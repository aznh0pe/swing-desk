"""Scan CLI: turns market data + dashboard state into alerts, exit signals and a digest.

    python -m engine.scan --mode auto --data <data-dir> --state <state-dir> --out <out-dir>

Never raises for per-ticker problems: errors are collected in the scan record. Writes:
    <out>/scan.json        full machine-readable result
    <out>/digest.md        email/notification-ready summary
    <out>/docs/...         one JSON file per dashboard document to write
    <out>/writes.json      ArtifactData batch manifest (chunks of <=50 writes)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import traceback

import numpy as np
import pandas as pd

from . import calendar as cal
from . import data as D
from . import indicators as I
from . import model as M
from . import strategies as S
from . import version as engine_version
from .monitor import check_trade, DYN_TEXT
from .portfolio import funds, load_collection, load_state, risk_governor, size_position

USER_FIELDS = {"status", "actual_price", "actual_shares", "done_at", "user_note", "trade_id",
               "skipped_at"}
# fields a later scan may refresh on an alert that already exists (sent as an overlay in the scan
# document, because existing documents are never overwritten); user fields are never touched
SCAN_OWNED = ("rank", "score", "win_pct", "model_win_pct", "grade", "news_check", "news_note", "capacity_note",
              "also_qualified", "topup_note", "shares", "fractional", "total_cost", "risk_usd", "pct_of_capital",
              "size_note", "exp_return_pct")
# the trade plan itself: refreshed on alerts nobody has acted on yet (a later scan on final prices can move
# the limit, stop or target by a few cents); the dashboard keeps the original plan once an order is placed
PLAN_FIELDS = ("entry", "stop", "target", "order", "rr", "risk_per_share", "reward_per_share", "last_close", "atr",
               "exit_signal", "earnings_next", "hold_max_days", "hold_typical_days")
TOKEN_RE = __import__("re").compile(r"^rt_[a-z0-9]{6,32}$")


def _r(x, nd=2):
    try:
        x = float(x)
    except Exception:
        return None
    return round(x, nd) if math.isfinite(x) else None


def resolve_mode(mode: str, now) -> str:
    if mode != "auto":
        return mode
    ph = cal.market_phase(now)
    return {"premarket": "premarket", "open": "preclose"}.get(ph, "postclose")


def load_params(state, repo_params_path):
    p = None
    if state.get("params_doc"):
        p = state["params_doc"].get("params") or state["params_doc"]
    if p is None and repo_params_path and pathlib.Path(repo_params_path).exists():
        try:
            p = json.loads(pathlib.Path(repo_params_path).read_text())
        except Exception:
            p = None
    return S.params_with_defaults(p)


def regime_summary(ind) -> dict:
    mk = ind["market"].iloc[-1]
    spy_up = bool(mk["spy_above200"] >= 0.5)
    b50 = float(mk["breadth50"]) if pd.notna(mk["breadth50"]) else None
    vix = float(mk["vix"]) if pd.notna(mk["vix"]) else None
    if spy_up and (b50 or 0) >= 0.5:
        label = "Bull"
    elif spy_up:
        label = "Bull, narrowing breadth"
    elif (b50 or 0) >= 0.4:
        label = "Correction"
    else:
        label = "Bear"
    return {"label": label, "spy": _r(mk["spy"]), "spy_vs_200d_pct": _r(100 * mk["spy_dist200"], 1),
            "spy_5d_pct": _r(100 * mk["spy_ret5"], 1), "breadth_above_50d_pct": _r(100 * b50, 0) if b50 is not None else None,
            "vix": _r(vix, 1), "vix_vs_20d": _r(mk["vix_ratio"], 2)}


def earnings_conflict(tk, earnings, start_day, hold) -> tuple[bool, str | None]:
    """True if the company reported in the last 3 sessions (the dip is an earnings reaction)
    or reports before the hold window ends (gap risk)."""
    e = (earnings or {}).get(tk) or {}
    recent_cut = cal.prev_trading_day(start_day, 4)
    end = start_day
    for _ in range(hold):
        end = cal.next_trading_day(end)
    hit, nxt_s = False, None
    for key in ("next", "last"):
        v = e.get(key)
        if not v:
            continue
        try:
            ed = cal._d(v)
        except Exception:
            continue
        if recent_cut <= ed <= end:
            hit = True
        if key == "next" and ed >= start_day:
            nxt_s = str(ed)
    return hit, nxt_s


# Trend-continuation setups whose premise is the trend itself: a close below the 50-day ends it.
# (Measured Nov 2026: the same rule on the dip-buying Leader 7-Day Pullback cut its win rate 68.5% -> 64.5%,
# so dip strategies keep their wide stop and time stop instead.)
TREND_FOLLOWING = ("TREND_PULLBACK", "BREAKOUT_RETEST", "GAP_HOLD")


def exit_signal_text(plan_name, sp, dyn, target, stop, hold, earnings_next):
    parts = []
    if dyn and dyn != "none":
        parts.append(f"Sell at the close if {DYN_TEXT.get(dyn, dyn)} (the daily scans alert you).")
    parts.append(f"Time stop: sell at the close on day {hold} if neither the target ${target:.2f} "
                 f"nor the stop ${stop:.2f} has filled.")
    if plan_name in TREND_FOLLOWING:
        parts.append("Exit early on a close below the 50-day average.")
    parts.append("Once up 2 ATR, raise the stop to entry + 1 ATR to lock in part of the gain (scan will flag it).")
    if earnings_next:
        parts.append(f"Next earnings {earnings_next}: exit before it if still open.")
    return " ".join(parts)


HELD_SECTORS: dict = {}  # open positions per sector bucket from the last build_entries call (saved in ctx)


def build_entries(ind, P, plans, state, earnings, universe, session_day, mode, partial_bar,
                  errors) -> tuple[list, list]:
    c = ind["close"]
    T = len(c) - 1
    asof = c.index[T].date()
    f = funds(state)
    gov = risk_governor(state, f["capital"], state["settings"])
    settings = {**state["settings"], "risk_pct": gov["risk_pct"]}
    open_tickers = {v["doc"].get("ticker") for v in state["trades"].values()
                    if v["doc"].get("status") == "open"}
    cands = []
    for name, plan in plans.items():
        try:
            entry_close = plan["entry_kind"] == "close"
            if mode == "preclose" and not entry_close:
                continue  # next-day limit setups are generated after the close
            if mode == "postclose" and entry_close:
                continue
            row = plan["mask"].values[T]
            js = np.nonzero(row)[0]
            if len(js) == 0:
                continue
            sp = P["strategies"][name]
            mdl = P.get("models", {}).get(name)
            off = float(P.get("live_adjust", {}).get(name, {}).get("offset", 0.0))
            feats = S.features_at(ind, np.full(len(js), T), js)
            p = M.predict(mdl, feats, off)
            for k, j in enumerate(js):
                tk = c.columns[j]
                try:
                    atr = float(ind["atr10"].values[T, j])
                    close = float(c.values[T, j])
                    entry = close if entry_close else float(plan["limit"].values[T, j])
                    if not (math.isfinite(entry) and entry > 0 and math.isfinite(atr) and atr > 0):
                        continue
                    if plan["stop"]["kind"] == "level":
                        stop = float(plan["stop"]["frame"].values[T, j])
                        if not math.isfinite(stop) or stop >= entry:
                            stop = entry - 2.0 * atr
                    else:
                        stop = entry - plan["stop"]["mult"] * atr
                    tg = plan["target"]
                    if tg["kind"] == "level":
                        lv = float(tg["frame"].values[T, j])
                        lo, hi = entry + tg.get("min_atr", 0.25) * atr, entry + tg.get("cap_atr", 4) * atr
                        target = lo if not math.isfinite(lv) else min(max(lv, lo), hi)
                    else:
                        target = entry + tg["mult"] * atr
                    rr = (target - entry) / max(entry - stop, 1e-6)
                    cands.append({"name": name, "j": int(j), "ticker": tk, "entry": entry,
                                  "stop": stop, "target": target, "rr": rr, "p": float(p[k]),
                                  "atr": atr, "close": close, "feats": feats.iloc[k].to_dict(),
                                  "model": mdl, "sp": sp, "plan": plan})
                except Exception as e:
                    errors.append(f"{name}/{tk}: {e!r}"[:200])
        except Exception as e:
            errors.append(f"strategy {name} failed: {e!r}"[:300])
    if not cands:
        return [], []
    # score: pooled robust-factor model when calibrated, else the per-strategy model
    pm = P.get("models", {}).get("_pooled")
    if pm and pm.get("kind") == "pooled":
        fdf = pd.DataFrame([cd["feats"] for cd in cands])
        names = [cd["name"] for cd in cands]
        try:
            p_, e_, ep_ = M.predict_pooled(pm, fdf, names, P.get("live_adjust"))
            sc_ = M.score_pooled(p_, ep_)
            for cd, pv, ev, epv, scv in zip(cands, p_, e_, ep_, sc_):
                cd.update(p=float(pv), exp_ret=float(ev), e_pct=float(epv), score=float(scv))
        except Exception as e:
            errors.append(f"pooled model failed, using per-strategy scores: {e!r}"[:200])
            pm = None
    if not (pm and pm.get("kind") == "pooled"):
        for cd in cands:
            cd["score"] = float(M.score(np.array([cd["p"]]), pd.DataFrame([cd["feats"]]), cd["model"], np.array([cd["rr"]]))[0])
    table = P["grading"].get("score_win_table") or []
    for cd in cands:
        cd["grade"] = M.grade(cd["score"], cd["p"], P["grading"])
        cd["model_p"] = cd["p"]
        for row in table:  # Win % = how often past setups with this score actually won
            if row["lo"] <= cd["score"] < row["hi"]:
                n = row["n"]
                cd["p"] = (row["win"] * n + cd["model_p"] * 30) / (n + 30)
                cd["band_avg"] = row["avg"]
                break
    # one setup per ticker: best score
    best = {}
    for cd in cands:
        if cd["ticker"] in open_tickers:
            continue
        b = best.get(cd["ticker"])
        if b is None or cd["score"] > b["score"]:
            best[cd["ticker"]] = cd
    ranked = sorted(best.values(), key=lambda x: (-x["score"], -x["p"]))
    # weekly minimum: from Wednesday's post-close scan on, top the week up to `min_week`
    # setups with the best grade-A candidates (clearly labelled as such)
    G = P["grading"]
    wk_start = (asof - dt.timedelta(days=asof.weekday())).isoformat()
    week_so_far = sum(1 for v in state["alerts"].values() if v["doc"].get("type") == "entry"
                      and str(v["doc"].get("signal_date", "")) >= wk_start)
    n_aplus = sum(1 for cd in ranked if cd["grade"] == "A+")
    need = max(0, int(G.get("min_week", 3)) - week_so_far - n_aplus) if (mode == "postclose" and asof.weekday() >= 2) else 0
    floor = float(G.get("topup_floor", G.get("aplus_score", 74) - 8))
    for cd in ranked:
        if need <= 0:
            break
        if cd["grade"] != "A+" and cd["score"] >= floor and cd["p"] >= G.get("aplus_pwin", 0.62):
            cd["grade"], cd["topup"] = "A", True
            need -= 1
    out, near = [], []
    cap_left = int(settings.get("max_open_positions", P["risk"]["max_open"])) - f["open_count"]
    cash_left = f["cash"]
    # sector concentration: correlated dips in one sector behave like one big position
    def sector_of(tk):
        if tk in universe.index:
            u = universe.loc[tk]
            sec = u["sector"] if isinstance(u["sector"], str) else None
            return sec or (u["name"] if u["type"] != "stock" else None)
        return None
    sec_count = {}
    for v in state["trades"].values():
        if v["doc"].get("status") == "open":
            sc = sector_of(v["doc"].get("ticker"))
            if sc:
                sec_count[sc] = sec_count.get(sc, 0) + 1
    HELD_SECTORS.clear()
    HELD_SECTORS.update(sec_count)
    FEATURED, PER_SECTOR = 8, 2
    rank, featured_sec = 0, {}
    for cd in ranked:
        hold = int(cd["sp"]["max_hold"])
        conflict, e_next = earnings_conflict(cd["ticker"], earnings, session_day, hold)
        if conflict:
            continue  # never trade an earnings reaction or carry a swing through earnings
        if cd["rr"] < 0.2:
            continue  # too lopsided: one stop-out erases five targets
        u = universe.loc[cd["ticker"]] if cd["ticker"] in universe.index else None
        meta = S.meta(cd["name"], P)
        size = size_position(cd["entry"], cd["stop"], f["capital"], cash_left, P, cd["score"], settings)
        e_win = cd["p"]
        mdl = cd["model"] or {}
        aw, al = mdl.get("avg_win", 0.025), mdl.get("avg_loss", -0.035)
        doc = {
            "type": "entry", "status": "new", "ticker": cd["ticker"],
            "name": (u["name"] if u is not None and isinstance(u["name"], str) else cd["ticker"]),
            "asset_type": (u["type"] if u is not None and isinstance(u["type"], str) else "stock"),
            "sector": (u["sector"] if u is not None and isinstance(u["sector"], str) else None),
            "strategy": cd["name"], "strategy_label": meta["label"], "family": meta["family"],
            "grade": cd["grade"], "score": round(cd["score"], 1), "win_pct": round(100 * e_win, 1),
            "model_win_pct": round(100 * cd.get("model_p", e_win), 1),
            "exp_return_pct": round(100 * (cd["exp_ret"] if "exp_ret" in cd else (e_win * aw + (1 - e_win) * al)), 2),
            "entry": _r(cd["entry"]), "stop": _r(cd["stop"]), "target": _r(cd["target"]),
            "rr": _r(cd["rr"], 2), "risk_per_share": _r(cd["entry"] - cd["stop"]),
            "reward_per_share": _r(cd["target"] - cd["entry"]),
            "order": ("Buy near the close today (limit at about the last price)" if cd["plan"]["entry_kind"] == "close"
                      else f"Buy LIMIT ${cd['entry']:.2f}, day order for {session_day} "
                           f"(place before the 6:30 AM PT open). Attach stop ${cd['stop']:.2f} and "
                           f"limit-sell target ${cd['target']:.2f} once filled (bracket/OCO)."),
            "valid_for": str(session_day if cd["plan"]["entry_kind"] != "close" else asof),
            "hold_max_days": hold, "valid_days": int(cd["plan"].get("valid_days", 1)),
            "hold_typical_days": int(max(1, round((mdl.get("avg_hold") or min(4, hold)) ))),
            "dyn_exit": cd["plan"]["dyn_exit"],
            "exit_signal": exit_signal_text(cd["name"], cd["sp"], cd["plan"]["dyn_exit"], cd["target"],
                                            cd["stop"], hold, e_next),
            "shares": size["shares"], "fractional": size["fractional"], "total_cost": size["total_cost"],
            "risk_usd": size["risk_usd"], "pct_of_capital": size["pct_of_capital"],
            "size_note": size["note"], "earnings_next": e_next,
            "last_close": _r(cd["close"]), "atr": _r(cd["atr"]),
            "signal_date": str(asof), "news_check": "pending", "news_note": None,
            "why": why_text(cd),
            "features": {k: _r(v, 4) for k, v in cd["feats"].items()},
            "partial_bar": bool(partial_bar),
        }
        if cd.get("topup"):
            doc["grade"] = "A"
            doc["topup_note"] = ("Weekly top-up: the best grade-A setup, sent because this week has fewer than "
                                 f"{G.get('min_week', 3)} A+ setups so far.")
        doc["sector_key"] = sector_of(cd["ticker"])
        if cd["grade"] == "A+" or cd.get("topup"):
            sc = doc["sector_key"]
            held_sec = sec_count.get(sc, 0) if sc else 0
            # featured list: best scores first, at most 2 per sector (counting open positions)
            is_featured = rank < FEATURED and (not sc or held_sec + featured_sec.get(sc, 0) < PER_SECTOR)
            if is_featured:
                rank += 1
                doc["rank"] = rank
                if sc:
                    featured_sec[sc] = featured_sec.get(sc, 0) + 1
                if rank > cap_left:
                    doc["capacity_note"] = (f"Ranked #{rank} tonight and you have {max(cap_left, 0)} free position "
                                            "slot(s). Place it only if you skip a higher-ranked setup, or once "
                                            "a position closes.")
                else:
                    cash_left -= size["total_cost"]
            else:
                doc["rank"] = None
                doc["also_qualified"] = True
                reason = (f"{held_sec + featured_sec.get(sc, 0)} positions/featured setups already in {sc}"
                          if sc and held_sec + featured_sec.get(sc, 0) >= PER_SECTOR else "outside tonight's top 8")
                doc["capacity_note"] = f"Also qualified ({reason}). Listed so you can swap into it if it suits you better."
            out.append(doc)
        elif cd["grade"] == "A" and len(near) < 8:
            near.append(doc)
    return out, near


def why_text(cd) -> list[str]:
    f = cd["feats"]
    w = []
    if f.get("rsi2") is not None and f["rsi2"] == f["rsi2"]:
        w.append(f"RSI(2) {f['rsi2']:.1f}: short-term oversold")
    if f.get("dist200") == f.get("dist200") and f.get("dist200") is not None:
        w.append(f"{100 * f['dist200']:+.1f}% vs 200-day average (long-term uptrend)")
    if f.get("rs_pct") is not None and f.get("is_etf", 0) < 0.5:
        rs = f["rs_pct"]
        w.append(f"Relative strength percentile {100 * rs:.0f}" + (" (weak: the model discounts it; the setup still cleared the bar)"
                                                                 if rs < 0.44 else " (a leader)" if rs >= 0.8 else ""))
    if f.get("depth_atr") == f.get("depth_atr") and f.get("depth_atr") is not None:
        w.append(f"{f['depth_atr']:.1f} ATR below its 5-day average")
    if f.get("spy_above200", 0) >= 0.5:
        w.append("Market (SPY) above its 200-day")
    if f.get("sector_up", 0) >= 0.75:
        w.append("Sector trend up")
    w.append(f"Reward:risk {cd['rr']:.2f} to the first target; win odds favour the bounce")
    return w


def current_overlay(state) -> dict:
    """Each alert's latest scan-side update, newest scan wins: the same view the dashboard shows."""
    scans = sorted((v["doc"] for v in (state.get("scans") or {}).values()), key=lambda d: str(d.get("started_at") or ""))
    ov: dict = {}
    for sc in scans:
        for aid, u in (sc.get("alert_updates") or {}).items():
            if isinstance(u, dict):
                ov[aid] = {**u, "_scan": sc.get("id"), "_at": sc.get("started_at")}
    return ov


def overlay_ctx(state, mode, session_day, today, now) -> dict:
    """What a later step needs to refresh alerts that already exist on the dashboard."""
    existing = [aid for aid, v in state["alerts"].items() if v["doc"].get("type") == "entry"]
    window = {}
    for aid, v in state["alerts"].items():
        a = v["doc"]
        if a.get("type") != "entry" or a.get("status") not in ("new", "placed", None):
            continue
        sid = str(a.get("scan_id", ""))
        if mode == "postclose" and str(a.get("valid_for")) == str(session_day) and not sid.endswith("preclose"):
            window[aid] = {"ticker": a.get("ticker"), "status": a.get("status")}
        elif mode == "preclose" and str(a.get("valid_for")) == str(today) and sid.endswith("preclose"):
            window[aid] = {"ticker": a.get("ticker"), "status": a.get("status")}
    return {"existing_entry_ids": existing, "window": window, "when": now.strftime("%-I:%M %p").lower()}


def build_overlay(scan_doc: dict, entries: list, ctx: dict) -> dict:
    ups = {}
    live = {e["id"] for e in entries}
    existing = set(ctx.get("existing_entry_ids") or [])
    for e in entries:
        if e["id"] in existing:
            u = {**{k: e.get(k) for k in SCAN_OWNED + PLAN_FIELDS}, "withdrawn": False}
            if e.get("news_check") in (None, "pending"):  # not news-checked this run: keep the earlier verdict
                u.pop("news_check", None)
                u.pop("news_note", None)
            ups[e["id"]] = u
    dropped = {d["ticker"]: d["reason"] for d in scan_doc.get("news_dropped") or []}
    for aid, a in (ctx.get("window") or {}).items():
        if aid in live:
            continue
        tk = a.get("ticker")
        reason = (f"Dropped by the {ctx.get('when')} scan's news check: {dropped[tk]}" if tk in dropped else
                  f"No longer qualifies in the {ctx.get('when')} scan (price or indicators moved, or a stronger setup "
                  "on the same ticker replaced it).")
        ups[aid] = {"withdrawn": True, "withdrawn_reason": reason, "rank": None, "ticker": tk}
    for c in scan_doc.get("premarket_checks") or []:
        if c.get("verdict") == "cancel" and c.get("alert_id"):
            ups[c["alert_id"]] = {"withdrawn": True, "withdrawn_reason": f"Pre-market check: {c.get('text')}", "rank": None,
                                  "ticker": c.get("ticker")}
    return ups


def headline(scan_doc: dict) -> str:
    if scan_doc.get("status") == "closed":
        return f"Market closed ({scan_doc.get('closed_reason') or 'holiday'})"
    c = scan_doc.get("counts") or {}
    bits = []
    if c.get("aplus"):
        bits.append(f"{c['aplus']} A+ setup{'s' if c['aplus'] != 1 else ''}")
    if c.get("exit_alerts"):
        bits.append(f"{c['exit_alerts']} exit alert{'s' if c['exit_alerts'] != 1 else ''}")
    if c.get("stop_raises"):
        bits.append(f"{c['stop_raises']} stop{'s' if c['stop_raises'] != 1 else ''} to raise")
    if c.get("fills_to_confirm"):
        bits.append(f"{c['fills_to_confirm']} fill{'s' if c['fills_to_confirm'] != 1 else ''} to confirm")
    if c.get("positions_to_check"):
        bits.append(f"{c['positions_to_check']} position{'s' if c['positions_to_check'] != 1 else ''} to check")
    pmc = scan_doc.get("premarket_checks") or []
    cancels = sum(1 for x in pmc if x.get("verdict") == "cancel" or (x.get("verdict") == "withdrawn" and x.get("placed")))
    if cancels:
        bits.append(f"{cancels} order{'s' if cancels != 1 else ''} to cancel")
    dont = sum(1 for x in pmc if x.get("verdict") == "withdrawn" and not x.get("placed"))
    if dont:
        bits.append(f"{dont} withdrawn setup{'s' if dont != 1 else ''} not to place")
    wd = sum(1 for v in (scan_doc.get("alert_updates") or {}).values() if v.get("withdrawn"))
    if wd:
        bits.append(f"{wd} withdrawn")
    out = " · ".join(bits) if bits else "No action needed"
    if (scan_doc.get("data") or {}).get("status") not in (None, "ok"):
        out = "⚠ data " + str(scan_doc["data"]["status"]) + " · " + out
    return out


# Which position signals mean what for the headline and the email. Every code the monitor can raise is in
# exactly one group, so nothing about an open position can fall through to "No action needed".
EXIT_CODES = {"EXIT_RULE", "TIME_STOP", "TREND_BREAK", "EARNINGS", "GAP_BELOW_STOP", "GAP_ABOVE_TARGET"}
FILL_CODES = {"STOP_HIT", "TARGET_HIT"}
RAISE_CODES = {"RAISE_STOP"}
NOTICE_CODES = {"TIME_STOP_NEXT", "MARKET_STRESS"}
CHECK_CODES = {"NO_DATA"}


def signal_counts(positions: list) -> dict:
    sigs = [sg for p in positions for sg in p.get("signals", [])]
    return {"exit_alerts": sum(1 for sg in sigs if sg["code"] in EXIT_CODES),
            "fills_to_confirm": sum(1 for sg in sigs if sg["code"] in FILL_CODES),
            "stop_raises": sum(1 for sg in sigs if sg["code"] in RAISE_CODES),
            "positions_to_check": sum(1 for sg in sigs if sg["code"] in CHECK_CODES),
            "notices": sum(1 for sg in sigs if sg["code"] in NOTICE_CODES)}


def email_headline(scan_doc: dict) -> str:
    """The first line of the emailed digest: what to act on, then the market label."""
    out = headline(scan_doc)
    label = (scan_doc.get("market") or {}).get("label")
    if label and scan_doc.get("status") != "closed":
        out += f" · market {label}"
    return out


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_done_doc(scan_doc: dict, token: str, at: str) -> dict:
    return {"token": token, "seq": 6, "state": "done", "pct": 100, "step": "Scan complete", "at": at,
            "scan_id": scan_doc["id"], "mode": scan_doc.get("mode"), "headline": headline(scan_doc),
            "market": (scan_doc.get("market") or {}).get("label"), "by": "scan"}


def run(mode: str, data_dir: str, state_dir: str | None, out_dir: str, params_path: str | None = None,
        now=None, run_token: str | None = None) -> dict:
    now = now or cal.now_pt()
    out = pathlib.Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    errors, notes = [], []
    requested = mode
    mode = resolve_mode(mode, now)
    if requested in ("premarket", "preclose", "postclose") and not cal.is_trading_day(now.date()):
        # a scheduled scan on a weekend or market holiday: nothing trades, so nothing to scan
        return closed_run(requested, now, out, run_token)
    _, close_pt = cal.session_times_pt(now.date())
    close_label = close_pt.strftime("%I:%M %p").lstrip("0") + " PT"
    if mode == "preclose" and cal.market_phase(now) == "postclose":
        # e.g. the 12:25 PM run on a half day (close 10:00 AM PT): the session is over, so selling
        # "before the close" is impossible; run as a post-close scan instead
        mode = "postclose"
        notes.append(f"The market already closed at {close_label} today (half day), so this ran as a post-close "
                     "scan: exits are for the next open and new setups for the next session.")
    scan_id = now.strftime("%Y%m%d-%H%M") + f"-{mode}"
    state = load_state(state_dir)
    P = load_params(state, params_path)
    meta = D.load_meta(data_dir)
    universe = D.load_universe(data_dir)
    earnings = D.load_earnings(data_dir)
    w = D.load_daily(data_dir)
    partial = False
    if mode == "preclose":
        intra = D.load_intraday(data_dir)
        if intra is not None and len(intra) and intra["date"].max().date() != now.date():
            intra = None  # a snapshot from another day is not today's price
        w, partial = D.append_partial_bar(w, intra)
        if not partial:
            notes.append("No live prices from today in the data feed yet, so buy-at-close setups were skipped and "
                         "exit checks used the last close. Check live prices in your broker before acting.")
    premarket = D.load_premarket(data_dir) if mode == "premarket" else None
    if premarket is not None and "asof" in premarket.columns:
        pm_day = str(premarket["asof"].dropna().max())[:10] if premarket["asof"].notna().any() else ""
        if pm_day != str(now.date()):
            notes.append(f"Pre-market prices in the feed are from {pm_day or 'an unknown time'}, not this morning, "
                         "so gap checks were skipped; earnings checks still ran.")
            premarket = None
    if mode == "preclose" and partial:
        mins_left = (close_pt - now).total_seconds() / 60
        if mins_left > 75:
            notes.append(f"Early read: prices as of {now.strftime('%I:%M %p').lstrip('0')} PT, "
                         f"{mins_left / 60:.1f} hours before the close. At-the-close exits and buy-at-close setups "
                         "can change by then; the pre-close scan 35 minutes before the close confirms them.")
    health = D.data_health({k: v.iloc[:-1] if partial else v for k, v in w.items()}, meta, mode, now)
    # trim to what indicators need
    w = {k: v.iloc[-400:] for k, v in w.items()}
    ind = I.compute(w, universe)
    plans = S.all_plans(ind, P)
    # the session new orders are for: today in the pre-close scan, otherwise the next session
    session_day = now.date() if mode == "preclose" else cal.next_session(now)
    if mode == "postclose" and cal.market_phase(now) == "open":
        session_day = cal.next_trading_day(now.date())
    entries, near = ([], [])
    if mode == "postclose" or (mode == "preclose" and partial):
        entries, near = build_entries(ind, P, plans, state, earnings, universe, session_day, mode,
                                      partial, errors)
    # news-check placeholders are filled by the session after web checks
    f = funds(state)
    gov = risk_governor(state, f["capital"], state["settings"])
    f["risk_pct"], f["risk_reason"] = gov["risk_pct"], gov["reason"]
    positions = []
    for tid, v in state["trades"].items():
        t = dict(v["doc"])
        t.setdefault("id", tid)
        if t.get("status") != "open":
            continue
        try:
            res = check_trade(t, ind, mode, earnings, premarket, P, partial, close_label)
            if res:
                positions.append(res)
        except Exception as e:
            errors.append(f"monitor {t.get('ticker')}: {e!r}"[:200])
    # pending entry alerts: expire / premarket re-check
    expire, pm_checks = [], []
    today = now.date()
    prev_ov = current_overlay(state) if mode == "premarket" else {}
    for aid, v in state["alerts"].items():
        a = v["doc"]
        # orders you marked as placed are checked before the open too: they are the ones that can fill
        if a.get("type") != "entry" or a.get("status") not in ("new", "placed", None):
            continue
        try:
            vf = cal._d(a.get("valid_for"))
        except Exception:
            continue
        if vf < today or (vf == today and mode == "postclose" and cal.market_phase(now) in ("postclose", "closed_day")):
            if a.get("status") in ("new", None):  # a placed order stays until you mark it filled or skipped
                expire.append((aid, v.get("version")))
        elif mode == "premarket" and vf == today:
            prev = prev_ov.get(aid) or {}
            if prev.get("withdrawn"):
                # a later scan already dropped it: say so instead of "keep", so a withdrawn order is never placed
                when = ""
                try:
                    when = dt.datetime.fromisoformat(str(prev.get("_at"))).strftime(" %a %-I:%M %p")
                except Exception:
                    pass
                placed = a.get("status") == "placed"
                pm_checks.append({"alert_id": aid, "ticker": a.get("ticker"), "verdict": "withdrawn", "placed": placed,
                                  "text": f"Withdrawn by the{when} scan: {prev.get('withdrawn_reason') or 'it no longer qualifies.'} "
                                  + ("You marked it placed: CANCEL it before the open." if placed else "Don't place it; cancel it if you did.")})
            else:
                c = premarket_check(a, premarket, earnings, today)
                c["placed"] = a.get("status") == "placed"
                pm_checks.append(c)
    pm_checks.sort(key=lambda c: ({"cancel": 0, "withdrawn": 1, "caution": 2}.get(c.get("verdict"), 3), str(c.get("ticker"))))
    # writes: create-only. A document that already exists (a re-run of the same scan
    # window, or one you've already acted on) is never overwritten; this scan's view of it
    # travels in the scan document's alert_updates instead.
    writes = []
    ctx = overlay_ctx(state, mode, session_day, today, now)
    ctx["held_sectors"] = dict(HELD_SECTORS)
    ts = now.isoformat(timespec="seconds")
    existing = state["alerts"]
    for doc in entries:
        did = f"e_{doc['valid_for']}_{doc['ticker']}_{doc['strategy']}".replace("^", "")
        doc.update({"id": did, "created_at": ts, "scan_id": scan_id})
        if did in existing:
            continue
        _write(out, writes, "set", "alerts", did, doc)
    for pos in positions:
        for sg in pos.get("signals", []):
            if sg["level"] not in ("alert", "good", "info") and sg["code"] not in CHECK_CODES:
                continue
            did = f"x_{pos['asof']}_{pos['trade_id']}_{sg['code']}"
            if did in existing:
                continue
            # a stop that already filled is an exit to record, not a stop to move
            is_exit = sg["code"] in EXIT_CODES or sg["code"] in FILL_CODES or sg["code"] in CHECK_CODES
            doc = {"id": did, "type": "exit" if is_exit else "adjust",
                   "status": "new", "trade_id": pos["trade_id"], "ticker": pos["ticker"],
                   "code": sg["code"], "action": sg["action"], "urgency": sg["urgency"],
                   "text": sg["text"], "new_stop": sg.get("new_stop"), "last": pos["last"],
                   "unrealized": pos["unrealized"], "days_held": pos["days_held"],
                   "days_left": pos["days_left"], "created_at": ts, "scan_id": scan_id}
            _write(out, writes, "set", "alerts", did, doc)
    week_start = today - dt.timedelta(days=today.weekday())
    week_aplus = sum(1 for v in existing.values() if v["doc"].get("type") == "entry"
                     and v["doc"].get("grade") == "A+" and str(v["doc"].get("signal_date", "")) >= str(week_start)) \
        + sum(1 for w_ in writes if w_["collection"] == "alerts" and w_["doc_id"].startswith("e_"))
    scan_doc = {
        "id": scan_id, "mode": mode, "started_at": ts, "status": "ok" if not errors else "ok_with_warnings",
        "data": {**health, "feed_generated_utc": meta.get("generated_utc"),
                 "intraday_asof": meta.get("intraday_asof"), "partial_bar": partial,
                 "tickers": int(ind["close"].shape[1])},
        "market": regime_summary(ind), "funds": f,
        "counts": {"aplus": len(entries), "near_misses": len(near), **signal_counts(positions),
                   "open_positions": len(positions), "expired": len(expire), "week_aplus": week_aplus,
                   "new_docs": len(writes)},
        "positions": positions, "near_misses": near[:8], "premarket_checks": pm_checks,
        "expired_alert_ids": [a for a, _ in expire][:50],
        "params_version": P.get("version"), "errors": errors[:40], "notes": notes,
        "session_day": str(session_day), "engine": engine_version.info(),
    }
    scan_doc["alert_updates"] = build_overlay(scan_doc, entries, ctx)
    digest = make_digest(scan_doc, entries, positions, near, pm_checks, week_aplus, P, now)
    scan_doc["digest"] = digest
    _write(out, writes, "set", "scans", scan_id, scan_doc)
    token = run_token if (run_token and TOKEN_RE.match(run_token)) else None
    if token:  # the dashboard's progress bar finishes when this lands, together with the results
        scan_doc["run_token"] = token
        _write(out, writes, "set", "runs", f"{token}-s6", run_done_doc(scan_doc, token, _utc_now()))
    (out / "digest.md").write_text(digest)
    (out / "headline.txt").write_text(email_headline(scan_doc))
    (out / "scan.json").write_text(json.dumps({"scan": scan_doc, "entries": entries, "ctx": ctx}, indent=1, default=str))
    chunks = [writes[i:i + 45] for i in range(0, len(writes), 45)]
    (out / "writes.json").write_text(json.dumps(chunks, indent=1))
    return {"scan_id": scan_id, "mode": mode, "entries": len(entries), "exits": scan_doc["counts"]["exit_alerts"],
            "writes": len(writes), "chunks": len(chunks), "status": scan_doc["status"],
            "data_status": health["status"], "errors": errors[:5], "headline": email_headline(scan_doc)}


def closed_run(mode: str, now, out: pathlib.Path, run_token: str | None = None) -> dict:
    """A scheduled scan on a weekend or holiday: record that it ran and why there was nothing to do."""
    reason = cal.closed_reason(now.date())
    nxt = cal.next_trading_day(now.date())
    scan_id = now.strftime("%Y%m%d-%H%M") + f"-{mode}"
    ts = now.isoformat(timespec="seconds")
    digest = "\n".join([
        f"SWING DESK | {mode.upper()} SCAN | {now.strftime('%a %b %d, %I:%M %p')} PT", "",
        f"Market closed today ({reason}). No orders fill and no exits trigger, so there was nothing to scan.",
        f"Next session: {nxt.strftime('%a %b %d')}, {cal.hours_label(nxt)}. Your open positions and their stops "
        "carry over unchanged.",
        "", "Model-generated signals from backtested rules, not financial advice."])
    scan_doc = {"id": scan_id, "mode": mode, "started_at": ts, "status": "closed", "closed_reason": reason,
                "session_day": str(nxt), "counts": {"aplus": 0, "exit_alerts": 0}, "positions": [],
                "notes": [f"Market closed ({reason})."], "digest": digest, "engine": engine_version.info()}
    writes = []
    _write(out, writes, "set", "scans", scan_id, scan_doc)
    token = run_token if (run_token and TOKEN_RE.match(run_token)) else None
    if token:
        scan_doc["run_token"] = token
        _write(out, writes, "set", "runs", f"{token}-s6", run_done_doc(scan_doc, token, _utc_now()))
    (out / "digest.md").write_text(digest)
    (out / "headline.txt").write_text(email_headline(scan_doc))
    (out / "scan.json").write_text(json.dumps({"scan": scan_doc, "entries": [], "ctx": {}}, indent=1, default=str))
    (out / "writes.json").write_text(json.dumps([writes], indent=1))
    return {"scan_id": scan_id, "mode": mode, "entries": 0, "exits": 0, "writes": len(writes), "chunks": 1,
            "status": "closed", "data_status": "n/a", "errors": [], "headline": email_headline(scan_doc)}


def premarket_check(a, premarket, earnings, today) -> dict:
    tk = a.get("ticker")
    res = {"alert_id": a.get("id"), "ticker": tk, "verdict": "keep", "text": "No pre-market issues."}
    e = (earnings or {}).get(tk) or {}
    if e.get("next") and str(e["next"]) == str(today):
        res.update(verdict="cancel", text="Reports earnings today: cancel the order.")
        return res
    if premarket is not None and tk in premarket.index:
        pm = premarket.loc[tk]
        pm_last = float(pm.get("pm_last", np.nan))
        lim, stop = float(a.get("entry") or 0), float(a.get("stop") or 0)
        atr = float(a.get("atr") or 0) or 1.0
        if math.isfinite(pm_last):
            res["pm_last"] = round(pm_last, 2)
            if stop and pm_last <= stop:
                res.update(verdict="cancel", text=f"Pre-market ${pm_last:.2f} is already below the stop "
                           f"${stop:.2f}: cancel (news-driven gap).")
            elif lim and pm_last < lim - 1.5 * atr:
                res.update(verdict="caution", text=f"Pre-market ${pm_last:.2f} is far below the limit "
                           f"${lim:.2f}. Check the news before letting it fill.")
    return res


def _write(out, writes, op, coll, did, data, version=None):
    p = out / "docs" / coll / f"{did}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, default=str))
    w = {"op": op, "collection": coll, "doc_id": did, "file_path": str(p.resolve())}
    if version is not None:
        w["if_version"] = int(version)
    writes.append(w)


_CAUTION = re.compile(r"(?<!\*\*)\b(caution)\b(?!\*\*)", re.IGNORECASE)


def mark_caution(text: str) -> str:
    """Make every news "caution" stand out in the email: a red marker and bold (markdown). Only the word is
    marked, and running it twice changes nothing."""
    return _CAUTION.sub(lambda m: f"🔴 **{m.group(1)}**", text or "")


def _hold_end(start, hold: int):
    """Last session of a hold window that starts (fills) on `start`."""
    try:
        d = cal._d(start)
    except Exception:
        return None
    for _ in range(max(0, int(hold) - 1)):
        d = cal.next_trading_day(d)
    return d


def _day(d) -> str:
    try:
        return cal._d(d).strftime("%a %b %d").replace(" 0", " ")
    except Exception:
        return str(d)


def _pct(a, b) -> str:
    try:
        return f"{100 * (float(a) / float(b) - 1):+.1f}%"
    except Exception:
        return ""


# Plain-text email: a coloured dot stands in for green / red / neutral text
TONE_MARK = {"sell": "🔴 ", "good": "🟢 ", "act": "🟠 ", "hold": "⚪ "}


def pnl_mark(x) -> str:
    """Open P&L for the email: 🟢 +$4.46 (gain), 🔴 -$0.63 (loss), ⚪ $0.00 (breakeven)."""
    try:
        v = round(float(x), 2)
    except Exception:
        return "–"
    if v > 0:
        return f"🟢 +${v:,.2f}"
    if v < 0:
        return f"🔴 -${abs(v):,.2f}"
    return "⚪ $0.00"


def _money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


SIGNAL_ORDER = {"GAP_BELOW_STOP": 0, "EARNINGS": 1, "EXIT_RULE": 2, "TIME_STOP": 3, "TREND_BREAK": 4, "GAP_ABOVE_TARGET": 5,
                "STOP_HIT": 6, "TARGET_HIT": 7, "RAISE_STOP": 8, "NO_DATA": 9, "TIME_STOP_NEXT": 10, "MARKET_STRESS": 11}


def make_digest(scan, entries, positions, near, pm_checks, week_aplus, P, now) -> str:
    """The email. Sections in the order you act on them: exits and fills first, then stop moves, then this
    morning's order checks, withdrawn setups, every open position with its full status, and finally each new
    setup with everything needed to place and manage it."""
    m = scan["market"]
    f = scan["funds"]
    mode_label = {"premarket": "PRE-MARKET CHECK", "preclose": "PRE-CLOSE SCAN", "postclose": "POST-CLOSE SCAN"}.get(scan["mode"], scan["mode"].upper() + " SCAN")
    lines = [f"SWING DESK | {mode_label} | {now.strftime('%a %b %d, %I:%M %p').replace(' 0', ' ')} PT", ""]
    lines.append(f"Market: {m['label']} | SPY {m['spy']} ({m['spy_vs_200d_pct']}% vs 200d, 5d {m['spy_5d_pct']}%) | "
                 f"breadth {m['breadth_above_50d_pct']}% above 50d | VIX {m['vix']}")
    lines.append(f"Capital {_money(f['capital'])} | cash free {_money(f['cash'])} | open positions {f['open_count']}"
                 f"/{P['risk']['max_open']}" + (f" | risk per trade {f['risk_pct']:g}% ({f['risk_reason']})" if f.get("risk_pct") else ""))
    d = scan["data"]
    if d["status"] != "ok":
        lines.insert(1, f"⚠ DATA WARNING ({d['status']}): " + " ".join(n for n in d["notes"] if "refreshed" not in n)
                     + " Verify prices in your broker before acting on anything below.")
    for n in scan.get("notes") or []:
        lines.append(f"Note: {n}")

    # before the open, today's session is the next hold day (the dashboard counts it the same way)
    day_add = 1 if scan["mode"] == "premarket" else 0

    def pos_line(p):
        return f"[last ${p['last']}, P&L {pnl_mark(p.get('unrealized'))} ({p.get('unrealized_pct')}%), day {p['days_held'] + day_add} of {p['max_hold']}]"

    sigs = sorted(((p, sg) for p in positions for sg in p.get("signals", [])), key=lambda x: SIGNAL_ORDER.get(x[1]["code"], 99))
    exits = [(p, sg) for p, sg in sigs if sg["code"] in EXIT_CODES]
    fills = [(p, sg) for p, sg in sigs if sg["code"] in FILL_CODES]
    raises = [(p, sg) for p, sg in sigs if sg["code"] in RAISE_CODES]
    checks = [(p, sg) for p, sg in sigs if sg["code"] in CHECK_CODES]
    notices = [(p, sg) for p, sg in sigs if sg["code"] in NOTICE_CODES]
    step = [0]

    def numbered(title):  # action sections are numbered in the order they appear
        step[0] += 1
        return f"=== {step[0]}. {title} ==="

    if exits or checks:
        lines += ["", numbered("EXIT NOW / ACT ON THESE POSITIONS")]
        for p, sg in exits + checks:
            hit = sorted({"target" if f["code"] == "TARGET_HIT" else "stop" for p2, f in fills if p2 is p}, reverse=True)
            if hit and sg["code"] in EXIT_CODES:  # the bracket has probably sold it already
                lines.append(f"- {p['ticker']}: ONLY IF STILL OPEN, {sg['action']} ({sg['urgency']}). Its {' and '.join(hit)} was reached, "
                             f"so your bracket has probably sold it: check your brokerage first. {sg['text']} {pos_line(p)}")
                continue
            lines.append(f"- {p['ticker']}: {sg['action']} ({sg['urgency']}). {sg['text']} {pos_line(p)}")
    if fills:
        lines += ["", "=== FILLS TO CONFIRM (your bracket order probably already handled these) ==="]
        for p, sg in fills:
            lines.append(f"- {p['ticker']}: {sg['action']}. {sg['text']} {pos_line(p)}")
    if raises:
        lines += ["", numbered("STOPS TO RAISE")]
        for p, sg in raises:
            lines.append(f"- {p['ticker']}: {sg['action']} (from ${p.get('stop')}). {sg['text']} {pos_line(p)}")
    if pm_checks:
        lines += ["", numbered("PRE-MARKET CHECK OF TODAY'S ORDERS")]
        for c in pm_checks:
            verdict = {"withdrawn": "WITHDRAWN, CANCEL IT" if c.get("placed") else "WITHDRAWN, DON'T PLACE"}.get(c["verdict"], c["verdict"].upper())
            lines.append(f"- {c['ticker']}{' (marked placed)' if c.get('placed') else ''}: {verdict}. {c['text']}")
    wd = [(k, v) for k, v in (scan.get("alert_updates") or {}).items() if v.get("withdrawn")]
    if wd:
        lines += ["", "=== WITHDRAWN SETUPS (cancel these orders if you placed them) ==="]
        for k, v in wd:
            tk = v.get("ticker") or (k.split("_")[2] if k.count("_") >= 3 else k)
            lines.append(f"- {tk}: {v.get('withdrawn_reason')}")
    pn = scan.get("position_news") or []
    if pn:
        lines += ["", "=== NEWS ON OPEN POSITIONS ==="]
        for x in pn:
            lines.append(f"- REVIEW {x['ticker']} position ({x['verdict']}): {x['reason']}")
    if positions:
        lines += ["", f"=== OPEN POSITIONS ({len(positions)}): STATUS AND WHAT TO DO ==="]
        for p in positions:
            label = S.meta(p.get("strategy") or "", P)["label"] if p.get("strategy") else "manual"
            st_ = p.get("status") or {}
            end = _hold_end(p.get("entry_date"), p.get("max_hold") or 0)
            sh = p.get("shares")
            sh = int(sh) if isinstance(sh, (int, float)) and float(sh).is_integer() else sh
            lines.append(f"- {p['ticker']}: {TONE_MARK.get(st_.get('tone'), '')}{st_.get('label', 'KEEP HOLDING')}. {st_.get('detail', '')}".rstrip())
            if p.get("last") is None:
                continue
            lines.append(f"    {label} | entered {_day(p.get('entry_date'))} at ${p.get('entry')} x {sh} | last ${p['last']} "
                         f"| P&L {pnl_mark(p.get('unrealized'))} ({_pct(p['last'], p.get('entry'))}) "
                         f"| day {p.get('day_now') or min(p['days_held'] + day_add, p['max_hold'])} of {p['max_hold']}"
                         f"{f', time stop at the close of {_day(end)}' if end else ''} | stop ${p.get('stop')} | target ${p.get('target')}")
            primary = st_.get("code")
            for p2, sg in notices:  # other notices that aren't already the status line
                if p2 is p and sg["code"] != primary and not (sg["code"] == "MARKET_STRESS" and primary == "HOLD"):
                    lines.append(f"    Note: {sg['text']}")
    featured = [e for e in entries if not e.get("also_qualified")]
    extra = [e for e in entries if e.get("also_qualified")]
    if entries:
        sess = _day(entries[0].get("valid_for"))
        lines += ["", f"=== NEW SETUPS FOR {sess.upper()} ({len(featured)} featured{f', {len(extra)} more below' if extra else ''}) ==="]
        lines.append("Buy near today's close (limit at about the last price); attach the stop and target as a bracket once filled."
                     if scan["mode"] == "preclose" else
                     "Place each as a DAY limit order before the 6:30 AM PT open; attach the stop and target as a bracket once filled.")
        for i, e in enumerate(featured, 1):
            lines += setup_block(e, i)
        if extra:
            lines += ["", "=== ALSO QUALIFIED (A+, outside the top 8 or sector-capped): swap in if you prefer one ==="]
            for e in extra:
                lines += setup_block(e, None)
    elif scan["mode"] == "postclose":
        lines += ["", "No A+ setups tonight. Standing aside is a position."]
    if near and (week_aplus < P["grading"].get("min_week", 3)):
        lines += ["", "=== WATCHLIST (grade A, not A+, shown because the weekly A+ count is low) ==="]
        for e in near[:4]:
            lines.append(f"- {e['ticker']} {e['strategy_label']}: score {e['score']}, win {e['win_pct']}%, "
                         f"limit ${e['entry']}, stop ${e['stop']}, target ${e['target']}")
    lines += ["", f"A+ setups this week: {week_aplus}"]
    eng = scan.get("engine") or {}
    if eng.get("source") and eng["source"] != "repo":
        lines.append(f"Engine: {eng['source']} {eng.get('fingerprint') or ''} (the repo couldn't be attached, so "
                     "this run used the identical copy published with the dashboard)")
    lines += ["", "Model-generated signals from backtested rules, not financial advice. Win % is an "
              "estimate from historical setups and can be wrong."]
    return mark_caution("\n".join(lines))


def setup_block(e: dict, rank) -> list[str]:
    """Everything needed to place and manage one setup, one labelled line per item."""
    head = f"{rank}. " if rank else "- "
    grade = e.get("grade", "A+") + (" (weekly top-up)" if e.get("topup_note") else "")
    out = ["", f"{head}{e['ticker']} | {e['name']} | {e['strategy_label']} | {grade} | score {e['score']} | win est. {e['win_pct']}%"
                f" | exp. {e.get('exp_return_pct', '')}% per trade"]
    frac = " (FRACTIONAL: market hours only)" if e.get("fractional") else ""
    out.append(f"   ORDER    {e['order']}")
    out.append(f"   BRACKET  Stop ${e['stop']} ({_pct(e['stop'], e['entry'])}) | Target ${e['target']} ({_pct(e['target'], e['entry'])}) | "
               f"reward:risk {e['rr']}")
    cap = e.get("pct_of_capital")
    out.append(f"   SIZE     {e['shares']} share{'s' if e['shares'] != 1 else ''}{frac} = ${e['total_cost']:,.2f}"
               f"{f' ({cap}% of capital)' if cap is not None else ''} | risk if stopped ${e['risk_usd']}")
    end = _hold_end(e.get("valid_for"), e.get("hold_max_days") or 0)
    out.append(f"   HOLD     up to {e['hold_max_days']} sessions (typical {e.get('hold_typical_days', '?')})"
               f"{f'; time stop at the close of {_day(end)}' if end else ''}")
    out.append(f"   EXIT     {e['exit_signal']}")
    why = e.get("why") or []
    if why:
        out.append(f"   WHY      {' · '.join(why)}")
    out.append(f"   EARNINGS {e['earnings_next'] + ' (after the hold window)' if e.get('earnings_next') else 'none inside the hold window'}")
    if e.get("news_check") and e["news_check"] != "pending":
        out.append(f"   NEWS     {e['news_check']}" + (f" · {e['news_note']}" if e.get("news_note") else ""))
    else:
        out.append("   NEWS     pending (the news check fills this in before the email goes out)")
    if e.get("capacity_note"):
        out.append(f"   NOTE     {e['capacity_note']}")
    if e.get("size_note"):
        out.append(f"   NOTE     {e['size_note']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--run-token", default=None, help="dashboard progress token (rt_...) from a Scan now press")
    a = ap.parse_args()
    try:
        res = run(a.mode, a.data, a.state, a.out, a.params, run_token=a.run_token)
    except Exception as e:
        res = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "failure.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
