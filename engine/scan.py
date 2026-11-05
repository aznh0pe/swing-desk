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
import traceback

import numpy as np
import pandas as pd

from . import calendar as cal
from . import data as D
from . import indicators as I
from . import model as M
from . import strategies as S
from .monitor import check_trade, DYN_TEXT
from .portfolio import funds, load_collection, load_state, size_position

USER_FIELDS = {"status", "actual_price", "actual_shares", "done_at", "user_note", "trade_id",
               "skipped_at"}


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
    e = (earnings or {}).get(tk) or {}
    nxt = e.get("next")
    if not nxt:
        return False, None
    try:
        ed = cal._d(nxt)
    except Exception:
        return False, None
    end = start_day
    for _ in range(hold):
        end = cal.next_trading_day(end)
    return (start_day <= ed <= end), str(ed)


def exit_signal_text(plan_name, sp, dyn, target, stop, hold, earnings_next):
    parts = []
    if dyn and dyn != "none":
        parts.append(f"Sell at the close if {DYN_TEXT.get(dyn, dyn)} (the daily scans alert you).")
    parts.append(f"Time stop: sell at the close on day {hold} if neither the target ${target:.2f} "
                 f"nor the stop ${stop:.2f} has filled.")
    if plan_name in ("TREND_PULLBACK", "BREAKOUT_RETEST", "GAP_HOLD", "LEADER_PULLBACK"):
        parts.append("Exit early on a close below the 50-day average.")
    parts.append("Raise the stop to breakeven once up 1 ATR (scan will flag it).")
    if earnings_next:
        parts.append(f"Next earnings {earnings_next}: exit before it if still open.")
    return " ".join(parts)


def build_entries(ind, P, plans, state, earnings, universe, session_day, mode, partial_bar,
                  errors) -> tuple[list, list]:
    c = ind["close"]
    T = len(c) - 1
    asof = c.index[T].date()
    f = funds(state)
    settings = state["settings"]
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
    # score
    for cd in cands:
        sc = M.score(np.array([cd["p"]]), pd.DataFrame([cd["feats"]]), cd["model"], np.array([cd["rr"]]))[0]
        cd["score"] = float(sc)
        cd["grade"] = M.grade(cd["score"], cd["p"], P["grading"])
    # one setup per ticker: best score
    best = {}
    for cd in cands:
        if cd["ticker"] in open_tickers:
            continue
        b = best.get(cd["ticker"])
        if b is None or cd["score"] > b["score"]:
            best[cd["ticker"]] = cd
    ranked = sorted(best.values(), key=lambda x: (-x["score"], -x["p"]))
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
    for cd in ranked:
        hold = int(cd["sp"]["max_hold"])
        conflict, e_next = earnings_conflict(cd["ticker"], earnings, session_day, hold)
        if conflict:
            continue  # never carry a swing through earnings
        u = universe.loc[cd["ticker"]] if cd["ticker"] in universe.index else None
        meta = S.STRATEGY_META[cd["name"]]
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
            "exp_return_pct": round(100 * (e_win * aw + (1 - e_win) * al), 2),
            "entry": _r(cd["entry"]), "stop": _r(cd["stop"]), "target": _r(cd["target"]),
            "rr": _r(cd["rr"], 2), "risk_per_share": _r(cd["entry"] - cd["stop"]),
            "reward_per_share": _r(cd["target"] - cd["entry"]),
            "order": ("Buy near the close today (limit at about the last price)" if cd["plan"]["entry_kind"] == "close"
                      else f"Buy LIMIT ${cd['entry']:.2f}, day order for {session_day} "
                           f"(place before the 6:30 AM PT open). Attach stop ${cd['stop']:.2f} and "
                           f"limit-sell target ${cd['target']:.2f} once filled (bracket/OCO)."),
            "valid_for": str(session_day if cd["plan"]["entry_kind"] != "close" else asof),
            "hold_max_days": hold,
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
        if cd["grade"] == "A+":
            sc = sector_of(cd["ticker"])
            if sc and sec_count.get(sc, 0) >= 2:
                doc["size_note"] = ((doc.get("size_note") or "") + f" Sector check: {sec_count[sc]} positions/setups "
                                    f"already in {sc}; taking more adds correlated risk.").strip()
            if sc:
                sec_count[sc] = sec_count.get(sc, 0) + 1
            if cap_left <= 0:
                doc["capacity_note"] = ("Max open positions reached. Only take this if it beats your "
                                        "weakest open trade (score/win%), or if a position exits first.")
            else:
                cap_left -= 1
                cash_left -= size["total_cost"]
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
        w.append(f"Relative strength percentile {100 * f['rs_pct']:.0f}")
    if f.get("depth_atr") == f.get("depth_atr") and f.get("depth_atr") is not None:
        w.append(f"{f['depth_atr']:.1f} ATR below its 5-day average")
    if f.get("spy_above200", 0) >= 0.5:
        w.append("Market (SPY) above its 200-day")
    if f.get("sector_up", 0) >= 0.75:
        w.append("Sector trend up")
    w.append(f"Reward:risk {cd['rr']:.2f} to the first target; win odds favour the bounce")
    return w


def run(mode: str, data_dir: str, state_dir: str | None, out_dir: str, params_path: str | None = None,
        now=None) -> dict:
    now = now or cal.now_pt()
    out = pathlib.Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    errors, notes = [], []
    mode = resolve_mode(mode, now)
    scan_id = now.strftime("%Y%m%d-%H%M") + f"-{mode}"
    state = load_state(state_dir)
    P = load_params(state, params_path)
    meta = D.load_meta(data_dir)
    universe = D.load_universe(data_dir)
    earnings = D.load_earnings(data_dir)
    w = D.load_daily(data_dir)
    partial = False
    if mode == "preclose":
        w, partial = D.append_partial_bar(w, D.load_intraday(data_dir))
        if not partial:
            notes.append("No intraday snapshot newer than the last daily bar; using the last close.")
    premarket = D.load_premarket(data_dir) if mode == "premarket" else None
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
    if mode in ("postclose", "preclose"):
        entries, near = build_entries(ind, P, plans, state, earnings, universe, session_day, mode,
                                      partial, errors)
    # news-check placeholders are filled by the session after web checks
    f = funds(state)
    positions = []
    for tid, v in state["trades"].items():
        t = dict(v["doc"])
        t.setdefault("id", tid)
        if t.get("status") != "open":
            continue
        try:
            res = check_trade(t, ind, mode, earnings, premarket, P, partial)
            if res:
                positions.append(res)
        except Exception as e:
            errors.append(f"monitor {t.get('ticker')}: {e!r}"[:200])
    # pending entry alerts: expire / premarket re-check
    expire, pm_checks = [], []
    today = now.date()
    for aid, v in state["alerts"].items():
        a = v["doc"]
        if a.get("type") != "entry" or a.get("status") not in ("new", None):
            continue
        try:
            vf = cal._d(a.get("valid_for"))
        except Exception:
            continue
        if vf < today or (vf == today and mode == "postclose" and cal.market_phase(now) in ("postclose", "closed_day")):
            expire.append((aid, v.get("version")))
        elif mode == "premarket" and vf == today:
            pm_checks.append(premarket_check(a, premarket, earnings, today))
    # writes: create-only. A document that already exists (a re-run of the same scan
    # window, or one you've already acted on) is never overwritten.
    writes = []
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
            if sg["level"] not in ("alert", "good", "info"):
                continue
            did = f"x_{pos['asof']}_{pos['trade_id']}_{sg['code']}"
            if did in existing:
                continue
            doc = {"id": did, "type": "exit" if sg["level"] in ("alert", "good") else "adjust",
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
        "counts": {"aplus": len(entries), "near_misses": len(near), "exit_alerts": sum(
            1 for p in positions for s in p.get("signals", []) if s["level"] in ("alert", "good")),
            "open_positions": len(positions), "expired": len(expire), "week_aplus": week_aplus,
            "new_docs": len(writes)},
        "positions": positions, "near_misses": near[:8], "premarket_checks": pm_checks,
        "expired_alert_ids": [a for a, _ in expire][:50],
        "params_version": P.get("version"), "errors": errors[:40], "notes": notes,
        "session_day": str(session_day),
    }
    digest = make_digest(scan_doc, entries, positions, near, pm_checks, week_aplus, P, now)
    scan_doc["digest"] = digest
    _write(out, writes, "set", "scans", scan_id, scan_doc)
    (out / "digest.md").write_text(digest)
    (out / "scan.json").write_text(json.dumps({"scan": scan_doc, "entries": entries}, indent=1, default=str))
    chunks = [writes[i:i + 45] for i in range(0, len(writes), 45)]
    (out / "writes.json").write_text(json.dumps(chunks, indent=1))
    return {"scan_id": scan_id, "mode": mode, "entries": len(entries), "exits": scan_doc["counts"]["exit_alerts"],
            "writes": len(writes), "chunks": len(chunks), "status": scan_doc["status"],
            "data_status": health["status"], "errors": errors[:5]}


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


def make_digest(scan, entries, positions, near, pm_checks, week_aplus, P, now) -> str:
    m = scan["market"]
    f = scan["funds"]
    lines = [f"SWING DESK | {scan['mode'].upper()} SCAN | {now.strftime('%a %b %d, %I:%M %p')} PT", ""]
    lines.append(f"Market: {m['label']} | SPY {m['spy']} ({m['spy_vs_200d_pct']}% vs 200d, 5d {m['spy_5d_pct']}%) | "
                 f"breadth {m['breadth_above_50d_pct']}% above 50d | VIX {m['vix']}")
    lines.append(f"Capital ${f['capital']:,.2f} | cash free ${f['cash']:,.2f} | open {f['open_count']}"
                 f"/{P['risk']['max_open']}")
    d = scan["data"]
    if d["status"] != "ok":
        lines.append(f"Data note: {d['status']}: " + " ".join(d["notes"]))
    exits = [(p, s) for p in positions for s in p.get("signals", []) if s["level"] in ("alert", "good")]
    if exits:
        lines += ["", "EXIT / ACTION ALERTS"]
        for p, s in exits:
            lines.append(f"- {p['ticker']}: {s['action']} ({s['urgency']}). {s['text']} "
                         f"[last ${p['last']}, P&L ${p['unrealized']}, day {p['days_held']}/{p['max_hold']}]")
    adj = [(p, s) for p in positions for s in p.get("signals", []) if s["level"] == "info"]
    if adj:
        lines += ["", "STOP / POSITION UPDATES"]
        for p, s in adj:
            lines.append(f"- {p['ticker']}: {s['action']}. {s['text']}")
    if pm_checks:
        lines += ["", "PRE-MARKET CHECK OF TODAY'S ORDERS"]
        for c in pm_checks:
            lines.append(f"- {c['ticker']}: {c['verdict'].upper()}. {c['text']}")
    if entries:
        lines += ["", f"A+ SETUPS ({len(entries)})"]
        for i, e in enumerate(entries, 1):
            frac = " (FRACTIONAL: market hours only)" if e["fractional"] else ""
            lines.append(f"{i}. {e['ticker']} ({e['name']}) | {e['strategy_label']} | score {e['score']} | win {e['win_pct']}%")
            lines.append(f"   Entry: {e['order']}")
            lines.append(f"   Stop ${e['stop']} | Target ${e['target']} | R:R {e['rr']} | hold up to {e['hold_max_days']} days")
            lines.append(f"   Size {e['shares']} sh{frac} = ${e['total_cost']:,.2f} ({e['pct_of_capital']}% of capital), risk ${e['risk_usd']}")
            lines.append(f"   Exit signal: {e['exit_signal']}")
            if e.get("capacity_note"):
                lines.append(f"   Note: {e['capacity_note']}")
            if e.get("size_note"):
                lines.append(f"   Note: {e['size_note']}")
    elif scan["mode"] == "postclose":
        lines += ["", "No A+ setups tonight. Standing aside is a position."]
    if near and (week_aplus < P["grading"].get("min_week", 3)):
        lines += ["", "WATCHLIST (grade A, not A+, shown because the weekly A+ count is low)"]
        for e in near[:4]:
            lines.append(f"- {e['ticker']} {e['strategy_label']}: score {e['score']}, win {e['win_pct']}%, "
                         f"limit ${e['entry']}, stop ${e['stop']}, target ${e['target']}")
    lines += ["", f"A+ setups this week: {week_aplus}", "",
              "Model-generated signals from backtested rules, not financial advice. Win % is an "
              "estimate from historical setups and can be wrong."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=None)
    a = ap.parse_args()
    try:
        res = run(a.mode, a.data, a.state, a.out, a.params)
    except Exception as e:
        res = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "failure.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
