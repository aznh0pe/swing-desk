"""Backtest grid search + daily learning loop.

    python -m engine.calibrate backtest --data <dir with history.parquet> --out <dir>
    python -m engine.calibrate learn    --data <data-dir> --state <state-dir> --out <dir>

`backtest` picks, per strategy, the variant with the highest in-sample win rate that still
clears profit-factor / expectancy floors, reports out-of-sample results honestly, fits the
win-probability model, calibrates the A+ cut-offs and writes params.json.

`learn` grades every past alert against what the market actually did (shadow results, so
the system learns even from setups you skipped), folds in your real fills, nudges each
strategy's win-probability toward its live record, flags drift, and writes a journal entry.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import pathlib
import time

import numpy as np
import pandas as pd

from . import backtest as B
from . import calendar as cal
from . import data as D
from . import indicators as I
from . import model as M
from . import rules as R
from . import strategies as S
from .portfolio import load_state

GRIDS = {
    "MR_STRETCH": {
        "entry": [{"rsi2_max": r, "k_atr": k} for r in (5, 10) for k in (0.0, 0.3, 0.6, 0.9)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "mean4", "min_atr": 0.3}, {"kind": "atr", "mult": 0.75},
                           {"kind": "atr", "mult": 1.25})
                 for s in (2.5, 3.5) for h in (5, 8)],
    },
    "LEADER_PULLBACK": {
        "entry": [{"rsi2_max": r, "k_atr": k, "rs_min": q} for r in (10, 20) for k in (0.0, 0.4, 0.8)
                  for q in (0.7, 0.85)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "max7c", "min_atr": 0.3}, {"kind": "atr", "mult": 1.0})
                 for s in (2.5, 3.5) for h in (6, 10)],
    },
    "ETF_IBS": {
        "entry": [{"ibs_max": i, "k_atr": k, "entry_kind": ek, "rsi2_max": 35}
                  for i in (0.1, 0.2) for ek, k in (("close", 0.0), ("limit_next", 0.0),
                                                    ("limit_next", 0.3), ("limit_next", 0.6))],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "high_t", "min_atr": 0.25}, {"kind": "atr", "mult": 0.75})
                 for s in (2.0, 3.0) for h in (4, 7)],
    },
    "TREND_PULLBACK": {
        "entry": [{"k_atr": k, "rs_min": q} for k in (0.0, 0.3, 0.6) for q in (0.7, 0.85)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "hh20", "min_atr": 0.75}, {"kind": "atr", "mult": 1.0})
                 for s in (2.0, 3.0) for h in (7, 12)],
    },
    "BREAKOUT_RETEST": {
        "entry": [{"base_max": b, "vol_mult": v, "valid_days": d} for b in (0.10, 0.15)
                  for v in (1.5, 2.0) for d in (1, 3)],
        "exit": [{"target": {"kind": "atr", "mult": m}, "stop_atr": s, "max_hold": h}
                 for m in (1.0, 2.0) for s in (1.5, 2.5) for h in (7, 12)],
    },
    "GAP_HOLD": {
        "entry": [{"gap_min": g, "vol_mult": v, "rsi2_max": r, "k_atr": k} for g in (0.03, 0.05)
                  for v in (2.0, 3.0) for r in (20, 35) for k in (0.0, 0.4)],
        "exit": [{"target": {"kind": "atr", "mult": m}, "max_hold": h} for m in (1.0, 1.5, 2.5)
                 for h in (6, 10)],
    },
    "FAIR_VALUE": {
        "entry": [{"z": z, "k_atr": k, "rsi2_max": r} for z in (1.5, 2.0, 2.5) for k in (0.0, 0.4)
                  for r in (15, 30)],
        "exit": [{"target": t, "stop_atr": s, "max_hold": h}
                 for t in ({"kind": "vwap20", "min_atr": 0.3}, {"kind": "mean4", "min_atr": 0.3})
                 for s in (2.5, 3.5) for h in (5, 8)],
    },
}

FLOORS = {"min_trades": 120, "min_pf": 1.35, "min_avg": 0.0035, "min_fill": 0.45}


def objective(s: dict) -> float:
    """Grid-search objective on 2008-2021 trades: win rate first, expectancy second, with floors.
    A limit that sits so deep that most orders never fill looks great per filled trade but produces
    few setups and misses the bounces that start without a dip, so fill rate has a floor too."""
    if not s or s.get("trades", 0) < FLOORS["min_trades"]:
        return -1.0
    pf = s.get("profit_factor") or 0
    pen = 0.0
    fr = s.get("fill_rate")
    if fr is not None and fr < FLOORS["min_fill"]:
        pen += 0.25 + (FLOORS["min_fill"] - fr)
    if pf < FLOORS["min_pf"]:
        pen += 0.25 + (FLOORS["min_pf"] - pf)
    if s["avg_ret"] < FLOORS["min_avg"]:
        pen += 0.25 + 20 * (FLOORS["min_avg"] - s["avg_ret"])
    return s["win_rate"] + 2.0 * s["avg_ret"] - pen


def _variant(P, name, v):
    P2 = copy.deepcopy(P)
    P2["strategies"][name].update(copy.deepcopy(v))
    return P2


def search_strategy(name, ind, P, split, log):
    g = GRIDS[name]
    best_entry, best_obj, tried = None, -9, []
    base_exit = {}
    for ev in g["entry"]:
        P2 = _variant(P, name, ev)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": ev, "train": st, "obj": round(o, 4)})
        if o > best_obj:
            best_obj, best_entry = o, ev
    if best_entry is None:
        return None
    best_v, best_obj2, best_tr = dict(best_entry), -9, None
    for xv in g["exit"]:
        v = {**best_entry, **xv}
        P2 = _variant(P, name, v)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": v, "train": st, "obj": round(o, 4)})
        if o > best_obj2:
            best_obj2, best_v, best_tr = o, v, tr
    # stage 3: market-regime gate
    for reg in ("spy200", "spy200_breadth", "vixcalm"):
        v = {**best_v, "regime": reg}
        P2 = _variant(P, name, v)
        plan = S.build(name, ind, P2)
        tr = B.simulate(plan, ind, P2, include_unfilled=True)
        if tr.empty:
            continue
        st = B.stats(tr[tr["signal_date"] < split])
        o = objective(st)
        tried.append({"variant": v, "train": st, "obj": round(o, 4)})
        if o > best_obj2 + 0.005:  # gate must earn its keep
            best_obj2, best_v, best_tr = o, v, tr
    log(f"  {name}: best train objective {best_obj2:.3f} with {best_v}")
    return {"variant": best_v, "trades": best_tr, "objective": best_obj2, "tried": tried}


def _isotonic(rows: list[dict]) -> list[dict]:
    """Pool adjacent bands until the win rate never falls as the score rises."""
    blocks = [{"lo": r["lo"], "hi": r["hi"], "w": r["win"] * r["n"], "a": r["avg"] * r["n"], "n": r["n"], "rows": [r]} for r in rows]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i]["w"] / blocks[i]["n"] > blocks[i + 1]["w"] / blocks[i + 1]["n"]:
            a, b = blocks[i], blocks.pop(i + 1)
            a.update(hi=b["hi"], w=a["w"] + b["w"], a=a["a"] + b["a"], n=a["n"] + b["n"], rows=a["rows"] + b["rows"])
            i = max(0, i - 1)
        else:
            i += 1
    out = []
    for bl in blocks:
        for r in bl["rows"]:
            out.append({**r, "win": round(bl["w"] / bl["n"], 4), "avg": round(bl["a"] / bl["n"], 5), "n_band": bl["n"]})
    return out


CAL_END = "2024-01-01"  # the post-split window is reported in two halves (2022-23 and 2024-now) so regime shifts show


def _aplus_window(tr_all: pd.DataFrame, sc: int, pw: float, min_week: int) -> dict:
    """A+ results inside one window of signals: filled-trade win rate, average trade, setups per week."""
    if tr_all.empty:
        return {"trades": 0}
    filled = tr_all[tr_all["ret"].notna() & tr_all["filled"]]
    sel = filled[(filled["score"] >= sc) & (filled["p"] >= pw)]
    s_all = tr_all[(tr_all["score"] >= sc) & (tr_all["p"] >= pw)].drop_duplicates(["signal_date", "ticker"])
    weeks_idx = pd.date_range(tr_all["signal_date"].min(), tr_all["signal_date"].max(), freq="W")
    weeks = max(1.0, len(weeks_idx))
    wk = s_all.groupby(pd.Grouper(key="signal_date", freq="W")).size().reindex(weeks_idx, fill_value=0)
    return {"trades": int(len(sel)), "win_rate": round(float((sel["ret"] > 0).mean()), 4) if len(sel) else None,
            "avg_ret": round(float(sel["ret"].mean()), 5) if len(sel) else None,
            "signals_per_week": round(float(len(s_all) / weeks), 1),
            "weeks_with_min_setups": round(float((wk >= min_week).mean()), 3),
            "window": [str(tr_all["signal_date"].min().date()), str(tr_all["signal_date"].max().date())]}


def _select_cutoffs(sig: pd.DataFrame, min_week: int, prefer_win: bool = False):
    """The (score, win-odds) bar with the highest filled-trade win rate whose average trade is at least
    +0.6% and that still produces enough setups a week across the universe; None if nothing clears it."""
    t = sig[sig["ret"].notna() & sig["filled"]]
    if t.empty:
        return None
    weeks_idx = pd.date_range(sig["signal_date"].min(), sig["signal_date"].max(), freq="W")
    weeks = max(1.0, len(weeks_idx))
    cands = []
    for sc in range(40, 94, 2):
        for pw in (0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
            sel = t[(t["score"] >= sc) & (t["p"] >= pw)]
            if len(sel) < 150:
                continue
            avg = sel["ret"].mean()
            s_all = sig[(sig["score"] >= sc) & (sig["p"] >= pw)].drop_duplicates(["signal_date", "ticker"])
            per_week = len(s_all) / weeks
            if avg < 0.006 or per_week < min_week * 1.6:
                continue
            wr = (sel["ret"] > 0).mean()
            wk = s_all.groupby(pd.Grouper(key="signal_date", freq="W")).size().reindex(weeks_idx, fill_value=0)
            cands.append((float(wr), float(avg), sc, pw, per_week, len(sel), float((wk >= min_week).mean())))
    if not cands:
        return None
    top_wr = max(c[0] for c in cands)
    top_avg = max(c[1] for c in cands if c[0] >= top_wr - 0.012)
    # adjacent score bands are statistically indistinguishable, so among cut-offs within 1.2 points
    # of the best win rate (and 85% of its average trade) take the one that covers the most weeks
    near = [c for c in cands if c[0] >= top_wr - 0.012 and c[1] >= 0.85 * top_avg] or cands
    if prefer_win:  # A+ is on watch (its live record trails the test): take the strictest of the near-best bars
        return max(near, key=lambda c: (c[0], c[2], c[3]))
    return max(near, key=lambda c: (c[6], c[0]))


# ----------------------------------------------------------------------------- weekly top-ups
TOPUP_MIN_TRADES = 150   # filled top-ups 2022-now before a top-up rule can be chosen
TOPUP_MIN_AVG = 0.004    # their average trade must clear +0.4%
TOPUP_MIN_PF = 1.25      # and their profit factor 1.25
TOPUP_STRAT_MIN = 30     # top-ups of one strategy before its own result counts
TOPUP_STRAT_PRIOR = 20


def _one_per_ticker(sig: pd.DataFrame) -> pd.DataFrame:
    """One setup per ticker and day, the best-scored, as the scanner sends them."""
    return sig.sort_values(["score", "p"], ascending=False).drop_duplicates(["signal_date", "ticker"])


def simulate_topups(s: pd.DataFrame, sc: float, pw: float, floor: float, tpw: float, min_week: int,
                    excluded=()) -> tuple[pd.DataFrame, pd.Series]:
    """Replays the scanner's weekly top-up rule day by day. Every A+ setup counts first; when the week is
    behind its pace (one more setup each weekday, up to min_week), the best-scored setups below the A+ bar
    with score >= floor and model win >= tpw are added (strategies in `excluded` never top up).
    s: one row per signal day and ticker (see _one_per_ticker). Returns (top-ups sent, setups per week)."""
    ap = (s["score"] >= sc) & (s["p"] >= pw)
    n_ap = s.loc[ap].groupby("signal_date").size()
    band = s[~ap & (s["score"] >= floor) & (s["p"] >= tpw)]
    if len(excluded):
        band = band[~band["strategy"].isin(list(excluded))]
    groups = {d: g for d, g in band.groupby("signal_date", sort=False)}
    picks, weeks, so_far, cur = [], {}, 0, None
    for d in sorted(s["signal_date"].unique()):
        d = pd.Timestamp(d)
        wk = d - pd.Timedelta(days=d.weekday())
        if wk != cur:
            cur, so_far = wk, 0
        na = int(n_ap.get(d, 0))
        need = max(0, min(min_week, d.weekday() + 1) - so_far - na)
        k = 0
        if need and d in groups:
            g = groups[d].head(need)
            picks.append(g)
            k = len(g)
        so_far += na + k
        weeks[wk] = so_far
    pk = pd.concat(picks) if picks else s.iloc[0:0]
    return pk, pd.Series(weeks, dtype=float)


def _trade_stats(df: pd.DataFrame) -> dict:
    f = df[df["filled"] & df["ret"].notna()]
    if not len(f):
        return {"trades": 0, "win_rate": None, "avg_ret": None, "profit_factor": None}
    w = f["ret"] > 0
    gl = -f.loc[~w, "ret"].sum()
    return {"trades": int(len(f)), "win_rate": round(float(w.mean()), 4), "avg_ret": round(float(f["ret"].mean()), 5),
            "profit_factor": round(float(f.loc[w, "ret"].sum() / gl), 2) if gl > 0 else None}


def _choose_topup_rule(s: pd.DataFrame, sc: float, pw: float, min_week: int, min_trades: int = TOPUP_MIN_TRADES):
    """The top-up bar with the highest win rate among rules whose replayed top-ups clear the trade-count,
    average-trade and profit-factor floors; among rules within 1.5 points of the best, the one that keeps
    the most weeks at the minimum. None if no rule clears the floors."""
    cands = []
    for floor in range(max(40, int(sc) - 24), int(sc), 2):
        for tpw in sorted({round(max(0.5, pw - d), 2) for d in (0.08, 0.06, 0.04, 0.02, 0.0)}):
            pk, wk = simulate_topups(s, sc, pw, floor, tpw, min_week)
            st = _trade_stats(pk)
            if st["trades"] < min_trades or st["avg_ret"] < TOPUP_MIN_AVG or (st["profit_factor"] or 0) < TOPUP_MIN_PF:
                continue
            cands.append((st["win_rate"], float((wk >= min_week).mean()) if len(wk) else 0.0, floor, tpw, st, pk, wk))
    if not cands:
        return None
    top = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= top - 0.015]
    return max(near, key=lambda c: (c[1], c[0]))


def _topup_exclusions(pk: pd.DataFrame, wr: float) -> dict:
    out = {}
    for name, g in pk.groupby("strategy"):
        st = _trade_stats(g)
        n = st["trades"]
        if n < TOPUP_STRAT_MIN:
            continue
        shr = (st["win_rate"] * n + wr * TOPUP_STRAT_PRIOR) / (n + TOPUP_STRAT_PRIOR)
        if shr < wr - 0.04 or (st["profit_factor"] or 0) < 1.0:
            out[name] = (f"its top-ups won {100 * st['win_rate']:.0f}% of {n} on 2022-now data (profit factor "
                         f"{st['profit_factor']}) vs {100 * wr:.0f}% for all top-ups; it can still send A+ setups")
    return out


def calibrate_topups(post: pd.DataFrame, sc: float, pw: float, min_week: int = 5, cal_end=CAL_END) -> dict:
    """The weekly top-ups, tested and tuned on their own: replays the exact weekly rule on 2022-now signals,
    picks the top-up bar, drops strategies whose own top-ups trail, and reports the result by year, by
    strategy and on the two halves of the window (bar chosen on the first half, measured on the second)."""
    if post.empty:
        return {"topup_ok": False, "topup_min_week": min_week, "topup_note": "No signals to test top-ups on."}
    s = _one_per_ticker(post if "strategy" in post else post.assign(strategy="?"))
    best = _choose_topup_rule(s, sc, pw, min_week)
    _, wk_ap = simulate_topups(s, sc, pw, 999, 9.0, min_week)
    ap_cover = round(float((wk_ap >= min_week).mean()), 3) if len(wk_ap) else None
    if best is None:
        return {"topup_ok": False, "topup_min_week": min_week, "oos_weeks_at_min_aplus_only": ap_cover,
                "topup_floor": int(sc) - 8, "topup_pwin": round(pw - 0.03, 2), "topup_excluded": {},
                "topup_note": (f"No top-up rule cleared the floors on 2022-now data (at least {TOPUP_MIN_TRADES} trades, "
                               f"average above +{100 * TOPUP_MIN_AVG:.1f}%, profit factor {TOPUP_MIN_PF}), so top-ups are off "
                               "until a recalibration finds one.")}
    wr, cover, floor, tpw, st, pk, wk = best
    # strategies whose own top-ups trail send none; removing one changes the picks, so check again (3 rounds at most)
    excl, cover0 = {}, cover
    for _ in range(3):
        more = {n: why for n, why in _topup_exclusions(pk, st["win_rate"]).items() if n not in excl}
        if not more:
            break
        trial = {**excl, **more}
        pk2, wk2 = simulate_topups(s, sc, pw, floor, tpw, min_week, excluded=list(trial))
        st2 = _trade_stats(pk2)
        c2 = float((wk2 >= min_week).mean()) if len(wk2) else 0.0
        if st2["trades"] < TOPUP_MIN_TRADES or st2["win_rate"] < st["win_rate"] or c2 < cover0 - 0.05:
            break
        excl, pk, wk, st, cover = trial, pk2, wk2, st2, c2
    weeks = max(1, len(wk))
    by_year = [{"year": int(y), **_trade_stats(g)} for y, g in pk.groupby(pk["signal_date"].dt.year)]
    by_strategy = {n: _trade_stats(g) for n, g in pk.groupby("strategy")}
    out = {"topup_ok": True, "topup_floor": int(floor), "topup_pwin": float(tpw), "topup_excluded": excl,
           "topup_min_week": min_week, "oos_topup_win_rate": st["win_rate"], "oos_topup_avg_ret": st["avg_ret"],
           "oos_topup_pf": st["profit_factor"], "oos_topup_trades": st["trades"],
           "oos_topup_sent_per_week": round(len(pk) / weeks, 2), "oos_weeks_at_min": round(cover, 3),
           "oos_weeks_at_min_aplus_only": ap_cover, "topup_by_year": by_year, "topup_by_strategy": by_strategy,
           "topup_note": ""}
    if cal_end is not None:
        ce = pd.Timestamp(cal_end)
        se, sl = s[s["signal_date"] < ce], s[s["signal_date"] >= ce]
        if len(se) and len(sl):
            ts = {"chosen_rule_early": _trade_stats(simulate_topups(se, sc, pw, floor, tpw, min_week, list(excl))[0]),
                  "chosen_rule_late": _trade_stats(simulate_topups(sl, sc, pw, floor, tpw, min_week, list(excl))[0]),
                  "window_early": [str(se["signal_date"].min().date()), str((ce - pd.Timedelta(days=1)).date())],
                  "window_late": [str(ce.date()), str(sl["signal_date"].max().date())]}
            eb = _choose_topup_rule(se, sc, pw, min_week, min_trades=60)
            if eb is not None:
                ts["early_selection"] = {"topup_floor": int(eb[2]), "topup_pwin": float(eb[3]), **eb[4]}
                ts["early_rule_on_late"] = _trade_stats(simulate_topups(sl, sc, pw, eb[2], eb[3], min_week)[0])
            out["topup_time_split"] = ts
    return out


# ----------------------------------------------------------------------------- market conditions
COND_GATE_TRAIN_N = 100     # trades in 2008-2021 before a strategy's result in one condition can propose a pause
COND_GATE_WIN, COND_GATE_PF = 0.62, 1.10


def _with_conditions(df: pd.DataFrame) -> pd.DataFrame:
    return df if "condition" in df else df.assign(condition=S.condition_series(df).values)


def _sent(post: pd.DataFrame, G: dict, min_week: int, pairs=frozenset()):
    """What the desk would have sent on 2022-now signals (every A+ setup plus the weekly top-ups) when the
    (strategy, condition) pairs in `pairs` are paused. Returns (sent, A+, top-ups, weeks)."""
    sig = post
    if pairs:
        sig = post[[(a, b) not in pairs for a, b in zip(post["strategy"], post["condition"])]]
    s = _one_per_ticker(sig)
    sc, pw = float(G["aplus_score"]), float(G["aplus_pwin"])
    ap = s[(s["score"] >= sc) & (s["p"] >= pw)]
    if G.get("topup_ok", True) and G.get("topup_floor") is not None:
        pk, wk = simulate_topups(s, sc, pw, float(G["topup_floor"]), float(G.get("topup_pwin", 0)), min_week,
                                 list((G.get("topup_excluded") or {}).keys()))
    else:
        pk, wk = s.iloc[0:0], pd.Series(dtype=float)
    return pd.concat([ap, pk]), ap, pk, wk


def condition_report(trades: pd.DataFrame, split, G: dict, min_week: int = 5) -> dict:
    """Which strategies work in which market conditions, and whether pausing a strategy in a condition where
    it was weak on 2008-2021 would have made the 2022-now alerts better. A pause is adopted only if, on
    2022-now data it was not chosen on, the setups the desk sends win more often (+0.2 points or more) with an
    average trade at least 98% as good, keeping at least half of the trades."""
    t = _with_conditions(trades)
    split = pd.Timestamp(split)
    f = t[t["filled"] & t["ret"].notna()]
    table = {}
    for (name, cond), g in f.groupby(["strategy", "condition"]):
        table.setdefault(name, {})[cond] = {"train": _trade_stats(g[g["signal_date"] < split]),
                                            "oos": _trade_stats(g[g["signal_date"] >= split])}
    days = t.drop_duplicates("signal_date")
    share = {c: round(float(v), 3) for c, v in days["condition"].value_counts(normalize=True).items() if c}
    share_oos = {c: round(float(v), 3) for c, v in days[days["signal_date"] >= split]["condition"].value_counts(normalize=True).items() if c}
    post = t[t["signal_date"] >= split]
    out = {"rule": S.CONDITION_RULE, "labels": dict(S.CONDITION_LABEL), "share_of_days": share, "share_of_days_oos": share_oos,
           "table": table, "gates": {}, "gates_tested": []}
    if post.empty or not G.get("aplus_score"):
        out["all_weather"] = {name: S.all_weather(v, share) for name, v in table.items()}
        out["all_weather_rule"] = S.ALL_WEATHER_RULE
        return out
    sent, ap, pk, wk = _sent(post, G, min_week)
    out["aplus"] = {c: _trade_stats(g) for c, g in ap.groupby("condition")}
    out["topup"] = {c: _trade_stats(g) for c, g in pk.groupby("condition")}
    best = {}
    for cond in S.CONDITIONS:
        rows = [(n, (v.get(cond) or {}).get("oos") or {}) for n, v in table.items()]
        rows = [(n, o) for n, o in rows if (o.get("trades") or 0) >= 100 and (o.get("profit_factor") or 0) >= 1.15]
        best[cond] = [n for n, o in sorted(rows, key=lambda x: (-(x[1]["win_rate"] or 0), -(x[1]["avg_ret"] or 0)))]
    out["best"] = best
    # all-weather strategies hold up in every condition: they trade everywhere and are never paused on old data
    out["all_weather"] = {name: S.all_weather(v, share) for name, v in table.items()}
    out["all_weather_rule"] = S.ALL_WEATHER_RULE
    # pauses proposed by 2008-2021 alone, each kept only if it improves 2022-now
    cands = []
    for name, v in table.items():
        if out["all_weather"][name]["all_weather"]:
            continue
        for cond, d in v.items():
            tr_ = d.get("train") or {}
            if (tr_.get("trades") or 0) >= COND_GATE_TRAIN_N and ((tr_.get("win_rate") or 1) < COND_GATE_WIN
                                                                  or (tr_.get("profit_factor") or 9) < COND_GATE_PF):
                cands.append((tr_.get("win_rate") or 0, name, cond))
    base = _trade_stats(sent)
    adopted = set()
    for _, name, cond in sorted(cands):
        trial = frozenset(adopted | {(name, cond)})
        st = _trade_stats(_sent(post, G, min_week, trial)[0])
        ok = (base["trades"] and st["trades"] >= 0.5 * base["trades"] and st["win_rate"] >= base["win_rate"] + 0.002
              and (st["avg_ret"] or 0) >= 0.98 * (base["avg_ret"] or 0))
        trn = (table[name][cond].get("train") or {})
        out["gates_tested"].append({"strategy": name, "condition": cond, "adopted": bool(ok),
                                    "train": trn, "sent_before": base, "sent_after": st})
        if ok:
            adopted.add((name, cond))
            out["gates"].setdefault(name, {})[cond] = (
                f"won {100 * (trn.get('win_rate') or 0):.0f}% (profit factor {trn.get('profit_factor')}) in "
                f"{S.CONDITION_PLURAL[cond]} on 2008-2021; pausing it there lifted the 2022-now alerts from "
                f"{100 * base['win_rate']:.1f}% to {100 * st['win_rate']:.1f}% wins")
            base = st
    out["sent_oos"] = base
    return out


def _all_weather_line(pc: dict, P: dict) -> str:
    """One changelog line: which live strategies are all-weather and which depend on the market condition."""
    live = {n for n, v in (P.get("strategies") or {}).items() if v.get("enabled", True)}
    aw = pc.get("all_weather") or {}
    yes = [S.meta(n, P)["label"] for n, v in aw.items() if n in live and v.get("all_weather")]
    dep = [f"{S.meta(n, P)['label']} (weaker in " + " and ".join(
               S.CONDITION_PLURAL[c] + (f": {(v.get('weak_detail') or {})[c]}" if (v.get("weak_detail") or {}).get(c) else "")
               for c in v["weak"]) + ")"
           for n, v in aw.items() if n in live and not v.get("all_weather") and v.get("weak")]
    return ("All-weather (held up in every market condition, so they trade in any market and are never paused on old "
            "data): " + (", ".join(yes) or "none") + ". Condition-dependent: " + ("; ".join(dep) or "none") + ".")


def calibrate_grades(all_tr: pd.DataFrame, split, min_week=3, cal_end=CAL_END, topup_min_week=5,
                     prefer_win=False) -> dict:
    """Pick A+ cut-offs on the post-split trades (2022-now): the highest win rate whose average trade is at
    least +0.6% and that still produces enough setups per week across the universe.

    Honesty notes that travel with the result: the cut-offs were chosen on the same trades the headline
    figures describe, so `by_year` shows how the bar did in each calendar year, and `time_split` repeats the
    whole selection on the first half of the window (to cal_end) and measures that bar on the second half,
    which it never saw. Both are stored so the dashboard and the journal can show them."""
    post = all_tr[all_tr["signal_date"] >= split]
    fallback = {"aplus_score": 70, "aplus_pwin": 0.66, "min_week": min_week, "note": "fallback cut-offs"}
    best = _select_cutoffs(post, min_week, prefer_win)
    if best is None:
        return fallback
    wr, avg, sc, pw, per_week, n, cover = best
    t = post[post["ret"].notna() & post["filled"]]
    # realised win rate by score band: what the alert's Win % shows
    table = []
    for lo, hi in ((40, 54), (54, 62), (62, 66), (66, 70), (70, 74), (74, 78), (78, 82), (82, 86), (86, 90), (90, 101)):
        b = t[(t["score"] >= lo) & (t["score"] < hi) & (t["p"] >= pw)]
        if len(b) >= 40:
            table.append({"lo": lo, "hi": hi, "win": round(float((b["ret"] > 0).mean()), 4),
                          "avg": round(float(b["ret"].mean()), 5), "n": int(len(b))})
    table = _isotonic(table)
    ap = t[(t["score"] >= sc) & (t["p"] >= pw)]
    by_year = [{"year": int(y), "trades": int(len(g)), "win": round(float((g["ret"] > 0).mean()), 4),
                "avg": round(float(g["ret"].mean()), 5)} for y, g in ap.groupby(ap["signal_date"].dt.year)]
    out = {"aplus_score": sc, "aplus_pwin": pw, "min_week": min_week,
           "oos_aplus_win_rate": round(wr, 4), "oos_aplus_avg_ret": round(float(avg), 5),
           "oos_aplus_trades": int(n), "oos_aplus_signals_per_week": round(float(per_week), 1),
           "oos_weeks_with_min_setups": round(cover, 3), "score_win_table": table,
           "by_year": by_year, "aplus_bar_rule": "strictest near-best bar (A+ on watch)" if prefer_win else "near-best bar covering most weeks",
           "oos_window": [str(post["signal_date"].min().date()), str(post["signal_date"].max().date())]}
    if cal_end is not None:
        cal_end = pd.Timestamp(cal_end)
        early, late = post[post["signal_date"] < cal_end], post[post["signal_date"] >= cal_end]
        if len(late[late["filled"] & late["ret"].notna()]) >= 200 and len(early) and len(late):
            ts = {"chosen_bar_late": _aplus_window(late, sc, pw, min_week),
                  "chosen_bar_early": _aplus_window(early, sc, pw, min_week)}
            e_best = _select_cutoffs(early, min_week)
            if e_best is None:
                ts["early_selection"] = None
                ts["note"] = (f"Choosing the bar on {early['signal_date'].min().date()} to {cal_end.date()} alone found no cut-off "
                              "that cleared the +0.6% average-trade floor: setups earned little in that stretch.")
            else:
                e_wr, e_avg, e_sc, e_pw, _, e_n, _ = e_best
                ts["early_selection"] = {"aplus_score": e_sc, "aplus_pwin": e_pw, "trades": int(e_n), "win_rate": round(e_wr, 4),
                                         "avg_ret": round(float(e_avg), 5)}
                ts["early_bar_on_late"] = _aplus_window(late, e_sc, e_pw, min_week)
            out["time_split"] = ts
    out.update(calibrate_topups(post, sc, pw, topup_min_week, cal_end))
    return out


def run_backtest(data_dir, out_dir, split="2022-01-01", strategies=None, log=print, merge_state=None,
                 cal_end=CAL_END) -> dict:
    t0 = time.time()
    cal_end = pd.Timestamp(cal_end) if cal_end else None
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(split)
    oldP = {}
    if merge_state:
        old = load_state(merge_state).get("params_doc") or {}
        oldP = old.get("params") or old
    w = D.load_history(data_dir)
    universe = D.load_universe(data_dir)
    log(f"history: {w['close'].shape[0]} days x {w['close'].shape[1]} tickers "
        f"({w['close'].index.min().date()} to {w['close'].index.max().date()})")
    ind = I.compute(w, universe)
    log(f"indicators done in {time.time() - t0:.0f}s")
    P = S.params_with_defaults(None)
    pre_changes = []
    # the desk's weekly minimum (top-ups are replayed against it) and whether A+ is on watch
    desk_min, prefer_win = 5, False
    if merge_state:
        try:
            desk_min = int((load_state(merge_state).get("settings") or {}).get("min_setups_week", 5))
        except Exception:
            desk_min = 5
        prefer_win = bool((oldP.get("grading") or {}).get("aplus_watch"))
    if merge_state:  # learn from real fills: stop slippage worse than assumed -> backtest with yours
        try:
            lp = ((load_state(merge_state).get("learning") or {}).get("portfolio") or {}).get("doc") or {}
            ss = lp.get("stop_slippage") or {}
            if (ss.get("n") or 0) >= 10 and ss.get("avg_pct") is not None and -float(ss["avg_pct"]) > P["risk"]["slippage_pct"]:
                P["risk"]["slippage_pct"] = round(-float(ss["avg_pct"]), 3)
                pre_changes.append(f"Backtests now assume {P['risk']['slippage_pct']:.2f}% slippage on stops and exits, "
                                   f"matching your last {ss['n']} stop fills.")
        except Exception as ex:
            log(f"learning/portfolio unavailable: {ex!r}")
    report = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "split": str(split.date()),
              "calibration_end": str(cal_end.date()) if cal_end is not None else None,
              "period": [str(w["close"].index.min().date()), str(w["close"].index.max().date())],
              "tickers": int(w["close"].shape[1]), "strategies": {}}
    pooled = []

    def evaluate(name, res, spec=None):
        if spec is not None:  # a research strategy: its rules travel with the params
            P.setdefault("custom", {})[name] = {"spec": spec, "params": res["variant"], "label": spec.get("label"),
                                                "family": spec.get("family"), "about": spec.get("about"),
                                                "source": spec.get("source")}
            P["strategies"][name] = R.strategy_entry(spec, res["variant"], True)
            if spec.get("market_conditions"):  # a specialist: trades only in the market conditions it has proven itself in
                P["strategies"][name]["market_conditions"] = list(spec["market_conditions"])
        else:
            P["strategies"][name].update(res["variant"])
        tr = res["trades"]
        filled = tr[tr["filled"] & tr["ret"].notna()]
        tr_train, tr_test = tr[tr["signal_date"] < split], tr[tr["signal_date"] >= split]
        st_train, st_test, st_all = B.stats(tr_train), B.stats(tr_test), B.stats(tr)
        mdl = M.fit(filled, split_date=split)
        mdl["avg_hold"] = st_all.get("avg_hold")
        # p for every row from a train-only fit (honest OOS scores for grade calibration)
        train_only = M.fit(filled[filled["signal_date"] < split])
        feats = tr[S.FEATURES]
        tr = tr.assign(p=M.predict(train_only, feats))
        rr = ((tr["target"] - tr["fill"]) / (tr["fill"] - tr["stop"])).where(tr["filled"],
              (tr["target"] - tr["limit"]) / (tr["limit"] - tr["stop"]))
        tr = tr.assign(score=M.score(tr["p"].values, feats, train_only, rr.fillna(1).values))
        pooled.append(tr)
        enabled, checks, reason = S.enable_checks(st_train, st_test)
        oos_weeks = max(1.0, (ind["close"].index[-1] - split).days / 7.0)
        # honest per-strategy figures for display and the learning loop: out-of-sample
        if st_test.get("trades", 0) >= 100:
            mdl["base_rate"] = st_test["win_rate"]
            mdl["avg_win"], mdl["avg_loss"] = st_test["avg_win"], st_test["avg_loss"]
        P["strategies"][name]["enabled"] = bool(enabled)
        P.setdefault("models", {})[name] = mdl
        mt = S.meta(name, P)
        rec = {"enabled": bool(enabled), "variant": res["variant"], "train": st_train,
               "test_oos": st_test, "all": st_all, "yearly": B.yearly(tr).reset_index().to_dict("records"),
               "model_validation": mdl.get("validation"), "label": mt["label"], "family": mt["family"],
               "checks": checks, "status_reason": reason,
               "path": None if enabled else S.path_to_pass(st_train, st_test, oos_weeks)}
        if spec is not None:
            rec.update(custom=True, about=spec.get("about"), source=spec.get("source"),
                       rules=R.describe({**spec, "params": res["variant"], "grid": {}}),
                       market_conditions=list(spec.get("market_conditions") or []) or None)
        report["strategies"][name] = rec
        log(f"  {name}: train win {st_train.get('win_rate')} avg {st_train.get('avg_ret')} PF "
            f"{st_train.get('profit_factor')} | OOS win {st_test.get('win_rate')} avg {st_test.get('avg_ret')} "
            f"PF {st_test.get('profit_factor')} n={st_test.get('trades')} | enabled={enabled}")

    for name in (strategies or list(GRIDS)):
        log(f"searching {name} ...")
        res = search_strategy(name, ind, P, split, log)
        if res is None:
            P["strategies"][name]["enabled"] = False
            report["strategies"][name] = {"enabled": False, "reason": "no trades", "status_reason": "Off: no trades in the backtest.",
                                          "label": S.STRATEGY_META[name]["label"], "family": S.STRATEGY_META[name]["family"]}
            continue
        evaluate(name, res)
    # research strategies: this week's candidates plus every research strategy promoted before
    research_docs = []
    if merge_state and not strategies:
        from .research import candidate_specs, evaluate_spec, retest_specs, search_spec
        from .scan import load_params as _load_live
        # ideas that fail only on 2022-now checks can pass as new data arrives: re-test them every week
        try:
            P_live, P_r, now_r = _load_live(load_state(merge_state), None), S.params_with_defaults(None), cal.now_pt()
            for spec in retest_specs(merge_state):
                d = evaluate_spec(spec, ind, P_r, P_live, split, now_r, by="weekly re-test", log=log)
                research_docs.append(d)
                log(f"  re-test {spec['id']}: {d['verdict']}")
        except Exception as ex:
            log(f"research re-tests failed: {ex!r}")
        try:
            cands = candidate_specs(merge_state, oldP, extra_results=research_docs)
        except Exception as ex:
            cands = {}
            log(f"research candidates unavailable: {ex!r}")
        for cid, spec in cands.items():
            errs = R.validate(spec)
            if errs:
                log(f"  skipping research strategy {cid}: {errs}")
                continue
            log(f"searching research strategy {cid} ...")
            try:
                res = search_spec(spec, ind, P, split, log)
            except Exception as ex:
                log(f"  {cid} failed: {ex!r}")
                continue
            if res is not None:
                evaluate(cid, res, spec)
    allp = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
    if not allp.empty:
        enabled_names = [n for n, v in report["strategies"].items() if v.get("enabled")]

        def fit_and_grade(names):
            en_ = allp[allp["strategy"].isin(names)].copy()
            pm_ = None
            # pooled robust-factor model: fit on pre-split trades, win odds recalibrated out of sample
            try:
                pm_ = M.fit_pooled(en_, names, split, cal_end)
                p_, e_, ep_ = M.predict_pooled(pm_, en_[S.FEATURES], en_["strategy"].values)
                en_["p"], en_["e"], en_["e_pct"] = p_, e_, ep_
                en_["score"] = M.score_pooled(p_, ep_)
            except Exception as ex:  # keep the per-strategy scores if the pooled fit fails
                log(f"pooled model failed: {ex!r}")
                pm_ = None
            return en_, pm_, calibrate_grades(en_, split, cal_end=cal_end, topup_min_week=desk_min, prefer_win=prefer_win)

        en, pm, G = fit_and_grade(enabled_names)
        # research strategies must earn their place inside the A+ set, not just pass on their own
        research_on = [n for n in enabled_names if (P["strategies"].get(n) or {}).get("custom")]
        if research_on:
            base_names = [n for n in enabled_names if n not in research_on]
            en0, pm0, G0 = fit_and_grade(base_names) if base_names else (None, None, {})
            oos_ = en[(en["signal_date"] >= split) & en["ret"].notna() & en["filled"]]
            ap_ = oos_[(oos_["score"] >= G["aplus_score"]) & (oos_["p"] >= G["aplus_pwin"])]
            a_win = float(G.get("oos_aplus_win_rate") or 0)
            per, weak = {}, []
            for n in research_on:
                sub = ap_[ap_["strategy"] == n]
                w_ = float((sub["ret"] > 0).mean()) if len(sub) else None
                per[n] = {"aplus_trades_oos": int(len(sub)), "aplus_win_oos": round(w_, 4) if w_ is not None else None,
                          "aplus_avg_oos": round(float(sub["ret"].mean()), 5) if len(sub) else None}
                if len(sub) >= 30 and w_ < a_win - 0.03:
                    weak.append(n)
            # keep or raise the A+ win rate and average trade without thinning out the weekly setups
            better = (not G0) or (a_win >= float(G0.get("oos_aplus_win_rate") or 0) - 0.005
                                  and float(G.get("oos_aplus_avg_ret") or 0) >= float(G0.get("oos_aplus_avg_ret") or 0) - 0.0005
                                  and float(G.get("oos_aplus_signals_per_week") or 0) >= 0.8 * float(G0.get("oos_aplus_signals_per_week") or 0))
            drop = research_on if not better else weak
            gkeys = ("aplus_score", "aplus_pwin", "oos_aplus_win_rate", "oos_aplus_avg_ret", "oos_aplus_trades",
                     "oos_aplus_signals_per_week", "oos_weeks_with_min_setups", "topup_floor", "topup_pwin",
                     "oos_topup_win_rate", "oos_topup_avg_ret", "oos_topup_trades", "oos_weeks_at_min")
            for n in research_on:  # each research strategy's own top-ups, judged apart from its A+ setups
                tb = (G.get("topup_by_strategy") or {}).get(n) or {}
                per[n].update(topup_trades_oos=tb.get("trades", 0), topup_win_oos=tb.get("win_rate"),
                              topup_avg_oos=tb.get("avg_ret"), topup_excluded=n in (G.get("topup_excluded") or {}))
            gate = {"with_research": {k: G.get(k) for k in gkeys}, "without_research": {k: G0.get(k) for k in gkeys},
                    "per_strategy": per, "dropped": drop}
            for n in drop:
                pn = per.get(n) or {}
                why = (f"Off: it passed every standalone check, but adding the research strategies didn't improve the A+ set "
                       f"on unseen data ({100 * a_win:.1f}% wins with them vs {100 * float(G0.get('oos_aplus_win_rate') or 0):.1f}% without)."
                       if not better else
                       f"Off: it passed every standalone check, but its A+-grade trades won only {100 * (pn.get('aplus_win_oos') or 0):.1f}% "
                       f"on unseen data vs {100 * a_win:.1f}% for the whole A+ set, so it would dilute the alerts.")
                P["strategies"][n]["enabled"] = False
                report["strategies"][n].update(enabled=False, status_reason=why + " Re-tested every Saturday.")
                log(f"  research gate: {n} off ({why})")
            if drop:
                enabled_names = [n for n in enabled_names if n not in drop]
                en, pm, G = (en0, pm0, G0) if set(drop) == set(research_on) and en0 is not None else fit_and_grade(enabled_names)
            for n in research_on:
                if n not in drop:
                    report["strategies"][n]["aplus_oos"] = per.get(n)
            report["research_gate"] = gate
        if pm is not None:
            P["models"]["_pooled"] = pm
            report["pooled_model"] = pm["validation"]
            log(f"pooled model: OOS AUC {pm['validation']['oos_auc']}, return corr {pm['validation']['oos_ret_corr']}")
        allp = en
        P["grading"].update(G)
        report["grading"] = G
        try:  # which strategies work in which market conditions, and validated pauses
            cr = condition_report(en, split, P["grading"], desk_min)
            report["conditions"] = cr
            P["conditions"] = {**{k: v for k, v in cr.items() if k != "gates_tested"},
                               "gates_tested": [{k: g[k] for k in ("strategy", "condition", "adopted")} for g in cr["gates_tested"]],
                               "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "source": "weekly recalibration"}
        except Exception as ex:
            log(f"market-condition report failed: {ex!r}")
        # score buckets (validates that a higher score = better setup)
        oos = en[(en["signal_date"] >= split) & en["ret"].notna()].copy()
        if len(oos) > 50:
            oos["bucket"] = pd.cut(oos["score"], [0, 50, 60, 70, 80, 90, 101], right=False)
            sb = oos.groupby("bucket", observed=True)["ret"].agg(trades="size", win_rate=lambda s: (s > 0).mean(), avg_ret="mean")
            report["score_buckets_oos"] = [{"bucket": str(k), **{kk: round(float(vv), 4) for kk, vv in r.items()}}
                                           for k, r in sb.iterrows()]
        # portfolio simulation, A+ only, $5k, 5 slots
        aplus = en[(en["score"] >= G["aplus_score"]) & (en["p"] >= G["aplus_pwin"])]
        for label, sub in (("all_years", aplus), ("oos", aplus[aplus["signal_date"] >= split]),
                           ("late", aplus[aplus["signal_date"] >= cal_end] if cal_end is not None else aplus.iloc[0:0])):
            if sub.empty:
                continue
            pf = B.portfolio(sub, 5000, P["risk"]["max_open"], P["risk"]["risk_pct"], P["risk"]["max_position_pct"])
            cv = pf.pop("curve", None)
            report.setdefault("portfolio", {})[label] = pf
            if label == "all_years" and cv is not None and len(cv):
                cv.assign(date=cv["date"].astype(str)).to_csv(out / "equity_curve.csv", index=False)
        allp.drop(columns=[c for c in ("t", "j", "fill_i", "exit_i") if c in allp]).to_parquet(out / "trades.parquet")
    P["version"] = int(time.time())
    P["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    P["source"] = "backtest"
    changes = list(pre_changes)
    if research_docs:
        report["research_retests"] = [{"idea": d.get("idea_id"), "verdict": d.get("verdict"), "summary": d.get("summary")}
                                      for d in research_docs]
        for d in research_docs:
            if d.get("verdict") == "candidate":
                changes.append(f"Research idea {d.get('label')} became a candidate on this week's data.")
    if merge_state:
        try:  # each grade's own live record, on top of this week's test
            from . import learnlog as LL_
            st_ = ((load_state(merge_state).get("learning") or {}).get("stats") or {}).get("doc")
            pol = LL_.topup_policy(st_, P["grading"])
            ex_live = LL_.topup_strategy_exclusions(st_, P["grading"])
            aw = LL_.aplus_watch(st_, P["grading"])
            P["grading"].update(topup_mode=pol["mode"], topup_excluded_live=ex_live)
            if aw:
                P["grading"]["aplus_watch"] = aw
            else:
                P["grading"].pop("aplus_watch", None)
            if pol["note"]:
                changes.append(pol["note"])
            for n_, why_ in ex_live.items():
                changes.append(f"{S.meta(n_, P)['label']} sends no top-ups for now: {why_}.")
            if aw:
                changes.append(aw["note"])
            if P.get("conditions"):
                gl = LL_.condition_gates_live(st_, P["conditions"])
                P["conditions"]["gates_live"] = gl
                for n_, cs_ in gl.items():
                    for c_, why_ in cs_.items():
                        changes.append(f"{S.meta(n_, P)['label']} paused in {S.CONDITION_PLURAL[c_]}: {why_}.")
        except Exception as ex:
            log(f"grade records unavailable: {ex!r}")
        oc_, nc_ = (oldP.get("conditions") or {}).get("gates") or {}, (P.get("conditions") or {}).get("gates") or {}
        pairs_ = lambda d: {(n, c) for n, v in d.items() for c in v}
        if pairs_(oc_) != pairs_(nc_):
            changes.append("Market-condition pauses validated on 2022-now data: " + (
                "; ".join(f"{S.meta(n_, P)['label']} in {S.CONDITION_PLURAL[c_]}" for n_, c_ in sorted(pairs_(nc_))) or "none") + ".")
        if (P.get("conditions") or {}).get("all_weather"):
            changes.append(_all_weather_line(P["conditions"], P))
        og_, ng_ = oldP.get("grading") or {}, P["grading"]
        if ng_.get("topup_ok") is False:
            changes.append(ng_.get("topup_note") or "Weekly top-ups are off: no top-up rule cleared the floors.")
        elif (og_.get("topup_floor"), og_.get("topup_pwin")) != (ng_.get("topup_floor"), ng_.get("topup_pwin")):
            changes.append(f"Weekly top-up bar: score {og_.get('topup_floor')} -> {ng_.get('topup_floor')}, model win "
                           f"{og_.get('topup_pwin', '-')} -> {ng_.get('topup_pwin')} (replayed top-ups won "
                           f"{100 * (ng_.get('oos_topup_win_rate') or 0):.1f}% of {ng_.get('oos_topup_trades')} on 2022-now data).")
        if set(og_.get("topup_excluded") or {}) != set(ng_.get("topup_excluded") or {}):
            changes.append("Strategies that send no top-ups (their own top-ups trailed on 2022-now data): "
                           + (", ".join(S.meta(n_, P)["label"] for n_ in ng_.get("topup_excluded") or {}) or "none") + ".")
        if oldP.get("live_adjust"):
            P["live_adjust"] = oldP["live_adjust"]  # keep what live trading has taught us
        for k in ("risk",):
            if isinstance(oldP.get(k), dict):
                P[k].update(oldP[k])
        for n, sp in P["strategies"].items():
            o = (oldP.get("strategies") or {}).get(n) or {}
            lab = S.meta(n, P)["label"]
            why = (report["strategies"].get(n) or {}).get("status_reason", "")
            if sp.get("custom") and not o:
                changes.append(f"Promoted research strategy {lab}: passed the recalibration and now trades live. {why}"
                               if sp.get("enabled") else f"Research candidate {lab} did not pass the recalibration. {why}")
                continue
            if o and o.get("enabled") != sp.get("enabled"):
                changes.append(f"{lab}: {'enabled' if sp.get('enabled') else 'disabled'} after recalibration. {why}")
            diff = {k: v for k, v in sp.items() if k not in ("enabled", "paused_reason") and o.get(k) != v}
            if o and diff:
                changes.append(f"{lab}: parameters now {json.dumps(diff, default=str)}.")
        og, ng = oldP.get("grading") or {}, P["grading"]
        if og and (og.get("aplus_score") != ng.get("aplus_score") or og.get("aplus_pwin") != ng.get("aplus_pwin")):
            changes.append(f"A+ cut-offs: score {og.get('aplus_score')} -> {ng.get('aplus_score')}, "
                           f"win {og.get('aplus_pwin')} -> {ng.get('aplus_pwin')}.")
    report["changes"] = changes
    (out / "params.json").write_text(json.dumps(P, indent=1, default=str))
    (out / "backtest_report.json").write_text(json.dumps(report, indent=1, default=str))
    # dashboard documents
    cv_p = out / "equity_curve.csv"
    curve = []
    if cv_p.exists():
        cv = pd.read_csv(cv_p)
        step = max(1, len(cv) // 300)
        curve = cv.iloc[::step].round(2).to_dict("records")
    bt_doc = {k: v for k, v in report.items()}
    bt_doc["equity_curve"] = curve
    (out / "doc_backtest.json").write_text(json.dumps(bt_doc, default=str))
    (out / "doc_params.json").write_text(json.dumps({"params": P, "updated_at": P["updated"],
                                                     "changelog_entry": changes or ["Weekly recalibration: no material changes."]}, default=str))
    # re-tested research ideas: new (create-only) result documents
    rw = []
    for d in research_docs:
        fp = out / "docs" / "research_results" / f"{d['id']}.json"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(d, default=str))
        rw.append({"op": "set", "collection": "research_results", "doc_id": d["id"], "file_path": str(fp.resolve())})
    (out / "research_writes.json").write_text(json.dumps([rw[i:i + 45] for i in range(0, len(rw), 45)], indent=1))
    log(f"backtest finished in {time.time() - t0:.0f}s")
    return report


# ----------------------------------------------------------------------------- learning

def shadow_outcome(a: dict, ind: dict) -> dict:
    """What happened to an entry alert if it had been placed exactly as sent."""
    tk = a.get("ticker")
    c = ind["close"]
    res = {"alert_id": a.get("id"), "ticker": tk, "strategy": a.get("strategy"), "status": "unknown"}
    if tk not in c.columns:
        return res
    j = c.columns.get_loc(tk)
    idx = c.index
    try:
        vf = pd.Timestamp(a.get("valid_for"))
    except Exception:
        return res
    if vf not in idx:
        res["status"] = "pending" if vf > idx[-1] else "no_session"
        return res
    d0 = idx.get_loc(vf)
    O, H, L, C = (ind[k].values[:, j] for k in ["open", "high", "low", "close"])
    entry, stop, tgt = float(a["entry"]), float(a["stop"]), float(a["target"])
    hold = int(a.get("hold_max_days") or 7)
    dyn = a.get("dyn_exit") or "none"
    at_close = str(a.get("order", "")).lower().startswith("buy near the close")
    valid = max(1, int(a.get("valid_days") or 1))
    # fill: a buy-near-the-close alert fills at that day's close; a limit fills on the first session
    # (within its validity) whose low trades through the limit (a touch is not a fill, as in the backtest),
    # at the better of the open and the limit
    fill_i, fill = -1, float("nan")
    if at_close:
        fill_i, fill = d0, float(C[d0])
    else:
        for d in range(d0, min(d0 + valid, len(idx))):
            if np.isfinite(L[d]) and L[d] < entry:
                fill_i, fill = d, float(min(O[d], entry))
                break
    if fill_i < 0:
        res["status"] = "unfilled" if d0 + valid - 1 <= len(idx) - 1 else "pending"
        return res
    sma5, rsi2, max7 = ind["sma5"].values[:, j], ind["rsi2"].values[:, j], ind["max7c"].values[:, j]
    held = 0
    start = fill_i + 1 if at_close else fill_i
    for d in range(start, len(idx)):
        held += 1
        if d == fill_i:
            if L[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": stop, "reason": "stop", "held": held}
        else:
            if O[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": O[d], "reason": "stop", "held": held}
            if O[d] >= tgt:
                return {**res, "status": "closed", "fill": fill, "exit": O[d], "reason": "target", "held": held}
            if L[d] <= stop:
                return {**res, "status": "closed", "fill": fill, "exit": stop, "reason": "stop", "held": held}
            if H[d] >= tgt:
                return {**res, "status": "closed", "fill": fill, "exit": tgt, "reason": "target", "held": held}
        sig = ((dyn == "sma5" and C[d] > sma5[d]) or (dyn == "prev_high" and d > 0 and C[d] > H[d - 1])
               or (dyn == "high7" and C[d] >= max7[d]) or (dyn == "rsi2_70" and rsi2[d] > 70))
        if sig:
            return {**res, "status": "closed", "fill": fill, "exit": C[d], "reason": "signal", "held": held}
        if held >= hold:
            return {**res, "status": "closed", "fill": fill, "exit": C[d], "reason": "time", "held": held}
    return {**res, "status": "open", "fill": fill, "last": C[-1], "held": held}


def run_topups(data_dir, state_dir, out_dir, split="2022-01-01", log=print, now=None) -> dict:
    """Re-tunes the weekly top-ups between recalibrations, with the live strategies and the live model:
    replays every live strategy's signals since 2008 exactly as the scanner scores them, then runs the same
    top-up calibration as the weekly recalibration and applies each grade's live record. Writes
    doc_params.json (live params with the new top-up rule, for strategy/params) and topups_report.json."""
    from .scan import load_params
    from . import learnlog as LL
    now = now or cal.now_pt()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state(state_dir)
    P = load_params(state, None)
    pm = (P.get("models") or {}).get("_pooled")
    if not (pm and pm.get("kind") == "pooled"):
        raise SystemExit("live params have no pooled model: run the weekly recalibration instead")
    w = D.load_history(data_dir)
    ind = I.compute(w, D.load_universe(data_dir))
    plans = S.all_plans(ind, P)
    trs = []
    for name, sp in P["strategies"].items():
        if not sp.get("enabled", True) or name not in plans:
            continue
        tr = B.simulate(plans[name], ind, P, include_unfilled=True)
        log(f"  {name}: {len(tr)} signals")
        if not tr.empty:
            trs.append(tr)
    t = pd.concat(trs, ignore_index=True)
    p_, e_, ep_ = M.predict_pooled(pm, t[S.FEATURES], t["strategy"].values)
    t["p"], t["score"] = p_, M.score_pooled(p_, ep_)
    post = t[t["signal_date"] >= pd.Timestamp(split)]
    G = P["grading"]
    desk_min = int((state.get("settings") or {}).get("min_setups_week", 5))
    old = {k: G.get(k) for k in ("topup_floor", "topup_pwin", "topup_excluded", "topup_mode", "oos_topup_win_rate")}
    res = calibrate_topups(post, float(G["aplus_score"]), float(G["aplus_pwin"]), desk_min, CAL_END)
    ap = _one_per_ticker(post)
    ap = ap[(ap["score"] >= G["aplus_score"]) & (ap["p"] >= G["aplus_pwin"])]
    try:
        Gc = {**G, **res}
        cr = condition_report(t, split, Gc, desk_min)
        P["conditions"] = {**{k: v for k, v in cr.items() if k != "gates_tested"},
                           "gates_tested": [{k: g[k] for k in ("strategy", "condition", "adopted")} for g in cr["gates_tested"]],
                           "updated": now.isoformat(timespec="seconds"), "source": "mid-week re-tune (live model)",
                           "gates_live": LL.condition_gates_live(((state.get("learning") or {}).get("stats") or {}).get("doc"), cr)}
    except Exception as ex:
        log(f"market-condition report failed: {ex!r}")
    res["aplus_check_live_model"] = {**_trade_stats(ap),  # the A+ bar in the same replay, for comparison
                                     "by_year": [{"year": int(y), **_trade_stats(g)} for y, g in ap.groupby(ap["signal_date"].dt.year)]}
    G.update(res)
    st_ = ((state.get("learning") or {}).get("stats") or {}).get("doc")
    pol = LL.topup_policy(st_, G)
    G.update(topup_mode=pol["mode"], topup_excluded_live=LL.topup_strategy_exclusions(st_, G))
    changes = []
    if not res.get("topup_ok"):
        changes.append(res.get("topup_note"))
    else:
        same = (old["topup_floor"], old["topup_pwin"]) == (res["topup_floor"], res["topup_pwin"])
        changes.append(
            f"Weekly top-ups re-tuned on their own (live strategies and model, replaying the exact {desk_min}-a-week rule on "
            f"{res['topup_by_year'][0]['year']}-now signals): "
            + (f"bar unchanged at score {res['topup_floor']}+ and model win {100 * res['topup_pwin']:.0f}%+. " if same else
               f"bar score {old['topup_floor']} -> {res['topup_floor']}, model win "
               f"{old['topup_pwin'] if old['topup_pwin'] is not None else 'not set'} -> {res['topup_pwin']:.2f}. ") +
            f"Replayed top-ups won {100 * res['oos_topup_win_rate']:.1f}% of {res['oos_topup_trades']} (average "
            f"{100 * res['oos_topup_avg_ret']:+.2f}%, profit factor {res['oos_topup_pf']}) and kept "
            f"{100 * res['oos_weeks_at_min']:.0f}% of weeks at {desk_min}+ setups (A+ alone: "
            f"{100 * (res['oos_weeks_at_min_aplus_only'] or 0):.0f}%).")
        if res.get("topup_excluded"):
            changes.append("No top-ups from: " + "; ".join(f"{S.meta(n, P)['label']} ({why})" for n, why in res["topup_excluded"].items()) + ".")
    if pol["note"]:
        changes.append(pol["note"])
    pc = P.get("conditions") or {}
    if pc.get("table"):
        tested = pc.get("gates_tested") or []
        changes.append("Market conditions learned for every live strategy (2008-2021 and 2022-now, per condition). "
                       + (f"Pauses adopted (they lifted 2022-now alerts): " + "; ".join(
                           f"{S.meta(n, P)['label']} in {S.CONDITION_PLURAL[c]}" for n, v in pc["gates"].items() for c in v)
                          if pc.get("gates") else
                          f"No pause adopted: {len(tested)} proposed by 2008-2021 results, none improved 2022-now alerts."))
        changes.append(_all_weather_line(pc, P))
    P["version"] = int(time.time())
    P["updated"] = now.isoformat(timespec="seconds")
    (out / "doc_params.json").write_text(json.dumps({"params": P, "updated_at": now.isoformat(timespec="seconds"),
                                                     "changelog_entry": changes}, default=str))
    (out / "topups_report.json").write_text(json.dumps({"before": old, "after": res, "changes": changes}, indent=1, default=str))
    return {"changes": changes, "topup_floor": res.get("topup_floor"), "topup_pwin": res.get("topup_pwin"),
            "win_rate": res.get("oos_topup_win_rate"), "trades": res.get("oos_topup_trades"),
            "weeks_at_min": res.get("oos_weeks_at_min"), "excluded": list((res.get("topup_excluded") or {}).keys())}


def run_learning(data_dir, state_dir, out_dir, now=None) -> dict:
    now = now or cal.now_pt()
    out = pathlib.Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    state = load_state(state_dir)
    from .scan import load_params, _write
    P = load_params(state, None)
    w = D.load_daily(data_dir)
    ind = I.compute({k: v.iloc[-400:] for k, v in w.items()}, D.load_universe(data_dir))
    alerts = [v["doc"] for v in state["alerts"].values() if v["doc"].get("type") == "entry"]
    shadow = [shadow_outcome(a, ind) for a in alerts]
    sh = pd.DataFrame([s for s in shadow if s.get("status") == "closed"])
    if not sh.empty:
        sh["ret"] = sh["exit"] / sh["fill"] - 1
    trades = [v["doc"] for v in state["trades"].values()]
    from .portfolio import merged_closed_trades
    closed_list = merged_closed_trades(trades)  # a position sold in parts counts once
    closed = pd.DataFrame(closed_list)
    if not closed.empty:
        closed["ret"] = closed["exit_price"].astype(float) / closed["entry_price"].astype(float) - 1
    from . import learnlog as LL
    prev_stats = ((state.get("learning") or {}).get("stats") or {}).get("doc")
    # the watchlist: next-best setups the scans listed but did not send, graded on paper
    scans = [v["doc"] for v in (state.get("scans") or {}).values()]
    watch = LL.watch_from_scans(scans, alerts)
    wshadow = [shadow_outcome(a, ind) for a in watch]
    stats = LL.stats_doc(prev_stats, alerts, shadow, closed_list, len(state["alerts"]) < 1000, now, watch, wshadow)
    by = {}
    changes, insights = [], []
    for name in P["strategies"]:
        mdl = P.get("models", {}).get(name) or {}
        base = float(mdl.get("base_rate", 0.65))
        s_sub = sh[sh["strategy"] == name] if not sh.empty else pd.DataFrame()
        l_sub = closed[closed["strategy"] == name] if not closed.empty else pd.DataFrame()
        sn, sw = len(s_sub), int((s_sub["ret"] > 0).sum()) if len(s_sub) else 0
        ln, lw = len(l_sub), int((l_sub["ret"] > 0).sum()) if len(l_sub) else 0
        # the permanent record (every month ever graded, live fills double, 6-month half-life)
        n_w, w_w = LL.weighted_record(stats["months"], name, now)
        n_eff, w_eff = int(round(n_w)), int(round(w_w))
        exp_p = np.mean([a.get("win_pct", 100 * base) for a in alerts if a.get("strategy") == name] or [100 * base]) / 100
        old = float(P.get("live_adjust", {}).get(name, {}).get("offset", 0.0))
        new = M.live_offset(base, w_eff, n_eff) if n_eff else 0.0
        by[name] = {"shadow_n": sn, "shadow_wins": sw, "live_n": ln, "live_wins": lw,
                    "expected_win": round(exp_p, 3), "base_rate": base, "offset_old": round(old, 3),
                    "offset_new": round(new, 3), "record_n": n_eff, "record_wins": w_eff,
                    "shadow_avg_ret": round(float(s_sub["ret"].mean()), 4) if sn else None,
                    "live_avg_ret": round(float(l_sub["ret"].mean()), 4) if ln else None}
        if abs(new - old) >= 0.02:
            P.setdefault("live_adjust", {})[name] = {"offset": round(new, 3), "n": n_eff, "wins": w_eff,
                                                     "updated": now.isoformat(timespec="seconds")}
            changes.append(f"{S.meta(name, P)['label']}: win-probability offset {old:+.2f} -> {new:+.2f} "
                           f"(record {w_eff}/{n_eff} weighted vs {base:.0%} backtest).")
        if n_eff >= 25:
            wr = w_eff / n_eff
            if wr < exp_p - 0.12:
                insights.append(f"{S.meta(name, P)['label']} is under-performing: {wr:.0%} live/shadow vs "
                                f"{exp_p:.0%} expected over {n_eff} weighted trades. Scores are being discounted.")
            if wr < 0.5 and P["strategies"][name].get("enabled", True):
                P["strategies"][name]["enabled"] = False
                P["strategies"][name]["paused_reason"] = (f"Paused by live results: {wr:.0%} win rate over {n_eff} weighted "
                                                          f"trades (live fills count double) is below the 50% floor, "
                                                          f"{now.date().isoformat()}.")
                changes.append(f"Paused {S.meta(name, P)['label']}: {wr:.0%} win rate over {n_eff} trades "
                               "is below the 50% floor. It will be re-tested at the weekly recalibration.")
    # A+ setups, weekly top-ups and the watchlist are graded apart; each grade's rules follow its own record
    gs = stats.get("grade_summary") or {}
    G_ = P["grading"]
    exp_ap, exp_tu = G_.get("oos_aplus_win_rate"), LL._expected_topup(G_)

    def _line(g, r, tested):
        if not r.get("n") and not r.get("live_n"):
            return None
        bits = [f"{LL.GRADE_LABEL[g]}: {r['wins']} of {r['n']} graded won" + (f" ({r['win_rate']:.0%}, average {100 * (r['avg_ret'] or 0):+.2f}%)" if r.get("n") else "")]
        if r.get("live_n"):
            bits.append(f"your trades {r['live_wins']} of {r['live_n']}")
        if tested:
            bits.append(f"tested {tested:.0%}")
        if (r.get("weighted_n") or 0) < (LL.APLUS_MIN_N if g == "aplus" else LL.TOPUP_MIN_N) and g != "watch":
            bits.append("too few results yet for the record to change the rules")
        return "; ".join(bits) + "."
    for g, tested in (("aplus", exp_ap), ("topup", exp_tu if G_.get("topup_ok", True) else None), ("watch", None)):
        ln = _line(g, gs.get(g) or {}, tested)
        if ln:
            insights.append(ln)
    pol = LL.topup_policy(stats, G_)
    old_mode = G_.get("topup_mode") if G_.get("topup_mode") in ("normal", "tight", "off") else "normal"
    if pol["mode"] != old_mode:
        G_["topup_mode"] = pol["mode"]
        changes.append(pol["note"] or (f"Weekly top-ups back to their normal rule: their record recovered to "
                                       f"{(pol['record'] or 0):.0%} over {pol['n']:.0f} weighted results."))
    ex_live = LL.topup_strategy_exclusions(stats, G_)
    old_ex = G_.get("topup_excluded_live") or {}
    if set(ex_live) != set(old_ex):
        G_["topup_excluded_live"] = ex_live
        for n_ in set(ex_live) - set(old_ex):
            changes.append(f"{S.meta(n_, P)['label']} sends no top-ups for now: {ex_live[n_]}.")
        for n_ in set(old_ex) - set(ex_live):
            changes.append(f"{S.meta(n_, P)['label']} can send top-ups again: its top-up record recovered.")
    aw = LL.aplus_watch(stats, G_)
    if bool(aw) != bool(G_.get("aplus_watch")):
        if aw:
            G_["aplus_watch"] = aw
            changes.append(aw["note"])
        else:
            G_.pop("aplus_watch", None)
            changes.append("A+ record back in line with its test: the A+ watch is lifted.")
    # market conditions: today's, how each grade and strategy has done in it, and live pauses
    mk = ind["market"].iloc[-1]
    cond_today = S.condition_of(mk.get("spy_above200"), mk.get("vix_ratio"), mk.get("spy_ret5"))
    PC = P.get("conditions") or {}
    if cond_today:
        lab = S.CONDITION_LABEL[cond_today]
        bits = []
        for g, nm in (("aplus", "A+"), ("topup", "top-ups")):
            d = ((gs.get(g) or {}).get("by_condition") or {}).get(cond_today) or {}
            if d.get("n"):
                bits.append(f"{nm} {d['wins']} of {d['n']} won here")
        best = [S.meta(n_, P)["label"] for n_ in ((PC.get("best") or {}).get(cond_today) or [])[:3]]
        tested = (PC.get("aplus") or {}).get(cond_today) or {}
        aw_t = [S.meta(n_, P)["label"] for n_, v_ in (PC.get("all_weather") or {}).items()
                if v_.get("all_weather") and ((P.get("strategies") or {}).get(n_) or {}).get("enabled", True)]
        insights.append(f"Today's market: {lab.lower()}. " + (f"Your record in it: {', '.join(bits)}. " if bits else "")
                        + (f"Tested 2022-now in this condition: A+ setups won {100 * tested['win_rate']:.0f}% of {tested['trades']}. " if tested.get("trades") else "")
                        + (f"Strongest strategies here: {', '.join(best)}. " if best else "")
                        + (f"All-weather in testing (they hold up in every condition, so they keep trading in any market): {', '.join(aw_t)}." if aw_t else ""))
    aw_live = LL.all_weather_live(stats)
    for n_, v_ in aw_live.items():
        tested_aw = ((PC.get("all_weather") or {}).get(n_) or {}).get("all_weather")
        if v_["all_weather"] and len(v_["tested"]) >= 3:
            insights.append(f"{S.meta(n_, P)['label']} is proving all-weather live: it held up in "
                            f"{', '.join(S.CONDITION_PLURAL[c] for c in v_['tested'])}"
                            + (" (as tested)." if tested_aw else " (testing had it condition-dependent)."))
        elif v_["weak"] and tested_aw:
            insights.append(f"{S.meta(n_, P)['label']} tested all-weather but its own record lags in "
                            f"{' and '.join(S.CONDITION_PLURAL[c] for c in v_['weak'])}: watched, paused there only on clear evidence.")
    if PC.get("table"):
        gl = LL.condition_gates_live(stats, PC)
        old_gl = PC.get("gates_live") or {}
        flat = lambda d: {(n_, c_) for n_, v_ in d.items() for c_ in v_}
        if flat(gl) != flat(old_gl):
            P["conditions"]["gates_live"] = gl
            for n_, c_ in sorted(flat(gl) - flat(old_gl)):
                changes.append(f"{S.meta(n_, P)['label']} paused in {S.CONDITION_PLURAL[c_]}: {gl[n_][c_]}.")
            for n_, c_ in sorted(flat(old_gl) - flat(gl)):
                changes.append(f"{S.meta(n_, P)['label']} trades again in {S.CONDITION_PLURAL[c_]}: its record there recovered.")
    journal_grades = {g: {k: v for k, v in (gs.get(g) or {}).items() if k not in ("exits",)} for g in LL.GRADES}
    for g in journal_grades.values():
        g["by_strategy"] = {n_: {k: r.get(k) for k in ("n", "wins", "win_rate", "avg_ret", "live_n", "live_wins")}
                            for n_, r in (g.get("by_strategy") or {}).items()}
    by_cond = {n_: r.get("by_condition") for n_, r in (stats.get("summary") or {}).items() if r.get("by_condition")}
    grade_rules = {"market_condition": cond_today, "aplus_score": G_.get("aplus_score"), "aplus_pwin": G_.get("aplus_pwin"),
                   "topup_floor": G_.get("topup_floor"), "topup_pwin": G_.get("topup_pwin"),
                   "topup_mode": G_.get("topup_mode", "normal"), "topup_ok": G_.get("topup_ok", True),
                   "topup_excluded": sorted((G_.get("topup_excluded") or {}).keys()),
                   "topup_excluded_live": sorted((G_.get("topup_excluded_live") or {}).keys()),
                   "aplus_watch": bool(G_.get("aplus_watch")), "tested_aplus": exp_ap, "tested_topup": exp_tu}
    # feature diagnostics on recent shadow results
    if not sh.empty and len(sh) >= 12:
        amap = {a.get("id"): a for a in alerts}
        rows = []
        for _, r in sh.iterrows():
            f = (amap.get(r["alert_id"]) or {}).get("features") or {}
            rows.append({**f, "win": r["ret"] > 0})
        fd = pd.DataFrame(rows)
        for col, label in (("vix_ratio", "VIX vs its 20-day average"), ("rsi2", "RSI(2) at signal"),
                           ("dist200", "distance above the 200-day"), ("spy_above200", "SPY above 200-day"),
                           ("sector_up", "sector trend up")):
            if col in fd and fd[col].notna().sum() >= 10 and fd["win"].nunique() == 2:
                mw, ml = fd.loc[fd["win"], col].median(), fd.loc[~fd["win"], col].median()
                if pd.notna(mw) and pd.notna(ml) and abs(mw - ml) > 0.15 * (abs(mw) + abs(ml) + 1e-9):
                    insights.append(f"Winners vs losers differ on {label}: median {mw:.2f} vs {ml:.2f}.")
    # headline stats
    def agg(df):
        if df is None or df.empty:
            return {"n": 0}
        return {"n": int(len(df)), "win_rate": round(float((df["ret"] > 0).mean()), 3),
                "avg_ret": round(float(df["ret"].mean()), 4)}
    worked = []
    if not sh.empty:
        top = sh.sort_values("ret", ascending=False).head(3)
        worked = [f"{r.ticker} ({S.meta(r.strategy, P)['label']}) {100 * r.ret:+.1f}% via {r.reason}"
                  for r in top.itertuples() if r.ret > 0]
        bad = sh.sort_values("ret").head(3)
        didnt = [f"{r.ticker} ({S.meta(r.strategy, P)['label']}) {100 * r.ret:+.1f}% via {r.reason}"
                 for r in bad.itertuples() if r.ret <= 0]
    else:
        didnt = []
    ledger = [v["doc"] for v in state["ledger"].values()]
    # the backtest's fill rate per strategy (how often its limit order fills), to compare your real fills with
    bt_doc = ((state.get("strategy") or {}).get("backtest") or {}).get("doc") or {}
    fill_rates = {}
    for name, rep in (bt_doc.get("strategies") or {}).items():
        fr = ((rep or {}).get("test_oos") or {}).get("fill_rate")
        fr = fr if fr is not None else ((rep or {}).get("all") or {}).get("fill_rate")
        if fr is not None:
            fill_rates[name] = float(fr)
    port = LL.portfolio_stats(alerts, shadow, trades, ledger, now.date().isoformat(), P, fill_rates)
    port["updated_at"] = now.isoformat(timespec="seconds")
    insights += [f"Portfolio: {x}" for x in port.get("lessons", [])]
    day = now.date().isoformat()
    journal = {"id": day, "date": day, "created_at": now.isoformat(timespec="seconds"),
               "shadow": agg(sh), "live": agg(closed), "by_strategy": by, "worked": worked,
               "didnt_work": didnt, "by_grade": journal_grades, "grade_rules": grade_rules,
               "market_condition": cond_today, "by_condition": by_cond, "all_weather_live": aw_live,
               "insights": insights or ["No statistically meaningful drift yet."],
               "changes": changes or ["No parameter changes today."], "params_version": P.get("version")}
    writes = []
    jid = now.strftime("%Y-%m-%d-%H%M")
    journal["id"] = jid
    _write(out, writes, "set", "journal", jid, journal)
    if changes:
        P["version"] = int(time.time())
        P["updated"] = now.isoformat(timespec="seconds")
        doc = {"params": P, "updated_at": now.isoformat(timespec="seconds"),
               "changelog_entry": changes}
        # existing doc: the session reads strategy/params first and fills in if_version
        _write(out, writes, "set", "strategy", "params", doc)
        writes[-1]["if_version"] = "VERSION_OF_strategy/params (get it first; omit only if the doc does not exist)"
    for did, doc in (("stats", stats), ("portfolio", port)):
        _write(out, writes, "set", "learning", did, doc)
        if (state.get("learning") or {}).get(did):
            writes[-1]["if_version"] = f"VERSION_OF_learning/{did} (get it first)"
    (out / "writes.json").write_text(json.dumps([writes], indent=1))
    (out / "journal.json").write_text(json.dumps(journal, indent=1, default=str))
    return {"journal": day, "changes": len(changes), "insights": len(insights), "writes": len(writes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "learn", "topups"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--only", default=None, help="comma list of strategies")
    ap.add_argument("--merge-state", default=None, help="dashboard state export to keep live adjustments from")
    ap.add_argument("--cal-end", default=CAL_END, help="start of the holdout window (cut-offs are chosen before it)")
    a = ap.parse_args()
    if a.cmd == "backtest":
        rep = run_backtest(a.data, a.out, a.split, a.only.split(",") if a.only else None, merge_state=a.merge_state,
                           cal_end=a.cal_end or None)
        print(json.dumps({k: v for k, v in rep.items() if k != "strategies"}, indent=1, default=str))
    elif a.cmd == "topups":
        print(json.dumps(run_topups(a.data, a.state, a.out, a.split), indent=1, default=str))
    else:
        print(json.dumps(run_learning(a.data, a.state, a.out), indent=1, default=str))


if __name__ == "__main__":
    main()
