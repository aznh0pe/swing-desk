"""Swing setups. Each builder turns indicator frames + params into an order plan.

A plan is evaluated on the signal day's close (day t) and trades on later sessions:
  entry_kind 'limit_next' : buy-limit order for the next session(s) at `limit`
  entry_kind 'close'      : buy near the close of day t (pre-close scan alert)
Stops / targets are fixed prices so they can sit in the brokerage as a bracket.
`dyn_exit` is the early-exit rule the daily scans watch (sell at the close).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

STRATEGY_META = {
    "MR_STRETCH": {
        "label": "RSI(2) Stretch Dip",
        "family": "Mean reversion",
        "about": "Uptrend (above rising 200-day) stock or ETF that just had a sharp 2-3 day "
                 "sell-off (RSI(2) deeply oversold, below its 5-day average). Buy a further "
                 "stretch below the close with a limit order; sell into the snap-back.",
    },
    "LEADER_PULLBACK": {
        "label": "Leader 7-Day Pullback",
        "family": "Momentum pullback",
        "about": "Top relative-strength stock (above 50 and 200-day) closing at a 7-day low. "
                 "Leaders that dip tend to be bought quickly; exit on the bounce.",
    },
    "ETF_IBS": {
        "label": "ETF Close-at-Low Snapback",
        "family": "Mean reversion / auction (poor low)",
        "about": "Index or sector ETF in an uptrend closing in the bottom of its daily range "
                 "(IBS low) on a down day. Sellers exhausted into the close; next day tends to "
                 "trade back above that day's high.",
    },
    "TREND_PULLBACK": {
        "label": "Trend Pullback to 20-EMA",
        "family": "Price action / trend continuation",
        "about": "Strong uptrend stock pulling back on lighter volume to its rising 20-day EMA "
                 "after a fresh high. Buy the test of support; target the recent swing high.",
    },
    "BREAKOUT_RETEST": {
        "label": "Tight-Base Breakout Retest",
        "family": "Breakout",
        "about": "Stock breaks out of a tight base to a 20-day high on heavy volume near its "
                 "52-week high. Buy the retest of the breakout level instead of chasing.",
    },
    "GAP_HOLD": {
        "label": "Power-Gap Hold",
        "family": "Event drift (post-earnings style)",
        "about": "A recent gap-up on 2.5x+ volume (news/earnings) is holding above the gap-day "
                 "low while it digests. Buy the dip into the gap; ride the drift.",
    },
    "FAIR_VALUE": {
        "label": "Below Fair Value (VWAP-20)",
        "family": "Fair value / auction market",
        "about": "Uptrend name trading far below its 20-day volume-weighted fair value "
                 "(VWAP minus z standard deviations). Auctions rotate back toward value; "
                 "target the 20-day VWAP.",
    },
}

DEFAULT_PARAMS = {
    "version": 1,
    "updated": None,
    "risk": {"risk_pct": 2.0, "risk_pct_top": 2.0, "max_position_pct": 25.0,
             "max_open": 5, "min_position_usd": 150.0, "slippage_pct": 0.05},
    "filters": {"min_price": 10.0, "min_dv_stock": 30e6, "min_dv_etf": 20e6,
                "earnings_buffer_days": 1, "allow_leveraged": True},
    # position management the scans apply live, replayed in every backtest so the two never drift apart:
    # the stop moves to entry + 1 ATR once a close is 2 ATR above the fill; a limit order only counts as
    # filled when the session's low trades through it (a touch is not a fill)
    "manage": {"lock_atr": 2.0, "lock_to": 1.0, "be_atr": 0, "fill": "through"},
    "grading": {"aplus_score": 80, "aplus_pwin": 0.70, "min_week": 3},
    "strategies": {
        "MR_STRETCH": {"enabled": True, "universe": "all", "rsi2_max": 10, "k_atr": 0.5,
                       "target": {"kind": "mean4", "min_atr": 0.3}, "stop_atr": 3.0,
                       "max_hold": 7, "dyn_exit": "sma5", "valid_days": 1},
        "LEADER_PULLBACK": {"enabled": True, "universe": "stock", "rs_min": 0.75,
                            "rsi2_max": 20, "k_atr": 0.3,
                            "target": {"kind": "max7c", "min_atr": 0.3}, "stop_atr": 3.0,
                            "max_hold": 8, "dyn_exit": "sma5", "valid_days": 1},
        "ETF_IBS": {"enabled": True, "universe": "etf", "ibs_max": 0.15, "rsi2_max": 35,
                    "entry_kind": "limit_next", "k_atr": 0.2,
                    "target": {"kind": "high_t", "min_atr": 0.25}, "stop_atr": 2.5,
                    "max_hold": 5, "dyn_exit": "prev_high", "valid_days": 1},
        "TREND_PULLBACK": {"enabled": True, "universe": "stock", "rs_min": 0.7, "k_atr": 0.1,
                           "target": {"kind": "hh20", "min_atr": 0.75}, "stop_atr": 2.0,
                           "max_hold": 10, "dyn_exit": "none", "valid_days": 1},
        "BREAKOUT_RETEST": {"enabled": True, "universe": "stock", "base_max": 0.12,
                            "vol_mult": 1.5, "target": {"kind": "atr", "mult": 2.0},
                            "stop_atr": 1.5, "max_hold": 10, "dyn_exit": "none",
                            "valid_days": 2},
        "GAP_HOLD": {"enabled": True, "universe": "stock", "gap_min": 0.03, "vol_mult": 2.5,
                     "rsi2_max": 30, "k_atr": 0.25, "target": {"kind": "atr", "mult": 1.5},
                     "stop_kind": "gap_low", "max_hold": 10, "dyn_exit": "none",
                     "valid_days": 1},
        "FAIR_VALUE": {"enabled": True, "universe": "all", "z": 2.0, "rsi2_max": 30,
                       "k_atr": 0.25, "target": {"kind": "vwap20", "min_atr": 0.3},
                       "stop_atr": 3.0, "max_hold": 7, "dyn_exit": "sma5", "valid_days": 1},
    },
    "models": {},
    "live_adjust": {},
}


def params_with_defaults(p: dict | None) -> dict:
    out = copy.deepcopy(DEFAULT_PARAMS)
    if not p:
        return out

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    merge(out, copy.deepcopy(p))
    return out


def base_masks(ind: dict, P: dict) -> dict:
    f = P["filters"]
    c = ind["close"]
    is_stock, is_etf, is_lev = ind["is_stock"], ind["is_etf"], ind["is_lev"]
    types = ind["types"]
    tradable = pd.DataFrame(np.broadcast_to((types != "index").values, c.shape),
                            index=c.index, columns=c.columns)
    min_dv = np.where(is_stock.values, f["min_dv_stock"], f["min_dv_etf"])
    liquid = (c >= f["min_price"]) & (ind["dv20"] >= min_dv) & ind["sma200"].notna() \
        & ind["atr10"].notna() & tradable
    if not f.get("allow_leveraged", True):
        liquid &= ~is_lev
    return {"liquid": liquid, "is_stock": is_stock, "is_etf": is_etf & tradable,
            "is_lev": is_lev}


def _universe_mask(b, which):
    if which == "stock":
        return b["is_stock"]
    if which == "etf":
        return b["is_etf"]
    return b["is_stock"] | b["is_etf"]


def _target(ind, sp):
    t = sp["target"]
    kind = t["kind"]
    if kind == "atr":
        return {"kind": "atr", "mult": float(t.get("mult", 1.0))}
    frame = {"mean4": ind["mean4"], "max7c": ind["max7c"], "high_t": ind["high"],
             "hh20": ind["hh20"], "vwap20": ind["vwap20"], "sma5": ind["sma5"]}[kind]
    return {"kind": "level", "frame": frame, "min_atr": float(t.get("min_atr", 0.25)),
            "cap_atr": float(t.get("cap_atr", 4.0))}


# ----------------------------------------------------------------------------- market conditions
# Four market conditions from SPY's trend and short-term stress. Every signal, alert and trade is tagged
# with the condition on its signal day; each strategy and each grade is learned per condition, and a
# strategy can be restricted to the conditions where it has proven itself.
CONDITIONS = ("bull_calm", "bull_stress", "bear_calm", "bear_stress")
CONDITION_LABEL = {"bull_calm": "Calm bull market", "bull_stress": "Pullback in a bull market",
                   "bear_calm": "Quiet bear market", "bear_stress": "Bear-market selloff"}
CONDITION_PLURAL = {"bull_calm": "calm bull markets", "bull_stress": "pullbacks in bull markets",
                    "bear_calm": "quiet bear markets", "bear_stress": "bear-market selloffs"}
STRESS_VIX, STRESS_SPY5 = 1.10, -0.014
CONDITION_RULE = ("Bull = SPY above its 200-day average. Stressed = VIX at least 1.10x its 20-day average, or SPY down "
                  "1.4% or more over 5 days. Calm = neither.")


def _f(x, default):
    try:
        v = float(x)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def condition_of(spy_above200, vix_ratio, spy_ret5) -> str | None:
    """The market condition for one day's SPY trend / VIX ratio / SPY 5-day return (None when unknown)."""
    up = _f(spy_above200, None)
    if up is None:
        return None
    stressed = _f(vix_ratio, 1.0) >= STRESS_VIX or _f(spy_ret5, 0.0) <= STRESS_SPY5
    return ("bull_" if up >= 0.5 else "bear_") + ("stress" if stressed else "calm")


def condition_of_features(f: dict | None) -> str | None:
    f = f or {}
    return condition_of(f.get("spy_above200"), f.get("vix_ratio"), f.get("spy_ret5"))


def condition_series(df: pd.DataFrame) -> pd.Series:
    """Per-row condition from spy_above200 / vix_ratio / spy_ret5 columns (a market frame or a trade table)."""
    up = df["spy_above200"].astype(float)
    stressed = (df["vix_ratio"].astype(float).fillna(1.0) >= STRESS_VIX) | (df["spy_ret5"].astype(float).fillna(0.0) <= STRESS_SPY5)
    out = np.where(up >= 0.5, np.where(stressed, "bull_stress", "bull_calm"), np.where(stressed, "bear_stress", "bear_calm"))
    return pd.Series(np.where(up.isna(), None, out), index=df.index)


def condition_gate(ind: dict, conds, like: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame (dates x tickers, shaped like `like`) that is True on days in one of `conds`."""
    day = condition_series(ind["market"]).reindex(like.index).isin(list(conds)).values
    return pd.DataFrame(np.broadcast_to(day[:, None], like.shape), index=like.index, columns=like.columns)


# All-weather strategies: some setups hold up in every market condition. They trade everywhere, are never
# paused in a condition on old data alone, need more live evidence before a live pause, and are never
# narrowed into specialists.
ALL_WEATHER = {"train_n": 50, "oos_n": 30, "train_win": 0.60, "train_pf": 1.15, "oos_win": 0.62, "oos_pf": 1.05,
               "min_conditions": 3, "min_share": 0.85}
ALL_WEATHER_RULE = ("All-weather = holds up in every market condition it has enough trades in (2008-2021: 60%+ wins and "
                    "profit factor 1.15+ on 50+ trades; 2022-now: 62%+ wins and profit factor 1.05+ on 30+ trades), "
                    "tested in at least 3 of the 4 conditions.")


def all_weather(by_cond: dict | None, share: dict | None = None, q: dict | None = None) -> dict:
    """Does a strategy hold up in every market condition? `by_cond` is {condition: {"train": stats, "oos": stats}}
    (stats with trades, win_rate and profit_factor; a missing profit factor is not held against it). A condition
    counts as tested when either period has enough trades there; the strategy is all-weather when it clears the
    bars in every tested condition and period, the tested conditions number at least 3 and (when `share` of
    trading days is given) cover at least 85% of days. Returns the verdict with the tested and weak conditions
    and the 2022-now win-rate range across conditions."""
    q = {**ALL_WEATHER, **(q or {})}
    tested, weak, wins, why = [], [], [], {}
    for c in CONDITIONS:
        d = (by_cond or {}).get(c) or {}
        tr, oo = d.get("train") or {}, d.get("oos") or {}
        t_ok, o_ok = (tr.get("trades") or 0) >= q["train_n"], (oo.get("trades") or 0) >= q["oos_n"]
        if not (t_ok or o_ok):
            continue
        tested.append(c)
        bad = False
        for ok_, st, wk, pk, per in ((t_ok, tr, "train_win", "train_pf", "2008-2021"), (o_ok, oo, "oos_win", "oos_pf", "2022-now")):
            if ok_ and ((st.get("win_rate") or 0) < q[wk] or (st.get("profit_factor") is not None and st["profit_factor"] < q[pk])):
                bad = True
                why.setdefault(c, []).append(f"{per} {100 * (st.get('win_rate') or 0):.0f}% wins"
                                             + (f", profit factor {st['profit_factor']}" if st.get("profit_factor") is not None else ""))
        if o_ok and oo.get("win_rate") is not None:
            wins.append(float(oo["win_rate"]))
        if bad:
            weak.append(c)
    cover = None if share is None else round(sum(float(share.get(c) or 0) for c in tested), 3)
    aw = len(tested) >= q["min_conditions"] and not weak and (cover is None or cover >= q["min_share"])
    return {"all_weather": bool(aw), "tested": tested, "weak": weak, "cover": cover,
            "weak_detail": {c: "; ".join(v) for c, v in why.items()},
            "oos_win_range": [round(min(wins), 4), round(max(wins), 4)] if wins else None}


def all_weather_sentence(aw: dict | None, label: str = "It") -> str:
    """One plain sentence on whether a strategy is all-weather."""
    if not aw or not aw.get("tested"):
        return ""
    rng = aw.get("oos_win_range")
    span = f" (won {100 * rng[0]:.0f}-{100 * rng[1]:.0f}% across them on 2022-now data)" if rng else ""
    if aw.get("all_weather"):
        return f"{label} is all-weather: it held up in every market condition{span}."
    if aw.get("weak"):
        det = aw.get("weak_detail") or {}
        return (f"{label} is condition-dependent: weaker in "
                + " and ".join(CONDITION_PLURAL[c] + (f" ({det[c]})" if det.get(c) else "") for c in aw["weak"]) + f"{span}.")
    return ""


def meta(name: str, P: dict | None = None) -> dict:
    """Label / family / description for a built-in or a promoted research strategy."""
    if name in STRATEGY_META:
        return STRATEGY_META[name]
    cu = ((P or {}).get("custom") or {}).get(name) or {}
    spec = cu.get("spec") or {}
    return {"label": cu.get("label") or spec.get("label") or name,
            "family": cu.get("family") or spec.get("family") or "Research",
            "about": cu.get("about") or spec.get("about") or "", "custom": True,
            "source": spec.get("source")}


# The bar every strategy must clear to trade live (re-checked at each weekly recalibration).
ENABLE_RULE = {"train_win": 0.58, "train_pf": 1.30, "train_avg": 0.002, "oos_pf": 1.05, "oos_decay": 0.08}


def enable_checks(st_train: dict, st_test: dict) -> tuple[bool, list, str]:
    """(enabled, checks, plain-English reason) for the live-trading bar."""
    R = ENABLE_RULE
    tw, tpf, tav = st_train.get("win_rate"), st_train.get("profit_factor"), st_train.get("avg_ret")
    ow, opf, on = st_test.get("win_rate"), st_test.get("profit_factor"), st_test.get("trades", 0)
    dec = (tw - ow) if (tw is not None and ow is not None) else None
    checks = [
        {"key": "train_win", "label": "Win rate 2008-2021", "value": tw, "need": f">= {R['train_win']:.0%}",
         "ok": tw is not None and tw >= R["train_win"]},
        {"key": "train_pf", "label": "Profit factor 2008-2021", "value": tpf, "need": f">= {R['train_pf']:.2f}",
         "ok": tpf is not None and tpf >= R["train_pf"]},
        {"key": "train_avg", "label": "Average trade 2008-2021", "value": tav, "need": f"> +{100 * R['train_avg']:.2f}%",
         "ok": tav is not None and tav > R["train_avg"]},
        {"key": "oos_pf", "label": "Profit factor 2022-now (unseen data)", "value": opf, "need": f">= {R['oos_pf']:.2f}",
         "ok": bool(on) and opf is not None and opf >= R["oos_pf"]},
        {"key": "oos_decay", "label": "Win-rate drop on unseen data", "value": dec,
         "need": f"<= {100 * R['oos_decay']:.0f} pts", "ok": dec is not None and dec <= R["oos_decay"] + 1e-9},
    ]
    ok = all(c["ok"] for c in checks)
    if ok:
        reason = (f"Active: passed every live-trading check ({100 * tw:.0f}% wins 2008-2021, "
                  f"{100 * ow:.0f}% on unseen 2022-now data, profit factor {opf:.2f}).")
    else:
        bits = []
        for c in checks:
            if c["ok"]:
                continue
            v = c["value"]
            if c["key"] == "train_win":
                bits.append(f"only {100 * v:.1f}% of 2008-2021 trades won (needs {R['train_win']:.0%})" if v is not None else "no trades")
            elif c["key"] == "train_pf":
                bits.append(f"profit factor {v:.2f} on 2008-2021 data is below {R['train_pf']:.2f}: winners too small for the losers"
                            if v is not None else "no profit factor")
            elif c["key"] == "train_avg":
                bits.append(f"average trade {100 * v:+.2f}% is too thin after costs" if v is not None else "no trades")
            elif c["key"] == "oos_pf":
                bits.append(f"on unseen 2022-now data it roughly broke even (profit factor {opf:.2f}, needs {R['oos_pf']:.2f})"
                            + (f" even though {100 * ow:.0f}% of trades won" if ow else "") if opf is not None else
                            "not enough unseen-data trades")
            elif c["key"] == "oos_decay":
                bits.append(f"win rate fell {100 * v:.1f} pts on unseen data (allowed {100 * R['oos_decay']:.0f}): the edge faded"
                            if v is not None else "no unseen-data trades")
        reason = "Off: " + "; ".join(bits) + ". Re-tested at every weekly recalibration; it switches back on by itself if it passes."
    return ok, checks, reason


FIXED_KEYS = {"train_win", "train_pf", "train_avg"}  # measured on 2008-2021: new data never changes them


def path_to_pass(st_train: dict, st_test: dict, oos_weeks: float, extra_checks: list | None = None,
                 need: dict | None = None, bumps=(0.0, 0.05)) -> dict:
    """What it would take for a strategy that is not live to pass as new weeks of data arrive.

    New data only ever lands in the unseen 2022-now window, so checks measured on 2008-2021
    can't change. For the unseen-data checks, assume future trades win at the strategy's
    2008-2021 rate (and 5 points better) with its usual win and loss sizes, and count the
    trades (and weeks at its usual pace) until every check passes.
    """
    ok, checks, _ = enable_checks(st_train, st_test)
    checks = checks + list(extra_checks or [])
    failing = [c for c in checks if not c["ok"]]
    if not failing:
        return {"status": "passing", "text": "Passes every check."}
    fixed = [c for c in failing if c["key"] in FIXED_KEYS]
    if fixed:
        short = {"train_win": "win rate", "train_pf": "profit factor", "train_avg": "average trade"}
        names = [short[c["key"]] for c in fixed]
        names = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return {"status": "fixed", "blocking": [c["key"] for c in fixed],
                "text": (f"New data can't switch it on: it fails the 2008-2021 {names} check{'s' if len(fixed) > 1 else ''}, and that period never "
                         "changes. Only a different rule can fix it; rule variants go through the research pipeline.")}
    only_overlap = all(c["key"] == "overlap" for c in failing)
    if only_overlap:
        return {"status": "overlap", "blocking": ["overlap"],
                "text": ("It passes on its own, but most of its trades are ones the live strategies already take. That "
                         "changes only if the live strategies change; it is re-checked every Saturday.")}
    need = dict(need or {})
    N = int(st_test.get("trades") or 0)
    p = float(st_test.get("win_rate") or 0)
    aw, al = float(st_test.get("avg_win") or 0), abs(float(st_test.get("avg_loss") or 0))
    tw = float(st_train.get("win_rate") or 0)
    pf_need = max(ENABLE_RULE["oos_pf"], float(need.get("oos_pf", 0)))
    win_need = max(tw - ENABLE_RULE["oos_decay"], float(need.get("oos_win", 0)))
    n_need = int(need.get("oos_trades", 0))
    per_week = N / max(1.0, oos_weeks)
    gp, gl = p * N * aw, (1 - p) * N * al
    scen = []
    for bump in bumps:
        wf = min(0.95, tw + bump)
        reqs = []
        coef = wf * aw - pf_need * (1 - wf) * al
        deficit = pf_need * gl - gp
        if deficit > 0:
            reqs.append(deficit / coef if coef > 0 else None)
        if p < win_need:
            reqs.append(N * (win_need - p) / (wf - win_need) if wf > win_need else None)
        if N < n_need:
            reqs.append(n_need - N)
        if any(r is None for r in reqs):
            scen.append({"future_win_rate": round(wf, 3), "trades": None, "weeks": None})
            continue
        n = int(np.ceil(max(reqs or [0])))
        wk = n / per_week if per_week > 0 else None
        scen.append({"future_win_rate": round(wf, 3), "trades": n, "weeks": round(wk, 1) if wk is not None else None})
    blocking = [c["key"] for c in failing if c["key"] != "overlap"]
    coef_now = p * aw - pf_need * (1 - p) * al
    stuck = (pf_need * gl - gp > 0 and coef_now <= 0) or (p < win_need)
    best = scen[-1]
    realistic = [sc for sc in scen if sc["weeks"] is not None and sc["weeks"] <= 104]
    if not realistic:
        how = ("it would never reach the bar" if best["trades"] is None else
               f"it would need about {best['trades']:,} more trades, roughly {best['weeks'] / 52:.0f} years")
        text = (f"New data alone is very unlikely to switch it on: even if its trades won "
                f"{100 * best['future_win_rate']:.0f}% from here, {how}. A better rule is the faster path, so the "
                "research pipeline tests rule variants.")
        status = "data_slow"
    else:
        parts = []
        for i, sc in enumerate(scen):
            w = f"{100 * sc['future_win_rate']:.0f}%" + (" (its 2008-2021 rate)" if i == 0 else "")
            if sc["trades"] is None:
                parts.append(f"if they win {w}, never")
            elif sc["weeks"] is None or sc["weeks"] > 104:
                parts.append(f"if they win {w}, more than 2 years")
            elif sc["trades"] == 0:
                parts.append(f"if they win {w}, right away")
            else:
                wk = max(1, round(sc["weeks"]))
                parts.append(f"if they win {w}, about {sc['trades']:,} more trades, roughly {wk} week{'s' if wk != 1 else ''}")
        text = (f"Can switch on with new data if it trades well. It makes about {per_week:.0f} trades a week across the "
                "universe: " + "; ".join(parts) + "." + (f" At its recent {100 * p:.0f}% it stays off." if stuck else ""))
        status = "data"
    if any(c["key"] == "overlap" for c in failing):
        text += " It also overlaps the live strategies too much, which only changes if they change."
    return {"status": status, "blocking": blocking, "per_week": round(per_week, 1), "scenarios": scen, "text": text}


def build(name: str, ind: dict, P: dict, b: dict | None = None) -> dict | None:
    sp = P["strategies"].get(name)
    if not sp or not sp.get("enabled", True):
        return None
    b = b or base_masks(ind, P)
    cu = (P.get("custom") or {}).get(name)
    if cu:
        from .rules import build_plan
        try:
            spec = {**cu["spec"], **({"market_conditions": sp["market_conditions"]} if sp.get("market_conditions") else {})}
            return build_plan(spec, ind, P, b, cu.get("params"), name=name)
        except Exception:
            return None
    c, o, h, l = ind["close"], ind["open"], ind["high"], ind["low"]
    atr = ind["atr10"]
    uni = _universe_mask(b, sp.get("universe", "all")) & b["liquid"]
    plan = {"name": name, "entry_kind": sp.get("entry_kind", "limit_next"),
            "valid_days": int(sp.get("valid_days", 1)), "max_hold": int(sp["max_hold"]),
            "dyn_exit": sp.get("dyn_exit", "none"),
            "stop": {"kind": "atr", "mult": float(sp.get("stop_atr", 3.0))},
            "target": _target(ind, sp)}
    k = float(sp.get("k_atr", 0.0))
    limit = c - k * atr

    if name == "MR_STRETCH":
        m = uni & (c > ind["sma200"]) & (ind["slope200"] > 0) & (ind["rsi2"] < sp["rsi2_max"])
        if sp.get("require_below_sma5", True):
            m &= c < ind["sma5"]
    elif name == "LEADER_PULLBACK":
        m = uni & (c > ind["sma50"]) & (ind["sma50"] > ind["sma200"]) \
            & (ind["rs_pct"] >= sp["rs_min"]) & (c <= ind["min7c"]) & (ind["rsi2"] < sp["rsi2_max"])
    elif name == "ETF_IBS":
        m = uni & (c > ind["sma200"]) & (ind["ibs"] < sp["ibs_max"]) & (c < c.shift(1)) \
            & (ind["rsi2"] < sp.get("rsi2_max", 100))
        if plan["entry_kind"] == "close":
            limit = c.copy()
    elif name == "TREND_PULLBACK":
        recent_high = ind["hh10"] >= ind["hh20"]  # 20-day high set within last 10 days
        touch = (l <= ind["ema20"] + 0.2 * atr) & (c >= ind["ema20"] - 0.3 * atr)
        m = uni & (ind["sma50"] > ind["sma200"]) & (c > ind["sma50"]) & (ind["slope50"] > 0) \
            & recent_high & touch & (ind["rsi14"] >= 38) & (ind["rsi14"] <= 58) & (ind["v5"] < ind["v20"]) \
            & (ind["rs_pct"] >= sp["rs_min"])
        limit = np.minimum(c, ind["ema20"] + 0.1 * atr) - k * atr
    elif name == "BREAKOUT_RETEST":
        prior_hh = ind["hh20"].shift(1)
        prior_ll = ind["ll20"].shift(1)
        tight = (prior_hh - prior_ll) / c < sp["base_max"]
        m = uni & (c > ind["sma50"]) & (ind["sma50"] > ind["sma200"]) & (c > prior_hh) \
            & (ind["vol_ratio"] > sp["vol_mult"]) & tight & (c >= 0.95 * ind["hh252"])
        limit = prior_hh + 0.1 * atr
        plan["stop"] = {"kind": "atr", "mult": float(sp.get("stop_atr", 1.5))}
    elif name == "GAP_HOLD":
        ev = (ind["gap_up"] >= sp["gap_min"]) & (ind["vol_ratio"] >= sp["vol_mult"]) \
            & (c >= (h + l) / 2) & b["liquid"]
        ev_low = l.where(ev)
        ev_close = c.where(ev)
        idx = np.arange(len(c))[:, None]
        last_ev = pd.DataFrame(np.where(ev.values, idx, np.nan), index=c.index,
                               columns=c.columns).ffill(limit=6)
        days_since = pd.DataFrame(idx - last_ev.values, index=c.index, columns=c.columns)
        g_low = ev_low.ffill(limit=6)
        g_close = ev_close.ffill(limit=6)
        m = uni & (days_since >= 2) & (days_since <= 6) & (c > g_low) & (c < g_close) \
            & (ind["rsi2"] < sp["rsi2_max"]) & (c > ind["sma200"])
        plan["stop"] = {"kind": "level", "frame": g_low - 0.5 * atr}
    elif name == "FAIR_VALUE":
        m = uni & (c > ind["sma200"]) & (c < ind["vwap20"] - sp["z"] * ind["vstd20"]) \
            & (ind["rsi2"] < sp["rsi2_max"])
    else:
        return None

    # optional market-regime gates (chosen by the backtest grid)
    reg = sp.get("regime")
    if reg:
        mk = ind["market"]
        gate = pd.Series(True, index=c.index)
        if "spy200" in reg:
            gate &= mk["spy_above200"] >= 0.5
        if "breadth" in reg:
            gate &= mk["breadth50"].fillna(0.5) >= 0.35
        if "vixcalm" in reg:
            gate &= mk["vix_ratio"].fillna(1.0) <= 1.35
        m = m & pd.DataFrame(np.broadcast_to(gate.values[:, None], c.shape), index=c.index, columns=c.columns)
    if sp.get("market_conditions"):  # a specialist trades only in the market conditions where it has proven itself
        m = m & condition_gate(ind, sp["market_conditions"], c)
    plan["mask"] = m.fillna(False).astype(bool)
    plan["limit"] = limit
    return plan


def all_plans(ind: dict, P: dict) -> dict:
    b = base_masks(ind, P)
    out = {}
    for name in P["strategies"]:
        pl = build(name, ind, P, b)
        if pl is not None:
            out[name] = pl
    return out


FEATURES = ["rsi2", "depth_atr", "dist200", "slope200", "atr_pct", "rs_pct", "ibs",
            "streak", "adx14", "vol_ratio", "dist50", "ret5", "spy_above200", "spy_dist200",
            "spy_ret5", "breadth50", "vix", "vix_ratio", "sector_up", "is_etf"]


def features_at(ind: dict, t_idx: np.ndarray, j_idx: np.ndarray) -> pd.DataFrame:
    def g(name):
        return ind[name].values[t_idx, j_idx]

    mk = ind["market"]
    out = pd.DataFrame({
        "rsi2": g("rsi2"),
        "depth_atr": (ind["sma5"].values[t_idx, j_idx] - g("close")) / g("atr10"),
        "dist200": g("dist200"), "slope200": g("slope200"), "atr_pct": g("atr_pct"),
        "rs_pct": g("rs_pct"), "ibs": g("ibs"), "streak": g("streak"), "adx14": g("adx14"),
        "vol_ratio": g("vol_ratio"), "dist50": g("dist50"), "ret5": g("ret5"),
        "spy_above200": mk["spy_above200"].values[t_idx],
        "spy_dist200": mk["spy_dist200"].values[t_idx],
        "spy_ret5": mk["spy_ret5"].values[t_idx],
        "breadth50": mk["breadth50"].values[t_idx],
        "vix": mk["vix"].values[t_idx], "vix_ratio": mk["vix_ratio"].values[t_idx],
        "sector_up": g("sector_up"),
        "is_etf": ind["is_etf"].values[t_idx, j_idx].astype(float),
    })
    out["rs_pct"] = out["rs_pct"].fillna(0.5)
    return out
