"""Exit / adjustment checks for open positions."""
from __future__ import annotations

import math

import numpy as np

from . import calendar as cal

DYN_TEXT = {
    "sma5": "the close is above its 5-day average",
    "prev_high": "the close is above the prior day's high",
    "high7": "the close is at a 7-day high",
    "rsi2_70": "RSI(2) is above 70",
}


def _f(x, nd=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def check_trade(tr: dict, ind: dict, mode: str, earnings: dict, premarket, P: dict,
                partial_bar: bool, close_label: str = "1:00 PM PT") -> dict | None:
    tk = tr.get("ticker")
    c = ind["close"]
    if tk not in c.columns:
        return {"trade_id": tr.get("id"), "ticker": tk, "signals": [{
            "code": "NO_DATA", "level": "warn", "action": "CHECK MANUALLY",
            "urgency": "today", "text": f"No price data for {tk} in the feed; check it manually."}]}
    j = c.columns.get_loc(tk)
    T = len(c) - 1
    last_day = c.index[T].date()
    O, H, L, C = (ind[k].values[:, j] for k in ["open", "high", "low", "close"])
    atr = float(ind["atr10"].values[T, j])
    sma5 = float(ind["sma5"].values[T, j])
    sma50 = float(ind["sma50"].values[T, j])
    entry = float(tr.get("entry_price") or tr.get("planned_entry") or 0)
    shares = float(tr.get("shares") or 0)
    stop = float(tr.get("stop") or 0)
    target = float(tr.get("target") or 0)
    max_hold = int(tr.get("max_hold") or 7)
    dyn = tr.get("dyn_exit") or "none"
    strategy = tr.get("strategy") or ""
    try:
        entry_day = cal._d(tr.get("entry_date"))
    except Exception:
        entry_day = last_day
    days_held = cal.trading_days_between(entry_day, last_day) + 1
    days_left = max_hold - days_held
    last = float(C[T])
    if mode == "premarket" and premarket is not None and tk in premarket.index:
        pm = premarket.loc[tk]
        pm_last = float(pm.get("pm_last", np.nan))
    else:
        pm_last = np.nan
    unreal = (last - entry) * shares if entry else 0.0
    sigs = []
    when_close = f"before the {close_label} close today" if mode == "preclose" else "at the next open"

    # fills the brokerage probably already handled (only for bars after the entry day)
    if last_day > entry_day or partial_bar:
        if stop and L[T] <= stop:
            sigs.append({"code": "STOP_HIT", "level": "info", "action": "CONFIRM STOP FILL",
                         "urgency": "now", "text": f"Low ${L[T]:.2f} reached your stop ${stop:.2f}. "
                         "Confirm the fill in your brokerage and log the exit price."})
        if target and H[T] >= target:
            sigs.append({"code": "TARGET_HIT", "level": "good", "action": "CONFIRM TARGET FILL",
                         "urgency": "now", "text": f"High ${H[T]:.2f} reached your target ${target:.2f}. "
                         "Confirm the fill and log the exit price."})
    if mode == "premarket" and math.isfinite(pm_last):
        if stop and pm_last <= stop:
            sigs.append({"code": "GAP_BELOW_STOP", "level": "alert", "action": "SELL AT OPEN",
                         "urgency": "open", "text": f"Pre-market ${pm_last:.2f} is below your stop "
                         f"${stop:.2f}. The stop will likely fill at the open; consider selling at the open."})
        elif target and pm_last >= target:
            sigs.append({"code": "GAP_ABOVE_TARGET", "level": "good", "action": "TARGET FILLS AT OPEN",
                         "urgency": "open", "text": f"Pre-market ${pm_last:.2f} is above your target; "
                         "your limit sell should fill at the open."})

    # earnings risk
    e = (earnings or {}).get(tk) or {}
    nxt = e.get("next")
    if nxt:
        try:
            ed = cal._d(nxt)
            sessions = cal.trading_days_between(last_day, ed)
            if 0 <= sessions <= 1 and ed >= last_day:
                when = e.get("when") or "?"
                sigs.append({"code": "EARNINGS", "level": "alert", "action": "SELL BEFORE EARNINGS",
                             "urgency": "today" if mode == "preclose" else "next",
                             "text": f"Earnings on {ed} ({when}). Swing setups don't hold through "
                             f"earnings: exit {when_close}."})
        except Exception:
            pass

    # strategy exit rule on the (projected) close
    rule_hit = False
    if dyn == "sma5":
        rule_hit = last > sma5
    elif dyn == "prev_high":
        rule_hit = last > H[T - 1]
    elif dyn == "high7":
        rule_hit = last >= float(np.nanmax(C[max(0, T - 6):T + 1]))
    elif dyn == "rsi2_70":
        rule_hit = float(ind["rsi2"].values[T, j]) > 70
    if rule_hit and (last_day > entry_day or partial_bar):
        verb = "is" if mode == "preclose" else "closed"
        sigs.append({"code": "EXIT_RULE", "level": "alert", "action": "SELL AT CLOSE" if mode == "preclose" else "SELL AT OPEN",
                     "urgency": "today" if mode == "preclose" else "next",
                     "text": f"Exit rule met: {DYN_TEXT.get(dyn, dyn)} (price {verb} ${last:.2f}). "
                     f"The bounce has done its job; take the {'gain' if last > entry else 'exit'} {when_close} "
                     "rather than waiting for the target."})
    if days_left <= 0 and not rule_hit:
        sigs.append({"code": "TIME_STOP", "level": "alert",
                     "action": "SELL AT CLOSE" if mode == "preclose" else "SELL AT OPEN",
                     "urgency": "today" if mode == "preclose" else "next",
                     "text": f"Time stop: day {days_held} of {max_hold}. The edge decays after the hold "
                     f"window; exit {when_close}."})
    if strategy in ("TREND_PULLBACK", "BREAKOUT_RETEST", "GAP_HOLD", "LEADER_PULLBACK") and math.isfinite(sma50):
        if last < sma50:
            sigs.append({"code": "TREND_BREAK", "level": "alert",
                         "action": "SELL AT CLOSE" if mode == "preclose" else "SELL AT OPEN",
                         "urgency": "today" if mode == "preclose" else "next",
                         "text": f"Trend break: price ${last:.2f} is below the 50-day average ${sma50:.2f}. "
                         "The setup's premise is gone."})
    # stop management
    if entry and math.isfinite(atr) and atr > 0:
        gain_atr = (last - entry) / atr
        if gain_atr >= 1.5 and stop < entry + 0.5 * atr:
            new = round(entry + 0.5 * atr, 2)
            sigs.append({"code": "RAISE_STOP", "level": "info", "action": f"RAISE STOP TO ${new:.2f}",
                         "urgency": "next", "new_stop": new,
                         "text": f"Up {gain_atr:.1f} ATR. Lock in profit: move the stop to ${new:.2f}."})
        elif gain_atr >= 1.0 and stop < entry:
            new = round(entry, 2)
            sigs.append({"code": "BREAKEVEN", "level": "info", "action": f"RAISE STOP TO ${new:.2f}",
                         "urgency": "next", "new_stop": new,
                         "text": f"Up {gain_atr:.1f} ATR. Move the stop to breakeven (${new:.2f}) so this "
                         "can't turn into a loss."})
    mk = ind["market"]
    try:
        spy_r1 = float(mk["spy"].iloc[-1] / mk["spy"].iloc[-2] - 1)
        vixr = float(mk["vix_ratio"].iloc[-1])
    except Exception:
        spy_r1, vixr = 0.0, 1.0
    if (spy_r1 <= -0.025 or vixr >= 1.35) and unreal < 0:
        sigs.append({"code": "MARKET_STRESS", "level": "warn", "action": "REVIEW",
                     "urgency": "today", "text": f"Market stress (SPY {spy_r1:+.1%} today, VIX {vixr:.2f}x its "
                     "20-day average). Keep the stop in place; do not average down."})
    return {"trade_id": tr.get("id"), "ticker": tk, "strategy": strategy,
            "last": _f(last), "asof": str(last_day), "partial_bar": bool(partial_bar),
            "unrealized": _f(unreal), "unrealized_pct": _f((last / entry - 1) * 100 if entry else 0, 2),
            "days_held": int(days_held), "days_left": int(days_left), "max_hold": max_hold,
            "sma5": _f(sma5), "atr": _f(atr), "pm_last": _f(pm_last), "signals": sigs}
