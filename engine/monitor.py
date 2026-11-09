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


def _day(d) -> str:
    try:
        return cal._d(d).strftime("%a %b %d").replace(" 0", " ")
    except Exception:
        return str(d)


def hold_last_day(entry_day, max_hold: int):
    """The last session of the hold window: the time stop sells at its close."""
    d = cal._d(entry_day)
    if not cal.is_trading_day(d):
        d = cal.next_trading_day(d)
    return cal.next_trading_day(d, max(0, int(max_hold) - 1)) if int(max_hold) > 1 else d


def _st(code, label, tone, when, detail) -> dict:
    return {"code": code, "label": label, "tone": tone, "when": when, "detail": detail}


def position_status(sigs: list, mode: str, *, stop, target, dyn, day_now: int, max_hold: int, last_day_str: str,
                    close_label: str = "1:00 PM PT", trigger: str = "") -> dict:
    """One plain instruction per open position, from its signals: what to do now and why.
    tone: sell (act by selling), good (a target filled or is filling), act (something to do but keep the trade), hold.
    when: when the instruction applies (now, open, close, next or none), so the dashboard can say when it has passed."""
    by = {s_["code"]: s_ for s_ in sigs}
    pre = mode == "preclose"
    sell = f"SELL AT TODAY'S CLOSE (before {close_label})" if pre else "SELL AT THE OPEN"
    sell_when = "close" if pre else "open"
    rule = DYN_TEXT.get(dyn)
    plan = ((f"It sells at the close once {rule}{f' ({trigger})' if trigger else ''}, or" if rule else "It sells")
            + f" at the close of {last_day_str} (time stop) at the latest; your bracket handles the target ${target:.2f} and stop ${stop:.2f}.")
    st = _st

    if "NO_DATA" in by:
        return st("CHECK", "CHECK IT MANUALLY", "act", "now", by["NO_DATA"]["text"])
    if "STOP_HIT" in by and "TARGET_HIT" in by:
        return st("CONFIRM", "STOP AND TARGET BOTH REACHED: CHECK WHICH FILLED", "act", "now",
                  "The day's range touched both your stop and your target. Check your brokerage for which order filled, "
                  "then record the sale on its dashboard card.")
    if "STOP_HIT" in by:
        return st("STOP_HIT", "STOP REACHED: RECORD THE SALE", "sell", "now",
                  by["STOP_HIT"]["text"].replace("log the exit price", "record the sale on its dashboard card"))
    if "TARGET_HIT" in by:
        return st("TARGET_HIT", "TARGET REACHED: RECORD THE SALE", "good", "now",
                  by["TARGET_HIT"]["text"].replace("log the exit price", "record the sale on its dashboard card"))
    if "GAP_BELOW_STOP" in by:
        return st("GAP_BELOW_STOP", "SELL AT THE OPEN", "sell", "open", by["GAP_BELOW_STOP"]["text"])
    if "GAP_ABOVE_TARGET" in by:
        return st("GAP_ABOVE_TARGET", "TARGET FILLS AT THE OPEN", "good", "open",
                  by["GAP_ABOVE_TARGET"]["text"] + " Record the sale once it fills.")
    if "EARNINGS" in by:
        return st("EARNINGS", f"SELL BEFORE EARNINGS: {sell}", "sell", sell_when, by["EARNINGS"]["text"])
    if "EXIT_RULE" in by:
        return st("EXIT_RULE", f"{sell}: EXIT RULE MET", "sell", sell_when, by["EXIT_RULE"]["text"])
    if "TREND_BREAK" in by:
        return st("TREND_BREAK", f"{sell}: TREND BROKE", "sell", sell_when, by["TREND_BREAK"]["text"])
    if "TIME_STOP" in by:
        return st("TIME_STOP", f"TIME STOP: {sell}", "sell", sell_when, by["TIME_STOP"]["text"])
    if "RAISE_STOP" in by:
        new = by["RAISE_STOP"].get("new_stop")
        return st("RAISE_STOP", f"KEEP HOLDING · RAISE THE STOP TO ${new:.2f}" if new else "KEEP HOLDING · RAISE THE STOP", "act", "next",
                  by["RAISE_STOP"]["text"] + " Do it before the next open, then press Stop moved on the dashboard. " + plan)
    if "TIME_STOP_NEXT" in by:
        return st("TIME_STOP_NEXT", "KEEP HOLDING · LAST DAY TOMORROW", "act", "next",
                  by["TIME_STOP_NEXT"]["text"])
    if "MARKET_STRESS" in by:
        return st("HOLD", "KEEP HOLDING", "hold", "none", by["MARKET_STRESS"]["text"] + " Otherwise nothing to do. " + plan)
    return st("HOLD", "KEEP HOLDING", "hold", "none", "Nothing to do today. " + plan)


def check_trade(tr: dict, ind: dict, mode: str, earnings: dict, premarket, P: dict,
                partial_bar: bool, close_label: str = "1:00 PM PT") -> dict | None:
    tk = tr.get("ticker")
    c = ind["close"]
    if tk not in c.columns:
        sig = {"code": "NO_DATA", "level": "warn", "action": "CHECK MANUALLY",
               "urgency": "today", "text": f"No price data for {tk} in the feed; check it in your brokerage."}
        return {"trade_id": tr.get("id"), "ticker": tk, "signals": [sig],
                "status": _st("CHECK", "CHECK IT MANUALLY", "act", "now", sig["text"])}
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
    elif days_left == 1 and mode == "postclose" and not rule_hit:
        sigs.append({"code": "TIME_STOP_NEXT", "level": "info", "action": "LAST DAY TOMORROW",
                     "urgency": "next", "text": f"Day {days_held} of {max_hold}: tomorrow is the last day of the hold "
                     "window. Unless the target or stop fills first, the pre-close scan will tell you to sell at the close."})
    if strategy in ("TREND_PULLBACK", "BREAKOUT_RETEST", "GAP_HOLD") and math.isfinite(sma50):
        if last < sma50:
            sigs.append({"code": "TREND_BREAK", "level": "alert",
                         "action": "SELL AT CLOSE" if mode == "preclose" else "SELL AT OPEN",
                         "urgency": "today" if mode == "preclose" else "next",
                         "text": f"Trend break: price ${last:.2f} is below the 50-day average ${sma50:.2f}. "
                         "The setup's premise is gone."})
    # stop management: measured in the backtest (Nov 2026). A breakeven stop after +1 ATR cut the A+ win
    # rate by about a point (bounces oscillate before they finish), so it is not used. Locking in +1 ATR
    # once a position is up 2 ATR changed nothing measurable, so that is the one adjustment kept.
    mg = P.get("manage") or {}
    lock_atr, lock_to = float(mg.get("lock_atr") or 2.0), float(mg.get("lock_to") if mg.get("lock_to") is not None else 1.0)
    if entry and math.isfinite(atr) and atr > 0 and lock_atr > 0:
        gain_atr = (last - entry) / atr
        if gain_atr >= lock_atr and stop < entry + lock_to * atr:
            new = round(entry + lock_to * atr, 2)
            sigs.append({"code": "RAISE_STOP", "level": "info", "action": f"RAISE STOP TO ${new:.2f}",
                         "urgency": "next", "new_stop": new,
                         "text": f"Up {gain_atr:.1f} ATR. Lock in part of the gain: move the stop to ${new:.2f} "
                         f"(entry + {lock_to:g} ATR). Leave the target in place."})
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
    # before the open, today's session is the next hold day (the dashboard counts it the same way)
    day_now = min(days_held + (1 if mode == "premarket" else 0), max_hold)
    try:
        end = hold_last_day(entry_day, max_hold)
    except Exception:
        end = None
    # the level the exit rule compares against at the next close
    trigger = ""
    try:
        if dyn == "sma5" and math.isfinite(sma5):
            trigger = f"now ${sma5:.2f}"
        elif dyn == "prev_high":
            ref_high = float(H[T - 1] if (mode == "preclose" and partial_bar) else H[T])
            trigger = f"${ref_high:.2f}"
        elif dyn == "high7":
            trigger = f"now ${float(np.nanmax(C[max(0, T - 6):T + 1])):.2f}"
        elif dyn == "rsi2_70":
            trigger = f"now {float(ind['rsi2'].values[T, j]):.0f}"
    except Exception:
        trigger = ""
    status = position_status(sigs, mode, stop=stop, target=target, dyn=dyn, day_now=day_now, max_hold=max_hold,
                             last_day_str=_day(end) if end else "the last hold day", close_label=close_label, trigger=trigger)
    return {"trade_id": tr.get("id"), "ticker": tk, "strategy": strategy, "status": status,
            "day_now": int(day_now), "time_stop_day": str(end) if end else None,
            "entry": _f(entry), "stop": _f(stop), "target": _f(target), "shares": shares,
            "entry_date": str(entry_day), "dyn_exit": dyn,
            "last": _f(last), "asof": str(last_day), "partial_bar": bool(partial_bar),
            "unrealized": _f(unreal), "unrealized_pct": _f((last / entry - 1) * 100 if entry else 0, 2),
            "days_held": int(days_held), "days_left": int(days_left), "max_hold": max_hold,
            "sma5": _f(sma5), "atr": _f(atr), "pm_last": _f(pm_last), "signals": sigs}
