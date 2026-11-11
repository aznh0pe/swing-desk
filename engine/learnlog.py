"""The desk's permanent learning record (written by the daily review, read by every later run).

learning/stats      monthly tallies per strategy: every graded alert (shadow) and real trade (live),
                    split by calm vs stressed market. Months are kept forever, so evidence is never
                    forgotten even after old alerts fall out of an export.
learning/portfolio  execution and portfolio lessons: how often orders fill, entry and stop slippage,
                    realized results vs. the alerts' estimates, what skipped setups and setups beyond
                    your free slots did, income taken. Stop slippage feeds the weekly backtest.
"""
from __future__ import annotations

import copy
import datetime as dt

import numpy as np

HALF_LIFE_MONTHS = 6.0  # older evidence counts less, never zero
STRESS_VIX, STRESS_SPY5 = 1.10, -0.014


def stressed(f: dict) -> bool:
    try:
        return float(f.get("vix_ratio") or 0) >= STRESS_VIX or float(f.get("spy_ret5") or 0) <= STRESS_SPY5
    except Exception:
        return False


def _blank():
    return {"shadow_n": 0, "shadow_wins": 0, "shadow_ret": 0.0, "live_n": 0, "live_wins": 0, "live_ret": 0.0,
            "stress_n": 0, "stress_wins": 0, "calm_n": 0, "calm_wins": 0, "exits": {}}


def monthly_tallies(alerts: list, shadow: list, closed_trades: list) -> dict:
    amap = {a.get("id"): a for a in alerts}
    out: dict = {}
    for s in shadow:
        if s.get("status") != "closed" or not s.get("fill"):
            continue
        a = amap.get(s.get("alert_id")) or {}
        m, st = str(a.get("valid_for", ""))[:7], s.get("strategy") or a.get("strategy")
        if len(m) != 7 or not st:
            continue
        ret = float(s["exit"]) / float(s["fill"]) - 1
        c = out.setdefault(m, {}).setdefault(st, _blank())
        c["shadow_n"] += 1
        c["shadow_wins"] += int(ret > 0)
        c["shadow_ret"] += ret
        reg = "stress" if stressed(a.get("features") or {}) else "calm"
        c[reg + "_n"] += 1
        c[reg + "_wins"] += int(ret > 0)
        r = s.get("reason") or "other"
        c["exits"][r] = c["exits"].get(r, 0) + 1
    for t in closed_trades:
        try:
            ret = float(t["exit_price"]) / float(t["entry_price"]) - 1
        except Exception:
            continue
        m, st = str(t.get("exit_date") or t.get("entry_date") or "")[:7], t.get("strategy")
        if len(m) != 7 or not st:
            continue
        c = out.setdefault(m, {}).setdefault(st, _blank())
        c["live_n"] += 1
        c["live_wins"] += int(ret > 0)
        c["live_ret"] += ret
    for cells in out.values():
        for c in cells.values():
            c["shadow_ret"], c["live_ret"] = round(c["shadow_ret"], 5), round(c["live_ret"], 5)
    return out


def merge_months(prev: dict, fresh: dict, export_complete: bool, oldest_month: str | None) -> dict:
    """Fresh tallies replace months the export fully covers; older months keep what was learned."""
    months = dict(prev or {})
    for m, v in fresh.items():
        if export_complete or oldest_month is None or m > oldest_month or m not in months:
            months[m] = v
    return dict(sorted(months.items()))


def _age_months(m: str, now) -> float:
    y, mo = int(m[:4]), int(m[5:7])
    return max(0.0, (now.year - y) * 12 + (now.month - mo))


def weighted_record(months: dict, strategy: str, now) -> tuple[float, float]:
    """(weighted trades, weighted wins): live fills count double, older months fade with a 6-month half-life."""
    n = w = 0.0
    for m, cells in (months or {}).items():
        c = (cells or {}).get(strategy)
        if not c:
            continue
        wt = 0.5 ** (_age_months(m, now) / HALF_LIFE_MONTHS)
        n += wt * (c.get("shadow_n", 0) + 2 * c.get("live_n", 0))
        w += wt * (c.get("shadow_wins", 0) + 2 * c.get("live_wins", 0))
    return n, w


def summary(months: dict, now) -> dict:
    """Per-strategy all-time, last-3-months and calm/stress records for the dashboard."""
    out = {}
    for m, cells in (months or {}).items():
        recent = _age_months(m, now) < 3
        for st, c in (cells or {}).items():
            r = out.setdefault(st, {"all_n": 0, "all_wins": 0, "all_ret": 0.0, "recent_n": 0, "recent_wins": 0,
                                    "stress_n": 0, "stress_wins": 0, "calm_n": 0, "calm_wins": 0,
                                    "live_n": 0, "live_wins": 0, "first_month": m})
            r["all_n"] += c.get("shadow_n", 0)
            r["all_wins"] += c.get("shadow_wins", 0)
            r["all_ret"] += c.get("shadow_ret", 0.0)
            if recent:
                r["recent_n"] += c.get("shadow_n", 0)
                r["recent_wins"] += c.get("shadow_wins", 0)
            for k in ("stress_n", "stress_wins", "calm_n", "calm_wins", "live_n", "live_wins"):
                r[k] += c.get(k, 0)
            r["first_month"] = min(r["first_month"], m)
    for r in out.values():
        tot = r.pop("all_ret")
        r["all_avg_ret"] = round(tot / r["all_n"], 5) if r["all_n"] else None
    return out


def portfolio_stats(alerts: list, shadow: list, trades: list, ledger: list, today: str, P: dict,
                    fill_rates: dict | None = None) -> dict:
    """Execution and portfolio record. Alert statuses you set on the dashboard:
    filled (or a trade) = the order filled; placed = order working / not yet resolved;
    unfilled = you placed it and the limit never filled; cancelled = you placed it and cancelled it
    (a withdrawn setup, or your choice); skipped = you chose not to place it.
    Fill rate counts filled / (filled + unfilled + still-placed): a cancelled order never had its full chance.
    "Skipped" is your choice only (plus orders you cancelled by choice), never an order that didn't fill."""
    smap = {s.get("alert_id"): s for s in shadow}
    fill_rates = fill_rates or {}
    past = [a for a in alerts if str(a.get("valid_for", "")) <= today and a.get("type") == "entry"]

    def order_state(a):
        st = a.get("status")
        if st == "filled" or a.get("trade_id"):
            return "filled"
        if st == "placed":  # not resolved on the dashboard: the grader knows whether the limit ever traded through
            return "unfilled" if (smap.get(a.get("id")) or {}).get("status") == "unfilled" else "placed"
        return st or "new"

    ostate = {a.get("id"): order_state(a) for a in past}
    placed = [a for a in past if ostate[a.get("id")] in ("placed", "filled", "unfilled")]
    filled = [a for a in past if ostate[a.get("id")] == "filled"]
    unfilled = [a for a in past if ostate[a.get("id")] == "unfilled"]
    cancelled = [a for a in past if ostate[a.get("id")] == "cancelled"]
    # still marked placed although the grader says the limit traded through: probably filled, never recorded
    unrecorded = [a for a in past if ostate[a.get("id")] == "placed"
                  and (smap.get(a.get("id")) or {}).get("status") in ("closed", "open")]

    def sret(a):
        s = smap.get(a.get("id")) or {}
        return float(s["exit"]) / float(s["fill"]) - 1 if s.get("status") == "closed" and s.get("fill") else None

    def rec(rows):
        rs = [r for r in rows if r is not None]
        return {"n": len(rs), "win_rate": round(float(np.mean([r > 0 for r in rs])), 3) if rs else None,
                "avg_ret": round(float(np.mean(rs)), 5) if rs else None}

    entry_gap = [float(a["actual_price"]) / float(a["entry"]) - 1 for a in filled
                 if a.get("actual_price") and a.get("entry")]
    from .portfolio import merged_closed_trades
    closed = merged_closed_trades(trades)  # a position sold in parts counts once
    stop_exits = [float(t["exit_price"]) / float(t["stop"]) - 1 for t in closed
                  if str(t.get("exit_reason", "")).lower().startswith("stop") and t.get("stop")]
    live_rets = [float(t["exit_price"]) / float(t["entry_price"]) - 1 for t in closed]
    amap = {a.get("id"): a for a in alerts}
    expected = [float((amap.get(t.get("alert_id")) or {}).get("win_pct") or t.get("win_pct") or 0) / 100 for t in closed]
    gaps = []
    for t in closed:
        s = smap.get(t.get("alert_id")) or {}
        if s.get("status") == "closed" and s.get("fill"):
            gaps.append((float(t["exit_price"]) / float(t["entry_price"]) - 1) - (float(s["exit"]) / float(s["fill"]) - 1))
    beyond = [sret(a) for a in past if str(a.get("capacity_note", "")).startswith("Ranked #")]
    within = [sret(a) for a in past if a.get("rank") and not str(a.get("capacity_note", "")).startswith("Ranked #")]
    skipped = [sret(a) for a in past if ostate[a.get("id")] == "skipped"
               or (ostate[a.get("id")] == "cancelled" and not a.get("withdrawn"))]
    # unfilled orders the grader counts as filled: the low traded through the limit on the order's day,
    # so a resting day order at the alert price should have filled (an execution gap worth checking)
    through = [a for a in unfilled if (smap.get(a.get("id")) or {}).get("status") in ("closed", "open")]
    exp_fill = [float(fill_rates[a.get("strategy")]) for a in placed if fill_rates.get(a.get("strategy")) is not None]
    by_st = {}
    for a in placed:
        r = by_st.setdefault(a.get("strategy") or "?", {"placed": 0, "filled": 0, "expected": fill_rates.get(a.get("strategy"))})
        r["placed"] += 1
        r["filled"] += int(ostate[a.get("id")] == "filled")
    taken = [sret(a) for a in filled]
    income = -sum(float(l.get("amount") or 0) for l in ledger if l.get("type") == "income")
    profit = sum(float(l.get("amount") or 0) for l in ledger if l.get("type") == "pnl")
    doc = {
        "orders": {"setups_past": len(past), "placed": len(placed), "filled": len(filled),
                   "unfilled": len(unfilled), "cancelled": len(cancelled),
                   "fill_rate": round(len(filled) / len(placed), 3) if placed else None,
                   "expected_fill_rate": round(float(np.mean(exp_fill)), 3) if exp_fill else None,
                   "by_strategy": by_st,
                   "unfilled_through_limit": [{"ticker": a.get("ticker"), "day": a.get("valid_for"), "limit": a.get("entry")}
                                              for a in through],
                   "probably_filled_unrecorded": [{"ticker": a.get("ticker"), "day": a.get("valid_for"), "limit": a.get("entry")}
                                                  for a in unrecorded]},
        "entry_vs_alert": {"n": len(entry_gap), "avg_pct": round(100 * float(np.mean(entry_gap)), 3) if entry_gap else None},
        "stop_slippage": {"n": len(stop_exits), "avg_pct": round(100 * float(np.mean(stop_exits)), 3) if stop_exits else None,
                          "assumed_pct": -float(P.get("risk", {}).get("slippage_pct", 0.05))},
        "realized": {**rec(live_rets), "expected_win_rate": round(float(np.mean(expected)), 3) if expected else None,
                     "avg_r": round(float(np.mean([float(t["r_mult"]) for t in closed if t.get("r_mult") is not None])), 2)
                     if any(t.get("r_mult") is not None for t in closed) else None,
                     "vs_alert_as_sent_pct": round(100 * float(np.mean(gaps)), 3) if gaps else None},
        "taken": rec(taken), "skipped": rec(skipped), "within_slots": rec(within), "beyond_slots": rec(beyond),
        "income_taken": round(income, 2), "realized_pnl": round(profit, 2),
    }
    lessons = []
    eg, ss, rz = doc["entry_vs_alert"], doc["stop_slippage"], doc["realized"]
    if eg["n"] >= 5 and eg["avg_pct"] is not None and eg["avg_pct"] > 0.05:
        lessons.append(f"Fills average {eg['avg_pct']:+.2f}% versus the alert price over {eg['n']} trades: paying up "
                       "costs edge, so keep the limit at the alert price.")
    if ss["n"] >= 5 and ss["avg_pct"] is not None and ss["avg_pct"] < ss["assumed_pct"]:
        lessons.append(f"Stops filled {ss['avg_pct']:.2f}% past the stop price on average ({ss['n']} stops), worse than "
                       f"the {ss['assumed_pct']:.2f}% the backtest assumes; the weekly recalibration will use your figure.")
    if rz["n"] >= 10 and rz["win_rate"] is not None and rz["expected_win_rate"]:
        d = rz["win_rate"] - rz["expected_win_rate"]
        lessons.append(f"Your closed trades won {100 * rz['win_rate']:.0f}% vs {100 * rz['expected_win_rate']:.0f}% "
                       f"expected over {rz['n']} trades ({'ahead of' if d >= 0 else 'behind'} plan by {abs(100 * d):.0f} pts).")
    if rz.get("vs_alert_as_sent_pct") is not None and rz["n"] >= 10 and rz["vs_alert_as_sent_pct"] < -0.3:
        lessons.append(f"Trades as you managed them returned {rz['vs_alert_as_sent_pct']:+.2f}% per trade less than the "
                       "same alerts followed exactly; sticking to the stop, target and exit rules would have done better.")
    b, wn = doc["beyond_slots"], doc["within_slots"]
    if b["n"] >= 15 and wn["n"] >= 15 and b["win_rate"] is not None and wn["win_rate"] is not None:
        if b["win_rate"] >= wn["win_rate"] and (b["avg_ret"] or 0) > 0.005:
            lessons.append(f"Setups ranked beyond your free slots won {100 * b['win_rate']:.0f}% ({b['n']} setups), as well "
                           f"as the ones within them ({100 * wn['win_rate']:.0f}%): more open positions at a smaller size "
                           "each would have captured them.")
    o = doc["orders"]
    if through:
        names = ", ".join(f"{a.get('ticker')} {a.get('valid_for')}" for a in through)
        lessons.append(f"{len(through)} order(s) recorded as not filled traded through the limit on their day ({names}): "
                       "a day limit order at the alert price, placed before the open, should have filled. "
                       "Check the order's price, time in force and when it went in.")
    if unrecorded:
        names = ", ".join(f"{a.get('ticker')} {a.get('valid_for')}" for a in unrecorded)
        lessons.append(f"Probably filled but not recorded ({names}): the low traded through the limit. If you hold it, "
                       "press Mark done (filled) on the dashboard so the scans watch its exits.")
    if o["placed"] >= 10 and o["fill_rate"] is not None and o["expected_fill_rate"] is not None \
            and o["fill_rate"] < o["expected_fill_rate"] - 0.15:
        lessons.append(f"Your limit orders filled {100 * o['fill_rate']:.0f}% of the time vs {100 * o['expected_fill_rate']:.0f}% "
                       f"in the backtest ({o['placed']} orders). Unfilled orders cost trades, not money; if it persists, "
                       "check the orders go in before the open at the alert's limit.")
    sk = doc["skipped"]
    if sk["n"] >= 10 and sk["win_rate"] is not None and sk["win_rate"] >= 0.7:
        lessons.append(f"Setups you skipped won {100 * sk['win_rate']:.0f}% ({sk['n']} setups); trusting A+ setups pays.")
    doc["lessons"] = lessons
    return doc


# ----------------------------------------------------------------------------- grades: A+ and weekly top-ups
# The two kinds of setup the desk sends are tracked, tested and tuned apart:
#   aplus  setups that clear the A+ bar (always sent);
#   topup  setups sent only to keep a week on its minimum pace (the best ones below the A+ bar);
#   watch  the next-best setups a scan listed but did not send (the watchlist), graded on paper so the
#          top-up band keeps learning even in weeks that need no top-ups.
GRADES = ("aplus", "topup", "watch")
GRADE_LABEL = {"aplus": "A+", "topup": "Weekly top-up", "watch": "Watchlist (not sent)"}

TOPUP_MIN_N = 12        # weighted top-up results before their record may change the rule
TOPUP_OFF_WIN = 0.60    # under this the top-ups stop
TOPUP_TIGHT_GAP = 0.07  # this far under the tested top-up win rate, top-ups must clear the A+ win estimate
STRAT_TOPUP_MIN_N = 8   # weighted top-ups of one strategy before its own record counts
STRAT_PRIOR_N = 20      # its record is shrunk toward the tested top-up win rate with this weight
STRAT_TOPUP_GAP = 0.08  # a strategy this far under (after shrinkage) stops sending top-ups
APLUS_MIN_N = 25        # weighted A+ results before the A+ record is judged
APLUS_GAP = 0.10        # this far under the tested A+ win rate puts A+ on watch
WATCH_KEEP_DAYS = 75    # watchlist ids remembered this long, so each paper result counts once


def grade_of(doc) -> str:
    d = doc or {}
    if d.get("grade_kind") in GRADES:
        return d["grade_kind"]
    return "topup" if d.get("topup_note") or d.get("grade") == "A" else "aplus"


def _cell():
    return {"shadow_n": 0, "shadow_wins": 0, "shadow_ret": 0.0, "live_n": 0, "live_wins": 0, "live_ret": 0.0,
            "stress_n": 0, "stress_wins": 0, "calm_n": 0, "calm_wins": 0, "exits": {}}


def _add_shadow(c: dict, ret: float, alert: dict, reason) -> None:
    c["shadow_n"] += 1
    c["shadow_wins"] += int(ret > 0)
    c["shadow_ret"] += ret
    reg = "stress" if stressed((alert or {}).get("features") or {}) else "calm"
    c[reg + "_n"] += 1
    c[reg + "_wins"] += int(ret > 0)
    r = reason or "other"
    c["exits"][r] = c["exits"].get(r, 0) + 1


def _round(c: dict) -> dict:
    c["shadow_ret"], c["live_ret"] = round(c["shadow_ret"], 5), round(c["live_ret"], 5)
    return c


def grade_tallies(alerts: list, shadow: list, closed_trades: list) -> dict:
    """{month: {grade: {strategy: cell}}} for the setups the desk sent: each alert's paper result in its
    session's month, your real trades in their exit month."""
    amap = {a.get("id"): a for a in alerts}
    out: dict = {}
    for s in shadow:
        if s.get("status") != "closed" or not s.get("fill"):
            continue
        a = amap.get(s.get("alert_id"))
        m = str((a or {}).get("valid_for", ""))[:7]
        st = s.get("strategy") or (a or {}).get("strategy")
        if not a or len(m) != 7 or not st:
            continue
        ret = float(s["exit"]) / float(s["fill"]) - 1
        _add_shadow(out.setdefault(m, {}).setdefault(grade_of(a), {}).setdefault(st, _cell()), ret, a, s.get("reason"))
    for t in closed_trades:
        a = amap.get(t.get("alert_id"))
        g = t.get("grade_kind") if t.get("grade_kind") in GRADES else (grade_of(a) if a else None)
        m, st = str(t.get("exit_date") or t.get("entry_date") or "")[:7], t.get("strategy")
        if not g or len(m) != 7 or not st:
            continue  # grade unknown (its alert left the export): the month already on record keeps it
        try:
            ret = float(t["exit_price"]) / float(t["entry_price"]) - 1
        except Exception:
            continue
        c = out.setdefault(m, {}).setdefault(g, {}).setdefault(st, _cell())
        c["live_n"] += 1
        c["live_wins"] += int(ret > 0)
        c["live_ret"] += ret
    for cells in out.values():
        for g in cells.values():
            for c in g.values():
                _round(c)
    return out


def watch_from_scans(scans: list, alerts: list) -> list:
    """The watchlist setups the scans listed but did not send: once per session, ticker and strategy."""
    sent = {(str(a.get("valid_for")), a.get("ticker")) for a in alerts}
    out: dict = {}
    for sc in sorted(scans or [], key=lambda x: str(x.get("started_at") or "")):
        for e in sc.get("near_misses") or []:
            tk, vf = e.get("ticker"), e.get("valid_for")
            if not tk or not vf or (str(vf), tk) in sent or e.get("entry") is None:
                continue
            wid = f"w_{vf}_{tk}_{e.get('strategy')}"
            out.setdefault(wid, {**e, "id": wid, "grade_kind": "watch", "type": "watch"})
    return list(out.values())


def accumulate_watch(prev_doc: dict | None, watch: list, wshadow: list, now) -> tuple[dict, dict]:
    """Paper results of the watchlist, each added once: the scans that list them are pruned later, so the
    months here are never rebuilt, only added to."""
    months = copy.deepcopy((prev_doc or {}).get("watch_months") or {})
    ids = dict((prev_doc or {}).get("watch_ids") or {})
    cutoff = (now.date() - dt.timedelta(days=WATCH_KEEP_DAYS)).isoformat()
    amap = {a["id"]: a for a in watch or []}
    for s in wshadow or []:
        a = amap.get(s.get("alert_id"))
        if not a or s.get("status") != "closed" or not s.get("fill") or a["id"] in ids:
            continue
        vf = str(a.get("valid_for"))
        if vf < cutoff:
            continue
        ret = float(s["exit"]) / float(s["fill"]) - 1
        c = months.setdefault(vf[:7], {}).setdefault(a.get("strategy") or "?", _cell())
        _add_shadow(c, ret, a, s.get("reason"))
        _round(c)
        ids[a["id"]] = vf
    return dict(sorted(months.items())), {k: v for k, v in ids.items() if v >= cutoff}


def _acc():
    return {"n": 0, "wins": 0, "ret": 0.0, "live_n": 0, "live_wins": 0, "live_ret": 0.0, "stress_n": 0,
            "stress_wins": 0, "calm_n": 0, "calm_wins": 0, "wn": 0.0, "ww": 0.0, "exits": {}}


def _add_acc(r: dict, c: dict, wt: float) -> None:
    r["n"] += c.get("shadow_n", 0)
    r["wins"] += c.get("shadow_wins", 0)
    r["ret"] += c.get("shadow_ret", 0.0)
    for k in ("live_n", "live_wins", "live_ret", "stress_n", "stress_wins", "calm_n", "calm_wins"):
        r[k] += c.get(k, 0)
    r["wn"] += wt * (c.get("shadow_n", 0) + 2 * c.get("live_n", 0))
    r["ww"] += wt * (c.get("shadow_wins", 0) + 2 * c.get("live_wins", 0))
    for k, v in (c.get("exits") or {}).items():
        r["exits"][k] = r["exits"].get(k, 0) + v


def _fin(r: dict) -> dict:
    n, ln = r["n"], r["live_n"]
    return {"n": n, "wins": r["wins"], "win_rate": round(r["wins"] / n, 4) if n else None,
            "avg_ret": round(r["ret"] / n, 5) if n else None,
            "live_n": ln, "live_wins": r["live_wins"], "live_win_rate": round(r["live_wins"] / ln, 4) if ln else None,
            "live_avg_ret": round(r["live_ret"] / ln, 5) if ln else None,
            "stress_n": r["stress_n"], "stress_wins": r["stress_wins"], "calm_n": r["calm_n"], "calm_wins": r["calm_wins"],
            "weighted_n": round(r["wn"], 1), "weighted_win_rate": round(r["ww"] / r["wn"], 4) if r["wn"] else None,
            "exits": r["exits"]}


def grade_summary(grade_months: dict, watch_months: dict, now) -> dict:
    """Per grade: all-time, last 3 months, calm vs stressed, your trades, the weighted record the rules use
    (real fills count double, 6-month half-life) and the same per strategy."""
    trees = {"aplus": {m: (c or {}).get("aplus") or {} for m, c in (grade_months or {}).items()},
             "topup": {m: (c or {}).get("topup") or {} for m, c in (grade_months or {}).items()},
             "watch": watch_months or {}}
    out = {}
    for g, tree in trees.items():
        tot, rec, by, first = _acc(), _acc(), {}, None
        for m, cells in sorted(tree.items()):
            if not cells:
                continue
            age = _age_months(m, now)
            wt = 0.5 ** (age / HALF_LIFE_MONTHS)
            first = first or m
            for st, c in cells.items():
                _add_acc(tot, c, wt)
                _add_acc(by.setdefault(st, _acc()), c, wt)
                if age < 3:
                    _add_acc(rec, c, wt)
        r = _fin(tot)
        r.update(first_month=first, label=GRADE_LABEL[g], recent={k: _fin(rec)[k] for k in ("n", "wins", "win_rate", "avg_ret")},
                 by_strategy={st: _fin(v) for st, v in sorted(by.items())})
        out[g] = r
    return out


def _expected_topup(G: dict) -> float:
    if G.get("oos_topup_win_rate"):
        return float(G["oos_topup_win_rate"])
    a = G.get("oos_aplus_win_rate")
    return float(a) - 0.06 if a else 0.69


def topup_policy(stats: dict | None, G: dict) -> dict:
    """normal / tight / off for the weekly top-ups, from their own weighted record against their tested win rate.
    tight = only top-ups that also clear the A+ win estimate; off = no top-ups. Applied daily and weekly."""
    t = ((stats or {}).get("grade_summary") or {}).get("topup") or {}
    exp = _expected_topup(G)
    n, wr = float(t.get("weighted_n") or 0), t.get("weighted_win_rate")
    base = {"mode": "normal", "expected": round(exp, 4), "record": wr, "n": n, "note": ""}
    if n < TOPUP_MIN_N or wr is None:
        return base
    if wr < TOPUP_OFF_WIN:
        return {**base, "mode": "off",
                "note": (f"Weekly top-ups stopped: they won {wr:.0%} over {n:.0f} weighted results (real trades count double), "
                         f"under the {TOPUP_OFF_WIN:.0%} floor; tested {exp:.0%}. The watchlist keeps being graded, and they "
                         "return once the record recovers or the losing stretch ages out.")}
    if wr < exp - TOPUP_TIGHT_GAP:
        return {**base, "mode": "tight",
                "note": (f"Weekly top-ups now need the full A+ win estimate: they won {wr:.0%} over {n:.0f} weighted results "
                         f"vs {exp:.0%} tested.")}
    return base


def topup_strategy_exclusions(stats: dict | None, G: dict) -> dict:
    """{strategy: reason} for strategies whose own top-ups trail the tested top-up win rate (record shrunk
    toward the tested rate, so a few unlucky results don't exclude a strategy)."""
    by = ((((stats or {}).get("grade_summary") or {}).get("topup") or {}).get("by_strategy")) or {}
    exp = _expected_topup(G)
    out = {}
    for st, r in by.items():
        n, wr = float(r.get("weighted_n") or 0), r.get("weighted_win_rate")
        if n < STRAT_TOPUP_MIN_N or wr is None:
            continue
        shr = (wr * n + exp * STRAT_PRIOR_N) / (n + STRAT_PRIOR_N)
        if shr < exp - STRAT_TOPUP_GAP:
            out[st] = (f"its own top-ups won {wr:.0%} over {n:.0f} weighted results (all top-ups tested at {exp:.0%}); "
                       "it can still send A+ setups")
    return out


def aplus_watch(stats: dict | None, G: dict) -> dict | None:
    """A+ setups well under their tested win rate: the weekly recalibration then picks the stricter A+ bar."""
    a = ((stats or {}).get("grade_summary") or {}).get("aplus") or {}
    exp = G.get("oos_aplus_win_rate")
    n, wr = float(a.get("weighted_n") or 0), a.get("weighted_win_rate")
    if not exp or n < APLUS_MIN_N or wr is None or wr >= float(exp) - APLUS_GAP:
        return None
    return {"record": round(wr, 4), "n": round(n, 1), "expected": float(exp),
            "note": (f"A+ setups won {wr:.0%} over {n:.0f} weighted results vs {float(exp):.0%} tested: the weekly "
                     "recalibration will favour the stricter A+ bar until the record recovers.")}


def stats_doc(prev_doc: dict | None, alerts: list, shadow: list, closed_trades: list, export_complete: bool, now,
              watch: list | None = None, wshadow: list | None = None) -> dict:
    fresh = monthly_tallies(alerts, shadow, closed_trades)
    oldest = min((str(a.get("valid_for", ""))[:7] for a in alerts if a.get("valid_for")), default=None)
    months = merge_months((prev_doc or {}).get("months") or {}, fresh, export_complete, oldest)
    prev_gm = {m: c for m, c in ((prev_doc or {}).get("grade_months") or {}).items()
               if isinstance(c, dict) and all(isinstance(v, dict) and "shadow_n" not in v for v in c.values())}
    gm = merge_months(prev_gm, grade_tallies(alerts, shadow, closed_trades), export_complete, oldest)
    wm, wids = accumulate_watch(prev_doc, watch or [], wshadow or [], now)
    return {"updated_at": now.isoformat(timespec="seconds"), "months": months, "summary": summary(months, now),
            "grade_months": gm, "watch_months": wm, "watch_ids": wids, "grade_summary": grade_summary(gm, wm, now),
            "half_life_months": HALF_LIFE_MONTHS,
            "about": ("Monthly tallies of every graded alert (shadow) and real trade (live) per strategy, kept "
                      "permanently; score adjustments use them with live fills counted double and a 6-month half-life. "
                      "grade_months splits the same record into A+ setups and weekly top-ups (per strategy); "
                      "watch_months grades the watchlist setups that were listed but not sent.")}
