"""Nov 7 audit: live position-management rules are replayed in the backtest, the shadow grader matches
the backtest, the research lab keeps a curated backlog and records variants, and grading reports its
honesty checks (by year and a time split)."""
import json
import pathlib

import numpy as np
import pandas as pd

from engine import backtest as B
from engine import calibrate as CB
from engine import monitor as MON
from engine import rules as R
from engine import strategies as S
from engine import research as RS

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _frames(closes, opens=None, highs=None, lows=None, start="2026-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.DataFrame({"X": closes}, index=idx, dtype=float)
    o = pd.DataFrame({"X": opens if opens is not None else closes}, index=idx, dtype=float)
    h = pd.DataFrame({"X": highs if highs is not None else np.maximum(o["X"], c["X"]) + 0.5}, index=idx, dtype=float)
    l = pd.DataFrame({"X": lows if lows is not None else np.minimum(o["X"], c["X"]) - 0.5}, index=idx, dtype=float)
    z = c * 0
    ind = {"open": o, "high": h, "low": l, "close": c, "volume": z + 1e6, "atr10": z + 2.0, "sma5": c.rolling(5, min_periods=1).mean(),
           "sma50": z + 50.0, "sma200": z + 50.0, "rsi2": z + 50.0, "max7c": c.rolling(7, min_periods=1).max(),
           "dist200": z, "slope200": z, "atr_pct": z + 0.02, "rs_pct": z + 0.5, "ibs": z + 0.5, "streak": z, "adx14": z + 20,
           "vol_ratio": z + 1, "dist50": z, "ret5": z, "sector_up": z + 1, "is_etf": z.astype(bool), "is_stock": z.astype(bool) | True,
           "is_lev": z.astype(bool), "market": pd.DataFrame({"spy_above200": 1.0, "spy_dist200": 0.05, "spy_ret5": 0.0, "breadth50": 0.6,
                                                              "vix": 15.0, "vix_ratio": 1.0, "spy": 500.0}, index=idx)}
    return ind


def _plan(ind, signal_row, limit, hold=10):
    mask = ind["close"] * 0 == 1
    mask.iloc[signal_row, 0] = True
    lim = ind["close"] * 0 + limit
    return {"name": "T", "entry_kind": "limit_next", "valid_days": 1, "max_hold": hold, "dyn_exit": "none",
            "stop": {"kind": "atr", "mult": 3.0}, "target": {"kind": "atr", "mult": 10.0}, "mask": mask, "limit": lim}


def test_lock_in_rule_is_replayed_in_the_backtest():
    # fill at 100, close jumps 2.5 ATR (ATR 2) -> stop moves to 102 -> next open 101 stops it out
    closes = [100, 100, 105, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    opens = [100, 101, 104, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    lows = [99, 99, 103, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    ind = _frames(closes, opens, lows=lows)
    P = S.params_with_defaults(None)
    tr = B.simulate(_plan(ind, 0, 100.0, hold=6), ind, P)
    assert tr.iloc[0]["reason"] == "stop" and abs(tr.iloc[0]["exit"] - 101 * (1 - 0.0005)) < 1e-6
    tr2 = B.simulate(_plan(ind, 0, 100.0, hold=6), ind, P, manage={"lock_atr": 0})
    assert tr2.iloc[0]["reason"] == "time"
    assert P["manage"]["be_atr"] == 0  # no breakeven stop: it cost about a point of A+ win rate when measured


def test_a_touch_of_the_limit_is_not_a_fill():
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    lows = [99, 100.0, 101, 102, 103, 104, 105, 106, 107, 108]  # day-1 low exactly at the limit
    ind = _frames(closes, lows=lows)
    P = S.params_with_defaults(None)
    assert P["manage"]["fill"] == "through"
    tr = B.simulate(_plan(ind, 0, 100.0), ind, P, include_unfilled=True)
    assert tr.iloc[0]["reason"] == "unfilled"
    tr2 = B.simulate(_plan(ind, 0, 100.0), ind, P, include_unfilled=True, manage={"fill": "touch"})
    assert tr2.iloc[0]["filled"]


def test_shadow_grader_matches_the_backtest_rules():
    closes = [100, 99, 103, 104, 110, 104, 104, 104, 104, 104]
    lows = [99, 98, 102, 103, 109, 103, 103, 103, 103, 103]
    ind = _frames(closes, lows=lows)
    ind["rsi2"].iloc[3, 0] = 80
    base = {"id": "a", "ticker": "X", "strategy": "T", "valid_for": str(ind["close"].index[1].date()),
            "entry": 99.0, "stop": 90.0, "target": 150.0, "hold_max_days": 8}
    r = CB.shadow_outcome({**base, "dyn_exit": "rsi2_70"}, ind)
    assert r["status"] == "closed" and r["reason"] == "signal" and r["exit"] == 104  # RSI(2) above 70 on day 3
    r = CB.shadow_outcome({**base, "dyn_exit": "high7"}, ind)
    assert r["status"] == "closed" and r["reason"] == "signal" and r["exit"] == 103  # 7-day closing high on day 2
    r = CB.shadow_outcome({**base, "dyn_exit": "none", "entry": 98.0}, ind)
    assert r["status"] == "unfilled"  # the low only touched 98
    r = CB.shadow_outcome({**base, "dyn_exit": "none", "entry": 98.0, "valid_days": 2}, ind)
    assert r["status"] == "unfilled"


def _trade(**kw):
    tr = {"id": "t1", "ticker": "X", "entry_price": 100.0, "shares": 1, "stop": 94.0, "target": 120.0, "max_hold": 8,
          "dyn_exit": "none", "strategy": "MR_STRETCH", "entry_date": "2026-01-02"}
    tr.update(kw)
    return tr


def test_monitor_stop_management_and_trend_break():
    P = S.params_with_defaults(None)
    ind = _frames([100, 102.5, 102.5])  # up 1.25 ATR: no breakeven nag any more
    codes = [s["code"] for s in MON.check_trade(_trade(), ind, "postclose", {}, None, P, False)["signals"]]
    assert "BREAKEVEN" not in codes and "RAISE_STOP" not in codes
    ind = _frames([100, 104.5, 104.5])  # up 2.25 ATR: lock in entry + 1 ATR
    sig = [s for s in MON.check_trade(_trade(), ind, "postclose", {}, None, P, False)["signals"] if s["code"] == "RAISE_STOP"]
    assert sig and sig[0]["new_stop"] == 102.0
    # a close below the 50-day ends trend-following setups, not the dip-buying leader pullback (measured: it hurt)
    ind = _frames([100, 45, 45])
    codes = [s["code"] for s in MON.check_trade(_trade(strategy="LEADER_PULLBACK"), ind, "postclose", {}, None, P, False)["signals"]]
    assert "TREND_BREAK" not in codes
    codes = [s["code"] for s in MON.check_trade(_trade(strategy="TREND_PULLBACK"), ind, "postclose", {}, None, P, False)["signals"]]
    assert "TREND_BREAK" in codes


def test_grading_reports_by_year_and_time_split():
    rng = np.random.default_rng(1)
    n = 6000
    dates = pd.to_datetime(rng.choice(pd.bdate_range("2022-01-03", "2026-11-05").values, n))
    score = rng.uniform(40, 100, n)
    p = rng.uniform(0.6, 0.8, n)
    win = rng.random(n) < (0.55 + 0.3 * (score - 40) / 60)
    ret = np.where(win, rng.uniform(0.005, 0.05, n), -rng.uniform(0.005, 0.04, n))
    tr = pd.DataFrame({"signal_date": dates, "ticker": rng.choice(list("ABCDEFGH"), n), "score": score, "p": p,
                       "ret": ret, "filled": True})
    G = CB.calibrate_grades(tr, pd.Timestamp("2022-01-01"))
    assert "by_year" in G and {r["year"] for r in G["by_year"]} >= {2022, 2026}
    ts = G["time_split"]
    assert ts["chosen_bar_early"]["window"][0].startswith("2022") and ts["chosen_bar_late"]["window"][0].startswith("2024")
    assert ts["early_selection"] is None or "early_bar_on_late" in ts


def test_seed_backlog_is_valid_distinct_and_consumed_in_order(tmp_path):
    seen = set()
    for f in RS.SEED_FILES:
        for sp in json.loads(f.read_text()):
            assert R.validate(sp) == [], sp["id"]
            assert sp["id"] not in seen, sp["id"]
            seen.add(sp["id"])
    assert len(RS.SEED_FILES) >= 3
    first = json.loads(RS.SEED_FILES[0].read_text())
    known = {s["id"]: {"spec": s, "signature": R.signature(s), "label": s["label"]} for s in first}
    backlog = RS.seed_backlog(known)
    assert backlog and backlog[0]["id"] not in known
    # a variant is detected even when its thresholds differ
    v2 = [s for f in RS.SEED_FILES for s in json.loads(f.read_text()) if s["id"] == "R_ETF_3DAY_HL_V2"][0]
    sim = RS.similar_ideas(v2, known)
    assert sim and sim[0]["id"] == "R_ETF_3DAY_HL"


def test_research_next_offers_the_backlog_when_the_queue_is_empty(tmp_path):
    st = tmp_path / "state"
    (st / "research").mkdir(parents=True)
    (st / "research_results").mkdir()
    RS.cmd_next(str(st), str(tmp_path / "out"))
    nx = json.loads((tmp_path / "out" / "next.json").read_text())
    assert nx["from_backlog"] and nx["next"] == nx["backlog"][0]
    spec = json.loads(pathlib.Path(nx["spec_file"]).read_text())
    assert spec["id"] == nx["next"] and R.validate(spec) == []


def test_every_position_signal_reaches_the_headline_and_the_email():
    """No code the monitor can raise may fall through to 'No action needed'."""
    from engine import scan as SC
    import re as _re
    codes = set(_re.findall(r'"code": "([A-Z_]+)"', pathlib.Path(MON.__file__).read_text()))
    grouped = SC.EXIT_CODES | SC.FILL_CODES | SC.RAISE_CODES | SC.NOTICE_CODES | SC.CHECK_CODES
    assert codes <= grouped, codes - grouped
    acting = SC.EXIT_CODES | SC.FILL_CODES | SC.RAISE_CODES | SC.CHECK_CODES
    for code in sorted(acting):
        pos = [{"trade_id": "t", "ticker": "X", "last": 1, "unrealized": 0, "days_held": 1, "max_hold": 5,
                "signals": [{"code": code, "level": "x", "action": "DO", "urgency": "now", "text": "t"}]}]
        doc = {"counts": {"aplus": 0, **SC.signal_counts(pos)}, "alert_updates": {}, "premarket_checks": [], "data": {"status": "ok"}}
        assert SC.headline(doc) != "No action needed", code


def test_time_stop_is_announced_the_day_before():
    P = S.params_with_defaults(None)
    closes = [100] * 8
    ind = _frames(closes, start="2026-01-02")  # entered Jan 2 -> day 8 is Jan 13; day 7 close = Jan 12
    ind = {k: (v.iloc[:7] if hasattr(v, "iloc") else v) for k, v in ind.items()}
    ind["market"] = ind["market"].iloc[:7]
    codes = [s["code"] for s in MON.check_trade(_trade(max_hold=8), ind, "postclose", {}, None, P, False)["signals"]]
    assert "TIME_STOP_NEXT" in codes and "TIME_STOP" not in codes
    ind8 = _frames(closes, start="2026-01-02")
    codes = [s["code"] for s in MON.check_trade(_trade(max_hold=8), ind8, "postclose", {}, None, P, False)["signals"]]
    assert "TIME_STOP" in codes


def test_risk_governor_steps_down_and_restores():
    from engine.portfolio import risk_governor
    base = {"settings": {"risk_pct": 2.0}, "trades": {}, "ledger": {"i": {"doc": {"type": "initial", "amount": 5000, "date": "2026-11-05"}}}, "learning": {}}
    g = risk_governor(base, 5000, base["settings"])
    assert g["risk_pct"] == 2.0 and not g["stepped_down"]
    # a 9% realised drawdown steps the size down to the floor
    dd = dict(base); dd["ledger"] = {**base["ledger"], "l": {"doc": {"type": "pnl", "amount": -450, "date": "2026-12-01"}}}
    g = risk_governor(dd, 4550, dd["settings"])
    assert g["risk_pct"] == 1.25 and g["stepped_down"] and "drawdown 9.0%" in g["reason"]
    # 20 closed trades at 55% wins step it down even with no drawdown
    tr = {f"t{i}": {"doc": {"status": "closed", "entry_price": 100, "exit_price": 101 if i < 11 else 99, "exit_date": f"2026-12-{i+1:02d}"}} for i in range(20)}
    wk = dict(base); wk["trades"] = tr
    g = risk_governor(wk, 5000, wk["settings"])
    assert g["risk_pct"] == 1.25 and "won 55%" in g["reason"]
    # the user's own setting below the floor is respected as is
    low = dict(base); low["settings"] = {"risk_pct": 1.0}
    assert risk_governor(low, 5000, low["settings"])["risk_pct"] == 1.0


def test_a_position_sold_in_parts_counts_once():
    from engine.portfolio import merged_closed_trades, risk_governor
    parent = {"id": "t_1", "ticker": "CRM", "status": "closed", "entry_price": 200.0, "shares": 1, "exit_price": 190.0,
              "exit_date": "2026-11-12", "exit_reason": "Stop filled", "pnl": -10.0, "r_mult": -0.5, "strategy": "FAIR_VALUE", "days_held": 4}
    part = {"id": "t_1_sabc", "parent_id": "t_1", "ticker": "CRM", "status": "closed", "entry_price": 200.0, "shares": 1, "exit_price": 230.0,
            "exit_date": "2026-11-10", "exit_reason": "Target filled", "pnl": 30.0, "r_mult": 1.5, "strategy": "FAIR_VALUE", "days_held": 2}
    other = {"id": "t_2", "status": "closed", "entry_price": 50.0, "shares": 2, "exit_price": 49.0, "exit_date": "2026-11-11", "pnl": -2.0}
    still_open = {"id": "t_3", "status": "open", "entry_price": 10.0, "shares": 5}
    m = {t["id"]: t for t in merged_closed_trades([parent, part, other, still_open])}
    assert set(m) == {"t_1", "t_2"}
    crm = m["t_1"]
    assert crm["shares"] == 2 and abs(crm["exit_price"] - 210.0) < 1e-9 and crm["pnl"] == 20.0 and crm["parts"] == 2
    assert crm["exit_date"] == "2026-11-12" and crm["r_mult"] == 0.5 and "parent_id" not in crm
    # a part sold while the rest is still open counts with the parts sold so far
    only_part = merged_closed_trades([part, {**parent, "status": "open"}])
    assert len(only_part) == 1 and only_part[0]["pnl"] == 30.0
    # the governor's trailing win rate sees one trade per position
    tr = {f"t{i}": {"doc": {"id": f"t{i}", "status": "closed", "entry_price": 100, "exit_price": 101 if i < 12 else 99,
                            "exit_date": f"2026-12-{i+1:02d}"}} for i in range(20)}
    tr.update({f"p{i}": {"doc": {"id": f"p{i}", "parent_id": "t0", "status": "closed", "entry_price": 100, "exit_price": 101,
                                 "exit_date": "2026-12-01"}} for i in range(5)})
    st = {"settings": {"risk_pct": 2.0}, "trades": tr, "learning": {},
          "ledger": {"i": {"doc": {"type": "initial", "amount": 5000, "date": "2026-11-05"}}}}
    g = risk_governor(st, 5000, st["settings"])
    assert g["trailing_win"] == 0.6  # 20 positions, 12 winners: the five parts of t0 don't add wins


def test_premarket_check_never_says_keep_for_a_withdrawn_order(tmp_path):
    """Nov 9: the pre-market email listed CRWD and NUE as KEEP although later scans had withdrawn them."""
    from engine import scan as SC
    state = {"scans": {
        "a": {"doc": {"id": "20261107-1243-postclose", "started_at": "2026-11-07T12:43:00-08:00",
                      "alert_updates": {"e_2026-11-09_CRWD_MR_STRETCH": {"withdrawn": True, "withdrawn_reason": "No longer qualifies."}}}},
        "b": {"doc": {"id": "20261107-0917-postclose", "started_at": "2026-11-07T09:17:00-08:00",
                      "alert_updates": {"e_2026-11-09_CRWD_MR_STRETCH": {"withdrawn": False}}}}}}
    ov = SC.current_overlay(state)
    assert ov["e_2026-11-09_CRWD_MR_STRETCH"]["withdrawn"] is True  # newest scan wins, whatever the export order
    doc = {"premarket_checks": [{"ticker": "CRWD", "verdict": "withdrawn", "placed": False},
                                {"ticker": "NUE", "verdict": "withdrawn", "placed": True},
                                {"ticker": "TWLO", "verdict": "keep"}],
           "counts": {"aplus": 0}, "alert_updates": {}, "data": {"status": "ok"}}
    h = SC.headline(doc)
    assert "1 order to cancel" in h and "1 withdrawn setup not to place" in h


def test_every_open_position_gets_one_plain_instruction():
    """Each open position carries one status line: keep holding, sell (when), raise the stop, record a fill ..."""
    from engine import scan as SC
    P = S.params_with_defaults(None)
    st = MON.check_trade(_trade(), _frames([100, 100.5, 100.2]), "postclose", {}, None, P, False)["status"]
    assert st["label"] == "KEEP HOLDING" and st["tone"] == "hold" and "(time stop)" in st["detail"]
    st = MON.check_trade(_trade(), _frames([100, 104.5, 104.5]), "postclose", {}, None, P, False)["status"]
    assert st["code"] == "RAISE_STOP" and st["label"] == "KEEP HOLDING · RAISE THE STOP TO $102.00" and st["tone"] == "act"
    # the day the hold window ends: pre-close says sell at today's close, the scans after it say sell at the open
    ind8 = _frames([100] * 8, start="2026-01-02")
    pre = MON.check_trade(_trade(max_hold=8), ind8, "preclose", {}, None, P, True)
    post = MON.check_trade(_trade(max_hold=8), ind8, "postclose", {}, None, P, False)
    assert pre["status"]["label"].startswith("TIME STOP: SELL AT TODAY'S CLOSE") and pre["status"]["when"] == "close"
    assert post["status"]["label"] == "TIME STOP: SELL AT THE OPEN" and post["status"]["when"] == "open"
    assert post["day_now"] == 8 and post["time_stop_day"] == "2026-01-13"
    # every code the monitor raises maps to a clear instruction in every scan
    for code in sorted(SC.EXIT_CODES | SC.FILL_CODES | SC.RAISE_CODES | SC.NOTICE_CODES | SC.CHECK_CODES):
        for mode in ("premarket", "preclose", "postclose"):
            st = MON.position_status([{"code": code, "text": "t.", "new_stop": 1.5}], mode, stop=1.0, target=2.0, dyn="sma5",
                                     day_now=1, max_hold=5, last_day_str="Fri Jan 9")
            assert st["label"] and st["tone"] in SC.TONE_MARK and st["when"] in {"now", "open", "close", "next", "none"}, (code, st)
            if code in {"EXIT_RULE", "TIME_STOP", "TREND_BREAK", "EARNINGS", "GAP_BELOW_STOP"}:
                assert st["tone"] == "sell" and "SELL" in st["label"], (code, mode, st)
            if code in SC.RAISE_CODES | SC.NOTICE_CODES:
                assert st["label"].startswith("KEEP HOLDING"), (code, st)
    # the email marks open P&L the way the dashboard colours it
    assert SC.pnl_mark(4.456).startswith("🟢 +$4.46") and SC.pnl_mark(-0.63).startswith("🔴 -$0.63")
    assert SC.pnl_mark(0.004).startswith("⚪ $0.00") and SC.pnl_mark(-0.004).startswith("⚪ $0.00")


def test_a_reached_target_with_an_exit_signal_reads_as_one_plan():
    """Target reached and exit rule met in the same scan: record the sale; sell only if the bracket didn't fill."""
    import datetime as _dt
    from engine import scan as SC
    hit = {"code": "TARGET_HIT", "level": "good", "action": "CONFIRM TARGET FILL", "urgency": "now",
           "text": "High $326.66 reached your target $324.96. Confirm the fill and log the exit price."}
    ex = {"code": "EXIT_RULE", "level": "alert", "action": "SELL AT CLOSE", "urgency": "today",
          "text": "Exit rule met: the close is above its 5-day average (price is $325.69)."}
    st = MON.position_status([hit, ex], "preclose", stop=276.74, target=324.96, dyn="sma5", day_now=2, max_hold=8,
                             last_day_str="Tue Nov 17")
    assert st["label"] == "TARGET REACHED: RECORD THE SALE" and st["tone"] == "good"
    assert st["detail"].endswith("If it didn't fill, sell it now, before the close: its exit rule is met too.")
    pos = [{"trade_id": "t", "ticker": "NDSN", "strategy": "", "last": 325.69, "unrealized": 15.2, "unrealized_pct": 4.9,
            "days_held": 2, "max_hold": 8, "entry": 310.49, "stop": 276.74, "target": 324.96, "shares": 1,
            "entry_date": "2026-11-06", "signals": [hit, ex], "status": st, "day_now": 2}]
    scan = {"mode": "preclose", "market": {"label": "Bull", "spy": 755.17, "spy_vs_200d_pct": 3.7, "spy_5d_pct": 0.4,
            "breadth_above_50d_pct": 68.0, "vix": 19.8}, "funds": {"capital": 5000, "cash": 4442.47, "open_count": 2},
            "data": {"status": "ok", "notes": []}, "alert_updates": {}}
    txt = SC.make_digest(scan, [], pos, [], [], 0, S.params_with_defaults(None), _dt.datetime(2026, 11, 9, 12, 31))
    assert "NDSN: ONLY IF STILL OPEN, SELL AT CLOSE (today). Its target was reached" in txt
    assert "NDSN: 🟢 TARGET REACHED: RECORD THE SALE." in txt and "P&L 🟢 +$15.20" in txt


def test_an_order_that_never_filled_expires_with_its_proof():
    """Day orders expire at the close: once the session is over, every placed order gets a verdict from that day's
    prices (strict, like the backtest: the low must trade BELOW the limit), so it never lingers as a live alert."""
    import datetime as _dt
    import pandas as pd
    from engine import scan as SC
    idx = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"])
    ind = {"low": pd.DataFrame({"TWLO": [197.57, 199.32, 203.26], "IEX": [222.60, 228.97, 234.04]}, index=idx),
           "open": pd.DataFrame({"TWLO": [205.01, 202.20, 204.92], "IEX": [223.82, 230.64, 234.43]}, index=idx)}
    a = {"id": "e1", "ticker": "TWLO", "status": "placed", "valid_for": "2026-11-09", "entry": 195.28, "stop": 144.73,
         "target": 220.56, "order": "Buy LIMIT $195.28, day order"}
    c = SC.order_fill_check(a, ind, _dt.date(2026, 11, 10), "premarket", False)
    assert c["filled"] is False and c["low"] == 199.32 and c["text"].startswith("Didn't fill") and "expired" in c["text"]
    touch = SC.order_fill_check({**a, "ticker": "IEX", "valid_for": "2026-11-06", "entry": 222.60}, ind, _dt.date(2026, 11, 9), "premarket", False)
    assert touch["filled"] is False and touch["touched"] is True and touch["text"].startswith("Probably didn't fill")
    assert "check your brokerage" in touch["text"]  # a touch can fill: ask, don't assume
    hit = SC.order_fill_check({**a, "valid_for": "2026-11-10", "entry": 210.0}, ind, _dt.date(2026, 11, 10), "postclose", False)
    assert hit["filled"] is True and hit["text"].startswith("Probably FILLED") and "opened at $204.92" in hit["text"]
    unmarked = SC.order_fill_check({**a, "status": "new", "valid_for": "2026-11-10", "entry": 210.0}, ind, _dt.date(2026, 11, 11), "premarket", False)
    assert unmarked["text"].startswith("If you placed it, it probably filled")
    # still working: today's order before the close, or today's order on a partial bar
    assert SC.order_fill_check({**a, "valid_for": "2026-11-10"}, ind, _dt.date(2026, 11, 10), "preclose", True) is None
    assert SC.order_fill_check({**a, "valid_for": "2026-11-10"}, ind, _dt.date(2026, 11, 10), "postclose", True) is None
    # no verdict without that day's bar, and none for a buy-near-the-close order (only the brokerage knows)
    assert SC.order_fill_check({**a, "valid_for": "2026-11-11"}, ind, _dt.date(2026, 11, 12), "premarket", False)["filled"] is None
    assert SC.order_fill_check({**a, "order": "Buy near the close today"}, ind, _dt.date(2026, 11, 10), "premarket", False)["filled"] is None
    # the email: a probable fill is an action, an expired order is information
    scan = {"mode": "postclose", "market": {"label": "Bull", "spy": 1, "spy_vs_200d_pct": 1, "spy_5d_pct": 1,
            "breadth_above_50d_pct": 50, "vix": 15}, "funds": {"capital": 5000, "cash": 5000, "open_count": 0},
            "data": {"status": "ok", "notes": []}, "alert_updates": {}, "fill_checks": [hit, {**c, "last_day": "2026-11-10"}],
            "counts": {"orders_to_record": 1}}
    txt = SC.make_digest(scan, [], [], [], [], 0, S.params_with_defaults(None), _dt.datetime(2026, 11, 10, 14, 5))
    assert "ORDERS TO RECORD" in txt and "ORDERS THAT DIDN'T FILL" in txt
    t2 = SC.make_digest({**scan, "fill_checks": [{**touch, "last_day": "2026-11-10"}]}, [], [], [], [], 0,
                        S.params_with_defaults(None), _dt.datetime(2026, 11, 10, 14, 5))
    assert "ORDERS TO RECORD" in t2 and "ORDERS THAT DIDN'T FILL" not in t2
    assert "1 order to check: filled or not?" in SC.headline({"counts": {"orders_to_check": 1}, "data": {"status": "ok"}})
    assert "1 order probably filled: record" in SC.headline({"counts": {"orders_to_record": 1}, "data": {"status": "ok"}})


def test_unfilled_orders_count_as_fill_rate_not_as_skips():
    from engine import learnlog as LL
    A = lambda i, st, **k: {"id": i, "type": "entry", "ticker": i.upper(), "strategy": "R_PCTB_3DAY", "valid_for": "2026-11-09",
                           "status": st, "entry": 100, **k}
    alerts = [A("f", "filled", trade_id="t1"), A("u", "unfilled"), A("p_unf", "placed"), A("p_hit", "placed"),
              A("s", "skipped"), A("cw", "cancelled", withdrawn=True), A("cc", "cancelled")]
    shadow = [{"alert_id": "p_unf", "status": "unfilled"}, {"alert_id": "p_hit", "status": "closed", "fill": 100, "exit": 103},
              {"alert_id": "s", "status": "closed", "fill": 100, "exit": 104}, {"alert_id": "cc", "status": "closed", "fill": 100, "exit": 98}]
    doc = LL.portfolio_stats(alerts, shadow, [], [], "2026-11-10", S.params_with_defaults(None), {"R_PCTB_3DAY": 0.455})
    o = doc["orders"]
    assert (o["placed"], o["filled"], o["unfilled"], o["cancelled"]) == (4, 1, 2, 2)  # p_unf counts as unfilled; p_hit unresolved
    assert o["fill_rate"] == 0.25 and o["expected_fill_rate"] == 0.455
    assert doc["skipped"]["n"] == 2  # your skip and the order you cancelled by choice; never an unfilled order
    assert o["probably_filled_unrecorded"] == [{"ticker": "P_HIT", "day": "2026-11-09", "limit": 100}]
    assert any("Probably filled but not recorded" in x for x in doc["lessons"])


def test_run_problems_lead_the_email_and_the_headline():
    """Anything that went wrong in a run is the first thing the email says, never a footnote."""
    import datetime as _dt
    from engine import scan as SC
    scan = {"mode": "postclose", "market": {"label": "Bull", "spy": 1, "spy_vs_200d_pct": 1, "spy_5d_pct": 1,
            "breadth_above_50d_pct": 50, "vix": 15}, "funds": {"capital": 5000, "cash": 5000, "open_count": 0},
            "data": {"status": "ok", "notes": []}, "alert_updates": {}, "counts": {},
            "engine": {"fingerprint": "abc", "source": "dashboard copy"},
            "run_issues": ["The engine files could not be read on the first try; the second try worked."]}
    near = [{"ticker": "OKTA", "strategy_label": "X", "score": 81.7, "win_pct": 74.2, "model_win_pct": 71.9,
             "entry": 1, "stop": 1, "target": 1}]
    P = S.params_with_defaults(None)
    P["grading"] = {**P["grading"], "aplus_score": 80, "aplus_pwin": 0.72}
    txt = SC.make_digest(scan, [], [], near, [], 0, P, _dt.datetime(2026, 11, 10, 14, 5))
    assert txt.splitlines()[1].startswith("⚠ RUN PROBLEM: The engine files")
    assert SC.headline(scan).startswith("⚠ 1 run problem · ")
    assert "couldn't be attached" not in txt and "Scanner version abc" in txt
    # a quiet night says why, and the watchlist says what each name is short of
    assert "No A+ setups tonight (A+ needs score 80+" in txt and "closest tonight was OKTA at score 81.7" in txt
    assert "not A+: model win 71.9% (needs" in txt


def test_weekly_minimum_is_five_and_quiet_scans_still_report():
    from engine import scan as SC
    assert SC.week_min({}) == 5 and SC.week_min({"min_setups_week": 7}) == 7 and SC.week_min(None) == 5
    quiet = {"mode": "premarket", "counts": {"open_positions": 2}, "data": {"status": "ok"}, "market": {"label": "Bull"}}
    h = SC.email_headline(quiet)
    assert h.startswith("No action needed (pre-market check done: 2 open positions checked, keep holding)") and h.endswith("market Bull")
    assert "no new A+ setups tonight" in SC.email_headline({**quiet, "mode": "postclose"})
    busy = SC.email_headline({**quiet, "counts": {"open_positions": 2, "exit_alerts": 1}})
    assert busy.startswith("1 exit alert") and "done:" not in busy


def test_grades_are_recorded_apart_per_strategy_and_their_records_set_the_rules():
    import datetime as dt
    from engine import learnlog as LL
    now = dt.datetime(2026, 11, 20, 15, 0)
    alerts, shadow = [], []
    for i in range(20):
        top = i < 14
        aid = f"e{i}"
        st_ = "S" if i % 2 == 0 else "T"
        alerts.append({"id": aid, "valid_for": "2026-11-1%d" % (i % 9), "strategy": st_, "grade": "A" if top else "A+",
                       "grade_kind": "topup" if top else "aplus", **({"topup_note": "Weekly top-up"} if top else {})})
        win = (i % 4 != 1) if top else True   # top-ups: S always wins, T wins 3 of 7; A+ 6 of 6
        shadow.append({"alert_id": aid, "status": "closed", "fill": 100.0, "exit": 102.0 if win else 97.0, "strategy": st_,
                       "reason": "target" if win else "stop"})
    trades = [{"alert_id": "e0", "entry_price": 100, "exit_price": 103, "exit_date": "2026-11-12", "strategy": "S",
               "grade_kind": "topup"}]
    st = LL.stats_doc(None, alerts, shadow, trades, True, now)
    gs = st["grade_summary"]
    assert gs["topup"]["n"] == 14 and gs["topup"]["wins"] == 10 and gs["aplus"]["n"] == 6 and gs["aplus"]["win_rate"] == 1.0
    assert gs["topup"]["live_n"] == 1 and gs["aplus"]["live_n"] == 0
    assert gs["topup"]["by_strategy"]["S"]["wins"] == 7 and gs["topup"]["by_strategy"]["T"]["n"] == 7
    assert st["grade_months"]["2026-11"]["topup"]["T"]["exits"] == {"target": 3, "stop": 4}
    G = {"aplus_pwin": 0.72, "oos_aplus_win_rate": 0.76, "oos_topup_win_rate": 0.735}
    assert LL.topup_policy(st, G)["mode"] == "normal"          # 10 of 14 + a real win (counted double): in line with the test
    # one strategy's own top-ups trail badly: only that strategy stops sending top-ups
    bad = [dict(s, exit=97.0) if a["strategy"] == "T" and a["grade_kind"] == "topup" else s for s, a in zip(shadow, alerts)]
    for k in range(6):  # more T top-ups that lost, so its record carries weight
        alerts.append({"id": f"x{k}", "valid_for": "2026-11-05", "strategy": "T", "grade": "A", "grade_kind": "topup"})
        bad.append({"alert_id": f"x{k}", "status": "closed", "fill": 100.0, "exit": 97.0, "strategy": "T"})
    st_bad = LL.stats_doc(None, alerts, bad, [], True, now)
    ex = LL.topup_strategy_exclusions(st_bad, G)
    assert set(ex) == {"T"} and "A+" in ex["T"]
    assert LL.topup_policy(st_bad, G)["mode"] in ("tight", "off")
    # all top-ups losing: off; clearly under the test: tight; too few results: normal
    allbad = [dict(s, exit=97.0) if a.get("grade_kind") == "topup" else s for s, a in zip(bad, alerts)]
    assert LL.topup_policy(LL.stats_doc(None, alerts, allbad, [], True, now), G)["mode"] == "off"
    few = LL.stats_doc(None, alerts[:4], shadow[:4], [], True, now)
    assert LL.topup_policy(few, G)["mode"] == "normal" and LL.topup_strategy_exclusions(few, G) == {}
    # A+ on watch only with enough results well under its test
    assert LL.aplus_watch(st, G) is None
    many = [{"id": f"a{k}", "valid_for": "2026-11-0%d" % (k % 9 + 1), "strategy": "S", "grade": "A+", "grade_kind": "aplus"} for k in range(30)]
    msh = [{"alert_id": f"a{k}", "status": "closed", "fill": 100.0, "exit": 102.0 if k % 2 else 97.0, "strategy": "S"} for k in range(30)]
    aw = LL.aplus_watch(LL.stats_doc(None, many, msh, [], True, now), G)
    assert aw and aw["record"] == 0.5 and "stricter A+ bar" in aw["note"]
    # the record survives when old alerts fall out of the export
    later = LL.stats_doc(st, [dict(alerts[0], valid_for="2026-12-02")], [], [], False, dt.datetime(2026, 12, 5))
    assert later["grade_months"]["2026-11"]["topup"]["S"]["shadow_n"] == 7


def test_watchlist_setups_are_graded_on_paper_once_each():
    import datetime as dt
    from engine import learnlog as LL
    now = dt.datetime(2026, 11, 20, 15, 0)
    nm = {"ticker": "W1", "valid_for": "2026-11-10", "strategy": "S", "entry": 10.0, "stop": 9.0, "target": 11.0, "grade": "A"}
    scans = [{"started_at": "2026-11-09T14:05", "near_misses": [nm, dict(nm, ticker="SENT")]},
             {"started_at": "2026-11-10T05:50", "near_misses": [nm]}]
    sent = [{"id": "e1", "ticker": "SENT", "valid_for": "2026-11-10"}]
    w = LL.watch_from_scans(scans, sent)
    assert [x["ticker"] for x in w] == ["W1"] and w[0]["grade_kind"] == "watch"
    sh = [{"alert_id": w[0]["id"], "status": "closed", "fill": 10.0, "exit": 11.0, "reason": "target"}]
    st1 = LL.stats_doc(None, [], [], [], True, now, w, sh)
    assert st1["grade_summary"]["watch"]["n"] == 1 and st1["grade_summary"]["watch"]["wins"] == 1
    st2 = LL.stats_doc(st1, [], [], [], True, now + dt.timedelta(days=1), w, sh)   # graded again next day: counted once
    assert st2["grade_summary"]["watch"]["n"] == 1
    st3 = LL.stats_doc(st2, [], [], [], True, now + dt.timedelta(days=120), [], [])  # ids age out, the record stays
    assert st3["grade_summary"]["watch"]["n"] == 1 and not st3["watch_ids"]


def test_topup_replay_follows_the_weekly_pace_and_is_tuned_on_its_own():
    import pandas as pd
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2022-01-03", "2026-10-30")
    rows = []
    for d in days:
        for k in range(6):
            strat = "BAD" if k == 5 else "GOOD"
            sc = float(rng.uniform(60, 94.9)) if strat == "BAD" else float(rng.uniform(40, 99))
            p = float(rng.uniform(0.6, 0.8))
            win = rng.random() < (0.35 if strat == "BAD" else 0.75)
            rows.append({"signal_date": d, "ticker": f"T{k}", "strategy": strat, "score": sc, "p": p, "filled": True,
                         "ret": float(rng.uniform(0.01, 0.04)) if win else -float(rng.uniform(0.01, 0.03))})
    sig = pd.DataFrame(rows)
    s = CB._one_per_ticker(sig)
    # nothing is A+: every weekday adds exactly the setups the pace asks for (1 Monday ... 5 Friday, capped by supply)
    pk, wk = CB.simulate_topups(s, 999, 9, 0, 0, 5)
    per_day = pk.groupby("signal_date").size()
    assert per_day.max() == 1 and float((wk == 5).mean()) > 0.9
    # A+ setups count first: a Monday with two A+ needs no top-up that day
    mon = days[days.weekday == 0][3]
    s2 = s.copy()
    s2.loc[(s2["signal_date"] == mon) & (s2["ticker"].isin(["T0", "T1"])), ["score", "p"]] = [99.5, 0.95]
    pk2, _ = CB.simulate_topups(s2, 99, 0.9, 0, 0, 5)
    assert not (pk2["signal_date"] == mon).any()
    # excluded strategies never top up
    pk3, _ = CB.simulate_topups(s, 999, 9, 0, 0, 5, excluded=["BAD"])
    assert not (pk3["strategy"] == "BAD").any()
    res = CB.calibrate_topups(sig[sig["signal_date"] >= "2022-01-01"], 95, 0.78, 5)
    assert res["topup_ok"] and res["topup_floor"] < 95 and "BAD" in res["topup_excluded"]
    assert res["oos_topup_trades"] >= CB.TOPUP_MIN_TRADES and res["oos_weeks_at_min"] >= res["oos_weeks_at_min_aplus_only"]
    assert {r["year"] for r in res["topup_by_year"]} >= {2022, 2026} and "chosen_rule_late" in res["topup_time_split"]
    # floors nobody can clear: top-ups off, said plainly
    awful = sig.assign(ret=-0.01)
    off = CB.calibrate_topups(awful, 95, 0.78, 5)
    assert off["topup_ok"] is False and "off" in off["topup_note"]


def test_market_conditions_are_learned_validated_and_used():
    import datetime as dt
    import pandas as pd
    from engine import learnlog as LL, strategies as S
    # one definition everywhere
    assert S.condition_of(1, 1.0, 0.01) == "bull_calm" and S.condition_of(1, 1.2, 0.0) == "bull_stress"
    assert S.condition_of(0, 1.0, 0.0) == "bear_calm" and S.condition_of(0, 1.0, -0.03) == "bear_stress"
    assert S.condition_of(None, 1, 0) is None
    # every graded result and trade is tagged with its condition, per strategy and per grade, and kept
    now = dt.datetime(2026, 11, 20, 15, 0)
    calm = {"spy_above200": 1, "vix_ratio": 1.0, "spy_ret5": 0.01}
    sell = {"spy_above200": 0, "vix_ratio": 1.5, "spy_ret5": -0.04}
    alerts, shadow = [], []
    for i in range(30):
        f = calm if i < 20 else sell
        alerts.append({"id": f"e{i}", "valid_for": "2026-11-0%d" % (i % 9 + 1), "strategy": "S", "grade": "A+",
                       "grade_kind": "aplus", "features": f})
        win = (i >= 20) or (i % 4 == 0)          # calm: 5 of 20 win; selloff: 10 of 10
        shadow.append({"alert_id": f"e{i}", "status": "closed", "fill": 100.0, "exit": 103.0 if win else 97.0, "strategy": "S"})
    trades = [{"alert_id": "e0", "entry_price": 100, "exit_price": 104, "exit_date": "2026-11-12", "strategy": "S",
               "grade_kind": "aplus", "market_condition": "bull_calm"}]
    st = LL.stats_doc(None, alerts, shadow, trades, True, now)
    bc = st["summary"]["S"]["by_condition"]
    assert bc["bull_calm"]["n"] == 20 and bc["bull_calm"]["wins"] == 5 and bc["bull_calm"]["live_n"] == 1
    assert bc["bear_stress"]["wins"] == 10 and st["grade_summary"]["aplus"]["by_condition"]["bear_stress"]["n"] == 10
    # a clearly failing record in one condition pauses the strategy there only
    conds = {"table": {"S": {"bull_calm": {"oos": {"win_rate": 0.70}}, "bear_stress": {"oos": {"win_rate": 0.75}}}}}
    gl = LL.condition_gates_live(st, conds)
    assert set(gl) == {"S"} and set(gl["S"]) == {"bull_calm"} and "calm bull market" in gl["S"]["bull_calm"]
    few = LL.stats_doc(None, alerts[:5], shadow[:5], [], True, now)
    assert LL.condition_gates_live(few, conds) == {}


def test_condition_pauses_are_adopted_only_when_unseen_alerts_improve():
    import pandas as pd
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2012-01-03", "2026-10-30")
    rows = []
    for d in days:
        bull_calm = (d.dayofyear % 3) != 0
        f = {"spy_above200": 1.0, "vix_ratio": 1.0 if bull_calm else 1.4, "spy_ret5": 0.01 if bull_calm else -0.03}
        for k in range(4):
            strat = "WEAK_IN_CALM" if k == 0 else "STEADY"
            p_win = 0.40 if (strat == "WEAK_IN_CALM" and bull_calm) else 0.74
            sc = float(rng.uniform(80, 99)) if strat == "WEAK_IN_CALM" else float(rng.uniform(60, 99))
            rows.append({"signal_date": d, "ticker": f"T{k}", "strategy": strat, "score": sc, "p": 0.75, "filled": True,
                         "ret": 0.02 if rng.random() < p_win else -0.02, **f})
    t = pd.DataFrame(rows)
    G = {"aplus_score": 85, "aplus_pwin": 0.7, "topup_ok": True, "topup_floor": 70, "topup_pwin": 0.7, "topup_excluded": {}}
    cr = CB.condition_report(t, "2022-01-01", G, 5)
    assert cr["table"]["WEAK_IN_CALM"]["bull_calm"]["oos"]["win_rate"] < 0.5
    assert cr["gates"] == {"WEAK_IN_CALM": {"bull_calm": cr["gates"]["WEAK_IN_CALM"]["bull_calm"]}}
    assert any(g["adopted"] for g in cr["gates_tested"]) and cr["sent_oos"]["win_rate"] > 0.7
    assert cr["best"]["bull_calm"][0] == "STEADY" and abs(sum(cr["share_of_days"].values()) - 1) < 0.01
    # a strategy that is fine everywhere is never paused
    t2 = t.assign(ret=np.where(rng.random(len(t)) < 0.74, 0.02, -0.02))
    assert CB.condition_report(t2, "2022-01-01", G, 5)["gates"] == {}


def test_all_weather_strategies_trade_everywhere_and_need_more_evidence_to_pause():
    from engine import learnlog as LL
    good = lambda n, w, pf: {"trades": n, "win_rate": w, "profit_factor": pf}
    every = {c: {"train": good(300, 0.70, 1.5), "oos": good(120, 0.68 + 0.02 * i, 1.4)} for i, c in enumerate(S.CONDITIONS)}
    v = S.all_weather(every, {"bull_calm": 0.66, "bull_stress": 0.17, "bear_calm": 0.09, "bear_stress": 0.08})
    assert v["all_weather"] and v["weak"] == [] and v["oos_win_range"] == [0.68, 0.74] and v["cover"] == 1.0
    assert "all-weather" in S.all_weather_sentence(v, "Leader 7-Day Pullback")
    # weak in one condition: condition-dependent, and the sentence names where
    dep = {**every, "bear_stress": {"train": good(300, 0.55, 0.9), "oos": good(120, 0.50, 0.8)}}
    d = S.all_weather(dep)
    assert not d["all_weather"] and d["weak"] == ["bear_stress"] and "bear-market selloffs" in S.all_weather_sentence(d)
    # too few conditions tested is not enough to call it all-weather
    two = {c: every[c] for c in ("bull_calm", "bull_stress")}
    assert not S.all_weather(two)["all_weather"] and S.all_weather(two)["weak"] == []
    # tested conditions must cover most trading days when the shares are known
    three = {c: every[c] for c in ("bull_stress", "bear_calm", "bear_stress")}
    assert S.all_weather(three)["all_weather"]
    assert not S.all_weather(three, {"bull_calm": 0.66, "bull_stress": 0.17, "bear_calm": 0.09, "bear_stress": 0.08})["all_weather"]

    # the weekly study marks them and never proposes a pause for them on old data
    rng = np.random.default_rng(11)
    days = pd.bdate_range("2010-01-04", "2026-10-30")
    regimes = [{"spy_above200": 1.0, "vix_ratio": 1.0, "spy_ret5": 0.01}, {"spy_above200": 1.0, "vix_ratio": 1.4, "spy_ret5": -0.03},
               {"spy_above200": 0.0, "vix_ratio": 1.0, "spy_ret5": 0.0}, {"spy_above200": 0.0, "vix_ratio": 1.5, "spy_ret5": -0.04}]
    rows = []
    for i, d0 in enumerate(days):
        f = regimes[i % 4]
        cond = S.condition_of(f["spy_above200"], f["vix_ratio"], f["spy_ret5"])
        for k, strat in enumerate(("ALL_WX", "SELLOFF_LOSER")):
            p_win = 0.45 if (strat == "SELLOFF_LOSER" and cond == "bear_stress") else 0.72
            rows.append({"signal_date": d0, "ticker": f"T{k}", "strategy": strat, "score": float(rng.uniform(60, 99)),
                         "p": 0.75, "filled": True, "ret": 0.02 if rng.random() < p_win else -0.02, **f})
    t = pd.DataFrame(rows)
    G = {"aplus_score": 85, "aplus_pwin": 0.7, "topup_ok": True, "topup_floor": 70, "topup_pwin": 0.7, "topup_excluded": {}}
    cr = CB.condition_report(t, "2022-01-01", G, 5)
    assert cr["all_weather"]["ALL_WX"]["all_weather"] and not cr["all_weather"]["SELLOFF_LOSER"]["all_weather"]
    assert cr["all_weather"]["SELLOFF_LOSER"]["weak"] == ["bear_stress"]
    assert {g["strategy"] for g in cr["gates_tested"]} == {"SELLOFF_LOSER"}
    line = CB._all_weather_line(cr, {"strategies": {"ALL_WX": {"enabled": True}, "SELLOFF_LOSER": {"enabled": True}}})
    assert "All-weather" in line and "ALL_WX" in line and "SELLOFF_LOSER (weaker in bear-market selloffs: 2008-2021 " in line

    # live pauses: a short bad streak pauses an ordinary strategy in that condition, not an all-weather one;
    # a long, clearly bad record pauses either
    def stats(n, wr):
        return {"summary": {"X": {"by_condition": {"bull_calm": {"weighted_n": n, "weighted_win_rate": wr}}}}}
    conds = {"table": {"X": {"bull_calm": {"oos": {"win_rate": 0.72}}}}}
    assert set(LL.condition_gates_live(stats(15, 0.45), conds)) == {"X"}
    aw = {**conds, "all_weather": {"X": {"all_weather": True}}}
    assert LL.condition_gates_live(stats(15, 0.45), aw) == {}
    long_bad = LL.condition_gates_live(stats(30, 0.40), aw)
    assert "all-weather in testing" in long_bad["X"]["bull_calm"]
    # the live record can prove a strategy all-weather too
    live = LL.all_weather_live({"summary": {"X": {"by_condition": {
        c: {"weighted_n": 14, "weighted_win_rate": 0.7} for c in ("bull_calm", "bull_stress", "bear_stress")}}}})
    assert live["X"]["all_weather"]


def test_research_never_narrows_an_all_weather_idea_into_a_specialist():
    good = lambda n, w, pf, a=0.004: {"trades": n, "win_rate": w, "profit_factor": pf, "avg_ret": a}
    # clears the specialist bars in every condition it has data in: that is all-weather, not a specialist
    bc = {c: {"train": good(400, 0.70, 1.5), "oos": good(200, 0.70, 1.5)} for c in S.CONDITIONS}
    assert RS.specialist_conditions(bc) == list(S.CONDITIONS) and S.all_weather(bc)["all_weather"]
    # a real specialist: strong in one condition, poor elsewhere
    sp = {c: {"train": good(400, 0.55, 1.0), "oos": good(200, 0.55, 0.9)} for c in S.CONDITIONS}
    sp["bear_calm"] = {"train": good(400, 0.72, 1.6), "oos": good(200, 0.73, 1.4)}
    assert RS.specialist_conditions(sp) == ["bear_calm"] and not S.all_weather(sp)["all_weather"]
    src = pathlib.Path(RS.__file__).read_text()
    assert "not aw[\"all_weather\"]" in src and "set(sc_) >= set(aw[\"tested\"])" in src


def test_a_plus_bar_is_chosen_by_account_growth_among_near_best_bars():
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2022-01-03", "2025-12-31")
    rows = []
    for d in days:
        for k in range(6):
            sc = float(rng.uniform(50, 99))
            band_bad = 80 <= sc < 86            # setups scored 80-86 are weak in this world
            win = rng.random() < (0.45 if band_bad else 0.78)
            ret = 0.03 if win else -0.03
            rows.append({"signal_date": d, "ticker": f"T{k}{d.dayofyear % 7}", "strategy": "S", "score": sc, "p": 0.75,
                         "filled": True, "ret": ret, "fill": 100.0, "stop": 94.0, "exit_date": d + pd.Timedelta(days=5)})
    post = pd.DataFrame(rows)
    n80 = int((post["score"] >= 80).sum()); n86 = int((post["score"] >= 86).sum())
    c80, c86 = (0.70, 0.010, 80, 0.62, 10.0, n80, 0.6), (0.78, 0.018, 86, 0.62, 6.0, n86, 0.4)
    pick, table = CB._growth_pick(post, [c80, c86], c80, 5, {"max_open": 5, "risk_pct": 2.0, "max_position_pct": 25.0})
    assert pick == c86 and sum(r["chosen"] for r in table) == 1
    assert {r["aplus_score"] for r in table} == {80, 86}
    best = next(r for r in table if r["chosen"]); other = next(r for r in table if not r["chosen"])
    assert best["account_cagr"] > other["account_cagr"] and "topup_rule" in best
    # without the columns a portfolio needs, the growth check steps aside (the coverage rule decides)
    assert CB._growth_pick(post.drop(columns=["stop"]), [c80, c86], c80, 5, None) == (None, [])


def _desk_world(seed=11, extra=None):
    """Synthetic 2022-2025 signals: strategy S sends A+ setups (score 90) on some days and top-up-band setups
    (score 60) on others; `extra` adds a second strategy's rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2022-01-03", "2025-12-31"):
        if rng.random() < 0.35:
            win = rng.random() < 0.78
            rows.append({"signal_date": d, "ticker": f"A{d.dayofyear % 9}", "strategy": "S", "score": 90.0, "p": 0.70,
                         "filled": True, "ret": 0.03 if win else -0.03, "fill": 100.0, "stop": 94.0,
                         "exit_date": d + pd.Timedelta(days=4)})
        if rng.random() < 0.25:
            win = rng.random() < 0.66
            rows.append({"signal_date": d, "ticker": f"B{d.dayofyear % 9}", "strategy": "S", "score": 60.0, "p": 0.58,
                         "filled": True, "ret": 0.02 if win else -0.025, "fill": 100.0, "stop": 94.0,
                         "exit_date": d + pd.Timedelta(days=4)})
        if extra:
            r_ = extra(d, rng)
            if r_:
                rows.append(r_)
    return pd.DataFrame(rows)


G_TEST = {"aplus_score": 86, "aplus_pwin": 0.62, "topup_floor": 56, "topup_pwin": 0.54, "topup_ok": True}
ACCT = {"max_open": 5, "risk_pct": 2.0, "max_position_pct": 25.0}


def test_daily_promotion_needs_win_rates_to_hold_and_the_account_to_grow():
    base = CB.desk_replay(_desk_world(), G_TEST, 5, ACCT)
    assert base["aplus"]["trades"] > 100 and base["topups"]["trades"] > 50 and 0 < base["weeks_at_min"] < 1

    def good(d, rng):  # a frequent source of strong top-up-band setups for the quiet days
        if rng.random() < 0.6:
            w = rng.random() < 0.80
            return {"signal_date": d, "ticker": f"G{d.dayofyear % 9}", "strategy": "NEW", "score": 62.0, "p": 0.60,
                    "filled": True, "ret": 0.025 if w else -0.025, "fill": 100.0, "stop": 94.0, "exit_date": d + pd.Timedelta(days=3)}

    def bad(d, rng):  # just as frequent, but its setups mostly lose: fills the weeks and dilutes the win rate
        if rng.random() < 0.6:
            w = rng.random() < 0.45
            return {"signal_date": d, "ticker": f"G{d.dayofyear % 9}", "strategy": "NEW", "score": 62.0, "p": 0.60,
                    "filled": True, "ret": 0.025 if w else -0.025, "fill": 100.0, "stop": 94.0, "exit_date": d + pd.Timedelta(days=3)}
    ok, checks = CB.promote_gate(base, CB.desk_replay(_desk_world(extra=good), G_TEST, 5, ACCT), "NEW")
    assert ok, [c for c in checks if not c["ok"]]
    ok_b, checks_b = CB.promote_gate(base, CB.desk_replay(_desk_world(extra=bad), G_TEST, 5, ACCT), "NEW")
    failed = {c["key"] for c in checks_b if not c["ok"]}
    assert not ok_b and {"topup_win", "own"} <= failed
    # a strategy the scanner never tops up from adds nothing below the A+ bar
    off = CB.desk_replay(_desk_world(extra=good), {**G_TEST, "topup_excluded": {"NEW": "test"}}, 5, ACCT)
    assert "NEW" not in off["by_strategy"]
    # with top-ups off, only A+ setups are sent
    aplus_only = CB.desk_replay(_desk_world(extra=good), {**G_TEST, "topup_ok": False}, 5, ACCT)
    assert aplus_only["topups"]["trades"] == 0 and aplus_only["sent"]["trades"] == aplus_only["aplus"]["trades"]


def test_graded_ideas_are_judged_on_the_setups_the_desk_would_send():
    # the lowest bar the desk sends at follows the live top-up state
    assert RS.graded_bars(G_TEST) == (56.0, 0.54, "topup")
    assert RS.graded_bars({**G_TEST, "topup_ok": False})[2] == "aplus"
    assert RS.graded_bars({**G_TEST, "topup_mode": "off"})[2] == "aplus"
    # every graded bar is in the checks, with the raw rules that must still hold
    keys = {"g_oos_n", "g_oos_win", "g_oos_pf", "g_oos_avg", "g_halves", "g_train"}
    src = pathlib.Path(RS.__file__).read_text()
    for k in keys | set(RS.GRADED_RAW):
        assert f'"{k}"' in src
    # research: graded comes before the specialist path, and a graded candidate is flagged for the weekly run
    assert "graded_ok = bool(gc and gc.get(\"ok\")) and q_ok[\"overlap\"]" in src
    assert "not graded_ok and not aw[\"all_weather\"]" in src
    assert 'sp_["graded"] = True' in src
    # the weekly run lets a graded strategy into the fit on its raw sanity checks, then replays the desk with and
    # without it (both recalibrated) and drops it unless the same promotion gate passes
    csrc = pathlib.Path(CB.__file__).read_text()
    assert 'if graded and not enabled and all(c["ok"] for c in checks if c["key"] in ("train_win", "oos_pf", "oos_decay"))' in csrc
    assert "ok_g, ch_g = promote_gate(rx, rw, n)" in csrc


def test_promotion_writes_the_params_and_the_strategy_card_together(tmp_path, monkeypatch):
    """run_promote end to end on a tiny state: one graded candidate that passes the replay gate."""
    st = tmp_path / "state"
    for c in ("strategy", "research", "research_results", "config"):
        (st / c).mkdir(parents=True)
    P = S.params_with_defaults(None)
    P["grading"].update(G_TEST)
    P["models"] = {"_pooled": {"kind": "pooled"}}
    P["strategies"]["S"] = {"enabled": True}
    (st / "strategy" / "params.json").write_text(json.dumps({"params": P, "changelog_entry": []}, default=str))
    (st / "strategy" / "backtest.json").write_text(json.dumps({"strategies": {"S": {"enabled": True, "label": "S"}}}))
    (st / "config" / "settings.json").write_text(json.dumps({"min_setups_week": 5, "risk_pct": 2, "max_open_positions": 5,
                                                              "max_position_pct": 25}))
    spec = {"id": "R_NEW", "label": "New idea", "family": "Price action", "about": "x", "universe": "stock",
            "conditions": ["close > open"], "entry": {"kind": "limit_next", "price": "close"}, "stop": {"atr": 3},
            "target": {"atr": 1}, "max_hold": 5}
    (st / "research" / "R_NEW.json").write_text(json.dumps({"id": "R_NEW", "spec": spec}))
    (st / "research_results" / "R_NEW-20261112-1300.json").write_text(json.dumps(
        {"id": "R_NEW-20261112-1300", "idea_id": "R_NEW", "verdict": "candidate", "graded_ok": True, "variant": {},
         "tested_at": "2026-11-12T13:00:00-08:00",
         "graded": {"checks": [{"key": "g_oos_win", "ok": True, "label": "x", "need": "y", "value": 0.8}],
                    "periods": {"oos": {"win_rate": 0.8}}}}))

    def good(d, rng):
        if rng.random() < 0.6:
            w = rng.random() < 0.80
            return {"signal_date": d, "ticker": f"G{d.dayofyear % 9}", "strategy": "R_NEW", "score": 62.0, "p": 0.60,
                    "filled": True, "ret": 0.025 if w else -0.025, "fill": 100.0, "stop": 94.0, "exit_date": d + pd.Timedelta(days=3)}
    extra = {"reason": "target", "held": 3, "r_mult": 0.5, "limit": 100.0, **{f: 0.0 for f in S.FEATURES}}
    world = _desk_world().assign(**extra)
    new_rows = _desk_world(extra=good)
    new_rows = new_rows[new_rows["strategy"] == "R_NEW"].assign(**extra)
    monkeypatch.setattr(CB.D, "load_history", lambda d: {"close": pd.DataFrame()})
    monkeypatch.setattr(CB.D, "load_universe", lambda d: {})
    monkeypatch.setattr(CB.I, "compute", lambda w, u: {})
    monkeypatch.setattr(CB.S, "all_plans", lambda ind, P_: {"S": {"name": "S"}})
    monkeypatch.setattr(CB.S, "build", lambda name, ind, P_: {"name": name})
    monkeypatch.setattr(CB.B, "simulate", lambda plan, ind, P_, include_unfilled=True: world if plan["name"] == "S" else new_rows)
    monkeypatch.setattr(CB, "_score_live", lambda t, pm, P_: t)  # the synthetic rows carry their own score and p
    rep = CB.run_promote("data", st, tmp_path / "out", now=pd.Timestamp("2026-11-12T13:00:00-08:00"))
    assert [p["id"] for p in rep["promoted"]] == ["R_NEW"], rep.get("note")
    doc = json.loads((tmp_path / "out" / "doc_params.json").read_text())
    assert doc["params"]["strategies"]["R_NEW"]["enabled"] and doc["params"]["custom"]["R_NEW"]["spec"]["graded"]
    assert doc["params"]["strategies"]["R_NEW"]["promoted"]["by"] == "daily promotion check"
    assert "Promoted New idea mid-week" in doc["changelog_entry"][0]
    bt = json.loads((tmp_path / "out" / "doc_backtest.json").read_text())
    assert bt["strategies"]["R_NEW"]["enabled"] and "daily promotion check" in bt["strategies"]["R_NEW"]["status_reason"]
    assert "S" in bt["strategies"] and bt["promotions"][0]["ids"] == ["R_NEW"]  # the rest of the document is kept
    # a strategy already live is never promoted again, and nothing is written when no candidate waits
    (st / "strategy" / "params.json").write_text(json.dumps(doc, default=str))
    rep2 = CB.run_promote("data", st, tmp_path / "out2", now=pd.Timestamp("2026-11-13T13:00:00-08:00"))
    assert rep2["promoted"] == [] and not (tmp_path / "out2" / "doc_params.json").exists()


def test_close_entry_setups_never_count_as_weekly_topups():
    """The pre-close scan buys at the close and never tops up; the replays must not either."""
    def etf(d, rng):
        if rng.random() < 0.9:
            return {"signal_date": d, "ticker": f"E{d.dayofyear % 5}", "strategy": "ETFC", "score": 70.0, "p": 0.60,
                    "filled": True, "ret": 0.01, "fill": 100.0, "stop": 97.0, "exit_date": d + pd.Timedelta(days=2)}
    w = _desk_world(extra=etf)
    P_ = {"strategies": {"S": {"entry_kind": "limit_next"}, "ETFC": {"entry_kind": "close"}}}
    with_flag = CB.desk_replay(CB.mark_entry(w, P_), G_TEST, 5, ACCT)
    assert "ETFC" not in with_flag["by_strategy"]
    assert "ETFC" in CB.desk_replay(w, G_TEST, 5, ACCT)["by_strategy"]  # what the old replay wrongly counted
    src = pathlib.Path(CB.__file__).read_text()
    assert 'entry_close=(P["strategies"][name].get("entry_kind") == "close")' in src and "t = mark_entry(t, P)" in src


def test_the_weekly_run_always_retests_every_promoted_strategy_and_new_candidates(tmp_path):
    """Nov 12, 2026: a cap of six covered promoted strategies and new candidates together, so with six promoted no
    candidate was ever tested on Saturday, and a seventh promoted strategy would have silently left the params."""
    st = tmp_path / "state"
    (st / "research").mkdir(parents=True)
    (st / "research_results").mkdir()
    spec = lambda i: {"id": i, "label": i, "family": "x", "about": "x", "conditions": ["close > open"],
                      "entry": {"kind": "limit_next", "price": "close"}, "stop": {"atr": 3}, "target": {"atr": 1}}
    old = {"custom": {f"R_P{k}": {"spec": spec(f"R_P{k}")} for k in range(7)}}
    for k, w in enumerate((0.70, 0.66, 0.72)):
        i = f"R_C{k}"
        (st / "research" / f"{i}.json").write_text(json.dumps({"id": i, "spec": spec(i)}))
        (st / "research_results" / f"{i}-1.json").write_text(json.dumps(
            {"id": f"{i}-1", "idea_id": i, "verdict": "candidate", "tested_at": "2026-11-12T13:00:00-08:00",
             "oos": {"win_rate": w}, **({"graded_ok": True, "graded": {"periods": {"oos": {"win_rate": w}}}} if k == 2 else {})}))
    got = RS.candidate_specs(st, old)
    assert set(old["custom"]) <= set(got)                     # all seven promoted, none dropped
    assert {"R_C0", "R_C1", "R_C2"} <= set(got)                # and the new candidates still get their test
    assert got["R_C2"].get("graded") is True                    # a graded candidate is tested as graded
    assert len(RS.candidate_specs(st, old, cap=1)) == 8 and "R_C2" in RS.candidate_specs(st, old, cap=1)


def test_a_recalibration_replaces_the_live_setup_only_if_it_does_at_least_as_well():
    """Nov 12, 2026: a dry run of Saturday's job would have cut the weeks reaching 5 setups from 78% to 32% and
    growth from 25% to 19% a year. The recalibrated setup is now adopted only if a replay on the same signals keeps
    the win rates, coverage, growth and drawdown of the live one."""
    live = {"aplus": {"win_rate": 0.767}, "sent": {"win_rate": 0.744}, "sent_late": {"win_rate": 0.817},
            "weeks_at_min": 0.779, "account": {"cagr": 0.252, "max_dd": -0.086}}
    worse = {"aplus": {"win_rate": 0.755}, "sent": {"win_rate": 0.757}, "sent_late": {"win_rate": 0.82},
             "weeks_at_min": 0.316, "account": {"cagr": 0.194, "max_dd": -0.077}}
    ok, checks = CB.adoption_gate(live, worse)
    assert not ok and {c["key"] for c in checks if not c["ok"]} == {"aplus_win", "weeks", "growth"}
    same = {**live, "account": {"cagr": 0.250, "max_dd": -0.09}}  # within the slack on every line
    assert CB.adoption_gate(live, same)[0]
    src = pathlib.Path(CB.__file__).read_text()
    assert "ok_a, checks_a = adoption_gate(champ, chall)" in src and 'P_old["source"] = "backtest (live setup kept)"' in src
