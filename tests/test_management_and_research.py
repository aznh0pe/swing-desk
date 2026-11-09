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
