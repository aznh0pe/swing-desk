"""A buy-near-the-close setup is not withdrawn only because the price lifted above its limit.

Nov 18, 2026: the 7:20 AM early read sent FAS (ETF Close-at-Low Snapback, A+) with a limit at $161.42. At 12:26 PM
FAS traded higher, off its low, so the pre-close scan withdrew it ("no longer qualifies"). It sold off into the
close and finished at $161.36, still A+: a working limit order would have filled on a valid setup, and the
withdrawal told the owner to cancel it. The limit already enforces "only while the price is down there", so the
pre-close scan now re-tests such a setup with the price at its limit and keeps it on when it still qualifies.
"""
import datetime as dt
import pathlib

import numpy as np
import pandas as pd
import pytest

from engine import scan as SC

FAS = "e_2026-11-18_FAS_ETF_IBS"
NOW = dt.datetime(2026, 11, 18, 12, 26)


def _wide(px_now, low_now=160.96, high_now=166.27):
    idx = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18"])
    cols = ["FAS", "XLE"]
    close = pd.DataFrame([[170.0, 63.0], [168.0, 62.5], [px_now, 61.3]], index=idx, columns=cols)
    low = pd.DataFrame([[168.0, 62.0], [166.0, 62.0], [low_now, 61.2]], index=idx, columns=cols)
    high = pd.DataFrame([[172.0, 64.0], [170.0, 63.0], [high_now, 63.3]], index=idx, columns=cols)
    return {"open": high.copy(), "high": high, "low": low, "close": close, "volume": close * 0 + 1e6}


def _state(withdrawn_reason=None, status="new", grade="A+", kind="close"):
    a = {"id": FAS, "type": "entry", "status": status, "ticker": "FAS", "strategy": "ETF_IBS", "grade": grade,
         "entry": 161.42, "entry_kind": kind, "valid_for": "2026-11-18", "signal_date": "2026-11-18",
         "scan_id": "20261118-0720-preclose",
         "order": "Buy near the close today: ..." if kind == "close" else "Buy LIMIT $161.42, day order"}
    scans = {}
    if withdrawn_reason:
        scans["s1"] = {"doc": {"id": "s1", "started_at": "2026-11-18T12:00:00-08:00",
                               "alert_updates": {FAS: {"withdrawn": True, "withdrawn_reason": withdrawn_reason}}}}
    return {"alerts": {FAS: {"doc": a, "version": 1}}, "scans": scans, "trades": {}}


@pytest.fixture
def fake(monkeypatch):
    seen = {"compute": 0}

    def compute(w, universe):
        seen["compute"] += 1
        seen["w"] = w
        return {"close": w["close"]}

    def build_entries(ind, P, plans, state, earnings, universe, session_day, mode, partial, errors):
        SC.CONDITION_STATE.clear()
        SC.BELOW_BAR.clear()
        SC.BELOW_BAR["XLE"] = {"changed": True}
        return [{"ticker": "FAS", "strategy": "ETF_IBS", "valid_for": "2026-11-18", "grade": seen.get("grade", "A+"),
                 "entry": float(ind["close"]["FAS"].iloc[-1])}], []

    monkeypatch.setattr(SC.I, "compute", compute)
    monkeypatch.setattr(SC.S, "all_plans", lambda ind, P: {})
    monkeypatch.setattr(SC, "build_entries", build_entries)
    return seen


def test_a_sent_close_setup_above_its_limit_stays_on_when_it_still_qualifies_at_the_limit(fake):
    SC.CONDITION_STATE.clear(); SC.CONDITION_STATE["condition"] = "bull_stress"
    SC.BELOW_BAR.clear(); SC.BELOW_BAR["AAA"] = {"score": 50}
    w = _wide(163.50)
    out = SC.hold_at_limit(w, None, {}, _state(), [], {}, NOW.date(), "preclose", NOW, [])
    assert len(out) == 1 and out[0]["held_at_limit"] and out[0]["price_now"] == 163.5
    note = out[0]["hold_note"]
    assert "$163.50, above your $161.42 limit" in note and "still A+" in note and "never higher" in note
    # re-tested with the price at the limit; the real frames are untouched
    assert float(fake["w"]["close"]["FAS"].iloc[-1]) == 161.42 and float(w["close"]["FAS"].iloc[-1]) == 163.50
    assert float(fake["w"]["low"]["FAS"].iloc[-1]) == 160.96  # the low was already under the limit
    assert float(fake["w"]["close"]["XLE"].iloc[-1]) == 61.3
    # the re-test leaves the real run's bookkeeping as it was
    assert SC.CONDITION_STATE == {"condition": "bull_stress"} and SC.BELOW_BAR == {"AAA": {"score": 50}}


def test_the_low_moves_down_with_the_price_when_the_limit_is_under_the_day_low(fake):
    SC.hold_at_limit(_wide(163.50, low_now=162.0), None, {}, _state(), [], {}, NOW.date(), "preclose", NOW, [])
    assert float(fake["w"]["low"]["FAS"].iloc[-1]) == 161.42


def test_an_a_plus_that_would_not_be_a_plus_at_its_limit_is_withdrawn_as_before(fake):
    fake["grade"] = "A"
    assert SC.hold_at_limit(_wide(163.50), None, {}, _state(), [], {}, NOW.date(), "preclose", NOW, []) == []


def test_at_or_under_its_limit_and_no_longer_a_setup_is_a_real_withdrawal(fake):
    assert SC.hold_at_limit(_wide(161.00), None, {}, _state(), [], {}, NOW.date(), "preclose", NOW, []) == []
    assert fake["compute"] == 0


def test_news_and_pre_market_withdrawals_are_never_re_tested_on_price(fake):
    for why in ("Dropped by the 12:00 pm scan's news check: lawsuit", "Pre-market check: gap down. Cancel."):
        assert SC.hold_at_limit(_wide(163.50), None, {}, _state(why), [], {}, NOW.date(), "preclose", NOW, []) == []
    # an earlier price withdrawal is re-tested, so a later scan can put the setup back on
    out = SC.hold_at_limit(_wide(163.50), None, {}, _state("No longer qualifies in the 12:00 pm scan (price moved)."),
                           [], {}, NOW.date(), "preclose", NOW, [])
    assert len(out) == 1
    assert fake["compute"] == 1


def test_only_todays_unfilled_close_setups_are_re_tested(fake):
    for st in (_state(status="filled"), _state(kind="limit")):
        assert SC.hold_at_limit(_wide(163.50), None, {}, st, [], {}, NOW.date(), "preclose", NOW, []) == []
    live = [{"ticker": "FAS", "strategy": "ETF_IBS", "valid_for": "2026-11-18"}]  # still qualifies on its own
    assert SC.hold_at_limit(_wide(163.50), None, {}, _state(), live, {}, NOW.date(), "preclose", NOW, []) == []
    assert fake["compute"] == 0


def test_the_hold_note_reaches_the_email_the_overlay_and_the_dashboard():
    e = {"ticker": "FAS", "name": "Financials 3x", "strategy_label": "ETF Close-at-Low Snapback", "grade": "A+",
         "score": 95, "win_pct": 78, "order": "Buy near the close today", "stop": 136.88, "entry": 161.42,
         "target": 167.55, "rr": 0.25, "shares": 4, "total_cost": 645.68, "risk_usd": 98.16, "hold_max_days": 7,
         "exit_signal": "x", "valid_for": "2026-11-18", "hold_note": "At 12:26 PM PT FAS was $163.50 ..."}
    assert any(l.startswith("   NOW      At 12:26 PM PT FAS") for l in SC.setup_block(e, 1))
    assert "hold_note" in SC.SCAN_OWNED and "price_now" in SC.SCAN_OWNED
    html = (pathlib.Path(SC.__file__).parent.parent / "dashboard" / "index.html").read_text()
    assert "a.hold_note && !a.withdrawn" in html


def test_the_pre_close_scan_runs_the_re_test_and_a_failure_never_costs_the_scan():
    src = pathlib.Path(SC.__file__).read_text()
    run = src.split("def run(")[1]
    assert "hold_at_limit(w, universe, P, state, entries" in run and "hold-at-limit check failed" in run


def test_the_pre_close_scan_keeps_each_etfs_day_so_far_and_the_post_close_record_uses_it():
    """The feed keeps no earlier snapshots. Nov 18, 2026: FAS, XLE, XLF and XLV met the close-at-low rule on the
    final bar (A+); the 12:26 PM scan had seen none of them qualify, and nothing said how far off they were."""
    from engine import strategies as S
    w = _wide(163.50)
    universe = pd.DataFrame({"type": ["etf_lev", "etf"]}, index=["FAS", "XLE"])
    universe.loc["AAPL"] = "stock"
    w2 = {k: v.assign(AAPL=200.0) for k, v in w.items()}
    snap = SC.close_snapshot(w2, universe, "2026-11-18T15:21-05:00")
    assert snap["day"] == "2026-11-18" and set(snap["rows"]) == {"FAS", "XLE"}      # ETFs only, never stocks
    assert snap["rows"]["FAS"] == [166.27, 160.96, 163.5]
    src = pathlib.Path(SC.__file__).read_text().split("def run(")[1]
    assert 'scan_doc["close_snapshot"] = close_snapshot(w, universe' in src and "close snapshot failed" in src
    # post-close: the sent and the late setups say where they were at the pre-close scan
    P = S.params_with_defaults(None)
    P["models"] = {}
    idx = pd.to_datetime(["2026-11-17", "2026-11-18"])
    close = pd.DataFrame({"FAS": [168.0, 161.36], "XLE": [62.5, 61.29]}, index=idx)
    mask = pd.DataFrame({"FAS": [False, True], "XLE": [False, True]}, index=idx)
    ind = {"close": close, "market": pd.DataFrame({"spy_above200": [1, 1], "vix_ratio": [1.2, 1.2],
                                                     "spy_ret5": [-0.03, -0.04]}, index=idx)}
    plans = {"ETF_IBS": {"entry_kind": "close", "mask": mask}}
    alert = {"doc": {"type": "entry", "ticker": "FAS", "strategy": "ETF_IBS", "grade": "A+", "entry": 161.42,
                     "valid_for": "2026-11-18", "entry_kind": "close"}}
    early = {"id": "20261118-0720-preclose", "mode": "preclose", "started_at": "2026-11-18T07:20:15-08:00",
             "close_snapshot": {"day": "2026-11-18", "asof": "10:11", "rows": {"FAS": [165.0, 160.0, 161.0]}}}
    late = {"id": "20261118-1226-preclose", "mode": "preclose", "started_at": "2026-11-18T12:26:45-08:00",
            "close_snapshot": {**snap, "rows": {**snap["rows"], "XLE": [63.3, 61.8, 62.4]}}}
    state = {"alerts": {"a": alert}, "scans": {s["id"]: {"doc": s} for s in (early, late)}}
    d = SC.close_drift(ind, P, plans, state, [])
    assert d["pre_close_scan"] == "20261118-1226-preclose"                         # the latest look of the day
    assert d["sent"][0]["pre_close_last"] == 163.5 and d["sent"][0]["pre_close_ibs"] == 0.478
    assert d["late"][0]["ticker"] == "XLE" and d["late"][0]["pre_close_last"] == 62.4
    # no pre-close snapshot that day: the record is as before
    d2 = SC.close_drift(ind, P, plans, {"alerts": {"a": alert}}, [])
    assert d2["pre_close_scan"] is None and "pre_close_last" not in d2["sent"][0]
