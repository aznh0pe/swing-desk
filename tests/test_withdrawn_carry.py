"""A withdrawal is never forgotten.

Nov 18, 2026: Monday's 5:24 AM pre-market check withdrew MRNA (news: a guidance change; "cancel today's order").
Each scan loads only the newest 6 scans, so by Tuesday evening the withdrawal had dropped out of view, and the
scans asked "did your MRNA order fill?" for two days. The dashboard loads 40 scans, still knew the setup was
withdrawn, and so offered no button to answer. Every scan now carries open withdrawals forward.
"""
import datetime as dt

from engine import scan as SC

MRNA = "e_2026-11-16_MRNA_LEADER_PULLBACK"
REASON = "Pre-market check: News: guidance change; stock fell 7% on it. Cancel today's order."


def _state(scans, status="new", valid_for="2026-11-16"):
    alerts = {MRNA: {"doc": {"id": MRNA, "type": "entry", "status": status, "ticker": "MRNA", "valid_for": valid_for}, "version": 1}}
    return {"alerts": alerts, "scans": {s["id"]: {"doc": s} for s in scans}}


def _scan(sid, at, ups):
    return {"id": sid, "started_at": at, "alert_updates": ups}


def test_every_scan_carries_an_open_withdrawal_so_the_six_scan_window_never_loses_it():
    mon = _scan("20261116-0524-premarket", "2026-11-16T05:24:47-08:00",
                {MRNA: {"withdrawn": True, "withdrawn_reason": REASON, "rank": None, "ticker": "MRNA"}})
    window = [mon]
    for i, at in enumerate(("2026-11-16T12:26", "2026-11-16T14:07", "2026-11-17T05:22", "2026-11-17T12:26",
                            "2026-11-17T14:07", "2026-11-18T05:23", "2026-11-18T12:26")):
        state = _state(window[-6:])                      # what a scheduled scan loads: the newest 6 scans
        prev = SC.current_overlay(state)
        assert prev[MRNA]["withdrawn"], at               # it still knows, at every step
        ups = SC.carry_withdrawn({}, prev, state, dt.date.fromisoformat(at[:10]))
        assert ups[MRNA]["withdrawn"] and ups[MRNA]["carried"] and ups[MRNA]["withdrawn_reason"] == REASON
        assert ups[MRNA]["withdrawn_at"] == "2026-11-16T05:24:47-08:00"     # when it really happened
        window.append(_scan(f"s{i}", at, ups))
    # the old code: with only the newest 6 scans and nothing carried, Monday's withdrawal is gone by Tuesday evening
    plain = [mon] + [_scan(f"p{i}", f"2026-11-1{6 + i // 3}T0{i}:00", {}) for i in range(6)]
    assert MRNA not in SC.current_overlay(_state(plain[-6:]))


def test_carried_withdrawals_stop_when_answered_or_old_and_never_override_a_fresh_view():
    prev = {MRNA: {"withdrawn": True, "withdrawn_reason": REASON, "ticker": "MRNA", "_at": "2026-11-16T05:24:47-08:00"}}
    wed = dt.date(2026, 11, 18)
    for status in ("cancelled", "unfilled", "skipped", "filled"):                # answered on the dashboard: let go
        assert SC.carry_withdrawn({}, prev, _state([], status=status), wed) == {}
    assert MRNA in SC.carry_withdrawn({}, prev, _state([], status="placed"), wed)  # placed and withdrawn: keep warning
    assert SC.carry_withdrawn({}, prev, _state([], valid_for="2026-11-06"), wed) == {}   # 8 sessions old: done
    fresh = {MRNA: {"withdrawn": False, "rank": 1}}                               # this scan's own view wins
    assert SC.carry_withdrawn(dict(fresh), prev, _state([]), wed) == fresh
    assert SC.carry_withdrawn({}, {MRNA: {"withdrawn": False}}, _state([]), wed) == {}


def test_the_email_lists_only_new_withdrawals():
    doc = {"alert_updates": {MRNA: {"withdrawn": True, "withdrawn_reason": REASON, "ticker": "MRNA", "carried": True},
                             "e_x_FAS": {"withdrawn": True, "withdrawn_reason": "No longer qualifies.", "ticker": "FAS"}},
           "counts": {}, "premarket_checks": [], "run_issues": []}
    head = SC.headline(doc)
    assert "1 withdrawn" in head and "2 withdrawn" not in head
    import inspect
    src = inspect.getsource(SC.make_digest)
    assert 'if v.get("withdrawn") and not v.get("carried")]' in src
    assert "carry_withdrawn(build_overlay(scan_doc, entries, ctx), prev_ov, state, today)" in inspect.getsource(SC.run)
