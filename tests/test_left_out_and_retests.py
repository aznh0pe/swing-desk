"""Nothing is excluded for good.

Nov 18, 2026, the owner: "if you trim or exclude tickers, you have to check back after a certain period, because
those tickers might perform better later. I really hope you factored that in." The audit found the screen re-admits
tickers daily and pauses are re-tested weekly, but two things were never looked at again: the scan's own fixed lines
(price, daily volume, leveraged ETFs), and 20 of 63 research ideas marked "only a different rule can fix it".
"""
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from engine import calibrate as CB
from engine import research as RS
from engine import strategies as S


def _state(tmp_path, rows):
    for iid, verdict, status, tested in rows:
        for coll, doc in (("research", {"id": iid, "spec": {"id": iid, "label": iid}}),
                          ("research_results", {"id": f"{iid}-1", "idea_id": iid, "verdict": verdict, "tested_at": tested,
                                                "path": {"status": status} if status else {}})):
            d = tmp_path / coll
            d.mkdir(exist_ok=True)
            (d / f"{doc['id']}.json").write_text(json.dumps(doc))
    return tmp_path


def test_rejected_ideas_are_retested_in_rotation(tmp_path):
    now = dt.datetime(2026, 11, 21, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=-8)))
    old, fresh = "2026-10-10T15:00:00-07:00", "2026-11-17T15:00:00-08:00"
    rows = [("DATA1", "watch", "data", fresh), ("OVER1", "redundant", "overlap", fresh), ("CAND", "candidate", None, old),
            ("FIX_OLD", "rejected", "fixed", old), ("FIX_NEW", "rejected", "fixed", fresh)]
    rows += [(f"FIX{i:02d}", "rejected", "fixed", f"2026-10-{i + 1:02d}T15:00:00-07:00") for i in range(10)]
    ids = [s["id"] for s in RS.retest_specs(_state(tmp_path, rows), now=now)]
    assert ids[:2] == ["DATA1", "OVER1"]                       # as before: verdicts new data can change, every week
    assert "CAND" not in ids and "FIX_NEW" not in ids          # a candidate is handled elsewhere; a fresh test waits
    stale = ids[2:]
    assert len(stale) == RS.STALE_CAP == 8 and stale[0] == "FIX00"    # oldest first, a few each week
    assert "FIX_OLD" not in stale                              # the 11th oldest: its turn comes next week
    assert RS.STALE_DAYS == 28


def test_quiet_names_are_left_out_and_the_line_is_a_parameter():
    P = S.params_with_defaults(None)
    assert P["filters"]["min_atr_pct"] == 0.015
    assert S.params_with_defaults({"filters": {"min_price": 10.0}})["filters"]["min_atr_pct"] == 0.015   # stored params get it
    src = pathlib.Path(S.__file__).read_text()
    assert 'liquid &= (ind["atr10"] / c) >= float(f["min_atr_pct"])' in src


def test_the_tier_check_rechecks_every_line_the_scan_draws():
    src = pathlib.Path(CB.__file__).read_text()
    assert "left_out = left_out_check(ind, P_all, pm, t, split, log=log)" in src
    assert 'block["ext"]["left_out"] = left_out' in src and 'reason + " " + left_out["text"]' in src
    assert "allow_leveraged=True, min_atr_pct=0.0)" in src            # the relaxed replay drops every line
    assert set(("min_price", "min_dv_stock", "min_dv_etf")) <= set(CB.LEFT_OUT)
    # a failure of the re-check never costs the tier check
    assert 'left_out = {"error": repr(ex)[:200]' in src
