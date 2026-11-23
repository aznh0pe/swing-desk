"""Owner, Nov 23, 2026: "the ADP alert didn't show up until the post-scan". ADP (buy limit $299.08 for Monday) was one
of ~250 setups on the "also qualified" swap list, never featured and never marked placed. The morning email showed it
only as one "KEEP" line among 250; the post-close scan then put "ADP: did your order fill?" at the top of the email
and the dashboard, as if it had been an order the owner was told to place (29 such questions that day).

A swap-list setup is never an order to record: it gets one summary line in the email and a collapsed table on the
dashboard (with a button, in case it was placed after all). Featured and placed orders are still asked about, and are
kept first so the 30-40 item cap can never push one of them out (Nov 23's pre-close scan dropped ADP that way)."""
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import scan as SC  # noqa: E402
from engine import strategies as S  # noqa: E402


def test_a_setup_is_swap_list_by_the_newest_view_of_it():
    assert SC.is_swap_list({"also_qualified": True}, None)
    assert not SC.is_swap_list({"rank": 1}, None)                                  # featured
    assert not SC.is_swap_list({"also_qualified": True}, {"also_qualified": False})  # promoted by a later scan
    assert SC.is_swap_list({"rank": 3}, {"also_qualified": True})                  # demoted by a later scan


def _fc(tk, swap=False, day="2026-11-23"):
    c = {"alert_id": f"e_{day}_{tk}_X", "id": f"e_{day}_{tk}_X", "ticker": tk, "filled": True, "touched": False,
         "status": "new", "valid_for": day, "last_day": day, "limit": 100.0, "low": 99.0,
         "text": f"If you placed it, it probably filled: the low $99.00 traded below your $100.00 limit."}
    if swap:
        c["swap"] = True
    return c


def test_the_email_asks_only_about_real_orders_and_sums_up_the_swap_list():
    scan = {"mode": "postclose", "market": {"label": "Correction", "spy": 1, "spy_vs_200d_pct": 1, "spy_5d_pct": 1,
            "breadth_above_50d_pct": 50, "vix": 26}, "funds": {"capital": 5000, "cash": 900, "open_count": 7},
            "data": {"status": "ok", "notes": []}, "alert_updates": {},
            "fill_checks": [_fc("SMH")] + [_fc(t, swap=True) for t in ("ADP", "ABT", "AIZ")],
            "counts": {"orders_to_record": 1, "swap_list_fills": 3}}
    txt = SC.make_digest(scan, [], [], [], [], 0, S.params_with_defaults(None), dt.datetime(2026, 11, 23, 14, 6))
    sec = txt.split("ORDERS TO RECORD", 1)[1].split("\n\n", 1)[0]
    assert "- SMH (" in sec and "- ADP (" not in sec and "- ABT (" not in sec
    assert 'Not orders: 3 setups from the "also qualified" swap list' in txt and "ABT, ADP, AIZ" in txt
    only_swap = SC.make_digest({**scan, "fill_checks": [_fc("ADP", swap=True)], "counts": {"swap_list_fills": 1}},
                               [], [], [], [], 0, S.params_with_defaults(None), dt.datetime(2026, 11, 23, 14, 6))
    assert "ORDERS TO RECORD" not in only_swap and "Not orders: 1 setup from" in only_swap
    # the headline counts only real orders
    assert "probably filled" not in SC.headline({"counts": {"orders_to_record": 0, "swap_list_fills": 3}, "data": {"status": "ok"}})


def test_the_scan_marks_swap_list_checks_and_keeps_real_orders_ahead_of_the_cap():
    src = (ROOT / "engine" / "scan.py").read_text()
    assert 'if a.get("status") != "placed" and is_swap_list(a, prev_ov.get(aid)):' in src
    assert 'sorted(fill_checks, key=lambda c: bool(c.get("swap")))[:40]' in src
    assert '"orders_to_record": sum(1 for c in fill_checks if c.get("filled") and not c.get("swap"))' in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_dashboard_never_asks_about_a_swap_list_setup_but_lists_it_apart(tmp_path):
    from tests.test_dashboard_orders import _functions
    alerts = [
        {"id": "A_ADP", "type": "entry", "status": "new", "valid_for": "2026-11-23", "valid_days": 1, "ticker": "ADP",
         "entry": 299.08, "stop": 279.37, "target": 308.94, "also_qualified": True},
        {"id": "A_SMH", "type": "entry", "status": "new", "valid_for": "2026-11-23", "valid_days": 1, "ticker": "SMH",
         "entry": 400.06, "stop": 316.59, "target": 479.96, "rank": 1},
    ]
    fcs = [{"alert_id": "A_ADP", "filled": True, "last_day": "2026-11-23", "low": 298.62, "limit": 299.08, "touched": False},
           {"alert_id": "A_SMH", "filled": True, "last_day": "2026-11-23", "low": 399.0, "limit": 400.06, "touched": False, "swap": False}]
    js = """
const RealDate = Date, NOW = "2026-11-23T14:50:00-08:00";
global.Date = class extends RealDate { constructor(...a) { if (a.length) super(...a); else super(NOW); } };
const esc = (v) => String(v);
const S = { alerts: %s, scans: [{ started_at: "2026-11-23T14:06:14-08:00", fill_checks: %s }] };
%s
console.log(JSON.stringify({ asks: orderResolutions().map((r) => r.a.ticker), swaps: swapListFills().map((r) => r.a.ticker) }));
""" % (json.dumps(alerts), json.dumps(fcs), _functions())
    f = tmp_path / "t.js"
    f.write_text(js)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, env={"TZ": "America/Los_Angeles", "PATH": "/usr/bin:/bin:/opt/node22/bin"})
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"asks": ["SMH"], "swaps": ["ADP"]}
    html = (ROOT / "dashboard" / "index.html").read_text()
    assert "Swap-list setups that traded through their limit" in html and 'data-act="fill"' in html
