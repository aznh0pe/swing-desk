"""Owner, Nov 25, 2026: "for the optional A+ setups, if it's already past a full day after the day of the alert,
there's no need to keep it." That morning's pre-market email still listed 54 optional setups from Fri Nov 20 and
Mon Nov 23 under "Not orders", and the dashboard kept them for three sessions. An optional setup now stays only until
the session after its own day has ended (scan.optional_kept / dashboard optionalKept), in the email's summary line,
the dashboard's traded-through table and the carried withdrawals. Featured and placed orders keep their windows."""
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import calendar as cal  # noqa: E402
from engine import scan as SC  # noqa: E402

PT = dt.timezone(dt.timedelta(hours=-8))


def test_an_optional_setup_lasts_through_the_session_after_its_day():
    done = lambda y, m, d, h, mi: cal.last_completed_session(dt.datetime(y, m, d, h, mi, tzinfo=PT))  # noqa: E731
    wed_pre, wed_post = done(2026, 11, 25, 5, 21), done(2026, 11, 25, 14, 6)
    assert (str(wed_pre), str(wed_post)) == ("2026-11-24", "2026-11-25")
    assert SC.optional_kept("2026-11-24", wed_pre)          # Tuesday's setup: still there Wednesday morning
    assert not SC.optional_kept("2026-11-23", wed_pre)      # Monday's: gone (the Nov 25 email had these)
    assert not SC.optional_kept("2026-11-20", wed_pre)
    assert not SC.optional_kept("2026-11-24", wed_post)     # Tuesday's: gone once Wednesday has closed
    assert SC.optional_kept("2026-11-25", wed_post)
    sun = done(2026, 11, 29, 16, 17)                        # the Sunday refresh: Friday is the last session
    assert str(sun) == "2026-11-27" and SC.optional_kept("2026-11-27", sun) and not SC.optional_kept("2026-11-26", sun)
    assert not SC.optional_kept(None, wed_pre)


def test_the_scan_uses_the_window_for_fill_checks_and_carried_withdrawals():
    src = (ROOT / "engine" / "scan.py").read_text()
    assert "done_session = cal.last_completed_session(now)" in src
    assert 'if not optional_kept(fc["last_day"], done_session):' in src
    assert "prev_ov, state, today, done_session)" in src
    state = {"alerts": {
        "e_opt_mon": {"doc": {"type": "entry", "status": "new", "valid_for": "2026-11-23", "also_qualified": True, "ticker": "AAA"}},
        "e_opt_tue": {"doc": {"type": "entry", "status": "new", "valid_for": "2026-11-24", "also_qualified": True, "ticker": "BBB"}},
        "e_top_mon": {"doc": {"type": "entry", "status": "new", "valid_for": "2026-11-23", "rank": 1, "ticker": "CCC"}},
    }}
    prev = {k: {"withdrawn": True, "withdrawn_reason": "news", "ticker": v["doc"]["ticker"]} for k, v in state["alerts"].items()}
    ups = SC.carry_withdrawn({}, prev, state, dt.date(2026, 11, 25), dt.date(2026, 11, 24))
    assert set(ups) == {"e_opt_tue", "e_top_mon"}     # Monday's optional withdrawal is no longer carried; a top pick's is


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_dashboard_drops_optional_setups_a_day_after_their_day(tmp_path):
    from tests.test_dashboard_orders import _functions
    alerts = [{"id": f"A_{t}", "type": "entry", "status": "new", "valid_for": d, "valid_days": 1, "ticker": t,
               "entry": 100, "stop": 90, "target": 105, "also_qualified": True}
              for t, d in (("FRI", "2026-11-20"), ("MON", "2026-11-23"), ("TUE", "2026-11-24"))]
    fcs = [{"alert_id": a["id"], "filled": True, "last_day": a["valid_for"], "low": 99, "limit": 100, "swap": True} for a in alerts]

    def run(now):
        js = """
const RealDate = Date, NOW = "%s";
global.Date = class extends RealDate { constructor(...a) { if (a.length) super(...a); else super(NOW); } };
const esc = (v) => String(v);
const S = { alerts: %s, scans: [{ started_at: "2026-11-25T05:21:00-08:00", fill_checks: %s }] };
%s
console.log(JSON.stringify(swapListFills().map((r) => r.a.ticker)));
""" % (now, json.dumps(alerts), json.dumps(fcs), _functions())
        f = tmp_path / "t.js"
        f.write_text(js)
        r = subprocess.run(["node", str(f)], capture_output=True, text=True,
                           env={"TZ": "America/Los_Angeles", "PATH": "/usr/bin:/bin:/opt/node22/bin"})
        assert r.returncode == 0, r.stderr[-800:]
        return json.loads(r.stdout.strip().splitlines()[-1])
    assert run("2026-11-25T07:12:00-08:00") == ["TUE"]   # the owner's message: Friday's and Monday's are gone
    assert run("2026-11-25T14:00:00-08:00") == []        # Wednesday has closed: Tuesday's go too


def test_an_optional_setup_you_placed_is_never_dropped_after_a_day():
    """Owner, Nov 25, 2026: "if an optional A+ alert was ordered placed, you won't remove those the day after right?"
    Right: once marked placed it is an order, followed like any other (fill questions for 3 sessions, withdrawals
    carried, the position watched once filled), whether it was a top pick or optional."""
    src = (ROOT / "engine" / "scan.py").read_text()
    assert 'if a.get("status") != "placed" and is_swap_list(a, prev_ov.get(aid)):' in src     # fill checks
    state = {"alerts": {
        "e_opt_mon_placed": {"doc": {"type": "entry", "status": "placed", "valid_for": "2026-11-23", "also_qualified": True, "ticker": "AAA"}},
        "e_opt_mon_new": {"doc": {"type": "entry", "status": "new", "valid_for": "2026-11-23", "also_qualified": True, "ticker": "BBB"}},
    }}
    prev = {k: {"withdrawn": True, "withdrawn_reason": "news", "ticker": v["doc"]["ticker"]} for k, v in state["alerts"].items()}
    ups = SC.carry_withdrawn({}, prev, state, dt.date(2026, 11, 25), dt.date(2026, 11, 24))
    assert set(ups) == {"e_opt_mon_placed"}
    html = (ROOT / "dashboard" / "index.html").read_text()
    i = html.index("function orderResolutions()")
    body = html[i:html.index("return out.sort", i)]
    # the dashboard asks about every placed order before the optional rule is even looked at
    assert body.index('if (a.status === "placed")') < body.index("if (isSwapList(a, fc)) continue;")
