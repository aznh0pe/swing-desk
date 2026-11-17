"""An order that was never marked placed but probably filled: the question stays until it is answered.

Nov 16, 2026: Friday's IFF order (limit $80.38, low $79.94) was asked about on Monday only. The card offered
"Mark done (filled)" and "I didn't place it", with no way to say "placed it, didn't fill", although the scan's own
text said to press Didn't fill. By Tuesday morning the card would have gone, answered or not.
"""
import datetime as dt
import json
import pathlib
import shutil
import subprocess

import pytest

from engine import scan as SC

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = (ROOT / "dashboard" / "index.html").read_text()
WANT = ("const fmtPT", "const todayPT", "const fmtHM", "const nowHM", "const HOLIDAYS", "const EARLY_CLOSES", "function isTradingDay",
        "function addDays", "function nextTD", "function prevTD", "function lastValidDay", "function sessionOver",
        "function fillCheckMap", "function orderResolutions", "function resolveCard", "const num", "const px", "function dShort")


def _functions():
    src, out = HTML.split("\n"), []
    for i, line in enumerate(src):
        if not line.startswith(WANT):
            continue
        blk, j = [line], i
        if line.startswith("function") and line.rstrip().endswith("{"):
            while not src[j + 1].startswith("}"):
                j += 1
                blk.append(src[j])
            blk.append(src[j + 1])
        elif line.startswith("const") and not line.rstrip().endswith(";"):
            while not src[j + 1].rstrip().endswith(";"):
                j += 1
                blk.append(src[j])
            blk.append(src[j + 1])
        out.append("\n".join(blk))
    return "\n".join(out)


def _cards(now, tmp_path, status="new"):
    js = """
const RealDate = Date, NOW = %s;
global.Date = class extends RealDate { constructor(...a) { if (a.length) super(...a); else super(NOW); } };
const esc = (v) => String(v);
const S = { alerts: [{ id: "A1", type: "entry", status: %s, valid_for: "2026-11-13", valid_days: 1, ticker: "IFF", entry: 80.38, stop: 70.73, target: 82.66 }],
  scans: [{ started_at: "2026-11-16T14:06:46-08:00", fill_checks: [] },
          { started_at: "2026-11-16T12:26:44-08:00", fill_checks: [{ alert_id: "A1", filled: true, last_day: "2026-11-13", low: 79.94, limit: 80.38, touched: false }] }] };
%s
console.log(JSON.stringify(orderResolutions().map((r) => ({ kind: r.kind, html: resolveCard(r) }))));
""" % (json.dumps(now), json.dumps(status), _functions())
    f = tmp_path / "t.js"
    f.write_text(js)
    r = subprocess.run(["node", str(f)], capture_output=True, text=True, env={"TZ": "America/Los_Angeles", "PATH": "/usr/bin:/bin:/opt/node22/bin"})
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_an_unanswered_probable_fill_stays_on_the_dashboard_for_three_sessions(tmp_path):
    for now in ("2026-11-16T18:59:00-08:00", "2026-11-17T06:00:00-08:00", "2026-11-18T12:00:00-08:00"):
        cards = _cards(now, tmp_path)
        assert [c["kind"] for c in cards] == ["unmarked"], now
        # all three answers are one tap away: filled, placed but not filled, never placed
        assert all(f'data-act="{a}"' in cards[0]["html"] for a in ("fill", "nofill", "skip")), now
        assert "Placed it, didn't fill" in cards[0]["html"]
    assert _cards("2026-11-19T06:00:00-08:00", tmp_path) == []          # after three sessions it lets go
    assert _cards("2026-11-17T06:00:00-08:00", tmp_path, "unfilled") == []  # and it goes the moment it is answered


def test_the_scan_keeps_asking_for_the_same_three_sessions():
    fri, mon, tue, wed = (dt.date(2026, 11, d) for d in (13, 16, 17, 18))
    assert SC.UNMARKED_ASK_SESSIONS == 3 and "prevTD(prevTD(prevTD(todayPT())))" in HTML
    assert SC.ask_unmarked("2026-11-13", fri)       # Monday's pre-market and pre-close scans (last session: Friday)
    assert SC.ask_unmarked("2026-11-13", mon)       # Monday's post-close scan: used to drop it here
    assert SC.ask_unmarked("2026-11-13", tue)       # through Wednesday's pre-close scan
    assert not SC.ask_unmarked("2026-11-13", wed)


def test_the_unmarked_text_names_all_three_answers():
    import numpy as np
    import pandas as pd
    idx = pd.to_datetime(["2026-11-12", "2026-11-13"])
    ind = {"low": pd.DataFrame({"IFF": [81.0, 79.94]}, index=idx), "open": pd.DataFrame({"IFF": [81.5, 81.04]}, index=idx)}
    a = {"id": "A1", "ticker": "IFF", "type": "entry", "status": "new", "valid_for": "2026-11-13", "entry": 80.38, "stop": 70.73, "target": 82.66}
    fc = SC.order_fill_check(a, ind, dt.date(2026, 11, 16), "premarket", False)
    assert fc["filled"] and "press Didn't fill; if you never placed it, press I didn't place it." in fc["text"]
    mine = SC.order_fill_check({**a, "status": "placed"}, ind, dt.date(2026, 11, 16), "premarket", False)
    assert mine["text"].endswith("If it didn't, press Didn't fill.")
