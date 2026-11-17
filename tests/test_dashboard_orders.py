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
        "function fillCheckMap", "function orderResolutions", "function resolveCard", "const num", "const px", "const pct",
        "function dShort", "function dtShort")


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


def _cards(now, tmp_path, status="new", extra=None):
    js = """
const RealDate = Date, NOW = %s;
global.Date = class extends RealDate { constructor(...a) { if (a.length) super(...a); else super(NOW); } };
const esc = (v) => String(v);
const S = { alerts: [{ id: "A1", type: "entry", status: %s, valid_for: "2026-11-13", valid_days: 1, ticker: "IFF", entry: 80.38, stop: 70.73, target: 82.66 }].concat(%s),
  scans: [{ started_at: "2026-11-16T14:06:46-08:00", fill_checks: [] },
          { started_at: "2026-11-16T12:26:44-08:00", fill_checks: [{ alert_id: "A1", filled: true, last_day: "2026-11-13", low: 79.94, limit: 80.38, touched: false }] }] };
%s
console.log(JSON.stringify(orderResolutions().map((r) => ({ kind: r.kind, html: resolveCard(r) }))));
""" % (json.dumps(now), json.dumps(status), json.dumps(extra or []), _functions())
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


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_buy_at_the_close_setup_nobody_recorded_is_asked_about(tmp_path):
    """Nov 16, 2026: XOP (A+, buy near the close) left the dashboard at 1 PM with status new. No price check can say
    whether a buy-at-the-close order was bought, so nothing ever asked, and a bought position would go unwatched."""
    xop = {"id": "X1", "type": "entry", "status": "new", "valid_for": "2026-11-16", "valid_days": 1, "ticker": "XOP",
           "entry": 179.99, "stop": 163.44, "target": 184.13, "entry_kind": "close", "grade": "A+", "score": 77.3,
           "win_pct": 77.8, "strategy_label": "ETF Close-at-Low Snapback", "created_at": "2026-11-16T12:26:44-08:00"}
    before = _cards("2026-11-16T12:40:00-08:00", tmp_path, "unfilled", [xop])
    assert before == []                                              # its session is still running: the ticket is live
    for now in ("2026-11-16T18:59:00-08:00", "2026-11-19T12:00:00-08:00"):
        cards = _cards(now, tmp_path, "unfilled", [xop])
        assert [c["kind"] for c in cards] == ["unmarked_close"], now
        assert "Did you buy it?" in cards[0]["html"] and "Buy near the close" in cards[0]["html"]
        assert all(f'data-act="{a}"' in cards[0]["html"] for a in ("fill", "nofill", "skip", "missed"))
        # the card says what the setup was and when it was sent: the owner may be seeing it for the first time
        assert all(x in cards[0]["html"] for x in ("A+ setup, ETF Close-at-Low Snapback", "score 77.3", "Nov 16", "12:26", "$163.44", "$184.13"))
    assert _cards("2026-11-20T06:00:00-08:00", tmp_path, "unfilled", [xop]) == []
    assert _cards("2026-11-17T06:00:00-08:00", tmp_path, "unfilled", [{**xop, "status": "skipped"}]) == []
    assert _cards("2026-11-17T06:00:00-08:00", tmp_path, "unfilled", [{**xop, "withdrawn": True}]) == []


def test_the_scan_asks_about_an_unrecorded_buy_at_the_close_too():
    a = {"id": "X1", "ticker": "XOP", "type": "entry", "status": "new", "valid_for": "2026-11-16", "entry": 179.99,
         "entry_kind": "close", "stop": 163.44, "target": 184.13}
    assert SC.order_fill_check(a, {}, dt.date(2026, 11, 16), "preclose", True) is None       # session still running
    fc = SC.order_fill_check(a, {}, dt.date(2026, 11, 17), "premarket", False)
    assert fc["at_close"] and fc["filled"] is None and "press I didn't place it, or Didn't see it in time" in fc["text"]
    src = pathlib.Path(SC.__file__).read_text()
    assert '(fc.get("filled") or fc.get("at_close")) and ask_unmarked(' in src
    assert 'how = "buy near the close" if c.get("at_close")' in src


def test_the_overview_never_calls_a_buy_at_the_close_setup_probably_filled():
    """Nov 16, 2026: the first version of this card put XOP under 'did this order fill? The day's low traded below
    the limit, so it probably filled', for a setup the owner had never seen."""
    assert 'resFill = res.filter((r) => r.kind !== "unmarked_close")' in HTML
    assert "nothing was recorded for" in HTML and 'act === "missed"' in HTML and "missed: true" in HTML


def test_a_setup_that_arrived_too_late_is_counted_as_missed_not_as_a_choice():
    from engine import learnlog as LL
    from engine import strategies as S
    alerts = [{"id": "X1", "type": "entry", "ticker": "XOP", "valid_for": "2026-11-16", "status": "skipped", "missed": True,
               "entry_kind": "close", "grade": "A+", "entry": 179.99, "strategy": "ETF_IBS"},
              {"id": "S1", "type": "entry", "ticker": "AAA", "valid_for": "2026-11-16", "status": "skipped", "entry": 10.0, "strategy": "ETF_IBS"}]
    shadow = [{"alert_id": "X1", "status": "closed", "fill": 180.0, "exit": 183.6}, {"alert_id": "S1", "status": "closed", "fill": 10.0, "exit": 9.0}]
    doc = LL.portfolio_stats(alerts, shadow, [], [], "2026-11-17", S.params_with_defaults(None))
    assert [m["ticker"] for m in doc["orders"]["not_seen_in_time"]] == ["XOP"] and doc["orders"]["not_seen_in_time"][0]["at_close"]
    assert doc["skipped"]["n"] == 1                      # only the one that was a choice
    assert any("reached you too late" in x and "1 of 1 graded since would have won" in x for x in doc["lessons"])
