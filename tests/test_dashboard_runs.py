"""Browser test: a pre-market check that hasn't come in gets a one-tap catch-up before the open.

Fri Nov 13, 2026: the scheduler never started the 5:50 AM pre-market check or its 6:02 AM backup (both were still
waiting at 6:26 AM), and the dashboard said "Running" and then "Backup running" for runs that had not started. The
check now runs at 5:17 AM (backup 5:47 AM); from 12 minutes after it was due until the 6:30 open, a check that hasn't
come in shows as "Not in yet" with Run now (Overview banner and Scans tab), which starts the Scan now task.
"""
import datetime as dt
import json
import pathlib

import pytest

pw = pytest.importorskip("playwright.sync_api")

ROOT = pathlib.Path(__file__).resolve().parent.parent
MOCK = """
window.__calls = [];
window.claude = { use: async (n) => {
  const data = window.__DATA;
  if (n === "mcp") return { callTool: async (server, tool, args) => { window.__calls.push({ server, tool, args }); return { payload: {} }; } };
  if (n === "user") return { isOwner: () => true, canEdit: () => true };
  if (n !== "db") return null;
  const snapDoc = (p) => ({ exists: !!data[p], data: () => data[p], id: p.split("/").pop(), metadata: { fromCache: false } });
  const snapCol = (c) => ({ docs: (data[c] || []).map((d) => ({ id: d.id, data: () => d, exists: true })), metadata: { fromCache: false } });
  const col = (c) => { const q = { orderBy: () => q, limit: () => q, where: () => q,
    onSnapshot: (cb) => { setTimeout(() => cb(snapCol(c)), 5); return () => {}; }, doc: (id) => docRef(c + "/" + id) }; return q; };
  const docRef = (p) => ({ onSnapshot: (cb) => { setTimeout(() => cb(snapDoc(p)), 5); return () => {}; },
    set: async () => {}, update: async () => {}, delete: async () => {} });
  return { doc: docRef, collection: col };
} };
"""

BANNER = "This morning's pre-market check hasn't come in yet"


def fixture(premarket_at=None):
    # yesterday's post-close scan, and (optionally) this morning's pre-market check
    scans = [{"id": "20261112-1442-postclose", "mode": "postclose", "started_at": "2026-11-12T14:42:00-08:00",
              "status": "ok", "session_day": "2026-11-12", "run_issues": [], "positions": []}]
    if premarket_at:
        scans.insert(0, {"id": "20261113-" + premarket_at.replace(":", "") + "-premarket", "mode": "premarket",
                         "started_at": f"2026-11-13T{premarket_at}:00-08:00", "status": "ok", "session_day": "2026-11-13",
                         "run_issues": [], "positions": []})
    return {"config/settings": {"initial_balance": 5000, "max_open_positions": 5, "max_position_pct": 25, "risk_pct": 2,
                                "income_default_pct": 30, "timezone": "America/Los_Angeles",
                                "manual_trigger_id": "trig_manual_test"},
            "alerts": [], "trades": [], "ledger": [{"id": "initial", "type": "initial", "amount": 5000, "date": "2026-11-05"}],
            "scans": scans, "journal": [], "learning": [], "research": [], "research_results": [], "runs": [],
            "strategy/params": {"params": {"grading": {"aplus_score": 86, "aplus_pwin": 0.62}, "strategies": {}}},
            "strategy/backtest": {}}


def visit(fx, clock, press=False):
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # no browser on this machine
            pytest.skip(f"chromium unavailable: {e}")
        ctx = b.new_context(viewport={"width": 1180, "height": 900}, timezone_id="America/Los_Angeles")
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
        page.clock.set_fixed_time(dt.datetime.fromisoformat(f"2026-11-13T{clock}:00-08:00"))
        page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://dash.test/")
        page.wait_for_timeout(700)
        page.click("#tab-overview")
        page.wait_for_timeout(200)
        overview = page.evaluate("() => document.querySelector('#main').innerText")
        calls = []
        if press:
            page.click("button[data-act='runnow'][data-run='premarket']")
            page.wait_for_timeout(300)
            calls = page.evaluate("() => window.__calls")
        page.click("#tab-scans")
        page.wait_for_timeout(200)
        row = page.evaluate("""() => { const r = [...document.querySelectorAll('tr')].find((t) => t.innerText.includes('Pre-market check'));
          return r ? { text: r.innerText, runNow: !!r.querySelector("button[data-act='runnow']") } : null; }""")
        b.close()
    assert not errs, errs
    return overview, row, calls


def test_before_the_check_is_late_it_is_only_waiting():
    overview, row, _ = visit(fixture(), "05:20")
    assert BANNER not in overview
    assert "Waiting for results" in row["text"] and "5:17 AM" in row["text"] and not row["runNow"]
    assert "Running" not in row["text"]  # nothing says a run started when the page can't know that


def test_a_late_check_offers_run_now_and_it_starts_the_scan_now_task():
    overview, row, calls = visit(fixture(), "05:35", press=True)
    assert BANNER in overview
    assert "Not in yet" in row["text"] and row["runNow"]
    assert [c["tool"] for c in calls] == ["fire_trigger"]
    assert calls[0]["args"]["trigger_id"] == "trig_manual_test"


def test_a_check_that_came_in_clears_it():
    overview, row, _ = visit(fixture(premarket_at="05:19"), "05:35")
    assert BANNER not in overview
    assert "Ran 5:19 AM" in row["text"] and not row["runNow"]


def test_a_catch_up_after_a_late_start_counts():
    # today's case: the check came in at 6:22 AM (a catch-up), well after the old 5:40 AM cut-off
    overview, row, _ = visit(fixture(premarket_at="06:22"), "06:40")
    assert BANNER not in overview and "Ran 6:22 AM" in row["text"]


def test_after_the_open_a_missing_check_is_reported_as_missed():
    overview, row, _ = visit(fixture(), "06:40")
    assert BANNER not in overview
    assert "Pre-market check didn't run today" in overview
    assert "Missed" in row["text"] and row["runNow"]


def visit_day(fx, day, clock):
    """Same as visit(), on any date (for the pre-close checks and half days)."""
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # no browser on this machine
            pytest.skip(f"chromium unavailable: {e}")
        ctx = b.new_context(viewport={"width": 1180, "height": 900}, timezone_id="America/Los_Angeles")
        page = ctx.new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
        page.clock.set_fixed_time(dt.datetime.fromisoformat(f"{day}T{clock}:00-08:00"))
        page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://dash.test/")
        page.wait_for_timeout(700)
        page.click("#tab-overview")
        page.wait_for_timeout(200)
        overview = page.evaluate("() => document.querySelector('#main').innerText")
        b.close()
    assert not errs, errs
    return overview


PRECLOSE = "Today's pre-close exit scan hasn't come in yet"


def test_a_late_pre_close_scan_offers_run_now_before_the_close():
    fx = fixture(premarket_at="05:19")
    assert PRECLOSE not in visit_day(fx, "2026-11-13", "12:30")       # not late yet
    ov = visit_day(fx, "2026-11-13", "12:40")
    assert PRECLOSE in ov and "before the 1:00 PM close" in ov
    assert PRECLOSE not in visit_day(fx, "2026-11-13", "12:57")       # too late to help: the backup / missed path


def test_a_pre_close_scan_that_came_in_clears_it():
    fx = fixture(premarket_at="05:19")
    fx["scans"].insert(0, {"id": "20261113-1227-preclose", "mode": "preclose", "started_at": "2026-11-13T12:27:00-08:00",
                           "status": "ok", "session_day": "2026-11-13", "run_issues": [], "positions": []})
    assert PRECLOSE not in visit_day(fx, "2026-11-13", "12:40")


def test_half_day_pre_close_catch_up_uses_the_early_close():
    fx = fixture()
    fx["scans"].insert(0, {"id": "20261127-0518-premarket", "mode": "premarket", "started_at": "2026-11-27T05:18:00-08:00",
                           "status": "ok", "session_day": "2026-11-27", "run_issues": [], "positions": []})
    ov = visit_day(fx, "2026-11-27", "09:35")
    assert PRECLOSE in ov and "before the 10:00 AM close" in ov
