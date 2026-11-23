"""Owner, Nov 22, 2026: every alert and every position shows its size as a % of the portfolio (cost over capital),
in the email and on the dashboard: entry alerts (featured and also-qualified), exit and stop alerts, position cards."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import scan as SC  # noqa: E402


def test_a_positions_size_is_its_cost_over_capital():
    assert SC.position_size(101.24, 6, 5026.70) == {"cost": 607.44, "size_pct": 12.1}
    assert SC.position_size(100, 1, 0) == {"cost": 100.0, "size_pct": None}
    assert SC.position_size(None, 3, 5000) == {} and SC.position_size(100, 0, 5000) == {}
    assert SC.size_text(12.1) == "12.1% of portfolio"


def test_the_one_line_alert_and_the_overflow_fields_carry_the_size():
    e = {"ticker": "XLF", "strategy_label": "ETF dip", "grade": "A+", "score": 88, "win_pct": 77, "entry": 50.0, "stop": 48.0,
         "target": 52.0, "shares": 10, "total_cost": 500.0, "pct_of_capital": 9.9, "risk_usd": 20.0, "hold_max_days": 7}
    line = SC.compact_line(e)
    assert "10 sh = $500.0, 9.9% of portfolio, risk $20.0" in line
    src = (ROOT / "engine" / "scan.py").read_text()
    assert '"total_cost", "pct_of_capital",' in src          # kept when a big night thins the alert refreshes
    assert "% of capital)" not in src                          # one wording everywhere: "% of portfolio"
    for needle in ('"size_pct": pos.get("size_pct")', "size_text(p.get('size_pct'))", "% of portfolio\"\n                    if f.get(\"open_cost\")"):
        assert needle in src, needle


def test_the_dashboard_shows_the_size_on_cards_alerts_and_the_section_header():
    pytest.importorskip("playwright.sync_api")
    import json
    import datetime as dt
    from playwright.sync_api import sync_playwright
    from tests.test_dashboard_positions import MOCK, fixture
    fx = fixture()
    fx["alerts"].append({"id": "e_2026-11-12_CCC_X", "type": "entry", "status": "new", "ticker": "CCC", "name": "C Corp", "strategy_label": "Dip",
                         "family": "Mean reversion", "grade": "A+", "score": 90, "win_pct": 78, "entry": 50.0, "stop": 47.0, "target": 53.0,
                         "rr": "1:1", "shares": 12, "total_cost": 600.0, "pct_of_capital": 12.0, "risk_usd": 36.0, "hold_max_days": 7,
                         "valid_for": "2026-11-12", "created_at": "2026-11-11T14:07:00-08:00", "order": "BUY 12 CCC LIMIT 50.00",
                         "exit_signal": "Sell on a close above the 5-day average.", "rank": 1})
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        page = b.new_context(viewport={"width": 1180, "height": 900}, timezone_id="America/Los_Angeles").new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
        page.clock.set_fixed_time(dt.datetime.fromisoformat("2026-11-11T17:05:00-08:00"))
        page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://dash.test/")
        page.wait_for_timeout(700)
        page.click("#tab-overview")
        page.wait_for_timeout(200)
        ov = page.evaluate("() => document.querySelector('#main').innerText")
        page.click("#tab-alerts")
        page.wait_for_timeout(200)
        al = page.evaluate("() => document.querySelector('#main').innerText")
        sizes = page.evaluate("() => [...document.querySelectorAll('article.pos-card .pc-size')].map((x) => x.textContent.trim())")
        b.close()
    assert not errs, errs
    assert "$200.00 invested = 4.0% of portfolio" in ov             # two $100 positions on $5,000
    assert sizes and all(s == "Size 2.0% of portfolio" for s in sizes), sizes
    assert "size 2.0% of portfolio" in al                           # the exit alerts of open positions
    assert "12.0% of portfolio" in al                               # the entry alert card
