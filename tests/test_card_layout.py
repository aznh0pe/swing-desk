"""Owner, Nov 23, 2026: the setup card's bottom notes (market condition record, wider-list note) sat flush against the
card edge with no padding, squished above the yellow strips. Every text block in a setup card keeps the card's inset."""
import datetime as dt
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_setup_card_notes_are_padded_like_the_rest_of_the_card():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from tests.test_dashboard_positions import MOCK, fixture
    fx = fixture()
    fx["alerts"].append({"id": "e_2026-11-12_AAA_X", "type": "entry", "status": "new", "ticker": "AAA", "name": "A Corp", "rank": 1,
                         "strategy_label": "Dip", "family": "Fair value", "grade": "A+", "score": 99, "win_pct": 78, "entry": 80.0,
                         "stop": 65.0, "target": 85.0, "rr": 0.3, "shares": 6, "total_cost": 480.0, "pct_of_capital": 9.6,
                         "risk_usd": 90.0, "hold_max_days": 8, "valid_for": "2026-11-12", "created_at": "2026-11-11T14:07:00-08:00",
                         "order": "Buy LIMIT $80.00", "exit_signal": "Sell above the 5-day average.",
                         "condition_note": "In bear-market selloffs (today's condition), A+ setups won 72% of 723.",
                         "grade_note": "Graded A+ on the wider list.", "list_note": "From the wider list: $3.0B company.",
                         "size_note": "No free cash at the moment.", "capacity_note": "Also qualified (2 already in Financials)."})
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        page = b.new_context(viewport={"width": 400, "height": 900}, timezone_id="America/Los_Angeles").new_page()
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
        page.clock.set_fixed_time(dt.datetime.fromisoformat("2026-11-11T17:05:00-08:00"))
        page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
        page.goto("http://dash.test/")
        page.wait_for_timeout(700)
        page.click("#tab-alerts")
        page.wait_for_timeout(300)
        res = page.evaluate("""() => {
          const card = document.querySelector('article.ticket[aria-label="AAA setup"]');
          const left = card.getBoundingClientRect().left;
          const aside = card.querySelector('.t-aside');
          // every direct child that holds text must start its text at least 10px inside the card edge
          const flush = [...card.children].filter((c) => c.textContent.trim() && !c.matches('.t-head, .t-body, .t-facts'))
            .map((c) => { const r = document.createRange(); r.selectNodeContents(c); return [c.className, r.getClientRects()[0].left - left]; })
            .filter(([, d]) => d < 10);
          return { notes: aside ? [...aside.querySelectorAll('p')].map((x) => x.textContent) : [], flush };
        }""")
        b.close()
    assert not errs, errs
    assert res["notes"] == ["In bear-market selloffs (today's condition), A+ setups won 72% of 723.",
                            "Graded A+ on the wider list.", "From the wider list: $3.0B company."]
    assert res["flush"] == [], res["flush"]
