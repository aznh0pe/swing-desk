"""A full trading day of scans: news found on an open position must stay in every email and on the dashboard
until a later news check looks at that stock again, and then follow that check.

Nov 13, 2026 went wrong twice: the 7:01 AM scan (no news check of its own) left the 6:22 AM ILMN caution out of
its email, and the 2:07 PM post-close scan (it belongs to the next session and checks only new setups) left it
out of its email and hid it on the dashboard. This replays such a day through the real scanner and news step,
then shows the dashboard every scan document they wrote, at the time each was written.
"""
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
from engine import postprocess as pp  # noqa: E402
from engine import scan  # noqa: E402

DAY = [  # (when, mode, news given to the news step or None, should the email carry the S01 caution)
    ("2026-11-05 06:22", "premarket", ["S01=caution:a director kept selling shares"], True),
    ("2026-11-05 06:27", "premarket", ["S01=clean"], True),        # a late re-run of the check: never softens it
    ("2026-11-05 07:01", "preclose", None, True),                  # nothing to news-check
    ("2026-11-05 12:31", "preclose", None, True),
    ("2026-11-05 14:07", "postclose", ["S05=clean"], True),        # next session; checks new setups only
    ("2026-11-06 05:18", "premarket", ["S01=clean"], False),       # the next pre-market check looks again
]


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    base = tmp_path_factory.mktemp("day")
    synth = base / "synth"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synth.py"), str(synth)], check=True, capture_output=True)
    st = base / "state"
    for sub in ("config", "ledger", "trades", "alerts", "scans"):
        (st / sub).mkdir(parents=True)
    (st / "config" / "settings.json").write_text(json.dumps({"initial_balance": 5000, "income_pct": 30, "max_open_positions": 5}))
    (st / "ledger" / "initial.json").write_text(json.dumps({"id": "initial", "type": "initial", "amount": 5000, "date": "2026-11-01"}))
    trade = {"id": "t1", "status": "open", "ticker": "S01", "strategy": "MR_STRETCH", "entry_price": 50.0,
             "entry_date": "2026-11-03", "shares": 5, "stop": 1.0, "target": 9999.0, "max_hold": 8, "dyn_exit": "none"}
    (st / "trades" / "t1.json").write_text(json.dumps(trade))
    steps = []
    for i, (when, mode, news, _) in enumerate(DAY):
        now = dt.datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=cal.PT)
        out = base / f"o{i}"
        scan.run(mode, str(synth), str(st), str(out), now=now)
        if news is not None:
            pp.apply(str(out), news, state_dir=str(st))
        doc = json.loads((out / "scan.json").read_text())["scan"]
        for f in (out / "docs" / "scans").glob("*.json"):  # what the run writes to the dashboard
            shutil.copy(f, st / "scans" / f.name)
        steps.append({"when": when, "doc": doc, "digest": (out / "digest.md").read_text(),
                      "scans": [json.loads(f.read_text()) for f in sorted((st / "scans").glob("*.json"))]})
    return {"steps": steps, "trade": trade}


@pytest.mark.parametrize("i", range(len(DAY)))
def test_every_email_of_the_day_carries_the_news_until_it_is_rechecked(day, i):
    step, expect = day["steps"][i], DAY[i][3]
    pn = [(x["ticker"], x["verdict"]) for x in step["doc"].get("position_news") or []]
    assert (("S01", "caution") in pn) == expect, (step["when"], pn)
    assert ("S01 · 🔴 **CAUTION**" in step["digest"]) == expect, step["when"]
    if expect:
        x = next(x for x in step["doc"]["position_news"] if x["ticker"] == "S01")
        assert x["found_at"].startswith("2026-11-05T06:22")  # always credited to the check that found it
        assert "What the Thu Nov 5, 6:22 AM news check found: A director kept selling shares." in step["digest"]


def test_the_dashboard_shows_it_after_every_scan_and_drops_it_after_the_recheck(day):
    pw = pytest.importorskip("playwright.sync_api")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_dashboard_runs import MOCK, fixture
    html = "<!doctype html><html><head><meta charset='utf-8'></head><body>" + (ROOT / "dashboard" / "index.html").read_text()
    seen = []
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # no browser on this machine
            pytest.skip(f"chromium unavailable: {e}")
        for i, step in enumerate(day["steps"]):
            fx = fixture()
            fx["scans"], fx["trades"] = step["scans"], [day["trade"]]
            page = b.new_context(viewport={"width": 1180, "height": 900}, timezone_id="America/Los_Angeles").new_page()
            errs = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.add_init_script(f"window.__DATA = {json.dumps(fx)};" + MOCK)
            t = dt.datetime.strptime(step["when"], "%Y-%m-%d %H:%M") + dt.timedelta(minutes=5)
            page.clock.set_fixed_time(t.replace(tzinfo=cal.PT))
            page.route("http://dash.test/", lambda r: r.fulfill(status=200, content_type="text/html", body=html))
            page.goto("http://dash.test/")
            page.wait_for_timeout(500)
            page.click("#tab-overview")
            page.wait_for_timeout(150)
            news = page.evaluate("() => [...document.querySelectorAll('.pos-card [data-pos-news]')].map((e) => e.innerText)")
            assert not errs, errs
            seen.append((step["when"], len(news) == 1 and "Caution" in news[0].title() and "6:22 AM" in news[0]))
            page.close()
        b.close()
    assert [s for _, s in seen] == [d[3] for d in DAY], seen
