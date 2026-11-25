"""A news verdict is only as good as what was read (owner, Nov 23, 2026).

The 5:25 AM pre-market check that day news-checked 8 of 13 open positions (its prompt said "at most 8"), searched
"<TICKER> stock news", got only quote pages with no dated headline, and marked all 8 clean on the absence of news.
Now: the scanner writes a plan of every ticker that must be read (no cap, ETFs apart, company names and searches
that return dated articles); clean needs the date of the newest headline read; anything else is shown as not
verified in the email, the headline and on the dashboard, never as clean.
"""
import datetime as dt
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import calendar as cal  # noqa: E402
from engine import postprocess as pp  # noqa: E402
from engine import scan  # noqa: E402

TODAY = dt.date(2026, 11, 23)


def test_clean_needs_the_date_of_a_headline_that_was_read():
    v, through = pp._verdicts(["AAA=clean", "BBB=clean@2026-11-20:Wins a contract", "CCC=avoid@2026-11-22:Files for an offering",
                               "DDD=clean@2026-12-30:typo in the date", "EEE=fine", "FFF=unverified:only quote pages",
                               "GGG=caution:CFO leaves"], TODAY)
    assert v["AAA"][0] == "unverified" and "no dated headline" in v["AAA"][1]          # the Nov 23 case
    assert v["BBB"] == ("clean", "Wins a contract") and through["BBB"] == "2026-11-20"
    assert v["CCC"] == ("avoid", "Files for an offering") and through["CCC"] == "2026-11-22"
    assert v["DDD"][0] == "unverified"                                                  # a date in the future proves nothing
    assert v["EEE"][0] == "unverified" and "unrecognized" in v["EEE"][1]                # a typo never becomes clean
    assert v["FFF"] == ("unverified", "only quote pages")
    assert v["GGG"] == ("caution", "CFO leaves")                                        # bad news counts with or without a date


def _pos(tk, last, pm=None):
    return {"ticker": tk, "last": last, "pm_last": pm, "trade_id": "t_" + tk}


def test_the_plan_lists_every_position_with_no_cap_and_puts_etfs_apart():
    stocks = ["CRL", "DUOL", "EL", "FRPT", "ICUI", "SHOP", "SNOW"]
    etfs = ["XOP", "IGV", "XBI", "XLY", "XRT", "FAS"]
    names = {**{t: [f"{t} Corp", "stock"] for t in stocks}, **{t: [t, "etf"] for t in etfs}, "NEWCO": ["Newco", "stock"],
             "PLACED": ["Placed Co", "stock"], "GAPPY": ["Gappy", "stock"], "LATER": ["Later", "stock"]}
    doc = {"session_day": "2026-11-23", "news_names": names,
           "positions": [_pos(t, 100) for t in stocks + etfs] + [_pos("GAPPY", 50, 46)],
           "premarket_checks": [{"ticker": "PLACED", "verdict": "keep", "placed": True},
                                {"ticker": "NEWCO", "verdict": "keep", "rank": 2},
                                {"ticker": "LATER", "verdict": "keep", "rank": None, "also_qualified": True}]}
    plan = scan.news_plan(doc, [])
    tks = [i["ticker"] for i in plan["items"]]
    assert tks[0] == "GAPPY" and "pre-market -8.0%" in plan["items"][0]["why"]           # a mover is read first
    assert set(stocks) <= set(tks) and len(tks) == 10                                   # all 8 stock positions, no cap at 8
    assert tks[-2:] == ["PLACED", "NEWCO"] and "LATER" not in tks                        # placed, then today's featured orders
    assert [e["ticker"] for e in plan["etfs"]] == etfs                                   # ETFs: no company news to search
    snow = next(i for i in plan["items"] if i["ticker"] == "SNOW")
    assert snow["queries"] == ["SNOW Corp stock news November 2026", "SNOW Corp SNOW news November 2026",
                               "SNOW stock latest news headlines November 23, 2026"]
    assert "never clean" in plan["how"]


def test_search_names_come_from_the_list_without_legal_suffixes():
    assert scan.short_name("Snowflake Inc. Class A Common Stock", "SNOW") == "Snowflake"
    assert scan.short_name("The Estée Lauder Companies Inc.", "EL") == "Estée Lauder Companies"
    assert scan.short_name(None, "brk.b") == "BRK.B"
    import pandas as pd
    u = pd.DataFrame([{"ticker": "FAS", "name": "Financials 3x", "type": "etf_lev"},
                      {"ticker": "SNOW", "name": "Snowflake", "type": "stock"}]).set_index("ticker", drop=False)
    assert scan.news_names(u, ["FAS", "SNOW", "ZZZ"]) == {"FAS": ["Financials 3x", "etf"], "SNOW": ["Snowflake", "stock"],
                                                           "ZZZ": ["ZZZ", "stock"]}


@pytest.fixture(scope="module")
def premarket(tmp_path_factory):
    base = tmp_path_factory.mktemp("pm")
    synth = base / "synth"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synth.py"), str(synth)], check=True, capture_output=True)
    uni = json.loads((synth / "universe.json").read_text())["items"]
    stocks = [u["ticker"] for u in uni if u["type"] == "stock"][:9]
    st = base / "state"
    for sub in ("config", "ledger", "trades", "alerts", "scans"):
        (st / sub).mkdir(parents=True)
    (st / "config" / "settings.json").write_text(json.dumps({"initial_balance": 5000, "max_open_positions": 7}))
    (st / "ledger" / "initial.json").write_text(json.dumps({"id": "initial", "type": "initial", "amount": 5000, "date": "2026-11-01"}))
    for tk in stocks + ["XLF", "TQQQ"]:     # 11 positions: 9 stocks (more than the old cap of 8) and 2 ETFs
        (st / "trades" / f"t_{tk}.json").write_text(json.dumps({
            "id": f"t_{tk}", "status": "open", "ticker": tk, "strategy": "MR_STRETCH", "entry_price": 50.0, "entry_date": "2026-11-03",
            "shares": 2, "stop": 1.0, "target": 9999.0, "max_hold": 8, "dyn_exit": "none"}))
    others = [u["ticker"] for u in uni if u["type"] == "stock"][9:39]
    for i, tk in enumerate(others):   # 30 standing setups for today: one placed, one featured #1, the rest also qualified
        (st / "alerts" / f"e_{tk}.json").write_text(json.dumps({
            "id": f"e_{tk}", "type": "entry", "status": "placed" if i == 0 else "new", "ticker": tk, "valid_for": "2026-11-05",
            "entry": 1.0, "stop": 0.5, "target": 2.0, "atr": 0.1, "rank": 1 if i == 1 else None, "also_qualified": i > 1,
            "created_at": "2026-11-04T14:05:00-08:00"}))
    return {"synth": synth, "st": st, "base": base, "stocks": stocks, "placed": others[0], "featured": others[1]}


def _run(pm, name, news):
    out = pm["base"] / name
    now = dt.datetime(2026, 11, 5, 5, 25, tzinfo=cal.PT)
    scan.run("premarket", str(pm["synth"]), str(pm["st"]), str(out), now=now)
    before = json.loads((out / "scan.json").read_text())["scan"]
    res = pp.apply(str(out), news, state_dir=str(pm["st"])) if news is not None else None
    return out, before, res


def test_before_the_news_step_nothing_counts_as_checked(premarket):
    out, doc, _ = _run(premarket, "o0", None)
    plan = json.loads((out / "news_plan.json").read_text())
    tks = [i["ticker"] for i in plan["items"]]
    assert set(tks[:9]) == set(premarket["stocks"])                                    # all 9 stock positions come first
    assert tks[9:] == [premarket["placed"], premarket["featured"]]                     # then the placed and featured orders
    assert {e["ticker"] for e in plan["etfs"]} == {"XLF", "TQQQ"}
    assert (out / "news_order.txt").read_text().split() == [i["ticker"] for i in plan["items"]]
    assert set(doc["news_unverified"]) == set(tks)
    assert doc["position_news_status"]["XLF"] == {"status": "etf"}
    assert doc["position_news_status"][premarket["stocks"][0]] == {"status": "not_checked"}
    assert "news not verified on 11" in scan.headline(doc)


def test_a_check_that_found_no_dated_headline_says_so_everywhere(premarket):
    s = premarket["stocks"]
    news = [f"{s[0]}=clean"] + [f"{t}=clean@2026-11-04:Routine product news" for t in s[1:8]]   # the 9th is left out
    out, _, res = _run(premarket, "o1", news)
    doc = json.loads((out / "scan.json").read_text())["scan"]
    digest = (out / "digest.md").read_text()
    pst = doc["position_news_status"]
    assert pst[s[0]]["status"] == "unverified" and pst[s[1]] == {"status": "clean", "through": "2026-11-04", "note": "Routine product news"}
    assert pst[s[8]] == {"status": "not_checked"} and pst["TQQQ"] == {"status": "etf"}
    left = [s[8], premarket["placed"], premarket["featured"]]   # the 9th position and both orders were not checked
    assert res["NEWS_CHECK_MISSING"].split(":")[1].split(".")[0].split() == left
    assert s[0] in res["NEWS_UNVERIFIED"]
    assert sorted(doc["news_unverified"]) == sorted([s[0]] + left)
    assert "news not verified on 4" in scan.email_headline(doc)
    assert "News: NOT VERIFIED: no dated headline was found." in digest
    assert "News: headlines read through Wed Nov 4: nothing material (newest: Routine product news)." in digest
    assert "News: not checked by this run." in digest
    assert "News: ETF: no company news to read (market risk only)." in digest
    assert "=clean\"" not in json.dumps(doc)       # nothing records a bare clean


def test_a_full_check_with_dated_headlines_reads_clean(premarket):
    s = premarket["stocks"]
    out, _, res = _run(premarket, "o2", [f"{t}=clean@2026-11-04:Routine news" for t in s + [premarket["placed"], premarket["featured"]]])
    doc = json.loads((out / "scan.json").read_text())["scan"]
    assert "NEWS_CHECK_MISSING" not in res and "NEWS_UNVERIFIED" not in res
    assert doc["news_unverified"] == [] and "news not verified" not in scan.headline(doc)


def test_the_premarket_email_lists_only_what_needs_reading(premarket):
    """Nov 23: ~250 standing setups each printed "KEEP. No pre-market issues.", which read as news-checked too."""
    s = premarket["stocks"]
    pl, ft = premarket["placed"], premarket["featured"]
    out, before, _ = _run(premarket, "o3", [f"{t}=clean@2026-11-04:Routine news" for t in s] + [f"{pl}=clean@2026-11-03:New plant"])
    plan = [i["ticker"] for i in json.loads((out / "news_plan.json").read_text())["items"]]
    assert plan[-2:] == [pl, ft]                                   # after the positions: the placed order, today's featured #1
    digest = (out / "digest.md").read_text()
    sect = digest.split("PRE-MARKET CHECK OF TODAY'S ORDERS ===")[1].split("===")[0]
    rows = [l for l in sect.splitlines() if l.startswith("- ")]
    assert len(rows) == 3, rows                                    # placed, featured, and one line for the other 28
    assert any(l.startswith(f"- {pl} (marked placed): KEEP.") and "headlines read through Tue Nov 3" in l for l in rows)
    assert any(l.startswith(f"- {ft} (featured #1): KEEP.") and "News: not checked by this run." in l for l in rows)
    assert rows[-1].startswith("- 28 more standing setups for today:") and "news was not checked" in rows[-1]
    assert "No pre-market issues." not in digest


def test_every_news_prompt_uses_the_plan_dates_and_no_cap():
    gen = ROOT / "ops" / "generated"
    for f in sorted(gen.glob("*.md")):
        t = f.read_text()
        if "engine.postprocess" not in t or "--news" not in t:
            continue
        assert "news_plan.json" in t and "@YYYY-MM-DD" in t and "never clean" in t, f.name
        assert "at most 8" not in t and '"<TICKER> stock news"' not in t, f.name
        assert "unverified" in t and "NEWS_UNVERIFIED" in t, f.name


def test_the_dashboard_card_says_whether_the_news_was_read():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    from tests.test_dashboard_positions import MOCK, fixture
    fx = fixture()
    fx["scans"][0]["position_news_status"] = {"AAA": {"status": "unverified", "note": "only quote pages"},
                                              "BBB": {"status": "clean", "through": "2026-11-10", "note": "New contract"}}
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
        got = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('article.pos-card')].map((c) => [
            c.querySelector('.tk').textContent.trim(), (c.querySelector('[data-news-status]') || {}).textContent || '']))""")
        b.close()
    assert not errs, errs
    assert got["AAA"].startswith("News not verified") and "only quote pages" in got["AAA"]
    assert got["BBB"].startswith("News read") and "Headlines through" in got["BBB"] and "New contract" in got["BBB"]
