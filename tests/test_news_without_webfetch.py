"""Nov 25, 2026: the 12:26 PM pre-close run had no WebFetch tool and left four featured setups unverified, though a
single "<name> <TICKER> news <Month YYYY>" search showed dated headlines for all four. The news check must never
depend on opening an article: it loads its tools, runs three queries, and a date printed in a search result counts."""
import importlib.util
import pathlib

from engine import scan

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _prompts():
    spec = importlib.util.spec_from_file_location("bp", ROOT / "ops" / "build_prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_plan_has_three_queries_with_the_name_ticker_month_one_second():
    q = scan.news_queries("GE HealthCare", "GEHC", "2026-11-25")
    assert q == ["GE HealthCare stock news November 2026", "GE HealthCare GEHC news November 2026",
                 "GEHC stock latest news headlines November 25, 2026"]


def test_the_plan_says_a_printed_date_counts_without_an_article():
    plan = scan.news_plan({"session_day": "2026-11-25", "news_names": {"ALLE": ["Allegion", "stock"]},
                           "positions": [{"ticker": "ALLE", "last": 1, "pm_last": None}]}, [])
    assert len(plan["items"][0]["queries"]) == 3
    assert "no article needs opening" in plan["how"] and "never clean" in plan["how"]


def test_every_news_step_loads_its_tools_and_works_without_webfetch():
    m = _prompts()
    for step in (m.NEWS_ENTRIES, m.NEWS_PREMARKET, m.NEWS_AUTO):
        assert 'select:WebSearch,WebFetch' in step
        assert "If WebFetch isn't found, carry on without it" in step
        assert "never needs an article opened" in step
        assert "counts in full, never as approximate" in step
        assert "extended" in step and "three search queries" in step
        assert "run every query you have not yet run" in step
        assert "second query, then a result article" not in step


def test_the_generated_prompts_carry_the_fallback():
    gen = sorted((ROOT / "ops" / "generated").glob("*.md"))
    news = [g for g in gen if "news_plan.json" in g.read_text()]
    assert news, "no generated prompt with a news check"
    for g in news:
        t = g.read_text()
        assert "select:WebSearch,WebFetch" in t and "If WebFetch isn't found, carry on without it" in t, g.name
