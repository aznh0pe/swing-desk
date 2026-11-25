"""Nov 25, 2026: the daily review found every published strategy page it opened paywalled and built its web idea
from one page's visible description with the desk's own parameters, logged under that page's name. Now a web idea
must carry the published rules it was built from (source.rules), the prompt sends the review to open sources first
with a fallback that never needs a paywalled page, and the 10-day-average exit common in published rules is
testable as written."""
import importlib.util
import json
import pathlib

import pandas as pd

from engine import backtest as B
from engine import monitor as MON
from engine import rules as R
from engine import research as RS
from engine import strategies as S

from test_management_and_research import _frames, _plan

ROOT = pathlib.Path(__file__).resolve().parents[1]

NOV25 = {"id": "R_ETF_STOCH_PULLBACK_X", "label": "ETF Low Stochastic above a Rising 200-Day", "family": "Mean reversion",
         "about": "A published SPY pullback rule applied to every ETF. The page hides its exact periods, so the 5-day "
                  "stochastic and the exit are the desk's choices.",
         "source": {"title": "Quantified Strategies: A Simple Stochastic Pullback Strategy",
                    "url": "https://quantifiedstrategies.substack.com/p/a-simple-stochastic-pullback-strategy"},
         "universe": "etf", "conditions": ["close > sma200", "slope200 > 0"], "entry": {"kind": "close"},
         "stop": {"atr": 3}, "target": {"atr": 3}, "exit": "sma5", "max_hold": 5, "valid_days": 1, "regime": [],
         "params": {}, "grid": {}}


def test_the_nov25_web_idea_would_now_be_refused():
    errs = RS.source_problems(NOV25)
    assert any("source.rules" in e for e in errs) and any("paywalled or hidden" in e for e in errs)


def test_a_description_only_idea_is_an_own_design_and_says_so():
    own = dict(NOV25, source={"title": "Swing Desk own design (inspired by Quantified Strategies, 2026-11-25)",
                              "url": "https://github.com/aznh0pe/swing-desk"})
    assert RS.source_problems(own) == []


def test_a_web_idea_with_its_published_rules_passes_and_adaptations_must_be_sentences():
    ok = json.loads((ROOT / "engine" / "research_seed6.json").read_text())[0]
    assert R.validate(ok) == [] and RS.check_new(ok, {}) == []
    fresh = dict(ok, id="R_FRESH_WEB_IDEA")                      # not in the curated list: checked in full
    assert RS.source_problems(fresh) == []
    bad = dict(fresh, source=dict(ok["source"], adapted="sells at the close"))
    assert any("adapted" in e for e in RS.source_problems(bad))


def test_the_curated_backlog_is_not_rechecked_and_still_reaches_next():
    ids = RS._curated_ids()
    assert "R_ETF_PULLBACK_10MA_OPEN" in ids
    assert any(sp["id"] == "R_ETF_PULLBACK_10MA_OPEN" for sp in RS.seed_backlog({}))


def test_the_10_day_average_exit_is_replayed_and_explained():
    assert "sma10" in R.EXITS and "10-day average" in MON.DYN_TEXT["sma10"]
    closes = [100, 99, 98, 97, 96, 95, 95, 95, 95, 95, 95, 99, 100, 100, 100]
    ind = _frames(closes)
    ind["sma10"] = ind["close"].rolling(10, min_periods=1).mean()
    P = S.params_with_defaults(None)
    plan = _plan(ind, 9, 200.0, hold=12)
    plan["dyn_exit"] = "sma10"
    tr = B.simulate(plan, ind, P)
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "signal"
    exit_day = pd.Timestamp(tr.iloc[0]["exit_date"]) if "exit_date" in tr.columns else None
    if exit_day is not None:
        assert ind["close"].loc[exit_day, "X"] > ind["sma10"].loc[exit_day, "X"]


def test_the_dashboard_names_the_10_day_exit():
    html = (ROOT / "dashboard" / "index.html").read_text()
    assert 'sma10: "the first day it closes above its 10-day average"' in html


def _prompts():
    spec = importlib.util.spec_from_file_location("bp", ROOT / "ops" / "build_prompts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_review_prompt_never_needs_a_paywalled_page():
    for name in ("learning.md", "backup_learning.md"):
        t = (ROOT / "ops" / "generated" / name).read_text()
        assert "source.rules" in t and "paywalled" in t, name
        assert "open sources first" in t and "StockCharts ChartSchool" in t, name
        assert "own design (inspired by" in t, name
        assert "a paywall is never a run problem" in t, name
