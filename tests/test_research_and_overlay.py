"""Quick checks: rule specs parse and evaluate, the seed backlog is valid, and the alert overlay works."""
import json
import pathlib

import numpy as np
import pandas as pd

from engine import rules as R
from engine.scan import build_overlay, headline

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _ind(n=300, k=4):
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(0)
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)), index=idx, columns=list("ABCD"))
    ind = {"close": c, "open": c.shift(1).fillna(c), "high": c * 1.01, "low": c * 0.99, "volume": c * 0 + 1e6,
           "atr10": c * 0.02, "sma200": c.rolling(200).mean(), "sma5": c.rolling(5).mean(), "rsi2": c * 0 + 50}
    ind["market"] = pd.DataFrame({"vix_ratio": 1.0, "spy_above200": 1.0}, index=idx)
    return ind


def test_seed_specs_are_valid():
    seeds = json.loads((ROOT / "config" / "research_seed.json").read_text())
    assert len(seeds) >= 8
    for s in seeds:
        assert R.validate(s) == [], s["id"]
        assert 1 <= len(R.combos(s)) <= R.MAX_COMBOS


def test_evaluator_lags_functions_calendar():
    ind = _ind()
    ev = R.Evaluator(ind)
    m = ev.mask("close > close[1] and rmin(low, 3) < low")
    assert m.shape == ind["close"].shape and m.dtypes.eq(bool).all()
    f = ev.frame("max(close - 2*atr10, low)")
    assert np.allclose(f.values[-1], np.maximum(ind["close"].values[-1] * 0.96, ind["low"].values[-1]))
    cal = ev.mask("tdom_rev == 0")
    assert cal.iloc[:, 0].sum() >= 10  # one last-session-of-month per month
    assert ev.mask("vix_ratio > 0.5").all().all()


def test_validate_rejects_unknown_names():
    bad = {"id": "R_BAD", "label": "x", "family": "x", "about": "x", "conditions": ["foo > 1", "__import__('os')"],
           "entry": {"kind": "limit_next", "price": "close"}, "stop": {"atr": 2}, "target": {"atr": 1}, "exit": "sma5"}
    errs = R.validate(bad)
    assert any("unknown field" in e for e in errs) and any("unknown function" in e for e in errs)


def test_overlay_refreshes_and_withdraws():
    ctx = {"existing_entry_ids": ["e_1", "e_2"], "window": {"e_1": {"ticker": "AAA"}, "e_2": {"ticker": "BBB"}}, "when": "3:13 pm"}
    entries = [{"id": "e_1", "ticker": "AAA", "rank": 2, "score": 81.0, "news_check": "clean", "shares": 3}]
    scan = {"news_dropped": [{"ticker": "BBB", "reason": "offering"}], "counts": {"aplus": 1}}
    ups = build_overlay(scan, entries, ctx)
    assert ups["e_1"]["rank"] == 2 and ups["e_1"]["withdrawn"] is False
    assert ups["e_2"]["withdrawn"] is True and "offering" in ups["e_2"]["withdrawn_reason"]
    scan["alert_updates"] = ups
    assert headline(scan) == "1 A+ setup · 1 withdrawn"
