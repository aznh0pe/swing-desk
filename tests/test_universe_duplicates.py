"""Nov 25, 2026, 5:21 AM pre-market scan: it crashed ("cannot reindex on an axis with duplicate labels") because the
day's universe file listed IBIT twice, as a core ETF ("Bitcoin") and as an extension-tier stock: the Nasdaq screener
lists the fund as a stock, and the builder checked the extension tier only against core STOCKS, then appended the
core funds without a check. The run patched its own copy and re-ran; these tests keep it from happening again:
the builder never writes a duplicate, and the engine de-duplicates on read whatever wrote the file."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from engine import data as D  # noqa: E402
from engine import indicators as I  # noqa: E402

NOV25 = [  # the two IBIT entries exactly as they were in that day's file, plus neighbours
    {"ticker": "AAPL", "name": "Apple", "type": "stock", "sector": "Information Technology", "tier": "core"},
    {"ticker": "IBIT", "name": "iShares Bitcoin Trust ETF Shares", "type": "stock", "sector": None, "tier": "ext"},
    {"ticker": "SONN", "name": "Sonnet", "type": "stock", "sector": "Health Care", "tier": "ext"},
    {"ticker": "IBIT", "name": "Bitcoin", "type": "etf", "sector": None, "tier": "core"},
    {"ticker": "SPY", "name": "S&P 500", "type": "etf", "sector": None, "tier": "core"},
]


def _write(tmp_path, items):
    (tmp_path / "universe.json").write_text(json.dumps({"built": "2026-11-25", "mode": "wide", "items": items}))
    return tmp_path


def test_the_engine_keeps_one_row_per_ticker_and_the_core_entry_wins(tmp_path):
    d = _write(tmp_path, NOV25)
    u = D.load_universe(d)
    assert u.index.is_unique and len(u) == 4
    assert u.loc["IBIT", "tier"] == "core" and u.loc["IBIT", "type"] == "etf" and u.loc["IBIT", "name"] == "Bitcoin"
    assert D.tier_tickers(d, "ext") == ["SONN"] and "IBIT" in D.tier_tickers(d, "core")
    info = D.universe_info(d)
    assert info["duplicates"] == ["IBIT"] and info["core"] + info["ext"] == 4
    note = D.duplicates_note(D.universe_items(d)[1])
    assert note.startswith("The universe list named IBIT twice") and "dropped 1 duplicate." in note
    clean = _write(tmp_path, [i for i in NOV25 if i["tier"] == "core"])
    assert D.duplicates_note(D.universe_items(clean)[1]) is None and "duplicates" not in D.universe_info(clean)


def test_the_indicator_step_that_crashed_now_runs_on_that_list(tmp_path):
    u = D.load_universe(_write(tmp_path, NOV25))
    idx = pd.bdate_range("2025-06-02", periods=260)
    rng = np.random.default_rng(0)
    w = {}
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 4)), axis=0)), index=idx,
                         columns=["AAPL", "IBIT", "SONN", "SPY"])
    w["close"] = close
    w["open"], w["high"], w["low"] = close * 0.999, close * 1.01, close * 0.99
    w["volume"] = close * 0 + 1e6
    ind = I.compute(w, u)  # Nov 25: ValueError('cannot reindex on an axis with duplicate labels')
    assert ind is not None


def test_the_scan_says_once_that_a_duplicate_was_dropped():
    src = (ROOT / "engine" / "scan.py").read_text()
    assert "dup_note = D.duplicates_note(D.universe_items(data_dir)[1])" in src


def test_the_builder_never_lists_a_core_fund_in_the_extension_tier(tmp_path, monkeypatch):
    import universe as U
    def no_net(*a, **k):
        raise OSError("offline test")
    monkeypatch.setattr(U, "_fetch", no_net)
    sp = [{"ticker": f"S{i}", "name": f"S{i}", "type": "stock", "sector": "Industrials", "index": "SP500", "tier": "core"}
          for i in range(450)]
    prev = tmp_path / "universe.json"
    prev.write_text(json.dumps({"items": sp}))
    big = {"cap": 50e9, "price": 60.0, "dv": 2e9, "sector": None}
    rows = ([{"ticker": "IBIT", "name": "iShares Bitcoin Trust ETF Shares", **big},       # the Nov 25 row
             {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", **big}]
            + [{"ticker": f"X{i}", "name": f"X{i}", **big} for i in range(2100)])
    items = U.build(previous=prev, mode="wide", today="2026-11-25", rows=rows)
    tks = [i["ticker"] for i in items]
    assert len(tks) == len(set(tks)), [t for t in set(tks) if tks.count(t) > 1]
    ibit = [i for i in items if i["ticker"] == "IBIT"]
    assert ibit == [{"ticker": "IBIT", "name": U.ETFS["IBIT"], "type": "etf", "sector": None, "index": "ETF", "tier": "core"}]
    assert not [i for i in items if i["ticker"] in U.ETFS and i.get("tier") == "ext"]
    assert U.dedupe([{"ticker": "A", "tier": "ext"}, {"ticker": "A", "tier": "core"}, {"ticker": "B"}]) == \
        [{"ticker": "A", "tier": "core"}, {"ticker": "B"}]
