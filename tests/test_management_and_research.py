"""Nov 7 audit: live position-management rules are replayed in the backtest, the shadow grader matches
the backtest, the research lab keeps a curated backlog and records variants, and grading reports its
honesty checks (by year and a time split)."""
import json
import pathlib

import numpy as np
import pandas as pd

from engine import backtest as B
from engine import calibrate as CB
from engine import monitor as MON
from engine import rules as R
from engine import strategies as S
from engine import research as RS

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _frames(closes, opens=None, highs=None, lows=None, start="2026-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.DataFrame({"X": closes}, index=idx, dtype=float)
    o = pd.DataFrame({"X": opens if opens is not None else closes}, index=idx, dtype=float)
    h = pd.DataFrame({"X": highs if highs is not None else np.maximum(o["X"], c["X"]) + 0.5}, index=idx, dtype=float)
    l = pd.DataFrame({"X": lows if lows is not None else np.minimum(o["X"], c["X"]) - 0.5}, index=idx, dtype=float)
    z = c * 0
    ind = {"open": o, "high": h, "low": l, "close": c, "volume": z + 1e6, "atr10": z + 2.0, "sma5": c.rolling(5, min_periods=1).mean(),
           "sma50": z + 50.0, "sma200": z + 50.0, "rsi2": z + 50.0, "max7c": c.rolling(7, min_periods=1).max(),
           "dist200": z, "slope200": z, "atr_pct": z + 0.02, "rs_pct": z + 0.5, "ibs": z + 0.5, "streak": z, "adx14": z + 20,
           "vol_ratio": z + 1, "dist50": z, "ret5": z, "sector_up": z + 1, "is_etf": z.astype(bool), "is_stock": z.astype(bool) | True,
           "is_lev": z.astype(bool), "market": pd.DataFrame({"spy_above200": 1.0, "spy_dist200": 0.05, "spy_ret5": 0.0, "breadth50": 0.6,
                                                              "vix": 15.0, "vix_ratio": 1.0, "spy": 500.0}, index=idx)}
    return ind


def _plan(ind, signal_row, limit, hold=10):
    mask = ind["close"] * 0 == 1
    mask.iloc[signal_row, 0] = True
    lim = ind["close"] * 0 + limit
    return {"name": "T", "entry_kind": "limit_next", "valid_days": 1, "max_hold": hold, "dyn_exit": "none",
            "stop": {"kind": "atr", "mult": 3.0}, "target": {"kind": "atr", "mult": 10.0}, "mask": mask, "limit": lim}


def test_lock_in_rule_is_replayed_in_the_backtest():
    # fill at 100, close jumps 2.5 ATR (ATR 2) -> stop moves to 102 -> next open 101 stops it out
    closes = [100, 100, 105, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    opens = [100, 101, 104, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    lows = [99, 99, 103, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    ind = _frames(closes, opens, lows=lows)
    P = S.params_with_defaults(None)
    tr = B.simulate(_plan(ind, 0, 100.0, hold=6), ind, P)
    assert tr.iloc[0]["reason"] == "stop" and abs(tr.iloc[0]["exit"] - 101 * (1 - 0.0005)) < 1e-6
    tr2 = B.simulate(_plan(ind, 0, 100.0, hold=6), ind, P, manage={"lock_atr": 0})
    assert tr2.iloc[0]["reason"] == "time"
    assert P["manage"]["be_atr"] == 0  # no breakeven stop: it cost about a point of A+ win rate when measured


def test_a_touch_of_the_limit_is_not_a_fill():
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    lows = [99, 100.0, 101, 102, 103, 104, 105, 106, 107, 108]  # day-1 low exactly at the limit
    ind = _frames(closes, lows=lows)
    P = S.params_with_defaults(None)
    assert P["manage"]["fill"] == "through"
    tr = B.simulate(_plan(ind, 0, 100.0), ind, P, include_unfilled=True)
    assert tr.iloc[0]["reason"] == "unfilled"
    tr2 = B.simulate(_plan(ind, 0, 100.0), ind, P, include_unfilled=True, manage={"fill": "touch"})
    assert tr2.iloc[0]["filled"]


def test_shadow_grader_matches_the_backtest_rules():
    closes = [100, 99, 103, 104, 110, 104, 104, 104, 104, 104]
    lows = [99, 98, 102, 103, 109, 103, 103, 103, 103, 103]
    ind = _frames(closes, lows=lows)
    ind["rsi2"].iloc[3, 0] = 80
    base = {"id": "a", "ticker": "X", "strategy": "T", "valid_for": str(ind["close"].index[1].date()),
            "entry": 99.0, "stop": 90.0, "target": 150.0, "hold_max_days": 8}
    r = CB.shadow_outcome({**base, "dyn_exit": "rsi2_70"}, ind)
    assert r["status"] == "closed" and r["reason"] == "signal" and r["exit"] == 104  # RSI(2) above 70 on day 3
    r = CB.shadow_outcome({**base, "dyn_exit": "high7"}, ind)
    assert r["status"] == "closed" and r["reason"] == "signal" and r["exit"] == 103  # 7-day closing high on day 2
    r = CB.shadow_outcome({**base, "dyn_exit": "none", "entry": 98.0}, ind)
    assert r["status"] == "unfilled"  # the low only touched 98
    r = CB.shadow_outcome({**base, "dyn_exit": "none", "entry": 98.0, "valid_days": 2}, ind)
    assert r["status"] == "unfilled"


def _trade(**kw):
    tr = {"id": "t1", "ticker": "X", "entry_price": 100.0, "shares": 1, "stop": 94.0, "target": 120.0, "max_hold": 8,
          "dyn_exit": "none", "strategy": "MR_STRETCH", "entry_date": "2026-01-02"}
    tr.update(kw)
    return tr


def test_monitor_stop_management_and_trend_break():
    P = S.params_with_defaults(None)
    ind = _frames([100, 102.5, 102.5])  # up 1.25 ATR: no breakeven nag any more
    codes = [s["code"] for s in MON.check_trade(_trade(), ind, "postclose", {}, None, P, False)["signals"]]
    assert "BREAKEVEN" not in codes and "RAISE_STOP" not in codes
    ind = _frames([100, 104.5, 104.5])  # up 2.25 ATR: lock in entry + 1 ATR
    sig = [s for s in MON.check_trade(_trade(), ind, "postclose", {}, None, P, False)["signals"] if s["code"] == "RAISE_STOP"]
    assert sig and sig[0]["new_stop"] == 102.0
    # a close below the 50-day ends trend-following setups, not the dip-buying leader pullback (measured: it hurt)
    ind = _frames([100, 45, 45])
    codes = [s["code"] for s in MON.check_trade(_trade(strategy="LEADER_PULLBACK"), ind, "postclose", {}, None, P, False)["signals"]]
    assert "TREND_BREAK" not in codes
    codes = [s["code"] for s in MON.check_trade(_trade(strategy="TREND_PULLBACK"), ind, "postclose", {}, None, P, False)["signals"]]
    assert "TREND_BREAK" in codes


def test_grading_reports_by_year_and_time_split():
    rng = np.random.default_rng(1)
    n = 6000
    dates = pd.to_datetime(rng.choice(pd.bdate_range("2022-01-03", "2026-11-05").values, n))
    score = rng.uniform(40, 100, n)
    p = rng.uniform(0.6, 0.8, n)
    win = rng.random(n) < (0.55 + 0.3 * (score - 40) / 60)
    ret = np.where(win, rng.uniform(0.005, 0.05, n), -rng.uniform(0.005, 0.04, n))
    tr = pd.DataFrame({"signal_date": dates, "ticker": rng.choice(list("ABCDEFGH"), n), "score": score, "p": p,
                       "ret": ret, "filled": True})
    G = CB.calibrate_grades(tr, pd.Timestamp("2022-01-01"))
    assert "by_year" in G and {r["year"] for r in G["by_year"]} >= {2022, 2026}
    ts = G["time_split"]
    assert ts["chosen_bar_early"]["window"][0].startswith("2022") and ts["chosen_bar_late"]["window"][0].startswith("2024")
    assert ts["early_selection"] is None or "early_bar_on_late" in ts


def test_seed_backlog_is_valid_distinct_and_consumed_in_order(tmp_path):
    seen = set()
    for f in RS.SEED_FILES:
        for sp in json.loads(f.read_text()):
            assert R.validate(sp) == [], sp["id"]
            assert sp["id"] not in seen, sp["id"]
            seen.add(sp["id"])
    assert len(RS.SEED_FILES) >= 3
    first = json.loads(RS.SEED_FILES[0].read_text())
    known = {s["id"]: {"spec": s, "signature": R.signature(s), "label": s["label"]} for s in first}
    backlog = RS.seed_backlog(known)
    assert backlog and backlog[0]["id"] not in known
    # a variant is detected even when its thresholds differ
    v2 = [s for s in json.loads(RS.SEED_FILES[-1].read_text()) if s["id"] == "R_ETF_3DAY_HL_V2"][0]
    sim = RS.similar_ideas(v2, known)
    assert sim and sim[0]["id"] == "R_ETF_3DAY_HL"


def test_research_next_offers_the_backlog_when_the_queue_is_empty(tmp_path):
    st = tmp_path / "state"
    (st / "research").mkdir(parents=True)
    (st / "research_results").mkdir()
    RS.cmd_next(str(st), str(tmp_path / "out"))
    nx = json.loads((tmp_path / "out" / "next.json").read_text())
    assert nx["from_backlog"] and nx["next"] == nx["backlog"][0]
    spec = json.loads(pathlib.Path(nx["spec_file"]).read_text())
    assert spec["id"] == nx["next"] and R.validate(spec) == []
