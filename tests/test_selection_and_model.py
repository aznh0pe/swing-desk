"""Evening of Nov 14, 2026: the bear-market input of the win-odds model, the A+ bar chosen on the whole desk, the
two-stage top-up floors, the research gate's volume rule, the faster top-up replay and the own-design queue.

Synthetic data only: these tests check the rules, not any market conclusion."""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import calibrate as CB  # noqa: E402
from engine import data as D  # noqa: E402
from engine import indicators as I  # noqa: E402
from engine import model as M  # noqa: E402
from engine import research as RS  # noqa: E402
from engine import rules as R  # noqa: E402
from engine import strategies as S  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synth.py"), str(d)], check=True, capture_output=True)
    return d


# ----------------------------------------------------------------------------- bear-market age
def test_bear_age_counts_sessions_in_a_row_under_the_200_day(synth):
    ind = I.compute(D.load_history(synth), D.load_universe(synth))
    mk = ind["market"]
    age, spy = mk["spy_bear_age"], mk["spy"]
    sma = spy.rolling(200, min_periods=100).mean()
    below = (spy <= sma) & sma.notna()
    assert (age[~below] == 0).all() and (age[below] >= 1).all()
    step = age.diff().dropna()
    assert set(step[below.loc[step.index]].unique()) <= {1.0}          # one more each session under the average
    assert "spy_bear_age" in S.FEATURES and "spy_bear_age" in R.MARKET_NAMES
    f = S.features_at(ind, np.array([len(spy) - 1]), np.array([0]))
    assert float(f["spy_bear_age"].iloc[0]) == float(age.iloc[-1])


def test_the_bear_age_feeds_strategy_rules_but_not_the_win_odds_model():
    """Tested Nov 14, 2026 and left out of the model: before 2022 old bear markets rewarded dip-buying (80% of 1,227
    raw trades, seven episodes of seven), 2022 did not (48.5%). The feature exists for strategy rules only."""
    assert not any("bear" in c for c in M.POOLED)
    assert "spy_bear_age" not in M.LOGIT_FEATURES
    rng = np.random.default_rng(5)
    n = 600
    t = pd.DataFrame({"signal_date": pd.bdate_range("2012-01-02", periods=n), "ticker": "T", "strategy": "S1", "filled": True,
                      "ret": np.where(rng.random(n) < 0.7, 0.03, -0.03), **{c: rng.normal(0, 1, n) for c in S.FEATURES}})
    pm = M.fit_pooled(t, ["S1"], pd.Timestamp("2013-06-01"))
    assert not any("bear" in c for c in pm["features"])
    a = t.iloc[:50]
    p1, _, _ = M.predict_pooled(pm, a[S.FEATURES], a["strategy"].values)
    p2, _, _ = M.predict_pooled(pm, a[S.FEATURES].assign(spy_bear_age=120.0), a["strategy"].values)
    p3, _, _ = M.predict_pooled(pm, a[S.FEATURES].drop(columns=["spy_bear_age"]), a["strategy"].values)   # a frame saved before the feature
    assert np.allclose(p1, p2) and np.allclose(p1, p3)
    assert M.fit(t.assign(score=1.0), min_trades=100)["features"] == M.LOGIT_FEATURES


# ----------------------------------------------------------------------------- top-up replay and floors
def _signals(seed=3, weeks=80, ap_weeks=0.5, band_wins=7, band_ret=0.03, band_loss=0.03):
    """Some weeks carry A+ setups; every day has setups in the band below them, `band_wins` of every 10 winning."""
    rng = np.random.default_rng(seed)
    rows, j = [], 0
    for w in range(weeks):
        mon = pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=w)
        has_ap = rng.random() < ap_weeks
        for dow in range(5):
            d = mon + pd.Timedelta(days=dow)
            j += 1
            for k in range(6):
                is_ap = has_ap and k < 2
                sc = float(rng.uniform(86, 99) if is_ap else rng.uniform(50, 80))
                win = (rng.random() < 0.85) if is_ap else ((j * 7) % 10 < band_wins)   # the band wins or loses by the day
                rows.append({"signal_date": d, "ticker": f"T{k}", "strategy": "A" if k % 2 else "B", "score": sc, "p": 0.80 if is_ap else 0.66,
                             "filled": True, "ret": (0.03 if is_ap else band_ret) if win else -(0.03 if is_ap else band_loss),
                             "fill": 100.0, "stop": 94.0, "exit_date": d + pd.Timedelta(days=4)})
    return pd.DataFrame(rows)


def test_fast_top_up_replay_matches_the_day_by_day_rule():
    s = CB._one_per_ticker(_signals())
    pk, wk = CB.simulate_topups(s, 84, 0.76, 60, 0.6, 5, excluded=["B"])
    # replay it the slow way
    ap = (s["score"] >= 84) & (s["p"] >= 0.76)
    band = s[~ap & (s["score"] >= 60) & (s["p"] >= 0.6) & (s["strategy"] != "B")]
    want, weeks, so_far, cur = [], {}, 0, None
    for d in sorted(s["signal_date"].unique()):
        d = pd.Timestamp(d)
        w = d - pd.Timedelta(days=d.weekday())
        if w != cur:
            cur, so_far = w, 0
        na = int(ap[s["signal_date"] == d].sum())
        need = max(0, min(5, d.weekday() + 1) - so_far - na)
        g = band[band["signal_date"] == d].head(need)
        want += list(g.index)
        so_far += na + len(g)
        weeks[w] = so_far
    assert list(pk.index) == want and wk.to_dict() == {k: float(v) for k, v in weeks.items()}
    assert set(pk["strategy"]) == {"A"} and (pk["score"] < 84).all()
    empty_pk, empty_wk = CB.simulate_topups(s.iloc[0:0], 84, 0.76, 60, 0.6, 5)
    assert empty_pk.empty and empty_wk.empty


def test_top_up_floors_relax_only_when_the_strict_ones_cannot_keep_the_week_on_pace(monkeypatch):
    monkeypatch.setattr(CB, "TOPUP_MIN_TRADES", 60)
    # top-ups win 70% and earn well: the strict floors pass, nothing is relaxed
    good = CB.calibrate_topups(_signals(), 84, 0.76, 5, None)
    assert good["topup_ok"] and good["topup_floors"] == "strict" and good["topup_note"] == ""
    # top-ups win 70% but earn little (profit factor about 1.2): the strict floor would switch them off
    thin = CB.calibrate_topups(_signals(band_ret=0.05, band_loss=0.097), 84, 0.76, 5, None)
    assert thin["topup_ok"] and thin["topup_floors"] == "soft" and 1.10 <= thin["oos_topup_pf"] < 1.25
    assert thin["oos_weeks_at_min"] >= 0.9 and "softer floors" in thin["topup_note"]
    # top-ups that lose money never pass, whatever the pace
    bad = CB.calibrate_topups(_signals(band_wins=6, band_ret=0.012), 84, 0.76, 5, None)
    assert not bad["topup_ok"]


# ----------------------------------------------------------------------------- research gate
def test_research_gate_compares_the_desks_when_both_sides_were_replayed():
    row = lambda aw, sw, wk, cg, dd: {"aplus_bar_growth": [{"chosen": True, "aplus_win": aw, "sent_win": sw, "weeks_at_min": wk,
                                                             "account_cagr": cg, "account_max_dd": dd}]}
    base = row(0.706, 0.707, 0.858, 0.127, -0.161)
    assert CB.research_gate_ok(row(0.770, 0.766, 0.945, 0.354, -0.146), base)            # better on every line
    assert not CB.research_gate_ok(row(0.690, 0.766, 0.945, 0.354, -0.146), base)        # the A+ win rate fell
    assert not CB.research_gate_ok(row(0.770, 0.766, 0.800, 0.354, -0.146), base)        # the weekly minimum fell 5.8 points
    assert not CB.research_gate_ok(row(0.770, 0.700, 0.945, 0.354, -0.146), base)        # everything sent won less often
    # growth and drawdown are not this gate's business: the final live-versus-recalibrated check judges them
    assert CB.research_gate_ok(row(0.751, 0.759, 0.941, 0.356, -0.144), row(0.744, 0.750, 0.283, 0.165, -0.105))
    # the desk rows win over the A+ counts: the same numbers with a smaller A+ set still pass
    more = {**row(0.770, 0.766, 0.945, 0.354, -0.146), "oos_aplus_win_rate": 0.77, "oos_aplus_trades": 100, "oos_aplus_signals_per_week": 1.0}
    assert CB.research_gate_ok(more, {**base, "oos_aplus_win_rate": 0.706, "oos_aplus_trades": 9000, "oos_aplus_signals_per_week": 40.0})


def test_research_gate_does_not_veto_a_smaller_set_that_wins_clearly_more_and_earns_as_much():
    """Nov 14, 2026: with the new model the A+ set with the research strategies won 85.5% (+3.41% a trade, 1,324
    trades) against 73.7% (+1.21%, 2,837) without them, and the gate dropped all of them for sending fewer setups."""
    g = lambda w, a, n, pw: {"oos_aplus_win_rate": w, "oos_aplus_avg_ret": a, "oos_aplus_trades": n, "oos_aplus_signals_per_week": pw}
    base = g(0.7367, 0.01206, 2837, 10.7)
    assert CB.research_gate_ok(g(0.855, 0.03413, 1324, 4.9), base)
    assert not CB.research_gate_ok(g(0.750, 0.03413, 1324, 4.9), base)      # fewer setups and under 2 points more wins
    assert not CB.research_gate_ok(g(0.855, 0.0200, 1324, 4.9), base)       # fewer setups and a smaller total return
    assert CB.research_gate_ok(g(0.750, 0.0125, 2700, 10.0), base)          # about as many setups: the old rule


# ----------------------------------------------------------------------------- own designs reach the research log
def test_own_designs_in_the_repo_are_queued_logged_and_credited(tmp_path, monkeypatch):
    own = json.loads(RS.OWN_FILE.read_text())
    assert own and all(not RS.check_new(sp, {}) for sp in own), "every own design must be a valid, new spec"
    assert len({sp["id"] for sp in own}) == len(own)
    seeds = {sp["id"] for f in RS.SEED_FILES for sp in json.loads(f.read_text())}
    assert not seeds & {sp["id"] for sp in own} and RS.OWN_FILE not in RS.SEED_FILES
    two = own[:2]
    f = tmp_path / "own.json"
    f.write_text(json.dumps(two))
    monkeypatch.setattr(RS, "OWN_FILE", f)
    assert [sp["id"] for sp in RS.own_queue({})] == [two[0]["id"], two[1]["id"]]
    assert [sp["id"] for sp in RS.own_queue({two[0]["id"]: {"signature": R.signature(two[0])}})] == [two[1]["id"]]   # logged ones leave the queue
    state = tmp_path / "state"
    state.mkdir()
    seen = {}
    monkeypatch.setattr(RS, "run_tests", lambda specs, data, st, out, split, by, **kw: seen.update(specs=specs, by=by, **kw))
    monkeypatch.setattr(sys, "argv", ["research", "next", "--state", str(state), "--out", str(tmp_path / "n")])
    RS.main()
    nxt = json.loads((tmp_path / "n" / "next.json").read_text())
    assert nxt["next"] == two[0]["id"] and nxt["queued"] == [two[0]["id"], two[1]["id"]] == nxt["own_queue"] and "spec_file" not in nxt
    monkeypatch.setattr(sys, "argv", ["research", "test", "--queued", "5", "--data", "data", "--state", str(state), "--out", str(tmp_path / "t"), "--by", "checks run"])
    RS.main()
    assert [sp["id"] for sp in seen["specs"]] == [two[0]["id"], two[1]["id"]] == [sp["id"] for sp in seen["ideas_to_add"]]
    assert seen["by"] == "checks run" and seen["by_idea"] == {two[0]["id"]: RS.OWN_BY, two[1]["id"]: RS.OWN_BY}
    monkeypatch.setattr(sys, "argv", ["research", "test", "--id", two[1]["id"], "--data", "data", "--state", str(state), "--out", str(tmp_path / "t2")])
    RS.main()
    assert [sp["id"] for sp in seen["specs"]] == [two[1]["id"]] and seen["by_idea"] == {two[1]["id"]: RS.OWN_BY}


def test_beta_and_downside_capture_describe_how_a_ticker_moves_with_the_market():
    """Nov 16, 2026: in bear-market selloffs the dips of high-capture stocks bounce most (both periods); the two
    frames are rule inputs and state-scan dimensions, not model features."""
    import numpy as np
    import pandas as pd
    from engine import indicators as I, rules as R, research as RS, strategies as S, model as M
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=420)
    m = rng.normal(0.0003, 0.012, len(idx))
    px = {"SPY": 100 * np.cumprod(1 + m), "HIGH": 50 * np.cumprod(1 + 1.8 * m + rng.normal(0, 0.002, len(idx))),
          "LOW": 80 * np.cumprod(1 + 0.4 * m + rng.normal(0, 0.002, len(idx)))}
    c = pd.DataFrame(px, index=idx)
    w = {"close": c, "open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "volume": c * 0 + 1e6}
    ind = I.compute(w)
    b, d = ind["beta250"].iloc[-1], ind["dcap250"].iloc[-1]
    assert abs(b["SPY"] - 1) < 0.02 and 1.6 < b["HIGH"] < 2.0 and 0.25 < b["LOW"] < 0.55
    assert abs(d["SPY"] - 1) < 0.02 and d["HIGH"] > 1.5 > 0.7 > d["LOW"]
    assert ind["beta250"].iloc[:100].isna().all().all()                  # no value before there is a history
    assert {"beta250", "dcap250", "attn"} <= R.FRAME_NAMES and RS.STATE_STOCK_EXTRA == ("beta250", "dcap250", "attn")
    a = ind["attn"].iloc[-1]                                              # steady volume: trading at its own norm
    assert abs(a["SPY"] - 1) < 0.25 and "attn" not in S.FEATURES
    assert not {"beta250", "dcap250"} & set(S.FEATURES) and not {"beta250", "dcap250"} & set(M.LOGIT_FEATURES)
    own = {x["id"] for x in __import__("json").load(open(RS.OWN_FILE))} if isinstance(__import__("json").load(open(RS.OWN_FILE)), list) else set()
    assert not own or {"R_BEAR_ANY_HIGH_CAPTURE_DIP", "R_SELLOFF_HIGH_BETA_LEADER"} <= own
