"""Reward against risk.

Nov 17, 2026, the owner's challenge: "all your A+ setups have higher risk than reward". The record agreed: A+ setups
won 81% on 2022-now data with wins averaging +2.8% and losses -4.9% (payoff 0.58). A win rate bought with a near
target and a far stop is not an edge by itself, so every research test reports the payoff, ideas built with
reward >= risk are judged in a lane of their own, and the research runs design one such idea every day.
"""
import pathlib

import numpy as np
import pandas as pd

from engine import backtest as B
from engine import research as RS

SPLIT = "2022-01-01"


def _trades(n_days, per_day, win_share, win, loss, start, fill=100.0, stop0=98.0, target=102.0):
    days = pd.bdate_range(start, periods=n_days)
    rows = []
    for d in days:
        for k in range(per_day):
            w = k < round(win_share * per_day)
            rows.append({"signal_date": d, "filled": True, "ret": win if w else loss, "fill": fill, "stop": stop0,
                         "stop0": stop0, "target": target, "reason": "target" if w else "stop"})
    return pd.DataFrame(rows)


def test_every_test_reports_what_winners_make_against_what_losers_cost():
    # the classic A+ shape: target 1 away, stop 3.5 away; 80% wins of +2%, losses of -5%
    tr = pd.concat([_trades(100, 10, 0.8, 0.02, -0.05, "2012-01-02", stop0=96.5, target=101.0),
                    _trades(60, 10, 0.8, 0.02, -0.05, "2023-01-02", stop0=96.5, target=101.0)])
    po = RS.payoff_profile(tr, SPLIT)
    o = po["oos"]
    assert o["payoff"] == 0.4 and o["planned_rr"] == 0.29 and o["breakeven_win"] == 0.7143 and o["win_rate"] == 0.8
    assert round(o["cushion"], 3) == 0.086 and o["days"] == 60 and o["day_win"] == 0.8
    assert po["even"] is False and po["lane"] is None
    assert "wins average +2.00% and losses -5.00% (payoff 0.40; it needs 71% wins to break even and won 80%)" in po["sentence"]
    assert "the plan risks 1 to make 0.29" in po["sentence"]


def test_an_idea_built_with_reward_at_least_risk_is_judged_in_its_own_lane():
    even = dict(stop0=98.0, target=102.0)                      # stop 2 away, target 2 away
    aplus = pd.concat([_trades(80, 5, 0.8, 0.03, -0.03, "2012-01-02", **even), _trades(40, 5, 0.8, 0.03, -0.03, "2023-01-02", **even)])
    po = RS.payoff_profile(aplus, SPLIT)
    assert po["even"] and po["lane"] == "aplus" and "A+ quality for a setup of this kind" in po["sentence"]
    cand = pd.concat([_trades(80, 5, 0.6, 0.04, -0.03, "2012-01-02", **even), _trades(40, 5, 0.6, 0.04, -0.03, "2023-01-02", **even)])
    assert RS.payoff_profile(cand, SPLIT)["lane"] == "candidate"
    # 55% wins is not there, whatever the payoff
    weak = pd.concat([_trades(80, 20, 0.55, 0.04, -0.03, "2012-01-02", **even), _trades(40, 20, 0.55, 0.04, -0.03, "2023-01-02", **even)])
    assert RS.payoff_profile(weak, SPLIT)["lane"] == "no"
    # a crash week: a thousand trades on five days are not a record (Nov 17: 334 trades, 75% wins, 7 signal days)
    crash = pd.concat([_trades(80, 5, 0.8, 0.03, -0.03, "2012-01-02", **even), _trades(5, 200, 0.9, 0.03, -0.03, "2025-04-07", **even)])
    po = RS.payoff_profile(crash, SPLIT)
    assert po["oos"]["trades"] == 1000 and po["oos"]["days"] == 5 and po["lane"] == "no"
    assert "30+ signal days of unseen data" in po["sentence"]
    # wins smaller than losses: the plan says 1:1 but the record does not pay it
    thin = pd.concat([_trades(80, 5, 0.8, 0.02, -0.03, "2012-01-02", **even), _trades(40, 5, 0.8, 0.02, -0.03, "2023-01-02", **even)])
    assert RS.payoff_profile(thin, SPLIT)["lane"] == "no"
    # by trade it wins 80%, but most days lose: every signal day counts once
    lumpy = pd.concat([_trades(80, 5, 0.8, 0.03, -0.03, "2012-01-02", **even),
                       _trades(1, 400, 1.0, 0.03, -0.03, "2023-01-02", **even), _trades(39, 2, 0.5, 0.03, -0.03, "2023-02-01", **even)])
    po = RS.payoff_profile(lumpy, SPLIT)
    assert po["oos"]["win_rate"] > 0.9 and po["oos"]["day_win"] < 0.55 and po["lane"] == "no"


def test_the_research_runs_design_one_such_idea_every_day():
    t = RS.payoff_track()
    assert "At least one of the new ideas you design today must be built with a reward at least as large as its risk" in t["every_day"]
    assert "70%+ wins in both periods" in t["how_it_is_judged"] and "60%+ a candidate" in t["how_it_is_judged"]
    assert any("payoff 0.58" in x for x in t["what_the_desk_knows"]) and any("1 ATR stop" in x for x in t["what_the_desk_knows"])
    assert "A tighter stop on an existing idea is not a new idea" in t["look_for"]
    src = pathlib.Path(RS.__file__).read_text()
    assert 'res["reward_at_least_risk"] = track' in src and 'focus["reward_at_least_risk"] =' in src      # printed by `research next`
    assert '"payoff": {k: v for k, v in po.items() if k != "sentence"}' in src and 'summary += po["sentence"]' in src


def test_a_winner_whose_stop_was_raised_keeps_its_planned_risk_and_its_place_in_the_replay():
    """The account replay sized every trade on fill minus stop, read after position management. A winner whose stop
    had been raised above its fill had no risk left and was dropped: 422 trades in the live record, all winners."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mr", pathlib.Path(__file__).with_name("test_management_and_research.py"))
    mr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mr)
    # fill at 100, a close 2.5 ATR higher (ATR 2) moves the stop to 102, the next open at 101 takes it out: +1%
    closes = [100, 100, 105, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    opens = [100, 101, 104, 101, 101, 101, 101, 101, 101, 101, 101, 101]
    lows = [99, 99, 103, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    ind = mr._frames(closes, opens, lows=lows)
    P = mr.S.params_with_defaults(None)
    tr = B.simulate(mr._plan(ind, 0, 100.0, hold=6), ind, P)
    r = tr.iloc[0]
    assert r["fill"] == 100.0 and r["stop0"] == 94.0 and r["stop"] > r["fill"] and r["ret"] > 0     # planned stop 3 ATR below
    assert np.isfinite(r["r_mult"]) and round(r["r_mult"], 3) == round((r["exit"] - 100.0) / 6.0, 3)   # R on the planned risk
    tr = tr.assign(score=80.0)
    pf = B.portfolio(tr, 5000, 7, 2.0, 14.0)
    assert pf["trades"] == 1 and pf["final"] > 5000
    # the same trade without the planned stop is what used to happen: no risk, dropped
    assert B.portfolio(tr.drop(columns="stop0"), 5000, 7, 2.0, 14.0).get("trades", 0) == 0
    # a trade list without the planned stop but with a normal stop (older data) is sized as before
    assert B.portfolio(tr.drop(columns="stop0").assign(stop=94.0), 5000, 7, 2.0, 14.0)["trades"] == 1
