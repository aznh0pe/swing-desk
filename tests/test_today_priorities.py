"""The desk's two standing research targets (owner, Nov 22, 2026): strategies that hold up in every market condition,
and strategies that win in the condition the market is in today. Both need today's condition to be right, every day.

What went wrong while building it: the research test worked the condition out from its own price history, which is
rebuilt once a week. On Nov 22 that file ended Nov 16 (a bull-market pullback) while the market had been in a
bear-market selloff since Nov 19: the day's research would have been aimed at the wrong market."""
import datetime as dt
import json
import pathlib
import sys
import zoneinfo

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import backtest as B  # noqa: E402
from engine import data as D  # noqa: E402
from engine import research as RS  # noqa: E402
from engine import rules as R  # noqa: E402
from engine import strategies as S  # noqa: E402

PT = zoneinfo.ZoneInfo("America/Los_Angeles")


def _closes(n=320, drop=4):
    """A steady climb, then a crash over the last `drop` sessions that takes SPY under its 200-day average."""
    idx = pd.bdate_range(end="2026-11-20", periods=n)
    spy = np.linspace(500, 700, n)
    spy[-drop:] = spy[-drop - 1] * np.array([0.93, 0.87, 0.82, 0.78][:drop])
    vix = np.full(n, 15.0)
    vix[-drop:] = [24, 30, 34, 37][:drop]
    return pd.DataFrame({"SPY": spy, "^VIX": vix, "AAA": 50.0}, index=idx)


def _market(c):
    spy = c["SPY"]
    sma = spy.rolling(200, min_periods=150).mean()
    vix = c["^VIX"]
    return pd.DataFrame({"spy_above200": (spy > sma).astype(float).where(sma.notna()), "spy_ret5": spy / spy.shift(5) - 1,
                         "vix_ratio": vix / vix.rolling(20, min_periods=10).mean(), "spy_dist200": spy / sma - 1}, index=c.index)


def test_todays_condition_comes_from_the_daily_file_not_the_weekly_history(monkeypatch, tmp_path):
    c = _closes()
    monkeypatch.setattr(D, "load_daily", lambda d: {"close": c})
    stale = {"market": _market(c.iloc[:-4])}                      # the weekly history: last bar four sessions ago
    assert RS._condition_from_frame(stale["market"], "test history")["condition"] == "bull_calm"
    now = RS.market_now(None, stale, tmp_path, now=dt.datetime(2026, 11, 22, 22, 0, tzinfo=PT))
    assert now["condition"] == "bear_stress" and now["source"] == "daily prices" and now["asof"] == "2026-11-20"
    assert now["days"] >= 1 and now["label"] == S.CONDITION_LABEL["bear_stress"]
    assert any("test history ends 2026-11-16" in s for s in now["older_sources_disagree"])
    assert "warning" not in now                                   # Friday's bar is the latest finished session on a Sunday
    assert now["market"]["spy_vs_200d_pct"] < 0 and now["market"]["vix_vs_20d"] > 1.1


def test_a_stale_daily_file_is_said_out_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "load_daily", lambda d: {"close": _closes()})
    now = RS.market_now(None, None, tmp_path, now=dt.datetime(2026, 11, 24, 15, 12, tzinfo=PT))   # Tuesday after the close
    assert now["sessions_behind"] == 2 and "ISSUES" in now["warning"] and "2026-11-20" in now["warning"]


def _scan(day, cond, started):
    return {"doc": {"status": "ok", "started_at": started, "data": {"last_bar": day},
                    "market": {"condition": cond, "label": "Correction", "vix": 37.2, "spy_vs_200d_pct": -5.4}}}


def test_without_prices_the_newest_scan_decides_and_the_newest_source_always_wins(monkeypatch, tmp_path):
    st = {"scans": {"a": _scan("2026-11-18", "bull_stress", "2026-11-18T14:05"), "b": _scan("2026-11-19", "bear_stress", "2026-11-19T14:05"),
                    "c": _scan("2026-11-20", "bear_stress", "2026-11-20T12:25"), "d": _scan("2026-11-20", "bear_stress", "2026-11-20T14:06")}}
    monkeypatch.setattr(RS, "load_state", lambda d: st)
    now = RS.market_now(tmp_path)
    assert (now["condition"], now["source"], now["asof"], now["days"]) == ("bear_stress", "newest scan", "2026-11-20", 2)
    # a history that ends earlier never overrides it
    assert RS.market_now(tmp_path, {"market": _market(_closes().iloc[:-4])})["condition"] == "bear_stress"
    # and when the daily file is as new as the scan, the prices decide and the scan's VIX and breadth are kept alongside
    monkeypatch.setattr(D, "load_daily", lambda d: {"close": _closes()})
    both = RS.market_now(tmp_path, None, tmp_path)
    assert both["source"] == "daily prices" and both["market"]["vix"] == 37.2 and both["market"]["label"] == "Correction"
    # a daily file too short for a 200-day average gives no answer of its own: the scan still decides
    monkeypatch.setattr(D, "load_daily", lambda d: {"close": _closes().iloc[-40:]})
    assert RS.market_now(tmp_path, None, tmp_path)["source"] == "newest scan"


def _per(win, n=200, pf=1.8):
    return {"trades": n, "win_rate": win, "profit_factor": pf}


def test_an_idea_is_ranked_all_weather_first_then_todays_condition_then_the_rest():
    good = {"bear_stress": {"train": _per(0.74), "oos": _per(0.78)}}
    assert RS.today_fit(good, {"all_weather": True}, "bear_stress")["priority"] == 1
    f = RS.today_fit(good, {"all_weather": False}, "bear_stress")
    assert f["priority"] == 2 and f["fits_today"] and "it works here" in f["sentence"] and "bear-market selloff" in f["sentence"]
    weak = RS.today_fit({"bear_stress": {"train": _per(0.74), "oos": _per(0.66)}}, None, "bear_stress")
    assert weak["priority"] == 3 and "not strong enough" in weak["sentence"]
    thin = RS.today_fit({"bear_stress": {"train": _per(0.9, 30), "oos": _per(0.9, 10)}}, None, "bear_stress")
    assert thin["priority"] == 3 and not thin["enough_trades"] and "too few trades" in thin["sentence"]
    elsewhere = RS.today_fit({"bull_calm": {"train": _per(0.8), "oos": _per(0.8)}}, None, "bear_stress")
    assert elsewhere["priority"] == 3                              # winning in another condition is not winning today
    assert RS.today_fit(good, None, None)["priority"] == 3 and "sentence" not in RS.today_fit(good, None, None)


def test_the_promotion_check_takes_all_weather_then_todays_candidates_first(tmp_path, monkeypatch):
    st = tmp_path / "state"
    (st / "research").mkdir(parents=True)
    (st / "research_results").mkdir()
    spec = lambda i: {"id": i, "label": i, "family": "x", "about": "x", "conditions": ["close > open"],
                      "entry": {"kind": "limit_next", "price": "close"}, "stop": {"atr": 3}, "target": {"atr": 3}}
    rows = {"R_BEST_ELSEWHERE": (0.86, {"bull_calm": {"train": _per(0.86), "oos": _per(0.86)}}, False),
            "R_TODAY": (0.74, {"bear_stress": {"train": _per(0.74), "oos": _per(0.74)}}, False),
            "R_ALL_WEATHER": (0.70, {c: {"train": _per(0.70), "oos": _per(0.70)} for c in S.CONDITIONS}, True)}
    for i, (w, bc, aw) in rows.items():
        (st / "research" / f"{i}.json").write_text(json.dumps({"id": i, "spec": spec(i)}))
        (st / "research_results" / f"{i}-1.json").write_text(json.dumps(
            {"id": f"{i}-1", "idea_id": i, "verdict": "candidate", "tested_at": "2026-11-20T15:00:00-08:00",
             "oos": {"win_rate": w}, "by_condition": bc, "all_weather": {"all_weather": aw}}))
    monkeypatch.setattr(RS, "market_now", lambda *a, **k: {"condition": "bear_stress"})
    assert list(RS.candidate_specs(st, {}, cap=3)) == ["R_ALL_WEATHER", "R_TODAY", "R_BEST_ELSEWHERE"]
    assert list(RS.candidate_specs(st, {}, cap=2)) == ["R_ALL_WEATHER", "R_TODAY"]     # the highest raw win rate waits its turn
    monkeypatch.setattr(RS, "market_now", lambda *a, **k: None)                         # no condition known: plain win-rate order
    assert list(RS.candidate_specs(st, {}, cap=3))[:2] == ["R_ALL_WEATHER", "R_BEST_ELSEWHERE"]


def _trades():
    """4,800 trades over four market conditions. A low `rsi2` wins 90% in bear-market selloffs only (50% elsewhere);
    a high `adx14` wins 80% in every condition; everything else is a coin flip."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2012-01-02", "2026-11-13")
    shape = {"bull_calm": (1.0, 0.95, 0.01), "bull_stress": (1.0, 1.3, -0.02), "bear_calm": (0.0, 0.95, 0.0), "bear_stress": (0.0, 1.4, -0.04)}
    rows = []
    for k, d in enumerate(days[::2]):
        cond = S.CONDITIONS[k % 4]
        up, vr, r5 = shape[cond]
        for j in range(3):
            rsi, adx = rng.uniform(0, 30), rng.uniform(10, 60)
            p = 0.5
            if adx >= 50:
                p = 0.8
            elif cond == "bear_stress" and rsi < 3:
                p = 0.9
            rows.append({"signal_date": d, "ticker": f"T{j}", "filled": True, "ret": 0.03 if rng.random() < p else -0.03,
                         "spy_above200": up, "vix_ratio": vr, "spy_ret5": r5, "rsi2": rsi, "adx14": adx})
    return pd.DataFrame(rows)


def test_the_states_scan_shows_what_wins_today_and_what_wins_everywhere(monkeypatch):
    t = _trades()
    monkeypatch.setattr(R, "build_plan", lambda *a, **k: None)
    monkeypatch.setattr(B, "simulate", lambda *a, **k: t.copy())
    monkeypatch.setattr(S, "base_masks", lambda *a, **k: None)
    ind = {"market": pd.DataFrame(index=t["signal_date"].unique())}
    out = RS.state_scan({"params": {}}, ind, {}, "2022-01-01", today={"condition": "bear_stress"})
    assert out["today"]["condition"] == "bear_stress" and out["today"]["label"] == S.CONDITION_LABEL["bear_stress"]
    assert set(out["by_condition"]) == set(S.CONDITIONS) and out["by_condition"]["bull_calm"]["oos"]["n"] > 100
    td = [(r["state"], r["hi"]) for r in out["today_standouts"]]
    assert any(s == "rsi2" and hi <= 3.5 for s, hi in td), td                 # the oversold bucket wins in today's condition
    aw = out["all_weather_standouts"]
    assert aw and aw[0]["state"] == "adx14" and aw[0]["lo"] >= 49 and aw[0]["worst_win"] >= 0.68
    assert len(aw[0]["conditions"]) >= 3 and not any(r["state"] == "rsi2" for r in aw)   # winning in one condition is not all-weather
    assert "today_standouts" in out["how_to_read"] and "all_weather_standouts" in out["how_to_read"]
    # with another condition today, the bear-selloff bucket is not offered as today's idea
    calm = RS.state_scan({"params": {}}, ind, {}, "2022-01-01", today={"condition": "bull_calm"})
    assert not any(r["state"] == "rsi2" for r in calm["today_standouts"])


def test_the_days_plan_names_both_targets_and_what_the_desk_already_has():
    st = {"params_doc": {"params": {"strategies": {"MR_STRETCH": {"enabled": True}},
                                    "conditions": {"table": {"MR_STRETCH": {"bear_stress": {"train": _per(0.71), "oos": _per(0.73)}}},
                                                   "all_weather": {"MR_STRETCH": {"all_weather": True}},
                                                   "aplus": {"bear_stress": _per(0.72, 723, 1.79)}}}}}
    latest = {"R_NEAR": {"label": "Near", "verdict": "watch", "by_condition": {"bear_stress": {"train": _per(0.72), "oos": _per(0.69)}},
                         "payoff": {"oos": {"planned_rr": 1.2}}},
              "R_FITS": {"label": "Fits", "verdict": "candidate", "by_condition": {"bear_stress": {"train": _per(0.8), "oos": _per(0.85)}}},
              "R_AW": {"label": "Everywhere", "verdict": "candidate", "all_weather": {"all_weather": True}, "oos": {"win_rate": 0.7}}}
    now = {"condition": "bear_stress", "label": "Bear-market selloff", "plural": "bear-market selloffs", "asof": "2026-11-20", "days": 2,
           "source": "daily prices"}
    plan = RS.today_plan(st, now, latest, {k: {} for k in latest})
    assert plan["condition"] == "bear_stress" and plan["sessions_in_a_row"] == 2 and plan["aplus_here_2022_now"]["trades"] == 723
    assert plan["live_strongest_here"][0]["strategy"] == "MR_STRETCH" and plan["live_all_weather"]
    assert [r["id"] for r in plan["ideas_closest_for_today"]] == ["R_FITS", "R_NEAR"]
    assert plan["ideas_closest_for_today"][1]["planned_rr"] == 1.2 and [r["id"] for r in plan["ideas_all_weather_not_live"]] == ["R_AW"]
    text = plan["every_day"]
    assert "bear-market selloff" in text and "all-weather" in text and "Win rate comes first" in text and "reward at least the risk" in text
    assert RS.today_plan(st, None, latest, {}) is None
    assert plan["own_design_today"]["aim"] == "today"            # no own design on record yet: start with today's condition


def test_own_designs_alternate_between_the_two_targets():
    own = lambda aim=None, mc=None: {"spec": {"source": {"title": "Swing Desk own design (2026-11-20)"},
                                              **({"aim": aim} if aim else {}), **({"market_conditions": mc} if mc else {})}}
    web = {"spec": {"source": {"title": "Some web page"}, "aim": "today"}}
    latest = {"R_A": {"tested_at": "2026-11-19T15:20"}, "R_B": {"tested_at": "2026-11-20T15:20"}, "R_W": {"tested_at": "2026-11-21T10:00"}}
    # the last own design was built for today's condition -> the next one is all-weather (a web idea does not count)
    nxt = RS.own_design_aim(latest, {"R_A": own("all_weather"), "R_B": own("today"), "R_W": web}, "bear_stress")
    assert nxt["aim"] == "all_weather" and nxt["last_own_design"] == "R_B" and '"aim": "all_weather"' in nxt["do"]
    # and the other way round
    nxt = RS.own_design_aim(latest, {"R_A": own("today"), "R_B": own("all_weather")}, "bear_stress")
    assert nxt["aim"] == "today" and "bear-market selloff" in nxt["do"]
    # an older idea without the field counts as built for a condition when its rule is tied to market conditions
    assert RS.spec_aim(own(mc=["bear_stress"])["spec"]) == "today" and RS.spec_aim(own()["spec"]) is None
    assert RS.own_design_aim(latest, {"R_B": own(mc=["bear_stress"])}, "bull_calm")["aim"] == "all_weather"
    assert R.validate({"id": "R_AIM", "label": "x", "family": "x", "about": "x", "aim": "all_weather", "conditions": ["close > open"],
                       "entry": {"kind": "close"}, "stop": {"atr": 3}, "target": {"atr": 3}, "exit": "sma5"}) == []


def test_the_daily_prompts_carry_the_two_targets_and_read_the_condition_from_fresh_prices():
    src = (ROOT / "ops" / "learning_prompt.md").read_text()
    for name in ("learning.md", "backup_learning.md"):
        p = ROOT / "ops" / "generated" / name
        if not p.exists():
            continue
        text = p.read_text()
        assert "engine.research next --state $W/state --data $W/data --out $W/rs" in text, name
        assert "today_standouts" in text and "all_weather_standouts" in text and '"ranking"' in text, name
        assert "at least one of the day's ideas is built for today's market condition" in text, name
        assert '"own_design_today"' in text and '"aim": "all_weather"' in text, name
    assert "(1) all-weather strategies" in src and "(2) strategies that win in today's market condition" in src
    assert "never read off the weekly history file" in src and "reward at least the risk" in src
