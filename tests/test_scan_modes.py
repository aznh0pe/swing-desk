"""End-to-end scan checks on synthetic data: every mode, closed days, half days and stale feeds.

These guard the behaviour scheduled runs depend on: a scan always finishes with a scan document,
never sends orders for a past session, and says plainly when its prices are not from today.
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
from engine import scan  # noqa: E402
from engine import strategies as S  # noqa: E402


# everything an alert (and its email) must carry
ENTRY_FIELDS = ("entry", "stop", "target", "win_pct", "score", "grade", "shares", "total_cost", "hold_max_days",
                "exit_signal", "strategy_label", "order", "valid_for", "rank", "risk_usd")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synth.py"), str(d)], check=True, capture_output=True)
    return d


def _state(base: pathlib.Path) -> pathlib.Path:
    st = base / "state"
    (st / "config").mkdir(parents=True)
    (st / "config" / "settings.json").write_text(json.dumps(
        {"initial_balance": 5000, "income_pct": 30, "max_open_positions": 5}))
    (st / "ledger").mkdir()
    (st / "ledger" / "initial.json").write_text(json.dumps(
        {"id": "initial", "type": "initial", "amount": 5000, "date": "2026-11-01"}))
    (st / "trades").mkdir()
    (st / "trades" / "t1.json").write_text(json.dumps(
        {"id": "t1", "status": "open", "ticker": "S01", "strategy": "MR_STRETCH", "entry_price": 50.0,
         "entry_date": "2026-11-03", "shares": 5, "stop": 1.0, "target": 9999.0, "max_hold": 8, "dyn_exit": "none"}))
    (st / "alerts").mkdir()
    (st / "alerts" / "e_pending.json").write_text(json.dumps(
        {"id": "e_pending", "type": "entry", "status": "new", "ticker": "S02", "entry": 40.0, "stop": 35.0,
         "target": 44.0, "atr": 1.0, "valid_for": "2026-11-06"}))
    return st


def _run(synth, tmp_path, mode, when, data=None):
    now = dt.datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=cal.PT)
    st = _state(tmp_path)
    out = tmp_path / "out"
    res = scan.run(mode, str(data or synth), str(st), str(out), now=now)
    doc = json.loads((out / "scan.json").read_text())
    chunks = json.loads((out / "writes.json").read_text())
    for chunk in chunks:  # every write points at a real JSON file
        for w in chunk:
            assert json.loads(pathlib.Path(w["file_path"]).read_text()) is not None
    return res, doc, (out / "digest.md").read_text(), chunks


@pytest.mark.parametrize("mode,when", [("premarket", "2026-11-26 05:50"), ("preclose", "2026-11-26 12:25"),
                                       ("postclose", "2026-11-26 14:05"), ("premarket", "2026-11-07 05:50")])
def test_scheduled_scan_on_a_closed_day_is_recorded_and_does_nothing(synth, tmp_path, mode, when):
    res, doc, digest, chunks = _run(synth, tmp_path, mode, when)
    assert res["status"] == "closed"
    assert doc["scan"]["status"] == "closed" and doc["entries"] == []
    assert "Market closed today" in digest
    assert [w["collection"] for c in chunks for w in c] == ["scans"]
    assert scan.headline(doc["scan"]).startswith("Market closed")


def test_manual_scan_on_a_weekend_still_runs_for_the_next_session(synth, tmp_path):
    res, doc, digest, _ = _run(synth, tmp_path, "auto", "2026-11-07 10:00")
    assert res["mode"] == "postclose" and res["status"] != "closed"
    assert doc["scan"]["session_day"] == "2026-11-09"


def test_preclose_ignores_an_intraday_snapshot_from_another_day(synth, tmp_path):
    # the synthetic intraday snapshot is from 2026-11-05; on the 6th it is not today's price
    res, doc, digest, _ = _run(synth, tmp_path, "preclose", "2026-11-06 12:25")
    assert doc["entries"] == []
    assert any("No live prices from today" in n for n in doc["scan"]["notes"])
    for e in doc["entries"]:
        assert e["valid_for"] >= "2026-11-06"


def test_preclose_with_todays_snapshot_uses_live_prices(synth, tmp_path):
    res, doc, digest, _ = _run(synth, tmp_path, "preclose", "2026-11-05 12:25")
    assert doc["scan"]["data"]["partial_bar"] is True
    assert not any("No live prices" in n for n in doc["scan"]["notes"])
    for e in doc["entries"]:
        assert e["valid_for"] >= "2026-11-05"


def test_early_preclose_read_is_labelled(synth, tmp_path):
    res, doc, digest, _ = _run(synth, tmp_path, "preclose", "2026-11-05 08:30")
    assert any(n.startswith("Early read") for n in doc["scan"]["notes"])
    assert "Note: Early read" in digest


def test_half_day_preclose_after_the_early_close_runs_as_postclose(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "EARLY_CLOSES", cal.EARLY_CLOSES | {"2026-11-05"})
    res, doc, digest, _ = _run(synth, tmp_path, "preclose", "2026-11-05 12:25")
    assert res["mode"] == "postclose"
    assert any("half day" in n for n in doc["scan"]["notes"])


def test_half_day_preclose_before_the_early_close_names_the_real_close(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "EARLY_CLOSES", cal.EARLY_CLOSES | {"2026-11-05"})
    st = _state(tmp_path)
    tr = json.loads((st / "trades" / "t1.json").read_text())
    tr.update(entry_date="2026-10-20", max_hold=3)  # past its time stop
    (st / "trades" / "t1.json").write_text(json.dumps(tr))
    now = dt.datetime(2026, 11, 5, 9, 20, tzinfo=cal.PT)
    scan.run("preclose", str(synth), str(st), str(tmp_path / "o2"), now=now)
    doc = json.loads((tmp_path / "o2" / "scan.json").read_text())
    texts = [s["text"] for p in doc["scan"]["positions"] for s in p["signals"]]
    assert any("10:00 AM PT close" in t for t in texts), texts


def test_premarket_skips_gap_checks_on_stale_premarket_prices(synth, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(synth, data)
    (data / "premarket.csv").write_text("ticker,pm_last,pm_high,pm_low,pm_volume,asof\n"
                                        "S02,1.0,1.0,1.0,0,2026-11-04T08:00-05:00\n")
    res, doc, digest, _ = _run(synth, tmp_path, "premarket", "2026-11-06 05:50", data=data)
    assert any("Pre-market prices in the feed are from 2026-11-04" in n for n in doc["scan"]["notes"])
    checks = {c["ticker"]: c for c in doc["scan"]["premarket_checks"]}
    assert checks["S02"]["verdict"] == "keep"  # a stale $1 print must not cancel the order


def test_normal_postclose_scan_is_complete(synth, tmp_path):
    res, doc, digest, chunks = _run(synth, tmp_path, "postclose", "2026-11-05 14:10")
    s = doc["scan"]
    assert res["status"] in ("ok", "ok_with_warnings") and s["data"]["status"] == "ok"
    assert s["engine"]["fingerprint"]
    assert digest.startswith("SWING DESK | POST-CLOSE SCAN")
    for e in doc["entries"]:
        for k in ENTRY_FIELDS:
            assert e.get(k) not in (None, ""), (k, e["ticker"])
        assert e["stop"] < e["entry"] < e["target"]
        assert e["valid_for"] == "2026-11-06"


def test_stale_data_is_flagged_at_the_top(synth, tmp_path):
    res, doc, digest, _ = _run(synth, tmp_path, "postclose", "2026-11-12 14:10")
    assert doc["scan"]["data"]["status"] == "stale"
    assert digest.splitlines()[1].startswith("⚠ DATA WARNING")
    assert scan.headline(doc["scan"]).startswith("⚠ data stale")


@pytest.mark.skipif(not (pathlib.Path(__import__("os").environ.get("SWING_DATA", "/nonexistent")) / "meta.json").exists(),
                    reason="set SWING_DATA (a clone of the data branch) and SWING_STATE (an exported state) to run")
def test_real_data_postclose_alerts_are_complete(tmp_path):
    import os
    st = tmp_path / "state"
    shutil.copytree(os.environ["SWING_STATE"], st)
    now = dt.datetime.combine(cal.prev_trading_day(dt.date.today() + dt.timedelta(days=1)), dt.time(14, 10), cal.PT)
    scan.run("postclose", os.environ["SWING_DATA"], str(st), str(tmp_path / "out"), now=now)
    doc = json.loads((tmp_path / "out" / "scan.json").read_text())
    for e in doc["entries"]:
        missing = [k for k in ENTRY_FIELDS if e.get(k) in (None, "")]
        assert not missing, (e["ticker"], missing)
        assert e["stop"] < e["entry"] < e["target"] and e["shares"] >= 1


def test_every_scan_writes_the_email_headline(synth, tmp_path):
    # the scheduled prompt's final message starts with this line, so it must always exist
    for mode, when, expect in (("postclose", "2026-11-05 14:10", "market "), ("premarket", "2026-11-26 05:50", "Market closed"),
                               ("postclose", "2026-11-12 14:10", "⚠ data stale")):
        sub = tmp_path / mode / when.replace(" ", "_").replace(":", "")
        sub.mkdir(parents=True)
        res, doc, digest, _ = _run(synth, sub, mode, when)
        line = (sub / "out" / "headline.txt").read_text()
        assert expect in line and "\n" not in line, line
        assert res["headline"] == line


def test_news_check_rewrites_digest_and_headline_together(synth, tmp_path):
    from engine import postprocess
    res, doc, digest, _ = _run(synth, tmp_path, "postclose", "2026-11-05 14:10")
    out = tmp_path / "out"
    tickers = [e["ticker"] for e in doc["entries"]]
    news = [f"{t}=clean" for t in tickers[1:]] + ([f"{tickers[0]}=avoid:Agreed to be acquired"] if tickers else [])
    r = postprocess.apply(str(out), news, str(tmp_path / "state"))
    line = (out / "headline.txt").read_text()
    assert r["headline"] == line
    after = json.loads((out / "scan.json").read_text())["scan"]
    assert after["counts"]["aplus"] == len(tickers) - (1 if tickers else 0)
    if tickers:
        assert "Dropped after news check" in (out / "digest.md").read_text()
    for chunk in json.loads((out / "writes.json").read_text()):
        for w in chunk:
            assert pathlib.Path(w["file_path"]).exists()


def _run_with_grading(synth, tmp_path, grading):
    now = dt.datetime(2026, 11, 5, 14, 10, tzinfo=cal.PT)
    st = _state(tmp_path)
    (st / "strategy").mkdir()
    (st / "strategy" / "params.json").write_text(json.dumps({"params": {"grading": grading}}))
    out = tmp_path / "out"
    scan.run("postclose", str(synth), str(st), str(out), now=now)
    return json.loads((out / "scan.json").read_text()), (out / "digest.md").read_text()


def test_weekly_topups_follow_their_own_rule_and_say_so(synth, tmp_path):
    base = {"aplus_score": 999, "aplus_pwin": 0.99, "topup_floor": 0, "topup_pwin": 0.0, "topup_ok": True,
            "oos_topup_win_rate": 0.735, "oos_topup_trades": 275, "oos_topup_avg_ret": 0.0085, "oos_aplus_win_rate": 0.76}
    doc, digest = _run_with_grading(synth, tmp_path / "a", base)
    tops = [e for e in doc["entries"] if e.get("topup_note")]
    pace = doc["scan"]["topup_pace"]
    assert pace["pace"] == 4 and pace["behind"] == 4 and len(tops) == pace["sent"] and 1 <= len(tops) <= 4
    for e in tops:
        assert e["grade_kind"] == "topup" and e["grade"] == "A" and "won 74% of 275" in e["topup_note"]
        assert abs(e["win_pct"] - 100 * (0.735 * 275 + e["model_win_pct"] / 100 * 30) / 305) < 0.2
    assert "Weekly top-up (below A+)" in digest and "TOP-UP" in digest
    assert doc["scan"]["counts"]["week_topups"] == len(tops) and "weekly top-up" in scan.email_headline(doc["scan"])
    # strategies excluded from top-ups (by the test or by their live record) never top up
    used = {e["strategy"] for e in tops}
    doc2, _ = _run_with_grading(synth, tmp_path / "b", {**base, "topup_excluded": {s: "test" for s in used},
                                                         "topup_excluded_live": {}})
    assert not ({e["strategy"] for e in doc2["entries"] if e.get("topup_note")} & used)
    # off: none, and the email says why the week is short
    doc3, digest3 = _run_with_grading(synth, tmp_path / "c", {**base, "topup_mode": "off"})
    assert not [e for e in doc3["entries"] if e.get("topup_note")] and "switched off" in digest3
    # nothing clears the bar: none forced, and the email says so
    doc4, digest4 = _run_with_grading(synth, tmp_path / "d", {**base, "topup_floor": 1000})
    assert not [e for e in doc4["entries"] if e.get("topup_note")] and "none was forced" in digest4


def test_alerts_say_how_their_strategy_does_in_todays_market_condition(synth, tmp_path):
    base = {"aplus_score": 999, "aplus_pwin": 0.99, "topup_floor": 0, "topup_pwin": 0.0, "topup_ok": True,
            "oos_topup_win_rate": 0.735, "oos_topup_trades": 275, "oos_topup_avg_ret": 0.0085, "oos_aplus_win_rate": 0.76}
    doc, _ = _run_with_grading(synth, tmp_path / "a", base)
    tops = [e for e in doc["entries"] if e.get("topup_note")]
    assert tops
    cond = doc["scan"]["market"]["condition"]
    names = {e["strategy"] for e in tops}
    aw_name = sorted(names)[0]
    cell = {"oos": {"trades": 400, "win_rate": 0.71, "avg_ret": 0.006, "profit_factor": 1.4}}
    conds = {"table": {n: {cond: cell} for n in names}, "topup": {cond: {"trades": 139, "win_rate": 0.719}},
             "all_weather": {aw_name: {"all_weather": True, "tested": list(S.CONDITIONS), "weak": [],
                                       "oos_win_range": [0.64, 0.75]}}}
    st = _state(tmp_path / "b")
    (st / "strategy").mkdir()
    (st / "strategy" / "params.json").write_text(json.dumps({"params": {"grading": base, "conditions": conds}}))
    out = tmp_path / "b" / "out"
    scan.run("postclose", str(synth), str(st), str(out), now=dt.datetime(2026, 11, 5, 14, 10, tzinfo=cal.PT))
    d2 = json.loads((out / "scan.json").read_text())
    t2 = [e for e in d2["entries"] if e.get("topup_note")]
    assert t2 and all(e["market_condition"] == cond and "today's condition" in e["condition_note"] for e in t2)
    assert all("weekly top-ups won 72% of 139" in e["condition_note"] for e in t2)
    aw = [e for e in t2 if e["strategy"] == aw_name]
    assert all(e.get("all_weather") and "is all-weather" in e["condition_note"] and "64-75%" in e["condition_note"] for e in aw)
    assert not any(e.get("all_weather") for e in t2 if e["strategy"] != aw_name)
    assert "all-weather (hold up in any market)" in (out / "digest.md").read_text() or not d2["scan"]["market"].get("condition")


def test_news_check_judges_top_ups_against_their_own_bar_and_recounts_the_week(synth, tmp_path):
    from engine import postprocess
    base = {"aplus_score": 999, "aplus_pwin": 0.99, "topup_floor": 0, "topup_pwin": 0.0, "topup_ok": True,
            "oos_topup_win_rate": 0.70, "oos_topup_trades": 649, "oos_topup_avg_ret": 0.011, "oos_aplus_win_rate": 0.76}
    doc, _ = _run_with_grading(synth, tmp_path / "a", base)
    out = tmp_path / "a" / "out"
    tops = [e for e in doc["entries"] if e.get("topup_note")]
    assert tops and doc["scan"]["topup_pace"]["sent"] == len(tops)
    st = tmp_path / "a" / "state"
    # a caution costs 8 points: a top-up that still clears the top-up bar stays (flagged), it is not judged against A+
    r = postprocess.apply(str(out), [f"{tops[0]['ticker']}=caution:Analyst downgrade"], str(st))
    after = json.loads((out / "scan.json").read_text())
    assert not r["dropped"] and any(e["ticker"] == tops[0]["ticker"] and e["news_check"] == "caution" for e in after["entries"])
    # avoid drops it, and the week's count, the pace and the email all say so
    week_before = after["scan"]["counts"]["week_topups"]
    r2 = postprocess.apply(str(out), [f"{tops[0]['ticker']}=avoid:Agreed to be acquired"], str(st))
    sc = json.loads((out / "scan.json").read_text())["scan"]
    assert r2["dropped"] and sc["counts"]["week_topups"] == week_before - 1
    assert sc["topup_pace"]["dropped_by_news"] == 1 and sc["topup_pace"]["sent"] == len(tops) - 1
    dg = (out / "digest.md").read_text()
    assert "dropped by the news check" in dg and f"{tops[0]['ticker']} (Agreed to be acquired)" in dg
    # running it again (say, to add a run problem) neither double-counts nor forgets the drop
    postprocess.apply(str(out), [], str(st), ["a test problem"])
    sc2 = json.loads((out / "scan.json").read_text())["scan"]
    assert sc2["counts"]["week_topups"] == week_before - 1 and sc2["topup_pace"]["dropped_by_news"] == 1
    assert "Dropped after news check" in (out / "digest.md").read_text()


def test_later_scans_refresh_or_supersede_open_exit_alerts():
    # the 12:26 pre-close scan left "sell at close" alerts open for PSX and ILMN; at 2:07 the post-close scan
    # says PSX: sell at the open (same exit rule, new wording), ILMN: keep holding (no signal any more)
    old = {
        "x_2026-11-11_tP_EXIT_RULE": {"doc": {"type": "exit", "status": "new", "trade_id": "tP", "code": "EXIT_RULE",
                                             "action": "SELL AT CLOSE", "text": "before the 1:00 PM PT close today"}},
        "x_2026-11-11_tI_EXIT_RULE": {"doc": {"type": "exit", "status": "new", "trade_id": "tI", "code": "EXIT_RULE"}},
        "x_2026-11-10_tP_RAISE_STOP": {"doc": {"type": "adjust", "status": "new", "trade_id": "tP", "code": "RAISE_STOP"}},
        "x_2026-11-11_tP_TARGET_HIT": {"doc": {"type": "exit", "status": "done", "trade_id": "tP", "code": "TARGET_HIT"}},
        "x_2026-11-11_tZ_EXIT_RULE": {"doc": {"type": "exit", "status": "new", "trade_id": "tZ", "code": "EXIT_RULE"}},
        "e_2026-11-11_ABC_S": {"doc": {"type": "entry", "status": "new"}},
    }
    positions = [
        {"trade_id": "tP", "ticker": "PSX", "asof": "2026-11-11", "last": 258.68, "unrealized": 11.64, "days_held": 4,
         "days_left": 4, "status": {"label": "SELL AT THE OPEN: EXIT RULE MET"},
         "signals": [{"code": "EXIT_RULE", "level": "alert", "action": "SELL AT OPEN", "urgency": "next",
                      "text": "take the gain at the next open"}]},
        {"trade_id": "tI", "ticker": "ILMN", "asof": "2026-11-11", "last": 212.93, "unrealized": -1.10, "days_held": 2,
         "days_left": 6, "status": {"label": "KEEP HOLDING"}, "signals": []},
    ]
    ups = scan.position_alert_updates(old, positions, "20261111-1407-postclose", "2:07 pm")
    # same signal, same day: refreshed to the latest wording, price and P&L
    p = ups["x_2026-11-11_tP_EXIT_RULE"]
    assert p["action"] == "SELL AT OPEN" and p["text"] == "take the gain at the next open" and p["last"] == 258.68
    assert "status" not in p
    # no signal any more: superseded, with what the latest scan says instead
    i = ups["x_2026-11-11_tI_EXIT_RULE"]
    assert i["status"] == "superseded" and "keep holding" in i["superseded_note"]
    # an older day's alert the latest scan no longer repeats is superseded too
    assert ups["x_2026-11-10_tP_RAISE_STOP"]["status"] == "superseded"
    # alerts you already acted on, positions this scan didn't check, and entry alerts are left alone
    assert not {"x_2026-11-11_tP_TARGET_HIT", "x_2026-11-11_tZ_EXIT_RULE", "e_2026-11-11_ABC_S"} & set(ups)
    # and they travel in the scan's alert_updates, which the dashboard applies (newest scan wins)
    assert scan.build_overlay({}, [], {"position_alert_updates": ups})["x_2026-11-11_tI_EXIT_RULE"]["status"] == "superseded"


def test_topups_count_only_what_is_sent_and_a_rerun_never_double_counts(synth, tmp_path, monkeypatch):
    """Nov 12: the post-close scan counted two top-ups, then dropped one at the earnings / reward-to-risk check, so
    the week stayed one short while the email said two were added. And a Scan now the same evening counted
    tonight's setups twice (in the week so far and again as new), so it could not add the missing one."""
    base = {"aplus_score": 999, "aplus_pwin": 0.99, "topup_floor": 0, "topup_pwin": 0.0, "topup_ok": True,
            "oos_topup_win_rate": 0.70, "oos_topup_trades": 600, "oos_topup_avg_ret": 0.01, "oos_aplus_win_rate": 0.76}
    doc, _ = _run_with_grading(synth, tmp_path / "a", base)
    tops = [e for e in doc["entries"] if e.get("topup_note")]
    assert len(tops) >= 2 and doc["scan"]["topup_pace"]["sent"] == len(tops)
    # the best one has an earnings report inside its hold: it is skipped and the next one takes its place
    first = tops[0]["ticker"]
    real = scan.earnings_conflict
    monkeypatch.setattr(scan, "earnings_conflict", lambda tk, e, d, h: (True, "2026-11-07") if tk == first else real(tk, e, d, h))
    doc_b, digest_b = _run_with_grading(synth, tmp_path / "b", base)
    tops_b = [e for e in doc_b["entries"] if e.get("topup_note")]
    assert first not in {e["ticker"] for e in tops_b}
    assert len(tops_b) == len(tops) == doc_b["scan"]["topup_pace"]["sent"]
    assert f"{len(tops_b)} weekly top-up(s) added" in digest_b
    monkeypatch.setattr(scan, "earnings_conflict", real)
    # a second run the same evening: tonight's top-ups count once, and the slot one of them leaves is filled anew
    now = dt.datetime(2026, 11, 5, 14, 40, tzinfo=cal.PT)
    st = tmp_path / "a" / "state"
    keep = tops[:-1]  # pretend the last one never went out (say the news check dropped it)
    for e in keep:
        (st / "alerts" / f"{e['id']}.json").write_text(json.dumps({**e, "status": "new"}))
    out = tmp_path / "a" / "out2"
    scan.run("postclose", str(synth), str(st), str(out), now=now)
    doc2 = json.loads((out / "scan.json").read_text())
    digest2 = (out / "digest.md").read_text()
    pace2 = doc2["scan"]["topup_pace"]
    assert pace2["week_so_far"] == len(keep) and pace2["sent"] == pace2["behind"] == len(tops) - len(keep)
    new_ids = {w["doc_id"] for ch in json.loads((out / "writes.json").read_text()) for w in ch if w["collection"] == "alerts"}
    assert not ({e["id"] for e in keep} & new_ids)  # nothing re-sent as new
    assert doc2["scan"]["counts"]["sent_earlier"] == len(keep) and doc2["scan"]["counts"]["topups"] == len(tops) - len(keep)
    assert digest2.count("ALREADY SENT") == len(keep)
    assert f"{len(keep)} sent earlier tonight" in scan.email_headline(doc2["scan"])


def test_live_boosts_never_loosen_grading_and_withdrawn_setups_never_count(synth, tmp_path):
    """Nov 12: positive live adjustments (from a short winning streak) let setups clear the A+ and top-up bars that
    the tested model would not send; replayed on 2022-now they more than doubled A+ setups and cut its win rate.
    Grading now uses only the adjustments that make it stricter; a boost shows in the win estimate alone. And a
    setup a later scan withdrew no longer counts toward the week's pace."""
    from engine import model as M
    adj = {"A": {"offset": 0.4, "wins": 12, "n": 12}, "B": {"offset": -0.3}}
    g = M.grading_offsets(adj)
    assert g["A"]["offset"] == 0.0 and g["B"]["offset"] == -0.3 and g["A"]["wins"] == 12
    base = {"aplus_score": 35, "aplus_pwin": 0.55, "topup_floor": 0, "topup_pwin": 0.0, "topup_ok": True,
            "oos_topup_win_rate": 0.70, "oos_topup_trades": 600, "oos_topup_avg_ret": 0.01, "oos_aplus_win_rate": 0.76}

    def run(sub, live_adjust, alerts=()):
        now = dt.datetime(2026, 11, 5, 14, 10, tzinfo=cal.PT)
        st = _state(tmp_path / sub)
        (st / "strategy").mkdir()
        models = {n: {"kind": "base", "base_rate": 0.62, "avg_win": 0.025, "avg_loss": -0.035}
                  for n in S.params_with_defaults(None)["strategies"]}
        (st / "strategy" / "params.json").write_text(json.dumps({"params": {"grading": base, "live_adjust": live_adjust,
                                                                             "models": models}}))
        for a in alerts:
            (st / "alerts" / f"{a['id']}.json").write_text(json.dumps(a))
        out = tmp_path / sub / "out"
        scan.run("postclose", str(synth), str(st), str(out), now=now)
        return json.loads((out / "scan.json").read_text())
    plain = run("a", {})
    names = {e["strategy"] for e in plain["entries"]}
    boosted = run("b", {n: {"offset": 0.6} for n in names})
    key = lambda d: sorted((e["ticker"], e["grade_kind"], e["score"]) for e in d["entries"])
    assert key(boosted) == key(plain)  # same setups, same grades, same scores
    assert any(b["win_pct"] > p["win_pct"] for b, p in zip(sorted(boosted["entries"], key=lambda e: e["ticker"]),
                                                          sorted(plain["entries"], key=lambda e: e["ticker"])))
    cut = run("c", {n: {"offset": -0.6} for n in names})
    assert len([e for e in cut["entries"] if e["grade_kind"] == "aplus"]) < len([e for e in plain["entries"] if e["grade_kind"] == "aplus"])
    # a setup withdrawn by a later scan earlier this week does not count toward the pace
    wd = [{"id": f"e_old{k}", "type": "entry", "status": "new", "ticker": f"Z{k}", "signal_date": "2026-11-03",
           "valid_for": "2026-11-04", "scan_id": "20261103-1405-postclose"} for k in range(2)]
    st_dir = tmp_path / "d"
    doc_kept = run("d", {}, wd)
    sdir = st_dir / "state" / "scans"
    sdir.mkdir()
    (sdir / "s1.json").write_text(json.dumps({"id": "s1", "started_at": "2026-11-04T05:50:00-08:00",
                                               "alert_updates": {"e_old0": {"withdrawn": True}}}))
    out = tmp_path / "d" / "out2"
    scan.run("postclose", str(synth), str(st_dir / "state"), str(out), now=dt.datetime(2026, 11, 5, 14, 10, tzinfo=cal.PT))
    doc_wd = json.loads((out / "scan.json").read_text())
    assert doc_wd["scan"]["topup_pace"]["week_so_far"] == doc_kept["scan"]["topup_pace"]["week_so_far"] - 1
