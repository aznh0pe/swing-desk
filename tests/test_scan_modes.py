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
    assert digest.startswith("SWING DESK | POSTCLOSE SCAN")
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
