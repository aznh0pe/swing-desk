"""The news check: what carries over between scans, and how the list is re-ranked after a drop.

Background (Nov 6-7, 2026): v1 rules told the news check to mark sector selloffs "caution" (-8 score),
which knocked dip-buy setups below A+ for exactly the condition the strategies trade. v2 judges
company-specific news only, and verdicts given under v1 are not carried forward.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import postprocess as pp  # noqa: E402


def _state(*scans):
    return {"scans": {d["id"]: {"doc": d} for d in scans}}


def _scan(id_, rules, reason):
    d = {"id": id_, "session_day": "2026-11-09", "status": "ok", "started_at": "2026-11-06T14:17:00-08:00",
         "news_dropped": [{"ticker": "CRM", "reason": "score fell below A+ after news caution: " + reason}]}
    if rules:
        d["news_rules"] = rules
    return d


def test_verdicts_from_older_rules_are_not_carried_forward():
    old = _scan("20261106-1417-postclose", None, "software-sector selloff")
    assert pp.prior_verdicts(_state(old), "2026-11-09") == {}


def test_current_rule_verdicts_still_never_soften():
    cur = _scan("20261107-1241-postclose", pp.NEWS_RULES, "analyst downgrade")
    got = pp.prior_verdicts(_state(cur), "2026-11-09")
    assert got["CRM"][0] == "caution" and "analyst downgrade" in got["CRM"][1]


def test_carried_note_is_not_nested():
    cur = _scan("20261107-1241-postclose", pp.NEWS_RULES, "CEO exit (from the 2:17 pm news check) (from the 4:24 pm news check)")
    note = pp.prior_verdicts(_state(cur), "2026-11-09")["CRM"][1]
    assert note.count("(from the") == 1, note


def _entry(tk, score, sector, rank=None, also=False):
    e = {"id": f"e_2026-11-09_{tk}_FAIR_VALUE", "type": "entry", "status": "new", "ticker": tk, "score": score,
         "grade": "A+", "sector_key": sector, "rank": rank, "news_check": "pending", "valid_for": "2026-11-09",
         "entry": 100.0, "stop": 90.0, "target": 104.0, "shares": 1, "total_cost": 100.0}
    if also:
        e["also_qualified"] = True
        e["capacity_note"] = "Also qualified (2 positions/featured setups already in Information Technology)."
    return e


def test_sector_capped_setup_moves_up_when_the_news_check_drops_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "make_digest", lambda *a, **k: "digest")
    entries = [_entry("CRM", 80.1, "Information Technology", 1), _entry("CRWD", 79.2, "Information Technology", 2),
               _entry("NTAP", 77.0, "Information Technology", also=True), _entry("NUE", 75.0, "Materials", 3)]
    scan = {"id": "20261107-1241-postclose", "mode": "postclose", "status": "ok", "started_at": "2026-11-07T12:41:00-08:00",
            "session_day": "2026-11-09", "counts": {"aplus": 4, "week_aplus": 9}, "positions": [], "near_misses": [],
            "premarket_checks": [], "funds": {"open_count": 2}, "market": {"label": "Bull"}, "data": {"status": "ok"}}
    out = tmp_path / "out"
    for sub in ("alerts", "scans"):
        (out / "docs" / sub).mkdir(parents=True)
    writes = []
    for e in entries:
        (out / "docs" / "alerts" / f"{e['id']}.json").write_text(json.dumps(e))
        writes.append({"op": "set", "collection": "alerts", "doc_id": e["id"],
                       "file_path": str(out / "docs" / "alerts" / f"{e['id']}.json")})
    (out / "docs" / "scans" / f"{scan['id']}.json").write_text(json.dumps(scan))
    writes.append({"op": "set", "collection": "scans", "doc_id": scan["id"], "file_path": str(out / "docs" / "scans" / f"{scan['id']}.json")})
    (out / "writes.json").write_text(json.dumps([writes]))
    (out / "scan.json").write_text(json.dumps({"scan": scan, "entries": entries, "ctx": {"held_sectors": {}}}))
    r = pp.apply(str(out), ["CRM=avoid:Agreed to be acquired", "CRWD=clean", "NTAP=clean", "NUE=clean"])
    kept = json.loads((out / "scan.json").read_text())
    by = {e["ticker"]: e for e in kept["entries"]}
    assert "CRM" not in by and r["kept"] == 3
    assert not by["NTAP"].get("also_qualified") and by["NTAP"]["rank"] is not None
    assert [by[t]["rank"] for t in ("CRWD", "NTAP", "NUE")] == [1, 2, 3]
    assert kept["scan"]["news_rules"] == pp.NEWS_RULES
    alert = json.loads((out / "docs" / "alerts" / "e_2026-11-09_NTAP_FAIR_VALUE.json").read_text())
    assert alert["rank"] == 2 and not alert.get("also_qualified")


def test_sector_cap_still_applies_with_open_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "make_digest", lambda *a, **k: "digest")
    entries = [_entry("CRM", 80.1, "Information Technology", 1), _entry("CRWD", 79.2, "Information Technology", None, also=True)]
    scan = {"id": "s", "mode": "postclose", "status": "ok", "started_at": "2026-11-07T12:41:00-08:00", "session_day": "2026-11-09",
            "counts": {"aplus": 2}, "positions": [], "near_misses": [], "premarket_checks": [], "funds": {"open_count": 1}}
    out = tmp_path / "out"
    for sub in ("alerts", "scans"):
        (out / "docs" / sub).mkdir(parents=True)
    for e in entries:
        (out / "docs" / "alerts" / f"{e['id']}.json").write_text(json.dumps(e))
    (out / "writes.json").write_text("[]")
    (out / "scan.json").write_text(json.dumps({"scan": scan, "entries": entries,
                                               "ctx": {"held_sectors": {"Information Technology": 1}}}))
    pp.apply(str(out), ["CRM=clean", "CRWD=clean"])
    by = {e["ticker"]: e for e in json.loads((out / "scan.json").read_text())["entries"]}
    assert by["CRM"]["rank"] == 1 and by["CRWD"].get("also_qualified")  # 1 held + 1 featured = the cap of 2


def test_overlay_refreshes_the_plan_of_existing_alerts():
    from engine import scan as sc
    e = _entry("CRM", 80.2, "Information Technology", 1)
    e.update(entry=213.65, stop=179.83, target=221.34, order="Buy LIMIT $213.65")
    ups = sc.build_overlay({"news_dropped": []}, [e], {"existing_entry_ids": [e["id"]], "window": {}})
    u = ups[e["id"]]
    assert u["entry"] == 213.65 and u["target"] == 221.34 and u["order"] == "Buy LIMIT $213.65" and u["withdrawn"] is False
