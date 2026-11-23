"""A night with hundreds of A+ setups (Fri Nov 20, 2026: 414 in a bear-market selloff).

Two things went wrong in the 2:06 PM post-close scan:
* the scan document was 891 KB (the digest alone 824 KB) and the dashboard refused it (limit 256 KB);
* the news check went by score, and with 414 setups tied near 99.8 it missed six of the eight featured ones.
The dashboard copy of the digest is now fitted under the limit, the email lists the long tail one line each,
and the scan writes news_order.txt (featured by rank first) and postprocess names any featured setup left unchecked.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import postprocess as pp  # noqa: E402
from engine import scan as S  # noqa: E402


def _entry(i, score, rank=None, also=False, sector=None):
    tk = f"T{i:03d}"
    e = {"id": f"e_2026-11-23_{tk}_FAIR_VALUE", "type": "entry", "status": "new", "ticker": tk, "score": score,
         "grade": "A+", "sector_key": sector or f"S{i}", "rank": rank, "news_check": "pending", "valid_for": "2026-11-23",
         "entry": 100.0, "stop": 90.0, "target": 104.0, "shares": 1, "total_cost": 100.0, "risk_usd": 10.0,
         "strategy_label": "Below Fair Value (VWAP-20)", "win_pct": 78.1, "hold_max_days": 8}
    if also:
        e["also_qualified"] = True
    return e


def test_news_order_puts_the_featured_list_first_even_when_others_score_higher():
    feat = [_entry(i, 99.8, rank=i + 1) for i in range(8)]
    extra = [_entry(100 + i, 100.0 - i / 100, also=True) for i in range(400)]
    order = S.news_order(extra + feat)
    assert order[:8] == [e["ticker"] for e in feat]
    assert order[8:10] == ["T100", "T101"] and len(order) == 10


def test_compact_line_carries_the_order():
    line = S.compact_line(_entry(1, 97.5, also=True))
    for part in ("T001", "buy limit $100.0", "stop $90.0", "target $104.0", "risk $10.0"):
        assert part in line


def test_fit_scan_doc_keeps_the_dashboard_copy_under_the_limit():
    calls = []

    def build(n):
        calls.append(n)
        return "F" * 3000 * n + "\n" + "c" * 150 * 414

    doc = {"id": "x", "digest": "F" * 3000 * 20 + "c" * 150 * 414 + "B" * 800_000}
    out = S.fit_scan_doc(doc, build)
    assert S.doc_bytes(out) <= S.DOC_LIMIT and out["digest_condensed"] is True and calls[0] == 5
    small = {"id": "y", "digest": "short"}
    assert S.fit_scan_doc(small, build) is small and "digest_condensed" not in small


def test_fit_scan_doc_last_resort_drops_the_long_list():
    doc = {"id": "x", "digest": "TOP\n=== ALSO QUALIFIED (A+...) ===\n" + "z" * 900_000}
    out = S.fit_scan_doc(doc, lambda n: doc["digest"])
    assert S.doc_bytes(out) <= S.DOC_LIMIT and out["digest"].startswith("TOP")


def _out(tmp_path, entries, scan, held=None):
    out = tmp_path / "out"
    for sub in ("alerts", "scans"):
        (out / "docs" / sub).mkdir(parents=True)
    writes = []
    for e in entries:
        (out / "docs" / "alerts" / f"{e['id']}.json").write_text(json.dumps(e))
        writes.append({"op": "set", "collection": "alerts", "doc_id": e["id"],
                       "file_path": str(out / "docs" / "alerts" / f"{e['id']}.json")})
    (out / "docs" / "scans" / f"{scan['id']}.json").write_text(json.dumps(scan))
    writes.append({"op": "set", "collection": "scans", "doc_id": scan["id"],
                   "file_path": str(out / "docs" / "scans" / f"{scan['id']}.json")})
    (out / "writes.json").write_text(json.dumps([writes[i:i + 45] for i in range(0, len(writes), 45)]))
    (out / "scan.json").write_text(json.dumps({"scan": scan, "entries": entries, "ctx": {"held_sectors": held or {}}}))
    return out


def test_414_setup_night_saves_and_names_unchecked_featured(tmp_path, monkeypatch):
    def fake_digest(scan, kept, *a, full_extra=None, **k):
        n = S.EXTRA_FULL if full_extra is None else full_extra
        extra = [e for e in kept if e.get("also_qualified")]
        return "HEAD\n=== ALSO QUALIFIED ===\n" + "F" * 2000 * min(n, len(extra)) + "\n".join(
            S.compact_line(e) for e in extra[n:])

    def fake_make(*a, **k):
        if len(a) > 8:
            k["full_extra"] = a[8]
        return fake_digest(a[0], a[1], **{kk: v for kk, v in k.items() if kk == "full_extra"})
    monkeypatch.setattr(pp, "make_digest", fake_make)
    feat = [_entry(i, 99.8, rank=i + 1) for i in range(8)]
    # the higher-scored ones sit in a sector the open positions already fill (as SDGR, GRAL, NTRA did)
    extra = [_entry(100 + i, 100.0 - i / 1000, also=True, sector="Health Care") for i in range(406)]
    entries = feat + extra
    scan = {"id": "20261120-1406-postclose", "mode": "postclose", "status": "ok", "started_at": "2026-11-20T14:06:39-08:00",
            "session_day": "2026-11-23", "counts": {"aplus": 414, "week_aplus": 478}, "positions": [], "near_misses": [],
            "premarket_checks": [], "funds": {"open_count": 13}, "market": {"label": "Correction"}, "data": {"status": "ok"},
            "close_snapshot": {"rows": {f"E{i}": [1, 2, 3] for i in range(160)}}}
    out = _out(tmp_path, entries, scan, held={"Health Care": 2})
    # the first check went by score: ten also-qualified setups, two featured
    first = [f"T{100 + i:03d}=clean" for i in range(8)] + ["T000=clean", "T001=clean"]
    r = pp.apply(str(out), first)
    assert "NEWS_CHECK_MISSING" in r
    for tk in ("T002", "T007"):
        assert tk in r["NEWS_CHECK_MISSING"]
    order = (out / "news_order.txt").read_text().split()
    assert order[:8] == [f"T{i:03d}" for i in range(8)]
    doc = json.loads((out / "docs" / "scans" / "20261120-1406-postclose.json").read_text())
    assert S.doc_bytes(doc) <= S.DOC_LIMIT
    email = (out / "digest.md").read_text()
    assert "T505" in email  # the email still lists every setup
    # after checking the featured ones too, nothing is missing
    r2 = pp.apply(str(out), first + [f"T{i:03d}=clean" for i in range(2, 8)])
    assert "NEWS_CHECK_MISSING" not in r2


def test_scan_prompts_check_news_in_news_order():
    gen = ROOT / "ops" / "generated"
    for f in ("preclose.md", "postclose.md", "manual.md", "weekend.md", "preclose_halfday.md"):
        t = (gen / f).read_text()
        assert "news_order.txt" in t and "NEWS_CHECK_MISSING" in t, f


def test_alert_refreshes_carry_only_what_differs_from_the_card():
    """Sun Nov 22, 2026: the weekend refresh re-scored 414 setups already on the dashboard; carried in full their
    refreshes made a 494 KB scan document (limit 256 KB). Now each refresh holds only what differs from its card."""
    entries, alerts = [], {}
    for i in range(414):
        e = _entry(i, 99.0, rank=i + 1 if i < 8 else None, also=i >= 8)
        e.update({"order": "Buy LIMIT $100.00, day order " + "x" * 300, "exit_signal": "Sell at the close " + "y" * 300,
                  "size_note": "No free cash at the moment: size shown assumes a slot frees up.", "hold_typical_days": 3})
        card = dict(e)
        if i % 3 == 0:
            e = {**e, "score": 98.5}   # the re-score moved a few
        entries.append(e)
        alerts[e["id"]] = {"doc": card, "version": 1}
    ctx = {"existing_entry_ids": list(alerts), "window": {}, "when": "4:19 pm"}
    full = S.build_overlay({}, entries, ctx)
    thin = S.build_overlay({}, entries, ctx, alerts)
    assert S.doc_bytes({"alert_updates": full}) > 256_000
    assert S.doc_bytes({"alert_updates": thin}) < 60_000
    first = thin[entries[0]["id"]]
    assert first == {"score": 98.5, "withdrawn": False, "rank": 1}
    # the dashboard lays the newest update over the card: same result either way
    for e in entries:
        card = alerts[e["id"]]["doc"]
        a, b = {**card, **full[e["id"]]}, {**card, **thin[e["id"]]}
        assert {k: v for k, v in a.items() if v is not None} == {k: v for k, v in b.items() if v is not None}


def test_fit_scan_doc_thins_alert_refreshes_as_a_last_resort():
    big = {f"e_{i}": {"withdrawn": False, "rank": None, "score": 99.0, "order": "z" * 1500, "exit_signal": "q" * 1000}
           for i in range(200)}
    big["e_w"] = {"withdrawn": True, "withdrawn_reason": "No longer qualifies", "rank": None, "ticker": "W", "carried": True}
    doc = {"id": "x", "digest": "short", "alert_updates": big}
    out = S.fit_scan_doc(doc, lambda n: "short")
    assert S.doc_bytes(out) <= S.DOC_LIMIT
    assert out["alert_updates"]["e_w"]["carried"] is True and "order" not in out["alert_updates"]["e_0"]


def test_postprocess_keeps_carried_withdrawals(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "make_digest", lambda *a, **k: "DIGEST")
    feat = [_entry(i, 99.8, rank=i + 1) for i in range(3)]
    scan = {"id": "20261122-1619-auto", "mode": "postclose", "status": "ok", "started_at": "2026-11-22T16:19:00-08:00",
            "session_day": "2026-11-23", "counts": {"aplus": 3, "week_aplus": 3}, "positions": [], "near_misses": [],
            "premarket_checks": [], "funds": {}, "market": {}, "data": {"status": "ok"},
            "alert_updates": {"e_old_MRNA": {"withdrawn": True, "carried": True, "ticker": "MRNA", "withdrawn_reason": "x"}}}
    out = _out(tmp_path, feat, scan)
    sj = json.loads((out / "scan.json").read_text())
    sj["ctx"].update({"existing_entry_ids": [], "window": {}, "when": "4:19 pm"})
    (out / "scan.json").write_text(json.dumps(sj))
    pp.apply(str(out), ["T000=clean", "T001=clean", "T002=clean"])
    doc = json.loads((out / "docs" / "scans" / "20261122-1619-auto.json").read_text())
    assert doc["alert_updates"]["e_old_MRNA"]["carried"] is True
