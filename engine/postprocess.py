"""Apply the scan session's news checks to a finished scan's output.

    python -m engine.postprocess --out <scan-out-dir> --news "NVDA=clean" "XYZ=caution:Analyst downgrade" "ABC=avoid:Agreed to be acquired"

* clean   -> news_check "clean"
* caution -> news_check "caution", note, score -8 (dropped if it falls below the A+ cut-off)
* avoid   -> alert removed (doc file, write entry and digest line)
Regenerates digest.md, the scan document and writes.json so everything stays consistent.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .scan import load_params, make_digest
from .portfolio import load_state
from . import calendar as cal


def apply(out_dir: str, news: list[str], state_dir: str | None = None) -> dict:
    out = pathlib.Path(out_dir)
    sj = json.loads((out / "scan.json").read_text())
    scan, entries = sj["scan"], sj["entries"]
    P = load_params(load_state(state_dir), None) if state_dir else None
    cutoff = (P or {}).get("grading", {}).get("aplus_score", 80) if P else scan.get("aplus_cutoff", 80)
    verdicts = {}
    for item in news:
        if "=" not in item:
            continue
        tk, rest = item.split("=", 1)
        kind, _, note = rest.partition(":")
        verdicts[tk.strip().upper()] = (kind.strip().lower(), note.strip() or None)
    kept, dropped = [], []
    for e in entries:
        v = verdicts.get(e["ticker"].upper())
        if v is None:
            kept.append(e)
            continue
        kind, note = v
        if kind == "avoid":
            dropped.append({"ticker": e["ticker"], "reason": note or "news risk"})
            continue
        if kind == "caution":
            e["news_check"], e["news_note"] = "caution", note
            e["score"] = round(float(e["score"]) - 8, 1)
            if e["score"] < cutoff:
                dropped.append({"ticker": e["ticker"], "reason": f"score fell below A+ after news caution: {note}"})
                continue
        else:
            e["news_check"], e["news_note"] = "clean", note
        kept.append(e)
    # re-rank the featured list on the news-adjusted scores
    feat = sorted([e for e in kept if not e.get("also_qualified")], key=lambda e: -float(e["score"]))
    max_open = int(((P or {}).get("risk") or {}).get("max_open", 5))
    if state_dir:
        try:
            max_open = int(load_state(state_dir)["settings"].get("max_open_positions", max_open))
        except Exception:
            pass
    cap_left = max(0, max_open - int(scan.get("funds", {}).get("open_count", 0)))
    for i, e in enumerate(feat, 1):
        e["rank"] = i
        if i > cap_left:
            e["capacity_note"] = (f"Ranked #{i} tonight and you have {cap_left} free position slot(s). Place it only "
                                  "if you skip a higher-ranked setup, or once a position closes.")
        elif str(e.get("capacity_note", "")).startswith("Ranked #"):
            e.pop("capacity_note", None)
    kept = feat + [e for e in kept if e.get("also_qualified")]
    keep_ids = {e["id"] for e in kept}
    drop_ids = {e["id"] for e in entries} - keep_ids
    # rewrite alert docs
    for e in entries:
        p = out / "docs" / "alerts" / f"{e['id']}.json"
        if e["id"] in drop_ids:
            p.unlink(missing_ok=True)
        elif p.exists():
            p.write_text(json.dumps(e, default=str))
    # rebuild the writes manifest
    chunks = json.loads((out / "writes.json").read_text())
    flat = [w for ch in chunks for w in ch if not (w["collection"] == "alerts" and w["doc_id"] in drop_ids)]
    # scan doc + digest
    scan["counts"]["aplus"] = len(kept)
    scan["news_dropped"] = dropped
    now = cal.now_pt()
    try:
        import datetime as dt
        now = dt.datetime.fromisoformat(scan["started_at"])
    except Exception:
        pass
    P2 = P or {"risk": {"max_open": scan.get("funds", {}).get("max_open", 5)}, "grading": {"min_week": 3}}
    digest = make_digest(scan, kept, scan.get("positions", []), scan.get("near_misses", []),
                         scan.get("premarket_checks", []), scan["counts"].get("week_aplus", len(kept)), P2, now)
    if dropped:
        digest += "\n\nDropped after news check: " + "; ".join(f"{d['ticker']} ({d['reason']})" for d in dropped)
    scan["digest"] = digest
    (out / "docs" / "scans" / f"{scan['id']}.json").write_text(json.dumps(scan, default=str))
    (out / "digest.md").write_text(digest)
    (out / "scan.json").write_text(json.dumps({"scan": scan, "entries": kept}, indent=1, default=str))
    chunks = [flat[i:i + 45] for i in range(0, len(flat), 45)]
    (out / "writes.json").write_text(json.dumps(chunks, indent=1))
    return {"kept": len(kept), "dropped": dropped, "writes": len(flat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--news", nargs="*", default=[])
    a = ap.parse_args()
    print(json.dumps(apply(a.out, a.news, a.state), indent=1))


if __name__ == "__main__":
    main()
