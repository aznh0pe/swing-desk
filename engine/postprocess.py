"""Apply the scan session's news checks to a finished scan's output.

    python -m engine.postprocess --out <scan-out-dir> --state <state-dir> --news "NVDA=clean" "XYZ=caution:Analyst downgrade" "ABC=avoid:Agreed to be acquired"

Post-close / pre-close scans (new setups):
* clean   -> news_check "clean"
* caution -> news_check "caution", note, score -8 (dropped if it falls below the A+ cut-off)
* avoid   -> setup removed (doc file, write entry and digest line)
Pre-market scans (today's orders and open positions):
* avoid on a pending order   -> CANCEL: the order is withdrawn on the dashboard
* caution on a pending order -> CAUTION line, order kept
* avoid / caution on an open position -> REVIEW line
Then re-ranks, refreshes the alert overlay (alerts that already exist on the dashboard), the
digest, the scan document, the Scan-now progress record and writes.json so everything agrees.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re

from .scan import _utc_now, build_overlay, email_headline, load_params, make_digest, mark_caution, run_done_doc
from .portfolio import load_state
from . import calendar as cal


def _verdicts(news: list[str]) -> dict:
    out = {}
    for item in news:
        if "=" not in item:
            continue
        tk, rest = item.split("=", 1)
        kind, _, note = rest.partition(":")
        out[tk.strip().upper()] = (kind.strip().lower(), note.strip() or None)
    return out


SEVERITY = {"clean": 0, "caution": 1, "avoid": 2}
# Version of the verdict rules in the scan prompts. Verdicts given under older rules are not carried
# forward (v1 marked sector selloffs as caution; v2 judges company-specific news only).
NEWS_RULES = 2
FEATURED, PER_SECTOR = 8, 2
_FROM = re.compile(r"\s*\(from the [^()]* news check\)")


def prior_verdicts(state: dict | None, session_day) -> dict:
    """Caution / avoid verdicts that earlier scans for the same session already gave (newest first)."""
    out = {}
    scans = sorted((v["doc"] for v in ((state or {}).get("scans") or {}).values()),
                   key=lambda d: str(d.get("started_at", "")), reverse=True)
    for sc in scans:
        if str(sc.get("session_day")) != str(session_day) or sc.get("status") == "failed":
            continue
        if int(sc.get("news_rules") or 1) < NEWS_RULES:
            continue  # judged under older rules
        try:
            when = dt.datetime.fromisoformat(sc["started_at"]).strftime("%-I:%M %p").lower()
        except Exception:
            when = "earlier"
        found = []
        for d in sc.get("news_dropped") or []:
            r = str(d.get("reason") or "")
            if r.startswith("score fell below A+ after news caution:"):
                found.append((d["ticker"], "caution", r.split(":", 1)[1].strip()))
            else:
                found.append((d["ticker"], "avoid", r))
        for aid, u in (sc.get("alert_updates") or {}).items():
            if u.get("news_check") == "caution":
                tk = u.get("ticker") or (aid.split("_")[2] if aid.count("_") >= 3 else None)
                if tk:
                    found.append((tk, "caution", u.get("news_note") or "notable news"))
        for tk, kind, note in found:
            tk = str(tk).upper()
            note = _FROM.sub("", str(note or "")).strip() or "notable news"
            if tk not in out or SEVERITY[kind] > SEVERITY[out[tk][0]]:
                out[tk] = (kind, f"{note} (from the {when} news check)")
    return out


def apply(out_dir: str, news: list[str], state_dir: str | None = None, issues: list[str] | None = None) -> dict:
    out = pathlib.Path(out_dir)
    sj = json.loads((out / "scan.json").read_text())
    scan, entries, ctx = sj["scan"], sj["entries"], sj.get("ctx") or {}
    state = load_state(state_dir) if state_dir else None
    P = load_params(state, None) if state else None
    cutoff = (P or {}).get("grading", {}).get("aplus_score", 80) if P else scan.get("aplus_cutoff", 80)
    verdicts = _verdicts(news)
    # a later check of the same session never softens an earlier caution or avoid (win rate first)
    kept_prior = []
    for tk, (kind, note) in prior_verdicts(state, scan.get("session_day")).items():
        cur = verdicts.get(tk)
        if cur is None or SEVERITY.get(cur[0], 0) < SEVERITY[kind]:
            verdicts[tk] = (kind, note)
            kept_prior.append(tk)
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
    # pre-market: news on today's orders and open positions
    for c in scan.get("premarket_checks") or []:
        v = verdicts.get(str(c.get("ticker", "")).upper())
        if not v or v[0] == "clean":
            continue
        kind, note = v
        if kind == "avoid":
            c.update(verdict="cancel", text=f"News: {note or 'material news'}. Cancel today's order.")
        elif c.get("verdict") != "cancel":
            c.update(verdict="caution", text=f"News: {note or 'notable news'}. Check it before letting the order fill.")
    pos_news = []
    for pos in scan.get("positions") or []:
        v = verdicts.get(str(pos.get("ticker", "")).upper())
        if v and v[0] in ("avoid", "caution"):
            pos_news.append({"ticker": pos["ticker"], "verdict": v[0], "reason": v[1] or "material news"})
    if pos_news:
        scan["position_news"] = pos_news
    # re-rank on the news-adjusted scores. Every kept setup competes for the top 8 again, so a setup that was
    # only "also qualified" (sector cap or outside the top 8) moves up when the news check drops one ahead of it.
    if all("sector_key" in e for e in kept):
        held = dict(ctx.get("held_sectors") or {})
        feat, extra, fsec = [], [], {}
        for e in sorted(kept, key=lambda e: -float(e["score"])):
            sk = e.get("sector_key")
            n = (held.get(sk, 0) + fsec.get(sk, 0)) if sk else 0
            if len(feat) < FEATURED and (not sk or n < PER_SECTOR):
                feat.append(e)
                e.pop("also_qualified", None)
                if str(e.get("capacity_note", "")).startswith("Also qualified"):
                    e.pop("capacity_note", None)
                if sk:
                    fsec[sk] = fsec.get(sk, 0) + 1
            else:
                e["also_qualified"], e["rank"] = True, None
                why = f"{n} positions/featured setups already in {sk}" if sk and n >= PER_SECTOR else "outside tonight's top 8"
                e["capacity_note"] = f"Also qualified ({why}). Listed so you can swap into it if it suits you better."
                extra.append(e)
        kept = feat + extra
    feat = sorted([e for e in kept if not e.get("also_qualified")], key=lambda e: -float(e["score"]))
    max_open = int(((P or {}).get("risk") or {}).get("max_open", 5))
    if state:
        try:
            max_open = int(state["settings"].get("max_open_positions", max_open))
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
    chunks = json.loads((out / "writes.json").read_text())
    flat = [w for ch in chunks for w in ch if not (w["collection"] == "alerts" and w["doc_id"] in drop_ids)]
    # scan doc, overlay for alerts that already exist, digest
    scan["counts"]["aplus"] = len(kept)
    scan["counts"]["topups"] = sum(1 for e in kept if e.get("topup_note"))
    scan["news_dropped"] = dropped
    scan["news_checked"] = sorted(verdicts)
    scan["news_rules"] = NEWS_RULES
    # anything that went wrong in the run itself (a step that failed and needed a fallback or a retry, a fix
    # made to the code, results that had to be resent): shown first in the email, the headline and the dashboard
    for t in issues or []:
        t = " ".join(str(t).split())[:400]
        if t and t not in scan.setdefault("run_issues", []):
            scan["run_issues"].append(t)
    if kept_prior:
        scan["news_kept_from_earlier"] = sorted(kept_prior)
    if ctx:
        scan["alert_updates"] = build_overlay(scan, kept, ctx)
    now = cal.now_pt()
    try:
        now = dt.datetime.fromisoformat(scan["started_at"])
    except Exception:
        pass
    P2 = P or {"risk": {"max_open": scan.get("funds", {}).get("max_open", 5)}, "grading": {"min_week": 3}}
    digest = make_digest(scan, kept, scan.get("positions", []), scan.get("near_misses", []),
                         scan.get("premarket_checks", []), scan["counts"].get("week_aplus", len(kept)), P2, now)
    if dropped:
        digest += "\n\nDropped after news check: " + "; ".join(f"{d['ticker']} ({d['reason']})" for d in dropped)
    digest = mark_caution(digest)
    scan["digest"] = digest
    (out / "docs" / "scans" / f"{scan['id']}.json").write_text(json.dumps(scan, default=str))
    if scan.get("run_token"):
        rp = out / "docs" / "runs" / f"{scan['run_token']}-s6.json"
        if rp.exists():
            rp.write_text(json.dumps(run_done_doc(scan, scan["run_token"], _utc_now()), default=str))
    (out / "digest.md").write_text(digest)
    (out / "headline.txt").write_text(email_headline(scan))
    (out / "scan.json").write_text(json.dumps({"scan": scan, "entries": kept, "ctx": ctx}, indent=1, default=str))
    chunks = [flat[i:i + 45] for i in range(0, len(flat), 45)]
    (out / "writes.json").write_text(json.dumps(chunks, indent=1))
    return {"kept": len(kept), "dropped": dropped, "writes": len(flat), "headline": email_headline(scan),
            "withdrawn": sum(1 for v in (scan.get("alert_updates") or {}).values() if v.get("withdrawn"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--news", nargs="*", default=[])
    ap.add_argument("--issue", action="append", default=[], help="a problem in this run, in plain words (repeatable)")
    a = ap.parse_args()
    print(json.dumps(apply(a.out, a.news, a.state, a.issue), indent=1))


if __name__ == "__main__":
    main()
