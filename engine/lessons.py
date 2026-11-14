"""The desk's permanent lessons: what testing and live trading have taught it, one dated line each.

    python -m engine.lessons list --state S
    python -m engine.lessons add  --state S --out O --topic T --lesson "..." [--evidence "..."] [--by "daily review"]

The research log remembers every idea and its verdict; this registry remembers the conclusions that outlive any
one idea (a ceiling that was measured, a rule that was tried and did not help, a trap that cost a week), so the
next design starts from them instead of rediscovering them. It lives in the dashboard database as learning/lessons
and is never trimmed. `add` writes O/docs/learning/lessons.json (the whole document with the new line) and
O/writes.json for ArtifactData; a lesson that says the same thing as one already on file is not added twice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

from . import calendar as cal
from .portfolio import load_collection

TOPICS = ("strategy design", "grading", "portfolio", "universe", "market conditions", "execution", "operations", "research method")


def load(state_dir) -> dict:
    doc = ((load_collection(state_dir, "learning") if state_dir else {}).get("lessons") or {}).get("doc") or {}
    doc.setdefault("items", [])
    return doc


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _same(a: str, b: str) -> bool:
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    return bool(wa and wb) and len(wa & wb) / max(1, min(len(wa), len(wb))) >= 0.85


def add(doc: dict, topic: str, lesson: str, evidence: str = "", by: str = "daily review", now=None) -> tuple[dict, bool]:
    """(document, added). Refuses an empty lesson and one already on file."""
    now = now or cal.now_pt()
    lesson = " ".join(str(lesson).split())
    if len(lesson) < 20 or any(_same(lesson, it.get("lesson", "")) for it in doc.get("items", [])):
        return doc, False
    out = dict(doc)
    out["items"] = list(doc.get("items", [])) + [{"date": now.date().isoformat(), "topic": topic if topic in TOPICS else "strategy design",
                                                  "lesson": lesson, "evidence": " ".join(str(evidence or "").split()), "by": by}]
    out.update(id="lessons", updated_at=now.isoformat(timespec="seconds"),
               about=("What the desk has learned for good: measured ceilings, rules that were tried and did not help, traps. "
                      "Read before designing or changing anything; added to by the daily review, the weekly recalibration and the chat."))
    return out, True


def brief(doc: dict, limit: int = 60) -> list:
    """The lessons as short lines, newest first, for `research next` and the prompts."""
    items = sorted(doc.get("items", []), key=lambda it: str(it.get("date", "")), reverse=True)[:limit]
    return [f"[{it.get('date')}] {it.get('topic')}: {it.get('lesson')}" for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "add"])
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--topic", default="strategy design", help=" | ".join(TOPICS))
    ap.add_argument("--lesson", default="")
    ap.add_argument("--evidence", default="")
    ap.add_argument("--by", default="daily review")
    a = ap.parse_args()
    doc = load(a.state)
    if a.cmd == "list":
        print(json.dumps({"count": len(doc["items"]), "lessons": brief(doc, 500)}, indent=1))
        return
    if not a.out:
        ap.error("add needs --out")
    new, added = add(doc, a.topic, a.lesson, a.evidence, a.by)
    out = pathlib.Path(a.out)
    res = {"added": added, "count": len(new["items"])}
    if added:
        p = out / "docs" / "learning" / "lessons.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(new, default=str))
        exists = bool(doc.get("items"))
        w = {"op": "set", "collection": "learning", "doc_id": "lessons", "file_path": str(p.resolve())}
        if exists:
            w["if_version"] = "VERSION_OF_learning/lessons"
        (out / "writes.json").write_text(json.dumps([[w]], indent=1))
    else:
        res["note"] = "not added: empty, too short, or the same as a lesson already on file"
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
