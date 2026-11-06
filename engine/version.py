"""Which engine is running: a fingerprint of its code and where it came from.

The same engine runs from two places: my GitHub repo (attached to the session) and an identical copy
published as files of the dashboard artifact, which scheduled runs use when the repo can't be attached.
The fingerprint shows which code produced a scan, and lets the daily review spot a stale copy.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def files(root: pathlib.Path = ROOT) -> list[str]:
    """The engine's files, relative to its root: everything the scanner, review and recalibration need
    (kept inside engine/, since file readers screen folders named config/)."""
    out = sorted(p.relative_to(root).as_posix() for p in (root / "engine").glob("*.py"))
    out += sorted(p.relative_to(root).as_posix() for p in (root / "engine").glob("*.json") if p.name != "MANIFEST.json")
    return out


def fingerprint(root: pathlib.Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in files(root):
        h.update(rel.encode())
        h.update(b"\0")
        h.update((root / rel).read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def source(root: pathlib.Path = ROOT) -> str:
    return "repo" if (root / ".git").exists() else "dashboard copy"


def info() -> dict:
    try:
        return {"fingerprint": fingerprint(), "source": source()}
    except Exception as e:  # never let bookkeeping stop a scan
        return {"fingerprint": None, "source": source(), "error": str(e)[:160]}


if __name__ == "__main__":
    print(json.dumps(info()))
