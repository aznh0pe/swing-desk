"""Refresh ops/engine_manifest.json, the list of files in the engine copy published with the dashboard.

Run this after any change to engine/ or config/, then republish the dashboard artifact with these
files (the printed map), so scheduled runs that can't attach the repo run the same code.
tests/test_engine_manifest.py fails until the manifest matches the code.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import version as V  # noqa: E402


def served(rel: str) -> str:
    """The name a file is published under. Nov 14, 2026: the artifact publisher this chat now has cannot serve
    .py files, so Python files are published as <name>.py.txt and the runs drop the .txt after reading them."""
    return rel + ".txt" if rel.endswith(".py") else rel


def main() -> None:
    files = V.files(ROOT)
    manifest = {
        "fingerprint": V.fingerprint(ROOT),
        "files": files,
        "published": [served(f) for f in files],
        "about": ("Swing Desk engine, identical to engine/ in the GitHub repo. Scheduled runs that can't attach the repo "
                  "read the \"published\" files with the Artifact tool, drop the .txt from the Python files (they are "
                  "served as <name>.py.txt) and run them."),
    }
    (ROOT / "ops" / "engine_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    publish = {served(f): f for f in files}
    publish["engine/MANIFEST.json"] = "ops/engine_manifest.json"
    print(json.dumps({"fingerprint": manifest["fingerprint"], "files": publish,
                      "remove_old": [f for f in files if served(f) != f]}, indent=1))


if __name__ == "__main__":
    main()
