import json
import pathlib

from engine import version as V

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_published_engine_copy_matches_the_code():
    m = json.loads((ROOT / "ops" / "engine_manifest.json").read_text())
    hint = "engine changed: run python3 ops/publish_engine.py, then republish the dashboard with those files"
    assert m["files"] == V.files(ROOT), hint
    assert m["fingerprint"] == V.fingerprint(ROOT), hint
