import json
import pathlib

from engine import version as V

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_published_engine_copy_matches_the_code():
    m = json.loads((ROOT / "ops" / "engine_manifest.json").read_text())
    hint = "engine changed: run python3 ops/publish_engine.py, then republish the dashboard with those files"
    assert m["files"] == V.files(ROOT), hint
    assert m["fingerprint"] == V.fingerprint(ROOT), hint


def test_engine_installs_missing_packages_itself_once(monkeypatch, tmp_path):
    import types
    import engine
    state = {"installed": False, "calls": []}
    real = engine._ilu.find_spec
    monkeypatch.setattr(engine._ilu, "find_spec",
                        lambda m: None if (m == "pyarrow" and not state["installed"]) else real(m))
    monkeypatch.setattr(engine, "_FAILED_MARK", str(tmp_path / "failed"))
    monkeypatch.delenv("SWING_DESK_NO_AUTO_INSTALL", raising=False)

    def ok_run(cmd, **kw):
        state["calls"].append(cmd)
        state["installed"] = True
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(engine._sp, "run", ok_run)
    logs = []
    assert engine.ensure_deps(logs.append) == ["pyarrow"]
    assert state["calls"][0][-1] == "pyarrow" and "--break-system-packages" in state["calls"][0]
    assert "routine setup, not a run problem" in logs[0]
    assert engine.ensure_deps(logs.append) == [] and len(state["calls"]) == 1   # nothing missing any more

    # no network: it says so once, marks the machine, and does not retry in every later process
    state.update(installed=False, calls=[])
    monkeypatch.setattr(engine._sp, "run", lambda cmd, **kw: (state["calls"].append(cmd),
                                                              types.SimpleNamespace(returncode=1, stdout="", stderr="x"))[1])
    logs.clear()
    assert engine.ensure_deps(logs.append) == [] and len(state["calls"]) == 2   # with and without the flag
    assert "could not install pyarrow" in logs[0] and (tmp_path / "failed").exists()
    assert engine.ensure_deps(logs.append) == [] and len(state["calls"]) == 2
