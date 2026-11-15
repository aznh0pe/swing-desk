"""The on-demand checks run's journal entry is built by the engine and saved with the document's version.

Nov 14, 2026: one checks run looked for the wider list's result under a key that did not exist in the tier report,
and two later runs the same day could not save their entry at all (the day's document existed, and a "set" of an
existing document is refused without its version). These tests pin both fixes: the engine composes the entry from
its own report files, a second run the same day is merged into the first with the version to pin, and the prompt
tells the run to save the engine's file and confirm it instead of composing anything.
"""
import json
import pathlib
import re
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine import calibrate as CB  # noqa: E402

NOW = pd.Timestamp("2026-11-16T13:40:00-08:00")
WL = "Wider list (weekly tier check): The wider list sends A+ setups only."


def _reports(tmp_path, promoted=True, tiers_changelog=True):
    promo, tiers, rs = tmp_path / "promo", tmp_path / "tiers", tmp_path / "rs1"
    for d in (promo, tiers, rs / "docs" / "research_results"):
        d.mkdir(parents=True)
    (promo / "promote_report.json").write_text(json.dumps({"note": "Promoted Demo Dip mid-week.", "promoted": ["R_DEMO"] if promoted else []}))
    ext = {"status": "aplus_only", "size_mult": 0.75, "reason": "The wider list sends A+ setups only."}
    (tiers / "tiers_report.json").write_text(json.dumps({"tiers": {"ext": ext}, **({"changelog": WL} if tiers_changelog else {})}))
    doc = {"params": {"tiers": {"ext": ext}}, "changelog_entry": ["Promoted Demo Dip.", WL] if promoted else [WL]}
    ((promo if promoted else tiers) / "doc_params.json").write_text(json.dumps(doc))
    (rs / "result.json").write_text(json.dumps({"results": [{"idea": "R_DEMO", "verdict": "candidate"}]}))
    (rs / "docs" / "research_results" / "R_DEMO-1.json").write_text(json.dumps({"idea_id": "R_DEMO", "label": "Demo Dip"}))
    return promo, tiers, rs


def test_first_run_of_the_day_builds_the_whole_entry(tmp_path):
    promo, tiers, rs = _reports(tmp_path)
    (tmp_path / "issues.txt").write_text("")
    r = CB.run_checksnote(promo, tiers, rs, tmp_path / "jv", tmp_path / "issues.txt", tmp_path / "jn", now=NOW)
    assert r["doc_id"] == "2026-11-16-checks" and r["if_version"] is None and r["exists"] is False
    doc = json.loads(pathlib.Path(r["file_path"]).read_text())
    assert doc["id"] == "2026-11-16-checks" and doc["date"] == "2026-11-16" and doc["created_at"] == doc["updated_at"]
    assert doc["changes"] == ["Promoted Demo Dip.", WL]
    assert doc["insights"][0].startswith("1 queued research idea was tested first: Demo Dip (candidate)")
    assert "Promoted Demo Dip mid-week." in doc["insights"] and WL in doc["insights"]
    assert "issues" not in doc and len(doc["runs"]) == 1


def test_wider_list_result_has_fallbacks_so_no_run_has_to_look_it_up(tmp_path):
    # the report's top-level text missing: the entry falls back to the tier block's reason, never to nothing
    promo, tiers, rs = _reports(tmp_path, promoted=False, tiers_changelog=False)
    doc = CB.checks_note(promo, tiers, None, None, None, NOW)["doc"]
    assert doc["insights"][-1] == "Wider list: The wider list sends A+ setups only."
    # no tier report at all (the tier check was skipped): the saved parameters still carry the status
    (tiers / "tiers_report.json").unlink()
    doc = CB.checks_note(promo, tiers, None, None, None, NOW)["doc"]
    assert doc["insights"][-1] == "Wider list: The wider list sends A+ setups only."
    # nothing ran: plain sentences, and the "no change" line
    doc = CB.checks_note(tmp_path / "none", tmp_path / "none2", None, None, None, NOW)["doc"]
    assert doc["changes"] == [CB.NO_CHANGE] and "wrote no result" in doc["insights"][0] and "wrote no result" in doc["insights"][1]


def test_second_run_the_same_day_is_merged_and_pins_the_version(tmp_path):
    promo, tiers, rs = _reports(tmp_path, promoted=False)
    jv = tmp_path / "jv" / "journal"
    jv.mkdir(parents=True)
    prev = {"id": "2026-11-16-checks", "date": "2026-11-16", "created_at": "2026-11-16T09:00:00-08:00", "insights": ["morning result"],
            "changes": ["Promoted Morning Dip."], "issues": ["the first write failed"],
            "issues_resolved": [{"issue": "the first write failed", "fix": "fixed"}], "evening": {"kept": True}}
    (jv / "2026-11-16-checks.json").write_text(json.dumps({"doc": prev, "version": 7}))
    (tmp_path / "issues.txt").write_text("the data feed is not the wide list yet\n\n")
    r = CB.run_checksnote(promo, tiers, rs, tmp_path / "jv", tmp_path / "issues.txt", tmp_path / "jn", now=NOW)
    assert r["if_version"] == 7 and r["exists"] is True and r["runs_today"] == 2
    doc = json.loads(pathlib.Path(r["file_path"]).read_text())
    assert doc["created_at"] == "2026-11-16T09:00:00-08:00" and doc["updated_at"] == NOW.isoformat(timespec="seconds")
    assert doc["changes"] == ["Promoted Morning Dip.", WL]          # the day's changes accumulate
    assert doc["insights"][-1] == WL and "morning result" not in doc["insights"]   # the headline is the latest run
    assert doc["runs"][0]["insights"] == ["morning result"] and doc["runs"][0]["issues"] == ["the first write failed"]
    assert doc["runs"][1]["changes"] == [WL] and all(WL not in i for i in doc["runs"][1].get("insights", []))   # a run's record does not repeat itself
    assert doc["issues"] == ["the data feed is not the wide list yet"]      # the banner follows the latest run
    assert doc["issues_resolved"] == prev["issues_resolved"] and doc["evening"] == {"kept": True}   # nothing else is lost
    # a clean third run clears the open problem
    (jv / "2026-11-16-checks.json").write_text(json.dumps({"doc": doc, "version": 8}))
    (tmp_path / "issues.txt").write_text("")
    r3 = CB.run_checksnote(promo, tiers, rs, tmp_path / "jv", tmp_path / "issues.txt", tmp_path / "jn3", now=NOW + pd.Timedelta(hours=1))
    doc3 = json.loads(pathlib.Path(r3["file_path"]).read_text())
    assert r3["if_version"] == 8 and len(doc3["runs"]) == 3 and "issues" not in doc3


def test_a_missing_document_export_is_not_mistaken_for_an_entry(tmp_path):
    promo, tiers, rs = _reports(tmp_path)
    jv = tmp_path / "jv"
    jv.mkdir()
    (jv / "2026-11-16-checks.json").write_text(json.dumps({"exists": False}))
    r = CB.run_checksnote(promo, tiers, rs, jv, None, tmp_path / "jn", now=NOW)
    assert r["if_version"] is None and r["exists"] is False


def test_an_export_without_a_version_still_says_the_entry_exists(tmp_path):
    """Nov 14, 2026 (late): the saved copy of the entry carried no version, the engine printed none, and the run had to
    notice by itself that the entry existed. The engine now says so, and the run pins the version its own read reported."""
    promo, tiers, rs = _reports(tmp_path)
    jv = tmp_path / "jv"
    jv.mkdir()
    (jv / "2026-11-16-checks.json").write_text(json.dumps({"id": "2026-11-16-checks", "date": "2026-11-16", "created_at": "2026-11-16T09:00:00-08:00",
                                                         "insights": ["morning"], "changes": ["Promoted Morning Dip."]}))
    r = CB.run_checksnote(promo, tiers, rs, jv, None, tmp_path / "jn", now=NOW)
    assert r["exists"] is True and r["if_version"] is None and r["runs_today"] == 2
    step = (ROOT / "ops" / "generated" / "checks.md").read_text()
    assert 'When exists is true' in step and 'the version that "get" just reported' in step


def test_verify_tells_saved_from_not_saved(tmp_path):
    promo, tiers, rs = _reports(tmp_path)
    r = CB.run_checksnote(promo, tiers, rs, None, None, tmp_path / "jn", now=NOW)
    doc = json.loads(pathlib.Path(r["file_path"]).read_text())
    jc = tmp_path / "jc"
    jc.mkdir()
    assert CB.run_checksnote(None, None, None, None, None, tmp_path / "jn", verify_dir=jc)["result"] == "NOT SAVED"   # nothing came back
    (jc / "2026-11-16-checks.json").write_text(json.dumps({"doc": {**doc, "updated_at": "2026-11-16T09:00:00-08:00"}, "version": 3}))
    assert CB.run_checksnote(None, None, None, None, None, tmp_path / "jn", verify_dir=jc)["result"] == "NOT SAVED"   # an older entry is still there
    (jc / "2026-11-16-checks.json").write_text(json.dumps({"doc": doc, "version": 4}))
    v = CB.run_checksnote(None, None, None, None, None, tmp_path / "jn", verify_dir=jc)
    assert v["result"] == "saved" and v["version"] == 4


def test_command_line_runs_without_market_data(tmp_path):
    promo, tiers, rs = _reports(tmp_path)
    out = subprocess.run([sys.executable, "-m", "engine.calibrate", "checksnote", "--promo", str(promo), "--tiers", str(tiers), "--rs", str(rs),
                          "--prev", str(tmp_path / "jv"), "--issues", str(tmp_path / "none.txt"), "--out", str(tmp_path / "jn")],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["doc_id"].endswith("-checks") and r["if_version"] is None and pathlib.Path(r["file_path"]).exists()


def test_the_checks_prompt_saves_the_engine_entry_with_its_version_and_confirms_it():
    text = (ROOT / "ops" / "generated" / "checks.md").read_text()
    step = text[text.index("7. Journal"):text.index("Run problems:")]
    assert "engine.calibrate checksnote --promo" in step and "--prev $W/jv" in step and "--verify $W/jc" in step
    assert "if_version" in step and "exists" in step and "NOT SAVED" in step and "could not be saved" in step
    # the run composes nothing by hand: no report file or key is named in the journal step
    assert not re.search(r"tiers_report\.json|promote_report\.json|changelog_entry|tiers\.ext", step)


def test_every_journal_write_in_a_prompt_is_confirmed():
    """A prompt that sets a journal document must also read it back: a refused write may not pass in silence."""
    for name in ("checks", "weekly", "learning"):
        text = (ROOT / "ops" / "generated" / f"{name}.md").read_text()
        assert re.search(r"[Cc]onfirm: ArtifactData \"get\" (strategy/params and )?journal/", text), name
