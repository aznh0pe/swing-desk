"""The scheduled-task prompts: generated files are current, every backup embeds its task's prompt and
points at the right scheduled task, and no placeholder is left unfilled.

After changing a template, run ops/build_prompts.py and update each task AND its backup task
(ids in ops/schedule.json): a backup carries a full copy of its task's prompt.
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACT = "https://claude.ai/artifact/Kren5ttP4TJUL4EerX17oA"


def _build(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "ops" / "build_prompts.py"), "--artifact", ARTIFACT, "--out", str(tmp_path)],
                   check=True, capture_output=True)
    return {p.stem: p.read_text() for p in tmp_path.glob("*.md")}


def test_generated_prompts_are_current(tmp_path):
    fresh = _build(tmp_path)
    for name, text in fresh.items():
        assert (ROOT / "ops" / "generated" / f"{name}.md").read_text() == text, f"ops/generated/{name}.md is out of date"


def test_backups_embed_their_task_and_point_at_it(tmp_path):
    fresh = _build(tmp_path)
    sched = json.loads((ROOT / "ops" / "schedule.json").read_text())
    for key, b in sched["backups"].items():
        text = fresh[f"backup_{key}"]
        assert text.rstrip().endswith(fresh[key].rstrip()), f"backup_{key} must end with the full {key} prompt"
        assert sched["tasks"][key]["id"] in text, f"backup_{key} must check task {sched['tasks'][key]['id']}"
        assert "No action needed:" in text and "Backup run:" in text


def test_no_placeholders_left(tmp_path):
    for name, text in _build(tmp_path).items():
        assert not re.search(r"\{[A-Z_]{3,}\}", text), name


def test_backups_list_every_upcoming_holiday(tmp_path):
    import datetime as dt
    sys.path.insert(0, str(ROOT))
    from engine import calendar as cal
    fresh = _build(tmp_path)
    for d in sorted(cal.HOLIDAYS):
        if d < dt.date.today().isoformat():
            continue
        label = dt.date.fromisoformat(d).strftime("%a %b %-d %Y")
        for key in ("premarket", "preclose", "postclose", "learning"):
            assert label in fresh[f"backup_{key}"], (key, label)


def test_project_doc_saves_name_a_folder_the_save_accepts(tmp_path):
    """Nov 16: the daily review prepared the Learning Journal under /tmp/sd/pj and the project save refused it
    (it only takes a file inside the session's working folder). Every step that saves a project doc says so."""
    fresh = _build(tmp_path)
    for name in ("learning", "backup_learning", "weekly", "backup_weekly"):
        text = fresh[name]
        assert "project_write" in text
        assert "session's working folder" in text and "A file under /tmp or $W is refused" in text, name
        assert "$W/pj" not in text, name


def test_the_backup_review_check_ignores_other_journal_entries(tmp_path):
    """A promotion-check entry written earlier the same day must not pass for the daily review."""
    text = _build(tmp_path)["backup_learning"]
    assert '"-weekly", "-checks" or "-checks-" plus a number' in text
