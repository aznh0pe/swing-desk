"""Nov 19, 2026: the daily review sent a research save naming a file that did not exist (a timestamp typed 1520
for 1519); it was refused and resent. Writes are now handed over exactly as the engine made them."""
import json
import pathlib

from engine import research as R
from engine import writes as W

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _made(tmp_path):
    out = tmp_path / "rs4"
    writes = []
    R._write(out, writes, "research_results", "R_X-20261119-1519", {"id": "R_X-20261119-1519"})
    R._write(out, writes, "research", "R_X", {"id": "R_X"})
    R._finish(out, writes, {"verdict": "rejected"})
    return out


def test_show_prints_each_chunk_exactly_as_written_and_ready_to_send(tmp_path, capsys):
    out = _made(tmp_path)
    capsys.readouterr()  # the test command's own printout
    assert W.main(["show", str(out / "writes.json")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("CHUNK 1/1") and "exactly as printed" in lines[0]
    sent = json.loads(lines[1])
    assert sent == json.loads((out / "writes.json").read_text())[0]
    assert all(pathlib.Path(w["file_path"]).is_file() for w in sent)
    res = json.loads((out / "result.json").read_text())
    assert "engine.writes show" in res["writes"] and "writes_problems" not in res


def test_a_retyped_minute_is_caught_before_anything_is_sent(tmp_path, capsys):
    out = _made(tmp_path)
    capsys.readouterr()
    ch = json.loads((out / "writes.json").read_text())
    ch[0][0]["file_path"] = ch[0][0]["file_path"].replace("1519", "1520")  # the Nov 19 typo
    bad = tmp_path / "typed.json"
    bad.write_text(json.dumps(ch))
    assert W.main(["show", str(bad)]) == 1
    msg = capsys.readouterr().out
    assert msg.startswith("WRITES NOT READY") and "R_X-20261119-1520.json does not exist" in msg
    ch[0][0]["file_path"] = ch[0][0]["file_path"].replace("1520", "1519")
    ch[0][0]["doc_id"] = "R_X-20261119-1520"
    bad.write_text(json.dumps(ch))
    assert W.main(["check", str(bad)]) == 1 and "does not match its file" in capsys.readouterr().out


def test_every_prompt_that_sends_research_or_lesson_writes_uses_the_printed_chunks():
    for f in ("learning.md", "backup_learning.md", "checks.md", "weekly.md", "backup_weekly.md"):
        txt = (ROOT / "ops" / "generated" / f).read_text()
        assert "python3 -m engine.writes show" in txt and "never type a doc id, a time or a file path" in txt, f
    learn = (ROOT / "ops" / "learning_prompt.md").read_text()
    assert learn.count("engine.writes show") >= 3  # review writes, research writes, lesson writes


def test_no_scheduled_prompt_sends_a_writes_file_any_other_way():
    """Nov 19 follow-up: every scheduled task that saves (scans, backups, review, checks, weekly) uses the printed
    chunks; a template that mentions a writes file and a batch without engine.writes show fails here."""
    gen = sorted((ROOT / "ops" / "generated").glob("*.md"))
    assert len(gen) >= 15
    for f in gen:
        txt = f.read_text()
        if "writes.json" in txt and '"batch"' in txt:
            assert "python3 -m engine.writes show" in txt, f.name
            assert "for each chunk call ArtifactData" not in txt and "for each chunk in $W/out/writes.json call" not in txt, f.name
    sched = json.loads((ROOT / "ops" / "schedule.json").read_text())["tasks"]
    for name, t in sched.items():
        p = ROOT / t["prompt"]
        if p.exists() and "writes.json" in p.read_text():
            assert "engine.writes show" in p.read_text(), name
