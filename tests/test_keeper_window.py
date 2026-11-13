"""The market-data refresh loop runs on trading days from 6:30 AM ET until 40 minutes after the close.

Nov 13, 2026: GitHub ran the scheduled data refresh about once an hour, so the 12:25 PM PT pre-close scan found
55-minute-old prices. The workflow now keeps itself running through the market day (pipeline/refresh_loop.sh).
"""
import datetime as dt
import importlib.util
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("keeper_window", ROOT / "pipeline" / "keeper_window.py")
kw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kw)
ET = kw.ET


def at(s):
    return dt.datetime.fromisoformat(s).replace(tzinfo=ET)


def test_covers_the_pre_market_check_the_session_and_the_final_bars():
    assert not kw.keep_running(at("2026-11-13T06:29"))
    assert kw.keep_running(at("2026-11-13T06:30"))       # 3:30 AM PT
    assert kw.keep_running(at("2026-11-13T08:17"))       # the 5:17 AM PT pre-market check
    assert kw.keep_running(at("2026-11-13T15:25"))       # the 12:25 PM PT pre-close scan
    assert kw.keep_running(at("2026-11-13T16:39"))       # the final daily bars land 10-20 minutes after the close
    assert not kw.keep_running(at("2026-11-13T16:40"))


def test_off_on_closed_days_and_shorter_on_half_days():
    assert not kw.keep_running(at("2026-11-14T10:00"))   # Saturday
    assert not kw.keep_running(at("2026-11-26T10:00"))   # Thanksgiving
    assert kw.keep_running(at("2026-11-27T13:39")) and not kw.keep_running(at("2026-11-27T13:40"))  # 1 PM ET close


def test_loop_script_is_valid_and_the_workflow_runs_it():
    subprocess.run(["bash", "-n", str(ROOT / "pipeline" / "refresh_loop.sh")], check=True)
    wf = (ROOT / ".github" / "workflows" / "market-data.yml").read_text()
    assert "bash pipeline/refresh_loop.sh" in wf and "actions: write" in wf and "timeout-minutes: 350" in wf
