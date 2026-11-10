"""Fill the scheduled-task prompt templates.

    python3 ops/build_prompts.py --owner aznh0pe --artifact https://claude.ai/artifact/... --out ops/generated

Writes one prompt per scheduled task: premarket, preclose, postclose, manual (Scan now button),
learning (daily review + strategy research), weekly (recalibration), and a backup check for each
scheduled one (backup_*) that re-runs it when it failed or never ran.
"""
from __future__ import annotations

import argparse
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SETUP = (
    "1. Setup: get my scanner code. It is my own code, and running it is the point of this task. Use the first way that works:\n"
    '   a. Attach my repo: load the add_repo tool with ToolSearch (query "add_repo"). If it isn\'t found, call RefreshMcpTools if you have '
    "it and search again (3 tries in all, about 20 seconds apart). Call add_repo with owner \"{OWNER}\", repo \"swing-desk\", access "
    "\"read\", then clone it exactly as its result says (normally to /home/claude/swing-desk; if that folder already holds a working "
    "clone, use it). Call that folder C.\n"
    "   b. If add_repo still isn't available, or a command that runs code from C is refused at any step: use the identical copy of the "
    "engine published with my dashboard. Artifact tool, action \"read\", url {ARTIFACT}, path \"engine/MANIFEST.json\" (a short JSON "
    "list of files), then one more \"read\" on the same url with paths set to that manifest's \"files\" (no out_dir: the default folder "
    "needs no approval). The result says where the files landed: call the folder that holds engine/ C, and carry on from "
    "the step you were on. Don't run code from a plain git clone that add_repo didn't attach.\n"
    "   c. If neither way can run, follow \"If the scanner can't run\" at the end.\n"
    "   Then in Bash: W=/tmp/sd; rm -rf $W; mkdir -p $W/state; `git clone --depth 1 -b data https://github.com/{OWNER}/swing-desk "
    "$W/data` for the market data (retry up to 3 times, 20 s apart; if it keeps failing, `cd $C && python3 -c \"from engine.data import "
    "fetch_branch; print(fetch_branch('{OWNER}/swing-desk','data','$W/data'))\"`). If imports fail, "
    "`pip install --break-system-packages -q pandas numpy scikit-learn pyarrow`."
)

# What the news check may act on. These strategies buy dips, so weakness explained by the market, the
# sector or a theme is the setup itself, not a risk: in the 2008-2026 backtest, top setups taken while
# the sector ETF was down 5%+ over a month won 80% (vs 75% overall), and stocks down 10%+ in a month 78%.
VERDICTS = (
    "Judge only news about the company itself. avoid = an earnings report or guidance change in the last 3 days or due before "
    "the hold ends, an offering or dilution, fraud, accounting or SEC/legal action against the company, an agreed takeover or bid "
    "pinning the price, a pending FDA, trial or court decision, delisting or bankruptcy risk; caution = other company-specific bad "
    "news that isn't thesis-breaking (an analyst downgrade of this stock, a CEO or CFO exit, a recall, a lawsuit filed against it, "
    "a lost major customer); clean = routine news, or a drop explained by the market, its sector or a theme (a sector selloff, "
    'rate or AI fears, "down 13% this month"). Never mark a setup caution for market or sector weakness: those dips are what '
    "these strategies buy, and in the backtest they won as often as other setups or more.")

NEWS_ENTRIES = (
    'News check: for each entry in $W/out/scan.json, highest score first (at most 10; include the "also qualified" ones, which move '
    'up if a ranked setup is dropped), run one WebSearch "<TICKER> stock news" and judge the last few days. ' + VERDICTS + " Apply every verdict in one command (it re-ranks the list and refreshes the "
    'dashboard cards): `cd $C && python3 -m engine.postprocess --out $W/out --state $W/state --news "TICKER=clean" '
    '"TICKER=caution:one-line reason" "TICKER=avoid:one-line reason"`. If there are no entries, skip this step.')

NEWS_PREMARKET = (
    "News check on today's orders and open positions: take the tickers in scan.premarket_checks and in scan.positions in "
    '$W/out/scan.json (at most 8). Run one WebSearch "<TICKER> stock news" each and look at the news since the last close. '
    "avoid = material company news (earnings surprise, guidance cut, offering, legal/FDA shock, takeover announcement); caution = "
    "other company-specific bad news; clean = routine news or moves explained by the market or the sector (a weak market is not a "
    "reason to cancel: the plan already allows for it). Apply every verdict in one command: "
    '`cd $C && python3 -m engine.postprocess --out $W/out --state $W/state --news "TICKER=clean" "TICKER=avoid:one-line reason"`. '
    "It turns avoid on a pending order into a CANCEL line and withdraws that order on the dashboard, and adds a REVIEW line for an "
    "open position. If there is nothing to check, skip this step.")

NEWS_AUTO = (
    "News check: if $W/out/scan.json has entries, check them, highest score first (at most 10, including the \"also qualified\" "
    "ones); if it is a pre-market "
    "scan, check the tickers in scan.premarket_checks and scan.positions instead (at most 8). Run one WebSearch "
    '"<TICKER> stock news" each and judge the last few days. ' + VERDICTS + " Apply every verdict in one command (it re-ranks, "
    "withdraws what you rule out and refreshes the dashboard cards): "
    '`cd $C && python3 -m engine.postprocess --out $W/out --state $W/state --news "TICKER=clean" "TICKER=caution:one-line reason" '
    '"TICKER=avoid:one-line reason"`. If there is nothing to check, skip this step.')

PROGRESS = '''
Progress for the dashboard's Scan now bar: the extra message after these instructions may contain a line "Run token: rt_...". If it does, call that value T and record progress by creating small documents in collection "runs" with ArtifactData action "set" (new documents, so no if_version), each with data {"token": T, "seq": n, "state": "running", "step": "<label>", "pct": <p>, "at": "<current UTC time, e.g. 2026-11-05T23:14:05Z, from `date -u +%Y-%m-%dT%H:%M:%SZ` chained onto that step's shell command>", "by": "scan"}:
- doc_id "T-s1", label "Loading scanner code and market data", pct 12: as soon as ArtifactData is loaded (send it in the same message as the add_repo call; for this first one, take the time from a `date -u +%Y-%m-%dT%H:%M:%SZ` Bash call made together with your first ToolSearch).
- "T-s2", "Reading your positions, funds and alerts", 28: in the same message as the step 3 exports.
- "T-s3", "Scanning 605 stocks & ETFs for A+ setups", 42: in the same message as the step 4 command.
- "T-s4", "Checking news on the top setups", 62: in the same message as the first step 5 search (skip it if step 5 is skipped).
- "T-s5", "Saving results to your dashboard", 86: in the same message as the first step 6 batch.
The scanner adds the final "T-s6" (done) record to the results batch itself because step 4 passes it the token (without a run token, leave `--run-token T` out of that command). If the run ends without results for any reason, create "T-s9" with {"token": T, "seq": 9, "state": "failed", "step": "Scan failed", "message": "<one plain line on what went wrong>", "at": ..., "by": "scan"}. If a progress write fails, ignore it and carry on: the scan is the job, progress is a convenience. Without a run token, skip all of this.
The extra message can reach you a little after you start. If the token turns up mid-run, write the record for the step you are on and carry on from there (skip the earlier ones); if step 4 already ran without it, create "T-s6" yourself right after the results are saved, with {"token": T, "seq": 6, "state": "done", "step": "Scan complete", "pct": 100, "scan_id": "<the scan id>", "at": ..., "by": "scan"}. Run the scan once per session: a message that arrives while you work or after you finish is never a request for another scan. If you had already finished, answer it with your final message again, unchanged (the headline line, a blank line, the full digest), without re-running anything.
'''

MODES = {
    "premarket": {
        "MODE": "premarket", "MODE_LABEL": "pre-market check (before the 6:30 AM PT open)",
        "FRESHNESS_RULE": "premarket_asof should be from today after 4:30 AM PT. If it is missing or older, re-clone the data branch once after 5 minutes, then continue regardless.",
        "NEWS_STEP": NEWS_PREMARKET, "PROGRESS_BLOCK": "", "SCAN_ARGS": ""},
    "preclose": {
        "MODE": "preclose", "MODE_LABEL": "pre-close exit scan (35 minutes before the 1:00 PM PT close)",
        "FRESHNESS_RULE": "intraday_asof should be within the last 45 minutes. If it is older, re-clone the data branch once after 5 minutes, then continue with what you have and mention the data time in the digest.",
        "NEWS_STEP": "If $W/out/scan.json lists entries (buy-at-close setups), run the news check. " + NEWS_ENTRIES,
        "PROGRESS_BLOCK": "", "SCAN_ARGS": ""},
    # one-off runs on half days (close 10:00 AM PT), when the regular 12:25 PM run would come after the close
    "preclose_halfday": {
        "MODE": "preclose", "MODE_LABEL": "pre-close exit scan for today's half day (40 minutes before the 10:00 AM PT early close)",
        "FRESHNESS_RULE": "intraday_asof should be within the last 45 minutes. If it is older, re-clone the data branch once after 5 minutes, then continue with what you have and mention the data time in the digest.",
        "NEWS_STEP": "If $W/out/scan.json lists entries (buy-at-close setups), run the news check. " + NEWS_ENTRIES,
        "PROGRESS_BLOCK": "", "SCAN_ARGS": ""},
    "postclose": {
        "MODE": "postclose", "MODE_LABEL": "post-close setup scan for the next session",
        "FRESHNESS_RULE": ("If today is a trading day and meta.json's daily_final_for is not today's date (Pacific), re-clone the data branch "
                           "every 5 minutes for up to 25 minutes until it is (use the Monitor tool or a background Bash loop to wait; don't "
                           "chain sleeps). If it still isn't, run anyway and put \"Provisional: data feed delayed, bars may be incomplete\" "
                           "at the top of the digest."),
        "NEWS_STEP": NEWS_ENTRIES, "PROGRESS_BLOCK": "", "SCAN_ARGS": ""},
    "weekend": {
        "MODE": "auto", "MODE_LABEL": ("weekend refresh of the next session's setups (runs Sunday afternoon: after Saturday's recalibration "
                                  "and its Sunday-morning backup, so the setups are scored by this week's strategies, and late enough "
                                  "that the news check covers the whole weekend)"),
        "FRESHNESS_RULE": "Use the newest data; there is nothing to wait for on a weekend.",
        "NEWS_STEP": NEWS_ENTRIES, "PROGRESS_BLOCK": "", "SCAN_ARGS": ""},
    "manual": {
        "MODE": "auto", "MODE_LABEL": "on-demand scan (I pressed Scan now on the dashboard)",
        "FRESHNESS_RULE": "Use the newest data. During market hours, if intraday_asof is more than 45 minutes old, re-clone the data branch once after 5 minutes.",
        "NEWS_STEP": NEWS_AUTO, "PROGRESS_BLOCK": PROGRESS, "SCAN_ARGS": " --run-token T"},
}


def _holidays() -> str:
    """Upcoming NYSE holidays, from the engine's calendar, for the backup checks' closed-day rule."""
    import datetime as dt
    import sys
    sys.path.insert(0, str(HERE.parent))
    from engine import calendar as cal
    days = sorted(d for d in cal.HOLIDAYS if d >= dt.date.today().isoformat())
    return ", ".join(dt.date.fromisoformat(d).strftime("%a %b %-d %Y") for d in days)


WEEKDAY_CLOSED = ('If today\'s Pacific date is a Saturday or Sunday, or one of these US market holidays: {HOLIDAYS}, end with '
                  'exactly: "Market closed today; nothing to check."')

# backup checks: each runs after a scheduled task and re-runs it if it failed or never ran.
# TARGET_ID is the scheduled task it backs up (to tell "still running" from "never ran").
BACKUPS = {
    "backup_premarket": {
        "TARGET_ID": "trig_01BC6UyGsrA3mesQsU6pFbRP", "TARGET": "premarket", "TARGET_LABEL": "today's pre-market check", "DUE": "5:50 AM PT", "CHECK_COLLECTION": "scans",
        "CHECK_ORDER": "started_at", "CLOSED_RULE": WEEKDAY_CLOSED,
        "CHECK_RULE": ('It ran if one of those scans has mode "premarket", a started_at today at or after 5:40 AM, and status "ok", '
                       '"ok_with_warnings" or "closed".'),
        "OK_LINE": "today's pre-market check ran at <its start time, e.g. 5:52 AM PT>.",
        "URGENCY": "The market opens at 6:30 AM PT, so go straight to it. "},
    "backup_preclose": {
        "TARGET_ID": "trig_015vu682N2v147NyT2kAdSLJ", "TARGET": "preclose", "TARGET_LABEL": "today's pre-close exit scan", "DUE": "12:25 PM PT", "CHECK_COLLECTION": "scans",
        "CHECK_ORDER": "started_at", "CLOSED_RULE": WEEKDAY_CLOSED,
        "CHECK_RULE": ('It ran if one of those scans has mode "preclose" or "postclose", a started_at today at or after 12:20 PM '
                       '(on a half day, when the market closes at 10:00 AM PT, at or after 9:15 AM), and status "ok", '
                       '"ok_with_warnings" or "closed".'),
        "OK_LINE": "today's pre-close scan ran at <its start time, e.g. 12:27 PM PT>.",
        "URGENCY": "The market closes at 1:00 PM PT, so go straight to it. "},
    "backup_postclose": {
        "TARGET_ID": "trig_01VJiBMjH2dJJnBrAaTr9BcX", "TARGET": "postclose", "TARGET_LABEL": "today's post-close setup scan", "DUE": "2:05 PM PT", "CHECK_COLLECTION": "scans",
        "CHECK_ORDER": "started_at", "CLOSED_RULE": WEEKDAY_CLOSED,
        "CHECK_RULE": ('It ran if one of those scans has mode "postclose", a started_at today at or after 1:55 PM (on a half day '
                       'at or after 10:55 AM), and status "ok", "ok_with_warnings" or "closed".'),
        "OK_LINE": "today's post-close scan ran at <its start time, e.g. 2:17 PM PT>.", "URGENCY": ""},
    "backup_learning": {
        "TARGET_ID": "trig_01V9Hsf7VTMtsod6AoHQnyiP", "TARGET": "learning", "TARGET_LABEL": "today's daily learning review", "DUE": "3:12 PM PT", "CHECK_COLLECTION": "journal",
        "CHECK_ORDER": "date", "CLOSED_RULE": WEEKDAY_CLOSED,
        "CHECK_RULE": 'It ran if one of those documents has date equal to today (YYYY-MM-DD) and an id that doesn\'t end in "-weekly".',
        "OK_LINE": "today's daily review ran (journal <its id>).", "URGENCY": ""},
    "backup_weekly": {
        "TARGET_ID": "trig_01QZiuM65EDHm187zYcSY7Kk", "TARGET": "weekly", "TARGET_LABEL": "this week's Saturday recalibration", "DUE": "Saturday 10:14 AM PT",
        "CHECK_COLLECTION": "journal", "CHECK_ORDER": "date",
        "CLOSED_RULE": "This check runs on Sunday; there is no closed-day rule.",
        "CHECK_RULE": 'It ran if one of those documents has an id ending in "-weekly" and a date of yesterday or today.',
        "OK_LINE": "this week's recalibration ran (journal <its id>).", "URGENCY": ""},
    "backup_weekend": {
        "TARGET_ID": "trig_01XGcnEVgDkEYRGBoFe9xijx", "TARGET": "weekend", "TARGET_LABEL": "this Sunday's weekend refresh",
        "DUE": "Sunday 4:17 PM PT", "CHECK_COLLECTION": "scans", "CHECK_ORDER": "started_at",
        "CLOSED_RULE": ("This check runs on Sunday; there is no closed-day rule (the refresh scores the next session's setups "
                        "even when Monday is a market holiday)."),
        "CHECK_RULE": ('It ran if one of those scans has mode "postclose", a started_at today at or after 4:05 PM, and status '
                       '"ok" or "ok_with_warnings" (a Scan now pressed after 4:05 PM today counts too: it is the same refresh).'),
        "OK_LINE": "this Sunday's weekend refresh ran at <its start time, e.g. 4:20 PM PT>.", "URGENCY": ""},
}


def fill(text: str, values: dict) -> str:
    for k, v in values.items():
        text = text.replace("{" + k + "}", v)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="aznh0pe")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", default=str(HERE / "generated"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = {"SETUP": SETUP, "OWNER": a.owner, "ARTIFACT": a.artifact}
    scan_t = (HERE / "scan_prompt.md").read_text()
    for name, vals in MODES.items():
        txt = fill(scan_t, {**vals, **common})
        txt = txt.replace("{MODE}", vals["MODE"])
        (out / f"{name}.md").write_text(txt)
    for name in ("learning", "weekly"):
        (out / f"{name}.md").write_text(fill((HERE / f"{name}_prompt.md").read_text(), common))
    backup_t = (HERE / "backup_prompt.md").read_text()
    for name, vals in BACKUPS.items():
        target = (out / f"{vals['TARGET']}.md").read_text().strip()
        txt = fill(backup_t, {k: v for k, v in vals.items() if k != "TARGET"})
        txt = fill(txt, {"HOLIDAYS": _holidays(), "ARTIFACT": a.artifact})
        (out / f"{name}.md").write_text(txt.replace("{TARGET_PROMPT}", target))
    for f in sorted(out.glob("*.md")):
        left = [s for s in ("{MODE", "{OWNER}", "{ARTIFACT}", "{NEWS_STEP}", "{PROGRESS_BLOCK}", "{SCAN_ARGS}", "{SETUP}",
                            "{TARGET", "{DUE}", "{CHECK_", "{OK_LINE}", "{URGENCY}", "{CLOSED_RULE}", "{HOLIDAYS}", "{TARGET_ID}") if s in f.read_text()]
        print(f.name, len(f.read_text()), "unfilled:" if left else "ok", *left)


if __name__ == "__main__":
    main()
