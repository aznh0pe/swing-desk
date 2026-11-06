"""Fill the scheduled-task prompt templates.

    python3 ops/build_prompts.py --owner aznh0pe --artifact https://claude.ai/artifact/... --out ops/generated

Writes one prompt per scheduled task: premarket, preclose, postclose, manual (Scan now button),
learning (daily review + strategy research) and weekly (recalibration).
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

NEWS_ENTRIES = (
    'News check: for each featured entry in $W/out/scan.json (the ones with a rank, at most 8) run one WebSearch "<TICKER> stock news" '
    "and judge the last few days. avoid = earnings or guidance shock, offering/dilution, fraud/SEC/legal trouble, an agreed takeover "
    "pinning the price, an FDA or other binary event, delisting risk; caution = notable but not thesis-breaking (downgrade, sector "
    "selloff, CEO exit); clean = routine. Apply every verdict in one command (it re-ranks the list and refreshes the dashboard cards): "
    '`cd $C && python3 -m engine.postprocess --out $W/out --state $W/state --news "TICKER=clean" "TICKER=caution:one-line reason" '
    '"TICKER=avoid:one-line reason"`. If there are no entries, skip this step.')

NEWS_PREMARKET = (
    "News check on today's orders and open positions: take the tickers in scan.premarket_checks and in scan.positions in "
    '$W/out/scan.json (at most 8). Run one WebSearch "<TICKER> stock news" each and look at the last day. avoid = material news '
    "(earnings surprise, guidance cut, offering, legal/FDA shock, takeover announcement); caution = notable but not thesis-breaking; "
    "clean = routine. Apply every verdict in one command: "
    '`cd $C && python3 -m engine.postprocess --out $W/out --state $W/state --news "TICKER=clean" "TICKER=avoid:one-line reason"`. '
    "It turns avoid on a pending order into a CANCEL line and withdraws that order on the dashboard, and adds a REVIEW line for an "
    "open position. If there is nothing to check, skip this step.")

NEWS_AUTO = (
    "News check: if $W/out/scan.json has featured entries (the ones with a rank, at most 8), check those; if it is a pre-market "
    "scan, check the tickers in scan.premarket_checks and scan.positions instead (at most 8). Run one WebSearch "
    '"<TICKER> stock news" each and judge the last few days. avoid = earnings or guidance shock, offering/dilution, fraud/SEC/legal '
    "trouble, an agreed takeover pinning the price, an FDA or other binary event, delisting risk; caution = notable but not "
    "thesis-breaking (downgrade, sector selloff, CEO exit); clean = routine. Apply every verdict in one command (it re-ranks, "
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
    "manual": {
        "MODE": "auto", "MODE_LABEL": "on-demand scan (I pressed Scan now on the dashboard)",
        "FRESHNESS_RULE": "Use the newest data. During market hours, if intraday_asof is more than 45 minutes old, re-clone the data branch once after 5 minutes.",
        "NEWS_STEP": NEWS_AUTO, "PROGRESS_BLOCK": PROGRESS, "SCAN_ARGS": " --run-token T"},
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
    for f in sorted(out.glob("*.md")):
        left = [s for s in ("{MODE", "{OWNER}", "{ARTIFACT}", "{NEWS_STEP}", "{PROGRESS_BLOCK}", "{SCAN_ARGS}", "{SETUP}") if s in f.read_text()]
        print(f.name, len(f.read_text()), "unfilled:" if left else "ok", *left)


if __name__ == "__main__":
    main()
