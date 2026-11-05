Run the Swing Desk daily learning review for me now. Work unattended: don't ask me anything.

Context: my swing-trading scanner lives in my public GitHub repo {OWNER}/swing-desk (branch main, market data on branch `data`). Its dashboard database is the artifact {ARTIFACT}; use the ArtifactData tool (load it with ToolSearch if needed). The goal of this review is to keep improving the strategies with a high win rate first and strong profits second.

Steps
1. Bash: W=/tmp/sd; rm -rf $W; mkdir -p $W/state; clone the repo to $W/code and the `data` branch to $W/data (`git clone --depth 1 [-b data] https://github.com/{OWNER}/swing-desk ...`, retry up to 3 times). pip install --break-system-packages -q pandas numpy scikit-learn pyarrow if imports fail.
2. Export state: ArtifactData "list" on {ARTIFACT} with out_dir "$W/state" for collections alerts (query.limit 1000), trades, strategy, config.
3. Run `cd $W/code && python3 -m engine.calibrate learn --data $W/data --state $W/state --out $W/out`. If it errors, fix the cause in the local copy and re-run (up to 3 attempts).
4. Read $W/out/journal.json. Add your own analysis in a "notes" field (2-4 sentences): what worked today, what didn't, and why, from the trades, shadow results and market regime. Save the file and the matching doc file under $W/out/docs/journal/.
5. Write: for each chunk in $W/out/writes.json call ArtifactData "batch". If a write targets strategy/params, first ArtifactData "get" collection strategy doc_id params; put its version in that entry's if_version (or drop if_version if the document doesn't exist yet).
6. Housekeeping: ArtifactData "list" scans (query.limit 1000, no out_dir needed if small). If there are more than 250 scan documents, delete the oldest ones beyond 250 with batch deletes.
7. If the Projects tool is available, append today's journal (date, shadow and live win rates, insights, changes, your notes) to the project doc at path "claude/Swing Desk - Learning Journal.md" (project_read it, add the entry at the top, project_write it back). This keeps the learning history outside the dashboard as well.
Final message: three lines at most: shadow/live win rate, the most useful insight, and any parameter change.
