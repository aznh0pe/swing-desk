Run the Swing Desk weekly recalibration for me now. Work unattended: don't ask me anything.

Context: my swing-trading scanner lives in my public GitHub repo {OWNER}/swing-desk (branch main; market data incl. history.parquet on branch `data`, rebuilt early Saturday). Its dashboard database is the artifact {ARTIFACT}; use the ArtifactData tool (load it with ToolSearch if needed). Priority: highest win rate that still clears profit-factor and expectancy floors.

Steps
1. Bash: W=/tmp/sd; rm -rf $W; mkdir -p $W/state; clone the repo to $W/code and the `data` branch to $W/data (retry up to 3 times). Check $W/data/meta.json history_asof is from the last 2 days; if not, continue with the history that exists and say so. pip install --break-system-packages -q pandas numpy scikit-learn pyarrow if needed.
2. Export state: ArtifactData "list" on {ARTIFACT} with out_dir "$W/state" for collections strategy and config.
3. Run `cd $W/code && python3 -m engine.calibrate backtest --data $W/data --out $W/bt --merge-state $W/state` (takes several minutes; run it with a long timeout or in the background and wait). Fix and re-run on errors (up to 3 attempts).
4. Sanity check before publishing: in $W/bt/backtest_report.json every enabled strategy must have train win_rate >= 0.58 and an out-of-sample (test_oos) win rate no more than 8 points below its train win rate. Disable any that fail by setting "enabled": false for it in both $W/bt/params.json and $W/bt/doc_params.json (inside "params"), and add a line saying why to doc_params.json's changelog_entry.
5. Write: ArtifactData "get" strategy/params and strategy/backtest to learn their versions (they may not exist yet). Then one ArtifactData "batch" with two "set" entries: collection strategy doc_id params file_path $W/bt/doc_params.json, and collection strategy doc_id backtest file_path $W/bt/doc_backtest.json, each with the if_version you just read (omit it for a document that didn't exist).
6. Journal: ArtifactData "set" collection journal doc_id "<today>-weekly" with fields id, date (today), created_at, insights (3-5 bullet strings on what the new backtest says), changes (from changelog_entry), shadow {}, live {}.
7. If the Projects tool is available, update the project doc at path "claude/Swing Desk - Strategy Playbook.md" with each strategy's current parameters and train/out-of-sample stats (project_read, edit, project_write).
Final message: what changed this week in at most four lines.
