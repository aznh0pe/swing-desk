Run the Swing Desk backup check for {TARGET_LABEL} for me now. Work unattended: don't ask me anything, decide and keep going.

Why this exists: {TARGET_LABEL} is scheduled for {DUE}. Those runs are crucial, so this check makes sure it really ran and, if it failed or never ran, runs it now so it never stays failed. The dashboard's database is the artifact {ARTIFACT}; read and write it with the ArtifactData tool (load it with ToolSearch if it isn't loaded yet).

Check
A. {CLOSED_RULE}
B. In Bash `rm -rf /tmp/bk`, then ArtifactData action "query" on url {ARTIFACT}, collection "{CHECK_COLLECTION}", query {"order_by": {"field": "{CHECK_ORDER}", "direction": "desc"}, "limit": 12}, out_dir "/tmp/bk" (the documents are large, so keep them on disk), then list them: `python3 -c "import json,glob; [print(d.get('id'), d.get('mode', '-'), d.get('started_at') or d.get('date'), d.get('status', '-')) for d in (json.load(open(f)) for f in glob.glob('/tmp/bk/**/*.json', recursive=True))]"`. {CHECK_RULE} Use Pacific time for "today" and for the times in those documents.
C. If it ran: end with exactly one line, "No action needed: {OK_LINE}", and nothing else.
D. If it hasn't, first make sure it isn't still running: load list_triggers with ToolSearch ("select:mcp__claude-code-remote__list_triggers"), call it with limit 100 (follow next_cursor) and find the scheduled task with id {TARGET_ID}. If its last_run fired today and hasn't finished (no finished_at, or a status saying it is running), it is still running: wait 3 minutes at a time (the Monitor tool, or a background Bash `sleep 180`; foreground sleeps are blocked) and repeat B after each wait, until it shows up or 20 minutes have passed. If list_triggers isn't available, wait 5 minutes and repeat B once. If it now shows up, go to C.
E. Otherwise (it didn't run, or only failed with status "failed"): run it now by carrying out every step of the instructions below, which are {TARGET_LABEL}'s own. {URGENCY}Start the final message with "Backup run: {TARGET_LABEL} ({DUE}) had failed or not run, so I ran it now." on its own line, then give everything those instructions ask the final message to contain.

==== Instructions for {TARGET_LABEL} ====
{TARGET_PROMPT}
