Run the Swing Desk backup check for {TARGET_LABEL} for me now. Work unattended: don't ask me anything, decide and keep going.

Why this exists: {TARGET_LABEL} is scheduled for {DUE}. Those runs are crucial, so this check makes sure it really ran and, if it failed or never ran, runs it now so it never stays failed. The dashboard's database is the artifact {ARTIFACT}; read and write it with the ArtifactData tool (load it with ToolSearch if it isn't loaded yet).

Check
A. {CLOSED_RULE}
B. ArtifactData action "query" on url {ARTIFACT}, collection "{CHECK_COLLECTION}", query {"order_by": {"field": "{CHECK_ORDER}", "direction": "desc"}, "limit": 12} (no out_dir). {CHECK_RULE} Use Pacific time for "today" and for the times in those documents.
C. If it ran: end with exactly one line, "No action needed: {OK_LINE}", and nothing else.
D. If it didn't run, or only failed (status "failed"): run it now by carrying out every step of the instructions below, which are {TARGET_LABEL}'s own. {URGENCY}Start the final message with "Backup run: {TARGET_LABEL} ({DUE}) had failed or not run, so I ran it now." and then give everything those instructions ask the final message to contain.

==== Instructions for {TARGET_LABEL} ====
{TARGET_PROMPT}
