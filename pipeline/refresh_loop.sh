#!/usr/bin/env bash
# Fetch market data and publish the `data` branch: one pass, or (an `auto` run on a trading day between 6:30 AM
# and 40 minutes after the close, ET) a pass every 10 minutes until then. See pipeline/keeper_window.py for why.
# Env: MODE (auto | daily | ...), PUSH_URL (authenticated repo URL), GH_TOKEN (for the hand-over), REPO.
set -u
MODE="${MODE:-auto}"
START=$(date +%s)
LIMIT=$((330 * 60))   # hand over to a new run well before GitHub's 6-hour job limit
FAILS=0

pass() {
  for attempt in 1 2 3; do
    python pipeline/fetch_data.py --mode "$MODE" --out out && return 0
    echo "fetch attempt $attempt failed, retrying in 30s"; sleep 30
  done
  return 1
}

publish() {
  (
    set -e
    cd out
    rm -rf .git
    git init -q -b data
    git config user.name "swing-desk-bot"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    git add -A
    git commit -qm "market data $(date -u +%Y-%m-%dT%H:%MZ)"
    for attempt in 1 2 3; do
      git push -q -f "$PUSH_URL" data:data && exit 0
      sleep 10
    done
    exit 1
  )
}

one() {
  if pass && publish; then FAILS=0; echo "published $(date -u +%H:%MZ)"; else FAILS=$((FAILS + 1)); echo "::warning::pass failed ($FAILS in a row)"; fi
}

one
[ "$MODE" = "auto" ] || exit $(( FAILS > 0 ))
while python pipeline/keeper_window.py; do
  if [ $(( $(date +%s) - START )) -ge "$LIMIT" ]; then
    # GitHub's scheduled runs are too sparse to rely on: start the next run ourselves (a workflow_dispatch made
    # with the run's own token does start a new run). It waits in the concurrency group until this one ends.
    gh workflow run market-data.yml --repo "$REPO" --ref main -f mode=auto && echo "handed over to a new run" \
      || echo "::warning::hand-over failed; the next scheduled run carries on"
    exit 0
  fi
  now=$(date +%s); sleep $(( 600 - now % 600 + 60 ))   # one minute past every 10-minute mark
  one
done
exit $(( FAILS > 2 ))
