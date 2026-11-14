#!/usr/bin/env bash
# Fetch market data and publish the `data` branch: one pass, or (an `auto` run on a trading day between 6:30 AM
# and 40 minutes after the close, ET) a pass every 10 minutes until then. See pipeline/keeper_window.py for why.
# Env: MODE (auto | daily | ...), PUSH_URL (authenticated repo URL), GH_TOKEN (for the hand-over), REPO,
# DATA_BRANCH (default data), UNIVERSE (core | wide; default set in pipeline/universe.py).
set -u
MODE="${MODE:-auto}"
BRANCH="${DATA_BRANCH:-data}"   # the live feed; a trial universe is built on another branch first
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
  # One commit with no parent, as before, but built in the checkout's own repository on top of what it already
  # knows the branch holds, so a push sends only the files that changed (the price snapshot, not 100+ MB of
  # history every 10 minutes).
  (
    set -e
    root="$PWD"
    export GIT_INDEX_FILE="$root/.git/data-index"
    rm -f "$GIT_INDEX_FILE"
    git --git-dir="$root/.git" --work-tree="$root/out" -C "$root/out" add -A .
    tree=$(git --git-dir="$root/.git" write-tree)
    commit=$(git --git-dir="$root/.git" -c user.name="swing-desk-bot" \
      -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
      commit-tree "$tree" -m "market data $(date -u +%Y-%m-%dT%H:%MZ)")
    for attempt in 1 2 3; do
      if git --git-dir="$root/.git" push -q -f "$PUSH_URL" "$commit:refs/heads/$BRANCH"; then
        git --git-dir="$root/.git" update-ref "refs/remotes/origin/$BRANCH" "$commit"
        exit 0
      fi
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
    gh workflow run market-data.yml --repo "$REPO" --ref main -f mode=auto -f branch="$BRANCH" -f universe="${UNIVERSE:-}" && echo "handed over to a new run" \
      || echo "::warning::hand-over failed; the next scheduled run carries on"
    exit 0
  fi
  now=$(date +%s); sleep $(( 600 - now % 600 + 60 ))   # one minute past every 10-minute mark
  one
done
exit $(( FAILS > 2 ))
