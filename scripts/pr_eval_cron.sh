#!/usr/bin/env bash
# The eval bot's cron wrapper. System cron runs this every 30 minutes; the
# bot only spins the GPU when there is an un-evaluated eligible PR head, so a
# quiet repo costs nothing but the poll.
#
# flock: evals serialize on one box; a poll that finds the previous one still
# running must exit, not queue. Cron's PATH is minimal, so gh/vastai
# (~/.local/bin) are added explicitly. The bot always runs from the local
# main tree — PR code never executes on this machine.
#
# Install (idempotent) — run from the repo root:
#   ( crontab -l 2>/dev/null | grep -v 'quench/scripts/pr_eval_cron.sh' ; \
#     echo "*/30 * * * * $PWD/scripts/pr_eval_cron.sh" ) | crontab -
#
# The path above was, until 2026-08-11, a copy-paste leftover from the braid
# port: it installed `~/Dev/plind-junior/imp/scripts/pr_eval_cron.sh` and
# filtered on the bare basename, so running it removed and re-added the OTHER
# repo's entry and never installed quench's at all. That is why no quench PR
# was ever evaluated on a schedule. Keep the filter specific to this repo's
# path so a second project's cron line survives a reinstall here.
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
# Polaris API key for attested receipts (optional — evals run without it).
# The key lives outside the repo on purpose; never move it into the tree.
if [ -f "$HOME/.config/quench/polaris.env" ]; then
  . "$HOME/.config/quench/polaris.env"
  export QUENCH_POLARIS_API_KEY
fi
cd "$(dirname "$0")/.."
LOG="${QUENCH_EVAL_LOG:-$HOME/quench-pr-eval.log}"

exec 9>"$HOME/.quench-pr-eval.lock"
flock -n 9 || exit 0

# The verdict policy is whatever is checked out HERE: the bot imports
# scripts/eval_scorer.py from this tree, and ships that same file into the TEE
# as the attested policy. A dev machine sitting on a feature branch would
# therefore grade other people's PRs — and publish attested receipts — with
# uncommitted, unreviewed rules. The bot's docstring has always claimed it
# "runs from the local main tree"; these two checks are what make that true.
# Skip rather than fail: a poll that lands mid-edit is normal, not an incident.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
if [ "$branch" != "main" ]; then
  echo "=== $(date -Is) skip: on '$branch', not main" >>"$LOG"
  exit 0
fi
if ! git diff --quiet HEAD -- scripts/ bench/ tests/; then
  echo "=== $(date -Is) skip: uncommitted changes under scripts/, bench/ or tests/" >>"$LOG"
  exit 0
fi
{
  echo "=== $(date -Is) pr-eval poll"
  # -u: unbuffered, so a 30-minute eval's progress is readable in the log
  # while it runs, not only after it exits.
  python3 -B -u scripts/pr_eval_bot.py
  echo "=== exit $?"
} >>"$LOG" 2>&1
