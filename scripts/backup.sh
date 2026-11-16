#!/usr/bin/env bash
# Backup the whole project (code + checkpoints + logs) to GitHub.
#
# One-time setup (on a machine where you have a token, NOT committed anywhere):
#   git remote add origin https://github.com/Underruness/dev.git
#   git config credential.helper store     # optional
#   git push -u origin main
#
# Then:  ./scripts/backup.sh "my message"
# Cron every 30 min:
#   */30 * * * * cd /path/to/project && ./scripts/backup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${1:-auto backup $(date -Is)}"
SIZE_WARN_MB="${SIZE_WARN_MB:-20}"

git add -A

# never let a multi-hundred-MB blob land in the history
big=$(git diff --cached --name-only | while read -r f; do
        [ -f "$f" ] || continue
        s=$(( $(stat -c%s "$f") / 1000000 ))
        [ "$s" -ge "$SIZE_WARN_MB" ] && echo "$f ($s MB)"
      done || true)
if [ -n "$big" ]; then
  echo "[backup] refusing to commit files >= ${SIZE_WARN_MB} MB:" >&2
  echo "$big" >&2
  exit 1
fi

if git diff --cached --quiet; then
  echo "[backup] nothing to commit"
else
  git commit -m "$MSG" >/dev/null
  echo "[backup] committed: $MSG"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push origin HEAD 2>&1 | tail -3
else
  echo "[backup] no 'origin' remote configured (see header of this script)"
fi
