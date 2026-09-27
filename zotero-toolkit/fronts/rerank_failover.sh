#!/usr/bin/env bash
# Supervisor for rerank_failover.py (see its docstring and README.md): restarts it 5 s after
# any exit and appends its output to a log. Settings come from the environment; an env file
# (RERANK_FO_ENV_FILE, default rerank_failover.env beside this script) is sourced first if
# present. Use absolute paths inside it: the front runs from this script's directory.
here="$(cd "$(dirname "$0")" && pwd)" || exit 1
env_file="${RERANK_FO_ENV_FILE:-$here/rerank_failover.env}"
if [ -f "$env_file" ]; then set -a; . "$env_file"; set +a; fi
cd "$here" || exit 1
export RERANK_FO_LOG="${RERANK_FO_LOG:-$here/rerank_failover.log}"
while :; do
  "${PYTHON:-python3}" ./rerank_failover.py >> "$RERANK_FO_LOG" 2>&1
  rc=$?
  echo "$(date -Is) rerank_failover exited rc=$rc - restarting in 5s" >> "$RERANK_FO_LOG"
  sleep 5
done
