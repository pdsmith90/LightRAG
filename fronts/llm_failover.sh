#!/usr/bin/env bash
# Supervisor for llm_failover.py (see its docstring and README.md): restarts it 5 s after any
# exit and appends its output to a log. Settings come from the environment; an env file
# (LLM_FO_ENV_FILE, default llm_failover.env beside this script) is sourced first if present.
# Use absolute paths inside it: the front runs from this script's directory.
here="$(cd "$(dirname "$0")" && pwd)" || exit 1
env_file="${LLM_FO_ENV_FILE:-$here/llm_failover.env}"
if [ -f "$env_file" ]; then set -a; . "$env_file"; set +a; fi
cd "$here" || exit 1
export LLM_FO_LOG="${LLM_FO_LOG:-$here/llm_failover.log}"
while :; do
  "${PYTHON:-python3}" ./llm_failover.py >> "$LLM_FO_LOG" 2>&1
  rc=$?
  echo "$(date -Is) llm_failover exited rc=$rc - restarting in 5s" >> "$LLM_FO_LOG"
  sleep 5
done
