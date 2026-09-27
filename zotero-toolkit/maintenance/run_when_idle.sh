#!/usr/bin/env bash
# run_when_idle.sh — run clean_dangling_refs.py, retrying while the LightRAG pipeline is busy.
#
# --sweep and --commit refuse to write while /documents/pipeline_status reports busy and
# exit 3, because extraction merges into exactly the rows they change. Idle windows can be
# short and unpredictable, so this retries until the tool gets through, then exits with
# its status. Arguments go to clean_dangling_refs.py; without a mode flag (--scan,
# --commit, --show, --sweep) it adds --sweep.
#
#   IDLE_RETRY_INTERVAL  seconds between attempts (default 300; use ~15 for --commit,
#                        whose plan goes stale as the pipeline keeps writing)
#   IDLE_MAX_WAIT        stop retrying after this many seconds (default 21600) and exit 3
#   PYTHON               interpreter with asyncpg installed (default python3). cron and systemd
#                        activate no virtual environment, so point it at the venv's bin/python
#                        there. Without asyncpg the wrapper exits 1 before its first attempt.
#
# Settings reach the tool through the environment or --env-file, as when run directly.
# One instance at a time when flock(1) is available. Timestamped lines on stdout.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if command -v flock >/dev/null 2>&1; then
  exec 9<"$here/$(basename "$0")"
  flock -n 9 || { echo "run_when_idle.sh is already running — exiting"; exit 0; }
fi
mode=
for a in "$@"; do
  case $a in --scan|--commit|--show|--sweep) mode=$a ;; esac
done
[ -n "$mode" ] || set -- --sweep "$@"
interval=${IDLE_RETRY_INTERVAL:-300}
max_wait=${IDLE_MAX_WAIT:-21600}
log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
py=${PYTHON:-python3}
if ! err=$("$py" -c 'import asyncpg' 2>&1); then
  log "cannot run clean_dangling_refs.py with $py: ${err##*$'\n'}"
  log "set PYTHON to the bin/python of the virtual environment that has maintenance/requirements.txt installed"
  exit 1
fi
start=$(date +%s)
attempts=0
while :; do
  attempts=$((attempts + 1))
  out=$("$py" "$here/clean_dangling_refs.py" "$@" 2>&1); rc=$?
  if [ "$rc" -ne 3 ]; then
    printf '%s\n' "$out"
    log "clean_dangling_refs.py $* -> rc=$rc after $attempts attempt(s)"
    exit "$rc"
  fi
  if [ $(( $(date +%s) - start )) -ge "$max_wait" ]; then
    log "pipeline still busy after $attempts attempt(s) over ${max_wait}s — giving up (exit 3)"
    exit 3
  fi
  sleep "$interval"
done
