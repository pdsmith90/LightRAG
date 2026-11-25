#!/usr/bin/env bash
# Daily citation-graph refresh. Run it from a timer (systemd/citegraph-refresh.timer.example):
#
#   refresh.sh --config FILE [--dry-run]        (or CITEGRAPH_CONFIG=FILE refresh.sh)
#
# FILE is a bash file with the site's settings; see citegraph.env.example. Steps:
#   1. copy zotero_metadata.json and zotero_references.json from their sources into $CITEGRAPH_DATA/input/
#   2. builder ONLINE: asks only for what the caches lack (new DOIs, outside works, unknown-id lookups, S2 records)
#   3. citation sweep of the papers that are due (new, failed, newly cited by the library, older than 180 days),
#      at most CITEGRAPH_SWEEP_MAX_HOURS hours
#   4. builder OFFLINE with the citation store: the build to load
#   5. gates: the hard checks and the reference check pass; no table shrinks by more than 2% against the live
#      schema on the database host; a build identical to the last loaded one (out.loaded/) is not loaded again
#   6. copy it to the database host as out.daily/ and load it there with load_zstruct.sh (no server restart). The
#      swap keeps the previous tables as <schema>_prev until the next load; the undo line is in
#      out.daily/swap_<schema>.sql.
# The run is skipped while any citation sweep runs (for instance a first full sweep). A failure calls
# citegraph_notify (when the config defines it) and leaves the live tables as they were.
#   --dry-run: no sweep-running skip, no sweep, no copy, no load; builds into out.daily.dryrun/ and runs the gates.
set -uo pipefail

usage() { echo "usage: $0 --config FILE [--dry-run]   (FILE may also come from CITEGRAPH_CONFIG)" >&2; }
DRY=0
CONF=${CITEGRAPH_CONFIG:-}
while [ $# -gt 0 ]; do
  case $1 in
    --config) [ $# -ge 2 ] || { usage; exit 2; }; CONF=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[ -n "$CONF" ] || { echo "refresh.sh: no config (--config FILE or CITEGRAPH_CONFIG)" >&2; usage; exit 2; }
[ -r "$CONF" ] || { echo "refresh.sh: cannot read the config $CONF" >&2; exit 2; }
CONF=$(realpath "$CONF")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# defaults the config may override
citegraph_with_key() { "$@"; }
citegraph_notify() { :; }
# shellcheck source=citegraph.env.example
. "$CONF"

for v in CITEGRAPH_DATA CITEGRAPH_METADATA_SRC CITEGRAPH_REFERENCES_SRC; do
  [ -n "${!v:-}" ] || { echo "refresh.sh: $v is not set in $CONF" >&2; exit 2; }
done
CODE=${CITEGRAPH_CODE:-$HERE}
SCHEMA=${CITEGRAPH_SCHEMA:-zstruct}
REMOTE=${CITEGRAPH_REMOTE:-}
if [ -n "$REMOTE" ]; then
  for v in CITEGRAPH_REMOTE_CODE CITEGRAPH_REMOTE_CONFIG CITEGRAPH_REMOTE_DATA; do
    [ -n "${!v:-}" ] || { echo "refresh.sh: $v is not set in $CONF (CITEGRAPH_REMOTE is)" >&2; exit 2; }
  done
fi
DB_CODE=${CITEGRAPH_REMOTE_CODE:-$HERE}      # where load_zstruct.sh and refresh_gates.py are, on the database host
DB_CONF=${CITEGRAPH_REMOTE_CONFIG:-$CONF}
DB_DATA=${CITEGRAPH_REMOTE_DATA:-$CITEGRAPH_DATA}
SWEEP_HOURS=${CITEGRAPH_SWEEP_MAX_HOURS:-3}
# pgrep -f pattern of a running citation sweep; the bracket keeps pgrep from matching its own command line
SWEEP_RUNNING='[s]2_citation_sweep.py'
[ -z "${CITEGRAPH_SWEEP_PROCESS_PATTERN:-}" ] || SWEEP_RUNNING=$CITEGRAPH_SWEEP_PROCESS_PATTERN
GATES=("$HERE/refresh_gates.py")

mkdir -p "$CITEGRAPH_DATA/logs" "$CITEGRAPH_DATA/input" "$CITEGRAPH_DATA/cache" || exit 1
cd "$CITEGRAPH_DATA" || exit 1
L=logs/refresh_$(date +%Y%m).log
say() { echo "$(date '+%F %T')$([ $DRY = 1 ] && echo ' [dry-run]') $*" >> "$L"; }
notify() { citegraph_notify "${1:0:250} -- log $CITEGRAPH_DATA/$L" || say "citegraph_notify failed"; }
fail() { say "FAILED: $*"; [ $DRY = 1 ] || notify "$*"; exit 1; }
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
# run a command line on the database host: over ssh when CITEGRAPH_REMOTE is set, else here
db_host() { if [ -n "$REMOTE" ]; then "${SSH[@]}" "$REMOTE" "$1"; else bash -c "$1"; fi; }
# psql on the database host with the config's connection settings; the SQL comes on stdin. The password file is
# read on that host (a leading ~/ is that host's home), so the password never travels.
pg_live() {
  local cmd="" v pw=${CITEGRAPH_PGPASSWORD_FILE:-}
  for v in PGHOST PGPORT PGUSER PGDATABASE; do
    [ -z "${!v:-}" ] || cmd+="$v=$(printf %q "${!v}") "
  done
  case $pw in
    '') ;;
    \~/*) cmd+="PGPASSWORD=\$(cat \"\$HOME\"/$(printf %q "${pw#\~/}")) " ;;
    *) cmd+="PGPASSWORD=\$(cat $(printf %q "$pw")) " ;;
  esac
  db_host "${cmd}psql -X -At -F ' ' -v ON_ERROR_STOP=1 -f -"
}

exec 9> logs/.refresh.lock
flock -n 9 || { say "skip: another refresh is running"; exit 0; }
if [ $DRY = 0 ] && pgrep -f "$SWEEP_RUNNING" > /dev/null; then
  say "skip: a citation sweep is running"; exit 0
fi
say "start"

# 1. inputs
rsync -a -e "${SSH[*]}" "$CITEGRAPH_METADATA_SRC" input/ || fail "copying zotero_metadata.json from its source failed"
rsync -a -e "${SSH[*]}" "$CITEGRAPH_REFERENCES_SRC" input/ || fail "copying zotero_references.json from its source failed"

# 2. builder online (its exit code also counts single failed requests: the gates below decide)
N=out.daily.new; [ $DRY = 1 ] && N=out.daily.dryrun
rm -rf "$N"
citegraph_with_key nice -n 19 python3 "$CODE/build_struct_layer.py" \
  --metadata input/zotero_metadata.json --out "$N" --cache cache/openalex_works.json --schema "$SCHEMA" \
  --reference-check input/zotero_references.json > logs/refresh_build_online.log 2>&1
say "online build rc=$?: $(tail -n 1 logs/refresh_build_online.log | cut -c1-220)"
[ -s "$N/papers.csv" ] || fail "the online build wrote no papers.csv (logs/refresh_build_online.log)"

# 3. sweep (0 = done or out of time, 3 = throttled: both leave a consistent store)
if [ $DRY = 0 ]; then
  nice -n 19 python3 "$CODE/s2_citation_sweep.py" --out "$N" --store s2store/api_citations.sqlite \
    --max-hours "$SWEEP_HOURS" > logs/refresh_sweep.log 2>&1
  rc=$?
  say "sweep rc=$rc: $(grep -a '^sweep:' logs/refresh_sweep.log | tail -n 1 | cut -c1-220)"
  case $rc in
    0 | 3) ;;
    *) notify "the citation sweep exited rc=$rc (logs/refresh_sweep.log); the build went on with the store as it was" ;;
  esac
fi

# 4. builder offline with the store
nice -n 19 python3 "$CODE/build_struct_layer.py" --metadata input/zotero_metadata.json --out "$N" \
  --cache cache/openalex_works.json --schema "$SCHEMA" --reference-check input/zotero_references.json \
  --s2-store s2store/api_citations.sqlite --offline > logs/refresh_build.log 2>&1
say "build rc=$?: $(tail -n 1 logs/refresh_build.log | cut -c1-220)"

# 5. gates
"${GATES[@]}" qa-gate "$N/qa.json" \
  || fail "the hard checks or the reference check did not pass ($N/QA.md, logs/refresh_build.log) -- not loaded"
live=$("${GATES[@]}" count-sql "$N/qa.json" --schema "$SCHEMA" | pg_live) \
  || fail "could not count the live $SCHEMA tables on the database host -- not loaded"
shrunk=$(printf '%s\n' "$live" | "${GATES[@]}" shrink "$N/qa.json" -)
case $? in
  0) ;;
  1) fail "tables would shrink by more than 2% against the live $SCHEMA: $shrunk -- not loaded" ;;
  *) fail "the shrink check did not run -- not loaded" ;;
esac
changed=0
for f in "$N"/*.csv; do cmp -s "$f" "out.loaded/${f##*/}" || changed=1; done
if [ $changed = 0 ]; then
  say "no change against the last loaded build (out.loaded): not loaded"
  rm -rf "$N"; exit 0
fi
counts=$("${GATES[@]}" summary "$N/qa.json")
if [ $DRY = 1 ]; then
  say "gates passed; would copy $N to the database host as out.daily and load it: $counts"; exit 0
fi

# 6. copy and load
rsync -a --delete -e "${SSH[*]}" "$N/" "${REMOTE:+$REMOTE:}$DB_DATA/out.daily/" \
  || fail "copying the build to the database host failed -- not loaded"
db_host "timeout 1800 bash $(printf %q "$DB_CODE/load_zstruct.sh") --config $(printf %q "$DB_CONF") out.daily" \
  > logs/refresh_load.log 2>&1
rc=$?
{ [ $rc = 0 ] && grep -q 'LOADED:' logs/refresh_load.log; } \
  || fail "the load on the database host failed (rc=$rc, logs/refresh_load.log); its log says whether the swap happened"
say "$(grep 'LOADED:' logs/refresh_load.log | cut -c10-200)"
rm -rf out.loaded.prev
[ -d out.loaded ] && mv out.loaded out.loaded.prev
mv "$N" out.loaded
say "done: $counts"
