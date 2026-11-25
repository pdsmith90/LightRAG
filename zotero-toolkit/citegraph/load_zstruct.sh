#!/usr/bin/env bash
# Load the citation-graph schema into the LightRAG PostgreSQL database: build <schema>_next from a build folder in
# one transaction, check every table's row count against the build's qa.json and run the pgvector `similar` query
# once, then swap it in as <schema> (the schema lightrag/zotero_struct.py reads; default zstruct). Nothing outside
# <schema>_next, <schema> and <schema>_prev is created or changed, and no server restart is needed.
# Undo after the swap: psql -c 'DROP SCHEMA zstruct CASCADE' (first load), or the undo line in DIR/swap_zstruct.sql.
#
# Run it on the database host (refresh.sh does, over ssh), where the build folder is:
#   load_zstruct.sh --config FILE [DIR]
# FILE is a bash file with the site's settings (citegraph.env.example); CITEGRAPH_CONFIG gives it as well.
# DIR: a build folder name in $CITEGRAPH_DATA (default out), such as out.daily or out.final-20261127.
set -euo pipefail

usage() { echo "usage: $0 --config FILE [DIR]   (FILE may also come from CITEGRAPH_CONFIG)" >&2; }
CONF=${CITEGRAPH_CONFIG:-}
DIR=out
while [ $# -gt 0 ]; do
  case $1 in
    --config) [ $# -ge 2 ] || { usage; exit 2; }; CONF=$2; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    -*) usage; exit 2 ;;
    *) DIR=$1; shift ;;
  esac
done
[ -n "$CONF" ] || { echo "load_zstruct.sh: no config (--config FILE or CITEGRAPH_CONFIG)" >&2; usage; exit 2; }
[ -r "$CONF" ] || { echo "load_zstruct.sh: cannot read the config $CONF" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=citegraph.env.example
. "$CONF"
[ -n "${CITEGRAPH_DATA:-}" ] || { echo "load_zstruct.sh: CITEGRAPH_DATA is not set in $CONF" >&2; exit 2; }
S=${CITEGRAPH_SCHEMA:-zstruct}
[[ "$S" =~ ^[a-z_][a-z0-9_]*$ ]] || { echo "load_zstruct.sh: CITEGRAPH_SCHEMA must be a lower-case identifier, got '$S'" >&2; exit 2; }
for v in PGHOST PGPORT PGUSER PGDATABASE; do
  [ -z "${!v:-}" ] || export "${v?}"  # the variable named by v
done
if [ -n "${CITEGRAPH_PGPASSWORD_FILE:-}" ]; then
  pwfile=${CITEGRAPH_PGPASSWORD_FILE/#\~\//$HOME/}   # a leading ~/ means this host's home
  [ -r "$pwfile" ] || { echo "load_zstruct.sh: cannot read CITEGRAPH_PGPASSWORD_FILE $pwfile" >&2; exit 2; }
  PGPASSWORD=$(cat "$pwfile")
  export PGPASSWORD
fi
pg() { psql -X -At -F ' ' -v ON_ERROR_STOP=1 "$@"; }

[[ "$DIR" =~ ^out[A-Za-z0-9._-]*$ ]] || { echo "DIR must be a build folder name in $CITEGRAPH_DATA, like out.final-20261127" >&2; exit 1; }
O=$CITEGRAPH_DATA/$DIR
mkdir -p "$CITEGRAPH_DATA/logs"
LOG=$CITEGRAPH_DATA/logs/load_zstruct.log
exec > >(tee -a "$LOG") 2>&1
die() { echo "$(date +%T) FATAL: $*" >&2; exit 1; }
echo "$(date '+%F %T') load_zstruct $S $DIR"

if [ -n "${CITEGRAPH_LOAD_WINDOW:-}" ]; then
  [[ "$CITEGRAPH_LOAD_WINDOW" =~ ^([0-9]{4})-([0-9]{4})$ ]] || die "CITEGRAPH_LOAD_WINDOW must look like 0845-2300, got '$CITEGRAPH_LOAD_WINDOW'"
  from=${BASH_REMATCH[1]}; to=${BASH_REMATCH[2]}
  hm=$(date +%H%M)
  { [ "$hm" -ge "$from" ] && [ "$hm" -lt "$to" ]; } \
    || die "it is $(date +%H:%M): run between ${from:0:2}:${from:2} and ${to:0:2}:${to:2} (CITEGRAPH_LOAD_WINDOW)"
fi
for f in "apply_$S.sql" "swap_$S.sql" qa.json papers.csv cites.csv; do [ -s "$O/$f" ] || die "missing $O/$f"; done
python3 "$HERE/refresh_gates.py" hard-checks "$O/qa.json" || die "qa.json: the build's hard checks did not pass"
[ "$(pg -c "SELECT count(*) FROM pg_extension WHERE extname = 'vector'")" = 1 ] || die "pgvector is not installed in the database"

echo "$(date +%T) apply: building ${S}_next"
( cd "$O" && pg -q -f "apply_$S.sql" ) || die "apply failed and rolled back -- ${S}_next not created, nothing changed"

drop_next() { pg -q -c "DROP SCHEMA IF EXISTS ${S}_next CASCADE" && echo "$(date +%T) ${S}_next dropped"; }
bad=0
while IFS=$'\t' read -r t n; do
  got=$(pg -c "SELECT count(*) FROM ${S}_next.\"$t\"") || { echo "  $t: cannot count"; bad=1; continue; }
  if [ "$got" = "$n" ]; then echo "  $t: $got ok"; else echo "  $t: $got in the DB, $n in qa.json"; bad=1; fi
done < <(python3 "$HERE/refresh_gates.py" row-counts "$O/qa.json")
[ "$bad" = 0 ] || { drop_next; die "row counts differ from qa.json"; }

# the server's `similar` query (lightrag/zotero_struct.py) against the new tables, once, for a paper with an embedding
sim=$(pg -c "WITH p AS (SELECT paper_id FROM ${S}_next.paper_embeddings WHERE embedding IS NOT NULL ORDER BY paper_id LIMIT 1),
q AS (SELECT embedding FROM ${S}_next.paper_embeddings WHERE paper_id = (SELECT paper_id FROM p))
SELECT count(*) || ' rows, top ' || coalesce(round(max(s)::numeric, 4)::text, 'none') FROM (
  SELECT 1 - (e.embedding <=> q.embedding) AS s FROM ${S}_next.paper_embeddings e, q
  WHERE e.paper_id <> (SELECT paper_id FROM p) ORDER BY e.embedding <=> q.embedding LIMIT 10) x") \
  || { drop_next; die "the similar query failed on ${S}_next"; }
echo "  similar (pgvector): $sim"
case "$sim" in "10 rows, top 0."*|"10 rows, top 1"*) ;; *) drop_next; die "similar query gave '$sim', expected 10 rows with a similarity";; esac

echo "$(date +%T) swap: ${S}_next -> $S"
pg -q -f "$O/swap_$S.sql" || die "swap failed and rolled back -- ${S}_next is still there, $S unchanged"
echo "$(date +%T) LOADED: schema $S = $(pg -c "SELECT count(*) FROM pg_tables WHERE schemaname = '$S'") tables;" \
     "papers $(pg -c "SELECT count(*) FROM $S.papers"), cites $(pg -c "SELECT count(*) FROM $S.cites")." \
     "${S}_prev: $(pg -c "SELECT count(*) FROM pg_namespace WHERE nspname = '${S}_prev'") (drop it with $DIR/drop_prev_$S.sql once the server reads the new one)"
