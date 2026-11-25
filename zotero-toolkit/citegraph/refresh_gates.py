#!/usr/bin/env python3
"""refresh_gates.py - the checks refresh.sh and load_zstruct.sh make on a build's qa.json.

Each subcommand reads the qa.json that build_struct_layer.py writes. Exit status: 0 = pass, 1 = the check
failed (the reason is on stderr, or on stdout for `shrink`), 2 = bad input (unreadable file, malformed JSON,
a table name that is not a plain identifier).

  qa-gate QA              the hard checks AND the reference check (oracle) passed
  hard-checks QA          the hard checks passed (the load script's own check; no reference check needed)
  row-counts QA           print "<table><TAB><rows>" per table, sorted: what the loaded tables must hold
  count-sql QA [--schema S]
                          print one "SELECT '<table>', count(*) FROM <S>.<table>;" per table, for psql
  shrink QA LIVE          LIVE = lines "<table> <rows>" counted in the live schema (a file, or - for stdin).
                          Print the tables whose new row count is under 98% of the live one, comma separated;
                          exit 1 if there are any. A table new to the build (not in LIVE) cannot shrink.
  summary QA              print "papers=N cites=N cite_contexts=N external_works=N paper_embeddings=N"
"""

from __future__ import annotations

import argparse
import json
import re
import sys

SHRINK_LIMIT_PERCENT = 98  # a table may lose at most 2% of the live rows
SUMMARY_TABLES = (
    "papers",
    "cites",
    "cite_contexts",
    "external_works",
    "paper_embeddings",
)
TABLE_RE = re.compile(r"[a-z_]+")
SCHEMA_RE = re.compile(r"[a-z_][a-z0-9_]*")


class BadInput(Exception):
    pass


def load_qa(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            qa = json.load(f)
    except (OSError, ValueError) as e:
        raise BadInput(f"cannot read {path}: {e}") from e
    if not isinstance(qa, dict):
        raise BadInput(f"{path}: not a JSON object")
    return qa


def row_counts(qa: dict) -> dict[str, int]:
    counts = qa.get("row_counts")
    if not isinstance(counts, dict) or not counts:
        raise BadInput("qa.json has no row_counts")
    for table in counts:
        if not TABLE_RE.fullmatch(table):
            raise BadInput(f"unexpected table name in row_counts: {table!r}")
    return counts


def hard_checks_passed(qa: dict) -> bool:
    return bool((qa.get("hard_checks") or {}).get("passed"))


def reference_check_passed(qa: dict) -> bool:
    return bool((qa.get("oracle") or {}).get("passed"))


def parse_live(text: str) -> dict[str, int]:
    live = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            raise BadInput(f"cannot read a live count from: {line!r}")
        live[parts[0]] = int(parts[1])
    return live


def shrunk_tables(new: dict[str, int], live: dict[str, int]) -> list[str]:
    """'table live -> new' for every table in both whose new count is under 98% of the live count."""
    return [
        f"{t} {live[t]} -> {n}"
        for t, n in sorted(new.items())
        if t in live and n * 100 < SHRINK_LIMIT_PERCENT * live[t]
    ]


def run(args: argparse.Namespace) -> int:
    qa = load_qa(args.qa)
    if args.cmd == "qa-gate":
        if not hard_checks_passed(qa):
            print("the hard checks did not pass", file=sys.stderr)
            return 1
        if not reference_check_passed(qa):
            print("the reference check is missing or did not pass", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "hard-checks":
        if not hard_checks_passed(qa):
            print("the hard checks did not pass", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "row-counts":
        for t, n in sorted(row_counts(qa).items()):
            print(f"{t}\t{n}")
        return 0
    if args.cmd == "count-sql":
        if not SCHEMA_RE.fullmatch(args.schema):
            raise BadInput(f"bad schema name {args.schema!r}")
        for t in sorted(row_counts(qa)):
            print(f'SELECT {t!r}, count(*) FROM {args.schema}."{t}";')
        return 0
    if args.cmd == "shrink":
        text = (
            sys.stdin.read()
            if args.live == "-"
            else open(args.live, encoding="utf-8").read()
        )
        shrunk = shrunk_tables(
            {t: int(n) for t, n in row_counts(qa).items()}, parse_live(text)
        )
        if shrunk:
            print(", ".join(shrunk))
            return 1
        return 0
    if args.cmd == "summary":
        counts = qa.get("row_counts") or {}
        print(" ".join(f"{t}={counts.get(t, 0)}" for t in SUMMARY_TABLES))
        return 0
    raise AssertionError(args.cmd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("qa-gate", "hard-checks", "row-counts", "summary"):
        sub.add_parser(name).add_argument("qa")
    p = sub.add_parser("count-sql")
    p.add_argument("qa")
    p.add_argument("--schema", default="zstruct")
    p = sub.add_parser("shrink")
    p.add_argument("qa")
    p.add_argument("live")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except (BadInput, OSError) as e:
        print(f"refresh_gates: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
