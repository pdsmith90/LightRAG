#!/usr/bin/env python3
"""
quarantine_junk.py — keep webpage-snapshot junk out of the LightRAG corpus.

Zotero saves HTML snapshots alongside (or instead of) the real PDF. build_corpus
converts those too, and the result is a .md that is mostly markup: navigation
menus, ad wrappers and base64-inlined images, with a little prose buried inside.
Ingesting one costs hundreds of chunks of noise, and the worst offenders stall
extraction outright — a chunk of pure base64 can make a small extraction LLM emit
tokens until it blows LLM_TIMEOUT, and the document never finishes.

TWO INDEPENDENT RULES, because one is not enough:

1. RATIO (automatic, catches new arrivals). Score = 1 - prose/total, where prose
   is what survives removing base64 runs and HTML tags. On a corpus of academic
   papers, every file scoring >=90% was a saved webpage and real documents stayed
   below about 30%, so 0.90 sits in a clean gap. This is the guard for
   freshly-converted files, which still carry their base64.

2. DENYLIST (explicit, survives regeneration). The ratio rule ALONE is
   self-defeating: quarantining a source makes build_corpus regenerate it from
   Zotero, the rebuild applies the current clean_md, and the file returns under
   any sane junk bar, so it sails back into the KB. That happened to most of the
   first files quarantined.

   And the ratio cannot simply be lowered. Three statistical discriminators were
   tested against a hand-labelled set and all failed:
     * structural HTML tags alone   -> would quarantine a real textbook of
       several hundred KB converted from HTML.
     * HTML + prose < 10KB          -> cuts straight through dozens of legitimate
       conference abstracts (6-11KB prose, same site template).
     * longest continuous prose block -> the ranges of known junk and known real
       documents overlap (both roughly 1-2.5K chars). Not separable.
   Once the base64 is stripped, a conference abstract and a news article are
   statistically identical; the difference is semantic. So confirmed junk is
   listed by Zotero storage key and quarantined on sight, regardless of score.

Scope: the corpus ROOT only. Those files are still pending ingest, so moving one
keeps it out of the KB. Files already in __parsed__/ have been ingested; moving
them does not retract them (that needs a doc delete) and would strand them from
build_corpus's up-to-date check.

Never deletes: moves to <corpus>_junk_quarantine/ and appends to quarantined.jsonl.
Always exits 0 so a bad heuristic can never abort a scheduled corpus update.

Usage:
  python3 quarantine_junk.py --corpus ~/Zotero/rag_corpus [--min-junk 0.90]
                             [--denylist junk_denylist.txt,dupe_denylist.txt] [--dry-run]
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import time

# Same 200-char bar as clean_md's base64 cut in build_corpus.py. Files built before
# that bar was lowered from 800 can still hold line-wrapped blobs (~76 cols, most runs
# 560-796 chars), and this scan finds them.
_b64_re = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_tag_re = re.compile(r"<[^>]+>")


def junk_ratio(text: str) -> float:
    """Fraction of the file that is base64 blob or HTML markup rather than prose."""
    if not text:
        return 0.0
    prose = _tag_re.sub("", _b64_re.sub("", text))
    return 1.0 - (len(prose) / len(text))


def load_denylist(path: str) -> set[str]:
    """Denylist entries, one per line; '#' starts a comment. Missing file = empty.

    An entry is either a bare Zotero storage key (blocks every file from that
    attachment) or an exact filename ending in .md (blocks just that one). The
    filename form matters when a key has both junk and salvaged output: e.g.
    EXAMPLE1__abstract.md is a useless abstract page, while a hand-recovered
    EXAMPLE1__Author_2020_recovered.md must be allowed through.

    Accepts a comma-separated list of files so concerns stay separate, e.g.
    junk_denylist.txt (webpage snapshots) and dupe_denylist.txt (superseded
    duplicate attachments). Missing files are skipped silently.
    """
    entries = set()
    for one in (path or "").split(","):
        one = one.strip()
        if not one or not os.path.exists(one):
            continue
        for line in open(one, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                entries.add(line.split()[0])
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--corpus",
        default=os.path.expanduser("~/Zotero/rag_corpus"),
        help="Corpus dir; only its top level (pending ingest) is scanned.",
    )
    ap.add_argument(
        "--quarantine",
        default=None,
        help="Destination (default: <corpus>_junk_quarantine).",
    )
    ap.add_argument(
        "--denylist",
        default="",
        help="Comma-separated denylist files (keys or exact .md filenames); "
        "pass build_corpus.py the same list. Default: none.",
    )
    ap.add_argument(
        "--min-junk",
        type=float,
        default=0.90,
        help="Move files at or above this junk ratio (default: 0.90).",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Ignore files smaller than this (default: 500).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only; move nothing.")
    a = ap.parse_args()

    corpus = os.path.abspath(os.path.expanduser(a.corpus))
    qdir = a.quarantine or (corpus.rstrip("/") + "_junk_quarantine")
    if not os.path.isdir(corpus):
        print(f"quarantine_junk: no corpus dir {corpus} — nothing to do")
        return 0

    deny = load_denylist(a.denylist)
    hits = []
    for name in sorted(os.listdir(corpus)):
        if not name.endswith(".md"):
            continue  # skips __parsed__/ too: it is a directory
        path = os.path.join(corpus, name)
        if not os.path.isfile(path):
            continue
        key = name.split("__", 1)[0]
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError as e:
            print(f"quarantine_junk: WARN cannot read {name}: {e}")
            continue
        if key in deny or name in deny:
            hits.append(("denylist", junk_ratio(text), len(text), name))
        elif len(text) >= a.min_chars and junk_ratio(text) >= a.min_junk:
            hits.append(("ratio", junk_ratio(text), len(text), name))

    if not hits:
        print(
            f"quarantine_junk: 0 junk files "
            f"(ratio>={a.min_junk:.0%} or on denylist[{len(deny)}]) in {corpus}"
        )
        return 0

    hits.sort(key=lambda h: -h[1])
    if not a.dry_run:
        os.makedirs(qdir, exist_ok=True)
    moved, failed, records = 0, 0, []
    stamp = time.strftime("%F %T")
    for reason, r, size, name in hits:
        print(
            f"quarantine_junk: {'DRY ' if a.dry_run else ''}[{reason:8s}] "
            f"{r:6.1%} junk, {size:9d} chars — {name}"
        )
        records.append(
            {
                "ts": stamp,
                "file": name,
                "reason": reason,
                "junk_ratio": round(r, 4),
                "chars": size,
            }
        )
        if a.dry_run:
            continue
        try:
            shutil.move(os.path.join(corpus, name), os.path.join(qdir, name))
            moved += 1
        except OSError as e:
            print(f"quarantine_junk: WARN could not move {name}: {e}")
            failed += 1

    if not a.dry_run and records:
        try:
            with open(
                os.path.join(qdir, "quarantined.jsonl"), "a", encoding="utf-8"
            ) as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
        except OSError as e:
            print(f"quarantine_junk: WARN could not write audit log: {e}")

    verb = "would quarantine" if a.dry_run else "quarantined"
    n = len(hits) if a.dry_run else moved
    byreason = {}
    for reason, *_ in hits:
        byreason[reason] = byreason.get(reason, 0) + 1
    print(
        f"quarantine_junk: {verb} {n} file(s) {byreason} -> {qdir}"
        + (f"; {failed} failed" if failed else "")
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never break a scheduled corpus update
        print(f"quarantine_junk: ERROR {type(e).__name__}: {e}")
        sys.exit(0)
