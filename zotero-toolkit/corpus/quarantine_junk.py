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

3. GLYPH CODES (automatic, independent of 1 and 2). Some scanned PDFs carry a text
   layer whose fonts map every glyph into the Unicode Private Use Area (U+E000-F8FF,
   plus planes 15 and 16). PDF text extractors, and Zotero's own full-text index,
   return those code points as "text", so such a file is neither empty (which would
   send it to OCR in build_corpus) nor markup (rule 1): one scanned encyclopedia
   became thousands of chunks of glyph codes that extracted nothing and kept the
   ingest pipeline busy for a night. On a corpus of several thousand academic PDFs
   the garbled files scored above 0.75 and every genuine file below 0.05 (symbol-font
   Greek in a mathematics handbook was the highest), so 0.30 sits in a clean gap.
   The ratio is PUA code points over NON-whitespace characters; files under 500 such
   characters are left to --min-chars. build_corpus applies the same bar
   (looks_garbled) and routes those PDFs to OCR; this rule is the second line.

Scope: the corpus ROOT only. Those files are still pending ingest, so moving one
keeps it out of the KB. Files already in __parsed__/ have been ingested; moving
them does not retract them (that needs a doc delete) and would strand them from
build_corpus's up-to-date check.

4. GLYPH GARBAGE (automatic, like 3). Three more shapes a broken text layer takes --
   U+FFFD replacement characters, C0 control codes, letter soup of single characters --
   with the bars build_corpus.garble_reason uses (the same code, duplicated here because
   build_corpus imports load_denylist from this file). Reason `garbled` in the log.

Never deletes: moves to <corpus>_junk_quarantine/ and appends to quarantined.jsonl.
Always exits 0 so a bad heuristic can never abort a scheduled corpus update.

Usage:
  python3 quarantine_junk.py --corpus ~/Zotero/rag_corpus [--min-junk 0.90] [--min-pua 0.30]
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
# Private Use Area: the BMP block and the two supplementary planes (U+F0000-10FFFD).
_pua_re = re.compile(r"[\ue000-\uf8ff\U000f0000-\U0010ffff]")
_ws_re = re.compile(r"\s+")
PUA_MIN_NONSPACE = 500  # below this, leave the verdict to --min-chars


def pua_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Private Use Area code points
    (glyph codes from a font with no Unicode mapping). 0.0 for tiny texts."""
    nonspace = len(_ws_re.sub("", text))
    if nonspace < PUA_MIN_NONSPACE:
        return 0.0
    return len(_pua_re.findall(text)) / nonspace


# The other garbage shapes build_corpus.garble_reason knows -- U+FFFD replacement characters,
# C0 control codes, letter soup -- duplicated here because build_corpus imports load_denylist
# from this file. Same bars; see build_corpus for how they were measured.
_ctrl_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_word_re = re.compile(r"[^\W\d_]{4,}")


def garble_reason(text: str):
    """'replacement', 'control', 'letter-soup' or 'no-words' when the text is glyph garbage of
    a shape other than Private Use Area codes (those are pua_ratio's), else None. Texts under
    PUA_MIN_NONSPACE non-space characters are never garbled."""
    ns = _ws_re.sub("", text)
    n = len(ns)
    if n < PUA_MIN_NONSPACE:
        return None
    toks = text.split()
    word = sum(len(w) for w in _word_re.findall(text)) / n
    if ns.count("\ufffd") / n >= 0.05:
        return "replacement"
    if len(_ctrl_re.findall(ns)) / n >= 0.05 and word < 0.25:
        return "control"
    single = sum(1 for t in toks if len(t) == 1 and not t.isdigit()) / len(toks)
    digit = sum(ch.isdigit() for ch in ns) / n
    if single >= 0.35 and word < 0.15 and digit < 0.30:
        return "letter-soup"
    alpha = sum(ch.isalpha() for ch in ns) / n
    if alpha >= 0.30 and word < 0.03:
        return "no-words"
    return None


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
    ap.add_argument(
        "--min-pua",
        type=float,
        default=0.30,
        help="Move files whose non-whitespace text is at least this fraction Private "
        "Use Area glyph codes (default: 0.30; garbled scans score above 0.75, "
        "genuine documents below 0.05).",
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
        jr, pr = junk_ratio(text), pua_ratio(text)
        if key in deny or name in deny:
            hits.append(("denylist", jr, len(text), name, pr, ""))
        elif len(text) >= a.min_chars and jr >= a.min_junk:
            hits.append(("ratio", jr, len(text), name, pr, ""))
        elif pr >= a.min_pua:
            hits.append(("glyphs", pr, len(text), name, pr, ""))
        else:
            why = garble_reason(text)
            if why:
                hits.append(("garbled", 1.0, len(text), name, pr, why))

    if not hits:
        print(
            f"quarantine_junk: 0 junk files "
            f"(ratio>={a.min_junk:.0%}, glyphs>={a.min_pua:.0%}, glyph garbage, or on "
            f"denylist[{len(deny)}]) in {corpus}"
        )
        return 0

    hits.sort(key=lambda h: -h[1])
    if not a.dry_run:
        os.makedirs(qdir, exist_ok=True)
    moved, failed, records = 0, 0, []
    stamp = time.strftime("%F %T")
    for reason, r, size, name, pr, why in hits:
        what = (
            "PUA glyph codes"
            if reason == "glyphs"
            else f"glyph garbage ({why})"
            if reason == "garbled"
            else "junk"
        )
        print(
            f"quarantine_junk: {'DRY ' if a.dry_run else ''}[{reason:8s}] "
            f"{r:6.1%} {what}, {size:9d} chars — {name}"
        )
        records.append(
            {
                "ts": stamp,
                "file": name,
                "reason": reason,
                "junk_ratio": round(r, 4),
                "pua_ratio": round(pr, 4),
                "chars": size,
                "garble": why,
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
