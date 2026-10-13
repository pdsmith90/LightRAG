#!/usr/bin/env python3
"""
edition_gate.py — a newer edition of a book supersedes the older ones in the corpus.

When a library holds two editions of the same book, ingesting both doubles the book's
chunks and entities, and the older text answers questions the newer one revised. The
policy is the same at download and at ingest: the newest edition is the one kept.

At download (outside this toolkit): a fetcher prefers the newest matching edition, and
the item record is updated to describe the edition attached. An older-edition record
whose newer edition the library already holds is merged into the newer record (its
collections move over, the newer record gets a dc:replaces relation, the old one goes
to the trash), the way Zotero's Merge Items does it.

At ingest (this script): for every pair of `book` items that are editions of one work,
every attachment of the older edition is appended to a denylist, so build_corpus.py
skips it and quarantine_junk.py moves any built output aside. Two `book` items are
editions of one work when

  * their first creators have the same surname (case and accents folded),
  * their main titles match (subtitle after ":" and a trailing "(... ed.)" dropped;
    difflib ratio >= 0.85),
  * when BOTH carry a subtitle, the subtitles match too: one author's "Wave Physics:
    an introduction" and "Wave Physics: nonlinear media" are two books, and so are an
    edited volume and a monograph that share a main title. (A subtitle that changed
    between editions then reads as two works -- both are ingested, the safe way to
    be wrong), and
  * any volume number or roman numeral in the main titles is the same, so
    "Field Theory I"/"II" and the "VII"/"VIII" volumes of one symposium series stay
    separate works.

The newer edition must be strictly later and hold a document attachment (PDF, EPUB or
DjVu); without one there is nothing to supersede with. Items in the trash are ignored.

Scope: like quarantine_junk.py, this keeps an edition OUT of the corpus. An older
edition that LightRAG has already ingested stays in the index until its document is
deleted there.

Usage:
  python3 edition_gate.py --zotero ~/Zotero --denylist dupe_denylist.txt [--dry-run]
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sqlite3
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quarantine_junk import load_denylist  # noqa: E402

DOC_TYPES = ("application/pdf", "application/epub+zip", "image/vnd.djvu")
ROMAN = re.compile(r"^(?=[ivxlc]+$)c{0,3}(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
RATIO = 0.85


def fold(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def main_title(t: str | None) -> list[str]:
    """Words of the title before any subtitle, edition note or bracket."""
    head = re.split(r"[:;(\[]| - ", t or "")[0]
    return re.sub(r"[^a-z0-9 ]", " ", fold(head)).split()


def subtitle(t: str | None) -> list[str]:
    """Words after the first ":", without an edition note or bracket."""
    parts = re.split(r":", t or "", maxsplit=1)
    tail = re.split(r"[(\[]", parts[1])[0] if len(parts) > 1 else ""
    return re.sub(r"[^a-z0-9 ]", " ", fold(tail)).split()


def numerals(words: list[str]) -> set[str]:
    return {w for w in words if w.isdigit() or ROMAN.match(w)}


def same_work(a: dict, b: dict) -> bool:
    if not a["author"] or fold(a["author"]) != fold(b["author"]):
        return False
    ta, tb = main_title(a["title"]), main_title(b["title"])
    if not ta or not tb or numerals(ta) != numerals(tb):
        return False
    if difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio() < RATIO:
        return False
    sa, sb = subtitle(a["title"]), subtitle(b["title"])
    return not (sa and sb) or (
        difflib.SequenceMatcher(None, " ".join(sa), " ".join(sb)).ratio() >= RATIO
    )


def superseded(books: list[dict]) -> list[tuple[dict, dict]]:
    """(older, newest) for every book with a strictly newer edition that holds a document.

    A book is {key, title, author, year (int or None), attachments [key], has_doc}.
    """
    out = []
    for old in books:
        if old["year"] is None:
            continue
        newer = [
            b
            for b in books
            if b is not old
            and b["has_doc"]
            and b["year"] is not None
            and b["year"] > old["year"]
            and same_work(old, b)
        ]
        if newer:
            out.append((old, max(newer, key=lambda b: (b["year"], b["key"]))))
    return out


def load_books(zotero_dir: str) -> list[dict]:
    db = os.path.join(zotero_dir, "zotero.sqlite")
    if not os.path.isfile(db):
        raise FileNotFoundError(f"no zotero.sqlite in {zotero_dir} (pass --zotero)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    trash = "SELECT itemID FROM deletedItems"

    def field(iid: int, name: str) -> str:
        row = con.execute(
            """SELECT v.value FROM itemData d JOIN fields f ON f.fieldID=d.fieldID
               JOIN itemDataValues v ON v.valueID=d.valueID
               WHERE d.itemID=? AND f.fieldName=?""",
            (iid, name),
        ).fetchone()
        return row[0] if row else ""

    books = []
    for iid, key in con.execute(
        f"""SELECT i.itemID, i.key FROM items i JOIN itemTypes t ON t.itemTypeID=i.itemTypeID
            WHERE t.typeName='book' AND i.itemID NOT IN ({trash})"""
    ).fetchall():
        author = con.execute(
            """SELECT c.lastName FROM itemCreators ic JOIN creators c ON c.creatorID=ic.creatorID
               WHERE ic.itemID=? ORDER BY ic.orderIndex LIMIT 1""",
            (iid,),
        ).fetchone()
        atts = con.execute(
            f"""SELECT i.key, a.contentType FROM itemAttachments a JOIN items i ON i.itemID=a.itemID
                WHERE a.parentItemID=? AND a.itemID NOT IN ({trash})""",
            (iid,),
        ).fetchall()
        ym = re.search(r"\b(\d{4})\b", field(iid, "date"))
        books.append(
            {
                "key": key,
                "title": field(iid, "title"),
                "author": author[0] if author else "",
                "year": int(ym.group(1)) if ym else None,
                "attachments": [k for k, _ in atts],
                "has_doc": any(ct in DOC_TYPES for _, ct in atts),
            }
        )
    con.close()
    return books


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zotero", default=os.path.expanduser("~/Zotero"))
    ap.add_argument(
        "--denylist",
        required=True,
        help="File the superseded attachment keys are appended to (dupe_denylist.txt).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only.")
    a = ap.parse_args()

    listed = load_denylist(a.denylist)
    lines = []
    for old, new in superseded(load_books(a.zotero)):
        for att in old["attachments"]:
            if att in listed:
                continue
            lines.append(
                f"{att}  # edition: {old['key']} ({old['year']}) superseded by "
                f"{new['key']} ({new['year']})"
            )
    for line in lines:
        print(line)
    print(
        f"{len(lines)} attachment(s) of older editions {'would be ' if a.dry_run else ''}denylisted"
    )
    if lines and not a.dry_run:
        with open(a.denylist, "a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
