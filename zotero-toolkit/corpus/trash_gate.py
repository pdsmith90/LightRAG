#!/usr/bin/env python3
"""
trash_gate.py — an attachment in the Zotero trash never enters the corpus.

Zotero keeps a trashed item, and its file, until the trash is emptied; a synced
library downloads the files of trashed attachments like any other. build_corpus.py
walks the storage folders and knows nothing of the trash, so without this gate a
trashed attachment is converted and ingested. The common case is a PDF renamed by
replacement: the old attachment goes to the trash and the same file is attached again
under a new key, so both copies would be built.

The gate keeps one denylist line per trashed attachment (the attachment itself or its
parent item in the trash), tagged `# trash-sync`, in the same file build_corpus.py and
quarantine_junk.py read:

  * an attachment newly in the trash gets a line, unless the file already lists its key;
  * an attachment restored from the trash loses its line;
  * an attachment erased from the library (trash emptied) keeps its line, so a storage
    folder left behind is still skipped.

Only lines with the tag are ever removed. Run it after build_metadata.py and before
build_corpus.py.

Scope: like quarantine_junk.py and edition_gate.py, this keeps a trashed attachment
OUT of the corpus. A document LightRAG has already ingested stays in the index until it
is deleted there, and a live copy that a duplicate gate blocked as a copy of the
trashed one stays blocked until that gate's line is removed.

Usage:
  python3 trash_gate.py --zotero ~/Zotero --denylist dupe_denylist.txt [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys

TAG = "# trash-sync "


def trashed_attachments(zotero_dir: str) -> dict[str, bool]:
    """attachment key -> True when it (or its parent item) is in the trash."""
    db = os.path.join(zotero_dir, "zotero.sqlite")
    if not os.path.isfile(db):
        raise FileNotFoundError(f"no zotero.sqlite in {zotero_dir} (pass --zotero)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """SELECT i.key, a.itemID IN (SELECT itemID FROM deletedItems)
                      OR a.parentItemID IN (SELECT itemID FROM deletedItems)
               FROM itemAttachments a JOIN items i ON i.itemID = a.itemID"""
        ).fetchall()
    finally:
        con.close()
    return {key: bool(t) for key, t in rows}


def entry(line: str) -> str:
    return line.split("#")[0].strip()


def sync(
    lines: list[str], attachments: dict[str, bool], today: str
) -> tuple[list[str], list[str], list[str]]:
    """(new lines, keys added, keys removed). `lines` are the denylist's lines with their
    newlines; `attachments` maps every attachment key the library knows to its trash state."""
    removed = [
        entry(ln) for ln in lines if TAG in ln and attachments.get(entry(ln)) is False
    ]
    keep = [
        ln for ln in lines if not (TAG in ln and attachments.get(entry(ln)) is False)
    ]
    listed = {entry(ln) for ln in keep if entry(ln)}
    added = sorted(k for k, t in attachments.items() if t and k not in listed)
    if added:
        if keep and not keep[-1].endswith("\n"):
            keep[-1] += "\n"
        keep.append(f"# {today}: attachments in the Zotero trash (trash_gate.py)\n")
        keep += [f"{k}   {TAG}{today}: in the Zotero trash\n" for k in added]
    return keep, added, removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zotero", default=os.path.expanduser("~/Zotero"))
    ap.add_argument(
        "--denylist",
        required=True,
        help="The denylist the trash lines are kept in (dupe_denylist.txt).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only.")
    a = ap.parse_args()

    lines = []
    if os.path.exists(a.denylist):
        with open(a.denylist, encoding="utf-8") as f:
            lines = f.readlines()
    new, added, removed = sync(
        lines, trashed_attachments(a.zotero), dt.date.today().isoformat()
    )
    would = "would be " if a.dry_run else ""
    for k in added:
        print(f"{k}  {would}denylisted (in the Zotero trash)")
    for k in removed:
        print(f"{k}  {would}un-denylisted (restored from the trash)")
    print(
        f"{len(added)} trashed attachment(s) {would}added, {len(removed)} restored one(s) {would}removed"
    )
    if new != lines and not a.dry_run:
        tmp = a.denylist + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(new))
        os.replace(
            tmp, a.denylist
        )  # build_corpus and quarantine_junk read the old or the new file, never half
    return 0


if __name__ == "__main__":
    sys.exit(main())
