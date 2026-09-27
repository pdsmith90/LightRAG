#!/usr/bin/env python3
"""
build_metadata.py — (Re)generate zotero_metadata.json from zotero.sqlite.

Maps every attachment's storage KEY -> its bibliographic metadata
(title, authors, year, DOI, publication, tags, abstract), pulling from the
attachment's PARENT item when present, else from the attachment itself.

Run this whenever your Zotero library changes, then re-run build_corpus.py.
Zotero normally locks zotero.sqlite while it is running; if you get
"database is locked", close Zotero or point --zotero at a copy of the data dir.

Usage:
  python3 build_metadata.py --zotero ~/Zotero --out ~/Zotero/zotero_metadata.json

--out defaults to $ZOTERO_METADATA, else <zotero>/zotero_metadata.json.
"""

from __future__ import annotations
import argparse, json, os, re, sqlite3


def _fields(con, iid):
    rows = con.execute(
        """SELECT f.fieldName, idv.value FROM itemData id
           JOIN fields f ON f.fieldID=id.fieldID
           JOIN itemDataValues idv ON idv.valueID=id.valueID
           WHERE id.itemID=?""",
        (iid,),
    ).fetchall()
    return {k: v for k, v in rows}


def _creators(con, iid):
    rows = con.execute(
        """SELECT cr.lastName, cr.firstName FROM itemCreators ic
           JOIN creators cr ON cr.creatorID=ic.creatorID
           WHERE ic.itemID=? ORDER BY ic.orderIndex""",
        (iid,),
    ).fetchall()
    return [(" ".join(x for x in [a, b] if x)).strip() for a, b in rows]


def _tags(con, iid):
    return [
        x[0]
        for x in con.execute(
            "SELECT t.name FROM itemTags it JOIN tags t ON t.tagID=it.tagID WHERE it.itemID=?",
            (iid,),
        ).fetchall()
    ]


def build(zotero_dir: str) -> dict:
    db = os.path.join(zotero_dir, "zotero.sqlite")
    if not os.path.isfile(db):  # sqlite3.connect would create an empty database there
        raise FileNotFoundError(
            f"no zotero.sqlite in {zotero_dir} (pass --zotero <Zotero data dir>)"
        )
    con = sqlite3.connect(db)

    # Fetch all attachment rows into memory first (avoids cursor exhaustion)
    # Every content type, not just PDF: build_corpus converts 7 file kinds
    # (EXT2KIND — pdf/epub/docx/pptx/djvu/html/txt), so filtering to PDFs here
    # would leave every non-PDF document in the corpus with no metadata entry and
    # a slug-derived citation (in one library a few percent of the documents,
    # mostly html, cited without authors/year/DOI).
    attachments = con.execute(
        """SELECT ia.itemID, i.key, ia.parentItemID, ia.path
           FROM itemAttachments ia JOIN items i ON i.itemID=ia.itemID"""
    ).fetchall()

    index = {}
    for iid, key, pid, path in attachments:
        src = pid or iid
        fl = _fields(con, src)
        fn = (
            path.split(":", 1)[1]
            if path and path.startswith("storage:")
            else (path or "")
        )
        ym = re.search(r"(\d{4})", fl.get("date", "") or "")
        pub = (
            fl.get("publicationTitle")
            or fl.get("bookTitle")
            or fl.get("proceedingsTitle")
            or fl.get("conferenceName")
            or ""
        )
        index[key] = {
            "filename": fn,
            "title": fl.get("title") or os.path.splitext(fn)[0],
            "authors": _creators(con, src),
            "year": ym.group(1) if ym else "",
            "doi": fl.get("DOI", ""),
            "publication": pub,
            "tags": _tags(con, src),
            "abstract": (fl.get("abstractNote", "") or "")[:1500],
            "source": "parent" if pid else "standalone",
        }
    con.close()
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zotero",
        default=os.path.expanduser("~/Zotero"),
        help="Zotero data dir (contains zotero.sqlite). Default: ~/Zotero.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON. Default: $ZOTERO_METADATA, else <zotero>/zotero_metadata.json.",
    )
    a = ap.parse_args()
    a.out = (
        a.out
        or os.environ.get("ZOTERO_METADATA")
        or os.path.join(a.zotero, "zotero_metadata.json")
    )
    idx = build(a.zotero)
    # Sticky orphans. An attachment deleted or merged away in Zotero keeps its
    # document in the KB (deleting there is a separate, gated, slow process), and
    # without an entry here that document cites as a bare filename (dozens did in one
    # library, some recoverable only because an older copy of this file still had
    # them). So a rebuild never forgets a key: every entry the previous output had
    # for a key Zotero no longer knows is carried forward, flagged, so a future
    # cleanup can list them.
    carried = 0
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out, encoding="utf-8"))
        except Exception as e:
            print(f"WARN: previous {a.out} unreadable ({e}); carrying nothing")
            prev = {}
        for key, entry in prev.items():
            if key not in idx:
                idx[key] = {**entry, "orphan": True}
                carried += 1
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:  # closed (flushed) before the rename
        json.dump(idx, fh, ensure_ascii=False, indent=1)
    os.replace(
        tmp, a.out
    )  # readers (server, MCP proxy, build_corpus) see old or new, never a partial file
    parented = sum(1 for v in idx.values() if v["source"] == "parent")
    print(
        f"wrote {a.out}: {len(idx)} entries ({parented} with parent metadata, {carried} carried orphans)"
    )


if __name__ == "__main__":
    main()
