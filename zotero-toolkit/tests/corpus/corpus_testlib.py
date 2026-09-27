"""Shared helpers for the corpus tests: paths, and a minimal synthetic Zotero data dir.

Every storage key here contains a 0 or a 1, which real Zotero keys never do (their
alphabet is 23456789ABCDEFGHIJKLMNPQRSTUVWXYZ), so no fixture can name a real item.
"""

import os
import sqlite3
import subprocess
import sys

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
CORPUS_DIR = os.path.join(REPO, "corpus")
if CORPUS_DIR not in sys.path:
    sys.path.insert(0, CORPUS_DIR)

# Exactly the tables and columns build_metadata.py reads; a real zotero.sqlite has
# many more of both.
SCHEMA = """
CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE);
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, path TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT UNIQUE);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value UNIQUE);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
CREATE TABLE itemCreators (itemID INT, creatorID INT, orderIndex INT);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE itemTags (itemID INT, tagID INT);
"""


def make_zotero_db(zotero_dir, items):
    """Write <zotero_dir>/zotero.sqlite holding `items`, in order.

    Each item is a dict: key (required); fields {fieldName: value}; creators
    [(lastName, firstName)] in author order; tags [name]; and, for an attachment,
    path ("storage:<file>", or None for a linked URL) plus optionally parent (the
    key of an item listed earlier).
    """
    os.makedirs(zotero_dir, exist_ok=True)
    db = os.path.join(zotero_dir, "zotero.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    ids = {}

    def rowid(sql_select, sql_insert, arg):
        row = con.execute(sql_select, (arg,)).fetchone()
        return row[0] if row else con.execute(sql_insert, (arg,)).lastrowid

    for it in items:
        iid = con.execute("INSERT INTO items (key) VALUES (?)", (it["key"],)).lastrowid
        ids[it["key"]] = iid
        for name, value in (it.get("fields") or {}).items():
            fid = rowid(
                "SELECT fieldID FROM fields WHERE fieldName=?",
                "INSERT INTO fields (fieldName) VALUES (?)",
                name,
            )
            vid = rowid(
                "SELECT valueID FROM itemDataValues WHERE value=?",
                "INSERT INTO itemDataValues (value) VALUES (?)",
                value,
            )
            con.execute("INSERT INTO itemData VALUES (?,?,?)", (iid, fid, vid))
        # insert creators in reverse so only orderIndex can put them in order
        creators = list(enumerate(it.get("creators") or []))
        for order, (last, first) in reversed(creators):
            cid = con.execute(
                "INSERT INTO creators (firstName, lastName) VALUES (?,?)", (first, last)
            ).lastrowid
            con.execute("INSERT INTO itemCreators VALUES (?,?,?)", (iid, cid, order))
        for name in it.get("tags") or []:
            tid = rowid(
                "SELECT tagID FROM tags WHERE name=?",
                "INSERT INTO tags (name) VALUES (?)",
                name,
            )
            con.execute("INSERT INTO itemTags VALUES (?,?)", (iid, tid))
        if "path" in it:
            parent = ids[it["parent"]] if it.get("parent") else None
            con.execute(
                "INSERT INTO itemAttachments VALUES (?,?,?)", (iid, parent, it["path"])
            )
    con.commit()
    con.close()
    return db


def run_script(name, *args, env_extra=None, env_drop=("ZOTERO_METADATA",)):
    """Run corpus/<name> as a CLI in a fresh interpreter; return the CompletedProcess."""
    env = {k: v for k, v in os.environ.items() if k not in env_drop}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, os.path.join(CORPUS_DIR, name), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
