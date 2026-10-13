"""Tests for corpus/edition_gate.py: the newest edition of a book supersedes the older ones."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_testlib import run_script  # noqa: E402  (also puts corpus/ on sys.path)
import edition_gate  # noqa: E402


def book(key, title, author, year, atts=(), has_doc=True):
    return {
        "key": key,
        "title": title,
        "author": author,
        "year": year,
        "attachments": list(atts),
        "has_doc": has_doc,
    }


def pairs(books):
    return {(o["key"], n["key"]) for o, n in edition_gate.superseded(books)}


def test_newer_edition_supersedes_older_ignoring_edition_note_and_subtitle():
    books = [
        book("TEST0001", "Matrix methods (3rd ed.)", "Alder", 1996),
        book("TEST0002", "Matrix Methods", "Alder", 2013),
        book(
            "TEST0003",
            "Orbit Surveying: Foundations, Methods, and Applications",
            "Brook",
            1993,
        ),
        book("TEST0004", "Orbit surveying", "Brook", 2003),
    ]
    assert pairs(books) == {("TEST0001", "TEST0002"), ("TEST0003", "TEST0004")}


def test_the_newest_of_several_editions_wins():
    books = [
        book("TEST0001", "Random fields", "Ledoux", 1960),
        book("TEST0002", "Random fields", "Ledóux", 1986),
        book("TEST0003", "Random Fields", "LEDOUX", 1999),
    ]
    assert pairs(books) == {("TEST0001", "TEST0003"), ("TEST0002", "TEST0003")}


def test_numbered_volumes_are_separate_works():
    books = [
        book("TEST0001", "Field Theory I", "Calder", 1977),
        book("TEST0002", "Field Theory II", "Calder", 1978),
        book("TEST0003", "VII Symposium on Applied Mechanics", "Dunmore", 2012),
        book("TEST0004", "VIII Symposium on Applied Mechanics", "Dunmore", 2016),
        book("TEST0005", "Handbook of Optics, volume 2", "Ellery", 1990),
        book("TEST0006", "Handbook of Optics, volume 3", "Ellery", 2001),
    ]
    assert pairs(books) == set()


def test_different_subtitles_under_one_main_title_are_different_books():
    books = [
        book("TEST0001", "Wave physics: an introduction", "Fenwick", 2023),
        book("TEST0002", "Wave Physics: Nonlinear Media", "Fenwick", 2026),
        book(
            "TEST0003",
            "Signal recovery: implementation and applications",
            "Garrow",
            1979,
        ),
        book(
            "TEST0004", "Signal recovery: the fundamentals of sampling", "Garrow", 1980
        ),
        # one side without a subtitle is still the same work
        book("TEST0005", "Global ocean circulation", "Hartley", 2004),
        book(
            "TEST0006",
            "Global Ocean Circulation: Applications of Tracer Theory",
            "Hartley",
            2016,
        ),
    ]
    assert pairs(books) == {("TEST0005", "TEST0006")}


def test_no_supersession_without_a_newer_document_or_a_matching_author():
    books = [
        book("TEST0001", "Learning machines", "Ivers", 2016),
        book("TEST0002", "Learning machines", "Ivers", 2021, has_doc=False),
        book("TEST0003", "Learning machines", "Jarrow", 2019),
        book("TEST0004", "Learning machines", "Ivers", 2016),  # same year: a duplicate
        book("TEST0005", "Learning machines", "Ivers", None),  # no year: no verdict
    ]
    assert pairs(books) == set()


SCHEMA = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT, key TEXT UNIQUE);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT UNIQUE);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value UNIQUE);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
CREATE TABLE itemCreators (itemID INT, creatorID INT, orderIndex INT);
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, contentType TEXT);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
INSERT INTO itemTypes VALUES (1, 'book'), (2, 'attachment'), (3, 'journalArticle');
INSERT INTO fields VALUES (1, 'title'), (2, 'date');
"""


def make_db(path, rows):
    """rows: (key, type, title, date, author, parent, contentType, trashed)."""
    con = sqlite3.connect(os.path.join(path, "zotero.sqlite"))
    con.executescript(SCHEMA)
    ids = {}
    for key, typ, title, date, author, parent, ct, trashed in rows:
        tid = {"book": 1, "attachment": 2, "journalArticle": 3}[typ]
        iid = con.execute(
            "INSERT INTO items (itemTypeID, key) VALUES (?,?)", (tid, key)
        ).lastrowid
        ids[key] = iid
        for fid, value in ((1, title), (2, date)):
            if value:
                con.execute(
                    "INSERT OR IGNORE INTO itemDataValues (value) VALUES (?)", (value,)
                )
                vid = con.execute(
                    "SELECT valueID FROM itemDataValues WHERE value=?", (value,)
                ).fetchone()[0]
                con.execute("INSERT INTO itemData VALUES (?,?,?)", (iid, fid, vid))
        if author:
            cid = con.execute(
                "INSERT INTO creators (lastName) VALUES (?)", (author,)
            ).lastrowid
            con.execute("INSERT INTO itemCreators VALUES (?,?,0)", (iid, cid))
        if parent:
            con.execute(
                "INSERT INTO itemAttachments VALUES (?,?,?)", (iid, ids[parent], ct)
            )
        if trashed:
            con.execute("INSERT INTO deletedItems VALUES (?)", (iid,))
    con.commit()
    con.close()


LIBRARY = [
    (
        "TEST0001",
        "book",
        "Matrix methods (3rd ed.)",
        "1996-11-01",
        "Alder",
        None,
        None,
        False,
    ),
    ("TEST0101", "attachment", None, None, None, "TEST0001", "application/pdf", False),
    ("TEST0102", "attachment", None, None, None, "TEST0001", "text/html", False),
    ("TEST0002", "book", "Matrix Methods", "2013", "Alder", None, None, False),
    ("TEST0201", "attachment", None, None, None, "TEST0002", "application/pdf", False),
    # a newer edition in the trash supersedes nothing
    ("TEST0003", "book", "Linear systems", "1980", "Kestrel", None, None, False),
    ("TEST0301", "attachment", None, None, None, "TEST0003", "application/pdf", False),
    ("TEST0004", "book", "Linear systems", "2001", "Kestrel", None, None, True),
    ("TEST0401", "attachment", None, None, None, "TEST0004", "application/pdf", False),
    # an article is never an edition of a book
    (
        "TEST0005",
        "journalArticle",
        "Matrix methods",
        "2020",
        "Alder",
        None,
        None,
        False,
    ),
    ("TEST0501", "attachment", None, None, None, "TEST0005", "application/pdf", False),
]


def test_cli_denylists_every_attachment_of_the_older_edition_once(tmp_path):
    make_db(str(tmp_path), LIBRARY)
    deny = tmp_path / "dupe_denylist.txt"
    deny.write_text("TEST0999  # an earlier entry\n", encoding="utf-8")

    dry = run_script(
        "edition_gate.py",
        "--zotero",
        str(tmp_path),
        "--denylist",
        str(deny),
        "--dry-run",
    )
    assert dry.returncode == 0, dry.stderr
    assert "2 attachment(s) of older editions would be denylisted" in dry.stdout
    assert deny.read_text(encoding="utf-8") == "TEST0999  # an earlier entry\n"

    run = run_script(
        "edition_gate.py", "--zotero", str(tmp_path), "--denylist", str(deny)
    )
    assert run.returncode == 0, run.stderr
    lines = deny.read_text(encoding="utf-8").splitlines()
    assert lines[1:] == [
        "TEST0101  # edition: TEST0001 (1996) superseded by TEST0002 (2013)",
        "TEST0102  # edition: TEST0001 (1996) superseded by TEST0002 (2013)",
    ]

    again = run_script(
        "edition_gate.py", "--zotero", str(tmp_path), "--denylist", str(deny)
    )
    assert "0 attachment(s)" in again.stdout
    assert len(deny.read_text(encoding="utf-8").splitlines()) == 3
