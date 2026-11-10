"""Tests for corpus/trash_gate.py: an attachment in the Zotero trash never enters the corpus."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_testlib import run_script  # noqa: E402  (also puts corpus/ on sys.path)
from quarantine_junk import load_denylist  # noqa: E402
import trash_gate  # noqa: E402

TODAY = "2026-11-10"
SCHEMA = """
CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT UNIQUE);
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, contentType TEXT);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
"""


def make_db(path, items, attachments, trashed):
    """items: {itemID: key}; attachments: {itemID: parentItemID or None}; trashed: itemIDs."""
    con = sqlite3.connect(os.path.join(path, "zotero.sqlite"))
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO items VALUES (?, ?)", items.items())
    con.executemany(
        "INSERT INTO itemAttachments VALUES (?, ?, 'application/pdf')",
        attachments.items(),
    )
    con.executemany("INSERT INTO deletedItems VALUES (?)", [(i,) for i in trashed])
    con.commit()
    con.close()


def test_trashed_attachment_and_attachment_of_a_trashed_parent_are_listed():
    new, added, removed = trash_gate.sync(
        [], {"TEST0001": True, "TEST0002": False, "TEST0003": True}, TODAY
    )
    assert added == ["TEST0001", "TEST0003"] and removed == []
    assert "TEST0001   # trash-sync 2026-11-10: in the Zotero trash\n" in new


def test_restored_loses_its_line_erased_keeps_it_others_untouched():
    lines = [
        "REST0001   # trash-sync 2026-11-01: in the Zotero trash\n",
        "GONE0001   # trash-sync 2026-11-01: in the Zotero trash\n",
        "LIVE0001  # edition: OLD (1990) superseded by NEW (2001)\n",
    ]
    new, added, removed = trash_gate.sync(
        lines, {"REST0001": False, "LIVE0001": False}, TODAY
    )
    assert removed == ["REST0001"] and added == []
    assert new == lines[1:]


def test_a_key_listed_for_another_reason_gets_no_second_line():
    lines = ["TEST0001  # edition: TEST0001 (1990) superseded by TEST0009 (2001)"]
    new, added, _ = trash_gate.sync(lines, {"TEST0001": True}, TODAY)
    assert added == [] and new == lines


def test_cli_lists_trashed_once_and_dry_run_writes_nothing(tmp_path):
    # 10 parent (live) -> attachment 11 (trashed); 20 parent (trashed) -> 21 (live row);
    # 31 standalone live; 41 standalone trashed
    make_db(
        str(tmp_path),
        {
            10: "PARE0010",
            11: "TEST0011",
            20: "PARE0020",
            21: "TEST0021",
            31: "TEST0031",
            41: "TEST0041",
        },
        {11: 10, 21: 20, 31: None, 41: None},
        [11, 20, 41],
    )
    deny = tmp_path / "dupe_denylist.txt"
    deny.write_text("KEEP0001  # junk\n", encoding="utf-8")
    dry = run_script(
        "trash_gate.py", "--zotero", tmp_path, "--denylist", deny, "--dry-run"
    )
    assert dry.returncode == 0, dry.stderr
    assert "3 trashed attachment(s) would be added" in dry.stdout
    assert deny.read_text(encoding="utf-8") == "KEEP0001  # junk\n"
    run = run_script("trash_gate.py", "--zotero", tmp_path, "--denylist", deny)
    assert run.returncode == 0, run.stderr
    assert load_denylist(str(deny)) == {"KEEP0001", "TEST0011", "TEST0021", "TEST0041"}
    again = run_script("trash_gate.py", "--zotero", tmp_path, "--denylist", deny)
    assert "0 trashed attachment(s) added, 0 restored one(s) removed" in again.stdout
    assert deny.read_text(encoding="utf-8").count("TEST0011") == 1


def test_cli_without_a_library_fails_loudly(tmp_path):
    r = run_script(
        "trash_gate.py", "--zotero", tmp_path, "--denylist", tmp_path / "d.txt"
    )
    assert r.returncode != 0 and "no zotero.sqlite" in r.stderr
