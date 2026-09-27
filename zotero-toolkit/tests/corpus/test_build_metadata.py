"""Tests for corpus/build_metadata.py against a synthetic zotero.sqlite."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_testlib import make_zotero_db, run_script
import build_metadata

LONG_ABSTRACT = "Deep networks learn representations. " * 60  # ~2 KB, over the 1500 cap

LIBRARY = [
    {
        "key": "TEST0001",
        "fields": {
            "title": "Deep learning",
            "date": "2015-05-00 May 2015",
            "DOI": "10.1038/nature14539",
            "publicationTitle": "Nature",
            "abstractNote": LONG_ABSTRACT,
        },
        "creators": [("LeCun", "Yann"), ("Bengio", "Yoshua"), ("Hinton", "Geoffrey")],
        "tags": ["neural networks", "review"],
    },
    {
        "key": "TEST0101",
        "parent": "TEST0001",
        "path": "storage:LeCun 2015 deep learning.pdf",
    },
    {
        "key": "TEST0002",
        "fields": {
            "title": "A chapter",
            "date": "2003",
            "bookTitle": "Handbook of Statistics",
            "conferenceName": "not preferred",
        },
        "creators": [("WHO", "")],
    },
    {"key": "TEST0102", "parent": "TEST0002", "path": "storage:chapter.epub"},
    # standalone snapshot: metadata comes from the attachment itself
    {
        "key": "TEST0103",
        "path": "storage:snapshot.html",
        "fields": {"title": "Saved page"},
    },
    # standalone attachment with no title: falls back to the file stem
    {"key": "TEST0104", "path": "storage:untitled notes.txt"},
    # linked URL: no path at all
    {"key": "TEST0105", "path": None},
]


def test_build_maps_every_attachment_with_parent_or_own_metadata(tmp_path):
    make_zotero_db(tmp_path, LIBRARY)
    idx = build_metadata.build(str(tmp_path))

    assert set(idx) == {
        "TEST0101",
        "TEST0102",
        "TEST0103",
        "TEST0104",
        "TEST0105",
    }  # parents are not keys
    pdf = idx["TEST0101"]
    assert pdf == {
        "filename": "LeCun 2015 deep learning.pdf",
        "title": "Deep learning",
        "authors": ["LeCun Yann", "Bengio Yoshua", "Hinton Geoffrey"],
        "year": "2015",
        "doi": "10.1038/nature14539",
        "publication": "Nature",
        "tags": ["neural networks", "review"],
        "abstract": LONG_ABSTRACT[:1500],
        "source": "parent",
    }
    epub = idx["TEST0102"]  # every content type, not just PDF
    assert (
        epub["publication"] == "Handbook of Statistics"
    )  # bookTitle outranks conferenceName
    assert (
        epub["authors"] == ["WHO"]
        and epub["year"] == "2003"
        and epub["source"] == "parent"
    )
    assert (
        idx["TEST0103"]["title"] == "Saved page"
        and idx["TEST0103"]["source"] == "standalone"
    )
    assert (
        idx["TEST0104"]["title"] == "untitled notes" and idx["TEST0104"]["year"] == ""
    )
    assert idx["TEST0105"]["filename"] == "" and idx["TEST0105"]["authors"] == []


def test_missing_database_raises_and_creates_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_metadata.build(str(tmp_path))
    assert not os.path.exists(tmp_path / "zotero.sqlite")


def test_rebuild_carries_forward_keys_zotero_no_longer_has(tmp_path):
    zdir, out = tmp_path / "zotero", tmp_path / "meta.json"
    make_zotero_db(zdir, LIBRARY)
    r = run_script("build_metadata.py", "--zotero", zdir, "--out", out)
    assert r.returncode == 0, r.stderr
    first = json.loads(out.read_text(encoding="utf-8"))
    assert "orphan" not in first["TEST0101"]

    # the PDF attachment is deleted in Zotero; its parent stays
    make_zotero_db(zdir, [it for it in LIBRARY if it["key"] != "TEST0101"])
    for _ in range(2):  # sticky: a second rebuild keeps it too
        r = run_script("build_metadata.py", "--zotero", zdir, "--out", out)
        assert r.returncode == 0, r.stderr
        assert "1 carried orphans" in r.stdout
        again = json.loads(out.read_text(encoding="utf-8"))
        assert again["TEST0101"] == {**first["TEST0101"], "orphan": True}
        assert "orphan" not in again["TEST0102"]
    assert not os.path.exists(str(out) + ".tmp")

    # the key comes back in Zotero: fresh entry, flag dropped
    make_zotero_db(zdir, LIBRARY)
    run_script("build_metadata.py", "--zotero", zdir, "--out", out)
    assert "orphan" not in json.loads(out.read_text(encoding="utf-8"))["TEST0101"]


def test_unreadable_previous_output_is_replaced_not_fatal(tmp_path):
    zdir, out = tmp_path / "zotero", tmp_path / "meta.json"
    make_zotero_db(zdir, LIBRARY)
    out.write_text("{not json", encoding="utf-8")
    r = run_script("build_metadata.py", "--zotero", zdir, "--out", out)
    assert r.returncode == 0 and "WARN: previous" in r.stdout
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 5


def test_default_output_path(tmp_path):
    zdir = tmp_path / "zotero"
    make_zotero_db(zdir, LIBRARY)
    r = run_script("build_metadata.py", "--zotero", zdir)
    assert r.returncode == 0, r.stderr
    assert (zdir / "zotero_metadata.json").is_file()  # <zotero>/zotero_metadata.json

    env_out = tmp_path / "elsewhere.json"
    r = run_script(
        "build_metadata.py",
        "--zotero",
        zdir,
        env_extra={"ZOTERO_METADATA": str(env_out)},
    )
    assert r.returncode == 0, r.stderr
    assert env_out.is_file()  # $ZOTERO_METADATA wins
