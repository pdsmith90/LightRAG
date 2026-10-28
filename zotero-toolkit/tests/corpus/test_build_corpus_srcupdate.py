#!/usr/bin/env python3
"""Behavioural tests for build_corpus's handling of a changed source whose document was
already ingested (2026-10-27).
Run:  .venv/bin/python test_build_corpus_srcupdate.py     (plain asserts; pytest also collects it)
LightRAG never indexes a rebuild of a name it has processed, so such a rebuild is parked in
__changed__/ when it gains enough text (a header-only stub whose real source arrived later),
and otherwise the ingested copy is marked current. All text is invented."""

import contextlib
import io
import os
import sys
import tempfile
import time

try:
    import corpus_testlib  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as bc

KEY = "0SRC1UPD0"  # contains 0 and 1, which real Zotero keys never do
NAME = f"{KEY}__notes.md"
META = {
    KEY: {
        "filename": "notes.txt",
        "title": "Notes on a synthetic survey",
        "authors": ["Doe J"],
        "year": "1970",
        "tags": [],
        "abstract": "",
    }
}
PROSE = (
    "Paragraph of invented field notes about thermometer drift and its correction. "
    * 40
)


def setup(td, source_text, ingested_body=None, force=False):
    """A library with one text attachment; ingested_body (when given) is the body of the
    copy LightRAG already moved to __parsed__/, one hour older than the source."""
    zotero = os.path.join(td, "zotero")
    folder = os.path.join(zotero, "storage", KEY)
    os.makedirs(folder)
    src = os.path.join(folder, "notes.txt")
    with open(src, "w", encoding="utf-8") as f:
        f.write(source_text)
    out = os.path.join(td, "rag_corpus")
    os.makedirs(os.path.join(out, "__parsed__"))
    bc.CFG.clear()
    bc.CFG.update(
        {
            "meta": META,
            "zotero": zotero,
            "out": out,
            "deny": set(),
            "min_chars": 200,
            "ocr": "off",
            "force": force,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )
    now = time.time()
    os.utime(src, (now - 60, now - 60))
    if ingested_body is not None:
        parsed = os.path.join(out, "__parsed__", NAME)
        with open(parsed, "w", encoding="utf-8") as f:
            f.write(bc.header(META, KEY) + ingested_body + "\n")
        os.utime(parsed, (now - 3600, now - 3600))
    return src, out


def run_convert():
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        key, method, path, n = bc.convert(KEY)
    return method, path, log.getvalue()


def test_is_upgrade_boundaries():
    assert not bc.is_upgrade(0, bc.UPDATE_MIN_GAIN - 1)
    assert bc.is_upgrade(0, bc.UPDATE_MIN_GAIN)
    assert bc.is_upgrade(bc.HEADER_ONLY_MAX - 1, bc.UPDATE_MIN_GAIN)
    assert bc.is_upgrade(3000, 6000)
    assert not bc.is_upgrade(3000, 5999)  # under twice the ingested body
    assert not bc.is_upgrade(1000, 2999)  # three times, but 1,999 characters more
    assert bc.is_upgrade(1000, 3000)
    assert not bc.is_upgrade(6000, 100)  # a shorter rebuild never replaces a document


def test_body_chars_skips_the_metadata_header():
    doc = bc.header(META, KEY) + "body text\n"
    assert bc.body_chars(doc) == len("body text")
    assert bc.body_chars(bc.header(META, KEY)) == 0
    assert bc.body_chars("no header here\n---\nmore") == len(
        "no header here\n---\nmore"
    )


def test_numbered_copies_are_the_names_lightrag_uses():
    with tempfile.TemporaryDirectory() as td:
        for n in (NAME, f"{KEY}__notes_001.md", f"{KEY}__notes_1700000000.md"):
            open(os.path.join(td, n), "w").close()
        for n in (
            f"{KEY}__notes_recovered.md",
            f"{KEY}__notes_0001.md",
            f"{KEY}__notes_01.md",
        ):
            open(os.path.join(td, n), "w").close()
        got = [os.path.basename(p) for p in bc.numbered_copies(td, NAME)]
        assert got[0] == NAME and sorted(got[1:]) == [
            f"{KEY}__notes_001.md",
            f"{KEY}__notes_1700000000.md",
        ], got
        assert bc.numbered_copies(os.path.join(td, "absent"), NAME) == []


def test_a_stub_whose_full_text_arrived_is_parked_in_changed():
    with tempfile.TemporaryDirectory() as td:
        _, out = setup(td, PROSE, ingested_body="")
        method, path, log = run_convert()
        parked = os.path.join(out, bc.CHANGED_DIR, NAME)
        assert method == "source_update" and path == parked, (method, path)
        assert "invented field notes" in open(parked, encoding="utf-8").read()
        assert not os.path.exists(os.path.join(out, NAME))
        assert f"SOURCE-UPDATE {NAME}: 0 -> " in log, log
        # the parked copy counts as built: the next run converts nothing
        method, path, log = run_convert()
        assert method == "skip" and path == parked and log == "", (method, path, log)


def test_the_same_text_marks_the_ingested_copy_current():
    with tempfile.TemporaryDirectory() as td:
        src, out = setup(td, PROSE, ingested_body=PROSE.strip())
        method, path, log = run_convert()
        parsed = os.path.join(out, "__parsed__", NAME)
        assert method == "skip" and path == parsed, (method, path)
        assert f"SOURCE-TOUCHED {NAME}" in log, log
        assert os.path.getmtime(parsed) >= os.path.getmtime(src)
        assert not os.path.exists(os.path.join(out, NAME))
        assert not os.path.exists(os.path.join(out, bc.CHANGED_DIR))
        assert run_convert()[2] == ""  # nothing converted again


def test_a_shorter_rebuild_keeps_the_ingested_copy():
    with tempfile.TemporaryDirectory() as td:
        _, out = setup(td, "Only the abstract survived. " * 5, ingested_body=PROSE * 2)
        method, path, log = run_convert()
        assert method == "skip" and f"SOURCE-SHRANK {NAME}" in log, (method, log)
        assert not os.path.exists(os.path.join(out, bc.CHANGED_DIR))
        assert not os.path.exists(os.path.join(out, NAME))


def test_a_newer_numbered_copy_counts_as_built():
    with tempfile.TemporaryDirectory() as td:
        src, out = setup(td, PROSE, ingested_body="")
        copy = os.path.join(out, "__parsed__", f"{KEY}__notes_001.md")
        with open(copy, "w", encoding="utf-8") as f:
            f.write(bc.header(META, KEY) + PROSE)
        method, path, log = run_convert()
        assert method == "skip" and path == copy and log == "", (method, path, log)


def test_a_document_never_ingested_is_built_at_the_root_as_before():
    with tempfile.TemporaryDirectory() as td:
        _, out = setup(td, PROSE)
        method, path, log = run_convert()
        assert method == "txt" and path == os.path.join(out, NAME), (method, path)
        assert "SOURCE-" not in log


def test_force_rebuilds_at_the_root_with_a_warning():
    with tempfile.TemporaryDirectory() as td:
        _, out = setup(td, PROSE, ingested_body="", force=True)
        method, path, log = run_convert()
        assert method == "txt" and path == os.path.join(out, NAME), (method, path)
        assert f"REBUILT {NAME}" in log, log
        assert not os.path.exists(os.path.join(out, bc.CHANGED_DIR))


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
