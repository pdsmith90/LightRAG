#!/usr/bin/env python3
"""Behavioural tests for build_corpus's not-a-PDF check (2026-10-06).
Run:  .venv/bin/python test_build_corpus_htmlpdf.py     (plain asserts; pytest also collects it)
A publisher landing page saved under the article's name as .pdf, or an empty .pdf, writes no document
and is named in the log; a real PDF goes on as before. All text is invented."""

import contextlib
import io
import os
import sys
import tempfile

try:
    import corpus_testlib  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_corpus as bc
from test_build_corpus_ocrlayer import patched

KEY = "0HTML1PDF0"  # contains 0 and 1, which real Zotero keys never do
HTML = b"<!DOCTYPE html>\n<html><head><title>A scanned paper</title></head><body>Please sign in.</body></html>\n"


def convert_with(td, content):
    zotero = os.path.join(td, "zotero")
    folder = os.path.join(zotero, "storage", KEY)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "paper.pdf"), "wb") as f:
        f.write(content)
    out = os.path.join(td, "rag_corpus")
    os.makedirs(out, exist_ok=True)
    bc.CFG.clear()
    bc.CFG.update(
        {
            "meta": {
                KEY: {
                    "filename": "paper.pdf",
                    "title": "A scanned paper",
                    "authors": ["Doe J"],
                    "year": "1968",
                    "tags": [],
                    "abstract": "",
                }
            },
            "zotero": zotero,
            "out": out,
            "deny": set(),
            "min_chars": 200,
            "ocr": "auto",
            "force": True,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )
    calls = []
    prose = "Real prose from the engine, long enough to count as a document. " * 10
    log = io.StringIO()
    with (
        patched(bc, "pdf_to_md", lambda path: (calls.append(("pdf", path)), prose)[1]),
        patched(
            bc,
            "ocr_pdf_to_md",
            lambda path, pages=None: (calls.append(("ocr", path)), "")[1],
        ),
        contextlib.redirect_stdout(log),
    ):
        key, method, path, n = bc.convert(KEY)
    return method, path, calls, log.getvalue()


def test_not_a_pdf_reasons():
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, "page.pdf")
        open(h, "wb").write(HTML)
        e = os.path.join(td, "empty.pdf")
        open(e, "wb").close()
        r = os.path.join(td, "real.pdf")
        open(r, "wb").write(b"%PDF-1.4\n%fake\n")
        assert (
            bc._not_a_pdf(h) == "HTML/text content"
            and bc._not_a_pdf(e) == "0 bytes"
            and bc._not_a_pdf(r) == ""
        )


def test_html_named_pdf_writes_nothing_and_skips_ocr():
    with tempfile.TemporaryDirectory() as td:
        method, path, calls, log = convert_with(td, HTML)
    assert method == "not_a_pdf" and path == "" and calls == [], (method, path, calls)
    assert f"NOT A PDF {KEY}: paper.pdf (HTML/text content)" in log, log


def test_empty_pdf_writes_nothing():
    with tempfile.TemporaryDirectory() as td:
        method, path, calls, log = convert_with(td, b"")
    assert method == "not_a_pdf" and path == "" and calls == [] and "(0 bytes)" in log


def test_real_pdf_goes_on_as_before():
    with tempfile.TemporaryDirectory() as td:
        method, path, calls, log = convert_with(td, b"%PDF-1.4 fake")
        assert (
            method == "pdf"
            and path.endswith(".md")
            and os.path.exists(path)
            and calls[0][0] == "pdf"
        )
    assert "NOT A PDF" not in log


if __name__ == "__main__":
    tests = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
