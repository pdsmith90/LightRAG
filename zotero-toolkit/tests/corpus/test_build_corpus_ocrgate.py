#!/usr/bin/env python3
"""Behavioural tests for the floor on whole-document ocrmypdf results (2026-10-17).
Run:  .venv/bin/python test_build_corpus_ocrgate.py     (plain asserts; pytest also collects it)
An ocrmypdf result under OCR_MIN_CHARS_PER_PAGE non-space characters per page is rejected: nothing is written
for an image-only scan (a header-only file would count as done for good), and a majority-garbled PDF keeps its
readable pages instead of the empty OCR text. A --pages run is not measured.
Synthetic PDFs via pymupdf; pymupdf4llm and ocrmypdf replaced by fakes; all text is invented."""

import contextlib
import io
import os
import sys
import tempfile

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc
from test_build_corpus_vlmocr import (
    OCRMYPDF_LINE,
    fake_llm,
    fake_ocrmypdf,
    image_only_pdf,
    no_vlm,
    patched,
    run_convert,
)


@contextlib.contextmanager
def blank_ocrmypdf():
    """ocrmypdf "succeeds" but its output carries no text -- what dropped page images produce."""
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        image_only_pdf(args[-1])

    with (
        patched(bc.shutil, "which", lambda name: "/usr/bin/ocrmypdf"),
        patched(bc.subprocess, "run", fake_run),
    ):
        yield calls


@contextlib.contextmanager
def own_cfg():
    """A private CFG for one test. run_convert() fills the module's shared CFG; leaving ocr_vlm == "auto" behind
    sends a later test module's synthetic PDFs to whatever VLM server answers (it reached a live one on 2026-10-17)."""
    with patched(bc, "CFG", {}):
        yield


def written(td):
    return os.listdir(os.path.join(td, "rag_corpus"))


def test_floor_arithmetic():
    floor = bc.OCR_MIN_CHARS_PER_PAGE
    assert bc.ocr_too_thin("x" * (2 * floor), 2) == ""
    assert bc.ocr_too_thin("x " * (2 * floor - 1), 2).startswith(
        f"{2 * floor - 1} non-space chars over 2 page(s)"
    )
    assert bc.ocr_too_thin("", 0) == ""  # page count unknown: no verdict


def test_blank_ocr_of_a_scan_writes_nothing():
    for pages, chars in (
        (["ii", ""], 2),
        (["", ""], 0),
    ):  # nearly empty, and wholly empty
        with (
            tempfile.TemporaryDirectory() as td,
            own_cfg(),
            fake_llm(pages),
            no_vlm(),
            blank_ocrmypdf() as runs,
        ):
            method, body, log = run_convert(td, image_only_pdf)
            assert method == "ocr_empty" and runs, (pages, method, runs)
            assert not written(td), (pages, written(td))  # retried by the next build
            assert (
                f"OCR-EMPTY scan.pdf: ocrmypdf gave {chars} non-space chars over 2 page(s)"
                in log
            ), (pages, log)
        assert bc._ocr_backend == "ocrmypdf:empty"


def test_real_ocr_text_is_still_written():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([OCRMYPDF_LINE * 20]),
        no_vlm(),
        fake_ocrmypdf(available=True),
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+ocr" and OCRMYPDF_LINE.strip() in body, (method, body[:80])
    assert "OCR-EMPTY" not in log, log


def test_thin_ocr_keeps_the_readable_pages_of_a_garbled_pdf():
    readable = (
        "A readable typeset paragraph of the invented paper about crustal loading. " * 6
    )

    def fake_pdf_to_md(path, is_ocr=False):
        if is_ocr:
            return "ii"  # ocrmypdf's blank rewrite
        bc._pdf_garbled_pages = [1, 2]  # 2 of 3 pages garbled: whole-document OCR
        bc._pdf_page_count = 3
        return readable

    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        patched(bc, "pdf_to_md", fake_pdf_to_md),
        no_vlm(),
        blank_ocrmypdf() as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=3))
    assert runs and "--pages" not in runs[0], runs
    assert method == "pdf_garbled_pages_dropped" and "crustal loading" in body, (
        method,
        body[:80],
    )
    assert (
        "OCR-EMPTY scan.pdf: ocrmypdf gave 2 non-space chars over 3 page(s)" in log
    ), log


def test_partial_ocr_is_not_measured():
    """--pages runs leave the other pages' own text in ocrmypdf's output, so its density says nothing."""
    with tempfile.TemporaryDirectory() as td, own_cfg():
        path = os.path.join(td, "ten.pdf")
        image_only_pdf(path, pages=10)
        bc.CFG.clear()
        bc.CFG.update({"ocr": "auto", "ocr_vlm": "auto"})
        with (
            patched(bc, "pdf_to_md", lambda p, is_ocr=False: "ii"),
            blank_ocrmypdf() as runs,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            assert bc.ocr_pdf_to_md(path, pages=[0]) == "ii"
        assert runs and "--pages" in runs[0] and "OCR-EMPTY" not in out.getvalue(), (
            runs,
            out.getvalue(),
        )


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
