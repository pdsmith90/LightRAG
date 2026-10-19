#!/usr/bin/env python3
"""Behavioural tests for the per-page Tesseract fallback inside VLM OCR (2026-10-19).
Run:  .venv/bin/python test_build_corpus_vlmpage.py     (plain asserts; pytest also collects it)
A page the vision model answers but cannot transcribe -- it runs out of tokens in a repetition loop, or returns
nothing -- is read by Tesseract from the same MuPDF render, and the VLM goes on with the next page. On 2026-10-19
one such page each (Gill p. 18, blank but for a sideways watermark; Efron p. 93, a table) sent two whole books to
ocrmypdf. More such pages than OCR_VLM_MAX_UNREADABLE_SHARE of the document (and than 3), a Tesseract failure on
such a page, or any other error still abandons the VLM as before.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP calls, Tesseract and ocrmypdf replaced by fakes (one test runs
the real Tesseract when it is installed); all text is invented."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import pymupdf
import build_corpus as bc
from test_build_corpus_vlmocr import (
    OCRMYPDF_LINE,
    VLM_PAGE,
    fake_llm,
    fake_ocrmypdf,
    image_only_pdf,
    patched,
    run_convert,
    vlm,
)

# Tesseract reads every page since the grounding check (2026-10-19): its fake reading contains the VLM's words, so
# the pages the VLM transcribes stay; a page the VLM cannot transcribe shows the extra sentence.
TESS_PAGE = (
    VLM_PAGE
    + "Invented Tesseract reading of a page the model looped on, about mantle viscosity. "
)


@contextlib.contextmanager
def own_cfg():
    """A private CFG for one test (see test_build_corpus_ocrgate.own_cfg: a shared one reached a live server)."""
    with patched(bc, "CFG", {}):
        yield


@contextlib.contextmanager
def tesseract(text=TESS_PAGE, fail=None):
    """bc._tesseract_png replaced by a fake that records the image sizes and returns `text` -- or text[i] on the
    i-th call when it is a list, the last repeating -- or raises `fail`."""
    calls = []

    def fake(png):
        calls.append(len(png))
        if fail:
            raise fail
        return text[min(len(calls), len(text)) - 1] if isinstance(text, list) else text

    with patched(bc, "_tesseract_png", fake):
        yield calls


def unreadable_on(*pages, answer=VLM_PAGE):
    """A VLM page function that cannot transcribe the given 1-based pages."""
    seen = []

    def page_fn(base, png):
        seen.append(1)
        if len(seen) in pages:
            raise bc.VlmPageUnreadable("finish_reason=length chars=6000")
        return answer

    return page_fn


def test_unreadable_page_is_read_by_tesseract():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(kind="remote", page_fn=unreadable_on(2)) as calls,
        tesseract() as tess,
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=3))
    assert method == "pdf+vlmocr", (method, log)
    assert "tidal loading" in body and "mantle viscosity" in body, body[:200]
    assert len(calls) == 3 and len(tess) == 3 and not runs, (
        calls,
        tess,
        runs,
    )  # the VLM went on; no ocrmypdf
    assert (
        "VLM-OCR scan.pdf page 2: finish_reason=length chars=6000 -> tesseract" in log
    ), log
    assert (
        "VLM-OCR scan.pdf: 3 page(s) via pepperocr-vl (remote)" in log
        and "1 page(s) by tesseract (2)" in log
    ), log
    assert "abandoned" not in log, log


def test_tesseract_reads_a_sharper_render_than_the_vlm():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(page_fn=unreadable_on(1)) as calls,
        tesseract() as tess,
        fake_ocrmypdf(available=False),
    ):
        run_convert(td, image_only_pdf)
    assert tess[0] > calls[0], (
        tess,
        calls,
    )  # OCR_VLM_TESSERACT_DPI 300 vs OCR_VLM_DPI 130


def test_blank_page_adds_nothing():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(page_fn=unreadable_on(1)),
        tesseract(text=["", TESS_PAGE]) as tess,
        fake_ocrmypdf(available=False),
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+vlmocr" and len(tess) == 2, (method, tess)
    assert "page 1: finish_reason=length chars=6000 -> tesseract (0 chars)" in log, log
    assert "tidal loading" in body


def test_too_many_unreadable_pages_abandon_the_vlm():
    """A model that loops on page after page is not reading this document: ocrmypdf, as before."""
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([OCRMYPDF_LINE * 20]),
        vlm(page_fn=unreadable_on(*range(1, 11))) as calls,
        tesseract() as tess,
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=10))
    assert method == "pdf+ocr" and runs, (method, runs)
    assert len(calls) == 4 and len(tess) == 4, (
        calls,
        tess,
    )  # max(3, 5% of 10): the 4th unreadable page gives up
    assert (
        "VLM OCR abandoned on scan.pdf after 3 page(s)" in log
        and "4 unreadable page(s)" in log
    ), log
    assert "tidal loading" not in body and "mantle viscosity" not in body, body[
        :200
    ]  # never a mixed abandon


def test_tesseract_failure_abandons_the_vlm():
    for err in (
        FileNotFoundError(2, "No such file or directory: 'tesseract'"),
        bc.subprocess.CalledProcessError(1, ["tesseract"]),
    ):
        with (
            tempfile.TemporaryDirectory() as td,
            own_cfg(),
            fake_llm([OCRMYPDF_LINE * 20]),
            vlm(page_fn=unreadable_on(2)),
            tesseract(fail=err),
            fake_ocrmypdf(available=True) as runs,
        ):
            method, body, log = run_convert(td, image_only_pdf)
        assert method == "pdf+ocr" and runs, (err, method, runs)
        assert "VLM OCR abandoned on scan.pdf after 1 page(s)" in log, (err, log)


def test_lane_down_after_a_blank_first_page_is_not_a_refusal():
    """The lane served page 1 (the model looped on it), so a 5xx on page 2 is a failure mid-document: ocrmypdf."""
    seen = []

    def page_fn(base, png):
        seen.append(1)
        if len(seen) == 1:
            raise bc.VlmPageUnreadable("finish_reason=length chars=6000")
        raise urllib.error.HTTPError(
            "http://127.0.0.1:1/v1/chat/completions",
            500,
            "Internal Server Error",
            {},
            None,
        )

    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([OCRMYPDF_LINE * 20]),
        vlm(kind="remote", page_fn=page_fn),
        tesseract(text=""),
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+ocr" and runs, (method, runs)
    assert "abandoned on scan.pdf after 1 page(s)" in log, log


def test_page_call_classifies_unreadable_answers():
    def answer(
        finish, content
    ):  # the streamed answer (2026-10-19): content, then the finish chunk
        chunks = (
            {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": finish}]},
        )
        body = (
            b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
            + b"data: [DONE]\n\n"
        )
        return lambda req, timeout=None: contextlib.closing(io.BytesIO(body))

    with patched(urllib.request, "urlopen", answer("stop", "## A page\n\nText.")):
        assert (
            bc._vlm_ocr_page("http://127.0.0.1:1", b"\x89PNG") == "## A page\n\nText."
        )
    for finish, content in (("length", "loop " * 50), ("stop", "   "), (None, "")):
        with patched(urllib.request, "urlopen", answer(finish, content)):
            try:
                bc._vlm_ocr_page("http://127.0.0.1:1", b"\x89PNG")
                raise AssertionError((finish, content))
            except bc.VlmPageUnreadable as e:
                assert isinstance(e, RuntimeError) and f"finish_reason={finish}" in str(
                    e
                ), e


def test_real_tesseract_reads_a_rendered_page():
    if not shutil.which("tesseract"):
        print("skip: no tesseract")
        return
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 120), "Bootstrap standard errors", fontsize=24)
    png = page.get_pixmap(dpi=bc.OCR_VLM_TESSERACT_DPI).tobytes("png")
    doc.close()
    assert "Bootstrap standard errors" in bc._tesseract_png(png)


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
