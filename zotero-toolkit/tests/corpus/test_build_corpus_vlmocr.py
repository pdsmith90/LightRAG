#!/usr/bin/env python3
"""Behavioural tests for build_corpus's VLM OCR route (2026-10-07).
Run:  .venv/bin/python test_build_corpus_vlmocr.py     (plain asserts; pytest also collects it)
Whole-document OCR (image-only PDFs, majority-OCR-layer scans) goes to the vision model first and falls back
to ocrmypdf when no server is usable or any page fails; the partial --pages route never touches the VLM;
the remote server is skipped inside its quiet window; --ocr-vlm off restores the old behaviour.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP calls and ocrmypdf replaced by fakes; all text is invented."""

import contextlib, io, os, subprocess, sys, tempfile, time, types

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import pymupdf
import build_corpus as bc

KEY = "0VLM1OCR"  # contains 0 and 1, which real Zotero keys never do
VLM_PAGE = (
    "# Loading of the crust\n\nInvented sentence about tidal loading of the crust and its viscous "
    "response, written for a test. "
) * 6
OCRMYPDF_LINE = "Fallback text written by the fake ocrmypdf for this invented scan. "


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


@contextlib.contextmanager
def fake_llm(pages):
    """pymupdf4llm.to_markdown replaced by a fake returning `pages` as page chunks."""
    fake = types.SimpleNamespace(
        to_markdown=lambda path, page_chunks=False, use_ocr=True, **kw: [
            {"text": t, "metadata": {"page_number": i + 1}} for i, t in enumerate(pages)
        ]
    )
    saved = sys.modules.get("pymupdf4llm")
    sys.modules["pymupdf4llm"] = fake
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["pymupdf4llm"]
        else:
            sys.modules["pymupdf4llm"] = saved


def image_only_pdf(path, pages=2):
    """Every page one full-page raster, no text layer -- a scan."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 170), False)
    pix.clear_with(200)
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()


def text_pdf(path, lines, pages=1):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        for i, line in enumerate(lines):
            page.insert_text((40, 80 + 16 * i), line, fontsize=7)
    doc.save(path)
    doc.close()


def layer_font(path):
    with pymupdf.open(path) as doc:
        for b in doc[0].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    return s["font"]


@contextlib.contextmanager
def vlm(kind="local", page_fn=None):
    """A usable VLM endpoint whose per-page call is `page_fn(base, png)`; records the calls."""
    calls = []

    def page(base, png):
        calls.append(len(png))
        return (page_fn or (lambda b, p: VLM_PAGE))(base, png)

    with (
        patched(bc, "_vlm_endpoint_cache", None),
        patched(bc, "_vlm_endpoint", lambda: (kind, "http://127.0.0.1:1")),
        patched(bc, "_vlm_ocr_page", page),
    ):
        yield calls


@contextlib.contextmanager
def no_vlm():
    with (
        patched(bc, "_vlm_endpoint_cache", None),
        patched(bc, "_vlm_endpoint", lambda: ("", "")),
    ):
        yield


@contextlib.contextmanager
def fake_ocrmypdf(available=True):
    """ocrmypdf present or absent; when run, it writes a text PDF to its output path."""
    calls = []

    def fake_run(args, **kw):
        calls.append(list(args))
        text_pdf(args[-1], [OCRMYPDF_LINE] * 20)

    with (
        patched(
            bc.shutil, "which", lambda name: "/usr/bin/ocrmypdf" if available else None
        ),
        patched(bc.subprocess, "run", fake_run),
    ):
        yield calls


def run_convert(td, src_builder, ocr="auto", ocr_vlm="auto"):
    zotero = os.path.join(td, "zotero")
    folder = os.path.join(zotero, "storage", KEY)
    os.makedirs(folder, exist_ok=True)
    src_builder(os.path.join(folder, "scan.pdf"))
    out = os.path.join(td, "rag_corpus")
    os.makedirs(out, exist_ok=True)
    bc.CFG.clear()
    bc.CFG.update(
        {
            "meta": {
                KEY: {
                    "filename": "scan.pdf",
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
            "ocr": ocr,
            "ocr_vlm": ocr_vlm,
            "force": True,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        key, method, path, n = bc.convert(KEY)
    body = open(path, encoding="utf-8").read() if path else ""
    return method, body, log.getvalue()


def test_image_only_scan_is_ocrd_by_the_vlm():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        vlm() as calls,
        fake_ocrmypdf(available=False),
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+vlmocr", method
    assert "tidal loading" in body
    assert len(calls) == 2, calls  # one call per page, in order
    assert "VLM-OCR scan.pdf: 2 page(s)" in log, log


def test_vlm_failure_falls_back_to_ocrmypdf():
    def boom(base, png):
        raise RuntimeError("finish_reason=length chars=0")

    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([[OCRMYPDF_LINE] * 20][0:0] or [OCRMYPDF_LINE * 20]),
        vlm(page_fn=boom),
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+ocr", method
    assert runs and runs[0][0] == "ocrmypdf", runs
    assert "VLM OCR abandoned on scan.pdf after 0 page(s)" in log, log
    assert bc._ocr_backend == "ocrmypdf"


def test_no_usable_server_means_the_old_route():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([OCRMYPDF_LINE * 20]),
        no_vlm(),
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+ocr" and runs, (method, runs)
    assert "VLM" not in log, log


def test_ocr_vlm_off_never_calls_the_vlm():
    def never(base, png):
        raise AssertionError("the VLM must not be called with --ocr-vlm off")

    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        vlm(page_fn=never),
        fake_ocrmypdf(available=False),
    ):
        method, body, log = run_convert(td, image_only_pdf, ocr_vlm="off")
    assert method == "pdf_needs_ocr", method


def test_partial_pages_never_use_the_vlm():
    def never(path):
        raise AssertionError(
            "whole-document VLM OCR must not run for a minority of pages"
        )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ten.pdf")
        text_pdf(
            path,
            ["A typeset line of the invented paper, repeated for size."] * 10,
            pages=10,
        )
        bc.CFG.clear()
        bc.CFG.update({"ocr": "auto", "ocr_vlm": "auto"})
        with patched(bc, "vlm_ocr_pdf_to_md", never), fake_ocrmypdf(available=False):
            assert bc.ocr_pdf_to_md(path, pages=[0]) == ""
        assert bc._ocr_backend == "ocrmypdf"
        # ... but a MAJORITY of pages is whole-document OCR and does use it
        with vlm() as calls, fake_ocrmypdf(available=False):
            md = bc.ocr_pdf_to_md(path, pages=list(range(6)))
        assert "tidal loading" in md and len(calls) == 10, (md[:40], calls)
        assert bc._ocr_backend == "vlm:local"


def test_quiet_window_skips_the_remote_server():
    inside = time.mktime((2026, 10, 7, 3, 0, 0, 0, 0, -1))
    outside = time.mktime((2026, 10, 7, 10, 0, 0, 0, 0, -1))
    with patched(bc, "OCR_VLM_REMOTE_QUIET", "01:30-08:45"):
        assert bc._vlm_quiet(inside) and not bc._vlm_quiet(outside)
    with patched(bc, "OCR_VLM_REMOTE_QUIET", "22:00-06:00"):
        assert bc._vlm_quiet(time.mktime((2026, 10, 7, 23, 30, 0, 0, 0, -1)))
        assert bc._vlm_quiet(time.mktime((2026, 10, 7, 5, 59, 0, 0, 0, -1)))
        assert not bc._vlm_quiet(time.mktime((2026, 10, 7, 6, 0, 0, 0, 0, -1)))
    alive = lambda base, timeout=3.0: (
        base == bc.OCR_VLM_REMOTE_URL
    )  # only the remote server answers
    with patched(bc, "_vlm_alive", alive), patched(bc, "OCR_VLM_LOCAL_START", ""):
        with (
            patched(bc, "_vlm_endpoint_cache", None),
            patched(bc, "_vlm_quiet", lambda now=None: True),
        ):
            assert bc._vlm_endpoint() == ("", "")
        with (
            patched(bc, "_vlm_endpoint_cache", None),
            patched(bc, "_vlm_quiet", lambda now=None: False),
        ):
            assert bc._vlm_endpoint() == ("remote", bc.OCR_VLM_REMOTE_URL)


def test_local_server_is_started_on_demand_for_a_build():
    started = []

    def fake_run(args, **kw):
        started.append(list(args))
        alive_now.append(True)

    alive_now = []
    alive = lambda base, timeout=3.0: base == bc.OCR_VLM_LOCAL_URL and bool(alive_now)
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "ocr_server.sh")
        open(script, "w").write("#!/bin/sh\n")
        os.chmod(script, 0o755)
        with (
            patched(bc, "_vlm_alive", alive),
            patched(bc, "OCR_VLM_LOCAL_START", script),
            patched(bc, "_vlm_quiet", lambda now=None: True),
            patched(bc.subprocess, "run", fake_run),
            patched(bc, "_vlm_endpoint_cache", None),
        ):
            assert bc._vlm_endpoint() == ("local", bc.OCR_VLM_LOCAL_URL)
            assert bc._vlm_endpoint() == (
                "local",
                bc.OCR_VLM_LOCAL_URL,
            )  # cached: started once
    assert started == [[script, "start", "--for-build"]], started


def test_majority_ocr_layer_scan_is_reocrd_by_the_vlm():
    left = [
        "The osculating elements of the satellite are estimated from pseudorange data."
    ] * 8
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "layer.pdf")
        text_pdf(path, left, pages=2)
        font = layer_font(path)
        bc.CFG.clear()
        bc.CFG.update({"ocr": "auto", "ocr_vlm": "auto"})
        pages = ["\n\n".join(left)] * 2
        with (
            fake_llm(pages),
            patched(bc, "OCR_LAYER_FONTS", {font}),
            vlm() as calls,
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            md = bc.pdf_to_md(path)
        assert "tidal loading" in md and len(calls) == 2, (md[:60], calls)
        assert bc._pdf_method == "pdf+vlmocr"
        assert "re-OCR'd by" in out.getvalue(), out.getvalue()
        # no server: the Tesseract layer is read in its own order, as before
        with (
            fake_llm(pages),
            patched(bc, "OCR_LAYER_FONTS", {font}),
            no_vlm(),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            md = bc.pdf_to_md(path)
        assert "osculating elements" in md and "tidal loading" not in md
        assert bc._pdf_method == "pdf+ocrlayer", bc._pdf_method


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
