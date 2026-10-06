#!/usr/bin/env python3
"""Behavioural tests for build_corpus's OCR-layer handling (2026-10-06).
Run:  .venv/bin/python test_build_corpus_ocrlayer.py     (plain asserts; pytest also collects it)
A page whose text layer was written by Tesseract (ocrmypdf) is read from the layer's own blocks, in
Tesseract's reading order, instead of pymupdf4llm's markdown -- which merges the two columns of such a
page line by line -- and convert() OCRs afresh before reusing a pre-OCR'd text cache.
Synthetic PDFs via pymupdf; pymupdf4llm and ocrmypdf replaced by fakes; all text is invented."""
import contextlib, io, os, sys, tempfile, types

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bundle layout: beside build_corpus.py
import pymupdf
import build_corpus as bc

LEFT = [
    "The osculating elements of the satellite are estimated from pseudorange data.",
    "Carrier-phase observations refine the along-track position of the spacecraft.",
    "A Kalman filter propagates the state between the tracking passes of each day.",
    "Residuals below two centimetres indicate a converged orbit solution overall.",
]
RIGHT = [
    "Thermal noise in the receiver front end limits the ranging precision reached.",
    "Multipath near the ground antenna adds a slowly varying bias to the ranges.",
    "Tropospheric delay is modelled with a mapping function of the elevation angle.",
    "Ionospheric refraction cancels in the dual-frequency linear combination used.",
]
INTERLEAVED = "\n\n".join(f"{a} {b}" for a, b in zip(LEFT, RIGHT))   # what pymupdf4llm makes of an OCR layer
SOUP = "\n".join("c ? c o o c o c c c o c o o c o c o" for _ in range(40))
KEY = "0OCR1LAYER"   # contains 0 and 1, which real Zotero keys never do


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


def two_column_pdf(path, left=LEFT, right=RIGHT, extra_pages=()):
    """Page 1: a two-column page written the way Tesseract's layer is -- one text object per
    line, the whole left column first. Further pages: one text block each."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(left):
        page.insert_text((40, 80 + 16 * i), line, fontsize=7)
    for i, line in enumerate(right):
        page.insert_text((320, 80 + 16 * i), line, fontsize=7)
    for text in extra_pages:
        doc.new_page(width=595, height=842).insert_text((40, 80), text, fontsize=7)
    doc.save(path)
    doc.close()


def layer_font(path):
    with pymupdf.open(path) as doc:
        for b in doc[0].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                for s in ln["spans"]:
                    return s["font"]


def run_pdf_to_md(path, pages, ocr_font):
    out = io.StringIO()
    with fake_llm(pages), patched(bc, "OCR_LAYER_FONTS", {ocr_font} if ocr_font else set()), \
         contextlib.redirect_stdout(out):
        md = bc.pdf_to_md(path)
    return md, out.getvalue()


def test_ocr_layer_page_is_told_by_its_font():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "scan.pdf")
        two_column_pdf(p)
        font = layer_font(p)
        with pymupdf.open(p) as doc:
            assert not bc._ocr_layer_page(doc[0])                      # a typeset page
            with patched(bc, "OCR_LAYER_FONTS", {font}):
                assert bc._ocr_layer_page(doc[0])                      # the same page, were its font Tesseract's
            assert not bc._ocr_layer_page(doc.new_page())              # no text at all


def test_ocr_layer_page_keeps_tesseract_column_order():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "scan.pdf")
        two_column_pdf(p)
        md, log = run_pdf_to_md(p, [INTERLEAVED], layer_font(p))
        pos = [md.index(line) for line in LEFT + RIGHT]                # every line present ...
        assert max(pos[: len(LEFT)]) < min(pos[len(LEFT):]), md        # ... the whole left column first
        assert not any(f"{a} {b}" in md for a, b in zip(LEFT, RIGHT))  # the merged lines are gone
        assert "OCR-LAYER scan.pdf: 1/1 page(s)" in log and "(0 dropped as letter soup)" in log, log
        assert bc._pdf_method == "pdf+ocrlayer" and bc._pdf_garbled_pages == []


def test_typeset_page_still_takes_pymupdf4llm_output():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "paper.pdf")
        two_column_pdf(p)
        md, log = run_pdf_to_md(p, [INTERLEAVED], None)
        assert md == bc.clean_md(INTERLEAVED) and "OCR-LAYER" not in log and bc._pdf_method == ""


def test_ocr_layer_scribble_is_dropped_and_not_sent_back_to_ocr():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "scan.pdf")
        two_column_pdf(p, extra_pages=[SOUP])                          # page 2: figure scribble in the OCR font
        md, log = run_pdf_to_md(p, [INTERLEAVED, SOUP], layer_font(p))
        assert all(line in md for line in LEFT + RIGHT) and "c ? c o o c" not in md
        assert "OCR-LAYER scan.pdf: 2/2 page(s)" in log and "(1 dropped as letter soup)" in log, log
        assert bc._pdf_garbled_pages == [] and "GARBLED pages" not in log


# ---------------------------------------------------------------- convert(): fresh OCR before the cache
def run_convert(td, fake_pdf_to_md, fake_ocr, ocr="auto"):
    zotero = os.path.join(td, "zotero")
    folder = os.path.join(zotero, "storage", KEY)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "scan.pdf"), "wb") as f:
        f.write(b"%PDF-1.4 fake")
    cache = os.path.join(zotero, "ocr_extracted")
    os.makedirs(cache, exist_ok=True)
    with open(os.path.join(cache, "scan.pdf.txt"), "w", encoding="utf-8") as f:
        f.write("Cached layout text of the scan.   Second column text beside it.\n" * 30)
    out = os.path.join(td, "rag_corpus")
    os.makedirs(out, exist_ok=True)
    bc.CFG.clear()
    bc.CFG.update({
        "meta": {KEY: {"filename": "scan.pdf", "title": "A scanned paper", "authors": ["Doe J"],
                       "year": "1968", "tags": [], "abstract": ""}},
        "zotero": zotero, "out": out, "deny": set(), "min_chars": 200, "ocr": ocr, "force": True,
        "held": None, "ocr_cache": cache, "fallback_md": None,
    })
    calls = []

    def ocr_fn(path, pages=None):
        calls.append(path)
        return fake_ocr

    log = io.StringIO()
    with patched(bc, "pdf_to_md", fake_pdf_to_md), patched(bc, "ocr_pdf_to_md", ocr_fn), \
         contextlib.redirect_stdout(log):
        key, method, path, n = bc.convert(KEY)
    body = open(path, encoding="utf-8").read() if path else ""
    return method, body, calls


FRESH = "Fresh OCR text of the scan, read in reading order. " * 30


def test_convert_prefers_fresh_ocr_over_the_layout_cache():
    with tempfile.TemporaryDirectory() as td:
        method, body, calls = run_convert(td, lambda path: "", FRESH)
    assert method == "pdf+ocr" and "Fresh OCR text" in body and "Cached layout" not in body and calls


def test_convert_falls_back_to_the_cache_when_ocr_fails_or_is_off():
    with tempfile.TemporaryDirectory() as td:
        method, body, calls = run_convert(td, lambda path: "", "")
    assert method == "pdf+ocr_cache" and "Cached layout" in body and calls
    with tempfile.TemporaryDirectory() as td:
        method, body, calls = run_convert(td, lambda path: "", FRESH, ocr="off")
    assert method == "pdf+ocr_cache" and "Cached layout" in body and not calls


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
