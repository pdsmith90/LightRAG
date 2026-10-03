"""build_corpus repairs pages pymupdf4llm drops from the PDF's own text layer.

Synthetic PDFs are built with pymupdf; pymupdf4llm is replaced by a fake, so the tests
exercise the decision (per-page screen -> repair -> method label), not the layout engine.
"""

import os
import sys
import types

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_testlib  # noqa: F401  (puts corpus/ on sys.path)
import build_corpus as bc

LINE = (
    "The scheduler assigns each queued job to the worker with the shortest expected "
    "completion time."
)
PAGE_TEXT = "\n".join(LINE for _ in range(16))  # ~1,400 non-space characters per page
WATERMARK = "Downloaded from an institutional subscription on 01 January 2000"
KEY = "TEST0002"


def make_pdf(path, pages=3, text=True, text_first=True):
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 60), False)
    pix.clear_with(220)
    for _ in range(pages):
        page = doc.new_page()
        if text and text_first:
            page.insert_text((60, 100), PAGE_TEXT, fontsize=9, render_mode=3)
        page.insert_image(page.rect, pixmap=pix)  # the full-page scan image
        if text and not text_first:
            page.insert_text((60, 100), PAGE_TEXT, fontsize=9, render_mode=3)
        page.insert_text((20, 700), WATERMARK, fontsize=6)  # the visible watermark
    doc.save(path)
    doc.close()


class FakeLLM:
    """Stands in for pymupdf4llm: one dict per page for page_chunks=True."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def to_markdown(self, path, page_chunks=False, use_ocr=True, **kw):
        assert use_ocr is False, "the engine's own OCR must stay off"
        self.calls.append(page_chunks)
        if page_chunks:
            return [
                {"text": t, "metadata": {"page_number": i + 1}}
                for i, t in enumerate(self.pages)
            ]
        return "".join(self.pages)


def convert_with(monkeypatch, fake, path):
    monkeypatch.setitem(
        sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=fake.to_markdown)
    )
    return bc.pdf_to_md(path)


def test_emptied_pages_are_repaired_from_the_text_layer(tmp_path, monkeypatch):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, pages=3)
    fake = FakeLLM([WATERMARK + "\n"] * 3)  # what the engine returns for such scans
    md = convert_with(monkeypatch, fake, p)
    assert bc._pdf_method == "pdf+textlayer"
    assert fake.calls == [True]
    assert md.count("shortest expected") >= 20
    assert "\n\n\n" not in md and md.count(LINE) >= 3  # merged into paragraphs


def test_partly_emptied_document_keeps_the_good_pages(tmp_path, monkeypatch):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, pages=4)
    good = "# Heading from the layout engine\n\n" + PAGE_TEXT.replace("\n", " ")
    fake = FakeLLM([good, WATERMARK, good, WATERMARK])
    md = convert_with(monkeypatch, fake, p)
    assert bc._pdf_method == "pdf+textlayer"
    assert md.count("# Heading from the layout engine") == 2  # kept verbatim
    assert md.count("shortest expected") >= 28


def test_good_document_is_untouched(tmp_path, monkeypatch):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, pages=3, text_first=False)
    good = "## Section\n\n" + PAGE_TEXT.replace("\n", " ")
    fake = FakeLLM([good] * 3)
    md = convert_with(monkeypatch, fake, p)
    assert bc._pdf_method == "" and fake.calls == [True]
    assert md == bc.clean_md(
        "".join([good] * 3)
    )  # chunks joined with "" == plain output


def test_image_only_scan_takes_the_ocr_route(tmp_path, monkeypatch):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, pages=3, text=False)
    fake = FakeLLM([WATERMARK] * 3)
    md = convert_with(monkeypatch, fake, p)
    assert md == "" and fake.calls == []  # convert() then runs ocrmypdf
    assert bc._pdf_method == ""


def test_convert_labels_the_method(tmp_path, monkeypatch):
    folder = tmp_path / "storage" / KEY
    folder.mkdir(parents=True)
    (folder / "scan.pdf").write_bytes(b"%PDF-1.4 fake")
    out = tmp_path / "rag_corpus"
    out.mkdir()
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
            "zotero": str(tmp_path),
            "out": str(out),
            "deny": set(),
            "min_chars": 200,
            "ocr": "off",
            "force": False,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )

    def fake_pdf_to_md(path):
        bc._pdf_method = "pdf+textlayer"
        return PAGE_TEXT

    monkeypatch.setattr(bc, "pdf_to_md", fake_pdf_to_md)
    key, method, path, n = bc.convert(KEY)
    assert method == "pdf+textlayer"
    assert "shortest expected" in open(path, encoding="utf-8").read()


def test_paragraph_merge():
    doc = pymupdf.open()
    page = doc.new_page()
    y = 80
    for line in (
        "Immediately after the event a continuing pro-",
        "gram of displacement measurements was undertaken.",
        "Several instruments were installed along the trace;",
        "the rate decreased logarithmically.",
    ):
        page.insert_text((60, y), line, fontsize=9, render_mode=3)
        y += 14
    md = bc._textlayer_page_md(page)
    assert "program of displacement" in md  # hyphen at a line break removed
    assert md.count("\n\n") == 1  # one paragraph break: after "undertaken."
    doc.close()
