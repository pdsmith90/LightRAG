"""build_corpus recognises glyph garbage beyond Private Use Area codes and keeps it out.

Three shapes a broken text layer takes -- U+FFFD replacement characters, C0 control codes
and letter soup -- are judged per page inside pdf_to_md (a garbled engine page is repaired
from a clean layer; a garbled layer is never injected and the page is dropped) and again on
the whole body in convert(), whose OCR route can now cover just the dropped pages. Numeric
tables and equation pages must stay. Synthetic PDFs are built with pymupdf where a test
needs one; pymupdf4llm and ocrmypdf are replaced by fakes. All text is invented.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_testlib  # noqa: F401  (puts corpus/ on sys.path)
import build_corpus as bc

LINE = (
    "The scheduler assigns each queued job to the worker with the shortest expected "
    "completion time."
)
PROSE = "\n".join(LINE for _ in range(16))  # ~1,350 non-space characters
FFFD = ("�" * 9 + " ") * 160  # what the engine emits for an undecodable font
CONTROL = " ".join(
    "\x01\x02\x03\x04\x05" for _ in range(200)
)  # glyphs mapped to control codes
SOUP_LINE = "c ? c o o c o c c c o c o o c o c o"
SOUP = "\n".join(
    SOUP_LINE for _ in range(40)
)  # OCR letter soup: 720 single-character tokens
TABLE = "\n".join(
    " ".join(f"{(i * j) % 977:5d}" for j in range(12)) for i in range(60)
)  # a page of numbers
TEX = (
    PROSE + "\n" + " ".join("\x01\x02" for _ in range(120))
)  # symbol-font codes in real prose
EQUATIONS = " ".join(
    f"x {i} = {i * i} y {i + 1}" for i in range(120)
)  # digit-heavy equations
GLYPHS = "".join(chr(0xF020 + (i % 90)) for i in range(5000))
PUA = " ".join(GLYPHS[i : i + 8] for i in range(0, 5000, 8))
KEY = "TEST0003"


def test_garble_reason_shapes():
    assert bc.garble_reason(PROSE) is None
    assert bc.garble_reason(TABLE) is None  # digits are not garbage
    assert (
        bc.garble_reason(EQUATIONS) is None
    )  # single-letter variables, but mostly digits
    assert bc.garble_reason(TEX) is None  # control codes inside real prose stay
    assert bc.garble_reason(FFFD) == "replacement"
    assert bc.garble_reason(CONTROL) == "control"
    assert bc.garble_reason(SOUP) == "letter-soup"
    assert bc.garble_reason(PUA) == "pua"
    assert bc.garble_reason(SOUP_LINE) is None  # under the 500 non-space floor
    assert bc.garble_reason(SOUP[:300], bc.GARBLED_PAGE_MIN_NONSPACE) is None
    assert bc.garble_reason(SOUP[:500], bc.GARBLED_PAGE_MIN_NONSPACE) == "letter-soup"
    assert bc.looks_garbled(SOUP) and not bc.looks_garbled(PROSE)
    assert not bc.looks_garbled("")
    assert "single-char tokens 100%" in bc.garble_detail(SOUP)


MOJIBAKE = " ".join("ÄóÓÒeÌôbÑÃØÏ¨¯¬¥" for _ in range(80))  # a font mapped to Latin-1 codes
FRENCH = "\n".join(
    "Les mesures gravimétriques révèlent une déformation élastique de la croûte."
    for _ in range(16)
)  # accented prose stays
DEGREES = "\n".join(
    f"{i}° ± {i / 10:.1f} µm  {i * 3}° ± 0.{i % 9} µm" for i in range(80)
)  # scientific symbols are not counted


def test_latin1_shape():
    assert bc.garble_reason(MOJIBAKE) == "latin1"
    assert bc.garble_reason(FRENCH) is None
    assert bc.garble_reason(DEGREES) is None
    assert "Latin-1 88%" in bc.garble_detail(MOJIBAKE)


def test_clean_md_strips_control_codes():
    out = bc.clean_md(
        "Intro\x00duction\x01 to the\x1f method.\n\nSecond paragraph of the text."
    )
    assert "\x00" not in out and "\x01" not in out and "\x1f" not in out
    assert "Introduction to the method." in out and "Second paragraph" in out
    assert "\t" in bc.clean_md("a\tb " * 100)  # tabs and newlines survive


# ---------------------------------------------------------------- pdf_to_md: the per-page gate
class FakeLLM:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def to_markdown(self, path, page_chunks=False, use_ocr=True, **kw):
        assert use_ocr is False
        self.calls.append(page_chunks)
        return [
            {"text": t, "metadata": {"page_number": i + 1}}
            for i, t in enumerate(self.pages)
        ]


def pdf_to_md_with(monkeypatch, fake, path):
    monkeypatch.setitem(
        sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=fake.to_markdown)
    )
    return bc.pdf_to_md(path)


def make_pdf(path, layers):
    """One page per entry of `layers`, each the page's text layer."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in layers:
        page = doc.new_page()
        page.insert_text((60, 80), text, fontsize=8)
    doc.save(path)
    doc.close()


def test_engine_garbage_is_repaired_from_a_clean_layer(tmp_path, monkeypatch):
    p = str(tmp_path / "paper.pdf")
    make_pdf(p, [PROSE, PROSE, PROSE])
    good = "## Section\n\n" + PROSE.replace("\n", " ")
    fake = FakeLLM([FFFD, good, good])  # the engine decoded page 1's font as U+FFFD
    md = pdf_to_md_with(monkeypatch, fake, p)
    assert "�" not in md
    assert md.count("shortest expected") >= 48
    assert bc._pdf_method == "pdf+textlayer"
    assert bc._pdf_garbled_pages == [] and bc._pdf_page_count == 3


def test_garbage_layer_is_dropped_whether_passed_through_or_emptied(
    tmp_path, monkeypatch, capsys
):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, [SOUP, SOUP])
    fake = FakeLLM([SOUP, ""])  # page 1 passed through, page 2 emptied (kept 0 %)
    md = pdf_to_md_with(monkeypatch, fake, p)
    assert md == "" and bc._pdf_garbled_pages == [0, 1]
    assert bc._pdf_method == ""  # nothing was repaired
    out = capsys.readouterr().out
    assert "GARBLED pages scan.pdf: 2/2" in out and "letter-soup 2" in out


def test_partly_garbled_document_keeps_the_good_pages(tmp_path, monkeypatch):
    p = str(tmp_path / "scan.pdf")
    make_pdf(p, [PROSE, SOUP, PROSE])
    good = "## Section\n\n" + PROSE.replace("\n", " ")
    fake = FakeLLM([good, SOUP, good])
    md = pdf_to_md_with(monkeypatch, fake, p)
    assert bc._pdf_garbled_pages == [1] and bc._pdf_method == ""
    assert SOUP_LINE not in md and md.count("shortest expected") >= 32


# ---------------------------------------------------------------- convert(): the OCR routes
def run_convert(tmp_path, monkeypatch, capsys, fake_pdf_to_md, fake_ocr, ocr="auto"):
    folder = tmp_path / "storage" / KEY
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scan.pdf").write_bytes(b"%PDF-1.4 fake")
    out = tmp_path / "rag_corpus"
    out.mkdir(exist_ok=True)
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
            "ocr": ocr,
            "force": True,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )
    monkeypatch.setattr(bc, "pdf_to_md", fake_pdf_to_md)
    monkeypatch.setattr(bc, "ocr_pdf_to_md", fake_ocr)
    capsys.readouterr()
    key, method, path, n = bc.convert(KEY)
    body = open(path, encoding="utf-8").read() if path else ""
    return method, body, capsys.readouterr().out


def _partial(pages, count):
    def pdf(path):
        bc._pdf_method = ""
        bc._pdf_garbled_pages = list(pages)
        bc._pdf_page_count = count
        return PROSE

    return pdf


def test_convert_ocrs_only_the_garbled_pages(tmp_path, monkeypatch, capsys):
    calls = []

    def ocr(path, pages=None):
        calls.append(pages)
        return PROSE + "\nOCR TEXT OF THE GARBLED PAGE"

    method, body, out = run_convert(
        tmp_path, monkeypatch, capsys, _partial([1], 3), ocr
    )
    assert method == "pdf+ocr_pages" and calls == [[1]]
    assert "OCR TEXT OF THE GARBLED PAGE" in body and "GARBLED text layer" not in out


def test_convert_labels_a_majority_of_garbled_pages_as_full_ocr(
    tmp_path, monkeypatch, capsys
):
    method, body, out = run_convert(
        tmp_path,
        monkeypatch,
        capsys,
        _partial([0, 1, 2, 3], 5),
        lambda path, pages=None: PROSE + "\nOCR TEXT",
    )
    assert method == "pdf+ocr" and "OCR TEXT" in body


def test_convert_keeps_readable_pages_when_ocr_fails_or_is_off(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def failing_ocr(path, pages=None):
        calls.append(pages)
        return ""

    method, body, out = run_convert(
        tmp_path, monkeypatch, capsys, _partial([1], 3), failing_ocr
    )
    assert method == "pdf_garbled_pages_dropped" and calls == [[1]]
    assert "shortest expected" in body
    method, body, out = run_convert(
        tmp_path, monkeypatch, capsys, _partial([1], 3), failing_ocr, ocr="off"
    )
    assert method == "pdf_garbled_pages_dropped" and calls == [
        [1]
    ]  # not called when off


def test_convert_routes_a_garbled_body_to_ocr_and_never_writes_garbage(
    tmp_path, monkeypatch, capsys
):
    def pdf(path):
        bc._pdf_method = ""
        return SOUP

    method, body, out = run_convert(
        tmp_path, monkeypatch, capsys, pdf, lambda path, pages=None: PROSE
    )
    assert method == "pdf+ocr" and "shortest expected" in body
    assert "GARBLED text layer scan.pdf" in out and "letter-soup" in out
    method, body, out = run_convert(
        tmp_path, monkeypatch, capsys, pdf, lambda path, pages=None: ""
    )
    assert method == "pdf_needs_ocr" and SOUP_LINE not in body
    assert body.startswith("# A scanned paper")


# ---------------------------------------------------------------- ocr_pdf_to_md: --pages and the time limit
def test_page_ranges():
    assert bc._page_ranges([0, 1, 2, 6, 8, 9, 10]) == "1-3,7,9-11"
    assert bc._page_ranges([4]) == "5"
    assert bc._page_ranges([3, 3, 2]) == "3-4"


def test_ocr_arguments_and_timeout(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    runs = []

    def fake_run(args, **kw):
        runs.append((args, kw))

    ten, big = str(tmp_path / "ten.pdf"), str(tmp_path / "big.pdf")
    for path, n in ((ten, 10), (big, 611)):
        doc = pymupdf.open()
        for _ in range(n):
            doc.new_page()
        doc.save(path)
        doc.close()
    monkeypatch.setattr(bc.shutil, "which", lambda name: "/usr/bin/ocrmypdf")
    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    monkeypatch.setattr(bc, "pdf_to_md", lambda path: "ocr text")
    assert bc.ocr_pdf_to_md(ten, [1]) == "ocr text"
    assert bc.ocr_pdf_to_md(ten, list(range(8))) == "ocr text"
    assert bc.ocr_pdf_to_md(ten) == "ocr text"
    assert bc.ocr_pdf_to_md(big) == "ocr text"
    a1, a2, a3, a4 = runs
    assert a1[0][:3] == ["ocrmypdf", "--force-ocr", "--quiet"]
    assert a1[0][3:5] == ["--pages", "2"]
    assert (
        "--pages" not in a2[0] and "--pages" not in a3[0]
    )  # 8 of 10 pages: OCR everything
    assert a1[1]["timeout"] == 600 and a3[1]["timeout"] == 600
    assert a4[1]["timeout"] == 5 * 611 + 60
