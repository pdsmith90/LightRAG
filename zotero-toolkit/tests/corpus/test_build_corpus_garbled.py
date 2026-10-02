"""build_corpus routes a PDF whose text layer is Private Use Area glyph codes to OCR.

pdf_to_md and ocr_pdf_to_md are replaced, so these tests exercise convert()'s decision,
not PyMuPDF or ocrmypdf, and run without either installed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_testlib  # noqa: F401  (puts corpus/ on sys.path)
import build_corpus as bc

PROSE = (
    "The scheduler assigns each queued job to the worker with the shortest expected "
    "completion time and rebalances the queues after every batch.\n"
) * 40
GLYPHS = "".join(chr(0xF020 + (i % 90)) for i in range(5000))
GARBAGE = " ".join(GLYPHS[i : i + 8] for i in range(0, 5000, 8)) + "\n###### \n"
GREEK_HEAVY = PROSE + " ".join(chr(0xF061 + (i % 20)) for i in range(150))  # ~4 %
KEY = "TEST0001"


def test_looks_garbled():
    assert bc.looks_garbled(GARBAGE)
    assert not bc.looks_garbled(PROSE)
    assert not bc.looks_garbled(GREEK_HEAVY)
    assert not bc.looks_garbled(GLYPHS[:300])  # under the 500 non-space floor
    assert not bc.looks_garbled("")


def run_convert(tmp_path, pdf_text, ocr_text, monkeypatch, ocr="auto"):
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
                    "title": "A scanned book",
                    "authors": ["Doe J"],
                    "year": "1999",
                    "tags": [],
                    "abstract": "",
                }
            },
            "zotero": str(tmp_path),
            "out": str(out),
            "deny": set(),
            "min_chars": 200,
            "ocr": ocr,
            "force": False,
            "held": None,
            "ocr_cache": None,
            "fallback_md": None,
        }
    )
    calls = {"ocr": 0}

    def fake_ocr(path):
        calls["ocr"] += 1
        return ocr_text

    monkeypatch.setattr(bc, "pdf_to_md", lambda path: pdf_text)
    monkeypatch.setattr(bc, "ocr_pdf_to_md", fake_ocr)
    key, method, path, n = bc.convert(KEY)
    body = open(path, encoding="utf-8").read() if path else ""
    return method, body, calls["ocr"]


def test_garbled_layer_is_routed_to_ocr(tmp_path, monkeypatch, capsys):
    method, body, ocr_calls = run_convert(tmp_path, GARBAGE, PROSE, monkeypatch)
    assert method == "pdf+ocr" and ocr_calls == 1
    assert "shortest expected completion time" in body
    assert not bc._pua_re.search(body)
    assert "GARBLED text layer scan.pdf" in capsys.readouterr().out


def test_garbled_layer_with_failed_ocr_writes_the_header_only(tmp_path, monkeypatch):
    method, body, ocr_calls = run_convert(tmp_path, GARBAGE, "", monkeypatch)
    assert method == "pdf_needs_ocr" and ocr_calls == 1
    assert body.startswith("# A scanned book") and not bc._pua_re.search(body)


def test_garbled_layer_with_ocr_off_writes_the_header_only(tmp_path, monkeypatch):
    method, body, ocr_calls = run_convert(
        tmp_path, GARBAGE, PROSE, monkeypatch, ocr="off"
    )
    assert method == "pdf_needs_ocr" and ocr_calls == 0
    assert not bc._pua_re.search(body)


def test_clean_pdf_is_untouched(tmp_path, monkeypatch, capsys):
    method, body, ocr_calls = run_convert(tmp_path, PROSE, "UNUSED", monkeypatch)
    assert method == "pdf" and ocr_calls == 0
    assert "shortest expected completion time" in body
    assert "GARBLED" not in capsys.readouterr().out


def test_greek_symbol_font_is_not_garbled(tmp_path, monkeypatch):
    method, body, ocr_calls = run_convert(tmp_path, GREEK_HEAVY, "UNUSED", monkeypatch)
    assert method == "pdf" and ocr_calls == 0


def test_empty_layer_still_goes_to_ocr(tmp_path, monkeypatch):
    method, body, ocr_calls = run_convert(tmp_path, "   \n", PROSE, monkeypatch)
    assert method == "pdf+ocr" and ocr_calls == 1
    assert "shortest expected completion time" in body
