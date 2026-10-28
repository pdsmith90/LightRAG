"""Unit tests for maintenance/thin_doc_audit.py: offline, no database (fetch_rows is replaced)."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pymupdf
import pytest

MAINTENANCE = Path(__file__).resolve().parents[2] / "maintenance"
spec = importlib.util.spec_from_file_location(
    "thin_doc_audit", MAINTENANCE / "thin_doc_audit.py"
)
tda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tda)

HEADER = "# A title\n\n**Authors:** A. Writer\n**Zotero:** zotero://select/library/items/0THIN1DOC0\n"
LINE = "Each queued job goes to the worker with the shortest expected completion time. "


def make_pdf(path, pages=3, text=True):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if text:
            page.insert_text((50, 80), "\n".join(LINE for _ in range(30)), fontsize=8)
    doc.save(str(path))
    doc.close()


def row(key="0THIN1DOC0", text=HEADER + "\n---\n", chunks=1, created="2026-01-10"):
    return {
        "doc_id": "doc-" + key,
        "file_path": f"{key}__slug.md",
        "chunks": chunks,
        "created": created,
        "text": text,
    }


def enriched(kb_chars, kb_body, pdf_chars, created="2026-01-10", ctime=None, key="0K1"):
    return {
        "key": key,
        "file_path": f"{key}__s.md",
        "chunks": 2,
        "created": created,
        "kb_chars": kb_chars,
        "kb_body": kb_body,
        "pdf": {
            "pages": 5,
            "chars": pdf_chars,
            "ctime": ctime if ctime is not None else time.time(),
            "mtime": 0,
        },
    }


# ---- header / body ----


def test_body_with_abstract():
    t = HEADER + "\n> **Abstract.** Short.\n\n---\nBody text here."
    assert tda.kb_body_chars(t) == len("Body text here.")


def test_body_header_only():
    assert tda.kb_body_chars(HEADER + "\n---\n") == 0


def test_body_without_header_is_whole_text():
    assert tda.kb_body_chars("one\n---\ntwo\n") == len("one\n---\ntwo")


def test_dash_line_before_zotero_line_is_not_the_header_end():
    t = "intro\n---\nmore\n" + HEADER + "\n---\nreal"
    assert tda.kb_body_chars(t) == 4


# ---- PDF measurement ----


def test_pdf_text_measures_text_pdf(tmp_path):
    make_pdf(tmp_path / "a.PDF", pages=3)
    info = tda.pdf_text(str(tmp_path))
    assert info["pages"] == 3 and info["chars"] > 3000
    assert info["ctime"] > 0 and info["mtime"] > 0


def test_pdf_text_empty_pages_and_missing(tmp_path):
    make_pdf(tmp_path / "blank.pdf", pages=2, text=False)
    assert tda.pdf_text(str(tmp_path))["chars"] == 0
    assert tda.pdf_text(str(tmp_path / "nope")) is None
    empty = tmp_path / "e"
    empty.mkdir()
    assert tda.pdf_text(str(empty)) is None


def test_pdf_text_picks_largest_and_scales(tmp_path):
    make_pdf(tmp_path / "small.pdf", pages=1)
    make_pdf(tmp_path / "big.pdf", pages=6)
    full = tda.pdf_text(str(tmp_path))
    capped = tda.pdf_text(str(tmp_path), max_pages=2)
    assert full["pages"] == 6
    assert abs(capped["chars"] - full["chars"]) < full["chars"] * 0.1


# ---- candidate selection ----


def test_min_gain_boundary_and_floor():
    # kb 5000 -> need > max(50000, 20000)
    assert not tda.select_candidates([enriched(5000, 4000, 50000)], 10)
    assert tda.select_candidates([enriched(5000, 4000, 50001)], 10)
    # kb 100 body 150 (stub rule needs <200 body, so use body 500) -> floor 20000 binds
    assert not tda.select_candidates([enriched(1000, 500, 20000)], 10)
    assert tda.select_candidates([enriched(1000, 500, 20001)], 10)


def test_header_only_rule():
    assert tda.select_candidates([enriched(171, 0, 2000)], 10)
    assert not tda.select_candidates([enriched(171, 0, 1999)], 10)
    assert not tda.select_candidates([enriched(400, 200, 3000)], 10)


def test_no_pdf_unreadable_and_excluded_are_skipped():
    r = enriched(100, 0, 90000)
    assert not tda.select_candidates([dict(r, pdf=None)])
    bad = dict(r, pdf=dict(r["pdf"], chars=-1))
    assert not tda.select_candidates([bad])
    assert not tda.select_candidates([r], exclude=["0K1"])


def test_classification_by_ctime_date():
    ct = time.mktime((2026, 3, 5, 12, 0, 0, 0, 0, -1))
    after = tda.select_candidates([enriched(100, 0, 90000, "2026-03-01", ct)])
    same = tda.select_candidates([enriched(100, 0, 90000, "2026-03-05", ct)])
    assert after[0]["class"] == "arrived after ingest"
    assert same[0]["class"] == "thin first conversion"


# ---- TSV ----


def test_write_targets_format(tmp_path):
    c = dict(
        enriched(171, 0, 90000, key="0STUB1K0"), **{"class": "arrived after ingest"}
    )
    t = dict(
        enriched(2459, 900, 90000, key="0THIN1K0"),
        dry={"body": 34022},
        **{"class": "thin first conversion"},
    )
    t["chunks"] = 3
    out = tmp_path / "t.tsv"
    tda.write_targets(str(out), [t, c], "confirmed", "kb-max-chars 4000")
    lines = out.read_text().splitlines()
    assert lines[0].startswith("#") and "kb-max-chars 4000" in lines[0]
    data = [x for x in lines if not x.startswith("#")]
    cols = [x.split("\t") for x in data]
    assert all(len(x) == 4 for x in cols)
    assert cols[0] == [
        "0THIN1K0",
        "thin conversion (KB 2459 chars); dry convert gives 34022",
        "3",
        "0THIN1K0__s.md",
    ]
    assert cols[1][1] == (
        "stub ingested before the full PDF arrived (KB 171 chars); unconfirmed"
    )


def test_reason_follows_the_class_not_the_body_size():
    # header-only, but the PDF was already there at the first conversion
    c = dict(
        enriched(237, 0, 7000, key="0HDR1K0"), **{"class": "thin first conversion"}
    )
    assert tda.reason(c).startswith("thin conversion (KB 237 chars)")


# ---- dry convert with a fake build_corpus ----

FAKE = """
import os
import sys
SEEN_ENV = {k: os.environ.get(k) for k in ("OCR_VLM_REMOTE_URL", "OCR_VLM_LOCAL_URL", "OCR_VLM_LOCAL_START")}
CFG = {}
OUTS = []
print("noise at import")
def convert(key):
    print("noise converting", key)
    OUTS.append(CFG["out"])
    assert CFG["ocr"] == "off" and CFG["ocr_vlm"] == "off" and CFG["force"] is True
    p = os.path.join(CFG["out"], key + "__x.md")
    with open(p, "w") as f:
        f.write("# T\\n**Zotero:** zotero://select/library/items/" + key + "\\n\\n---\\n" + "z" * 5000)
    return (key, "ok", p, 1)
"""


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch):
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "build_corpus.py").write_text(FAKE)
    (b / "zotero_metadata.json").write_text("{}")
    for k in ("OCR_VLM_REMOTE_URL", "OCR_VLM_LOCAL_URL", "OCR_VLM_LOCAL_START"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield b
    sys.modules.pop("build_corpus", None)


def test_dry_convert(fake_bundle, capsys):
    res = tda.dry_convert(
        ["0K1", "0K2"], str(fake_bundle), str(fake_bundle / "zotero_metadata.json")
    )
    assert res == {
        "0K1": {"status": "ok", "body": 5000},
        "0K2": {"status": "ok", "body": 5000},
    }
    bc = sys.modules["build_corpus"]
    assert bc.SEEN_ENV == {
        "OCR_VLM_REMOTE_URL": "http://127.0.0.1:9",
        "OCR_VLM_LOCAL_URL": "http://127.0.0.1:9",
        "OCR_VLM_LOCAL_START": "",
    }
    assert bc.OUTS and not os.path.exists(bc.OUTS[0])
    cap = capsys.readouterr()
    assert "noise" not in cap.out
    assert str(fake_bundle) not in sys.path


def test_dry_convert_cleans_up_on_error(fake_bundle):
    with pytest.raises(OSError):
        tda.dry_convert(["0K1"], str(fake_bundle), str(fake_bundle / "missing.json"))
    assert str(fake_bundle) not in sys.path


# ---- main with the DB layer mocked ----


def test_main_end_to_end(tmp_path, monkeypatch, capsys, fake_bundle):
    zot = tmp_path / "zot"
    (zot / "storage" / "0THIN1DOC0").mkdir(parents=True)
    make_pdf(zot / "storage" / "0THIN1DOC0" / "p.pdf", pages=20)
    (zot / "storage" / "0OKAY1DOC0").mkdir(parents=True)
    make_pdf(zot / "storage" / "0OKAY1DOC0" / "p.pdf", pages=1)
    rows = [row(), row("0OKAY1DOC0", text="x" * 3000), row("0GONE1DOC0")]
    seen = {}

    def fake_fetch(ws, mx, env=None):
        seen.update(ws=ws, mx=mx)
        return rows

    monkeypatch.setattr(tda, "fetch_rows", fake_fetch)
    out = tmp_path / "t.tsv"
    rc = tda.main(
        [
            "--zotero",
            str(zot),
            "--workspace",
            "w1",
            "--out",
            str(out),
            "--dry-convert",
            str(fake_bundle),
            "--exclude",
            "0GONE1DOC0",
        ]
    )
    assert rc == 0 and seen == {"ws": "w1", "mx": 4000}
    printed = capsys.readouterr().out
    assert "0THIN1DOC0" in printed and "0OKAY1DOC0" not in printed
    assert "noise" not in printed
    data = [x for x in out.read_text().splitlines() if not x.startswith("#")]
    assert len(data) == 1 and data[0].startswith("0THIN1DOC0\tstub ingested")


def test_main_unconfirmed_without_dry_convert(tmp_path, monkeypatch):
    zot = tmp_path / "zot"
    (zot / "storage" / "0THIN1DOC0").mkdir(parents=True)
    make_pdf(zot / "storage" / "0THIN1DOC0" / "p.pdf", pages=20)
    monkeypatch.setattr(tda, "fetch_rows", lambda *a, **k: [row()])
    out = tmp_path / "t.tsv"
    assert tda.main(["--zotero", str(zot), "--out", str(out)]) == 0
    assert "unconfirmed" in out.read_text().splitlines()[-1]


def test_main_db_unreachable(monkeypatch, capsys):
    def boom(*a, **k):
        raise ConnectionRefusedError("no route")

    monkeypatch.setattr(tda, "fetch_rows", boom)
    assert tda.main([]) == 1
    assert "database unreachable" in capsys.readouterr().err
