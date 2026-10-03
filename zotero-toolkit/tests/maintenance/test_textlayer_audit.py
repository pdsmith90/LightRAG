"""Unit tests for maintenance/textlayer_audit.py on a synthetic corpus (pymupdf builds the PDFs)."""

import importlib.util
from pathlib import Path

import pymupdf

MAINTENANCE = Path(__file__).resolve().parents[2] / "maintenance"
spec = importlib.util.spec_from_file_location(
    "textlayer_audit", MAINTENANCE / "textlayer_audit.py"
)
textlayer_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(textlayer_audit)

LINE = "Each queued job goes to the worker with the shortest expected completion time. "
PAGE_TEXT = "\n".join(LINE for _ in range(20))  # ~1,500 non-space characters per page
HEADER = "# A title\n\n**Year:** 2000\n\n---\n"


def make_pdf(path, pages=4, scan=True):
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 60), False)
    pix.clear_with(220)
    for _ in range(pages):
        page = doc.new_page()
        if scan:
            page.insert_image(page.rect, pixmap=pix)
        page.insert_text((60, 100), PAGE_TEXT, fontsize=9, render_mode=3 if scan else 0)
    doc.save(str(path))
    doc.close()


def make_corpus(tmp_path):
    zotero = tmp_path / "zotero"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for key, scan, body in (
        (
            "KEY00001",
            True,
            "Downloaded from an institutional subscription\n" * 4,
        ),  # lossy scan
        (
            "KEY00002",
            False,
            PAGE_TEXT.replace("\n", " ") + "\n" * 4,
        ),  # clean, but 4 pages' worth missing
        (
            "KEY00003",
            False,
            (PAGE_TEXT.replace("\n", " ") + "\n\n") * 4,
        ),  # clean, complete
    ):
        (zotero / "storage" / key).mkdir(parents=True)
        make_pdf(zotero / "storage" / key / "paper.pdf", scan=scan)
        (corpus / f"{key}__paper.md").write_text(HEADER + body, encoding="utf-8")
    (corpus / "KEY00004__no_pdf.md").write_text(HEADER + "orphan\n", encoding="utf-8")
    return zotero, corpus


def test_lossy_conversion_is_reported_and_clean_one_is_not(tmp_path, capsys):
    zotero, corpus = make_corpus(tmp_path)
    keys = tmp_path / "keys.txt"
    rc = textlayer_audit.main(
        ["--zotero", str(zotero), "--corpus", str(corpus), "--keys-out", str(keys)]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert keys.read_text().split() == [
        "KEY00001",
        "KEY00002",
    ]  # worst first in the listing
    assert "KEY00003" not in out and "KEY00004" not in out
    assert "scan" in out and "out of 3 audited" in out


def test_thresholds_and_exit_codes(tmp_path):
    zotero, corpus = make_corpus(tmp_path)
    assert (
        textlayer_audit.main(
            ["--zotero", str(zotero), "--corpus", str(corpus), "--max-ratio", "0.01"]
        )
        == 0
    )
    assert (
        textlayer_audit.main(
            [
                "--zotero",
                str(zotero),
                "--corpus",
                str(corpus),
                "--min-layer",
                "10000000",
            ]
        )
        == 0
    )
    assert (
        textlayer_audit.main(
            ["--zotero", str(zotero), "--corpus", str(tmp_path / "missing")]
        )
        == 2
    )


def test_json_rows_carry_the_measurements(tmp_path):
    zotero, corpus = make_corpus(tmp_path)
    js = tmp_path / "rows.json"
    textlayer_audit.main(
        [
            "--zotero",
            str(zotero),
            "--corpus",
            str(corpus),
            "--json-out",
            str(js),
            "--quiet",
        ]
    )
    import json

    rows = {r["key"]: r for r in json.load(open(js))}
    assert rows["KEY00001"]["scan"] is True and rows["KEY00001"]["ratio"] < 0.1
    assert rows["KEY00003"]["ratio"] > 0.6 and rows["KEY00003"]["pages"] == 4
