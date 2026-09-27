"""End to end: a tiny synthetic Zotero data dir -> build_metadata.py -> build_corpus.py.

The HTML/TXT leg needs only the standard library. The PDF leg generates its PDFs
with PyMuPDF and converts them with pymupdf4llm; it is skipped when either is missing.
"""
import json
import os
import shutil
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_testlib import make_zotero_db, run_script

PROSE = "\n".join(
    f"Paragraph {w}: pooled estimates reduce sampling error relative to single-study results, "
    f"and the residual bias after the {w} correction is small in large samples."
    for w in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
              "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon"]) + "\n"
REFS = """References
Benjamini, Y., and Y. Hochberg (1995), Controlling the false discovery rate: A practical and powerful approach to multiple testing, J. R. Stat. Soc. B, 57, 289–300, doi:10.1111/j.2517-6161.1995.tb02031.x.
Hansen, J., R. Ruedy, M. Sato, and K. Lo (2010), Global surface temperature change, Rev. Geophys., 48, RG4004, doi:10.1029/2010RG000345.
Hurrell, J. W. (1995), Decadal trends in the North Atlantic Oscillation: Regional temperatures and precipitation, Science, 269, 676–679, doi:10.1126/science.269.5224.676.
Tversky, A., and D. Kahneman (1974), Judgment under uncertainty: Heuristics and biases, Science, 185, 1124–1131, doi:10.1126/science.185.4157.1124.
Watts, D. J., and S. H. Strogatz (1998), Collective dynamics of 'small-world' networks, Nature, 393(6684), 440–442.
"""
SCHEMA_KEYS = {"filename", "title", "authors", "year", "doi", "publication", "tags", "abstract", "source"}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _manifest(out):
    rows = [json.loads(x) for x in (out / ".manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["key"]: r for r in rows}


def _build(zdir, out, meta, *extra):
    r = run_script("build_corpus.py", "--zotero", zdir, "--out", out, "--meta", meta,
                   "--jobs", "1", "--ocr", "off", *extra)
    assert r.returncode == 0, r.stdout + r.stderr
    return _manifest(out)


@pytest.fixture
def library(tmp_path):
    zdir = tmp_path / "Zotero"
    st = zdir / "storage"
    make_zotero_db(zdir, [
        {"key": "TEST0001",
         "fields": {"title": "Synthetic survey evaluation", "date": "2019-03-00 March 2019",
                    "DOI": "10.0000/synthetic.0001", "publicationTitle": "Journal of Tests",
                    "abstractNote": "We   evaluate\nsynthetic surveys."},
         "creators": [("Doe", "Jane"), ("Roe", "Richard")], "tags": ["surveys", "test"]},
        {"key": "TEST0101", "parent": "TEST0001", "path": "storage:snapshot.html"},
        {"key": "TEST0102", "path": "storage:field notes.txt", "fields": {"title": "Field notes"}},
        {"key": "TEST0103", "path": "storage:page.html"},
        {"key": "TEST0104", "path": "storage:blocked.txt"},
        {"key": "TEST0105", "path": "storage:missing.pdf"},
    ])
    # HTML snapshot: the page itself is navigation junk; Zotero's full-text cache is clean
    _write(st / "TEST0101" / "snapshot.html",
           "<html><body><nav>Subscribe to our newsletter | Sign in</nav>"
           "<p>Cookie banner text</p></body></html>")
    _write(st / "TEST0101" / ".zotero-ft-cache", PROSE + "\n" + REFS)
    _write(st / "TEST0102" / "field notes.txt", "Field notes on thermometer drift.\n" + PROSE)
    # HTML without a cache: falls back to pandoc, then BeautifulSoup, whichever is present
    _write(st / "TEST0103" / "page.html", "<html><body><p>" + PROSE.replace("\n", " ") + "</p></body></html>")
    _write(st / "TEST0104" / "blocked.txt", PROSE)
    (st / "TEST0105").mkdir(parents=True)                       # attachment file never synced
    _write(st / "TEST0106" / "stray.txt", "Stray text with no metadata entry.\n" + PROSE)
    deny = tmp_path / "deny.txt"
    deny.write_text("TEST0104\n", encoding="utf-8")
    return zdir, deny


def test_metadata_then_corpus_then_resume(tmp_path, library):
    zdir, deny = library
    meta, out = tmp_path / "zotero_metadata.json", tmp_path / "rag_corpus"

    r = run_script("build_metadata.py", "--zotero", zdir, "--out", meta)
    assert r.returncode == 0, r.stderr
    md = json.loads(meta.read_text(encoding="utf-8"))
    assert set(md) == {"TEST0101", "TEST0102", "TEST0103", "TEST0104", "TEST0105"}
    assert all(set(v) == SCHEMA_KEYS for v in md.values())

    m = _build(zdir, out, meta, "--denylist", deny)
    assert {k: v["method"] for k, v in m.items()} == {
        "TEST0101": "html+ftcache", "TEST0102": "txt", "TEST0103": m["TEST0103"]["method"],
        "TEST0104": "denylist", "TEST0105": "no_source", "TEST0106": "txt"}
    assert m["TEST0103"]["method"] in ("html", "html_empty")
    assert m["TEST0101"]["out"] == "TEST0101__snapshot.md"

    html = (out / "TEST0101__snapshot.md").read_text(encoding="utf-8")
    assert html.startswith("# Synthetic survey evaluation\n\n**Authors:** Doe Jane; Roe Richard\n")
    assert "**Year:** 2019  **Published in:** Journal of Tests  **DOI:** 10.0000/synthetic.0001" in html
    assert "**Tags:** surveys, test" in html
    assert "**Zotero:** zotero://select/library/items/TEST0101" in html
    assert "> **Abstract.** We evaluate synthetic surveys." in html
    assert "Paragraph upsilon" in html and "Subscribe" not in html          # the cache, not the page
    assert "Benjamini" not in html and "Strogatz" not in html                    # reference list stripped
    assert (out / "TEST0102__field_notes.md").read_text(encoding="utf-8").startswith("# Field notes\n")
    assert (out / "TEST0106__stray.md").read_text(encoding="utf-8").startswith("# (untitled)\n")
    if m["TEST0103"]["method"] == "html":                                   # pandoc wraps lines
        assert "Paragraph upsilon" in " ".join((out / "TEST0103__page.md").read_text(encoding="utf-8").split())
    assert not any(n.startswith("TEST0104") for n in os.listdir(out))
    assert not any(n.endswith(".tmp") for n in os.listdir(out))

    # up to date: nothing is rebuilt
    m = _build(zdir, out, meta, "--denylist", deny)
    assert {m[k]["method"] for k in ("TEST0101", "TEST0102", "TEST0103", "TEST0106")} == {"skip"}

    # LightRAG moved an ingested file into __parsed__/: still counts as built
    (out / "__parsed__").mkdir()
    shutil.move(out / "TEST0102__field_notes.md", out / "__parsed__" / "TEST0102__field_notes.md")
    m = _build(zdir, out, meta, "--denylist", deny)
    assert m["TEST0102"]["method"] == "skip" and not (out / "TEST0102__field_notes.md").exists()

    # a changed source is rebuilt at the corpus root, even when the old output was ingested,
    # and the log says so; a file in __parsed__/ with some other suffix is not a copy of it
    now = time.time()
    os.utime(out / "__parsed__" / "TEST0102__field_notes.md", (now - 3600, now - 3600))
    os.utime(zdir / "storage" / "TEST0102" / "field notes.txt", (now - 60, now - 60))
    (out / "__parsed__" / "TEST0102__field_notes_recovered.md").write_text("x", encoding="utf-8")
    r = run_script("build_corpus.py", "--zotero", zdir, "--out", out, "--meta", meta,
                   "--jobs", "1", "--ocr", "off", "--denylist", deny)
    assert r.returncode == 0, r.stdout + r.stderr
    m = _manifest(out)
    assert m["TEST0102"]["method"] == "txt" and (out / "TEST0102__field_notes.md").exists()
    assert "REBUILT TEST0102__field_notes.md" in r.stdout

    # a scan does not index a name LightRAG has already processed: it archives the file as
    # __parsed__/<stem>_001.md. That copy is as new as the source, so it counts as built
    # and the next run does not rebuild it (and hand LightRAG another copy) again.
    shutil.move(out / "TEST0102__field_notes.md", out / "__parsed__" / "TEST0102__field_notes_001.md")
    m = _build(zdir, out, meta, "--denylist", deny)
    assert m["TEST0102"]["method"] == "skip" and m["TEST0102"]["out"] == "TEST0102__field_notes_001.md"
    assert not (out / "TEST0102__field_notes.md").exists()

    # a hold dir counts as built only when --held-dir names it (default: none)
    held = tmp_path / "held"
    held.mkdir()
    shutil.move(out / "TEST0101__snapshot.md", held / "TEST0101__snapshot.md")
    m = _build(zdir, out, meta, "--denylist", deny, "--held-dir", held)
    assert m["TEST0101"]["method"] == "skip" and not (out / "TEST0101__snapshot.md").exists()
    m = _build(zdir, out, meta, "--denylist", deny)
    assert m["TEST0101"]["method"] == "html+ftcache" and (out / "TEST0101__snapshot.md").exists()

    # without the denylist the blocked attachment is built
    m = _build(zdir, out, meta)
    assert m["TEST0104"]["method"] == "txt" and (out / "TEST0104__blocked.md").exists()


def test_pdf_text_layer_and_image_only_fallbacks(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    zdir = tmp_path / "Zotero"
    make_zotero_db(zdir, [
        {"key": "TEST0002", "fields": {"title": "A generated paper", "date": "2020"},
         "creators": [("Doe", "Jane")]},
        {"key": "TEST0201", "parent": "TEST0002", "path": "storage:Doe 2020 generated.pdf"},
        {"key": "TEST0202", "path": "storage:scan.pdf", "fields": {"title": "A scan"}},
    ])
    text_pdf = zdir / "storage" / "TEST0201" / "Doe 2020 generated.pdf"
    scan_pdf = zdir / "storage" / "TEST0202" / "scan.pdf"
    text_pdf.parent.mkdir(parents=True)
    scan_pdf.parent.mkdir(parents=True)
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(PROSE.splitlines()[:12]):
        page.insert_text((72, 72 + 18 * i), line[:90], fontsize=9)
    doc.save(str(text_pdf))
    doc.close()
    doc = pymupdf.open()
    doc.new_page()                                    # no text layer: stands in for a scan
    doc.save(str(scan_pdf))
    doc.close()

    meta, out = tmp_path / "zotero_metadata.json", tmp_path / "rag_corpus"
    assert run_script("build_metadata.py", "--zotero", zdir, "--out", meta).returncode == 0

    m = _build(zdir, out, meta)
    assert m["TEST0201"]["method"] == "pdf"
    body = (out / "TEST0201__Doe_2020_generated.md").read_text(encoding="utf-8")
    assert body.startswith("# A generated paper\n\n**Authors:** Doe Jane\n") and "Paragraph alpha" in body
    assert m["TEST0202"]["method"] == "pdf+ocr_empty"          # --ocr off, no cache: header only

    cache, legacy = tmp_path / "ocr_cache", tmp_path / "legacy_md"
    _write(cache / "scan.pdf.txt", "OCR cache text for the scanned page.\n" + PROSE)
    _write(legacy / "scan.md", "Legacy extraction of the scanned page.\n" + PROSE)
    m = _build(zdir, out, meta, "--force", "--ocr-cache-dir", cache, "--fallback-md-dir", legacy)
    assert m["TEST0202"]["method"] == "pdf+ocr_cache"
    assert "OCR cache text" in (out / "TEST0202__scan.md").read_text(encoding="utf-8")
    m = _build(zdir, out, meta, "--force", "--fallback-md-dir", legacy)
    assert m["TEST0202"]["method"] == "pdf_legacy_md"
    assert "Legacy extraction" in (out / "TEST0202__scan.md").read_text(encoding="utf-8")
