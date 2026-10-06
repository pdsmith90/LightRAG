#!/usr/bin/env python3
"""Behavioural tests for build_corpus's column-sanity gate (2026-10-06).
Run:  .venv/bin/python test_build_corpus_columns.py     (plain asserts; pytest also collects it)
A two-column page whose engine text lost the left column's line order is re-read from its blocks, whatever
font the layer uses; a page the engine read correctly, or one with too few lines to judge, is left alone.
Synthetic PDFs via pymupdf; pymupdf4llm replaced by a fake; all text is invented."""
import contextlib, io, os, sys, tempfile

try:
    import corpus_testlib  # noqa: F401  -- fork layout
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bundle layout
import pymupdf
import build_corpus as bc
from test_build_corpus_ocrlayer import fake_llm

LEFT = [
    "The osculating elements of the satellite are estimated from pseudorange data",
    "carrier-phase observations refine the along-track position of the spacecraft",
    "a Kalman filter propagates the state between the tracking passes of each day",
    "residuals below two centimetres indicate a converged orbit solution overall",
    "the empirical accelerations absorb the mismodelled drag along the orbit arc",
    "solar radiation pressure is scaled by one parameter per arc and per satellite",
    "ocean tide loading displaces the stations by a few millimetres each cycle",
    "the reference frame is realised through the fiducial station coordinates",
    "a priori sigmas constrain the loosely observed parameters of the solution",
    "the normal equations are stacked over consecutive arcs before inversion",
    "post-fit residuals are screened for outliers above four times their sigma",
    "the final orbit is compared with the laser ranging residuals as a check",
]
RIGHT = [
    "thermal noise in the receiver front end limits the ranging precision reached",
    "multipath near the ground antenna adds a slowly varying bias to the ranges",
    "tropospheric delay is modelled with a mapping function of the elevation angle",
    "ionospheric refraction cancels in the dual-frequency linear combination used",
    "clock offsets are eliminated by differencing between two receivers and two satellites",
    "the ambiguities are fixed to integers once their fractional parts converge",
    "antenna phase centre variations are calibrated on a robot before deployment",
    "the sampling interval of thirty seconds resolves the orbital signal fully",
    "receiver tracking loops lose lock briefly during strong scintillation events",
    "cycle slips are repaired from the geometry-free combination of the phases",
    "the elevation cutoff of ten degrees keeps the low observations out of the fit",
    "data gaps longer than one hour start a new arc in the batch estimator",
]
INTERLEAVED = "\n\n".join(f"{a} {b}" for a, b in zip(LEFT, RIGHT))
CORRECT = "\n\n".join(LEFT + RIGHT)


def two_column_pdf(path, left, right, single=False):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(left):
        page.insert_text((40, 80 + 16 * i), line, fontsize=7)
    if not single:
        for i, line in enumerate(right):
            page.insert_text((320, 80 + 16 * i), line, fontsize=7)
    doc.save(path)
    doc.close()


def run(path, pages):
    out = io.StringIO()
    with fake_llm(pages), contextlib.redirect_stdout(out):
        md = bc.pdf_to_md(path)
    return md, out.getvalue()


def test_two_column_lines_and_succession():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "paper.pdf")
        two_column_pdf(p, LEFT, RIGHT)
        with pymupdf.open(p) as doc:
            lines = bc._two_column_lines(doc[0])
        assert len(lines) == 12 and lines[0][:3] == LEFT[0].split()[:3], lines
        assert bc._succession(CORRECT, lines) == (11, 11)
        kept, pairs = bc._succession(INTERLEAVED, lines)
        assert pairs == 11 and kept == 0, (kept, pairs)
        assert bc._columns_merged(INTERLEAVED, pymupdf.open(p)[0]) and not bc._columns_merged(CORRECT, pymupdf.open(p)[0])
        s = os.path.join(td, "single.pdf")
        two_column_pdf(s, LEFT, RIGHT, single=True)
        with pymupdf.open(s) as doc:
            assert bc._two_column_lines(doc[0]) is None                  # one column: never judged


def test_merged_columns_are_re_read_from_blocks():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "paper.pdf")
        two_column_pdf(p, LEFT, RIGHT)
        md, log = run(p, [INTERLEAVED])
        pos = [md.index(line) for line in LEFT + RIGHT]
        assert max(pos[:12]) < min(pos[12:]), md                           # the whole left column first
        assert not any(f"{a} {b}" in md for a, b in zip(LEFT, RIGHT))
        assert "COLUMNS paper.pdf: 1/1 two-column page(s)" in log, log
        assert bc._pdf_method == "pdf+columns" and bc._pdf_garbled_pages == []


def test_correct_engine_order_is_kept():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "paper.pdf")
        two_column_pdf(p, LEFT, RIGHT)
        md, log = run(p, [CORRECT])
        assert md == bc.clean_md(CORRECT) and "COLUMNS" not in log and bc._pdf_method == ""


def test_too_few_pairs_are_not_judged():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "short.pdf")
        two_column_pdf(p, LEFT[:5], RIGHT[:5])                               # 4 pairs < COLUMN_MIN_PAIRS
        merged = "\n\n".join(f"{a} {b}" for a, b in zip(LEFT[:5], RIGHT[:5]))
        md, log = run(p, [merged])
        assert md == bc.clean_md(merged) and "COLUMNS" not in log


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
