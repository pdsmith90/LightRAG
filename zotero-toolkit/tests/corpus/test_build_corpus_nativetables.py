#!/usr/bin/env python3
"""Behavioural tests for tables in clean_md (2026-10-19: native tables that are numeric only are dropped too).
Run:  .venv/bin/python test_build_corpus_nativetables.py     (plain asserts; pytest also collects it)
Every conversion path ends in clean_md, so there a table of numbers -- a pymupdf4llm pipe table from a PDF's own text
layer, a pandoc table, a VLM table -- becomes the one-line marker of _drop_numeric_tables, before the tag and <br>
passes. The tables that stay keep their cells apart: an HTML table becomes a pipe table (the tag pass used to run
its cells together -- a row "-8.9 | 5 | -0.3" read "-8.95-0.3"), and a <br> inside a pipe row becomes a space
instead of splitting the row.
Synthetic PDFs via pymupdf; pymupdf4llm replaced by a fake; all text is invented."""

import contextlib
import io
import os
import sys
import tempfile

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc
from test_build_corpus_vlmocr import fake_llm, patched, run_convert, text_pdf

PROSE = (
    "The invented survey measured the loading response of the crust at several stations and compared the "
    "observed displacements with the predictions of an elastic model. "
) * 4
NATIVE_NUMERIC = "|Degree|C|S|Sigma|\n|---|---|---|---|\n" + "".join(
    f"|{n}|0.{n:03d}1|-0.{n:03d}7|0.00{n % 9}|\n" for n in range(2, 30)
)
NATIVE_TEXT = (
    "|Station|Receiver|Antenna|\n|---|---|---|\n|INVA|Invented receiver model one|Choke ring|\n"
    "|INVB|Invented receiver model two|Dome<br>radome|\n|INVC|Invented receiver three|Patch|\n"
)
HTML_KEPT = (
    "<table><tr><th>Station<br>names</th><th>Coordinates</th><th>Min</th><th>Max</th></tr>"
    "<tr><td>Inventedville</td><td>(49°N, 3°W)</td><td>-8.9</td><td>5</td></tr>"
    "<tr><td>Otherplace</td><td>(51°N, 1°W)</td><td>-8.5</td><td>4.8</td></tr></table>"
)


def test_native_numeric_table_is_dropped_in_a_conversion():
    with (
        tempfile.TemporaryDirectory() as td,
        patched(bc, "CFG", {}),
        fake_llm(
            [
                PROSE
                + "\n\nTable 2. Invented coefficients.\n\n"
                + NATIVE_NUMERIC
                + "\n\n"
                + PROSE
            ]
        ),
    ):
        method, body, log = run_convert(
            td, lambda p: text_pdf(p, [PROSE[:90]] * 30), ocr="off", ocr_vlm="off"
        )
    assert (
        method == "pdf" and "Invented coefficients" in body and "0.0071" not in body
    ), (method, body[-400:])
    assert (
        "[Table of numbers omitted: 29 rows x 4 columns; header: Degree | C | S | Sigma]"
        in body
    ), body
    assert "NUMERIC-TABLES 0VLM1OCR__scan.md: 1 table(s) of numbers dropped" in log, log


def test_native_text_table_stays_with_its_rows_whole():
    md = bc.clean_md(PROSE + "\n\n" + NATIVE_TEXT)
    assert "|INVB|Invented receiver model two|Dome radome|" in md, (
        md
    )  # <br> inside a row: a space, one line
    assert md.count("\n|") == 5, md  # header, separator, three rows


def test_html_table_becomes_a_pipe_table():
    md = bc.clean_md(PROSE + "\n\n" + HTML_KEPT)
    assert (
        "| Station names | Coordinates | Min | Max |" in md
        and "|---|---|---|---|" in md
    ), md
    assert "| Inventedville | (49°N, 3°W) | -8.9 | 5 |" in md and "-8.95" not in md, md
    assert "<t" not in md, md


def test_vlm_numeric_html_table_is_dropped_by_clean_md_too():
    rows = "".join(
        f"<tr><td>{n}</td><td>0.{n}1</td><td>0.{n}2</td></tr>" for n in range(1, 9)
    )
    md = bc.clean_md(
        "Table 1. Invented.\n\n<table><tr><td>n</td><td>a</td><td>b</td></tr>"
        + rows
        + "</table>"
    )
    assert (
        md.startswith("Table 1. Invented.")
        and "[Table of numbers omitted: 9 rows x 3 columns; header: n | a | b]" in md
    )


def test_clean_md_is_idempotent_on_tables():
    doc = PROSE + "\n\n" + NATIVE_TEXT + "\n\n" + HTML_KEPT + "\n\n" + NATIVE_NUMERIC
    once = bc.clean_md(doc)
    assert bc.clean_md(once) == once
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        bc.clean_md(once)
    assert "NUMERIC-TABLES" not in log.getvalue()  # a marker is not a table


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
