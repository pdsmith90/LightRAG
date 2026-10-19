#!/usr/bin/env python3
"""Behavioural tests for VLM OCR tables (2026-10-19: tables of numbers are not worth importing; the table markup
fixed). Run:  .venv/bin/python test_build_corpus_vlmtables.py     (plain asserts; pytest too)
1. The page request bans the "style" token (pepperocr-vl wrote style='text-align: center; word-wrap: break-word;' on
   every table cell, ~70 % of a table page) and gives the remote lane OCR_VLM_REMOTE_MAX_TOKENS (its context is 16k;
   the local server's 8k keeps OCR_VLM_MAX_TOKENS).
2. A table whose data cells are numbers (NUMERIC_TABLE_SHARE) is dropped from the page, its caption kept,
   a one-line marker in its place; tables with words or formulas stay.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP calls, Tesseract and ocrmypdf replaced by fakes; all text is invented."""

import contextlib
import io
import json
import os
import sys
import tempfile
import urllib.request

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc
from test_build_corpus_vlmocr import (
    VLM_PAGE,
    fake_llm,
    fake_ocrmypdf,
    image_only_pdf,
    patched,
    run_convert,
    vlm,
)

CELL = "<td style='text-align: center; word-wrap: break-word;'>"
NUMERIC = (
    "<table border=1><tr>"
    + "".join(f"{CELL}{h}</td>" for h in ("n", "m", "c", "$\\sigma(c)$"))
    + "</tr>"
    + "".join(
        f"<tr>{CELL}{n}</td>{CELL}{n % 3}</td>{CELL}$-0.{n:04d}$</td>{CELL}0.{n + 7:03d}</td></tr>"
        for n in range(1, 30)
    )
    + "</table>"
)
LABELLED = (
    '<table><tr><td></td><td colspan="4">$\\sigma = 0.125$</td></tr><tr><td>REPL</td><td>ISUBM</td>'
    "<td>ISUBV</td><td>ISUBM</td><td>ISUBV</td></tr>"
    + "".join(
        f'<tr><td rowspan="2">n = {32 * k}</td><td>1.{k}21</td><td>1.{k}04</td><td>1.6{k}</td>'
        f"<td>1.0{k}</td></tr><tr><td>1.2{k}</td><td>1.1{k}</td><td>1.8{k}</td><td>1.3{k}</td></tr>"
        for k in range(1, 8)
    )
    + "</table>"
)
STATIONS = (
    "<table><tr><td>Station</td><td>Coordinates</td><td>Years</td><td>Mean</td></tr>"
    + "".join(
        f"<tr><td>Invented{n}ville</td><td>(4{n}°N, {n}°W)</td><td>1971-1984</td><td>-{n}.5</td></tr>"
        for n in range(1, 9)
    )
    + "</table>"
)
FORMULAS = (
    "<table><tr><td>Algorithm</td><td>Count</td></tr>"
    + "".join(
        f"<tr><td>({n})</td><td>$({n}n^2 + 9n + 4.5)m$</td></tr>" for n in range(1, 8)
    )
    + "</table>"
)
PIPE_NUMERIC = "| t | 05 | 10 | 15 |\n|---|---|---|---|\n" + "".join(
    f"| {n} | 0.{n}1 | 0.{n}2 | 0.{n}3 |\n" for n in range(1, 9)
)
PIPE_TEXT = "| Symbol | Meaning |\n|---|---|\n| $a$ | invented semi-major axis |\n| $e$ | invented eccentricity |\n| $i$ | inclination |\n"


@contextlib.contextmanager
def own_cfg():
    with patched(bc, "CFG", {}):
        yield


def test_numeric_tables_are_dropped():
    for table in (NUMERIC, LABELLED, PIPE_NUMERIC):
        md, n = bc._drop_numeric_tables(
            "Table 15. Invented harmonic coefficients.\n\n"
            + table
            + "\n\nAfter the table."
        )
        assert (
            n == 1
            and "Invented harmonic coefficients" in md
            and "After the table." in md
        ), (table[:40], md)
        assert "[Table of numbers omitted:" in md and "0.0" not in md.replace(
            "Table 15", ""
        ), md


def test_tables_with_words_or_formulas_stay():
    for table in (
        STATIONS,
        FORMULAS,
        PIPE_TEXT,
        "<table><tr><td>n</td></tr><tr><td>1</td><td>2</td></tr></table>",
    ):
        md, n = bc._drop_numeric_tables("Before.\n\n" + table)
        assert n == 0 and table in md, table[:60]


def test_marker_keeps_the_shape_and_header():
    md, n = bc._drop_numeric_tables(NUMERIC)
    assert (
        n == 1
        and md.strip()
        == "[Table of numbers omitted: 30 rows x 4 columns; header: n | m | c | $\\sigma(c)$]"
    ), md


def test_named_rows_keep_a_table():
    """2026-10-19: only pure number grids go -- a table that names its rows (models, regions) stays."""
    models = (
        "<table><tr><td>Model</td><td>M2</td><td>K1</td><td>O1</td></tr>"
        + "".join(
            f"<tr><td>INV{k}.{k}b</td><td>2.{k}</td><td>1.{k}</td><td>0.{k}</td></tr>"
            for k in range(1, 8)
        )
        + "</table>"
    )
    toc = (
        "<table>"
        + "".join(
            f"<tr><td>{k}. Invented chapter {k}</td><td>{9 * k}</td></tr>"
            for k in range(1, 9)
        )
        + "</table>"
    )
    indexed = (
        "<table><tr><td>k</td><td>a</td><td>b</td></tr>"
        + "".join(
            f"<tr><td>{k}</td><td>0.{k}1</td><td>-0.{k}2</td></tr>" for k in range(1, 8)
        )
        + "</table>"
    )
    assert bc._drop_numeric_tables(models)[1] == 0  # model names label the rows
    assert bc._drop_numeric_tables(toc)[1] == 0  # chapter titles
    assert (
        bc._drop_numeric_tables(indexed)[1] == 1
    )  # rows indexed by numbers: a pure grid


def test_page_request_bans_style_and_sizes_the_budget_by_lane():
    seen = []

    def urlopen(req, timeout=None):
        seen.append((json.loads(req.data), timeout))
        return contextlib.closing(
            io.BytesIO(
                b'data: {"choices": [{"delta": {"content": "A page."}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n'
            )
        )

    with (
        patched(urllib.request, "urlopen", urlopen),
        patched(bc, "OCR_VLM_REMOTE_URL", "http://127.0.0.1:18000"),
        patched(bc, "OCR_VLM_LOCAL_URL", "http://127.0.0.1:19530"),
    ):  # distinct even under the :9 test guard
        bc._vlm_ocr_page(bc.OCR_VLM_REMOTE_URL, b"\x89PNG")
        bc._vlm_ocr_page(bc.OCR_VLM_LOCAL_URL, b"\x89PNG")
    (remote, t1), (local, _) = seen
    assert (
        remote["max_tokens"] == bc.OCR_VLM_REMOTE_MAX_TOKENS == 14000
        and local["max_tokens"] == bc.OCR_VLM_MAX_TOKENS
    )
    assert [" style", False] in remote["logit_bias"] and ["style", False] in local[
        "logit_bias"
    ], remote["logit_bias"]
    assert (
        t1 == bc.OCR_VLM_PAGE_TIMEOUT_S >= 900
    )  # a queue wait behind a 14k-token table page


def test_numeric_tables_are_dropped_in_a_conversion():
    page = VLM_PAGE + "\n\nTable 3. Invented coefficients of the model.\n\n" + NUMERIC
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(page_fn=lambda b, p: page),
        fake_ocrmypdf(available=False),
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert (
        method == "pdf+vlmocr"
        and "Invented coefficients" in body
        and "-0.0012" not in body
    ), (method, body[:300])
    assert "; 2 numeric table(s) dropped" in log, log


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
