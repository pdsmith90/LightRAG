#!/usr/bin/env python3
"""Behavioural tests for build_corpus._strip_back_index.
Run:  python -m pytest tests/corpus      (plain asserts; also runnable as a script)

BODY is book-sized against the index blocks, so an index at the end sits in the last
quarter of the document, as it does in real books. The index blocks use the shapes
pymupdf4llm produces from book PDFs: volume-and-page entries ("I 214; II 37"), page
running headers ("612 Subject Index"), letter groups (bold, underlined, or dashed as
"-G-"), an entry with sub-entries rendered as a heading, and back matter after the
index (a table of constants, a publisher's statement)."""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "corpus")
)
from build_corpus import _strip_back_index, clean_md

BODY = (
    "\n\n".join(
        f"## {k}. Job scheduling\n\nThe scheduler assigns each queued job to the worker with the "
        f"shortest expected completion time and rebalances the queues after every batch, as chapter {k} shows."
        for k in range(1, 61)
    )
    + "\n"
)
NAME_INDEX = """###### **Name Index**

Abara, T., I 52 Brandt, K. L., I 218 Castell, M. O., I 64 Dorsey, P., I 271
Ekwueme, N., I 9, 77; II 31 Farrow, D. A., I 12, 240 Hadley, S. V., II 146
"""
SUBJECT_INDEX = """###### **Subject Index**

Admission control, I 271 Batching, I 271 Backpressure, I 58; II 72
Cache eviction, II 23 Warm-up, II 114 Write-back, II 114
612 Subject Index
###### **<u>G</u>**
Garbage collection, I 322 Gang scheduling, I 325 Greedy dispatch, I 297
Graph partitioning, see also Load balancing
Subject Index 615
Queue depth, I 11; II 6 Round-robin dispatch, I 284, 291– 296
Shortest job first, I 8, 131–133, 302; II 97, 101
Work stealing, I 8, 176–201 Worker pools, I 180
"""


def book(*tail):
    return "# Scheduling Systems\n\n" + BODY + "\n" + "\n".join(tail)


def test_book_index_removed_body_kept():
    out = _strip_back_index(book(SUBJECT_INDEX))
    assert (
        "Subject Index" not in out
        and "Admission control" not in out
        and "Work stealing" not in out
    )
    assert out.count("rebalances the queues") == 60 and "chapter 60 shows" in out


def test_name_and_subject_index_both_removed():
    out = _strip_back_index(book(NAME_INDEX, SUBJECT_INDEX))
    assert (
        "Name Index" not in out
        and "Brandt, K. L." not in out
        and "Shortest job first, I 8" not in out
    )
    assert "chapter 60 shows" in out


def test_back_matter_after_index_kept():
    constants = (
        "###### **Conversion Factors**\n\n|Mebibyte|MiB|=|1048576 bytes|\n|---|---|---|---|\n"
        "|Kibibyte|KiB|=|1024 bytes|\n"
    )
    statement = (
        "#### **About the Series Editors**\n\nThe series editors welcome proposals for new "
        "volumes on every aspect of distributed systems and invite readers to write to them.\n"
    )
    out = _strip_back_index(book(SUBJECT_INDEX, constants))
    assert (
        "Admission control" not in out
        and "Conversion Factors" in out
        and "|Mebibyte|MiB|" in out
    )
    out = _strip_back_index(book(SUBJECT_INDEX, statement))
    assert "Admission control" not in out and "welcome proposals" in out


def test_subentry_heading_inside_index_removed():
    sub = "###### Scheduling\ncooperative, 142 fair-share, 38 preemptive, 61\nreal-time, 63, 177 Semaphores, 38\n"
    more = (
        "Thread pools, 402 Throughput, 9, 128, 154\nTimeouts, 266, 318, 322, 327, 349\n"
    )
    out = _strip_back_index(book(SUBJECT_INDEX, sub, more))
    assert (
        "fair-share" not in out and "Timeouts" not in out and "chapter 60 shows" in out
    )


def test_dashed_letter_group_continues():
    # "-G-" opens a group whose sub-entry lines mostly carry no page number, so only
    # the letter-group rule (punctuation dropped) keeps the region going past it.
    group = "### -G-\ngateway, 872\ngauges\nin load balancers 701-702\nconstraints\nlimits\n"
    out = _strip_back_index(book(SUBJECT_INDEX, group))
    assert "-G-" not in out and "gauges" not in out and "load balancers" not in out


def test_table_row_ends_region():
    table = "|Tier|Latency (ms)|Throughput (req/s)|\n|---|---|---|\n|Edge|12|4800|\n"
    out = _strip_back_index(book(SUBJECT_INDEX, table))
    assert "Admission control" not in out and "|Edge|12|4800|" in out


def test_index_heading_early_in_document_kept():
    doc = (
        "# Handbook\n\n## Index\n\nGarbage collection, I 322 Gang scheduling, I 325\n"
        "Queues, 12, 45\nPools, 7\nThreads, 209\nTimers, 83\n\n" + BODY
    )
    assert _strip_back_index(doc) == doc


def test_titles_that_only_contain_index_kept():
    for title in (
        "## Refractive Index",
        "## Index Terms",
        "## **Index construction**",
        "## Consumer price index 1990-2020",
    ):
        doc = book(
            title,
            "\n".join(f"Sample {k}: value {k * 3}, year {1990 + k}" for k in range(12)),
        )
        assert _strip_back_index(doc) == doc, title


def test_prose_under_an_index_heading_kept():
    prose = "\n\n".join(
        "The index is rebuilt every night from the transaction log; we model its growth "
        "with a simple linear approximation whose error budget is discussed at length in the text."
        for _ in range(8)
    )
    doc = book("## Index\n", prose)
    assert _strip_back_index(doc) == doc


def test_german_sachverzeichnis_removed():
    idx = "## Sachverzeichnis\n\nAlgorithmus, 12, 45\nDatenbank, 7, 88\nKompilierung, 15\nNetzwerk, 99, 101\nSpeicher, 3\n"
    out = _strip_back_index(book(idx))
    assert "Sachverzeichnis" not in out and "Datenbank" not in out


def test_clean_md_strips_index_and_is_idempotent():
    once = clean_md(book(NAME_INDEX, SUBJECT_INDEX))
    assert (
        "Baker, F. W." not in once
        and "Admission control" not in once
        and "chapter 60 shows" in once
    )
    assert clean_md(once) == once


def test_logs_the_document_only_during_a_build():
    import contextlib
    import io

    import build_corpus as bc

    doc = book(SUBJECT_INDEX)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bc._strip_back_index(doc)  # no document named: silent
    assert buf.getvalue() == ""
    bc._ref_doc = "ABCD1234__book.md"
    try:
        with contextlib.redirect_stdout(buf):
            bc._strip_back_index(doc)
            bc._strip_back_index(doc)  # clean_md runs twice on PDFs
    finally:
        bc._ref_doc = ""
    assert buf.getvalue().count("INDEX stripped ABCD1234__book.md") == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
