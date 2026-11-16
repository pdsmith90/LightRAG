"""get_ocr_progress_lines: the OCR line the Pipeline Status dialog shows.

The build leaves several traces in its bundle directory: a running marker, one
page cache per document in flight, its own build totals, a log of finished
documents and a list of deferred ones. Only a page cache being appended to right
now means OCR is running; the other four outlive the build by days and describe
work that is not happening now, so they are never reported. With no document in
flight the answer is an empty list.

Best effort throughout: a missing, half-written, stale or unparseable page cache
costs the line, never the call. Fixtures are invented; no Zotero, PDF or OCR
service is involved.

``ocr_line_to_append`` decides what the endpoint writes into the shared pipeline
history: nothing while no document is in flight, the line once when it changes,
and nothing on the next poll while it stays the same.
"""

import json
import os
import time

import pytest

from lightrag.zotero_ocr_progress import (
    OCR_LINE_PREFIX,
    _clear_cache,
    _short_name,
    get_ocr_progress_lines,
    ocr_line_to_append,
)

pytestmark = pytest.mark.offline

NOW = 1_700_000_000.0

INFLIGHT_LINE = "OCR in progress: scan: page 4 of 10 (40%)"
EXISTING = "Starting document processing"


@pytest.fixture(autouse=True)
def _no_cached_answers():
    """The module caches per bundle for 5 s; a reused tmp path must not leak."""
    _clear_cache()
    yield
    _clear_cache()


def _page_cache(bundle, name, pages, done, path="/pdfs/scan.pdf"):
    """Write a page cache the way the build does: header, then one line per page."""
    cache_dir = bundle / "ocr_vlm_pages"
    cache_dir.mkdir(exist_ok=True)
    f = cache_dir / name
    header = json.dumps(
        {"v": 1, "path": path, "size": 1234, "pages": pages, "dpi": 200}
    )
    lines = [header] + [json.dumps({"i": i, "text": f"page {i}"}) for i in range(done)]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _totals(bundle, **fields):
    """Write the converter's ``ocr_build_progress.json`` for this build."""
    body = {
        "started": "2026-11-15T12:30:45Z",
        "updated": "2026-11-15T13:12:45Z",
        "docs_done": 4,
        "pages_done": 130,
        "ocr_seconds": 2530,
        "last_doc": "scan.pdf",
    }
    body.update(fields)
    f = bundle / "ocr_build_progress.json"
    f.write_text(json.dumps(body), encoding="utf-8")
    return f


def _age(f, mtime):
    os.utime(f, (mtime, mtime))


def _inflight_cache(bundle, pages=10, done=4, path="/pdfs/scan.pdf"):
    """A page cache appended to seconds ago: what says OCR is running right now."""
    f = _page_cache(
        bundle, "papers-0123456789.jsonl", pages=pages, done=done, path=path
    )
    _age(f, NOW)
    return f


def test_empty_bundle_has_nothing_to_say(tmp_path):
    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_in_flight_document_reports_page_and_percent(tmp_path):
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=4)
    _age(f, NOW)

    lines = get_ocr_progress_lines(tmp_path, now=NOW)

    assert lines == [INFLIGHT_LINE]


def test_in_flight_progress_never_exceeds_all_the_pages(tmp_path):
    # A retried page is appended twice; 5 > 3 must not read as 166%.
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=3, done=5)
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR in progress: scan: page 3 of 3 (100%)"
    ]


def test_in_flight_document_goes_quiet_after_fifteen_minutes(tmp_path):
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=4)
    _age(f, NOW - 10000)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_newest_page_cache_is_the_one_in_flight(tmp_path):
    old = _page_cache(
        tmp_path, "old-aaaaaaaaaa.jsonl", pages=8, done=2, path="/pdfs/old.pdf"
    )
    new = _page_cache(
        tmp_path, "new-bbbbbbbbbb.jsonl", pages=5, done=3, path="/pdfs/new.pdf"
    )
    _age(old, NOW - 600)
    _age(new, NOW - 30)

    lines = get_ocr_progress_lines(tmp_path, now=NOW)

    assert lines == ["OCR in progress: new: page 3 of 5 (60%)"]


def test_equation_pass_cache_is_not_a_document(tmp_path):
    f = _page_cache(tmp_path, "papers-0123456789.eq.jsonl", pages=10, done=4)
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_bundle_dir_comes_from_the_environment_when_not_given(tmp_path, monkeypatch):
    _inflight_cache(tmp_path)
    monkeypatch.setenv("ZOTERO_OCR_BUNDLE", str(tmp_path))

    assert get_ocr_progress_lines(now=NOW) == [INFLIGHT_LINE]


def test_an_empty_bundle_env_var_falls_back_to_the_working_directory(
    tmp_path, monkeypatch
):
    _inflight_cache(tmp_path)
    monkeypatch.setenv("ZOTERO_OCR_BUNDLE", "")
    monkeypatch.chdir(tmp_path)

    assert get_ocr_progress_lines(now=NOW) == [INFLIGHT_LINE]


def test_an_unreadable_page_cache_means_nothing_is_running(tmp_path):
    # The newest cache cannot be read, so no document is provably in flight.
    (tmp_path / "ocr_vlm_pages").mkdir()
    f = tmp_path / "ocr_vlm_pages" / "torn-0123456789.jsonl"
    f.write_bytes(b'{"path": "/pdfs/torn.pdf", "pages"\n{"i": 0}\n{"i": 1}\n')
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_header_without_a_page_count_is_skipped(tmp_path):
    (tmp_path / "ocr_vlm_pages").mkdir()
    f = tmp_path / "ocr_vlm_pages" / "odd-0123456789.jsonl"
    f.write_text(json.dumps({"path": "/pdfs/odd.pdf"}) + "\n{}\n", encoding="utf-8")
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_only_an_in_flight_page_cache_produces_lines(tmp_path):
    # Every build artifact at once -- running marker, totals, log, deferred list
    # -- all fresh, plus a document being converted. Only the document is worth
    # a line: the others describe work that is not happening now.
    (tmp_path / ".corpus_build_running").write_text("", encoding="utf-8")
    _totals(tmp_path)
    (tmp_path / "vm_weekly_update.log").write_text(
        "[build_corpus] VLM-OCR done.pdf: 7 page(s) via pepperocr-vl (remote) in 70 s\n",
        encoding="utf-8",
    )
    (tmp_path / "ocr_vlm_deferred.txt").write_text("wait.pdf\n", encoding="utf-8")
    f = _inflight_cache(tmp_path)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [INFLIGHT_LINE]

    # The same bundle, once its only page cache has stopped being appended to.
    _age(f, NOW - 10000)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_two_polls_within_five_seconds_serve_the_cached_answer(tmp_path):
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=4)
    _age(f, time.time())
    first = get_ocr_progress_lines(tmp_path)

    _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=7)
    _age(f, time.time())
    second = get_ocr_progress_lines(tmp_path)

    assert first == [INFLIGHT_LINE]
    assert second == first


# ---------------------------------------------------------------------------
# ocr_line_to_append: one history entry per change, in time order
# ---------------------------------------------------------------------------


def test_the_first_line_of_a_run_goes_into_an_empty_history():
    assert ocr_line_to_append([INFLIGHT_LINE], []) == INFLIGHT_LINE


def test_the_same_line_is_not_written_twice():
    assert ocr_line_to_append([INFLIGHT_LINE], [INFLIGHT_LINE]) is None


def test_pipeline_messages_below_the_line_do_not_make_it_write_again():
    # The line is no longer the tail; the newest OCR entry alone decides.
    history = [EXISTING, INFLIGHT_LINE, "Processing file 2/9: other.pdf"]
    assert ocr_line_to_append([INFLIGHT_LINE], history) is None


def test_a_changed_line_is_written_after_the_newest_entry():
    history = [INFLIGHT_LINE, "Processing file 2/9: other.pdf"]
    next_line = "OCR in progress: scan: page 5 of 10 (50%)"

    assert ocr_line_to_append([next_line], history) == next_line


def test_only_the_newest_ocr_entry_is_compared():
    # A history that was cleared and reprinted leaves an older OCR entry above
    # the current page; the current page is still a change.
    history = [
        INFLIGHT_LINE,
        "Processing file 2/9: other.pdf",
        "OCR in progress: scan: page 1 of 10 (10%)",
    ]

    assert ocr_line_to_append([INFLIGHT_LINE], history) == INFLIGHT_LINE


def test_a_history_without_an_ocr_entry_gets_the_line():
    assert ocr_line_to_append([INFLIGHT_LINE], [EXISTING]) == INFLIGHT_LINE


def test_nothing_in_flight_writes_nothing():
    assert ocr_line_to_append([], [INFLIGHT_LINE]) is None
    assert ocr_line_to_append([], []) is None


def test_every_line_carries_the_prefix_the_writer_looks_for(tmp_path):
    # _inflight builds the text from OCR_LINE_PREFIX; if the two drift apart the
    # writer stops recognising its own entries and re-appends the same line.
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=4)
    _age(f, NOW)
    (line,) = get_ocr_progress_lines(tmp_path, now=NOW)

    assert line.startswith(OCR_LINE_PREFIX)
    assert ocr_line_to_append([line], [line]) is None


# ---------------------------------------------------------------------------
# _short_name: author and year, not the title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basename, expected",
    [
        # The shape Zotero writes, and the reason the title is dropped: it would
        # push the dialog's one line past its width.
        (
            "Backus et al. - 2005 - Foundations of Geomagnetism.pdf",
            "Backus et al. 2005",
        ),
        # A title may itself contain ' - ' and ':', which must not be mistaken
        # for the separators, and never reaches the answer anyway.
        ("Smith - 2019 - Ocean - Atmosphere: A - B Study.pdf", "Smith 2019"),
        ("Иванов - 2012 - Некоторые вопросы теории.pdf", "Иванов 2012"),
        # Several works in one year carry a letter suffix on the year.
        ("Doe - 2005a - Second Paper.pdf", "Doe 2005a"),
        # Not that shape: keep the name, but still drop the extension.
        ("Remote sensing method [E9TAYZ6G].pdf", "Remote sensing method [E9TAYZ6G]"),
        ("Scan - draft.pdf", "Scan - draft"),
        ("report.PDF", "report"),
    ],
)
def test_short_name_keeps_author_and_year_only(basename, expected):
    assert _short_name(basename) == expected


def test_a_zotero_name_reaches_the_line_shortened(tmp_path):
    path = "/pdfs/Backus et al. - 2005 - Foundations of Geomagnetism.pdf"
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=384, done=191, path=path)
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR in progress: Backus et al. 2005: page 191 of 384 (49%)"
    ]
