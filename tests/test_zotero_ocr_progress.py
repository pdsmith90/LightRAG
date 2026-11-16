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
"""

import json
import os
import time

import pytest

from lightrag.zotero_ocr_progress import _clear_cache, get_ocr_progress_lines

pytestmark = pytest.mark.offline

NOW = 1_700_000_000.0

INFLIGHT_LINE = "OCR in progress: scan.pdf: page 4 of 10 (40%)"


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
        "OCR in progress: scan.pdf: page 3 of 3 (100%)"
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

    assert lines == ["OCR in progress: new.pdf: page 3 of 5 (60%)"]


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
