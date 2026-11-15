"""get_ocr_progress_lines: the OCR lines the Pipeline Status dialog shows.

The build leaves five traces in its bundle directory -- a running marker, one
page cache per document in flight, its own build totals, a log of finished
documents and a list of deferred ones. This module turns them into a few plain
lines, best effort: a missing, half-written or stale file costs its own line,
never the call. Fixtures are invented; no Zotero, PDF or OCR service is involved.
"""

import json
import os

import pytest

from lightrag.zotero_ocr_progress import _clear_cache, get_ocr_progress_lines

pytestmark = pytest.mark.offline

NOW = 1_700_000_000.0


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


def test_empty_bundle_has_nothing_to_say(tmp_path):
    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_running_marker_comes_first(tmp_path):
    (tmp_path / ".corpus_build_running").write_text("", encoding="utf-8")
    assert get_ocr_progress_lines(tmp_path, now=NOW) == ["OCR: corpus build running"]


def test_a_leaked_running_marker_goes_quiet_after_twelve_hours(tmp_path):
    marker = tmp_path / ".corpus_build_running"
    marker.write_text("", encoding="utf-8")
    _age(marker, NOW - 13 * 3600)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_in_flight_document_reports_page_and_percent(tmp_path):
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=10, done=4)
    _age(f, NOW)

    lines = get_ocr_progress_lines(tmp_path, now=NOW)

    assert lines == ["OCR in progress: scan.pdf: page 4 of 10 (40%)"]


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


def test_build_totals_report_what_this_build_has_done(tmp_path):
    _totals(tmp_path)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR this build: 4 document(s), 130 page(s), 42 min of OCR "
        "(started 2026-11-15 12:30)"
    ]


def test_build_totals_without_a_start_time_drop_the_parenthesis(tmp_path):
    _totals(tmp_path, started="not a timestamp")

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR this build: 4 document(s), 130 page(s), 42 min of OCR"
    ]


def test_build_totals_without_docs_or_pages_say_nothing(tmp_path):
    _totals(tmp_path, docs_done="4")

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_a_week_old_build_totals_file_says_nothing(tmp_path):
    f = _totals(tmp_path)
    _age(f, NOW - 8 * 86400)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_corrupt_build_totals_say_nothing(tmp_path):
    (tmp_path / "ocr_build_progress.json").write_text(
        '{"docs_done": 4', encoding="utf-8"
    )

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_bundle_dir_comes_from_the_environment_when_not_given(tmp_path, monkeypatch):
    (tmp_path / ".corpus_build_running").write_text("", encoding="utf-8")
    monkeypatch.setenv("ZOTERO_OCR_BUNDLE", str(tmp_path))

    assert get_ocr_progress_lines(now=NOW) == ["OCR: corpus build running"]


def test_an_empty_bundle_env_var_falls_back_to_the_working_directory(
    tmp_path, monkeypatch
):
    (tmp_path / "ocr_vlm_deferred.txt").write_text("wait.pdf\n", encoding="utf-8")
    monkeypatch.setenv("ZOTERO_OCR_BUNDLE", "")
    monkeypatch.chdir(tmp_path)

    assert get_ocr_progress_lines(now=NOW) == [
        "OCR deferred: 1 document(s) waiting for a free OCR lane"
    ]


def test_last_three_finished_documents_oldest_first(tmp_path):
    log = tmp_path / "vm_weekly_update.log"
    log.write_text(
        "\n".join(
            f"[build_corpus] VLM-OCR {name}.pdf: {n} page(s) via pepperocr-vl "
            f"(remote) in {n * 10} s"
            for name, n in (("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5))
        )
        + "\n[build_corpus] done: 5 documents\n",
        encoding="utf-8",
    )

    lines = get_ocr_progress_lines(tmp_path, now=NOW)

    assert lines == [
        "OCR done: c.pdf: 3 page(s) via pepperocr-vl (remote) in 30 s",
        "OCR done: d.pdf: 4 page(s) via pepperocr-vl (remote) in 40 s",
        "OCR done: e.pdf: 5 page(s) via pepperocr-vl (remote) in 50 s",
    ]


def test_a_log_nobody_touched_for_a_week_says_nothing(tmp_path):
    log = tmp_path / "vm_weekly_update.log"
    log.write_text(
        "[build_corpus] VLM-OCR old.pdf: 7 page(s) via pepperocr-vl (remote) in 70 s\n",
        encoding="utf-8",
    )
    _age(log, NOW - 8 * 86400)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_a_finished_line_with_a_tail_still_matches(tmp_path):
    log = tmp_path / "vm_weekly_update.log"
    log.write_text(
        "[build_corpus] VLM-OCR scan.pdf: 12 page(s) via pepperocr-vl (local) "
        "in 900 s; 2 page(s) by tesseract\n",
        encoding="utf-8",
    )

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR done: scan.pdf: 12 page(s) via pepperocr-vl (local) in 900 s"
    ]


def test_a_cut_first_line_does_not_hide_the_matches_after_it(tmp_path):
    # The tail read starts mid-line when the log is bigger than the window.
    log = tmp_path / "vm_weekly_update.log"
    log.write_text(
        "[build_corpus] VLM-OCR cut.pdf: 12 pa\n"
        "[build_corpus] VLM-OCR later.pdf: 4 page(s) via pepperocr-vl (remote) "
        "in 40 s\n",
        encoding="utf-8",
    )

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR done: later.pdf: 4 page(s) via pepperocr-vl (remote) in 40 s"
    ]


def test_names_with_colons_and_cyrillic_survive(tmp_path):
    # Non-ASCII and ": " are both common in a paper title, and the line format
    # separates the name from the count with a colon.
    f = _page_cache(
        tmp_path,
        "ru-0123456789.jsonl",
        pages=6,
        done=2,
        path="/pdfs/Квантовая механика: часть 2.pdf",
    )
    _age(f, NOW)
    (tmp_path / "vm_weekly_update.log").write_text(
        "[build_corpus] VLM-OCR Отчёт: том 1.pdf: 3 page(s) via pepperocr-vl "
        "(remote) in 30 s\n",
        encoding="utf-8",
    )

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR in progress: Квантовая механика: часть 2.pdf: page 2 of 6 (33%)",
        "OCR done: Отчёт: том 1.pdf: 3 page(s) via pepperocr-vl (remote) in 30 s",
    ]


def test_deferred_documents_are_counted(tmp_path):
    (tmp_path / "ocr_vlm_deferred.txt").write_text(
        "AAAA0001__one.pdf\n\nBBBB0002__two.pdf\nCCCC0003__three.pdf\n",
        encoding="utf-8",
    )

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR deferred: 3 document(s) waiting for a free OCR lane"
    ]


def test_empty_deferred_list_says_nothing(tmp_path):
    (tmp_path / "ocr_vlm_deferred.txt").write_text("\n\n", encoding="utf-8")

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_unreadable_header_costs_only_its_own_line(tmp_path):
    (tmp_path / "ocr_vlm_pages").mkdir()
    f = tmp_path / "ocr_vlm_pages" / "torn-0123456789.jsonl"
    f.write_bytes(b'{"path": "/pdfs/torn.pdf", "pages"\n{"i": 0}\n{"i": 1}\n')
    _age(f, NOW)
    (tmp_path / "ocr_vlm_deferred.txt").write_text("x.pdf\n", encoding="utf-8")

    lines = get_ocr_progress_lines(tmp_path, now=NOW)

    assert lines == ["OCR deferred: 1 document(s) waiting for a free OCR lane"]


def test_header_without_a_page_count_is_skipped(tmp_path):
    (tmp_path / "ocr_vlm_pages").mkdir()
    f = tmp_path / "ocr_vlm_pages" / "odd-0123456789.jsonl"
    f.write_text(json.dumps({"path": "/pdfs/odd.pdf"}) + "\n{}\n", encoding="utf-8")
    _age(f, NOW)

    assert get_ocr_progress_lines(tmp_path, now=NOW) == []


def test_all_sources_in_order(tmp_path):
    (tmp_path / ".corpus_build_running").write_text("", encoding="utf-8")
    f = _page_cache(tmp_path, "papers-0123456789.jsonl", pages=4, done=1)
    _age(f, NOW)
    _totals(tmp_path, docs_done=2, pages_done=9, ocr_seconds=600)
    (tmp_path / "vm_weekly_update.log").write_text(
        "[build_corpus] VLM-OCR done.pdf: 7 page(s) via pepperocr-vl (remote) in 70 s\n",
        encoding="utf-8",
    )
    (tmp_path / "ocr_vlm_deferred.txt").write_text("wait.pdf\n", encoding="utf-8")

    assert get_ocr_progress_lines(tmp_path, now=NOW) == [
        "OCR: corpus build running",
        "OCR in progress: scan.pdf: page 1 of 4 (25%)",
        "OCR this build: 2 document(s), 9 page(s), 10 min of OCR "
        "(started 2026-11-15 12:30)",
        "OCR done: done.pdf: 7 page(s) via pepperocr-vl (remote) in 70 s",
        "OCR deferred: 1 document(s) waiting for a free OCR lane",
    ]


def test_two_polls_within_five_seconds_serve_the_cached_answer(tmp_path):
    deferred = tmp_path / "ocr_vlm_deferred.txt"
    deferred.write_text("one.pdf\n", encoding="utf-8")
    first = get_ocr_progress_lines(tmp_path)

    deferred.write_text("one.pdf\ntwo.pdf\n", encoding="utf-8")
    second = get_ocr_progress_lines(tmp_path)

    assert first == ["OCR deferred: 1 document(s) waiting for a free OCR lane"]
    assert second == first
