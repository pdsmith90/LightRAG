#!/usr/bin/env python3
"""Behavioural tests for the VLM page cache (2026-11-10): a document longer than its lane's day runs in passes.
Run:  .venv/bin/python test_build_corpus_vlmresume.py     (plain asserts; pytest also collects it)
A 988-page scan needed more hours than the remote lane is open in a day, and a lane that went down mid-book threw
away every page done. Each finished page is now appended to a per-document cache; a document on the remote lane
pauses when its quiet window begins, one whose lane goes down mid-book is deferred, and the next pass resumes at the
first page not done.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP calls, Tesseract and ocrmypdf replaced by fakes; text invented."""

import contextlib
import json
import os
import sys
import tempfile
import urllib.error

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc
from test_build_corpus_vlmocr import (
    OCRMYPDF_LINE,
    VLM_PAGE,
    fake_llm,
    fake_ocrmypdf,
    image_only_pdf,
    no_vlm,
    patched,
    run_convert,
    vlm,
)


def marked(k):
    return VLM_PAGE + f"\n\nInvented marker {k:03d} on this page."


def counter(down_after=None):
    """A VLM page function answering marked(1), marked(2), ... across passes (state shared through the dict it
    returns); with down_after=k, the call after page k raises a refused connection, as a lane that went down."""
    state = {"page": 0, "down_after": down_after}

    def page_fn(base, png):
        if state["down_after"] is not None and state["page"] == state["down_after"]:
            raise urllib.error.URLError(
                ConnectionRefusedError(111, "Connection refused")
            )
        state["page"] += 1
        return marked(state["page"])

    return state, page_fn


def quiet_after(k):
    """bc._vlm_quiet replaced: the checks before the first k pages find the remote lane open, the next its quiet
    window. Enter it after vlm(), which keeps the window shut."""
    seq = [False] * k
    return patched(bc, "_vlm_quiet", lambda now=None: seq.pop(0) if seq else True)


def cache_lines(cache_dir):
    files = [f for f in os.listdir(cache_dir) if f.endswith(".jsonl")]
    assert len(files) == 1, files
    with open(os.path.join(cache_dir, files[0]), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f.read().splitlines()]


def in_order(body, pages):
    at = [body.find(f"marker {k:03d}") for k in range(1, pages + 1)]
    return (
        all(a >= 0 for a in at)
        and at == sorted(at)
        and all(body.count(f"marker {k:03d}") == 1 for k in range(1, pages + 1))
    )


def five_pages(p):
    image_only_pdf(p, pages=5)


def keep(p):
    """The source as the earlier pass left it (same file, same mtime): the cache must match it."""


def test_a_lane_that_goes_down_mid_book_resumes_where_it_stopped():
    state, page_fn = counter(down_after=2)
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, five_pages)
        assert (
            method.startswith("vlm_deferred:") and "went down after 2 page(s)" in method
        ), method
        assert "resumes at page 3" in method and not body and len(calls) == 3, (
            method,
            calls,
        )
        assert "VLM-OCR deferred scan.pdf" in log and not runs, (log, runs)
        lines = cache_lines(cache_dir)
        assert lines[0]["pages"] == 5 and [r["i"] for r in lines[1:]] == [0, 1], lines
        assert "marker 002" in lines[2]["text"], lines[2]
        state["down_after"] = None
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 3 and not runs, (
            method,
            calls,
            runs,
        )
        assert in_order(body, 5), body
        assert "VLM-OCR scan.pdf: resuming at page 3 of 5 (remote)" in log, log
        assert "; resumed at page 3" in log, log
        assert not os.listdir(cache_dir), os.listdir(
            cache_dir
        )  # written: the cache goes


def test_a_book_pauses_when_the_quiet_window_begins_and_resumes():
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            method, body, log = run_convert(td, five_pages)
        assert method.startswith(
            "vlm_deferred:5 page(s), 2 done; paused as the remote lane's quiet window"
        ), method
        assert "resumes at page 3" in method and len(calls) == 2 and not runs, (
            method,
            calls,
            runs,
        )
        assert [r["i"] for r in cache_lines(cache_dir)[1:]] == [0, 1]
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 3 and in_order(body, 5), (
            method,
            calls,
            body,
        )
        assert not os.listdir(cache_dir)


def test_a_local_lane_never_pauses_for_the_remote_quiet_window():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="local") as calls,
            patched(bc, "_vlm_quiet", lambda now=None: True),
        ):
            method, body, log = run_convert(td, five_pages)
    assert method == "pdf+vlmocr" and len(calls) == 5 and not runs, (
        method,
        calls,
        runs,
    )


def test_passes_that_keep_no_new_page_defer_a_few_times_then_go_to_ocrmypdf():
    """A page that takes the server down every time: the pass that resumes on it fails at once, and so does each one
    after. OCR_VLM_MAX_STALLS such passes in a row defer; the next abandons and drops the cache."""
    top = bc.OCR_VLM_MAX_STALLS
    state, page_fn = counter(down_after=2)
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
    ):
        for k in range(top + 2):
            with (
                vlm(kind="remote", page_fn=page_fn),
                patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            ):
                method, body, log = run_convert(td, keep if k else five_pages)
            if k == 0:
                assert "went down after 2 page(s)" in method, method
                assert "pass(es) in a row" not in method, method
            elif k <= top:
                assert method.startswith("vlm_deferred:") and not runs, (k, method)
                assert "resumes at page 3" in method, method
                assert f"; {k} of {top} pass(es) in a row with no new page" in method, (
                    method
                )
        assert method == "pdf+ocr" and runs, (method, runs)
        assert f"{top + 1} pass(es) in a row with no new page kept" in log, log
        assert "VLM OCR abandoned on scan.pdf after 2 page(s)" in log, log
        assert not os.listdir(cache_dir), os.listdir(cache_dir)
    state, page_fn = counter(
        down_after=0
    )  # a fresh document whose first page is refused
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, five_pages)
    assert method.startswith("vlm_deferred:") and not runs, (method, runs)
    assert f"resumes at page 1; 1 of {top} pass(es)" in method, method


def test_a_pass_that_keeps_a_page_starts_the_count_over():
    top = bc.OCR_VLM_MAX_STALLS
    state, page_fn = counter(down_after=1)
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):

        def one_pass(builder=keep, quiet=None):
            with contextlib.ExitStack() as st:
                st.enter_context(vlm(kind="remote", page_fn=page_fn))
                st.enter_context(patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir))
                if quiet is not None:
                    st.enter_context(quiet_after(quiet))
                method, body, log = run_convert(td, builder)
            return method, body

        def counted():
            return [f for f in os.listdir(cache_dir) if f.endswith(".stalls")]

        assert "pass(es) in a row" not in one_pass(five_pages)[0]
        assert f"; 1 of {top} pass(es)" in one_pass()[0]
        assert f"; 2 of {top} pass(es)" in one_pass()[0] and counted()
        state["down_after"] = (
            2  # page 2 is kept, then the lane goes down: the count starts over
        )
        method = one_pass()[0]
        assert "resumes at page 3" in method and "pass(es) in a row" not in method, (
            method
        )
        assert not counted()
        assert f"; 1 of {top} pass(es)" in one_pass()[0] and counted()
        state["down_after"] = (
            None  # page 3 is kept, then the quiet window: a pause starts it over too
        )
        method = one_pass(quiet=1)[0]
        assert "paused" in method and "resumes at page 4" in method, method
        assert not counted()
        method, body = one_pass()
        assert method == "pdf+vlmocr" and in_order(body, 5) and not runs, (method, runs)
        assert not os.listdir(cache_dir), os.listdir(cache_dir)


def test_a_replaced_file_does_not_inherit_the_count():
    state, page_fn = counter(down_after=0)

    def replaced(p):
        five_pages(p)
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))

    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        methods = []
        for builder in (five_pages, replaced):
            with (
                vlm(kind="remote", page_fn=page_fn),
                patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            ):
                methods.append(run_convert(td, builder)[0])
    assert all("; 1 of" in m for m in methods), methods


def test_a_build_with_no_lane_drops_the_cache_and_takes_the_old_route():
    state, page_fn = counter(down_after=2)
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            run_convert(td, five_pages)
        assert len(cache_lines(cache_dir)) == 3
        with no_vlm(), patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+ocr" and runs, (method, runs)
        assert not os.listdir(cache_dir), os.listdir(cache_dir)


def test_a_changed_file_starts_over():
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            run_convert(td, five_pages)
        assert len(cache_lines(cache_dir)) == 3

        def replaced(p):
            five_pages(p)
            st = os.stat(p)
            os.utime(
                p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10)
            )  # a new copy synced over the old one

        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, replaced)
        assert method == "pdf+vlmocr" and len(calls) == 5 and "resuming" not in log, (
            method,
            calls,
            log,
        )


def test_a_torn_last_line_is_redone_and_rewritten_away():
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            run_convert(td, five_pages)
        cp = os.path.join(cache_dir, os.listdir(cache_dir)[0])
        with open(cp, "a", encoding="utf-8") as f:
            f.write(
                '{"i": 2, "text": "the worker was killed whi'
            )  # no newline: the write was cut off
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(1),
        ):
            method, body, log = run_convert(td, keep)
        assert len(calls) == 1 and "5 page(s), 3 done" in method, (calls, method)
        assert [r["i"] for r in cache_lines(cache_dir)[1:]] == [
            0,
            1,
            2,
        ]  # every line parses again
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert (
            method == "pdf+vlmocr"
            and in_order(body, 5)
            and "the worker was killed" not in body
        ), (method, body)


def test_line_separators_and_a_lone_surrogate_inside_a_page_survive_a_resume():
    """Raw U+2028 / U+0085 would let str.splitlines cut a record, and a lone surrogate cannot be written as UTF-8 at
    all (it abandoned the VLM before the cache was written as ASCII JSON); the document is written with it replaced."""
    state = {"page": 0}

    def page_fn(base, png):
        state["page"] += 1
        return (
            marked(state["page"])
            + " Caf\u00e9 \u2028 next\u0085line \u2028 glyph \udc80 end."
        )

    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(3),
        ):
            method, body, log = run_convert(td, five_pages)
        assert method.startswith("vlm_deferred:5 page(s), 3 done"), (method, log)
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 2 and in_order(body, 5), (
            method,
            calls,
        )
        assert body.count("glyph ? end") == 5 and "abandoned" not in log, (body, log)


def test_a_record_cut_mid_character_keeps_the_pages_before_it():
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            run_convert(td, five_pages)
        with open(os.path.join(cache_dir, os.listdir(cache_dir)[0]), "ab") as f:
            f.write(
                b'{"i": 2, "text": "caf\xc3'
            )  # the first byte of a two-byte character
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 3 and in_order(body, 5), (
            method,
            calls,
        )


def test_tesseract_pages_and_the_unreadable_cap_carry_across_a_resume():
    """5 pages: the cap is max(3, 5 %) = 3 unreadable pages with text. Two in the first pass and two in the second
    abandon the VLM, as four in one pass would; a resumed run that finishes lists every Tesseract page."""

    def unreadable(base, png):
        raise bc.VlmPageUnreadable("finish_reason=length chars=6000")

    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=unreadable),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            method, body, log = run_convert(td, five_pages)
        assert method.startswith("vlm_deferred:5 page(s), 2 done") and not runs, (
            method,
            runs,
        )
        assert [r["unreadable"] for r in cache_lines(cache_dir)[1:]] == [True, True]
        with (
            vlm(kind="remote", page_fn=unreadable),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+ocr" and runs and "4 unreadable page(s)" in log, (
            method,
            log,
        )
        assert "VLM OCR abandoned on scan.pdf after 3 page(s)" in log, log
        assert not os.listdir(cache_dir), os.listdir(cache_dir)
    seen = []

    def second_unreadable(base, png):
        seen.append(1)
        if len(seen) == 2:
            raise bc.VlmPageUnreadable("finish_reason=length chars=6000")
        return marked(len(seen))

    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=second_unreadable),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(3),
        ):
            run_convert(td, five_pages)
        with (
            vlm(kind="remote", page_fn=second_unreadable),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and not runs, (method, runs)
        assert "; 1 page(s) by tesseract (2); resumed at page 4" in log, log


def test_a_changed_setting_starts_over():
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True),
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            run_convert(td, five_pages)
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            patched(bc, "OCR_VLM_DPI", bc.OCR_VLM_DPI + 20),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 5 and "resuming" not in log, (
            method,
            calls,
            log,
        )


def test_an_unwritable_cache_never_costs_the_ocr():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
    ):
        blocker = os.path.join(td, "not-a-dir")
        open(blocker, "w").close()  # makedirs under a file: NotADirectoryError
        with (
            vlm(kind="remote"),
            patched(bc, "OCR_VLM_PAGE_CACHE", os.path.join(blocker, "pages")),
        ):
            method, body, log = run_convert(td, five_pages)
    assert method == "pdf+vlmocr" and not runs and "tidal loading" in body, (
        method,
        runs,
    )
    assert "page cache" in log and "not writable" in log and "abandoned" not in log, log


def test_a_lane_down_with_an_unwritable_cache_abandons_rather_than_deferring_for_good():
    """Nothing is kept, so a deferral would redo the same pages on every pass and fail again where a page takes the
    server down: ocrmypdf, as before the cache."""
    state, page_fn = counter(down_after=2)
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
    ):
        blocker = os.path.join(td, "not-a-dir")
        open(blocker, "w").close()
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", os.path.join(blocker, "pages")),
        ):
            method, body, log = run_convert(td, five_pages)
    assert method == "pdf+ocr" and runs and len(calls) == 3, (method, runs, calls)
    assert "not writable" in log and "abandoned on scan.pdf after 2 page(s)" in log, log


def test_a_disk_that_fills_mid_book_costs_neither_the_ocr_nor_a_pause():
    """Every append after the cache is opened fails as on a full disk (its descriptor pointed at /dev/full). A buffered
    cache kept the failed record and raised again when closed -- after the last page, or over a pause -- so a finished
    document was abandoned for ocrmypdf; unbuffered, closing it has nothing left to write."""
    if not os.path.exists("/dev/full"):
        return
    real_open = bc._vlm_cache_open

    def full_disk(path, n, done):
        f = real_open(path, n, done)
        if f is not None:
            fd = os.open("/dev/full", os.O_WRONLY)
            os.dup2(fd, f.fileno())
            os.close(fd)
        return f

    for pause in (False, True):
        with (
            tempfile.TemporaryDirectory() as td,
            fake_llm([OCRMYPDF_LINE * 20]),
            fake_ocrmypdf(available=True) as runs,
        ):
            with (
                vlm(kind="remote"),
                patched(bc, "_vlm_cache_open", full_disk),
                quiet_after(2 if pause else 5),
            ):
                method, body, log = run_convert(td, five_pages)
        assert "page 1 not saved" in log and "abandoned" not in log and not runs, (
            pause,
            method,
            log,
        )
        if pause:
            assert method.startswith("vlm_deferred:5 page(s), 2 done; paused"), method
        else:
            assert method == "pdf+vlmocr" and "tidal loading" in body, method


def test_abandoning_the_vlm_drops_the_cache():
    state, page_fn = counter()

    def timeout(base, png):
        raise TimeoutError("timed out")

    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
    ):
        with (
            vlm(kind="remote", page_fn=page_fn),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
            quiet_after(2),
        ):
            run_convert(td, five_pages)
        with (
            vlm(kind="remote", page_fn=timeout),
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+ocr" and runs, (method, runs)
        assert "VLM OCR abandoned on scan.pdf after 2 page(s)" in log, log
        assert not os.listdir(cache_dir), os.listdir(cache_dir)


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
