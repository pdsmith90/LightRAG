#!/usr/bin/env python3
"""Behavioural tests for a remote OCR lane that is reserved, or whose launcher refuses the model.
Run:  .venv/bin/python test_build_corpus_vlmreserve.py     (plain asserts; pytest also collects it)
llama-swap with sendLoadingState answers a STREAMED request for a model whose launch fails with HTTP 200: the loading
banner as reasoning_content, an error event, [DONE]. The page call read that as a page with no text and no
finish_reason, so the page went to Tesseract (a formula crop: to its text layer) where the same request unstreamed, an
HTTP 500, defers the document. An error event, or a stream that ends with no answer at all, is now a lane that cannot
serve (VlmLaneError). And unscheduled work that needs the remote GPU can reserve it through a file holding an epoch
second (OCR_VLM_REMOTE_RESERVED_FILE, off when unset), read as fronts/llm_failover.py reads LLM_FO_RESERVED_FILE:
while it names a future moment the lane choice skips the remote lane, a document in progress on it pauses before its
next page and resumes from the page cache, and a formula crop is not sent.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP stream, Tesseract and ocrmypdf replaced by fakes; text invented."""

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request

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
    patched,
    run_convert,
    vlm,
)
from test_build_corpus_vlmresume import cache_lines, counter, five_pages, in_order, keep
from test_pdf_equations import LATEX, chunk_for, converting, page_with_equation
from test_pdf_equations import lane as eq_lane

URL = "http://127.0.0.1:1"
LOCAL, REMOTE = (
    "http://127.0.0.1:2",
    "http://127.0.0.1:3",
)  # never reached: the probes are faked


def sse(*events):
    """A streamed response body: one "data: {json}" line per event, a blank line between, then [DONE]."""
    return (
        b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        + b"data: [DONE]\n\n"
    )


def answer(finish, content):
    """What the server streams for an answer: the content, then the chunk with the finish_reason."""
    return sse(
        {"choices": [{"delta": {"content": content}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": finish}]},
    )


# What llama-swap (sendLoadingState: true) streams when the model's launcher refuses to start: HTTP 200, the loading
# banner as reasoning_content one piece at a time, an error event, [DONE]. Unstreamed, the same request gets HTTP 500.
BANNER = [
    "=====",
    "\n",
    "llama",
    "-swap",
    " loading",
    " model",
    ":",
    " ",
    bc.OCR_VLM_MODEL,
    "\n\n",
    "Done",
    "!",
    " (",
    "0.40s",
    ")",
    "\n",
    "=====",
    "\n",
    " ",
    "\n",
]
REFUSED = sse(
    *({"choices": [{"delta": {"reasoning_content": c}}]} for c in BANNER),
    {
        "src": "llama-swap",
        "error": {
            "message": "upstream command exited prematurely",
            "type": "server_error",
            "param": None,
            "code": "internal_error",
        },
    },
)
CUT = sse(
    {
        "choices": [
            {"delta": {"content": "Text of a page cut off."}, "finish_reason": None}
        ]
    }
)


def http500():
    return urllib.error.HTTPError(
        URL + "/v1/chat/completions", 500, "Internal Server Error", {}, None
    )


@contextlib.contextmanager
def serving(*bodies):
    """urlopen replaced: request i is answered bodies[i] (the last one repeats; an exception is raised instead).
    Yields the list of URLs requested."""
    sent = []

    def urlopen(req, timeout=None):
        body = bodies[min(len(sent), len(bodies) - 1)]
        sent.append(req.full_url)
        if isinstance(body, Exception):
            raise body
        return contextlib.closing(io.BytesIO(body))

    with patched(urllib.request, "urlopen", urlopen):
        yield sent


@contextlib.contextmanager
def streaming(*bodies, kind="remote"):
    """The real _vlm_ocr_page on lane `kind` behind serving(*bodies). Yields a namespace: sent (the URLs requested),
    tess (the Tesseract calls; each reads VLM_PAGE) and cache (the page cache's directory)."""
    seen = types.SimpleNamespace(sent=[], tess=[], cache="")

    def tesseract(png):
        seen.tess.append(len(png))
        return VLM_PAGE

    with (
        tempfile.TemporaryDirectory() as cache_dir,
        serving(*bodies) as sent,
        patched(bc, "_vlm_endpoint_cache", None),
        patched(bc, "_vlm_endpoint", lambda *a, **k: (kind, URL)),
        patched(bc, "_tesseract_png", tesseract),
        patched(bc, "_vlm_quiet", lambda now=None: False),
        patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
    ):
        seen.sent, seen.cache = sent, cache_dir
        yield seen


def write(path, content):
    with open(path, "wb") as f:
        f.write(content if isinstance(content, bytes) else content.encode())


@contextlib.contextmanager
def reserved(content=None):
    """OCR_VLM_REMOTE_RESERVED_FILE naming a temporary file that holds `content` (None: the file does not exist).
    Yields its path."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "reserved_until")
        if content is not None:
            write(path, content)
        with patched(bc, "OCR_VLM_REMOTE_RESERVED_FILE", path):
            yield path


def hhmm(epoch):
    return time.strftime("%H:%M", time.localtime(epoch))


@contextlib.contextmanager
def lanes(local=False, quiet=False):
    """The real _vlm_endpoint between two fake servers: the remote one lists the model, the local one only when
    `local`; nothing is started and the cache is fresh. Yields the servers probed, in order."""
    probed = []

    def probe(base, timeout=3.0):
        probed.append(base)
        return "model" if base == REMOTE or (local and base == LOCAL) else "down"

    with (
        patched(bc, "OCR_VLM_LOCAL_URL", LOCAL),
        patched(bc, "OCR_VLM_REMOTE_URL", REMOTE),
        patched(bc, "OCR_VLM_LOCAL_START", ""),
        patched(bc, "_vlm_probe", probe),
        patched(bc, "_vlm_quiet", lambda now=None: quiet),
        patched(bc, "_vlm_endpoint_cache", None),
    ):
        yield probed


# ---- a refused launch is a lane that cannot serve -------------------------------------------------------------------


def test_the_page_call_tells_a_refused_launch_from_an_unreadable_page():
    def call(body):
        with serving(body):
            return bc._vlm_ocr_page(URL, b"\x89PNG")

    try:
        call(REFUSED)
        raise AssertionError("an error event")
    except bc.VlmLaneError as e:
        assert isinstance(e, RuntimeError) and not isinstance(
            e, bc.VlmPageUnreadable
        ), e
        assert str(e) == "upstream command exited prematurely" and bc._vlm_lane_down(e)
    try:
        call(sse({"error": "a plain-string error"}))
        raise AssertionError("an error event whose error is a string")
    except bc.VlmLaneError as e:
        assert str(e) == "a plain-string error" and bc._vlm_lane_down(e), e
    for body in (
        sse({"choices": [{"delta": {"content": ""}, "finish_reason": None}]}),
        sse({"choices": [{"delta": {"reasoning_content": "x"}}]}),
        sse(),
    ):  # a stream that ends with no text and no finish_reason
        try:
            call(body)
            raise AssertionError(body)
        except bc.VlmLaneError as e:
            assert "no answer" in str(e) and bc._vlm_lane_down(e), e
    # unchanged: a page the model could not transcribe is unreadable, never a lane that is down
    for finish, content in (
        ("length", "loop " * 50),
        ("length", "partial text"),
        ("stop", "   "),
        ("stop", ""),
    ):
        try:
            call(answer(finish, content))
            raise AssertionError((finish, content))
        except bc.VlmPageUnreadable as e:
            assert f"finish_reason={finish}" in str(e) and not bc._vlm_lane_down(e), e
    try:
        call(CUT)  # text but no finish_reason: unreadable, as before
        raise AssertionError("a cut-off stream with text")
    except bc.VlmPageUnreadable as e:
        assert str(e) == "finish_reason=None chars=23" and not bc._vlm_lane_down(e), e
    try:
        call(http500())
        raise AssertionError("HTTP 500")
    except urllib.error.HTTPError as e:
        assert bc._vlm_lane_down(e)  # unchanged: the unstreamed refusal
    assert call(answer("stop", "## A page\n\nText.")) == "## A page\n\nText."


def test_a_refused_launch_on_the_first_page_defers_the_document():
    """Deferred like any lane that is down on a document's first page: no Tesseract page is written, and the passes
    that keep no new page count toward OCR_VLM_MAX_STALLS; the pass after that many goes to ocrmypdf."""
    top = bc.OCR_VLM_MAX_STALLS
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([OCRMYPDF_LINE * 20]),
        fake_ocrmypdf(available=True) as runs,
        streaming(REFUSED) as s,
    ):
        method, body, log = run_convert(td, five_pages)
        assert method == (
            "vlm_deferred:5 page(s); the remote lane went down after 0 page(s) "
            "(VlmLaneError: upstream command exited prematurely), resumes at page 1"
            f"; 1 of {top} pass(es) in a row with no new page kept"
        ), method
        assert body == "" and not runs and s.tess == [], (body[:40], runs, s.tess)
        assert len(s.sent) == 1 and "VLM-OCR deferred scan.pdf" in log, (s.sent, log)
        assert "page 1:" not in log and "abandoned" not in log, log
        for k in range(2, top + 1):
            method = run_convert(td, keep)[0]
            assert (
                method.startswith("vlm_deferred:")
                and f"; {k} of {top} pass(es)" in method
            ), (k, method)
        method, body, log = run_convert(td, keep)
        assert method == "pdf+ocr" and runs and s.tess == [], (method, runs, s.tess)
        assert f"{top + 1} pass(es) in a row with no new page kept" in log, log


def test_a_refused_launch_after_some_pages_keeps_them_and_defers():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        streaming(answer("stop", VLM_PAGE), answer("stop", VLM_PAGE), REFUSED) as s,
    ):
        method, body, log = run_convert(td, five_pages)
        assert method == (
            "vlm_deferred:5 page(s); the remote lane went down after 2 page(s) "
            "(VlmLaneError: upstream command exited prematurely), resumes at page 3"
        ), method
        assert not runs and body == "" and len(s.tess) == 2, (runs, s.tess)
        assert [r["i"] for r in cache_lines(s.cache)[1:]] == [0, 1]


def test_a_refused_launch_defers_a_formula_box_rather_than_writing_its_text_layer():
    with tempfile.TemporaryDirectory() as td:
        path = page_with_equation(pathlib.Path(td))
        with (
            streaming(REFUSED) as s,
            converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}),
        ):
            try:
                md = bc.pdf_to_md(path)
            except bc.VlmDeferred as e:
                assert "formula box(es); the remote lane went down" in str(e), e
                assert "VlmLaneError: upstream command exited prematurely" in str(e), e
                assert len(s.sent) == 1, s.sent
                return
    raise AssertionError(f"a refused launch must defer the document, not write {md!r}")


def test_a_page_the_model_cannot_read_still_goes_to_tesseract():
    for bad in (
        answer("length", "partial text " * 20),
        answer("stop", ""),
        answer(None, "loop " * 100),
        CUT,
    ):
        with (
            tempfile.TemporaryDirectory() as td,
            fake_llm([]),
            fake_ocrmypdf(available=True) as runs,
            streaming(bad, answer("stop", VLM_PAGE)) as s,
        ):
            method, body, log = run_convert(td, image_only_pdf)
            assert method == "pdf+vlmocr" and not runs, (bad, method, runs)
            assert "scan.pdf page 1:" in log and "-> tesseract" in log, (bad, log)
            assert "; 1 page(s) by tesseract (1)" in log, (bad, log)
            assert len(s.sent) == 2 and not os.listdir(s.cache), (bad, s.sent)


def test_an_http_500_is_still_a_lane_that_is_down():
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        streaming(http500()) as s,
    ):
        method, body, log = run_convert(td, five_pages)
    assert method.startswith("vlm_deferred:") and "(HTTPError: " in method, method
    assert "went down after 0 page(s)" in method and not runs and s.tess == []


# ---- a reserved remote lane -----------------------------------------------------------------------------------------


def test_a_reservation_closes_the_remote_lane():
    until = int(time.time()) + 3600
    for content in (f"{until}\n", f"{until} benchmark\n", f"{until}.5"):
        with reserved(content), lanes() as probed:
            assert bc._vlm_reserved_until() == until, content
            assert bc._vlm_reserved() == hhmm(until), content
            assert bc._vlm_endpoint() == ("", ""), content
            assert REMOTE not in probed, (content, probed)  # never even probed
    with reserved(f"{until}\n"), lanes(local=True) as probed:
        assert bc._vlm_endpoint() == ("local", LOCAL)  # the local lane serves meanwhile
        assert REMOTE not in probed, probed
    with reserved(f"{until}\n"), lanes(quiet=True) as probed:
        assert bc._vlm_endpoint() == ("", "") and REMOTE not in probed


def test_an_absent_expired_or_garbled_reservation_changes_nothing():
    past = int(time.time()) - 60
    for content in (
        None,
        f"{past}\n",
        "0",
        "",
        "  \n",
        "soon",
        "note 123",
        "nan",
        "inf",
        b"\x00\xff",
    ):
        with reserved(content), lanes() as probed:
            assert bc._vlm_reserved() == "", content
            assert bc._vlm_endpoint() == ("remote", REMOTE), content
            assert probed == [LOCAL, REMOTE], (content, probed)
    with patched(bc, "OCR_VLM_REMOTE_RESERVED_FILE", ""), lanes() as probed:
        assert bc._vlm_reserved_until() == 0 and bc._vlm_reserved() == ""
        assert bc._vlm_endpoint() == ("remote", REMOTE) and probed == [LOCAL, REMOTE]
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        reserved(f"{past}\n"),
        vlm(kind="remote") as calls,
    ):
        method, body, log = run_convert(td, five_pages)
    assert method == "pdf+vlmocr" and len(calls) == 5 and not runs, (method, calls)


def test_the_reservation_is_off_unless_the_environment_names_a_file():
    here = os.path.dirname(os.path.abspath(bc.__file__))

    def value(**extra):
        env = {
            k: v for k, v in os.environ.items() if k != "OCR_VLM_REMOTE_RESERVED_FILE"
        }
        env.update(extra)
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "import build_corpus as b; print(repr(b.OCR_VLM_REMOTE_RESERVED_FILE))",
            ],
            cwd=here,
            env=env,
            capture_output=True,
            text=True,
        ).stdout.strip()

    assert value() == "''"
    assert value(OCR_VLM_REMOTE_RESERVED_FILE="/tmp/reserved_until") == repr(
        "/tmp/reserved_until"
    )


def test_a_reserved_remote_lane_is_skipped_as_in_its_quiet_window():
    """No request reaches the remote lane. With no local lane the scan then takes the route it takes inside the quiet
    window (ocrmypdf): the lane choice has no per-document wait."""
    until = int(time.time()) + 3600
    sent = []

    def no_request(req, timeout=None):
        sent.append(req.full_url)
        raise AssertionError("no request may reach the remote lane")

    outcome = []
    for quiet, content in ((False, f"{until}\n"), (True, None)):
        with (
            tempfile.TemporaryDirectory() as td,
            fake_llm([OCRMYPDF_LINE * 20]),
            fake_ocrmypdf(available=True) as runs,
            reserved(content),
            lanes(quiet=quiet) as probed,
            patched(urllib.request, "urlopen", no_request),
        ):
            method, body, log = run_convert(td, five_pages)
        assert not sent and REMOTE not in probed, (quiet, sent, probed)
        outcome.append((method, len(runs)))
    assert outcome[0] == outcome[1] == ("pdf+ocr", 1), outcome


def test_a_document_on_a_remote_lane_chosen_earlier_waits_while_it_is_reserved():
    """The lane choice is cached per build: a reservation that begins after it keeps the next document off the lane."""
    until = int(time.time()) + 3600
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        reserved(f"{until}\n"),
        vlm(kind="remote") as calls,
    ):
        method, body, log = run_convert(td, five_pages)
        assert method == (
            f"vlm_deferred:5 page(s), 0 done; paused as the remote lane is reserved until {hhmm(until)}, "
            "resumes at page 1"
        ), method
        assert calls == [] and not runs and body == "", (calls, runs)
        stalls = [f for f in os.listdir(bc.OCR_VLM_PAGE_CACHE) if f.endswith(".stalls")]
        assert not stalls, stalls  # a pause is no stall


def test_a_reservation_that_begins_mid_book_pauses_it_and_the_next_pass_resumes():
    until = int(time.time()) + 3600
    state, page_fn = counter()
    with (
        tempfile.TemporaryDirectory() as td,
        tempfile.TemporaryDirectory() as cache_dir,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        reserved(None) as path,
    ):

        def reserving(base, png):  # the reservation is written while page 2 is read
            text = page_fn(base, png)
            if state["page"] == 2:
                write(path, f"{until}\n")
            return text

        with (
            vlm(kind="remote", page_fn=reserving) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, five_pages)
        assert method == (
            f"vlm_deferred:5 page(s), 2 done; paused as the remote lane is reserved until {hhmm(until)}, "
            "resumes at page 3"
        ), method
        assert len(calls) == 2 and not runs and body == "", (calls, runs)
        assert [r["i"] for r in cache_lines(cache_dir)[1:]] == [0, 1]
        assert not [
            f for f in os.listdir(cache_dir) if f.endswith(".stalls")
        ]  # a pause is no stall
        os.remove(path)  # the holder released the lane
        with (
            vlm(kind="remote", page_fn=page_fn) as calls,
            patched(bc, "OCR_VLM_PAGE_CACHE", cache_dir),
        ):
            method, body, log = run_convert(td, keep)
        assert method == "pdf+vlmocr" and len(calls) == 3 and not runs, (method, calls)
        assert in_order(body, 5) and "resuming at page 3 of 5 (remote)" in log, log
        assert not os.listdir(cache_dir), os.listdir(cache_dir)


def test_a_local_lane_never_pauses_for_a_remote_reservation():
    until = int(time.time()) + 3600
    with (
        tempfile.TemporaryDirectory() as td,
        fake_llm([]),
        fake_ocrmypdf(available=True) as runs,
        reserved(f"{until}\n"),
        vlm(kind="local") as calls,
    ):
        method, body, log = run_convert(td, five_pages)
    assert method == "pdf+vlmocr" and len(calls) == 5 and not runs, (method, calls)


def test_a_formula_crop_is_not_sent_to_a_reserved_remote_lane():
    until = int(time.time()) + 3600
    with tempfile.TemporaryDirectory() as td, reserved(f"{until}\n"):
        path = page_with_equation(pathlib.Path(td))
        calls, ep, pg = (
            eq_lane()
        )  # the remote lane, chosen before the reservation began
        with converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}), ep, pg:
            try:
                bc.pdf_to_md(path)
                raise AssertionError("a reserved lane must defer the formula box")
            except bc.VlmDeferred as e:
                assert (
                    f"formula box(es); the remote lane is reserved until {hhmm(until)}"
                    in str(e)
                ), e
        assert calls == [], calls
        calls, ep, pg = eq_lane(kind="local")
        with converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}), ep, pg:
            md = bc.pdf_to_md(path)
        assert f"${LATEX} \\tag{{5}}$" in md and len(calls) == 1, md


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
