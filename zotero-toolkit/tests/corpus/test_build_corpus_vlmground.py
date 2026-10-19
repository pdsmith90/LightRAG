#!/usr/bin/env python3
"""Behavioural tests for VLM OCR v2 (2026-10-19): a streamed page request cut off when the model loops, and a
per-page grounding check against Tesseract that rejects invented text.
Run:  .venv/bin/python test_build_corpus_vlmground.py     (plain asserts; pytest also collects it)
On a blank or title-only page pepperocr-vl at temperature 0 can write a whole invented paper and stop normally (Gill
pp. 379/405, Kingman p. 2, Finite Mixture Models p. 1), or loop until max_tokens (Gill p. 18: "# 10.2.3.4.1.1.1...").
The loop is now seen within a few hundred characters; the invented page is caught because Tesseract reads few or none
of its prose words on that page. Either way Tesseract's reading replaces the page. Blank pages (Tesseract under
OCR_VLM_BLANK_WORDS words) do not count toward OCR_VLM_MAX_UNREADABLE_SHARE.
Synthetic PDFs via pymupdf; pymupdf4llm, the HTTP calls, Tesseract and ocrmypdf replaced by fakes; all text is invented."""

import contextlib
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
    OCRMYPDF_LINE,
    VLM_PAGE,
    fake_llm,
    fake_ocrmypdf,
    image_only_pdf,
    patched,
    run_convert,
    vlm,
)

INVENTED = (
    "# The Influence of Coastal Tourism on Regional Economies\n\nThis paper explores how seasonal visitors "
    "reshape employment, housing markets and municipal budgets across several coastal communities, drawing "
    "on surveys, interviews and administrative records collected over a decade by the authors and their "
    "partners in local government. "
) * 4  # ~156 prose words: above OCR_VLM_GROUND_MIN_WORDS
FIGURE_LABELS = (
    "a\n\nINVENTED_DIFFERENCES Level: Low Season: All UNITS\n\nApex Latitude\n\nMagnetic Local Time "
    "average difference= 4.31\n\nb\n\nINVENTED_DIFFERENCES Level: Medium Season: All UNITS\n\nApex "
    "Latitude\n\nMagnetic Local Time average difference= 4.18\n\nColour scale shows invented values"
)
CAPTION = (
    "Table 4.2. Invented values of the smoothed estimate at two loadings; also standard errors based on "
    "twenty replications of the invented experiment.\n\n"
)


@contextlib.contextmanager
def own_cfg():
    with patched(bc, "CFG", {}):
        yield


@contextlib.contextmanager
def tesseract(*texts, fail=None):
    """bc._tesseract_png replaced by a fake returning texts[i] on its i-th call (pages are read in order; the last
    text repeats), or raising `fail`."""
    calls = []

    def fake(png):
        calls.append(len(png))
        if fail:
            raise fail
        return texts[min(len(calls), len(texts)) - 1]

    with patched(bc, "_tesseract_png", fake):
        yield calls


def answers(*pages):
    """A VLM page function returning pages[i] on the i-th call (raising it when it is an exception)."""
    seen = []

    def page_fn(base, png):
        seen.append(1)
        a = pages[min(len(seen), len(pages)) - 1]
        if isinstance(a, Exception):
            raise a
        return a

    return page_fn


def prefixes(text, step=64):
    return (text[:n] for n in range(step, len(text) + step, step))


# ---- the loop detector ----
def test_loop_detector_catches_degenerate_tails():
    for tail, within in (
        ("# 10.2.3.4" + ".1" * 3000, 600),
        ("Digitized page\n\n" + "░" * 4000, 600),
        (CAPTION * 40, 2600),
    ):
        hit = next((len(p) for p in prefixes(tail) if bc._vlm_looping(p)), None)
        assert hit is not None and hit <= within, (tail[:30], hit)


def test_loop_detector_leaves_real_repetition_alone():
    toc = "".join(
        f"{n}. Invented chapter heading number {n} " + ". " * 55 + f" {3 * n}\n\n"
        for n in range(1, 30)
    )
    cell = "<td style='text-align: center; word-wrap: break-word;'>"
    header = (
        "<table><tr>"
        + (cell + "ISUBM</td>" + cell + "ISUBV</td>") * 8
        + "</tr>"
        + "".join(f"<tr>{cell}{n}</td>{cell}{n / 7:.3f}</td></tr>" for n in range(40))
    )
    for text in (
        toc,
        header,
        INVENTED,
    ):  # (VLM_PAGE is one sentence x6: a loop by construction)
        assert not any(bc._vlm_looping(p) for p in prefixes(text)), text[:40]


# ---- the streamed request ----
class FakeStream:
    """An SSE body: iterating it yields `data:` lines from chunks(); records how many were read and whether closed."""

    def __init__(self, chunks):
        self.chunks, self.read, self.closed = chunks, 0, False

    def __iter__(self):
        for c in self.chunks():
            self.read += 1
            yield b"data: " + json.dumps(c).encode() + b"\n"
            yield b"\n"
        yield b"data: [DONE]\n"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True


def delta(text, finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def serve(stream):
    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        assert body["stream"] is True and body["max_tokens"] == bc.OCR_VLM_MAX_TOKENS, (
            body
        )
        return stream

    return patched(urllib.request, "urlopen", urlopen)


def test_stream_returns_the_page():
    s = FakeStream(
        lambda: iter(
            [delta("## A page\n\n"), delta("Invented text."), delta("", "stop")]
        )
    )
    with serve(s):
        assert (
            bc._vlm_ocr_page("http://127.0.0.1:1", b"\x89PNG")
            == "## A page\n\nInvented text."
        )
    assert s.closed


def test_stream_is_cut_when_the_model_loops():
    def forever():
        yield delta("# 10.2.3.4")
        while True:
            yield delta(".1")

    s = FakeStream(forever)
    with serve(s):
        try:
            bc._vlm_ocr_page("http://127.0.0.1:1", b"\x89PNG")
            raise AssertionError("a loop must raise")
        except bc.VlmPageUnreadable as e:
            assert "loop" in str(e), e
    assert s.closed and s.read < 400, s.read  # hundreds of tokens, not 6000


def test_stream_unusable_endings_are_unreadable():
    for chunks, want in (
        ([delta("x" * 50), delta("", "length")], "finish_reason=length"),
        ([delta("   "), delta("", "stop")], "finish_reason=stop chars=0"),
        ([delta("cut off")], "finish_reason=None"),
    ):
        with serve(FakeStream(lambda chunks=chunks: iter(chunks))):
            try:
                bc._vlm_ocr_page("http://127.0.0.1:1", b"\x89PNG")
                raise AssertionError(want)
            except bc.VlmPageUnreadable as e:
                assert want in str(e), (want, e)


# ---- grounding ----
def test_grounding_rule():
    assert bc._vlm_unsupported(INVENTED, "Finite Mixture Models").startswith(
        "unsupported ("
    )
    assert bc._vlm_unsupported(INVENTED, "").startswith("unsupported (")
    assert bc._vlm_unsupported(VLM_PAGE, VLM_PAGE) == ""  # read by both
    assert (
        bc._vlm_unsupported("$$\\sum_{k=1}^{K} w_k f_k(x)$$\n\n$$\\int_0^1 g = 1$$", "")
        == ""
    )  # formulas only
    assert (
        bc._vlm_unsupported("# Chapter 3\n\nA short invented heading page.", "") == ""
    )  # too few words
    assert (
        bc._vlm_unsupported(FIGURE_LABELS, "") == ""
    )  # labels on a colour map Tesseract cannot read (S2ZD3FHG)


def test_invented_page_is_replaced_by_tesseract():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(kind="remote", page_fn=answers(VLM_PAGE, INVENTED, VLM_PAGE)) as calls,
        tesseract(VLM_PAGE, "Finite Mixture Models", VLM_PAGE) as tess,
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=3))
    assert method == "pdf+vlmocr" and not runs and len(calls) == 3 and len(tess) == 3, (
        method,
        runs,
        calls,
        tess,
    )
    assert (
        "Coastal Tourism" not in body
        and "Finite Mixture Models" in body
        and "tidal loading" in body
    ), body[:200]
    assert (
        "VLM-OCR scan.pdf page 2: unsupported (" in log
        and "-> tesseract (21 chars)" in log
    ), log
    assert "1 page(s) by tesseract (2)" in log and "abandoned" not in log, log


def test_blank_pages_do_not_count_toward_the_cap():
    """Half the pages blank and invented on (a one-sided scan): every one replaced, the VLM kept for the rest."""
    pages = [INVENTED if i % 2 else VLM_PAGE for i in range(10)]
    reads = ["" if i % 2 else VLM_PAGE for i in range(10)]
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(page_fn=answers(*pages)),
        tesseract(*reads),
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=10))
    assert method == "pdf+vlmocr" and not runs, (method, log)
    assert (
        "Coastal Tourism" not in body
        and "5 page(s) by tesseract (2, 4, 6, 8, 10)" in log
    ), log


def test_invented_text_pages_count_toward_the_cap():
    """Pages Tesseract reads plenty on, where the VLM writes something else: the model is not reading this document."""
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([OCRMYPDF_LINE * 20]),
        vlm(page_fn=answers(INVENTED)),
        tesseract(VLM_PAGE),
        fake_ocrmypdf(available=True) as runs,
    ):
        method, body, log = run_convert(td, lambda p: image_only_pdf(p, pages=10))
    assert method == "pdf+ocr" and runs, (method, runs)
    assert (
        "VLM OCR abandoned on scan.pdf after 3 page(s)" in log
        and "4 unreadable page(s)" in log
    ), log


def test_tesseract_failure_skips_the_check():
    with (
        tempfile.TemporaryDirectory() as td,
        own_cfg(),
        fake_llm([]),
        vlm(page_fn=answers(VLM_PAGE)),
        tesseract(fail=FileNotFoundError(2, "No such file or directory: 'tesseract'")),
        fake_ocrmypdf(available=False),
    ):
        method, body, log = run_convert(td, image_only_pdf)
    assert method == "pdf+vlmocr" and "tidal loading" in body, (method, log)
    assert "grounding skipped" in log, log


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
