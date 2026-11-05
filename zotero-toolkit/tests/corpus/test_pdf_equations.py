#!/usr/bin/env python3
"""Behavioural tests for pdf_equations.py and its hooks in build_corpus (2026-11-05): display equations and merged table rows.
Run:  .venv/bin/python test_pdf_equations.py     (plain asserts; pytest also collects it)
Synthetic PDFs via pymupdf; pymupdf4llm replaced by a fake that returns page chunks with page_boxes (the real layout
model is not run); every formula, label and number is invented. The OCR client is a fake, an HTTP stand-in or the
build's own lane (patched)."""

import ast
import contextlib
import http.server
import io
import json
import os
import sys
import threading
import types

import pymupdf

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc
import pdf_equations as eq
from test_build_corpus_vlmocr import patched

LEAD = (
    "The invented model relates the filtered signal to the instrument noise through the transfer function "
    "defined below, where the coefficients were chosen for this fixture only and carry no physical meaning. "
) * 3
AFTER = (
    "where the symbols are explained in the invented list that follows this equation."
)
BETWEEN = "A second invented paragraph that sits between the two equations."
EQ_LAYER = "x2 + y2 = z2                              (5)"  # one text line: formula, gap, equation number
LATEX = "x^{2} + y^{2} = z^{2}"


def page_with_equation(tmp_path, name="eq.pdf", layer=EQ_LAYER, image=False):
    """One page: a lead-in paragraph, an equation line at y=300 (box 40..560 x 290..312), a closing paragraph."""
    p = str(tmp_path / name)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(40, 60, 555, 270), LEAD, fontsize=9)
    if layer:
        page.insert_text((60, 305), layer, fontsize=10)
    page.insert_textbox(pymupdf.Rect(40, 330, 555, 400), AFTER, fontsize=9)
    if image:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 60), False)
        pix.clear_with(220)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(p)
    doc.close()
    return p


def chunk_for(slot="\n\n", box=(40, 290, 560, 312), lead_class="text"):
    """A page chunk the way pymupdf4llm hands it over: text and page_boxes with `pos` slices of that text."""
    lead = LEAD.strip() + "\n\n"
    text = lead + slot + AFTER + "\n\n"
    return {
        "text": text,
        "metadata": {"page_number": 1},
        "page_boxes": [
            {"index": 0, "class": "page-header", "bbox": (0, 0, 1, 1), "pos": (0, 0)},
            {
                "index": 1,
                "class": lead_class,
                "bbox": (40, 60, 555, 270),
                "pos": (0, len(lead)),
            },
            {
                "index": 2,
                "class": "formula",
                "bbox": box,
                "pos": (len(lead), len(lead) + len(slot)),
            },
            {
                "index": 3,
                "class": "text",
                "bbox": (40, 330, 555, 400),
                "pos": (len(lead) + len(slot), len(text)),
            },
        ],
    }


def page_with_two_equations(tmp_path, name="twoeq.pdf"):
    """One page: a lead-in, equation (5) at y=305, a paragraph, equation (6) at y=405, a closing paragraph."""
    p = str(tmp_path / name)
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(40, 60, 555, 200), LEAD, fontsize=9)
    page.insert_text((60, 305), EQ_LAYER, fontsize=10)
    page.insert_textbox(pymupdf.Rect(40, 330, 555, 370), BETWEEN, fontsize=9)
    page.insert_text((60, 405), "a + b = c                    (6)", fontsize=10)
    page.insert_textbox(pymupdf.Rect(40, 420, 555, 460), AFTER, fontsize=9)
    doc.save(p)
    doc.close()
    return p


def chunk_for_two_boxes():
    """A page chunk with two formula slots and the paragraph between them."""
    lead = LEAD.strip() + "\n\n"
    mid = BETWEEN + "\n\n"
    slot = "\n\n"
    text = lead + slot + mid + slot + AFTER + "\n\n"
    p0 = len(lead)
    p1, p2 = p0 + len(slot), p0 + len(slot) + len(mid)
    p3 = p2 + len(slot)
    return {
        "text": text,
        "metadata": {"page_number": 1},
        "page_boxes": [
            {"index": 0, "class": "text", "bbox": (40, 60, 555, 200), "pos": (0, p0)},
            {
                "index": 1,
                "class": "formula",
                "bbox": (40, 290, 560, 312),
                "pos": (p0, p1),
            },
            {
                "index": 2,
                "class": "text",
                "bbox": (40, 330, 555, 370),
                "pos": (p1, p2),
            },
            {
                "index": 3,
                "class": "formula",
                "bbox": (40, 390, 560, 412),
                "pos": (p2, p3),
            },
            {
                "index": 4,
                "class": "text",
                "bbox": (40, 420, 555, 460),
                "pos": (p3, len(text)),
            },
        ],
    }


def run_apply(path, chunk, client, **kw):
    doc = pymupdf.open(path)
    e = eq.Equations(doc, client, **kw)
    out = e.apply(0, chunk, chunk["text"])
    doc.close()
    return out, e


def ok_client(png):
    assert png.startswith(b"\x89PNG")
    return f"$${LATEX}$$"


# ------------------------------------------------------------------ the equation slot


def test_verified_ocr_replaces_the_blank_slot_with_one_tagged_line_in_its_paragraph(
    tmp_path,
):
    out, e = run_apply(page_with_equation(tmp_path), chunk_for(), ok_client)
    assert f"carry no physical meaning.\n${LATEX} \\tag{{5}}$\n\n{AFTER}" in out, out
    assert "$$" not in out and e.stats["ocr"] == 1 and e.stats["text_layer"] == 0
    assert e.stats["pair_order"] == 1


def test_marker_block_of_pymupdf_2_is_replaced_too(tmp_path):
    slot = "\n\n<!-- Start of picture text -->\n\ufffd<br>x2 + y2 = z2 (5)<br><!-- End of picture text -->\n\n"
    out, e = run_apply(page_with_equation(tmp_path), chunk_for(slot), ok_client)
    assert "picture text" not in out and f"${LATEX} \\tag{{5}}$" in out, out


def test_no_ocr_client_writes_the_text_layer_never_a_blank(tmp_path):
    out, e = run_apply(page_with_equation(tmp_path), chunk_for(), None)
    assert "x2 + y2 = z2 (5)" in out and "$" not in out, out
    assert e.stats["text_layer"] == 1 and e.reasons["no OCR"] == 1


def test_ocr_that_misses_a_number_falls_back_to_the_text_layer(tmp_path):
    bad = lambda png: "$$x^{3} + y^{2} = z^{2}$$"  # noqa: E731  (the layer has no 3: it has x2)
    out, e = run_apply(page_with_equation(tmp_path), chunk_for(), lambda p: "$$y = 7$$")
    assert "x2 + y2 = z2 (5)" in out and "$y = 7" not in out, out
    assert e.stats["text_layer"] == 1 and any("unverified" in k for k in e.reasons), (
        e.reasons
    )
    assert bad(b"")  # the fixture itself is valid


def test_ocr_that_misses_letters_falls_back(tmp_path):
    # numbers right, but the layer's letters x, y, z are not in the formula
    out, e = run_apply(
        page_with_equation(tmp_path), chunk_for(), lambda p: "$$a^{2} + b^{2} = c^{2}$$"
    )
    assert "x2 + y2 = z2 (5)" in out and e.stats["ocr"] == 0, out


def test_unbalanced_or_undelimited_ocr_falls_back(tmp_path):
    for answer in ("$$x^{2 + y^{2} = z^{2}$$", "x2 + y2 = z2"):
        out, e = run_apply(
            page_with_equation(tmp_path), chunk_for(), lambda p, a=answer: a
        )
        assert "x2 + y2 = z2 (5)" in out and e.stats["ocr"] == 0, (answer, out)


def test_ocr_failure_is_that_boxes_text_layer_and_three_in_a_row_stop_the_ocr(tmp_path):
    calls = []

    def failing(png):
        calls.append(1)
        raise eq.OcrFailed("boom")

    path = page_with_equation(tmp_path)
    doc = pymupdf.open(path)
    e = eq.Equations(doc, failing)
    for _ in range(5):
        c = chunk_for()
        assert "x2 + y2 = z2 (5)" in e.apply(0, c, c["text"])
    doc.close()
    assert len(calls) == eq.EQ_FAIL_RUN and e.stats["text_layer"] == 5, (
        len(calls),
        e.stats,
    )
    assert e.reasons["OCR lane failing"] == 2


def test_a_client_that_raises_something_else_propagates(tmp_path):
    class LaneGone(Exception):
        pass

    def gone(png):
        raise LaneGone()

    try:
        run_apply(page_with_equation(tmp_path), chunk_for(), gone)
    except LaneGone:
        return
    raise AssertionError(
        "a lane that went away must reach the caller, not become a text-layer line"
    )


def test_budget_and_oversized_boxes_use_the_text_layer(tmp_path):
    out, e = run_apply(page_with_equation(tmp_path), chunk_for(), ok_client, max_ocr=0)
    assert "x2 + y2 = z2 (5)" in out and e.reasons["per-document OCR budget"] == 1
    big = chunk_for(
        box=(40, 20, 555, 450)
    )  # taller than 45 % of the page: not one equation
    out, e = run_apply(page_with_equation(tmp_path), big, ok_client)
    assert e.stats["ocr"] == 0 and e.reasons["too big for one equation"] == 1


def test_a_box_without_a_text_layer_is_left_as_it_was(tmp_path):
    out, e = run_apply(page_with_equation(tmp_path, layer=""), chunk_for(), ok_client)
    assert (
        out == chunk_for()["text"] and e.reasons["no text layer"] == 1
    )  # never an unverifiable OCR guess


def test_a_scanned_page_is_left_to_the_ocr_route(tmp_path):
    path = page_with_equation(tmp_path, image=True)
    out, e = run_apply(path, chunk_for(), ok_client)
    assert out == chunk_for()["text"] and e.stats["boxes"] == 0


def test_slices_that_are_not_blank_are_never_touched(tmp_path):
    c = chunk_for(slot="Some real paragraph text the layout put here.\n\n")
    out, _ = run_apply(page_with_equation(tmp_path), c, ok_client)
    assert out == c["text"]


def test_two_numbered_equations_in_one_box_pair_by_order_and_a_stray_number_is_dropped(
    tmp_path,
):
    layer = "x2 + y2 = z2     (5)"
    path = str(tmp_path / "two.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(40, 60, 555, 270), LEAD, fontsize=9)
    page.insert_text((60, 305), layer, fontsize=10)
    page.insert_text(
        (60, 325), "a = 2 b                                    (6)", fontsize=10
    )
    doc.save(path)
    doc.close()
    answer = f"$${LATEX}$$\n\n$$a = 2b \\quad (6)$$"
    out, e = run_apply(path, chunk_for(box=(40, 290, 560, 332)), lambda p: answer)
    assert f"${LATEX} \\tag{{5}}$\n$a = 2b \\tag{{6}}$" in out, (
        out
    )  # the model's own "(6)" is not repeated
    assert e.stats["pair_order"] == 1


def test_two_boxes_in_one_chunk_keep_their_slots_and_the_text_between_them(tmp_path):
    answers = iter(["$$a + b = c \\quad (6)$$", f"$${LATEX}$$"])

    def client(png):
        # apply() replaces the boxes from the last one, so box (6) is sent to the client first
        assert png.startswith(b"\x89PNG")
        return next(answers)

    out, e = run_apply(page_with_two_equations(tmp_path), chunk_for_two_boxes(), client)
    assert (
        out
        == LEAD.strip()
        + "\n"
        + f"${LATEX} \\tag{{5}}$"
        + "\n\n"
        + BETWEEN
        + "\n"
        + "$a + b = c \\tag{6}$"
        + "\n\n"
        + AFTER
        + "\n\n"
    ), out
    assert out.count("$") == 4 and e.stats["ocr"] == 2, e.stats
    assert e.stats["pair_order"] == 2 and e.stats["boxes"] == 2, e.stats


def test_a_box_whose_ocr_is_only_its_number_falls_back_to_the_text_layer(tmp_path):
    # the layer is the number alone, so the transcription passes the check; with its own number
    # stripped there is no formula left and the box must not become a \tag-only line
    out, e = run_apply(
        page_with_equation(tmp_path, layer="(6)"), chunk_for(), lambda p: "$$(6)$$"
    )
    assert "(6)" in out and "$" not in out, out
    assert e.stats["text_layer"] == 1 and e.stats["ocr"] == 0, e.stats
    assert e.reasons["only the equation number"] == 1, e.reasons


def test_assemble_cases():
    ent = lambda *xs: [{"latex": x, "tag": None} for x in xs]  # noqa: E731
    assert eq.assemble(ent("a", "b"), ["1", "2"]) == (
        ["$a \\tag{1}$", "$b \\tag{2}$"],
        "order",
    )
    assert eq.assemble(ent("a", "b", "c"), ["7"]) == (
        ["$a$", "$b$", "$c \\tag{7}$"],
        "single-number",
    )
    assert eq.assemble(ent("a", "b"), []) == (["$a$", "$b$"], "unnumbered")
    assert eq.assemble(ent("a", "b"), ["1", "2", "3"]) == (["$a$", "$b$"], "unpaired")
    assert eq.assemble([], ["1"]) == ([], "none")
    assert (
        eq.strip_own_number("g(x) = f(2)", "2") == "g(x) = f(2)"
    )  # a call is not a number
    assert eq.strip_own_number("y = 1 \\quad (10)", "10") == "y = 1"


def test_verify_numbers_letters_and_greek():
    words = ["2", "\u03c0", "f", "10", "\u22128", "Hz"]
    layer = " ".join(words)
    assert eq.verify(layer, words, ["2\\pi f 10^{-8}\\,\\text{Hz}"]) == ""
    assert "numbers missing: 8" in eq.verify(layer, words, ["2\\pi f 10\\,\\text{Hz}"])
    assert eq.verify(layer, words, ["2 f 10^{-8}\\,\\text{Hz}"]).startswith(
        "letters"
    )  # the pi is missing
    assert eq.verify("x", ["x"], ["x^{"]) == "malformed LaTeX"
    assert eq.verify("x", ["x"], ["x $ y"]) == "malformed LaTeX"


# ------------------------------------------------------------------ the shared helpers and the prompt


def _fn_dump(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    return {
        n.name: ast.dump(n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    } | {
        t.id: ast.dump(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name) and t.id in names
    }


def test_shared_helpers_match_the_mcp_proxys_pdf_tools():
    shared = (
        "norm_number tag_of _flat split_math tagged number_check line_end_words tags_on_page _center words_in "
        "join_label_lines _ROMAN _TAG_WORD _MATH _TAG_CMD"
    ).split()
    mine = os.path.abspath(eq.__file__)
    theirs = os.path.join(os.path.dirname(mine), "..", "mcp", "paper_pdf_tools.py")
    if not os.path.exists(theirs):
        return  # bundle layout: the proxy's helper lives in mcp/, not beside this file
    a, b = _fn_dump(mine, shared), _fn_dump(theirs, shared)
    assert set(a) == set(shared) == set(b), (
        sorted(set(shared) - set(a)),
        sorted(set(shared) - set(b)),
    )
    assert [n for n in shared if a[n] != b[n]] == [], (
        "paper_pdf_tools.py and pdf_equations.py have drifted apart"
    )


def test_prompt_is_the_ocr_lanes_prompt():
    assert eq.EQ_PROMPT == bc.OCR_VLM_PROMPT


# ------------------------------------------------------------------ tables


MERGED = (
    "|**Parameter**|**A**|**B**|\n|---|---|---|\n"
    "|Mass (kg)<br>Volume (L)<br>Range<br>(m/s2)|9.4<br>10.6<br>6.6|11.4<br>13.7<br>5|\n"
    "|Name|plain|cell|\n"
    "|Note<br>other|one<br>two<br>three|x|\n"
)


def test_merged_row_is_split_by_order_and_others_keep_their_br():
    out, split, merged = eq.split_pipe_tables(MERGED)
    lines = out.split("\n")
    assert lines[:2] == ["|**Parameter**|**A**|**B**|", "|---|---|---|"]
    assert lines[2:5] == [
        "|Mass (kg)|9.4|11.4|",
        "|Volume (L)|10.6|13.7|",
        "|Range (m/s2)|6.6|5|",
    ], out
    assert lines[5] == "|Name|plain|cell|"  # no <br>: untouched
    assert (
        lines[6] == "|Note<br>other|one<br>two<br>three|x|"
    )  # 2 labels vs 3 and 1 lines: not split, kept
    assert (split, merged) == (1, 1)


def test_text_outside_tables_and_rows_without_br_are_untouched():
    md = "Intro | not a table\n\n|a|b|\n|---|---|\n|1|2|\n\nline one<br>line two\n"
    assert eq.split_pipe_tables(md) == (md, 0, 0)


def test_clean_md_keeps_br_in_pipe_rows_and_is_idempotent_and_still_breaks_other_br():
    doc = (
        "Some invented prose line.<br>second line of the prose.\n\n"
        + MERGED
        + "\n\n"
        + MERGED.replace("Mass", "Weight")
    )
    once = bc.clean_md(doc)
    assert "|Note<br>other|one<br>two<br>three|x|" in once
    assert "|Mass (kg)|9.4|11.4|" in once and "|Weight (kg)|9.4|11.4|" in once
    assert (
        "prose line.\nsecond line" in once
    )  # <br> outside a table: a newline, as before
    assert "\x1d" not in once
    assert bc.clean_md(once) == once


def test_a_repeated_equation_line_is_not_page_furniture():
    line = "$" + "\\mathbf{x}_{k+1} = \\mathbf{F}\\mathbf{x}_k + \\mathbf{w}_k" + "$"
    doc = "\n\n".join(
        f"Invented paragraph number {i} with enough words to be prose.\n{line}"
        for i in range(8)
    )
    assert bc.clean_md(doc).count(line) == 8


# ------------------------------------------------------------------ in build_corpus


class FakeLLM:
    def __init__(self, chunk):
        self.chunk = chunk
        self.calls = 0

    def to_markdown(self, path, page_chunks=False, use_ocr=True, **kw):
        assert use_ocr is False
        self.calls += 1
        return [self.chunk]


@contextlib.contextmanager
def converting(chunk, cfg=None, mode="auto", ocr_url=""):
    fake = types.SimpleNamespace(to_markdown=FakeLLM(chunk).to_markdown)
    saved = sys.modules.get("pymupdf4llm")
    sys.modules["pymupdf4llm"] = fake
    try:
        with (
            patched(bc, "CFG", cfg if cfg is not None else {}),
            patched(eq, "EQ_MODE", mode),
            patched(eq, "EQ_OCR_URL", ocr_url),
        ):
            yield
    finally:
        if saved is None:
            del sys.modules["pymupdf4llm"]
        else:
            sys.modules["pymupdf4llm"] = saved


def lane(answer=None, kind="remote"):
    calls = []

    def page(base, png):
        calls.append(len(png))
        if isinstance(answer, Exception):
            raise answer
        return answer or f"$${LATEX}$$"

    return (
        calls,
        patched(bc, "_vlm_endpoint", lambda *a, **k: (kind, "http://127.0.0.1:1")),
        patched(bc, "_vlm_ocr_page", page),
    )


def test_pdf_to_md_uses_the_ocr_lane_for_formula_boxes(tmp_path):
    path = page_with_equation(tmp_path)
    calls, ep, pg = lane()
    with converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}), ep, pg:
        md = bc.pdf_to_md(path)
    assert f"${LATEX} \\tag{{5}}$" in md and len(calls) == 1, md


def test_pdf_to_md_without_ocr_writes_the_text_layer(tmp_path):
    path = page_with_equation(tmp_path)
    calls, ep, pg = lane()
    with converting(chunk_for(), {"ocr": "off", "ocr_vlm": "off"}), ep, pg:
        md = bc.pdf_to_md(path)
    assert "x2 + y2 = z2 (5)" in md and calls == [] and "$" not in md


def test_an_unreadable_crop_is_the_text_layer_not_a_failed_document(tmp_path):
    path = page_with_equation(tmp_path)
    calls, ep, pg = lane(answer=bc.VlmPageUnreadable("loop on '.1' after 400 chars"))
    with converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}), ep, pg:
        md = bc.pdf_to_md(path)
    assert "x2 + y2 = z2 (5)" in md and "carry no physical meaning" in md


def test_a_lane_that_goes_down_defers_the_document(tmp_path):
    path = page_with_equation(tmp_path)
    calls, ep, pg = lane(answer=ConnectionError("refused"))
    with converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}), ep, pg:
        try:
            bc.pdf_to_md(path)
        except bc.VlmDeferred as e:
            assert "went down" in str(e)
            return
    raise AssertionError(
        "a lane that cannot serve must defer, not write text-layer lines"
    )


def test_a_configured_lane_that_is_not_usable_defers_the_document(tmp_path):
    path = page_with_equation(tmp_path)
    busy = patched(
        bc, "_vlm_endpoint", lambda *a, **k: ("defer", "remote lane closed until 08:45")
    )
    if not hasattr(
        bc, "_vlm_lane_close"
    ):  # the fork's endpoint has no per-document "defer": it says ("", "")
        busy = patched(bc, "_vlm_endpoint", lambda *a, **k: ("", ""))
    with (
        converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}),
        busy,
        patched(bc, "OCR_VLM_REMOTE_URL", "http://127.0.0.1:1"),
    ):
        try:
            bc.pdf_to_md(path)
        except bc.VlmDeferred:
            return
    raise AssertionError("a busy lane must defer the document")


def test_convert_writes_nothing_for_a_deferred_document():
    import pathlib
    import tempfile

    from test_build_corpus_vlmocr import run_convert

    def build(p):
        page_with_equation(pathlib.Path(os.path.dirname(p)), os.path.basename(p))

    with tempfile.TemporaryDirectory() as td:
        calls, ep, pg = lane(answer=ConnectionError("refused"))
        with converting(chunk_for()), ep, pg:
            method, body, log = run_convert(td, build)
        assert method.startswith("vlm_deferred:") and body == "", (method, body[:100])
        assert "VLM-OCR deferred" in log


def test_no_lane_configured_at_all_is_the_text_layer(tmp_path):
    path = page_with_equation(tmp_path)
    with (
        converting(chunk_for(), {"ocr": "auto", "ocr_vlm": "auto"}),
        patched(bc, "_vlm_endpoint", lambda *a, **k: ("", "")),
        patched(bc, "OCR_VLM_REMOTE_URL", ""),
        patched(bc, "OCR_VLM_LOCAL_START", ""),
    ):
        md = bc.pdf_to_md(path)
    assert "x2 + y2 = z2 (5)" in md and "$" not in md


def test_equations_off_is_the_old_blank_line(tmp_path):
    path = page_with_equation(tmp_path)
    with converting(chunk_for(), {"ocr": "off"}, mode="off"):
        md = bc.pdf_to_md(path)
    assert "x2 + y2" not in md and "$" not in md


def test_http_stand_in_server_serves_the_validation_client(tmp_path):
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            out = json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": f"$${LATEX}$$"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        path = page_with_equation(tmp_path)
        with converting(
            chunk_for(), {"ocr": "off"}, ocr_url=f"http://127.0.0.1:{srv.server_port}"
        ):
            md = bc.pdf_to_md(path)
    finally:
        srv.shutdown()
    assert f"${LATEX} \\tag{{5}}$" in md
    body = seen[0]
    assert (
        body["temperature"] == 0
        and body["messages"][0]["content"][1]["text"] == eq.EQ_PROMPT
    )
    assert body["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_a_formula_heavy_page_is_not_replaced_by_its_text_layer(tmp_path):
    # the page's text layer is mostly equations: without counting them as kept, the keep ratio (< 40 %) repairs the page
    path = str(tmp_path / "heavy.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        pymupdf.Rect(40, 60, 555, 120),
        "A short invented lead-in sentence for the page.",
        fontsize=9,
    )
    rows = [
        "term%d + coef%d * value%d = result%d      (%d)" % (i, i, i, i, 20 + i)
        for i in range(1, 25)
    ]
    for i, r in enumerate(rows):
        page.insert_text((60, 150 + 14 * i), r, fontsize=10)
    doc.save(path)
    doc.close()
    lead = "A short invented lead-in sentence for the page.\n\n"
    chunk = {
        "text": lead + "\n\n",
        "metadata": {"page_number": 1},
        "page_boxes": [
            {
                "index": 0,
                "class": "text",
                "bbox": (40, 60, 555, 120),
                "pos": (0, len(lead)),
            },
            {
                "index": 1,
                "class": "formula",
                "bbox": (40, 140, 560, 490),
                "pos": (len(lead), len(lead) + 2),
            },
        ],
    }
    with converting(chunk, {"ocr": "off"}, mode="text"):
        md = bc.pdf_to_md(path)
    assert bc._pdf_method != "pdf+textlayer" and "term1 + coef1" in md, (
        bc._pdf_method,
        md[:300],
    )
    with converting(chunk, {"ocr": "off"}, mode="off"):
        bc.pdf_to_md(path)
    assert bc._pdf_method == "pdf+textlayer"  # as before this change


def test_log_line_names_the_counts(tmp_path, capsys=None):
    path = page_with_equation(tmp_path)
    buf = io.StringIO()
    with (
        converting(chunk_for(), {"ocr": "off"}, mode="text"),
        contextlib.redirect_stdout(buf),
    ):
        bc.pdf_to_md(path)
    assert (
        "EQUATIONS eq.pdf: 1 formula box(es); 0 transcribed and checked; 1 as text layer"
        in buf.getvalue()
    )


if __name__ == "__main__":
    import tempfile

    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        fn = globals()[n]
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as td:
                import pathlib

                fn(pathlib.Path(td))
        else:
            fn()
        print("ok", n)
    print(f"{len(names)} passed")
