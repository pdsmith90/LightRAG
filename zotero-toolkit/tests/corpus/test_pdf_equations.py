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


# ------------------------------------------------------------------ the corrupted text layer
# Crops from the Phase C validation (results/validation-phaseC.md section 9, probe_all.json): a
# broken PDF text layer, the OCR answer served from its cache, and the verdict. They pin the
# normalisation of the layer's known glyph corruptions; the checks themselves still bind.


def test_tag_of_reads_the_glyphs_a_broken_font_prints_for_parentheses_and_an_en_dash():
    assert eq.tag_of("ð7Þ") == "7"  # VCNXJ5JV p3: '(' printed as eth, ')' as thorn
    assert eq.tag_of("(1–30)") == "1-30"  # TCZLPVMV p26: an en dash between the digits
    assert eq.tag_of("‹A12›") == "A12"
    assert eq.tag_of("ðT") is None  # a mapped glyph alone is not a number
    assert eq.tag_of("273:15Þ") is None


def test_three_equations_in_one_box_with_the_fraction_glyph_and_colon_decimals_verify():
    # VCNXJ5JV p3: the font prints '=' as U+2044 ('1⁄4') and '.' as ':' ('77:689'), so the layer
    # manufactured the numbers 1, 4, 7, 8, 9, 15, 77, 269, ... OCR right; it was "numbers missing".
    words = [
        "DLtrop", "¼", "10", "6", "N", "D", "ð7Þ", "N", "¼", "77:689", "P", "e", "T",
        "þ", "71:2952", "e", "T", "þ", "375463", "e", "T", "2", "ð8Þ", "e", "¼",
        "6:1078", "exp", "17:269", "ðT", "273:15Þ", "237:30", "þ", "ðT", "273:15Þ", "f",
        "ð9Þ",
    ]  # fmt: skip
    latexes = [
        "\\Delta L_{trop} = 10^{-6} \\cdot N \\cdot D",
        "N = 77.689 \\cdot \\frac{P - e}{T} + 71.2952 \\cdot \\frac{e}{T} + 375463 \\cdot \\frac{e}{T^2}",
        "e = 6.1078 \\cdot \\exp\\left(\\frac{17.269 \\cdot (T - 273.15)}{237.30 + (T - 273.15)}\\right) \\cdot f",
    ]  # fmt: skip
    rest = [w for w in words if not eq.tag_of(w)]  # _box_layer() drops the tag words
    assert len(words) - len(rest) == 3  # ð7Þ, ð8Þ, ð9Þ
    assert eq.number_check(rest, latexes) == []
    assert eq.verify(eq._flat(" ".join(rest)), rest, latexes) == ""


def test_number_check_still_binds_on_a_real_digit():
    assert eq.number_check(["10"], ["2"]) == ["10"]
    assert eq.number_check(["¼", "10"], ["10"]) == []  # the '=' glyph is not a number
    # the colon decimal point is a glyph corruption too: only a layer that shows the font is read so
    assert eq.number_check(["¼", "77:689"], ["77.689"]) == []
    assert eq.number_check(["77:689"], ["77.689"]) == [
        "77",
        "689",
    ]  # a healthy 'a:b' is a ratio
    assert eq.number_check(["A1"], ["x_i = x_T + b_i,"]) == ["1"]


def test_split_math_keeps_the_formula_whose_spacing_and_number_survive_beside_it():
    # CHE7U6NC p2: the model wrote "$...$. \qquad (2)"; \qquad read as a two-letter word, so the
    # line counted as inline prose and the box fell back with "no formula in the transcription".
    raw = "$a_x = -\\frac{1}{2}\\rho C_d \\frac{A_x}{m} V_x^2$. \\qquad (2)"
    entries, delimited = eq.split_math(raw)
    assert delimited and entries == [
        {"latex": "a_x = -\\frac{1}{2}\\rho C_d \\frac{A_x}{m} V_x^2", "tag": None}
    ], entries
    words = ["ax", "¼", "1", "2", "rCd", "Ax", "m", "Vx", "2:", "ð2Þ"]
    rest = [w for w in words if not eq.tag_of(w)]
    assert eq.verify(eq._flat(" ".join(rest)), rest, [entries[0]["latex"]]) == ""
    # prose beside the math is still prose, not a display formula
    assert eq.split_math("the mantle is that of $n=0$ samples") == ([], True)


def test_split_math_takes_a_trailing_eqno_as_the_equation_number():
    for raw, tag in (
        ("$$a=b \\eqno(2.2)$$", "2.2"),  # WCLPVYVE p2
        ("$$a=b \\eqno (4)$$", "4"),  # 5LFWGZJY p2
        ("$$a=b \\eqno{(42)}$$", "42"),  # 9WPVX7DW p8
    ):
        entries, _ = eq.split_math(raw)
        assert entries == [{"latex": "a=b", "tag": tag}], (raw, entries)


def test_a_real_half_and_a_real_q_on_a_healthy_layer_are_checked_strictly():
    # a layer with no glyph marker: '1⁄2' is a half, q and S are the letters q and S
    words = ["x", "=", "1⁄2", "y"]
    assert not eq.layer_garbled(" ".join(words))
    assert eq.number_check(words, ["x = 1 y"]) == ["2"]  # the OCR dropped the 2
    assert eq.number_check(words, ["x = \\frac{1}{2} y"]) == []
    assert eq.number_check(words, ["x = 10 y"]) == [
        "1",
        "2",
    ]  # a half's digits are checked
    assert eq.letter_recall("q", ["\\theta"]) == (0, 1)  # a real q is not theta
    assert eq.letter_recall("S", ["\\mathcal{J}"]) == (0, 1)
    # 9WPVX7DW p2, real crop: "∇S · ∇S = n2, (3)" shows no marker, so its S is a real S and the
    # OCR's \mathcal{J} is a miss (this box passed between aee9fc49 and this gate)
    words = ["∇S", "·", "∇S", "=", "n2,"]
    latex = "\\nabla \\mathcal{J} \\cdot \\nabla \\mathcal{J} = n^2,"
    assert eq.verify(eq._flat(" ".join(words)), words, [latex]) == "letters 1/3"


def test_the_same_layer_in_a_broken_font_is_normalised():
    # the same words, plus the marker the broken font prints for its parentheses: now the '1⁄2' is
    # its '=' glyph (with a fabricated 1 and 2) and its q is theta
    words = ["x", "=", "1⁄2", "y", "BðzÞ"]
    assert eq.layer_garbled(" ".join(words))
    assert eq.number_check(words, ["x = 1 y"]) == []  # the fabricated 2 is not wanted
    assert eq.number_check(words, ["x = 10 y"]) == []  # nor the 1
    assert eq.letter_recall("q ð", ["\\theta"]) == (1, 1)  # q credited as theta
    assert eq.letter_recall("q ð", ["\\xi"]) == (
        0,
        1,
    )  # a glyph the table does not name


def test_the_marker_is_the_fraction_glyph_standing_for_equals_not_a_real_fraction():
    assert eq.layer_garbled("E ¼ m c2")  # '¼' alone: the font's '='
    assert eq.layer_garbled("E 1⁄4 m c2")  # the same glyph written out
    assert not eq.layer_garbled("x = ½ y")  # a real half is not a marker
    assert not eq.layer_garbled("x = 1⁄2 y")
    assert eq.layer_garbled("BðzÞ")


def test_the_ellipsis_and_dagger_count_only_as_the_parentheses_they_replaced():
    # the 2002 paper prints '(' as '…' and ')' as '†', around the equation number
    assert eq.layer_garbled("…4:6†") and eq.layer_garbled("…2:2†")
    assert eq.layer_garbled("r2 …4:6†")
    # a healthy ellipsis in a range of subscripts and a conjugate transpose are not markers
    assert not eq.layer_garbled("x_1, …, x_n")
    assert not eq.layer_garbled("P…Q")
    assert not eq.layer_garbled("A†")
    assert not eq.layer_garbled("A^†")
    # the guillemets are markers wherever they appear
    assert eq.layer_garbled("‹7›") and eq.layer_garbled("›")


def test_an_eqno_number_counts_as_written_and_the_wclpvyve_box_passes():
    # WCLPVYVE p9: the printed tag is '...4:6†' -- '(' is the font's ellipsis, ')' its dagger and
    # ':' its decimal point -- and the number the model wrote as \eqno(4.6) is out of the formula by
    # the time the check runs, so it counts as written only if the tag is credited.
    raw = (
        "$$r_i^2 = (\\hat{y}_i - \\bar{\\hat{y}}_i)^2 / \\hat{\\sigma}^2 \\eqno(4.6)$$"
    )
    entries, _ = eq.split_math(raw)
    assert entries[0]["tag"] == "4.6"
    words = ["r2", "i", "ˆ", "…yi", "¡", "^yi†2=^s2", "…4:6†"]
    assert eq.number_check(words, [entries[0]["latex"]]) == ["4.6"]
    assert eq.number_check(words, [entries[0]["latex"]], ["4.6"]) == []
    # only the number the model actually wrote is credited
    assert eq.number_check(words, [entries[0]["latex"]], ["4.7"]) == ["4.6"]
    assert (
        eq.verify(eq._flat(" ".join(words)), words, [entries[0]["latex"]], ["4.6"])
        == ""
    )


def test_a_number_in_a_broken_font_is_read_from_the_layer_and_the_box_verifies(
    tmp_path,
):
    # the same eth/thorn pair through the whole pipeline: the tag is theirs, the formula is OCR's
    layer = "x2 + y2 = z2                  ð7Þ"
    out, e = run_apply(
        page_with_equation(tmp_path, layer=layer),
        chunk_for(),
        lambda p: f"${LATEX}$. \\qquad (7)",
    )
    assert f"${LATEX} \\tag{{7}}$" in out, out
    assert e.stats["ocr"] == 1 and e.stats["pair_order"] == 1, e.stats


def test_a_ragged_pair_of_numbers_in_a_broken_layer_still_verifies():
    # TCZLPVMV p26: the tag is '(1–30)' — an en dash — so it used to stay in the words and '30'
    # was wanted; the OCR wrote "\\tag{1-30}" beside the formula, spaced out.
    layer = "ds2 = dr2 + r2 dθ2 + r2 sin2θ dλ2 . (1–30)"
    latex = "d s ^ { 2 } = d r ^ { 2 } + r ^ { 2 } \\, d \\vartheta ^ { 2 } + r ^ { 2 } \\sin ^ { 2 } \\! \\vartheta \\; d \\lambda ^ { 2 } \\, ."
    assert eq.tag_of("(1–30)") == "1-30"
    words = [
        w for w in layer.split() if not eq.tag_of(w)
    ]  # the tag word never reaches the check
    assert eq.verify(eq._flat(" ".join(words)), words, [latex]) == ""
    assert eq.strip_own_number(latex + " \\qquad \\qquad ( 1 - 3 0 )", "1-30") == latex


def test_the_hand_audited_genuine_ocr_errors_stay_rejected():
    # 3RM7ZTU7 p3: the OCR put the product under a fraction bar and invented a \\cdot
    words = ["SNRR,i", "=", "IPR,i", "2", "+", "QPR,i", "2", "·", "√Fs", "PN,i"]
    snr = "\\text{SNR}_{R,i} = \\frac{\\sqrt{\\text{IP}_{R,i}{}^2 + \\text{QP}_{R,i}{}^2}}{\\cdot\\sqrt{F_s}}"
    assert eq.verify(eq._flat(" ".join(words)), words, [snr]) == "letters 15/18"
    # 3RM7ZTU7 p3: subscript I read as 1
    words = ["sTEC", "j", "R", "=", "sTEC", "j", "1", "+", "sTEC", "j", "2"]
    stec = "\\text{sTEC}_R^j = \\text{sTEC}_I^j + \\text{sTEC}_{2}^j"
    assert eq.verify(eq._flat(" ".join(words)), words, [stec]) == "numbers missing: 1"
    # ZHT6NICB p4: the derivative is with respect to θ_i; the OCR read the printed θ as v
    words = ["Ci", "=", "∂C(θ)", "∂ϑi", "."]
    ci = "\\mathbf{C}_i = \\frac{\\partial\\mathbf{C}(\\boldsymbol{\\theta})}{\\partial v_i}."
    assert eq.verify(eq._flat(" ".join(words)), words, [ci]) == "letters 5/6"
    # 3VKPNJIQ p9: the layer's subscript 1 read as a capital I (the tag was 'A1', untagged)
    words = ["xi", "xT", "bi", "i,", "A1"]
    xf = "x_i = x_T + b_i + \\varepsilon_i,"
    assert eq.verify(eq._flat(" ".join(words)), words, [xf]) == "numbers missing: 1"
    # GJJX2V4E p4: the layer says zeta where the OCR wrote xi (unclear, so still rejected)
    words = [
        "ζP(ϕ,", "λ,rP)", "=", "ζ0(ϕ,", "λ,rE)", "+", "K", "k=1", "1", "k!", "∂kζ",
        "∂rk", "rE", "hk", "+", "C1B(ϕ,", "λ,", "h)",
    ]  # fmt: skip
    xi = "\\xi_P(\\varphi,\\lambda,r_P)=\\xi_0(\\varphi,\\lambda,r_{\\mathrm{E}})+\\left\\{\\sum_{k=1}^{K}\\frac{1}{k!}\\left.\\frac{\\partial^k\\xi}{\\partial r^k}\\right|_{r_{\\mathrm{E}}}h^k\\right\\}+C_{1\\mathrm{B}}(\\varphi,\\lambda,h)"
    assert eq.verify(eq._flat(" ".join(words)), words, [xi]) == "letters 24/27"


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
        "join_label_lines _ROMAN _TAG_WORD _MATH _TAG_CMD _EQNO layer_garbled _GLYPH_MARKS _ELISION_OPEN _ELISION_CLOSE"
    ).split()
    mine = os.path.abspath(eq.__file__)
    here = os.path.dirname(mine)
    for theirs in (
        os.path.join(
            here, "mcp", "paper_pdf_tools.py"
        ),  # bundle layout: beside pdf_equations.py
        os.path.join(
            here, "..", "mcp", "paper_pdf_tools.py"
        ),  # fork layout: corpus/../mcp
    ):
        if os.path.exists(theirs):
            break
    else:
        reason = f"no mcp/paper_pdf_tools.py in or beside {here}"
        try:
            import pytest
        except ImportError:  # the plain runner the installer uses has no pytest
            print("skip:", reason)
            return
        pytest.skip(reason)
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
