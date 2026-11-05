#!/usr/bin/env python3
"""
pdf_equations.py: display equations and merged table rows for build_corpus.py (2026-11-05).

pymupdf4llm's layout model labels every display equation a `formula` box, and the Markdown
writer turns that box into a blank line (1.28, images off) or a picture-text block of
unmapped glyphs (2.0), so a born-digital paper loses every numbered equation on its way into
the knowledge base. This module puts them back, page by page, from the data build_corpus
already has:

  * Where: each page chunk's `page_boxes` list gives every box's class, bbox and `pos`, the
    (start, stop) of that box's text inside the chunk's Markdown. A formula box's slice is the
    blank line (1.28) or the marker block (2.0); it is replaced in place. No `write_images`
    link matching, no dependence on 2.0's markers; the same on both versions.
  * What: the box is cropped at 300 dpi and sent to an OCR client (a callable PNG -> LaTeX
    text; build_corpus's own OCR lane in production, an HTTP server for validation). The
    equation number comes from the page's text layer (a "(N)" word that ends a line inside the
    box, at its right edge) as \\tag{N}; the model leaves numbers out.
  * Check: the LaTeX must contain every number and (almost) every letter of the box's text
    layer, with balanced braces. A layer whose font mapping corrupts what the checks read is
    normalised where the corruption is known (`=` printed as `1⁄4`, `(N)` as `ðNÞ` or an en
    dash in the number, a colon for the decimal point, `q` for θ); the checks themselves are
    not loosened. A formula that fails the check, a box too big to be one
    equation, an OCR failure or no OCR client at all is written as the box's text-layer text.
    Nothing is written for a box without a text layer unless the check can run: a blank line
    stays a blank line, but an OCR guess that nothing can contradict is never written.
  * Format: ONE `$...$` line per equation (never `$$`: the server's Markdown parser turns a
    `$$` block into its own LLM-analysed item), glued to the paragraph that leads into it.

Table rows: pymupdf writes a table without row rules as one row whose cells hold every line,
the breaks as `<br>`. split_pipe_tables() splits such a row into real rows when the first cell
holds several labels (wrapped and unit lines joined) and every other cell holds exactly as
many lines; any other row keeps its `<br>`.

The helpers marked "shared" are copies of zotero-toolkit/mcp/paper_pdf_tools.py (the MCP
proxy's paper_equation / paper_table); tests/corpus/test_pdf_equations.py compares them with
that file so the two cannot drift. Standard library only; pymupdf objects are passed in.

Config (environment, read at import; build_corpus passes nothing else):
    BUILD_EQUATIONS    auto (OCR when a lane is free) | text (text layer only) | off (as before), default auto
    BUILD_EQ_MAX       formula boxes per document that may be sent to OCR, default 300 (the rest: text layer)
    BUILD_EQ_OCR_URL   an OpenAI-compatible server to use instead of build_corpus's OCR lane (validation)
"""

from __future__ import annotations

import base64
import collections
import json
import os
import re
import time
import unicodedata
import urllib.request

EQ_MODE = os.environ.get("BUILD_EQUATIONS", "auto").strip().lower()
EQ_MAX = int(os.environ.get("BUILD_EQ_MAX", "300"))
EQ_OCR_URL = os.environ.get("BUILD_EQ_OCR_URL", "").strip()
EQ_CROP_S = 3.0  # seconds per crop on the GPU lane (computed from 86 tok/s decode, ~230 tokens); a lane-time estimate only
EQ_DPI = 300
EQ_MARGIN_PT = 4
EQ_MAX_SIDE_PX = 2400  # a very wide box is rendered at a lower dpi than 300
EQ_MAX_LAYER_WORDS = (
    150  # a "formula" box with more words than this is not one equation
)
EQ_MAX_HEIGHT_FRAC = 0.45  # ... nor one taller than this share of the page
EQ_LETTER_MIN = 0.90  # share of the layer's letters the LaTeX must contain (below 10 letters: all of them)
EQ_TEXT_CAP = 600  # characters of a text-layer fallback line
EQ_SCAN_COVER = 0.7  # a page this covered by images is a scan: its formulas are left to the whole-page OCR route
EQ_FAIL_RUN = (
    3  # this many OCR failures in a row and the rest of the document goes without OCR
)
EQ_MAX_TOKENS = 1200

# The model card's instruction, verbatim (the same as build_corpus's OCR_VLM_PROMPT; a test compares them).
EQ_PROMPT = (
    "You are an advanced hybrid OCR engine capable of processing multilingual text mixed with mathematical "
    "notation. Your goal is to transcribe the content with high fidelity.Strict Rules: 1. Multilingual Precision: "
    "Transcribe text exactly as it appears in the original language. Do not translate, summarize, or correct "
    "original spelling errors. 2. Math Formatting: Identify all mathematical expressions and convert them into "
    "LaTeX. 3. Use single dollar signs ($x$) for inline math (formulas within a sentence). 4. Use double dollar "
    "signs ($$x$$) for display math (standalone formulas on their own lines). 5. Layout & Structure: Use Markdown "
    "to preserve the visual structure (headers, paragraphs, lists). 6. Output Only: Output the transcribed text "
    "directly without any conversational filler."
)


class OcrFailed(RuntimeError):
    """The OCR client could not transcribe this crop (the document goes on with the text layer for it)."""


# ---------------------------------------------------------------- shared with mcp/paper_pdf_tools.py

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_TAG_WORD = re.compile(
    r"^\(\s*([A-Za-z]{0,2}[.\-]?\d{1,3}(?:[.\-]\d{1,3})*[a-z]?)\s*\)$"
)


def norm_number(n, roman: bool = False) -> str:
    """'(6)', 'Eq. 6', ' 6 ' -> '6'; 'A.1' -> 'a.1'. With roman=True 'II' -> '2'."""
    s = unicodedata.normalize("NFKC", str(n)).strip()
    s = re.sub(r"^(?:eqs?|equations?|tables?|tab)\b\.?\s*", "", s, flags=re.I)
    s = s.strip("()[] ").replace("\u2013", "-").replace("\u2212", "-")
    s = re.sub(r"\s+", "", s).lower()
    if roman and re.fullmatch(r"[ivxlc]+", s):
        total, prev = 0, 0
        for ch in reversed(s):
            v = _ROMAN[ch]
            total += v if v >= prev else -v
            prev = max(prev, v)
        return str(total)
    return s


def tag_of(word: str) -> str | None:
    """The equation number inside a text-layer word like '(6)', else None. A broken font maps the
    parentheses to other glyphs ('ð7Þ', '‹7›') and the hyphen to an en dash ('(1–30)')."""
    for glyph, plain in (
        ("ð", "("),
        ("Þ", ")"),
        ("‹", "("),
        ("›", ")"),
        ("\u2013", "-"),
    ):
        word = word.replace(glyph, plain)
    m = _TAG_WORD.match(word.strip())
    return m.group(1) if m else None


def _flat(s: str) -> str:
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", unicodedata.normalize("NFKC", s))
    return " ".join(s.split())


_MATH = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?)\}(?P<e>.+?)\\end\{(?P=env)\}"
    r"|\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|\$(?P<d>[^$]+?)\$",
    re.S,
)
_TAG_CMD = re.compile(r"\\tag\*?\{([^}]*)\}")
_EQNO = re.compile(
    r"\\eqno\s*[({]*\s*([A-Za-z]{0,2}[.\-]?\d{1,3}(?:[.\-]\d{1,3})*[a-z]?)\s*[)}]*\s*$"
)


def split_math(text: str) -> tuple[list[dict], bool]:
    """The display formulas of an OCR transcription, in order, as [{'latex', 'tag'}] (a
    \\tag the model wrote is taken out and kept in 'tag'); and whether it had any math
    delimiters. A `$...$` inside a line of prose ("(c) Mantle for $n=0$") is not a display
    formula and is left out; the spacing commands and the equation number the model leaves
    beside it ("$...$. \\qquad (2)") are not prose, so that formula is kept."""
    out, had = [], False
    for m in _MATH.finditer(text):
        had = True
        if m.group("d") is not None:
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            line = text[ls : len(text) if le < 0 else le]
            rest = re.sub(
                r"\\(?:q?quad|hfill|eqno|tag\*?\{[^}]*\}|hspace\*?\{[^}]*\})",
                " ",
                _MATH.sub(" ", line),
            )
            rest = " ".join(w for w in rest.split() if not tag_of(w))
            if re.search(r"[^\W\d_]{2,}", rest):
                continue
        body = next(
            m.group(g) for g in ("e", "dd", "br", "d") if m.group(g) is not None
        )
        tag = _TAG_CMD.search(body)
        body = _TAG_CMD.sub("", body)
        if tag is None:
            tag = _EQNO.search(body)
        body = _EQNO.sub("", body)
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        body = " ".join(body.split())
        if body:
            out.append({"latex": body, "tag": tag.group(1).strip() if tag else None})
    if had:
        return out, True
    paras = [
        " ".join(p.split()) for p in re.split(r"\n\s*\n", text.strip()) if p.strip()
    ]
    return [{"latex": p, "tag": None} for p in paras], False


def tagged(latex: str, number: str) -> str:
    return f"${latex} \\tag{{{number}}}$"


def number_check(layer_words: list[str], latexes: list[str]) -> list[str]:
    """Numbers that the text layer has and the transcription lacks (a misread digit). The layer is
    normalised first for the two glyph corruptions of a broken font, and nothing else: U+2044, the
    fraction glyph this font prints for '=' ('1⁄4' manufactures a 1 and a 4), and a colon between
    digits, which is its decimal point ('77:689' is 77.689)."""
    have = set(re.findall(r"\d+(?:\.\d+)?", " ".join(latexes)))
    layer = re.sub(r"\d*\u2044\d*", " ", _flat(" ".join(layer_words)))
    layer = re.sub(r"(?<=\d):(?=\d)", ".", layer)
    want = set(re.findall(r"\d+(?:\.\d+)?", layer))
    return sorted(want - have, key=lambda x: (len(x), x))


def line_end_words(words: list) -> list:
    """Words (x0, y0, x1, y1, text, block, line, n) that end their text line."""
    last: dict = {}
    for w in words:
        k = (w[5], w[6])
        if k not in last or w[7] > last[k][7]:
            last[k] = w
    return list(last.values())


def tags_on_page(words: list) -> list[tuple[str, tuple]]:
    """[(number, word)] of the '(N)' words that end a text line."""
    return [(t, w) for w in line_end_words(words) if (t := tag_of(w[4]))]


def _center(w):
    return ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)


def words_in(words: list, rect) -> list:
    return [
        w
        for w in words
        if rect.x0 - 1 <= _center(w)[0] <= rect.x1 + 1
        and rect.y0 - 1 <= _center(w)[1] <= rect.y1 + 1
    ]


def join_label_lines(lines: list[str]) -> list[str]:
    """Label lines with wrapped and unit lines joined to the line above: a line that
    starts with '(' or a lower-case letter (not 'a)') continues the previous one."""
    out: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if out and (
            ln.startswith("(") or (ln[:1].islower() and not re.match(r"^[a-z]\)", ln))
        ):
            out[-1] += " " + ln
        elif ln:
            out.append(ln)
    return out


# ---------------------------------------------------------------- checking a transcription

# layer glyph -> the LaTeX command names that may stand for it
_GREEK = {
    "α": ("alpha",), "β": ("beta",), "γ": ("gamma",), "δ": ("delta",),
    "ε": ("epsilon", "varepsilon"), "ϵ": ("epsilon", "varepsilon"), "ζ": ("zeta",),
    "η": ("eta",), "θ": ("theta", "vartheta"), "ϑ": ("theta", "vartheta"), "ι": ("iota",),
    "κ": ("kappa", "varkappa"), "λ": ("lambda",), "μ": ("mu",), "µ": ("mu",), "ν": ("nu",),
    "ξ": ("xi",), "π": ("pi", "varpi"), "ϖ": ("pi", "varpi"), "ρ": ("rho", "varrho"),
    "ϱ": ("rho", "varrho"), "σ": ("sigma", "varsigma"), "ς": ("sigma", "varsigma"),
    "τ": ("tau",), "υ": ("upsilon",), "φ": ("phi", "varphi"), "ϕ": ("phi", "varphi"),
    "χ": ("chi",), "ψ": ("psi",), "ω": ("omega",),
    "Γ": ("Gamma",), "Δ": ("Delta",), "Θ": ("Theta",), "Λ": ("Lambda",), "Ξ": ("Xi",),
    "Π": ("Pi",), "Σ": ("Sigma",), "Υ": ("Upsilon",), "Φ": ("Phi",), "Ψ": ("Psi",), "Ω": ("Omega",),
}  # fmt: skip

# A broken font's substitution for a letter, beside the Greek ones above: the layer prints this
# ASCII letter where the page has the symbol, so the LaTeX holds one of these commands instead.
_GLYPH_CMDS = {"q": ("theta", "vartheta"), "S": ("mathcal",)}


def letter_recall(layer_text: str, latexes: list[str]) -> tuple[int, int]:
    """(letters found, letters wanted): the ASCII letters (case-folded, multiset) and the Greek
    letters of the text layer that the LaTeX holds; a Greek letter counts when the LaTeX has one
    of its commands. Glyphs the font could not map (U+FFFD) and other symbols are not counted."""
    layer = unicodedata.normalize("NFKC", layer_text)
    latex = "\n".join(latexes)
    want = collections.Counter(c.lower() for c in layer if c.isascii() and c.isalpha())
    have = collections.Counter(c.lower() for c in latex if c.isascii() and c.isalpha())
    cmds = collections.Counter(re.findall(r"\\([A-Za-z]+)", latex))
    for glyph, names in _GLYPH_CMDS.items():
        if glyph in layer:
            have[glyph.lower()] += sum(cmds[v] for v in names)
    wanted = hit = 0
    for ch, n in collections.Counter(c for c in layer if c in _GREEK).items():
        wanted += n
        hit += min(n, sum(cmds[v] for v in _GREEK[ch]))
    for ch, n in want.items():
        wanted += n
        hit += min(n, have[ch])
    return hit, wanted


def balanced(latex: str) -> bool:
    depth = 0
    for m in re.finditer(r"\\.|[{}]", latex):
        if m.group(0) == "{":
            depth += 1
        elif m.group(0) == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def verify(layer_text: str, layer_words: list[str], latexes: list[str]) -> str:
    """'' when the transcription holds the box's text layer, else why not."""
    if not latexes:
        return "no formula in the transcription"
    for lx in latexes:
        if "$" in lx or not balanced(lx):
            return "malformed LaTeX"
    missing = number_check(layer_words, latexes)
    if missing:
        return "numbers missing: " + ", ".join(missing[:4])
    hit, wanted = letter_recall(layer_text, latexes)
    if wanted and wanted - hit > int(round((1 - EQ_LETTER_MIN) * wanted, 6)):
        return f"letters {hit}/{wanted}"
    return ""


def strip_own_number(latex: str, number: str) -> str:
    """`latex` without a trailing "(N)" (and the spacing commands before it) when N is the equation's
    number: the model sometimes copies the printed number into the formula, and \\tag{N} must not repeat it.
    The copy may have spaces between its characters ('( 1 - 3 0 )')."""
    n = r"\s*".join(re.escape(c) for c in norm_number(number))
    if re.fullmatch(r"\s*\(\s*" + n + r"\s*\)\s*", latex):
        return ""  # nothing but the number: no formula at all
    return re.sub(
        r"(?:\\(?:q?quad|hfill|hspace\*?\{[^}]*\}|[;:,! ])|\s)+\(\s*"
        + n
        + r"\s*\)\s*$",
        "",
        latex,
        flags=re.I,
    ).rstrip()


def assemble(entries: list[dict], box_tags: list[str]) -> tuple[list[str], str]:
    """(lines of `$...$`, how): a box's formulas paired with the equation numbers its text layer
    carries. As many numbers as formulas pair by order; one number is the box's last formula (a
    long equation the model wrote as several lines); no number leaves them unnumbered; any other
    count writes the formulas without numbers. A formula that was only its own number is dropped
    (no lines: the box falls back to its text layer)."""
    n, t = len(entries), len(box_tags)
    if not n:
        return [], "none"
    if t == 0:
        return [f"${e['latex']}$" for e in entries], "unnumbered"
    if t == n:
        pairs = [
            (strip_own_number(e["latex"], box_tags[i]), box_tags[i])
            for i, e in enumerate(entries)
        ]
        return [tagged(lx, k) for lx, k in pairs if lx], "order"
    if t == 1:
        lines = [f"${e['latex']}$" for e in entries[:-1]]
        last = strip_own_number(entries[-1]["latex"], box_tags[0])
        if last:
            lines.append(tagged(last, box_tags[0]))
        return lines, "single-number"
    return [f"${e['latex']}$" for e in entries], "unpaired"


# ---------------------------------------------------------------- OCR clients


def http_client(base: str, model: str | None = None, timeout: float = 300.0):
    """A client for an OpenAI-compatible server at `base` (a llama-server, validation): PNG -> text.
    Raises OcrFailed when the request fails or the model does not finish."""

    def client(png: bytes) -> str:
        body = {
            "temperature": 0, "repeat_penalty": 1.15, "max_tokens": EQ_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": EQ_PROMPT},
            ]}],
        }  # fmt: skip
        if model:
            body["model"] = model
        req = urllib.request.Request(
            base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ch = json.load(r)["choices"][0]
        except Exception as e:
            raise OcrFailed(f"request failed ({type(e).__name__})") from e
        text = ((ch.get("message") or {}).get("content") or "").strip()
        if ch.get("finish_reason") != "stop" or not text:
            raise OcrFailed(f"the model did not finish ({ch.get('finish_reason')})")
        return text

    return client


# ---------------------------------------------------------------- the pages


_SLOT_OK = re.compile(
    r"^\s*(?:<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->|!\[[^\]]*\]\([^)]*\))?\s*$",
    re.S | re.I,
)
_GLUE = (
    "text",
    "list-item",
    "formula",
)  # a formula follows these inside their paragraph


def count_formulas(chunks) -> int:
    """Formula boxes over all page chunks (dict chunks with page_boxes only)."""
    return sum(
        1
        for ch in chunks
        if isinstance(ch, dict)
        for b in ch.get("page_boxes") or []
        if isinstance(b, dict) and b.get("class") == "formula"
    )


def lane_pages(n_formulas: int, seconds_per_page: float) -> int:
    """The lane-time estimate for n crops, in whole-page units for build_corpus's lane check."""
    return max(1, -(-int(n_formulas * EQ_CROP_S) // max(1, int(seconds_per_page))))


class Equations:
    """Formula boxes of one document: page by page, their slice of the chunk text replaced.

    `client`: PNG bytes -> transcription text, raising OcrFailed (None: no OCR, text layer only);
    any other exception it raises (a lane that went away: build_corpus's VlmDeferred) propagates."""

    def __init__(self, doc, client=None, max_ocr: int | None = None, pymupdf=None):
        self.doc, self.client = doc, client
        self.max_ocr = EQ_MAX if max_ocr is None else max_ocr
        self.stats: collections.Counter = collections.Counter()
        self.reasons: collections.Counter = collections.Counter()
        self.seconds = 0.0
        self._pm = pymupdf
        self._words: dict = {}
        self._fail_run = 0

    def _mu(self):
        if self._pm is None:
            import pymupdf

            self._pm = pymupdf
        return self._pm

    def _scanned(self, pno: int) -> bool:
        page = self.doc[pno]
        try:
            area = abs(page.rect) or 1.0
            pm = self._mu()
            return (
                sum(abs(pm.Rect(i["bbox"])) for i in page.get_image_info()) / area
                >= EQ_SCAN_COVER
            )
        except Exception:
            return False

    def _formula_boxes(self, chunk, text: str, pno: int | None = None) -> list[dict]:
        """Formula boxes whose slice of `text` is a blank line, a marker block or an image link
        (none on a scanned page)."""
        if not isinstance(chunk, dict) or (pno is not None and self._scanned(pno)):
            return []
        out = []
        for b in chunk.get("page_boxes") or []:
            if not (
                isinstance(b, dict) and b.get("class") == "formula" and b.get("bbox")
            ):
                continue
            pos = b.get("pos")
            if not (pos and 0 <= pos[0] <= pos[1] <= len(text)):
                continue
            if _SLOT_OK.match(text[pos[0] : pos[1]]):
                out.append(b)
        return out

    def _page_words(self, pno: int):
        if pno not in self._words:
            self._words = {pno: self.doc[pno].get_text("words")}
        return self._words[pno]

    def _box_layer(self, pno: int, rect):
        """(words of the box without its equation numbers, [(number, word)] at its right edge, text)."""
        inside = words_in(self._page_words(pno), rect)
        tw = tags_on_page(inside)
        right = [
            x for x in tw if x[1][2] >= rect.x1 - max(30.0, 0.15 * rect.width)
        ] or tw
        tw = sorted(right, key=lambda x: _center(x[1])[1])
        tag_ids = {id(w) for _, w in tw}
        rest = [w for w in inside if id(w) not in tag_ids]
        text = _flat(self.doc[pno].get_text("text", clip=rect))
        return rest, tw, text

    def recovered_chars(self, pno: int, chunk, text: str) -> int:
        """Non-space characters of the text layer inside this page's formula boxes: what the page
        text will hold again once the boxes are written (the text-layer repair gate counts them)."""
        n = 0
        for b in self._formula_boxes(chunk, text, pno):
            rect = self._mu().Rect(b["bbox"])
            n += sum(
                len("".join(w[4].split()))
                for w in words_in(self._page_words(pno), rect)
            )
        return n

    def _crop(self, pno: int, rect) -> bytes:
        clip = rect + (-EQ_MARGIN_PT, -EQ_MARGIN_PT, EQ_MARGIN_PT, EQ_MARGIN_PT)
        dpi = EQ_DPI
        side = max(clip.width, clip.height) * dpi / 72.0
        if side > EQ_MAX_SIDE_PX:
            dpi = max(72, int(dpi * EQ_MAX_SIDE_PX / side))
        return self.doc[pno].get_pixmap(clip=clip, dpi=dpi).tobytes("png")

    def _lines(self, pno: int, box: dict) -> list[str]:
        """The Markdown lines for one formula box ([] leaves its slice as it was)."""
        pm = self._mu()
        rect = pm.Rect(box["bbox"])
        page = self.doc[pno]
        words, tw, text = self._box_layer(pno, rect)
        self.stats["boxes"] += 1
        if not words and not tw:
            self.reasons["no text layer"] += 1
            return []
        fallback = [text[:EQ_TEXT_CAP]] if text else []
        too_big = (
            len(words) > EQ_MAX_LAYER_WORDS
            or rect.height > EQ_MAX_HEIGHT_FRAC * page.rect.height
        )
        if self._fail_run >= EQ_FAIL_RUN:
            self.client = None
        if self.client is None or too_big or self.stats["ocr_tried"] >= self.max_ocr:
            why = (
                ("OCR lane failing" if self._fail_run else "no OCR")
                if self.client is None
                else "too big for one equation"
                if too_big
                else "per-document OCR budget"
            )
            self.reasons[why] += 1
            self.stats["text_layer"] += 1
            return fallback
        self.stats["ocr_tried"] += 1
        t0 = time.monotonic()
        try:
            raw = self.client(self._crop(pno, rect))
        except OcrFailed as e:
            self._fail_run += 1
            self.seconds += time.monotonic() - t0
            self.reasons["OCR failed: " + str(e)[:40]] += 1
            self.stats["text_layer"] += 1
            return fallback
        self.seconds += time.monotonic() - t0
        self._fail_run = 0
        entries, delimited = split_math(raw)
        why = "" if delimited else "no math delimiters"
        layer_words = [w[4] for w in words]
        why = why or verify(
            _flat(" ".join(layer_words)), layer_words, [e["latex"] for e in entries]
        )
        if why:
            self.reasons["unverified: " + " ".join(why.split(":")[0].split()[:2])] += 1
            self.stats["text_layer"] += 1
            return fallback
        lines, how = assemble(entries, [t for t, _ in tw])
        if not lines:
            self.reasons["only the equation number"] += 1
            self.stats["text_layer"] += 1
            return fallback
        self.stats["ocr"] += 1
        self.stats["eq_lines"] += len(lines)
        self.stats["pair_" + how] += 1
        return lines

    def apply(self, pno: int, chunk, text: str) -> str:
        """`text` (a page chunk's Markdown) with each formula box's slice replaced."""
        boxes = self._formula_boxes(chunk, text, pno)
        if not boxes:
            return text
        allboxes = [
            b for b in chunk["page_boxes"] if isinstance(b, dict) and b.get("pos")
        ]
        try:
            for b in sorted(boxes, key=lambda b: b["pos"][0], reverse=True):
                s, e = b["pos"]
                lines = self._lines(pno, b)
                if not lines:
                    continue
                prev = next(
                    (
                        x.get("class")
                        for x in sorted(
                            allboxes, key=lambda x: x["pos"][1], reverse=True
                        )
                        if x["pos"][1] <= s and x["pos"][1] > x["pos"][0]
                    ),
                    None,
                )
                head = text[:s]
                if prev in _GLUE and head.endswith("\n\n"):
                    head = head.rstrip("\n") + "\n"
                text = head + "\n".join(lines) + "\n\n" + text[e:]
        finally:
            self._words.pop(pno, None)
        return text

    def summary(self) -> str:
        s = self.stats
        parts = [f"{s['boxes']} formula box(es)", f"{s['ocr']} transcribed and checked"]
        if s["text_layer"]:
            parts.append(f"{s['text_layer']} as text layer")
        if self.reasons:
            parts.append(
                "("
                + ", ".join(f"{k}: {v}" for k, v in sorted(self.reasons.items()))
                + ")"
            )
        pairs = {k[5:]: v for k, v in s.items() if k.startswith("pair_")}
        if pairs:
            parts.append(
                "numbers " + ", ".join(f"{k} {v}" for k, v in sorted(pairs.items()))
            )
        if self.seconds:
            parts.append(f"OCR {self.seconds:.0f} s")
        return "; ".join(parts)


# ---------------------------------------------------------------- tables

_SEP_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_BR = re.compile(r"<br\s*/?>", re.I)


def _is_row(ln: str) -> bool:
    s = ln.strip()
    return s.startswith("|") and s.endswith("|")


def _cells(line: str) -> list[str]:
    s = line.strip()[1:-1]
    return re.split(r"(?<!\\)\|", s)


def _cell_lines(cell: str) -> list[str]:
    return [p.strip() for p in _BR.split(cell) if p.strip()]


def split_pipe_tables(md: str) -> tuple[str, int, int]:
    """(md, rows split, rows left merged): in each pipe table (header row, `|---|` row, data rows) a
    row whose first cell holds several labels (wrapped and unit lines joined) and whose other
    non-empty cells each hold as many lines becomes one row per label. Other rows are untouched
    (their `<br>` stays). A row is only touched when it holds a `<br>`."""
    lines = md.split("\n")
    out: list[str] = []
    split = merged = 0
    i = 0
    while i < len(lines):
        if (
            _is_row(lines[i])
            and i + 1 < len(lines)
            and _is_row(lines[i + 1])
            and _SEP_ROW.match(lines[i + 1])
        ):
            j = i + 2
            while j < len(lines) and _is_row(lines[j]):
                j += 1
            out += lines[i : i + 2]
            for ln in lines[i + 2 : j]:
                if not _BR.search(ln):
                    out.append(ln)
                    continue
                cells = _cells(ln)
                cl = [_cell_lines(c) for c in cells]
                labels = join_label_lines(cl[0]) if cl else []
                counts = {len(c) for c in cl[1:] if c}
                if len(labels) > 1 and counts == {len(labels)}:
                    for k, lab in enumerate(labels):
                        out.append(
                            "|"
                            + "|".join([lab] + [c[k] if c else "" for c in cl[1:]])
                            + "|"
                        )
                    split += 1
                else:
                    out.append(ln)
                    merged += 1
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), split, merged
