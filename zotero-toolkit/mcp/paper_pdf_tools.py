#!/usr/bin/env python3
"""
paper_pdf_tools.py: the PDF work behind the MCP proxy's paper_equation and paper_table.

The proxy (lightrag_mcp_server.py) runs under an interpreter without PyMuPDF, so it
calls this script with the one that has it, passes a mode, a PDF and a number, and reads
one JSON object from stdout. Everything else (layout model, text layer, OCR) lives here.

    paper_pdf_tools.py equation PDF NUMBER [--page N]
    paper_pdf_tools.py table PDF NUMBER

equation: pages with a text line ending "(NUMBER)" are run through PyMuPDF's layout
model (one page at a time, cached per file); the formula box whose text layer holds that
number is cropped at 300 dpi and transcribed to LaTeX by an OCR model on a CPU-only
llama-server that is started for the call and stopped after it. The equation number
comes from the text layer, as \\tag{N}. When no OCR model is available, or it fails or
times out, the answer is the box's text layer and says so.

table: finds the "Table N" caption, the nearest layout table box and the pipe table the
layout converter wrote for it; cell line breaks stay as <br>, and a row of several
labelled lines is split into rows when every column has as many lines as there are
labels. No OCR model is involved.

Pages too heavy for the layout model (it holds some 25 N x N matrices for N text spans)
are never given to it. Config (environment variables, all optional):
    LIGHTRAG_MCP_CACHE              layout / OCR cache directory
    LIGHTRAG_MCP_OCR                "off" disables the OCR model
    LIGHTRAG_MCP_OCR_TIMEOUT        seconds for server start + transcription, default 120
    LIGHTRAG_MCP_OCR_LLAMA_SERVER   llama-server binary
    LIGHTRAG_MCP_OCR_MODEL          the OCR model's GGUF
    LIGHTRAG_MCP_OCR_MMPROJ         its multimodal projector GGUF
    LIGHTRAG_MCP_OCR_THREADS        CPU threads, default 8
    LIGHTRAG_MCP_PDF_MEM_GIB        address-space ceiling for this process, default 7
Deps: pymupdf and pymupdf4llm with its layout package (never imported by the proxy).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request

CACHE_DIR = os.path.expanduser(
    os.environ.get("LIGHTRAG_MCP_CACHE")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache", "lightrag-mcp-pdf")
)
# No defaults for the OCR model: set these to enable it (without them, paper_equation answers
# with the text layer and says so). llama-server is looked up on PATH.
LLAMA_SERVER = os.environ.get("LIGHTRAG_MCP_OCR_LLAMA_SERVER") or (
    shutil.which("llama-server") or ""
)
OCR_MODEL = os.environ.get("LIGHTRAG_MCP_OCR_MODEL") or ""
OCR_MMPROJ = os.environ.get("LIGHTRAG_MCP_OCR_MMPROJ") or ""
OCR_THREADS = int(os.environ.get("LIGHTRAG_MCP_OCR_THREADS") or "8")
OCR_BUDGET_S = float(os.environ.get("LIGHTRAG_MCP_OCR_TIMEOUT") or "120")
OCR_MAX_TOKENS = 1200
OCR_DPI = 300
OCR_MARGIN_PT = 4
MEM_GIB = float(os.environ.get("LIGHTRAG_MCP_PDF_MEM_GIB") or "7")

# Layout-model guards, as build_corpus has them (2026-10-15 OOM): spans, content bytes.
LAYOUT_MAX_SPANS = 3500
LAYOUT_MAX_CONTENT_BYTES = 30 * 2**20

# The model card's instruction, verbatim (the same as build_corpus's OCR_VLM_PROMPT).
OCR_PROMPT = (
    "You are an advanced hybrid OCR engine capable of processing multilingual text mixed with mathematical "
    "notation. Your goal is to transcribe the content with high fidelity.Strict Rules: 1. Multilingual Precision: "
    "Transcribe text exactly as it appears in the original language. Do not translate, summarize, or correct "
    "original spelling errors. 2. Math Formatting: Identify all mathematical expressions and convert them into "
    "LaTeX. 3. Use single dollar signs ($x$) for inline math (formulas within a sentence). 4. Use double dollar "
    "signs ($$x$$) for display math (standalone formulas on their own lines). 5. Layout & Structure: Use Markdown "
    "to preserve the visual structure (headers, paragraphs, lists). 6. Output Only: Output the transcribed text "
    "directly without any conversational filler."
)

LAYOUT_VERSION = "1"
RAW_LIMIT = 1500  # characters of text layer returned


# ---------------------------------------------------------------- numbers and tags

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
    """The equation number inside a text-layer word like '(6)', else None."""
    m = _TAG_WORD.match(word.strip())
    return m.group(1) if m else None


def _flat(s: str) -> str:
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", unicodedata.normalize("NFKC", s))
    return " ".join(s.split())


# ---------------------------------------------------------------- LaTeX entries

_MATH = re.compile(
    r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?)\}(?P<e>.+?)\\end\{(?P=env)\}"
    r"|\$\$(?P<dd>.+?)\$\$"
    r"|\\\[(?P<br>.+?)\\\]"
    r"|\$(?P<d>[^$]+?)\$",
    re.S,
)
_TAG_CMD = re.compile(r"\\tag\*?\{([^}]*)\}")


def split_math(text: str) -> tuple[list[dict], bool]:
    """The display formulas of an OCR transcription, in order, as [{'latex', 'tag'}] (a
    \\tag the model wrote is taken out and kept in 'tag'); and whether it had any math
    delimiters. A `$...$` inside a line of prose ("(c) Mantle for $n=0$") is not a display
    formula and is left out."""
    out, had = [], False
    for m in _MATH.finditer(text):
        had = True
        if m.group("d") is not None:
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            line = text[ls : len(text) if le < 0 else le]
            if re.search(r"[^\W\d_]{2,}", _MATH.sub(" ", line)):
                continue
        body = next(
            m.group(g) for g in ("e", "dd", "br", "d") if m.group(g) is not None
        )
        tag = _TAG_CMD.search(body)
        body = _TAG_CMD.sub("", body)
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


def pick_equation(entries: list[dict], box_tags: list[str], number: str):
    """(lines of `$...$`, how). The formula of equation `number` among a box's formulas.
    A \\tag the model wrote wins; else formulas and numbers pair by order when their
    counts match; a box with one number is one equation, however many formulas the
    model made of it. Otherwise nothing is chosen ('how' says why)."""
    want = norm_number(number)
    for e in entries:
        if e["tag"] and norm_number(e["tag"]) == want:
            return [tagged(e["latex"], e["tag"])], "model-tag"
    norms = [norm_number(t) for t in box_tags]
    if want in norms and len(entries) == len(box_tags):
        i = norms.index(want)
        return [tagged(entries[i]["latex"], box_tags[i])], "order"
    if norms == [want] and entries:
        lines = [f"${e['latex']}$" for e in entries[:-1]]
        lines.append(tagged(entries[-1]["latex"], box_tags[0]))
        return lines, "single-number"
    return [], f"{len(entries)} formulas for {len(box_tags)} equation number(s)"


def number_check(layer_words: list[str], latexes: list[str]) -> list[str]:
    """Numbers that the text layer has and the transcription lacks (a misread digit)."""
    have = set(re.findall(r"\d+(?:\.\d+)?", " ".join(latexes)))
    want = set(re.findall(r"\d+(?:\.\d+)?", _flat(" ".join(layer_words))))
    return sorted(want - have, key=lambda x: (len(x), x))


# ---------------------------------------------------------------- citing sentences

_ABBREV = re.compile(
    r"(?:\b(?:Eqs?|Figs?|Tabs?|Sec|Secs|Ref|Refs|No|Nos|vs|cf|al|approx|Dr|Prof|Vol|pp|ca|resp))\.$"
    r"|\b(?:i\.e|e\.g)\.$"
)
_EQ_REF = re.compile(r"\b(?:Equations?|Eqs?\.?)\s*", re.I)
_REF_TOKENS = re.compile(
    r"(?:\(\s*|\s*)(?P<n>[A-Za-z]{0,2}[.\-]?\d{1,3}(?:\.\d{1,3})*[a-z]?)\s*\)?"
    r"(?:(?P<rng>\s*(?:[\u2013\u2014-]|to)\s*)|(?P<sep>\s*(?:,|;|and|&|or)\s*)|(?=[^\w]|$))"
)


def _refs_after(text: str, start: int) -> set[str]:
    """Equation numbers listed right after an 'Eq.'/'Eqs.'/'Equation': '(5)-(7)', '5 and 6'."""
    nums: set[str] = set()
    pos, prev, rng = start, None, False
    for _ in range(12):
        m = _REF_TOKENS.match(text, pos)
        if not m or m.end() == pos:
            break
        n = norm_number(m.group("n"))
        if (
            rng
            and prev is not None
            and prev.isdigit()
            and n.isdigit()
            and 0 < int(n) - int(prev) < 60
        ):
            nums.update(str(i) for i in range(int(prev), int(n) + 1))
        nums.add(n)
        prev, rng = n, bool(m.group("rng"))
        if not (m.group("rng") or m.group("sep")):
            break
        pos = m.end()
    return nums


def _sentence(text: str, a: int, b: int, reach: int = 700) -> str:
    """The sentence of `text` around [a, b): boundaries are '. ' before a capital, but not
    after an abbreviation such as 'Eq.'"""
    lo = max(0, a - reach)
    start = lo
    for m in re.finditer(r"[.!?][\"')\]]*\s+(?=[A-Z(\[])", text[lo:a]):
        head = text[lo : lo + m.end()].rstrip()
        if not _ABBREV.search(head):
            start = lo + m.end()
    end = min(len(text), b + reach)
    for m in re.finditer(r"[.!?][\"')\]]*(?=\s+[A-Z(\[]|\s*$)", text[b:end]):
        head = text[: b + m.end()]
        if not _ABBREV.search(head):
            end = b + m.end()
            break
    s = text[start:end].strip()
    return s if len(s) <= 900 else s[:900] + "\u2026"


def citing_sentences(pages_text: list[str], number: str, limit: int = 2) -> list[dict]:
    """Sentences that cite equation `number` ('Eq. (6)', 'Eqs. (5)-(7)', 'Equation 6'),
    in document order: [{'page', 'text'}], at most `limit`."""
    want = norm_number(number)
    found, seen = [], set()
    for pno, raw in enumerate(pages_text, start=1):
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
        text = " ".join(text.split())
        for m in _EQ_REF.finditer(text):
            if want in _refs_after(text, m.end()):
                sent = _sentence(text, m.start(), m.end())
                key = sent[:80]
                if key not in seen and len(re.findall(r"[^\W\d_]{3,}", sent)) >= 4:
                    seen.add(key)
                    found.append({"page": pno, "text": sent})
                    if len(found) >= limit:
                        return found
    return found


# ---------------------------------------------------------------- tables

_CAPTION = re.compile(
    r"^\s*(?:Table|TABLE|Tab\.)\s+(?P<n>[A-Za-z]{0,2}[.\-]?[0-9IVXLCivxlc]+(?:[.\-]\d+)*[a-z]?)"
    r"(?P<after>\s*[.:|\u2013\u2014-]|\s*$|\s*\n)",
    re.S,
)
_SUP = str.maketrans(
    "0123456789+-()ni",
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207d\u207e\u207f\u2071",
)
_SUB = str.maketrans(
    "0123456789+-()",
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208d\u208e",
)
_RAISED_SYMBOLS = set("\u221a\u25e6\u00b0\u2218*\u2020\u2021\u2032\u2033")


def _script(m, table, word) -> str:
    inner = re.sub(r"(?<!\w)_|_(?!\w)", "", m.group(1)).replace("\u2212", "-")
    inner = inner.replace("\u25e6", "\u00b0")
    if inner and all(c in _RAISED_SYMBOLS for c in inner):
        return inner
    if inner and all(
        c in "0123456789+-()ni" if table is _SUP else c in "0123456789+-()"
        for c in inner
    ):
        return inner.translate(table)
    return f"{word}({inner})"


def clean_cell_line(s: str, vocab: set[str] | None = None) -> str:
    """A table cell line from the layout converter's Markdown, as plain text: tags and
    emphasis marks gone, <sup>/<sub> as Unicode scripts, ring/root signs kept inline, and
    a dropped 'fl'/'fi' ligature restored from the text layer's vocabulary."""
    s = unicodedata.normalize("NFKC", html.unescape(s))
    s = re.sub(
        r"</sup>\s*<sup>|</sub>\s*<sub>", "", s
    )  # km<sup>-</sup><sup>1</sup>: one exponent
    s = re.sub(r"<sup>(.*?)</sup>", lambda m: _script(m, _SUP, "^"), s)
    s = re.sub(r"<sub>(.*?)</sub>", lambda m: _script(m, _SUB, "_"), s)
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)
    s = s.replace("~~", "").replace("**", "")
    s = re.sub(r"_([^\w\s])_", r"\1", s)
    s = re.sub(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", r"\1", s)
    s = s.replace("\u25e6", "\u00b0")
    s = " ".join(re.sub(r"\s*\u00d7\s*", " \u00d7 ", s).split())
    if vocab:
        s = re.sub(r"[^\W\d_]+", lambda m: _ligature(m.group(0), vocab), s)
    return s


def _ligature(word: str, vocab: set[str]) -> str:
    if word.lower() in vocab or "f" not in word:
        return word
    for i, ch in enumerate(word):
        if ch == "f":
            for rep in ("fl", "fi", "ff", "ffi", "ffl"):
                cand = word[:i] + rep + word[i + 1 :]
                if cand.lower() in vocab:
                    return cand
    return word


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


def md_tables(text: str) -> list[str]:
    """The pipe tables of a page's Markdown, each as its lines joined by newlines."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.match(
            r"^\s*\|\s*:?-{3,}", lines[i + 1]
        ):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append("\n".join(lines[i:j]))
            i = j
        else:
            i += 1
    return out


def parse_pipe_table(block: str) -> list[list[str]]:
    """Rows of cells (header first, separator row dropped); a cell keeps its <br>."""
    rows = []
    for ln in block.split("\n"):
        if re.match(r"^\s*\|\s*:?-{3,}", ln):
            continue
        cells = re.split(r"(?<!\\)\|", ln.strip())
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        rows.append([c.strip().replace("\\|", "|") for c in cells])
    return rows


def _cell_lines(cell: str, vocab) -> list[str]:
    parts = re.split(r"<br\s*/?>", cell, flags=re.I)
    return [x for x in (clean_cell_line(p, vocab) for p in parts) if x]


def tidy_table(rows: list[list[str]], vocab: set[str] | None = None) -> dict:
    """Rows as lists of cell lines, then order-split: in a row whose first cell holds
    several labels (wrapped and unit lines joined) and whose other cells each hold as many
    lines as there are labels, every label starts its own row. Other rows keep their
    lines, joined by <br>, and are counted in 'unsplit'."""
    out, split, unsplit = [], 0, 0
    for r, cells in enumerate(rows):
        lines = [_cell_lines(c, vocab) for c in cells]
        if r == 0 or all(len(c) <= 1 for c in lines):
            out.append(["<br>".join(c) for c in lines])
            continue
        labels = join_label_lines(lines[0]) if lines else []
        counts = {len(c) for c in lines[1:] if c}
        if len(labels) > 1 and counts == {len(labels)}:
            for k in range(len(labels)):
                out.append([labels[k]] + [c[k] if c else "" for c in lines[1:]])
            split += 1
        else:
            out.append(["<br>".join(c) for c in lines])
            unsplit += 1
    return {"rows": out, "split_rows": split, "unsplit_rows": unsplit}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", s).lower()))


def match_md_table(tables: list[str], box_text: str) -> str | None:
    """The Markdown pipe table that holds most of the box text layer's tokens (>= half)."""
    want = _tokens(box_text)
    best, score = None, 0.0
    for t in tables:
        have = _tokens(re.sub(r"<[^>]+>", " ", t))
        s = len(want & have) / len(want) if want else 0.0
        if s > score:
            best, score = t, s
    return best if score >= 0.5 else None


def nearest_box(
    caption: tuple, boxes: list[tuple], page_h: float, max_gap: float = 150.0
):
    """The box (x0, y0, x1, y1) nearest the caption rectangle, sharing some of its width;
    one below the caption is preferred (tables carry their caption above)."""
    cx0, cy0, cx1, cy1 = caption
    best, best_score = None, None
    for b in boxes:
        x0, y0, x1, y1 = b
        overlap = min(cx1, x1) - max(cx0, x0)
        if overlap < 0.2 * min(cx1 - cx0, x1 - x0):
            continue
        if y0 >= cy1 - 2:
            gap = y0 - cy1
        elif y1 <= cy0 + 2:
            gap = cy0 - y1 + 15.0
        else:
            gap = 0.0
        if gap <= max_gap and (best_score is None or gap < best_score):
            best, best_score = b, gap
    return best


# ---------------------------------------------------------------- the OCR server


class OcrUnavailable(RuntimeError):
    pass


def _pdeathsig():
    """Child setup: killed with this process; address-space limit lifted for the server."""
    import ctypes
    import resource

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1, signal.SIGKILL
        )  # PR_SET_PDEATHSIG
    except Exception:
        pass
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (hard, hard))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OcrServer:
    """A CPU-only llama-server with the OCR model, alive for one `with` block."""

    def __init__(self, deadline: float):
        self.deadline, self.proc, self.port = deadline, None, 0
        self._lock = None

    def _left(self) -> float:
        return self.deadline - time.monotonic()

    def __enter__(self):
        if os.environ.get("LIGHTRAG_MCP_OCR", "").lower() in (
            "off",
            "0",
            "no",
            "false",
        ):
            raise OcrUnavailable("the OCR model is switched off (LIGHTRAG_MCP_OCR=off)")
        for what, path in (
            ("llama-server", LLAMA_SERVER),
            ("OCR model", OCR_MODEL),
            ("OCR projector", OCR_MMPROJ),
        ):
            if not (
                path
                and os.path.isfile(path)
                and (what != "llama-server" or os.access(path, os.X_OK))
            ):
                raise OcrUnavailable(
                    f"{what} not found on this host ({path or 'unset'})"
                )
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._lock = open(os.path.join(CACHE_DIR, "ocr.lock"), "w")
        while True:  # one OCR server at a time, whatever the number of callers
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if self._left() <= 0:
                    raise OcrUnavailable(
                        "another OCR call held the model for the whole time budget"
                    )
                time.sleep(0.5)
        self.port = _free_port()
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        env.update(
            HIP_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1"
        )  # no GPU, whatever the build
        cmd = [
            "nice", "-n", "19", LLAMA_SERVER,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-m", OCR_MODEL, "--mmproj", OCR_MMPROJ,
            "-c", "8192", "--parallel", "1",
            "--device", "none", "-ngl", "0", "-t", str(OCR_THREADS),
            "--jinja", "--no-warmup",
        ]  # fmt: skip
        started_at = time.monotonic()
        self._log = open(os.path.join(CACHE_DIR, "ocr-server.log"), "wb")
        self.proc = subprocess.Popen(
            cmd, env=env, stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=_pdeathsig,
        )  # fmt: skip
        try:
            self._wait_ready()
            self.load_s = time.monotonic() - started_at
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait_ready(self):
        url = f"http://127.0.0.1:{self.port}/health"
        while True:
            if self.proc.poll() is not None:
                raise OcrUnavailable(
                    f"llama-server exited with {self.proc.returncode}: {self._log_tail()}"
                )
            if self._left() <= 0:
                raise OcrUnavailable("the OCR model did not finish loading in time")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)

    def _log_tail(self) -> str:
        try:
            with open(os.path.join(CACHE_DIR, "ocr-server.log"), "rb") as f:
                return (
                    f.read()[-300:]
                    .decode("utf-8", "replace")
                    .strip()
                    .replace("\n", " | ")
                )
        except OSError:
            return ""

    def transcribe(self, png: bytes) -> str:
        body = {
            "temperature": 0, "repeat_penalty": 1.15, "max_tokens": OCR_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
        }  # fmt: skip
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )  # fmt: skip
        left = self._left()
        if left <= 1:
            raise OcrUnavailable("no time left for the transcription")
        try:
            with urllib.request.urlopen(req, timeout=left) as r:
                d = json.load(r)
        except Exception as e:
            raise OcrUnavailable(
                f"the transcription request failed or timed out ({type(e).__name__})"
            ) from e
        ch = d["choices"][0]
        text = (ch.get("message") or {}).get("content") or ""
        if ch.get("finish_reason") != "stop" or not text.strip():
            raise OcrUnavailable(
                f"the model did not finish (finish_reason={ch.get('finish_reason')})"
            )
        return text.strip()

    def __exit__(self, *exc):
        p = self.proc
        if p is not None and p.poll() is None:
            for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(p.pid, sig)
                try:
                    p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        for f in (getattr(self, "_log", None), self._lock):
            if f:
                with contextlib.suppress(Exception):
                    f.close()
        return False


# ---------------------------------------------------------------- PDF, layout and cache


def _file_key(path: str) -> str:
    st = os.stat(path)
    import pymupdf

    raw = f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}|{pymupdf.__version__}|{LAYOUT_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


class Cache:
    """Layout (boxes and pipe tables) and OCR text per PDF, in one JSON file."""

    def __init__(self, path: str):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.file = os.path.join(CACHE_DIR, _file_key(path) + ".json")
        try:
            with open(self.file, encoding="utf-8") as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}
        self.data.setdefault("pages", {})
        self.data.setdefault("ocr", {})

    def save(self):
        tmp = f"{self.file}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.file)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def page_too_heavy(page) -> str:
    """'' when the layout model may see the page, else why not."""
    words = page.get_text("words")
    if len(words) > LAYOUT_MAX_SPANS:
        spans = sum(
            len(ln["spans"])
            for b in page.get_text("dict")["blocks"]
            if b.get("type") == 0
            for ln in b["lines"]
        )
        if spans > LAYOUT_MAX_SPANS:
            return f"{spans} text spans"
    if len(page.read_contents() or b"") > LAYOUT_MAX_CONTENT_BYTES:
        return "a very large content stream"
    return ""


def layout_page(doc, pno: int, cache: Cache) -> dict | None:
    """{'boxes': [[class, x0, y0, x1, y1]], 'tables': [pipe table, ...]} for a 0-based
    page, or None (with cache['skipped'] noting why) when the layout model cannot see it."""
    key = str(pno)
    if key in cache.data["pages"]:
        return cache.data["pages"][key] or None
    why = page_too_heavy(doc[pno])
    if why:
        cache.data.setdefault("skipped", {})[key] = why
        return None
    import pymupdf4llm

    try:
        with contextlib.redirect_stdout(sys.stderr):
            ch = pymupdf4llm.to_markdown(
                doc, pages=[pno], page_chunks=True, use_ocr=False
            )[0]
    except MemoryError:
        cache.data.setdefault("skipped", {})[key] = "the layout model ran out of memory"
        return None
    except Exception as e:
        cache.data.setdefault("skipped", {})[key] = (
            f"the layout model failed ({type(e).__name__})"
        )
        return None
    res = {
        "boxes": [
            [b["class"], *map(float, b["bbox"])]
            for b in ch.get("page_boxes") or []
            if isinstance(b, dict) and b.get("bbox")
        ],
        "tables": md_tables(ch.get("text") or ""),
    }
    cache.data["pages"][key] = res
    cache.save()
    return res


def skipped_reason(cache: Cache, pno: int) -> str:
    return (cache.data.get("skipped") or {}).get(str(pno), "")


def page_is_scan(page) -> bool:
    area = page.rect.width * page.rect.height
    try:
        return any(
            (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) >= 0.8 * area
            for i in page.get_image_info()
        )
    except Exception:
        return False


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


def band_text(page, words: list, tag_word) -> str:
    """Text-layer lines just above and level with a '(N)' word: the fallback when the
    layout model found no formula box. A right-hand column is told by an empty middle gap."""
    import pymupdf

    w = page.rect.width
    x0 = 0.0
    if tag_word[2] > 0.55 * w:
        band = [
            x for x in words if tag_word[1] - 45 <= _center(x)[1] <= tag_word[3] + 8
        ]
        mid_empty = not any(
            x[0] < 0.52 * w < x[2] or 0.48 * w < x[0] < 0.52 * w for x in band
        )
        left = [x for x in band if x[2] < 0.48 * w]
        right = [x for x in band if x[0] > 0.52 * w and not tag_of(x[4])]
        if mid_empty and len(left) >= 3 and len(right) >= 3:
            x0 = 0.5 * w
    rect = pymupdf.Rect(x0, tag_word[1] - 45, tag_word[2] + 2, tag_word[3] + 8)
    return page.get_text("text", clip=rect)


# ---------------------------------------------------------------- equation


def _ocr_box(doc, pno, rect, cache: Cache, deadline_s: float):
    """(text, seconds, from_cache, seconds to load the model) for a box; raises OcrUnavailable."""

    key = f"{pno}:{rect.x0:.0f},{rect.y0:.0f},{rect.x1:.0f},{rect.y1:.0f}:{os.path.basename(OCR_MODEL)}"
    if key in cache.data["ocr"]:
        return cache.data["ocr"][key], 0.0, True, 0.0
    started = time.monotonic()
    with OcrServer(started + deadline_s) as srv:
        png = (
            doc[pno]
            .get_pixmap(
                clip=rect
                + (-OCR_MARGIN_PT, -OCR_MARGIN_PT, OCR_MARGIN_PT, OCR_MARGIN_PT),
                dpi=OCR_DPI,
            )
            .tobytes("png")
        )
        text = srv.transcribe(png)
    cache.data["ocr"][key] = text
    cache.save()
    return text, time.monotonic() - started, False, srv.load_s


def _equation_box(words: list, boxes: list, want: str):
    """(rect, [(number, word)], class) of the layout box that holds equation `want`: a
    'formula' box first; else a 'table' or 'picture' box up to 150 pt high (the model
    sometimes calls a one-line equation that); else a short text-like box (<= 60 pt high,
    <= 40 words) with the number at its right edge. None when there is none."""
    import pymupdf

    tiers = (
        lambda c, r, n: c == "formula",
        lambda c, r, n: c in ("table", "picture") and r.height <= 150,
        lambda c, r, n: c in ("text", "list-item") and r.height <= 60 and n <= 40,
    )
    for ok in tiers:
        for cls, *bb in boxes:
            rect = pymupdf.Rect(bb)
            inside = words_in(words, rect)
            if not ok(cls, rect, len(inside)):
                continue
            tw = tags_on_page(inside)
            if cls != "formula":
                tw = [
                    x for x in tw if x[1][2] >= rect.x1 - max(30.0, 0.15 * rect.width)
                ]
            if any(norm_number(t) == want for t, _ in tw):
                return rect, tw, cls
    return None


def cmd_equation(path: str, number: str, only_page: int = 0) -> dict:
    import pymupdf

    started = time.monotonic()
    doc = pymupdf.open(path)
    if doc.needs_pass:
        return {
            "status": "error",
            "message": "the PDF is encrypted and needs a password",
        }
    want = norm_number(number)
    res: dict = {"number": number, "pages": doc.page_count}
    cache = Cache(path)
    page_words, cands, found_tags, has_text = {}, [], set(), False
    for pno in range(doc.page_count):
        words = doc[pno].get_text("words")
        has_text = has_text or bool(words)
        page_words[pno] = words
        tags = tags_on_page(words)
        found_tags.update(norm_number(t) for t, _ in tags)
        if any(norm_number(t) == want for t, _ in tags) and (
            not only_page or pno + 1 == only_page
        ):
            cands.append(pno)
    if not has_text:
        return {**res, "status": "no-text-layer"}
    if not cands:
        shown = sorted(found_tags, key=lambda x: (len(x), x))[:60]
        return {**res, "status": "not-found", "numbers_seen": shown}
    res["other_pages"] = [p + 1 for p in cands[1:]]
    chosen = None
    for pno in cands:
        lay = layout_page(doc, pno, cache)
        if not lay:
            continue
        chosen = _equation_box(page_words[pno], lay["boxes"], want)
        if chosen:
            chosen = (pno, *chosen)
            break
    first = cands[0]
    if not chosen:
        why = (
            skipped_reason(cache, first)
            or "the layout model found no formula box carrying this number"
        )
        w = next(
            w for t, w in tags_on_page(page_words[first]) if norm_number(t) == want
        )
        text = band_text(doc[first], page_words[first], w)
        return {
            **res, "status": "text-layer", "page": first + 1, "scanned": page_is_scan(doc[first]),
            "text_layer": _flat(text)[:RAW_LIMIT], "reason": f"{why}; this is the text near the number",
            "number_label": w[4].strip("() "),
            "citing": citing_sentences([doc[p].get_text("text") for p in range(doc.page_count)], number),
            "seconds": round(time.monotonic() - started, 1),
        }  # fmt: skip
    pno, rect, tw, box_class = chosen
    page = doc[pno]
    right = [x for x in tw if x[1][2] >= rect.x1 - max(30.0, 0.15 * rect.width)] or tw
    tw = sorted(right, key=lambda x: _center(x[1])[1])
    box_tags = [t for t, _ in tw]
    tag_words = {id(w) for _, w in tw}
    layer = [w for w in words_in(page_words[pno], rect) if id(w) not in tag_words]
    res.update(
        status="text-layer", page=pno + 1, scanned=page_is_scan(page), box_tags=box_tags, box_class=box_class,
        text_layer=_flat(page.get_text("text", clip=rect))[:RAW_LIMIT],
        number_label=next(t for t in box_tags if norm_number(t) == want),
    )  # fmt: skip
    res["citing"] = citing_sentences(
        [doc[p].get_text("text") for p in range(doc.page_count)], number
    )
    try:
        text, secs, cached, load_s = _ocr_box(doc, pno, rect, cache, OCR_BUDGET_S)
    except OcrUnavailable as e:
        res["reason"] = str(e)
        res["seconds"] = round(time.monotonic() - started, 1)
        return res
    entries, delimited = split_math(text)
    lines, how = pick_equation(entries, box_tags, number)
    res.update(
        status="ocr",
        ocr_seconds=round(secs, 1),
        ocr_load_seconds=round(load_s, 1),
        ocr_cached=cached,
        ocr_model=os.path.basename(OCR_MODEL),
    )
    if lines:
        res["equation"] = lines
        res["pick"] = how
    else:
        res["status"] = "ocr-unmatched"
        res["pick"] = how
        res["ocr_block"] = [f"${e['latex']}$" for e in entries]
    if not delimited:
        res["warning_delimiters"] = True
    res["missing_numbers"] = (
        []
        if res["scanned"]
        else number_check([w[4] for w in layer], [e["latex"] for e in entries])
    )
    res["seconds"] = round(time.monotonic() - started, 1)
    return res


# ---------------------------------------------------------------- table


def cmd_table(path: str, number: str) -> dict:
    import pymupdf

    started = time.monotonic()
    doc = pymupdf.open(path)
    if doc.needs_pass:
        return {
            "status": "error",
            "message": "the PDF is encrypted and needs a password",
        }
    want = norm_number(number, roman=True)
    res: dict = {"number": number, "pages": doc.page_count}
    cache = Cache(path)
    cands, captions_seen, has_text = [], [], False
    for pno in range(doc.page_count):
        blocks = doc[pno].get_text("blocks")
        for b in blocks:
            if b[6] != 0:
                continue
            has_text = True
            m = _CAPTION.match(b[4])
            if m:
                label = norm_number(m.group("n"), roman=True)
                captions_seen.append(label)
                if label == want:
                    cands.append((pno, b))
    if not has_text:
        return {**res, "status": "no-text-layer"}
    if not cands:
        return {
            **res,
            "status": "not-found",
            "captions_seen": sorted(set(captions_seen), key=lambda x: (len(x), x))[:60],
        }
    chosen = None
    for pno, b in cands:
        lay = layout_page(doc, pno, cache)
        if not lay:
            continue
        cap = tuple(b[:4])
        tables = [tuple(x[1:]) for x in lay["boxes"] if x[0] == "table"]
        tb = nearest_box(cap, tables, doc[pno].rect.height)
        tpno = pno
        if (
            tb is None
            and cap[3] > 0.7 * doc[pno].rect.height
            and pno + 1 < doc.page_count
        ):
            nxt = layout_page(doc, pno + 1, cache)
            if nxt:
                top = [
                    tuple(x[1:])
                    for x in nxt["boxes"]
                    if x[0] == "table" and x[2] < 0.35 * doc[pno + 1].rect.height
                ]
                tb, tpno, lay = (top[0] if top else None), pno + 1, nxt
        if tb is not None:
            chosen = (pno, b, tpno, tb, lay)
            break
    pno, b = cands[0]
    caption = _flat(b[4])
    if not chosen:
        why = (
            skipped_reason(cache, pno)
            or "the layout model found no table box next to the caption"
        )
        return {
            **res,
            "status": "no-table",
            "page": pno + 1,
            "caption": caption,
            "reason": why,
        }
    pno, b, tpno, tb, lay = chosen
    caption = _flat(b[4])
    page = doc[tpno]
    rect = pymupdf.Rect(tb)
    layer_text = page.get_text("text", clip=rect)
    res.update(page=tpno + 1, caption=caption, caption_page=pno + 1)
    block = match_md_table(lay["tables"], layer_text)
    if block is None:
        res.update(status="text-layer", text_layer=_flat(layer_text)[:RAW_LIMIT],
                   lines=[x.strip() for x in layer_text.split("\n") if x.strip()][:200],
                   reason="the layout converter did not produce a pipe table for this box; this is the box's text layer")  # fmt: skip
        return res
    vocab = {
        w.lower()
        for w in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", layer_text))
    }
    tidy = tidy_table(parse_pipe_table(block), vocab)
    res.update(status="table", **tidy, seconds=round(time.monotonic() - started, 1))
    return res


# ---------------------------------------------------------------- main


def _limit_memory():
    import resource

    with contextlib.suppress(Exception):
        gib = int(MEM_GIB * 2**30)
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        resource.setrlimit(
            resource.RLIMIT_AS,
            (gib if hard == resource.RLIM_INFINITY else min(gib, hard), hard),
        )
    with contextlib.suppress(
        Exception
    ):  # the kernel's first victim, not the LightRAG server
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mode", choices=("equation", "table"))
    ap.add_argument("pdf")
    ap.add_argument("number")
    ap.add_argument("--page", type=int, default=0)
    a = ap.parse_args(argv)
    signal.signal(
        signal.SIGTERM, lambda *_: sys.exit(143)
    )  # runs the `with` blocks' cleanup
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # nothing but the result may reach stdout
    _limit_memory()
    try:
        if not os.path.isfile(a.pdf):
            out = {"status": "error", "message": f"no such file: {a.pdf}"}
        elif a.mode == "equation":
            out = cmd_equation(a.pdf, a.number, a.page)
        else:
            out = cmd_table(a.pdf, a.number)
    except MemoryError:
        out = {"status": "error", "message": "ran out of memory reading this PDF"}
    except Exception as e:  # a corrupt file, a missing dependency: say so, once
        out = {"status": "error", "message": f"{type(e).__name__}: {e}"}
    real_stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
