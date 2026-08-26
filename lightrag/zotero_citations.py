"""Deterministic bibliographic citations for a Zotero-backed corpus.

LightRAG identifies a source only by its corpus filename, and asks the answering
LLM to turn that into a ``### References`` section. Measured against this
deployment's own answer cache, that produces raw filenames, fabricated APA
entries with invented volume/issue/DOI, duplicated headings -- or no list at
all. This module replaces that guesswork: corpus files are named
``<8-char ZoteroKey>__<slug>.md``, so the key resolves against
``zotero_metadata.json`` to the real authors, year, title, publication and DOI.

Fork-local (no upstream counterpart), stdlib only, and inert when the metadata
file is absent -- every entry point degrades to LightRAG's previous behaviour
rather than raising.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

logger = logging.getLogger("lightrag")

# NOTE: deliberately not `from lightrag.utils import logger` -- utils.py imports
# this module, so that would be a circular import.

METADATA_PATH = os.environ.get(
    "ZOTERO_METADATA", "/home/pat/Zotero/lightrag-bundle/zotero_metadata.json"
)

# The reference list is rebuilt inside the chunk-truncation retry loop
# (utils._truncate_chunks_for_unified_context), which runs dozens of times per
# query on the tokenizer thread executor. stat() at most this often.
_STAT_INTERVAL = 5.0

_lock = threading.Lock()
_meta: dict = {}
_mtime: float | None = None
_next_stat: float = 0.0
_warned = False

_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

# A references heading the answering LLM may have emitted. Anchored to the start
# of a line; the bare form must be alone on its line so ordinary prose beginning
# with "References" is not mistaken for a heading.
_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*references\b"
    r"|\*\*references\*\*:?[ \t]*$"
    r"|references[ \t]*:?[ \t]*$)",
    re.IGNORECASE | re.MULTILINE,
)

SHORT_MAX = 160
_HOLDBACK = 64


def _load() -> dict:
    """Return the metadata index, reloading when the file changes on disk."""
    global _meta, _mtime, _next_stat, _warned
    now = time.monotonic()
    with _lock:
        if now < _next_stat:
            return _meta
        _next_stat = now + _STAT_INTERVAL
        try:
            mtime = os.path.getmtime(METADATA_PATH)
        except OSError as e:
            if not _warned:
                logger.warning(
                    f"zotero_citations: metadata unavailable at {METADATA_PATH} "
                    f"({e}); citations fall back to file paths"
                )
                _warned = True
            return _meta
        if mtime == _mtime:
            return _meta
        try:
            with open(METADATA_PATH, encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as e:
            if not _warned:
                logger.warning(f"zotero_citations: cannot parse {METADATA_PATH}: {e}")
                _warned = True
            return _meta
        if not isinstance(loaded, dict):
            return _meta
        _meta, _mtime, _warned = loaded, mtime, False
        logger.info(f"zotero_citations: loaded {len(_meta)} entries from {METADATA_PATH}")
        return _meta


def zotero_key(file_path: str) -> str | None:
    """Extract the 8-character Zotero storage key from a corpus file path."""
    if not file_path:
        return None
    name = os.path.basename(file_path)
    if name.endswith(".md"):
        name = name[:-3]
    key = name.split("__", 1)[0]
    return key if _KEY_RE.match(key) else None


def _deslug(file_path: str) -> str:
    """Human-readable fallback: the slug tail with underscores turned back into spaces."""
    name = os.path.basename(file_path or "")
    if name.endswith(".md"):
        name = name[:-3]
    tail = name.split("__", 1)[1] if "__" in name else name
    return re.sub(r"_+", " ", tail).strip()


def _authors(entry: dict) -> str:
    """Author names verbatim.

    Zotero stores these as "Family Given" with no comma and sometimes multi-word
    families ("de Paula E. R.", "Lo Min-Hui"), so they cannot be reliably split
    into an initialised form. Print them as-is.
    """
    names = [a for a in (entry.get("authors") or []) if a]
    if not names:
        return ""
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def _join(parts) -> str:
    """Join citation parts with ". ", without doubling a period.

    Author strings routinely end in an initial ("Featherstone W. E.") or in
    "et al.", so a blind ". ".join produces "W. E.." -- join on a bare space
    when the previous part already closed itself.
    """
    out = ""
    for part in parts:
        if not out:
            out = part
        elif out.endswith("."):
            out += " " + part
        else:
            out += ". " + part
    return out


def _entry(file_path: str) -> dict | None:
    key = zotero_key(file_path)
    if not key:
        return None
    got = _load().get(key)
    return got if isinstance(got, dict) else None


def citation_for(file_path: str) -> str:
    """Full bibliographic citation for the appended reference block.

    Returns "" only when there is nothing at all to say (empty path or the
    ``unknown_source`` sentinel); otherwise always yields something printable.
    """
    try:
        if not file_path or file_path == "unknown_source":
            return ""
        entry = _entry(file_path)
        if entry is None:
            # Unknown key, or a free-text source with no key at all (a manually
            # added note keeps its own description as its path).
            return _deslug(file_path) if "__" in os.path.basename(file_path) else file_path
        title = (entry.get("title") or "").strip().rstrip(".")
        parts = [
            p
            for p in (
                _authors(entry),
                f"({entry['year']})" if (entry.get("year") or "").strip() else "",
                title,
                (entry.get("publication") or "").strip(),
                f"https://doi.org/{entry['doi'].strip()}"
                if (entry.get("doi") or "").strip()
                else "",
            )
            if p
        ]
        return _join(parts) or _deslug(file_path)
    except Exception as e:  # never break a query over a citation
        logger.warning(f"zotero_citations: citation_for({file_path!r}) failed: {e}")
        return ""


def citation_short(file_path: str, max_chars: int = SHORT_MAX) -> str:
    """Compact "Authors (Year). Title" for the LLM's Reference Document List.

    ``max_chars`` is a hard budget, and callers rendering into a prompt must set
    it to the length of the string being replaced. ``reference_list_str`` is
    built against a fixed ``buffer_tokens`` reserve computed *before* the list
    exists, and a citation is not reliably shorter than the slug filename it
    replaces: measured across this corpus the mean change is +2.6 tokens per
    entry, and a ten-entry list can grow by ~279 -- past the 200-token reserve.
    Capping to the filename's own length keeps the substitution non-inflating.
    """
    try:
        if not file_path or file_path == "unknown_source":
            return ""
        entry = _entry(file_path)
        if entry is None:
            return _deslug(file_path) if "__" in os.path.basename(file_path) else file_path
        title = (entry.get("title") or "").strip().rstrip(".")
        parts = [
            p
            for p in (
                _authors(entry),
                f"({entry['year']})" if (entry.get("year") or "").strip() else "",
                title,
            )
            if p
        ]
        out = _join(parts) or _deslug(file_path)
        # No lower floor: the cap is a hard promise that the replacement is
        # never longer than what it replaces.
        cap = min(SHORT_MAX, max_chars)
        if cap < 1:
            return ""
        return out if len(out) <= cap else out[: cap - 1].rstrip() + "…"
    except Exception as e:
        logger.warning(f"zotero_citations: citation_short({file_path!r}) failed: {e}")
        return ""


def strip_llm_references(text: str) -> str:
    """Drop an LLM-written references heading and everything after it."""
    if not text:
        return text
    m = _HEADING_RE.search(text)
    return text[: m.start()].rstrip() if m else text


def format_reference_block(references: list, answer_text: str) -> str:
    """Render the ``### References`` block appended to an answer.

    Lists only the references the answer actually cites with an ``[n]`` marker,
    so a retrieved-but-unused document is not passed off as a source. If the
    answer cites nothing, every retrieved reference is listed instead.
    """
    try:
        if not references:
            return ""
        cited = set(_MARKER_RE.findall(answer_text or ""))
        chosen = [r for r in references if str(r.get("reference_id", "")) in cited]
        if not chosen:
            chosen = list(references)
        lines = []
        for ref in chosen:
            rid = ref.get("reference_id", "?")
            path = ref.get("file_path", "")
            cite = ref.get("citation") or citation_for(path) or path or "(unknown source)"
            lines.append(f"- [{rid}] {cite}")
        if not lines:
            return ""
        return "\n\n### References\n\n" + "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning(f"zotero_citations: format_reference_block failed: {e}")
        return ""


class ReferenceStripper:
    """Incremental :func:`strip_llm_references` for streamed answers.

    Text already sent to the client cannot be retracted, so a short tail is held
    back on every ``feed`` -- long enough that a references heading split across
    chunk boundaries is still recognised before any of it is emitted.
    """

    def __init__(self, holdback: int = _HOLDBACK):
        self._acc: list[str] = []
        self._len = 0
        self._emitted = 0
        self._done = False
        self._holdback = holdback

    def _text(self) -> str:
        joined = "".join(self._acc)
        self._acc = [joined]
        return joined

    def feed(self, chunk: str) -> str:
        if self._done or not chunk:
            return ""
        self._acc.append(chunk)
        self._len += len(chunk)
        acc = self._text()
        m = _HEADING_RE.search(acc)
        if m:
            self._done = True
            keep = acc[: m.start()].rstrip()
            # The heading was caught before we emitted it in all but pathological
            # cases; if some of it did escape, we simply stop here.
            return keep[self._emitted :] if len(keep) > self._emitted else ""
        cut = max(self._emitted, self._len - self._holdback)
        out = acc[self._emitted : cut]
        self._emitted = cut
        return out

    def flush(self) -> str:
        if self._done:
            return ""
        self._done = True
        out = "".join(self._acc)[self._emitted :]
        self._emitted = self._len
        return out

    @property
    def text(self) -> str:
        """Everything fed so far, references heading and all."""
        return "".join(self._acc)

    @property
    def kept(self) -> str:
        """The answer as the client saw it, with any references section removed.

        Scan this -- not :attr:`text` -- for ``[n]`` markers: the LLM's own
        references block lists every retrieved id, so counting markers there
        would defeat the cited-only filtering in
        :func:`format_reference_block`.
        """
        return strip_llm_references("".join(self._acc))
