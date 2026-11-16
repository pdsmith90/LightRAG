"""Human-readable OCR progress for a Zotero corpus bundle directory.

The nightly build converts scanned PDFs page by page, and a multi-hour document
looks like no progress at all from outside: the log line only appears when the
document is written. This module reads the newest per-document page cache and
turns it into one plain line the web UI can show in the Pipeline Status dialog --
the document being converted right now, and page N of M.

Nothing else is reported. The build's running marker, totals file, log and
deferred list stay on disk for days after a build, so they describe work that is
not happening now and would sit in the dialog forever; with no document in
flight the answer is an empty list.

The line is also the one thing the dialog writes back: the endpoint appends it to
the shared pipeline history once per change, so it takes its place in time order
with the pipeline's own messages instead of always sitting at the bottom.

Fork-local (no upstream counterpart), stdlib only, and best effort throughout: a
missing, half-written, stale or unparseable file costs its own line, never the
call. The bundle directory is the explicit argument when the caller has one, else
``$ZOTERO_OCR_BUNDLE`` when that is set and non-empty, else the working
directory -- the same convention ``zotero_citations`` uses for its metadata file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("lightrag")

# The dialog polls while a build writes, and every call stats a directory and
# counts page-cache lines. Cache per bundle directory, with the same 5 s stat
# throttle as zotero_citations._load. An explicit `now` bypasses the cache so
# tests stay deterministic.
_CACHE_INTERVAL = 5.0
_lock = threading.Lock()
_cache: dict[str, tuple[float, list[str]]] = {}

# A page cache not appended to for this long is not an in-flight document: a
# killed worker leaves one behind, and the dialog must not show it forever.
_INFLIGHT_MAX_AGE = 900.0

# Every line this module returns starts with this, which is how the history
# writer recognises the newest OCR entry already in the pipeline log.
OCR_LINE_PREFIX = "OCR in progress:"


def _clear_cache() -> None:
    """Drop the per-bundle cache (tests; also safe to call at any time)."""
    with _lock:
        _cache.clear()


def _inflight(bundle: Path, now: float) -> list[str]:
    """The document the newest page cache belongs to, and how far it has got.

    ``<bundle>/ocr_vlm_pages/<name>-<hash>.jsonl`` is a header line
    (``{"path": ..., "pages": n, ...}``) followed by one line per finished page.
    The ``.eq.jsonl`` siblings are a different pass, not the document's pages.
    """
    best: tuple[float, Path] | None = None
    for p in (bundle / "ocr_vlm_pages").glob("*.jsonl"):
        if p.name.endswith(".eq.jsonl"):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, p)
    if best is None or now - best[0] > _INFLIGHT_MAX_AGE:
        return []

    try:
        with open(best[1], "rb") as f:
            header = json.loads(f.readline())
            if not isinstance(header, dict):
                return []
            path = header.get("path")
            pages = header.get("pages")
            if not isinstance(path, str) or not path:
                return []
            if not isinstance(pages, int) or isinstance(pages, bool) or pages <= 0:
                return []
            # Count the finished pages without parsing them: the file runs to
            # megabytes of transcribed text.
            done = sum(1 for line in f if line.strip())
    except (OSError, ValueError):
        return []

    # A retried page can be appended twice; never claim more than all of them.
    done = min(done, pages)
    pct = int(done * 100 / pages)
    return [
        f"{OCR_LINE_PREFIX} {os.path.basename(path)}: page {done} of {pages} ({pct}%)"
    ]


def _compute(bundle: Path, now: float) -> list[str]:
    # A page cache appended to moments ago is the ONLY trace that says OCR is
    # running; every other source is a build artifact that survives for days.
    # Without one, show nothing at all.
    try:
        return _inflight(bundle, now)
    except Exception as e:  # a broken bundle must not break the dialog
        logger.debug(f"zotero_ocr_progress: _inflight failed for {bundle}: {e}")
        return []


def get_ocr_progress_lines(
    bundle_dir: Path | None = None, now: float | None = None
) -> list[str]:
    """OCR status lines for `bundle_dir`.

    The bundle is the explicit argument, else ``$ZOTERO_OCR_BUNDLE`` when that is
    set and non-empty, else the working directory. Returns ``[]`` when nothing
    applies. `now` (epoch seconds) overrides the clock for the staleness tests
    and skips the cache, so callers that need a deterministic answer are never
    served a stale one.
    """
    if bundle_dir is not None:
        bundle = Path(bundle_dir)
    else:
        override = os.environ.get("ZOTERO_OCR_BUNDLE", "").strip()
        bundle = Path(override) if override else Path.cwd()
    if now is not None:
        return _compute(bundle, now)

    key = str(bundle)
    mono = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and mono < hit[0]:
            return list(hit[1])

    lines = _compute(bundle, time.time())
    with _lock:
        _cache[key] = (mono + _CACHE_INTERVAL, lines)
    return list(lines)


def ocr_line_to_append(lines: list[str], history: list[str]) -> str | None:
    """The OCR line to write into the shared pipeline history now, or ``None``.

    The dialog polls this line, and each answer is appended to the pipeline's
    history so it lands in time order with the pipeline's own messages. That
    must happen once per change: an unchanged answer (same document, same page)
    is already the newest OCR entry in the log and is skipped, and a poll with
    nothing in flight writes nothing.

    The newest OCR entry is found in `history` (oldest first) rather than
    remembered in a module global: the server runs several worker processes over
    one shared history list, so per-process state would write every change once
    per worker. Reading the log also keeps the answer right after the pipeline
    clears or trims its history.
    """
    line = " | ".join(lines)
    if not line:
        return None
    for entry in reversed(history):
        if isinstance(entry, str) and entry.startswith(OCR_LINE_PREFIX):
            return None if entry == line else line
    return line
