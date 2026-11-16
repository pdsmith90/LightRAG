"""Human-readable OCR progress for a Zotero corpus bundle directory.

The nightly build converts scanned PDFs page by page, and a multi-hour document
looks like no progress at all from outside: the log line only appears when the
document is written. This module turns what the build leaves on disk -- its
running marker, the per-document page cache, its own build totals, its log and
its deferred list -- into a few plain lines the web UI can show in the Pipeline
Status dialog.

Those lines are shown only while OCR is actually running, i.e. only while a page
cache is being written: the totals, log and deferred list stay on disk for days
after a build and would otherwise sit in the dialog forever, so they are shown as
context for an in-flight document and not on their own.

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
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("lightrag")

# The dialog polls while a build writes, and every call stats a directory, reads
# a log tail and counts page-cache lines. Cache per bundle directory, with the
# same 5 s stat throttle as zotero_citations._load. An explicit `now` bypasses
# the cache so tests stay deterministic.
_CACHE_INTERVAL = 5.0
_lock = threading.Lock()
_cache: dict[str, tuple[float, list[str]]] = {}

# A page cache not appended to for this long is not an in-flight document: a
# killed worker leaves one behind, and the dialog must not show it forever.
_INFLIGHT_MAX_AGE = 900.0

# The same guard for the two single-file claims: a build killed mid-run leaves
# its running marker behind, and a log nobody has touched this week describes a
# build that is long over.
_MARKER_MAX_AGE = 12 * 3600.0
_LOG_MAX_AGE = 7 * 86400.0

# The build's own totals file, rewritten as each page lands. Last week's counts
# are not "this build".
_TOTALS_MAX_AGE = 7 * 86400.0
_TOTALS_FILE = "ocr_build_progress.json"

_LOG_TAIL_BYTES = 65536
_LOG_MAX_LINES = 3

# The exact line build_corpus prints when it finishes a document (a tail follows
# when pages were read by Tesseract, so this is a search, not a full match).
_DONE_RE = re.compile(
    r"\[build_corpus\] VLM-OCR (?P<name>.+?): (?P<pages>\d+) page\(s\) via "
    r"(?P<model>\S+) \((?P<kind>\w+)\) in (?P<secs>\d+) s"
)


def _clear_cache() -> None:
    """Drop the per-bundle cache (tests; also safe to call at any time)."""
    with _lock:
        _cache.clear()


def _is_old(mtime: float, now: float, max_age: float) -> bool:
    """True when a file last written at `mtime` is too old to still be current."""
    return now - mtime > max_age


def _running(bundle: Path, now: float) -> list[str]:
    """``.corpus_build_running``: the build is converting something right now."""
    marker = bundle / ".corpus_build_running"
    try:
        mtime = marker.stat().st_mtime
    except OSError:
        return []
    if _is_old(mtime, now, _MARKER_MAX_AGE):
        return []
    return ["OCR: corpus build running"]


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
        f"OCR in progress: {os.path.basename(path)}: page {done} of {pages} ({pct}%)"
    ]


def _started_label(value: object) -> str | None:
    """``2026-11-15T12:30:00Z`` -> ``2026-11-15 12:30`` (UTC), or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        started = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is not None:
        started = started.astimezone(timezone.utc)
    return started.strftime("%Y-%m-%d %H:%M")


def _build_totals(bundle: Path, now: float) -> list[str]:
    """``ocr_build_progress.json``: what this build has converted so far."""
    path = bundle / _TOTALS_FILE
    try:
        if _is_old(path.stat().st_mtime, now, _TOTALS_MAX_AGE):
            return []
        with open(path, "rb") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []

    docs = data.get("docs_done")
    pages = data.get("pages_done")
    for value in (docs, pages):
        if not isinstance(value, int) or isinstance(value, bool):
            return []

    seconds = data.get("ocr_seconds")
    if not isinstance(seconds, int) or isinstance(seconds, bool):
        seconds = 0
    line = (
        f"OCR this build: {docs} document(s), {pages} page(s), "
        f"{seconds // 60} min of OCR"
    )
    started = _started_label(data.get("started"))
    if started:
        line += f" (started {started})"
    return [line]


def _done_lines(bundle: Path, now: float) -> list[str]:
    """The last few documents the log says the build finished, oldest first."""
    log = bundle / "vm_weekly_update.log"
    try:
        if _is_old(log.stat().st_mtime, now, _LOG_MAX_AGE):
            return []
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    matches = list(_DONE_RE.finditer(tail))[-_LOG_MAX_LINES:]
    return [
        f"OCR done: {m['name']}: {m['pages']} page(s) via {m['model']} "
        f"({m['kind']}) in {m['secs']} s"
        for m in matches
    ]


def _deferred(bundle: Path) -> list[str]:
    """Documents no OCR lane could take, waiting for a later build."""
    try:
        with open(bundle / "ocr_vlm_deferred.txt", "rb") as f:
            n = sum(1 for line in f if line.strip())
    except OSError:
        return []
    if n <= 0:
        return []
    return [f"OCR deferred: {n} document(s) waiting for a free OCR lane"]


def _compute(bundle: Path, now: float) -> list[str]:
    # A page cache appended to moments ago is the ONLY trace that says OCR is
    # running; every other source is a build artifact that survives for days.
    # Without one, show nothing at all -- no totals, no log, no deferred count.
    try:
        inflight = _inflight(bundle, now)
    except Exception as e:  # a broken bundle must not break the dialog
        logger.debug(f"zotero_ocr_progress: _inflight failed for {bundle}: {e}")
        inflight = []
    if not inflight:
        return []

    head: list[str] = []
    tail: list[str] = []
    for fn, args in (
        (_running, (bundle, now)),
        (_build_totals, (bundle, now)),
        (_done_lines, (bundle, now)),
        (_deferred, (bundle,)),
    ):
        try:
            lines = fn(*args)
        except Exception as e:  # a broken bundle must not break the dialog
            logger.debug(f"zotero_ocr_progress: {fn.__name__} failed for {bundle}: {e}")
            continue
        # The running marker, when present, is the first line; the in-flight
        # document follows it and precedes the build's context lines.
        (head if fn is _running else tail).extend(lines)
    return head + inflight + tail


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
