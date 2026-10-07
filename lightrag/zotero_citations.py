"""Deterministic bibliographic citations for a Zotero-backed corpus.

LightRAG identifies a source only by its corpus filename, and asks the answering
LLM to turn that into a ``### References`` section. Given nothing but file
names, the model writes raw filenames, fabricated APA entries with invented
volume/issue/DOI, duplicated headings -- or no list at all. This module
replaces that guesswork: corpus files are named
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

# Relative to the process working directory, the directory the server also
# reads its .env from, so zotero_metadata.json can sit beside it. Resolved once
# at import so every stat()/open() and log line uses the same absolute path.
# ZOTERO_METADATA overrides it.
METADATA_PATH = os.path.abspath(
    os.environ.get("ZOTERO_METADATA", "zotero_metadata.json")
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

# A references heading the answering LLM may have emitted. Anchored to the start
# of a line; the bare form must be alone on its line so ordinary prose beginning
# with "References" is not mistaken for a heading.
_HEADING_RE = re.compile(
    r"^[ \t]{0,3}(?:#{1,6}[ \t]*references\b"
    r"|\*\*references\*\*:?[ \t]*$"
    r"|references[ \t]*:?[ \t]*$"
    # "Sources" must be the WHOLE heading. Unanchored it would eat legitimate
    # prose sections -- "### Sources of error" is ordinary scientific writing.
    r"|#{1,6}[ \t]*sources[ \t]*:?[ \t]*$"
    r"|\*\*sources\*\*:?[ \t]*$"
    r"|sources[ \t]*:[ \t]*$)",
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
        logger.info(
            f"zotero_citations: loaded {len(_meta)} entries from {METADATA_PATH}"
        )
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
    families ("van der Berg A. B.", "De Luca Maria"), so they cannot be reliably
    split into an initialised form. Print them as-is.
    """
    names = [a for a in (entry.get("authors") or []) if a]
    if not names:
        return ""
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def _join(parts) -> str:
    """Join citation parts with ". ", without doubling a period.

    Author strings routinely end in an initial ("Doe J. A.") or in
    "et al.", so a blind ". ".join produces "J. A.." -- join on a bare space
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
            return (
                _deslug(file_path) if "__" in os.path.basename(file_path) else file_path
            )
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


def citation_for_key(key: str) -> str:
    """Full bibliographic citation of the library item with storage ``key``;
    "" when the metadata does not know it."""
    try:
        entry = _load().get(key)
        if not isinstance(entry, dict):
            return ""
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
        return _join(parts)
    except Exception as e:  # never break a query over a citation
        logger.warning(f"zotero_citations: citation_for_key({key!r}) failed: {e}")
        return ""


def named_work_notices(
    query: str,
    reference_file_paths,
    named: list[tuple[str, int]],
    indexed,
) -> list[str]:
    """Fork (NAMED_WORK_NOTICE): one sentence per work the query names by author
    and year that is not among the answer's sources, saying why -- not in the
    knowledge base at all, or indexed but not retrieved -- so a reader never
    mistakes an answer built from a paper's citers for one built from the paper.

    ``named`` is :func:`find_works`'s result for the query, ``indexed`` the keys
    among them that hold chunks. Empty when the query names no year (a bare
    surname is too weak a naming to warn about), when the work the query names
    specifically (:func:`specific_named_work`) is among the references, or,
    for an ambiguous naming, when any of the tied works is; and when nothing
    was named. At most three notices.
    """
    if not named or not query_has_year(query):
        return []
    cited = {zotero_key(path or "") for path in reference_file_paths}
    specific = specific_named_work(query, named)
    if specific is not None:
        if specific in cited:
            return []
        candidates = [(specific, 0)]
    else:
        if any(key in cited for key, _ in named):
            return []
        candidates = named[:3]
    notices = []
    for key, _ in candidates:
        cite = citation_for_key(key) or key
        if key in indexed:
            notices.append(
                f"{cite} is in the knowledge base but was not among the sources "
                "retrieved for this question."
            )
        else:
            notices.append(
                f"{cite} is in the Zotero library but not in this knowledge base; "
                "the answer relies on other sources."
            )
    return notices


_WORK_TITLE_MIN = (
    20  # shorter normalised titles ("Preface", "Introduction") name no single work
)


def work_key(file_path: str) -> str:
    """Identity of the work a corpus file holds, for treating copies of one paper
    filed under several Zotero items as one.

    The normalised title (markup, case and punctuation dropped) when it is long
    enough to name a single work, else the DOI, else the path itself -- a file
    the metadata cannot resolve is its own work. Title before DOI because copies
    of a paper often carry different DOIs (an arXiv preprint and the journal
    version), while a short generic title would merge unrelated works.
    """
    entry = _entry(file_path)
    if entry:
        title = re.sub(r"<[^>]+>", "", entry.get("title") or "")
        title = re.sub(r"[^0-9a-z]", "", title.lower())
        if len(title) >= _WORK_TITLE_MIN:
            return "title:" + title
        doi = (entry.get("doi") or "").strip().lower()
        if doi:
            return "doi:" + doi
    return "path:" + (file_path or "")


def citation_short(file_path: str, max_chars: int = SHORT_MAX) -> str:
    """Compact "Authors (Year). Title" for the LLM's Reference Document List.

    ``max_chars`` is a hard budget, and callers rendering into a prompt must set
    it to the length of the string being replaced. ``reference_list_str`` is
    built against a fixed ``buffer_tokens`` reserve computed *before* the list
    exists, and a citation is not reliably shorter than the slug filename it
    replaces: a long author list or title tokenizes larger, and over a
    ten-entry list the growth can exceed the 200-token reserve. Capping to the
    filename's own length keeps the substitution non-inflating.
    """
    try:
        if not file_path or file_path == "unknown_source":
            return ""
        entry = _entry(file_path)
        if entry is None:
            return (
                _deslug(file_path) if "__" in os.path.basename(file_path) else file_path
            )
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


def format_reference_block(references: list) -> str:
    """Render the ``### Sources`` block appended to an answer.

    Lists every retrieved reference, in reference_id order.

    An earlier version listed only ids the answer marked with ``[n]``. That was
    wrong twice over: answering models place markers unreliably (an answer may
    carry none at all, or cite [2] and [3] when retrieval returned a single
    reference), and filtering left visible gaps -- a lone "[2]" entry
    reads as a broken list even though it correctly matches its inline marker.
    Listing everything retrieved has no gaps, and "Sources" claims only what is
    true: these are the documents the retrieval returned. Whether the model
    actually leaned on each one is not something the reference list can know.
    """
    try:
        if not references:
            return ""
        lines = []
        for ref in references:
            rid = ref.get("reference_id", "?")
            path = ref.get("file_path", "")
            cite = (
                ref.get("citation") or citation_for(path) or path or "(unknown source)"
            )
            lines.append(f"- [{rid}] {cite}")
        if not lines:
            return ""
        return "\n\n### Sources\n\n" + "\n".join(lines) + "\n"
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


# ---------------------------------------------------------------------------
# Author-year lookup (fork): which library works does a query or a citation name?
# ---------------------------------------------------------------------------

_AY_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_AY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019\-\u2010\u2011]{2,}")
_AY_TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
# Lower-case words that continue a surname ("van der Lind Tamás", "Ferreira Da
# Costa Inês"); any other token after the first ends it.
_NAME_PARTICLES = frozenset(
    "van der den de del della di da do dos das du le la von zu zur ter ten af av al el bin ibn".split()
)
# A creator string that is an institution, a programme or a mailbox rather than a
# person contributes no surname: its first token is an ordinary word ("Data",
# "University", "Satellites") that would otherwise name the work in every
# question that uses it.
_ORG_WORDS = frozenset(
    "university corporation institute institution center centre centres laboratory "
    "laboratories agency administration program programme project group team "
    "consortium committee commission council office department division service "
    "support data mission observatory satellites association society foundation "
    "survey bureau network".split()
)
# Function words a malformed creator string can put in the surname position
# ("For Mei"); never a surname form.
_NOT_SURNAMES = frozenset(
    "and the for with from that this which what when where who how not but nor".split()
)
# "Okafor (2019)", "Okafor and Lindqvist 2019", "Okafor et al., 2019a", "(Okafor, 2019)".
_CITE_RE = re.compile(
    r"(?<![A-Za-z])([A-Z][A-Za-z'\u2019\-]{2,})"
    r"(?:\s+(?:and|&)\s+([A-Z][A-Za-z'\u2019\-]{2,})|\s+et\s+al\.?)?"
    r"[\s,]*\(?\s*((?:19|20)\d{2})[a-z]?\)?"
)
# When one surname and one year still name more works than this, the query is
# too ambiguous to act on (a common surname); the other legs keep running.
_AY_MAX_TIES = 9
_CITE_MAX_WORKS_PER_MENTION = 3

_ay_index: tuple | None = (
    None  # (mtime, surname -> keys, year -> keys, key -> (surnames, year, title words))
)


def _surname_forms(author: str) -> set[str]:
    """Lower-case surname forms of a creator string ("Last First" or "Last, First").

    The whole surname, plus each hyphen- or space-separated part of at least
    four letters, so "Ghobadi-Far Khosro" answers to "ghobadi-far" and "ghobadi".
    """
    author = (author or "").strip()
    if not author or "@" in author:
        return set()
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", author)}
    # A person carries a given name or an initial; a lone token is an organisation.
    if words & _ORG_WORDS or ("," not in author and len(author.split()) < 2):
        return set()
    if "," in author:
        last = author.split(",", 1)[0]
    else:
        # "Last First [Middle]": the surname is the first token, continued only
        # through particles ("van der Lind Tamás"), so the given names that
        # follow it ("Varga Grace E.") never become surname forms.
        parts = author.split()
        keep = parts[:1]
        for prev, tok in zip(parts, parts[1:]):
            if tok.lower() in _NAME_PARTICLES or prev.lower() in _NAME_PARTICLES:
                keep.append(tok)
            else:
                break
        last = " ".join(keep)
    last = last.lower().strip()
    forms = {last} if len(last) >= 3 else set()
    forms.update(p for p in re.split(r"[\s\-\u2010\u2011]+", last) if len(p) >= 4)
    return forms - _NOT_SURNAMES


def _author_year_index() -> tuple:
    """Surname and year indexes over the metadata, rebuilt when the file changes."""
    global _ay_index
    meta = _load()
    if _ay_index is not None and _ay_index[0] == _mtime:
        return _ay_index
    by_surname: dict[str, set[str]] = {}
    by_year: dict[str, set[str]] = {}
    works: dict[str, tuple[frozenset[str], str, frozenset[str]]] = {}
    for key, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        year = str(entry.get("year") or "").strip()[:4]
        names: set[str] = set()
        for author in entry.get("authors") or []:
            if isinstance(author, str):
                names |= _surname_forms(author)
        for name in names:
            by_surname.setdefault(name, set()).add(key)
        if year:
            by_year.setdefault(year, set()).add(key)
        title_words = frozenset(
            w.lower() for w in _AY_TITLE_WORD_RE.findall(str(entry.get("title") or ""))
        )
        works[key] = (frozenset(names), year, title_words)
    _ay_index = (_mtime, by_surname, by_year, works)
    return _ay_index


def find_works(query: str, max_works: int = 3) -> list[tuple[str, int]]:
    """Zotero keys of the works a query names by author surname(s) and year,
    best first, each with its match score (surnames matched, +1 for the year).

    A year in the query must match; a work of an adjacent year stands in
    without the year point when no work matches the year exactly, or when it
    matches more of the surnames than any exact-year work does (a journal's
    online-first and print years differ by one). Without a year, a single surname counts
    only when it names at most two works (a rare name), two or more surnames
    always. Works tied at the best score are all returned, past ``max_works``
    if need be: a tie is broken by how many of the query's other content words
    a title shares, then by key, and an alphabetical cut would drop the right
    paper when a pair of authors published several times in one year. When more
    than ``_AY_MAX_TIES`` works tie the query is too ambiguous and nothing is
    returned. Empty when the metadata is absent.
    """
    _, by_surname, _by_year, works = _author_year_index()
    if not works or not query:
        return []
    years = set(_AY_YEAR_RE.findall(query))
    tokens = {t.lower() for t in _AY_TOKEN_RE.findall(query)}
    surnames = {t for t in tokens if t in by_surname}
    if not surnames:
        return []
    matched: dict[str, int] = {}
    for name in surnames:
        for key in by_surname[name]:
            matched[key] = matched.get(key, 0) + 1
    scored: list[tuple[str, int]] = []
    if years:
        adjacent = {str(int(y) + d) for y in years for d in (-1, 1)} - years
        scored = [(k, h + 1) for k, h in matched.items() if works[k][1] in years]
        floor = max((h for _, h in scored), default=0)
        scored += [
            (k, h)
            for k, h in matched.items()
            if works[k][1] in adjacent and h + 1 > floor
        ]
    else:
        for key, hits in matched.items():
            if hits == 1:
                name = next(n for n in surnames if n in works[key][0])
                if len(by_surname[name]) > 2:
                    continue
            scored.append((key, hits))
    if not scored:
        return []
    content = tokens - surnames
    scored.sort(key=lambda item: (-item[1], -len(works[item[0]][2] & content), item[0]))
    best = scored[0][1]
    ties = sum(1 for _, score in scored if score == best)
    if ties > _AY_MAX_TIES:
        return []
    return scored[: max(max_works, ties)]


def query_has_year(query: str) -> bool:
    """Whether the query carries a four-digit year, i.e. names a work firmly
    enough for :func:`find_works`'s result to be pinned or missed."""
    return bool(_AY_YEAR_RE.search(query or ""))


def specific_named_work(query: str, named: list[tuple[str, int]]) -> str | None:
    """The one work ``query`` names beyond doubt, else None.

    ``named`` is :func:`find_works`'s result. The first work is specific when
    it is alone, when it beats the runner-up on the match score, or when, at
    equal score, its title shares at least two more of the query's words than
    the runner-up's -- "Scanlon 2016 global evaluation of mascon products"
    means one of the three Scanlon 2016 papers, while one shared word such as
    the mission's name decides nothing. Otherwise the query is ambiguous
    between them.
    """
    if not named:
        return None
    if len(named) == 1:
        return named[0][0]
    _, _, _, works = _author_year_index()
    tokens = {t.lower() for t in _AY_TOKEN_RE.findall(query or "")}

    def rank(item):
        key, score = item
        return (score, len(works.get(key, ((), "", frozenset()))[2] & tokens))

    (score, overlap), (runner_score, runner_overlap) = rank(named[0]), rank(named[1])
    if score > runner_score or (
        score == runner_score and overlap >= runner_overlap + 2
    ):
        return named[0][0]
    return None


def cited_works_in(
    text: str, exclude: set[str] | None = None, max_works: int = 4
) -> list[str]:
    """Zotero keys of the library works ``text`` cites by author and year, the
    most-cited first.

    A mention resolves when its first surname and year name at most
    ``_CITE_MAX_WORKS_PER_MENTION`` works (a second surname narrows further);
    ambiguous mentions are skipped. ``exclude`` drops works already in hand.
    """
    _, by_surname, by_year, works = _author_year_index()
    if not works or not text:
        return []
    counts: dict[str, int] = {}
    for match in _CITE_RE.finditer(text):
        first, second, year = (
            match.group(1).lower(),
            (match.group(2) or "").lower(),
            match.group(3),
        )
        candidates = by_surname.get(first, set()) & by_year.get(year, set())
        if second:
            candidates = {k for k in candidates if second in works[k][0]}
        if not candidates or len(candidates) > _CITE_MAX_WORKS_PER_MENTION:
            continue
        for key in candidates:
            counts[key] = counts.get(key, 0) + 1
    if exclude:
        counts = {k: c for k, c in counts.items() if k not in exclude}
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _ in ordered[:max_works]]
