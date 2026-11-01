#!/usr/bin/env python3
"""
lightrag_mcp_server.py — expose a Zotero-backed LightRAG knowledge base to any
MCP client (Claude Code, Claude Desktop, other agents) over stdio.

It's a thin, self-contained proxy over the LightRAG Server REST API. Five tools:

    search_library(query, mode, top_k, answer=False, include_graph=False,
                   passages=8)         -> compiled citations, then the retrieved
                                          passages (answer=True: the server's
                                          synthesized answer, with citations)
    paper_passages(paper, question)    -> up to 3 passages of ONE paper, ranked
                                          against a question
    find_paper(paper)                  -> lookup: is a paper indexed, only in
                                          Zotero, or in neither?
    add_note(text, description)        -> insert a note/snippet into the KB under
                                          a unique document name
    health()                           -> is the LightRAG server up?

Citations are compiled from the `references` array that stock LightRAG returns
(one {reference_id, file_path} per retrieved document): corpus files named
"<ZoteroKey>__<slug>.md" resolve through zotero_metadata.json to authors, year,
title, publication, DOI and a zotero:// link. Anything the answering LLM writes
as its own reference section is stripped and replaced.

Config (environment variables):
    LIGHTRAG_BASE_URL                  default http://localhost:9621
    LIGHTRAG_API_KEY                   sent as the X-API-Key header; must match
                                       LIGHTRAG_API_KEY in the server's .env
    ZOTERO_METADATA                    default ~/Zotero/zotero_metadata.json, the
                                       corpus builder's default output (~ is expanded)
    ZOTERO_STORAGE                     default ~/Zotero/storage, Zotero's attachment
                                       folders: paper_passages names a paper's file
                                       there when it exists (~ is expanded)
    LIGHTRAG_MCP_TIMEOUT               seconds per query, default 300
    LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT  seconds for the /query/data fallback, default 90
    LIGHTRAG_MCP_HEALTH_TIMEOUT        seconds for health(), default 15
    LIGHTRAG_MCP_MAX_TOP_K             upper bound applied to top_k, default 40
    LIGHTRAG_MCP_MAX_CHARS             longest retrieved-context result; lower-ranked
                                       passages are left out beyond it, default
                                       48000 (Claude Code saves a tool result over
                                       50,000 characters to a file instead)

find_paper and paper_passages ask the server's GET /graph/papers/search (the
zlightrag fork's paper lookup) which papers are indexed, and paper_passages asks
POST /graph/papers/passages for one paper's passages. A stock LightRAG Server has
neither route: the lookup then falls back to zotero_metadata.json alone, and
paper_passages to a /query that names the paper.

Passage headings keep the section path only: the levels of the server's
content_headings that do not place a passage (the paper's title, journal and author
line, an Abstract or Summary heading made the parent of later sections, fragments)
are dropped.

Run:  python3 lightrag_mcp_server.py     (normally launched by the MCP client)
Deps: pip install -r requirements.txt    (mcp 2.x, httpx)
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import httpx

# mcp 2.0 removed mcp.server.fastmcp; MCPServer is the successor.
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

BASE = os.environ.get("LIGHTRAG_BASE_URL", "http://localhost:9621").rstrip("/")
KEY = os.environ.get("LIGHTRAG_API_KEY", "")
HEADERS = {"X-API-Key": KEY} if KEY else {}
VALID_MODES = {"mix", "hybrid", "local", "global", "naive"}
QUERY_TIMEOUT = float(os.environ.get("LIGHTRAG_MCP_TIMEOUT", "300"))
REF_FALLBACK_TIMEOUT = float(os.environ.get("LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT", "90"))
HEALTH_TIMEOUT = float(os.environ.get("LIGHTRAG_MCP_HEALTH_TIMEOUT", "15"))
MAX_TOP_K = max(1, int(os.environ.get("LIGHTRAG_MCP_MAX_TOP_K", "40")))
MAX_CHARS = max(1000, int(os.environ.get("LIGHTRAG_MCP_MAX_CHARS", "48000")))
# The server's paper lookup caps its answer at 100 papers.
PAPER_SEARCH_LIMIT = 100
# search_library's passages (the server's chunk_top_k), about 4 KB each.
DEFAULT_PASSAGES = 8
MAX_PASSAGES = 12
# Paper lookups are cached per request for PAPER_CACHE_TTL seconds.
PAPER_CACHE_TTL = 120.0
PAPER_CACHE_SIZE = 256
# Rows of a lookup list shown before the rest are counted.
LIST_ROWS = 10
# A lookup asks the server about at most CONFIRM_MAX Zotero items by key: 50 keys
# to a request on a server with the folding lookup, else one request per key, 4 at
# a time. Items past it are listed as "index status not checked".
CONFIRM_MAX = 100
CONFIRM_BATCH = 50
CONFIRM_THREADS = 4

# Zotero metadata for deterministic citations. Answering LLMs fabricate the
# "### References" block (placeholder titles, wrong years), so we strip that and
# compile real citations from the server's structured `references` array
# (chunk file_paths are "<ZoteroKey>__<slug>.md").
META_PATH = os.path.expanduser(
    os.environ.get("ZOTERO_METADATA") or "~/Zotero/zotero_metadata.json"
)
# Zotero's attachment folders, one per corpus key: paper_passages names the paper's
# file there, for what the index lacks (display equations).
STORAGE = os.path.expanduser(os.environ.get("ZOTERO_STORAGE") or "~/Zotero/storage")
_META_SIGNATURE = None
_BAD_SIGNATURE = None


def _signature(stat):
    return stat.st_ino, stat.st_mtime_ns, stat.st_size


try:
    with open(META_PATH, encoding="utf-8") as _f:
        _META = json.load(_f)
        if not isinstance(_META, dict):
            raise ValueError("metadata must be a JSON object")
        _META_SIGNATURE = _signature(os.fstat(_f.fileno()))
except Exception as _e:
    _META = {}
    # stderr, never stdout: stdout carries the MCP protocol.
    print(
        f"lightrag_mcp_server: no citation metadata loaded from "
        f"{os.path.abspath(META_PATH)} ({type(_e).__name__}); citations fall "
        f"back to file names",
        file=sys.stderr,
    )


def _metadata() -> dict:
    """Refresh citations after an atomic metadata rebuild, without restarting MCP."""
    global _META, _META_SIGNATURE, _BAD_SIGNATURE
    try:
        signature = _signature(os.stat(META_PATH))
    except OSError:
        return _META
    if signature == _META_SIGNATURE:
        return _META
    try:
        with open(META_PATH, encoding="utf-8") as source:
            updated = json.load(source)
            if not isinstance(updated, dict):
                raise ValueError("metadata must be a JSON object")
            signature = _signature(os.fstat(source.fileno()))
    except (OSError, ValueError) as exc:
        if signature != _BAD_SIGNATURE:
            print(
                f"lightrag_mcp_server: cannot refresh {META_PATH}: {exc}",
                file=sys.stderr,
            )
            _BAD_SIGNATURE = signature
        return _META
    _META, _META_SIGNATURE, _BAD_SIGNATURE = updated, signature, None
    return _META


mcp = MCPServer("zotero-lightrag")


def _strip_llm_refs(text: str) -> str:
    """Remove a References/Sources heading and its entry lines wherever they appear.

    Two producers: the answering LLM fabricates them (and sometimes continues
    prose afterwards), and a server that compiles its own citations appends a
    "### Sources" block. Strip both, or this proxy's own compiled block is a
    duplicate rather than a replacement. The heading must end the line, so prose
    headings such as "### Sources of error" are kept.
    """
    out, skipping = [], False
    for line in text.splitlines():
        if re.match(
            r"\s*(#{1,6}\s*)?(References|Sources)\s*:?\s*$", line, re.IGNORECASE
        ):
            skipping = True
            continue
        if skipping:
            if not line.strip() or re.match(r"\s*[-*]?\s*\[\d+\]", line):
                continue
            skipping = False
        out.append(line)
    return "\n".join(out).rstrip()


def _authors(m: dict) -> str | None:
    """Authors as 'Family, Given; ...' from structured creators, else the 'authors' strings."""
    names = []
    for p in m.get("creators") or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            name = ", ".join(
                x for x in (str(p[0] or "").strip(), str(p[1] or "").strip()) if x
            )
            if name:
                names.append(name)
    if names:
        return "; ".join(names)
    return ", ".join(m.get("authors") or []) or None


def _cite(file_path: str, server_citation: str = "") -> str:
    name = os.path.basename(file_path or "")
    if not re.match(r"^[A-Z0-9]{8}__", name):
        return name or "(untitled source)"  # notes / non-corpus sources
    key = name[:8]
    if server_citation:
        cite = server_citation
    elif m := _metadata().get(key):
        authors = _authors(m)
        parts = [
            p
            for p in (
                authors,
                f"({m['year']})" if m.get("year") else None,
                m.get("title") or None,
                m.get("publication") or None,
                f"doi:{m['doi']}" if m.get("doi") else None,
            )
            if p
        ]
        cite = ". ".join(parts)
    else:
        cite = re.sub(r"_+", " ", re.sub(r"\.md$", "", name)[10:]).strip()
    return f"{cite or name} [zotero://select/library/items/{key}]"


def _compiled_refs(references: list, label: str) -> str:
    if not references:
        return ""
    lines = [
        f"- [{r.get('reference_id', '?')}] {_cite(r.get('file_path', ''), r.get('citation') or '')}"
        for r in references
    ]
    return f"\n\n### {label}\n" + "\n".join(lines)


def _top_k(top_k: int) -> int:
    """Clamp top_k to [1, MAX_TOP_K]. The server rejects values below 1, and a
    large top_k fills its token budget with graph entities and relations,
    leaving fewer (or no) document chunks and therefore fewer citations."""
    return max(1, min(int(top_k), MAX_TOP_K))


# References for an identical (query, mode, top_k) within this server process.
# Lives only as long as the process; a stdio MCP server is per-session.
_REF_MEMO: dict = {}


def _error_detail(r: httpx.Response) -> str:
    """The server's explanation of a refusal: FastAPI's {"detail": "..."} when it
    is a string, otherwise the raw body, truncated."""
    try:
        detail = r.json()["detail"]
    except Exception:
        detail = None
    return (detail if isinstance(detail, str) else r.text).strip()[:500]


def _tool_error(path: str, e: httpx.HTTPError) -> ToolError:
    # mcp >= 2.1 shows the client only "Error executing tool <name>" for an
    # unexpected exception; a ToolError's message reaches it on every 2.x.
    msg = f"LightRAG {path} failed: {e}"
    # The status line does not say why; LightRAG's body does (a 409 names the
    # document that already exists, or the scan or deletion blocking inserts).
    if isinstance(e, httpx.HTTPStatusError) and (detail := _error_detail(e.response)):
        msg += f"\nServer response: {detail}"
    return ToolError(msg)


def _post(path: str, payload: dict, timeout: float | None = None) -> dict:
    try:
        r = httpx.post(
            f"{BASE}{path}",
            json=payload,
            headers=HEADERS,
            timeout=QUERY_TIMEOUT if timeout is None else timeout,
        )
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise _tool_error(path, e) from e
    try:
        return r.json()
    except Exception:
        return {"response": r.text}


def _zotero_key(file_path: str) -> str:
    """The Zotero key a corpus file is named by ("<KEY>__<slug>.md"), else ''."""
    name = os.path.basename(file_path or "")
    return name[:8] if re.match(r"^[A-Z0-9]{8}__", name) else ""


def _norm_doi(text: str) -> str:
    """Lowercase DOI without a resolver prefix or 'doi:'; '' when not a DOI. The
    same normalisation as the server's paper lookup."""
    d = (text or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d if d.startswith("10.") and "/" in d else ""


def _entry(key: str) -> dict:
    m = _metadata().get(key) if key else None
    return m if isinstance(m, dict) else {}


def _first_family(m: dict) -> str:
    """The first author's family name in a metadata entry: structured creators
    first, then the plain author string."""
    creators = m.get("creators") or []
    if creators and isinstance(creators[0], (list, tuple)) and creators[0]:
        if family := str(creators[0][0] or "").strip():
            return family
    return str(m["authors"][0]).strip() if m.get("authors") else ""


def _first_author_year(key: str, server_citation: str = "") -> tuple[str, str]:
    """(first author's family name, year) of a corpus document, for passage
    labels and query leads: structured creators first, then the server's
    citation ("Family, Given; ... (Year)."), then the plain author string."""
    m = _entry(key)
    creators = m.get("creators") or []
    family = ""
    if creators and isinstance(creators[0], (list, tuple)) and creators[0]:
        family = str(creators[0][0] or "").strip()
    if not family and (c := re.match(r"\s*([^,;.(]+)", server_citation or "")):
        family = c.group(1).strip()
    if not family:
        family = _first_family(m)
    year = str(m.get("year") or "")
    if not year and (y := re.search(r"\((\d{4})\)", server_citation or "")):
        year = y.group(1)
    return family, year


def _label(ref: dict) -> str:
    """'Family Year' for a passage heading, so that passages of several papers on
    one topic are not attributed to the wrong one."""
    file_path = ref.get("file_path") or ""
    key = _zotero_key(file_path)
    if not key:
        return os.path.basename(file_path) or "(untitled source)"
    family, year = _first_author_year(key, ref.get("citation") or "")
    return " ".join(p for p in (family, year) if p) or key


def _chunks(ctx: str) -> list[dict]:
    """The document chunks of a context-only response, in the server's ranked
    order: one JSON object per line carrying reference_id and content (and
    content_headings on servers that send it)."""
    chunks = []
    for line in ctx.splitlines():
        if not line.startswith('{"reference_id"'):
            continue
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if isinstance(c, dict) and isinstance(c.get("content"), str):
            chunks.append(c)
    return chunks


# The block corpus/build_corpus.py header() puts at the top of every corpus file,
# so at the top of each paper's first chunk. The chunker may drop its blank lines.
_HEADER = re.compile(
    r"# [^\n]*\n\n?"
    r"(?:\*\*Authors:\*\* [^\n]*\n)?"
    r"(?:\*\*(?:Year|Published in|DOI):\*\* [^\n]*\n)?"
    r"(?:\*\*Tags:\*\* [^\n]*\n)?"
    r"\*\*Zotero:\*\* zotero://select/library/items/[A-Z0-9]{8}[ \t]*\n"
    r"(?:\n?(> \*\*Abstract\.\*\* [^\n]*)\n)?"
    r"\n?---[ \t]*(?:\n|\Z)"
)
_ABSTRACT = "> **Abstract.** "
# The word Zotero's copy of an abstract often starts with, glued or not
# ("AbstractThe widget census ...").
_ABSTRACT_WORD = re.compile(
    r"^\s*(?:Abstract|ABSTRACT|Summary|SUMMARY)[\s.:\u2014-]*(?=\S)"
)


def _strip_header(content: str) -> str:
    """A chunk's text without the corpus header, which the citation already
    carries. Its abstract line is kept only when the text after the header does
    not repeat that abstract (the PDF's own first page usually does). They are
    compared as bare letters and digits, without a leading "Abstract": the PDF's
    copy carries emphasis and a heading of its own."""
    m = _HEADER.match(content.lstrip())
    if not m:
        return content
    rest = content.lstrip()[m.end() :].lstrip("\n")
    abstract = m.group(1)
    if abstract:
        text = _ABSTRACT_WORD.sub("", abstract[len(_ABSTRACT) :])
        probe = _plain(text).replace(" ", "")[:80]
        if probe and probe in _plain(rest).replace(" ", ""):
            abstract = None
    if abstract:
        return f"{abstract}\n\n{rest}" if rest.strip() else abstract
    return rest if rest.strip() else content


# The server's content_headings is a chunk's parent heading chain, "h1 → h2", each
# level cut at about 80 characters. In a sample of 40 papers most levels did not place
# the passage: the paper's own title led the chain in most of them (cut, bold or
# garbled by the PDF converter, while the citation already names the paper), and
# journal names, author lines, an Abstract or SUMMARY heading that the converter made
# the parent of every later section, and equation or link fragments took levels of
# their own.
_HEADING_SEP = " → "
_TAGS = re.compile(r"<[^>]*>")
_EMPHASIS = re.compile(r"\*\*|__|~~|`|(?<!\w)_|_(?!\w)")
_JUNK_LEVEL = re.compile(
    r"=|\||https?:|www\.|terms and conditions|downloaded from"
    r"|\b(?:received|accepted|revised)\b.*\b(?:1[89]|20)\d\d\b",
    re.IGNORECASE,
)
# Levels that label the text rather than place it, as bare letters.
_LABEL_LEVELS = {
    "abstract",
    "summary",
    "letter",
    "article",
    "originalarticle",
    "researcharticle",
    "reviewarticle",
    "researchpaper",
}


def _section(headings, key: str = "", citation: str = "") -> str:
    """A chunk's heading chain without the levels that do not place it in its
    paper: the paper's title (the citation names it), its journal and author
    line, Abstract/Summary and article-type labels, and fragments with no word
    of three letters, an '=' or '|', a link or a received/accepted date. Bold,
    italics, HTML tags and backticks are removed. '' when no level is left."""
    if not headings:
        return ""
    m = _entry(key)
    # Without a metadata entry, the title from the server's citation:
    # "Family, Given; ... (Year). Title. Journal. doi:..."
    cited = re.search(r"\(\d{4}\)\.\s*(.+?)(?:\.\s|\.?$)", citation or "")
    title = m.get("title") or (cited.group(1) if cited else "")
    title = _plain(_TAGS.sub("", title)).replace(" ", "")
    journal = _plain(m.get("publication") or "").replace(" ", "")
    families = {
        f
        for f in (
            _plain(c[0])
            for c in m.get("creators") or []
            if isinstance(c, (list, tuple)) and c
        )
        if len(f) >= 2
    }
    kept = []
    for level in str(headings).split(_HEADING_SEP):
        text = " ".join(_EMPHASIS.sub("", _TAGS.sub("", level)).split())
        text = text.strip(" -–—:|")
        plain = _plain(text)
        flat = plain.replace(" ", "")
        if (
            not re.search(r"[^\W\d_]{3}", text)
            or _JUNK_LEVEL.search(text)
            or flat in _LABEL_LEVELS
            or flat == journal
        ):
            continue
        words = [w for w in plain.split() if len(w) >= 3]
        if (
            title
            and words
            and len(flat) >= min(0.6 * len(title), 50)
            and sum(w in title for w in words) >= 0.8 * len(words)
        ):
            continue  # the title, whole or cut ("…"), or garbled into word pieces
        if (
            len(families) >= 2
            and sum(bool(re.search(rf"\b{re.escape(f)}\b", plain)) for f in families)
            >= 2
        ):
            continue  # the author line
        if not kept or kept[-1] != text:
            kept.append(text)
    return _HEADING_SEP.join(kept)


def _passages(
    chunks: list[dict], labels: dict, sources: dict | None = None
) -> list[str]:
    """Each chunk as plain text headed by its [n], label and section, its corpus
    header stripped. A chunk whose text repeats an earlier one's (another Zotero
    copy of the same paper) is dropped, and its [n] joins the earlier heading.
    sources maps [n] to (Zotero key, server citation) for the section cleanup."""
    groups, seen = [], {}
    for c in chunks:
        rid = str(c.get("reference_id", "?"))
        body = _strip_header(c["content"]).strip()
        same = " ".join(body.split())
        if same in seen:
            ids = groups[seen[same]][0]
            if rid not in ids:
                ids.append(rid)
            continue
        seen[same] = len(groups)
        groups.append(([rid], c, body))
    out = []
    for ids, c, body in groups:
        head = ("".join(f"[{i}]" for i in ids) + f" {labels.get(ids[0], '')}").rstrip()
        key, citation = (sources or {}).get(ids[0], ("", ""))
        if section := _section(c.get("content_headings"), key, citation):
            head += f" · {section}"
        out.append(f"{head}\n{body}")
    return out


def _sources(references: list) -> dict:
    """[n] -> (Zotero key, server citation), for _passages."""
    return {
        str(r.get("reference_id")): (
            _zotero_key(r.get("file_path", "")),
            r.get("citation") or "",
        )
        for r in references
    }


def _fit(passages: list[str], budget: int) -> list[str]:
    """The leading passages that fit in budget characters (always the first)."""
    shown, used = [], 0
    for p in passages:
        if shown and used + len(p) + 2 > budget:
            break
        shown.append(p)
        used += len(p) + 2
    return shown


def _render_context(data: dict, include_graph: bool) -> str:
    """Compiled citations FIRST, then the passages as plain text, each headed by its
    [n], first author and year, and section; the knowledge-graph entities and
    relations only when asked for. Lower-ranked passages are left out beyond
    MAX_CHARS, and the result says how many."""
    ctx = data.get("response") or data.get("context") or str(data)
    references = data.get("references") or []
    refs = _compiled_refs(
        references, "Retrieved documents (compiled citations)"
    ).strip()
    chunks = _chunks(ctx)
    if not chunks:
        # Not the chunk format this proxy reads: pass the context through whole.
        return f"{refs}\n\n{ctx}" if refs else ctx
    labels = {str(r.get("reference_id")): _label(r) for r in references}
    # Stock LightRAG puts the entities and relations ahead of the chunks.
    graph = ctx.split("Document Chunks", 1)[0].strip() if include_graph else ""
    head = (f"{refs}\n\n" if refs else "") + (
        "### Passages (the server's ranked order; [n] is the document listed above)"
    )
    budget = MAX_CHARS - len(head) - len(graph) - 300
    passages = _passages(chunks, labels, _sources(references))
    shown = _fit(passages, budget)
    out = f"{head}\n\n" + "\n\n".join(shown)
    if left := len(passages) - len(shown):
        out += (
            f"\n\n[{left} lower-ranked passage(s) left out to keep this result under "
            f"{MAX_CHARS} characters. Ask a narrower question, or use paper_passages "
            "for one paper.]"
        )
    return f"{out}\n\n{graph}" if graph else out


def _context(
    query: str, mode: str, top_k: int, passages: int, include_graph: bool = False
) -> str:
    """Retrieval only — no LLM answer."""
    data = _post(
        "/query",
        {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": passages,
            "only_need_context": True,
        },
    )
    return _render_context(data, include_graph)


# --- paper lookup ----------------------------------------------------------------
#
# Words are folded the same way on both sides, here and in the server's paper
# lookup: Zotero titles carry U+2010 hyphens, curly apostrophes and accents that
# a typed query does not.

_FOLD_LETTERS = str.maketrans(
    {
        "ı": "i",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "ß": "ss",
        "ẞ": "SS",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
        "þ": "th",
        "Þ": "TH",
    }
)
_FOLD_MARKS = str.maketrans(
    {
        **{chr(c): "-" for c in range(0x2010, 0x2016)},  # hyphens and dashes
        "\u2212": "-",  # minus sign
        "\u00ad": "-",  # soft hyphen
        **{c: None for c in "'\u2019\u2018\u02bc\u2032`"},  # apostrophes, primes
    }
)
# stripped from both ends of a query token: brackets, sentence punctuation, '*'
# (emphasis) and typographic double quotes and guillemets
_PUNCT = '()[]{},.;:"!?*\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a'
_BIBTEX = re.compile(r"^([a-z][a-z]*)((?:[A-Z][A-Za-z0-9]*)+)(\d{4})[a-z]?$")
# camel-case words, acronym runs ("ABCFollow" -> ABC, Follow) and digit runs, as
# the server's (lightrag/zotero_papers.py)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_YEAR = re.compile(r"^\d{4}$")


def _fold(text) -> str:
    """Lowercase, without accents, apostrophes or primes, every hyphen and dash
    an ASCII '-'."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.translate(_FOLD_LETTERS).translate(_FOLD_MARKS).lower()


def _plain(text: str) -> str:
    """Folded text as its alphanumeric words joined by single spaces."""
    return " ".join(re.findall(r"[a-z0-9]+", _fold(text)))


def _word_in(word: str, hay: str, flat: str | None = None, loose: bool = True) -> bool:
    """Whether a folded query word occurs in a folded haystack: as it is, and
    (``loose``) without its hyphens (against the haystack without hyphens,
    ``flat``) or as every part of it split on '-' or '/'. A DOI-shaped query is
    matched whole, not loosely: its parts ("10.1", "x") are in most haystacks."""
    if word in hay:
        return True
    if not loose:
        return False
    flat = hay.replace("-", "") if flat is None else flat
    if word.replace("-", "") in flat:
        return True
    parts = [p for p in re.split(r"[-/]", word) if p]
    return len(parts) > 1 and all(p in hay for p in parts)


def _tokens(q: str) -> list[str]:
    """The whitespace-separated tokens of a lookup query, each without its edge
    punctuation, and without the "et al" that follows a name in a citation
    ("Doe et al. 2011"). The server's paper search carries the same rule
    (``_query_tokens`` in lightrag/zotero_papers.py)."""
    raw = [t for t in (tok.strip(_PUNCT) for tok in (q or "").split()) if t]
    tokens: list[str] = []
    i = 0
    while i < len(raw):
        if (
            tokens
            and i + 1 < len(raw)
            and _fold(raw[i]) == "et"
            and _fold(raw[i + 1]) == "al"
        ):
            i += 2
            continue
        tokens.append(raw[i])
        i += 1
    return tokens


def _query_words(q: str) -> list[tuple[str, bool]]:
    """The words of a lookup query, folded, as ``(word, is_title_word)``, in the
    server's order (``query_words``): a token with no letter or digit left (the
    " - " of "Author - Year - Title", an "&") is no word; a BibTeX key
    ("smithSomeTitle2001") becomes its family, its camel-case title words
    (matched against the haystack without its non-alphanumerics) and its year.
    ``tests/mcp/paper_search_cases.json`` holds the cases both sides run."""
    words: list[tuple[str, bool]] = []
    for token in _tokens(q):
        if b := _BIBTEX.match(token):
            words.append((b.group(1), False))
            words += [(w.lower(), True) for w in _CAMEL.findall(b.group(2))]
            words.append((b.group(3), False))
            continue
        word = _fold(token).strip(_PUNCT)
        if any(c.isalnum() for c in word):
            words.append((word, False))
    return words


def _terms(q: str) -> tuple[list[str], list[str], str]:
    """(words, BibTeX title words, BibTeX family) of a lookup query."""
    pairs = _query_words(q)
    family = next((b.group(1) for t in _tokens(q) if (b := _BIBTEX.match(t))), "")
    return (
        [w for w, title in pairs if not title],
        [w for w, title in pairs if title],
        family,
    )


_ROWS: dict = {"meta": None, "rows": {}}


def _rows() -> tuple[dict, dict]:
    """(metadata, key -> folded haystacks of its entry), rebuilt when the
    metadata is."""
    meta = _metadata()
    if _ROWS["meta"] is not meta:
        rows = {}
        for key, m in meta.items():
            if not isinstance(m, dict):
                continue
            title = _fold(m.get("title") or "")
            authors = _fold(" ".join(str(a) for a in m.get("authors") or []))
            year = str(m.get("year") or "")
            doi = _norm_doi(m.get("doi") or "")
            hay = " ".join([title, authors, year, doi, key.lower()])
            families = set()
            for c in m.get("creators") or []:
                if isinstance(c, (list, tuple)) and c:
                    families.update(re.findall(r"[a-z0-9]+", _fold(c[0])))
            if not families:  # no structured names: any word of the author strings
                families.update(re.findall(r"[a-z0-9]+", authors))
            rows[key] = {
                "hay": hay,
                "flat": hay.replace("-", ""),
                "alnum": re.sub(r"[^a-z0-9]", "", hay),
                "title": title,
                "title_alnum": re.sub(r"[^a-z0-9]", "", title),
                "title_words": " ".join(re.findall(r"[a-z0-9]+", title)),
                "families": families,
                "first": _fold(_first_family(m)),
                "year": year,
                "doi": doi,
            }
        _ROWS.update(meta=meta, rows=rows)
    return _ROWS["meta"], _ROWS["rows"]


def _matches(
    row: dict, words: list[str], title_words: list[str], loose: bool = True
) -> bool:
    return all(_word_in(w, row["hay"], row["flat"], loose) for w in words) and all(
        t in row["alnum"] for t in title_words
    )


def _title_hits(row: dict, words: list[str], title_words: list[str]) -> int:
    return sum(_word_in(w, row["title"]) for w in words) + sum(
        t in row["title_alnum"] for t in title_words
    )


def _whole_word(word: str, hay: str) -> bool:
    """Whether a folded query word occurs in a folded haystack with no letter or
    digit on either side (the server's ``_whole_word``)."""
    return re.search(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", hay) is not None


def _newest(row: dict) -> int:
    return -int(row["year"]) if row["year"].isdigit() else 0


def _metadata_matches(q: str) -> list[tuple[str, dict]]:
    """Corpus metadata entries matching q: an exact Zotero key or DOI, else every
    word of q (folded) in the title, authors, year, DOI or key; most title words
    first, then the newest."""
    key_q, doi_q = q.strip().upper(), _norm_doi(q)
    meta, rows = _rows()
    exact = [
        (k, meta[k])
        for k, r in rows.items()
        if k == key_q or (doi_q and r["doi"] == doi_q)
    ]
    if exact:
        return exact
    words, title_words, _ = _terms(q)
    if not words and not title_words:
        return []
    found = [k for k, r in rows.items() if _matches(r, words, title_words, not doi_q)]
    # whole-word matches first: a short surname ("ho") is inside many title words
    found.sort(
        key=lambda k: (
            not all(_whole_word(w, rows[k]["hay"]) for w in words),
            -_title_hits(rows[k], words, title_words),
            _newest(rows[k]),
        )
    )
    return [(k, meta[k]) for k in found]


def _near_matches(q: str) -> list[tuple[str, dict]]:
    """Up to 5 entries that miss one condition of q: every word but a year that is
    off by one, then the entries by the query's surname (the BibTeX family, else
    its first non-numeric word), within a year of a year it gives."""
    words, title_words, family = _terms(q)
    years = [w for w in words if _YEAR.match(w)]
    others = [w for w in words if not _YEAR.match(w)]
    surname = family or next((w for w in words if not w.isdigit()), "")
    meta, rows = _rows()

    def within_a_year(r):
        return r["year"].isdigit() and all(
            abs(int(r["year"]) - int(y)) <= 1 for y in years
        )

    def rank(k):
        return -_title_hits(rows[k], others, title_words), _newest(rows[k])

    by_year = [
        k
        for k, r in rows.items()
        if years
        and (others or title_words)
        and within_a_year(r)
        and _matches(r, others, title_words, not _norm_doi(q))
    ]
    by_name = [
        k
        for k, r in rows.items()
        if surname
        and surname in r["families"]
        and k not in by_year
        and (not years or within_a_year(r))
    ]
    near, dois = [], set()
    for k in sorted(by_year, key=rank) + sorted(by_name, key=rank):
        doi = rows[k]["doi"]
        if doi and doi in dois:
            continue
        dois.add(doi)
        near.append((k, meta[k]))
        if len(near) == 5:
            break
    return near


_PAPER_CACHE: dict = {}
_PAPER_CACHE_LOCK = threading.Lock()


def _paper_search(**params) -> tuple[list[dict] | None, bool]:
    """(indexed papers matching ``q`` or carrying one of ``keys``, best match first,
    or None when the server has no paper lookup, as a stock LightRAG Server;
    whether it is the lookup that folds words and takes keys=). Cached for
    PAPER_CACHE_TTL seconds."""
    path = "/graph/papers/search"
    params = {**params, "limit": PAPER_SEARCH_LIMIT}
    cache_key = tuple(sorted(params.items()))
    with _PAPER_CACHE_LOCK:
        hit = _PAPER_CACHE.get(cache_key)
    if hit and time.monotonic() - hit[0] < PAPER_CACHE_TTL:
        return (None if hit[1] is None else list(hit[1])), hit[2]
    try:
        r = httpx.get(
            f"{BASE}{path}", params=params, headers=HEADERS, timeout=QUERY_TIMEOUT
        )
        if r.status_code == 404:
            papers = None
        else:
            r.raise_for_status()
            papers = r.json()
            papers = (
                [p for p in papers if isinstance(p, dict)]
                if isinstance(papers, list)
                else []
            )
    except httpx.HTTPError as e:
        raise _tool_error(path, e) from e
    except ValueError as e:
        raise ToolError(f"LightRAG {path} returned no JSON: {e}") from e
    new = r.headers.get("X-Paper-Search") == "2"
    with _PAPER_CACHE_LOCK:
        _PAPER_CACHE.pop(cache_key, None)
        while len(_PAPER_CACHE) >= PAPER_CACHE_SIZE:
            _PAPER_CACHE.pop(next(iter(_PAPER_CACHE)))
        _PAPER_CACHE[cache_key] = (time.monotonic(), papers, new)
    return (None if papers is None else list(papers)), new


def _confirm(keys: list[str], new: bool) -> dict[str, dict]:
    """Zotero key -> the indexed paper carrying it, for the keys the server knows:
    keys= requests of CONFIRM_BATCH keys on a new server, else one request per key,
    CONFIRM_THREADS at a time."""
    if not keys:
        return {}
    if new:
        chunks = [
            ",".join(keys[i : i + CONFIRM_BATCH])
            for i in range(0, len(keys), CONFIRM_BATCH)
        ]
        batches = [_paper_search(keys=c)[0] or [] for c in chunks]
    else:
        with ThreadPoolExecutor(CONFIRM_THREADS) as pool:
            batches = list(pool.map(lambda k: _paper_search(q=k)[0] or [], keys))
    found = {}
    for papers in batches:
        for p in papers:
            for k in p.get("keys") or []:
                found.setdefault(k, p)
    return {k: found[k] for k in keys if k in found}


def _ranked(q: str, papers: list[dict]) -> list[dict]:
    """Indexed papers whose title is the query's title words (those that are not
    an author's name or a year) first, then those holding them as a phrase, then
    those holding every query word as a whole word, then by query words in the
    title, then in the server's order."""
    words, title_words, _ = _terms(q)
    rows = _rows()[1]

    def rank(item):
        i, p = item
        m = next((_entry(k) for k in p.get("keys") or [] if _entry(k)), {})
        title = _fold(m.get("title") or p.get("title") or "")
        authors = (
            " ".join(str(a) for a in m.get("authors") or [])
            if m
            else (p.get("citation") or "").split("(")[0]
        )
        named = set(re.findall(r"[a-z0-9]+", _fold(authors)))
        content = [w for w in words if not _YEAR.match(w) and w not in named]
        phrase = _plain(" ".join(content + title_words))
        flat = " ".join(re.findall(r"[a-z0-9]+", title))
        row = {"title": title, "title_alnum": re.sub(r"[^a-z0-9]", "", title)}
        local = next((rows[k] for k in p.get("keys") or [] if k in rows), None)
        hay = (
            local["hay"]
            if local
            else _fold(
                " ".join(
                    [p.get("title") or "", p.get("citation") or "", p.get("doi") or ""]
                    + list(p.get("keys") or [])
                )
            )
        )
        return (
            not (phrase and flat == phrase),
            not (phrase and f" {phrase} " in f" {flat} "),
            not all(_whole_word(w, hay) for w in words),
            -_title_hits(row, words, title_words),
            i,
        )

    return [p for _, p in sorted(enumerate(papers), key=rank)]


def _lookup(paper: str):
    """(indexed papers, or None when the server cannot say; Zotero items that match
    but are not indexed, one per DOI; whether the server cut its list at its limit;
    near matches when nothing has every word, as (key, entry, the indexed paper,
    None when not indexed, or "?" when the server cannot say); Zotero items whose
    index status was not asked, one per DOI).

    An item is reported as not indexed only after the server, asked for its exact
    key, does not know it either. Items are asked about in order until LIST_ROWS
    of them are not indexed or CONFIRM_MAX keys were asked; the rest are returned
    as not checked."""
    indexed, new = _paper_search(q=paper)
    local = _metadata_matches(paper)
    if indexed is None:
        near = [] if local else [(k, m, "?") for k, m in _near_matches(paper)]
        return None, local, False, near, []
    if len(indexed) >= PAPER_SEARCH_LIMIT:
        return _ranked(paper, indexed), [], True, [], []
    keys = {k for p in indexed for k in p.get("keys") or []}
    dois = {_norm_doi(p.get("doi") or "") for p in indexed} - {""}
    step = CONFIRM_BATCH if new else CONFIRM_THREADS
    queue, rest, asked = list(local), [], 0

    def take(limit: int) -> list[tuple[str, dict]]:
        """The next items not yet indexed or listed, at most ``limit``."""
        batch = []
        while queue and len(batch) < limit:
            key, m = queue.pop(0)
            doi = _norm_doi(m.get("doi") or "")
            if key not in keys and not (doi and doi in dois):
                batch.append((key, m))
        return batch

    while queue and len(rest) < LIST_ROWS and asked < CONFIRM_MAX:
        batch = take(min(step, CONFIRM_MAX - asked))
        asked += len(batch)
        confirmed = _confirm([k for k, _ in batch], new)
        for key, m in batch:
            doi = _norm_doi(m.get("doi") or "")
            if key in keys or (doi and doi in dois):
                continue  # a copy of a paper this batch found indexed or listed
            if found := confirmed.get(key):
                indexed.append(found)
                keys |= set(found.get("keys") or [])
                dois |= {_norm_doi(found.get("doi") or "")} - {""}
                continue
            if doi:
                dois.add(doi)  # list one Zotero copy of a paper, not every attachment
            rest.append((key, m))
    unchecked = []
    for key, m in take(len(queue)):
        doi = _norm_doi(m.get("doi") or "")
        if not (doi and doi in dois):
            if doi:
                dois.add(doi)
            unchecked.append((key, m))
    near = []
    if not indexed and not rest and not unchecked:
        candidates = _near_matches(paper)
        found = _confirm([k for k, _ in candidates], new)
        near = [(k, m, found.get(k)) for k, m in candidates]
    return _ranked(paper, indexed), rest, False, near, unchecked


def _zotero_line(key: str, m: dict) -> str:
    family, year = _first_author_year(key)
    if len(m.get("creators") or m.get("authors") or []) > 1:
        family += " et al."
    doi = f" · doi:{m['doi']}" if m.get("doi") else ""
    return f"- {key} · {family} ({year or 'n.d.'}). {m.get('title') or ''}{doi}"


def _listing(lines: list[str]) -> list[str]:
    if len(lines) <= LIST_ROWS:
        return lines or ["  (none)"]
    return lines[:LIST_ROWS] + [
        f"  … and {len(lines) - LIST_ROWS} more NOT shown. The paper you want may be "
        "among them: add a title word or the author, or pass a Zotero key or DOI. "
        "A cut list does not show absence."
    ]


def _near_text(near) -> list[str]:
    if not near:
        return []
    status = {True: "indexed", None: "Zotero only", "?": "index status unknown"}
    return [
        "\nNo item has every word. Near matches (NOT exact — check before citing):",
        *(
            f"{_zotero_line(k, m)} · {status[True if isinstance(p, dict) else p]}"
            for k, m, p in near
        ),
    ]


def _lookup_text(
    paper: str, indexed, rest, truncated: bool, near=(), unchecked=()
) -> str:
    out = [
        f'Lookup of "{paper}" (a Zotero key, a DOI, or words that must all occur in '
        "the title, authors or year)."
    ]
    zotero = [_zotero_line(k, m) for k, m in rest]
    if indexed is None:
        out += [
            "Index status unknown: this LightRAG server has no paper lookup "
            "(/graph/papers/search).",
            f"\nIn the corpus metadata ({len(zotero)}):",
            *_listing(zotero),
            *_near_text(near),
        ]
        return "\n".join(out)
    rows = [
        f"- {', '.join(p.get('keys') or [])} · {p.get('citation') or p.get('title') or p.get('paper')}"
        + (f" · doi:{p['doi']}" if p.get("doi") else "")
        for p in indexed
    ]
    out += [f"\nIndexed, so its text is searchable ({len(rows)}):", *_listing(rows)]
    if truncated:
        out.append(
            f"\n{PAPER_SEARCH_LIMIT} or more indexed papers match, so the papers that "
            "are only in Zotero are not listed: add title words, or pass a key or DOI."
        )
    else:
        out += [
            f"\nIn Zotero but NOT indexed, so no tool here can read its text ({len(zotero)}):",
            *_listing(zotero),
        ]
        if unchecked:
            out += [
                f"\nIn Zotero, index status NOT checked ({len(unchecked)}): the "
                "server was asked about the items above and not about these, so "
                "any of them may be indexed. Pass a Zotero key or DOI, or add a "
                "title word, to check one:",
                *_listing([_zotero_line(k, m) for k, m in unchecked]),
            ]
        if not rows and not zotero and not unchecked:
            out.append(
                "\nNo indexed paper or Zotero item contains every word (hyphens, "
                "apostrophes, accents and quotes are ignored). That does not show "
                "the paper is missing: a wrong year, a name spelled differently from "
                "the metadata or a topic word that is not in the title gives this "
                "answer too. "
                + (
                    "The near matches below may be the paper itself: check one."
                    if near
                    else "Try 'Author Year' alone."
                    if _norm_doi(paper)
                    else "Try the DOI, the Zotero key, or 'Author Year' alone."
                )
            )
        out += _near_text(near)
    return "\n".join(out)


def _lead_key(keys: list[str]) -> str:
    """The copy of a work that names it: the one with a DOI, then the longest
    title, then the first key in sorted order."""
    return min(
        keys,
        key=lambda k: (
            not _norm_doi(_entry(k).get("doi") or ""),
            -len(str(_entry(k).get("title") or "")),
            k,
        ),
    )


def _query_lead(keys: list[str], lead_key: str, family: str, year: str) -> str:
    """'Family Year' to put before the question, plus up to 3 of the paper's title
    words that the metadata's other works by that first author in that year lack."""
    lead = " ".join(p for p in (family, year) if p)
    meta, rows = _rows()
    mine = rows.get(lead_key)
    if not (family and year and mine):
        return lead
    first = _fold(family)
    others = [
        r
        for k, r in rows.items()
        if k not in keys
        and r["first"] == first
        and r["year"] == year
        and not (mine["doi"] and r["doi"] == mine["doi"])
    ]
    if not others:
        return lead
    taken = {w for r in others for w in r["title_words"].split()}
    extra = []
    for word in re.findall(r"[^\W_]+", str(meta[lead_key].get("title") or "")):
        folded = _fold(word)
        if len(folded) >= 4 and not folded.isdigit() and folded not in taken:
            taken.add(folded)
            extra.append(word)
            if len(extra) == 3:
                break
    return " ".join([lead, *extra])


def _other_line(ref: dict) -> str:
    """'- KEY · Family Year · title' (its first 60 characters) of a document."""
    file_path = ref.get("file_path") or ""
    key = _zotero_key(file_path)
    if not key:
        return f"- {os.path.basename(file_path) or '(untitled source)'}"
    family, year = _first_author_year(key, ref.get("citation") or "")
    title = (
        str(_entry(key).get("title") or "")
        or re.sub(
            r"_+", " ", re.sub(r"\.md$", "", os.path.basename(file_path))[10:]
        ).strip()
    )
    if len(title) > 60:
        title = title[:60] + "…"
    who = " ".join(p for p in (family, year) if p)
    return " · ".join(p for p in (f"- {key}", who, title) if p)


def _post_or_absent(path: str, payload: dict) -> dict | None:
    """_post, but None when the server has no such route (it answers 404)."""
    try:
        r = httpx.post(
            f"{BASE}{path}", json=payload, headers=HEADERS, timeout=QUERY_TIMEOUT
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        raise _tool_error(path, e) from e
    except ValueError as e:
        raise ToolError(f"LightRAG {path} returned no JSON: {e}") from e
    return data if isinstance(data, dict) else {}


_NO_PASSAGE = (
    "No passage of this paper matched the question. Ask again with a sentence in "
    'the paper\'s own terms, or call search_library(mode="naive") with the claim '
    "sentence alone: it searches every paper and can reach a sentence this misses."
)


def _local_file(key: str) -> str:
    """The paper's attachment in this host's Zotero storage, '' when it has none."""
    name = _entry(key).get("filename") or ""
    path = os.path.join(STORAGE, key, name) if key and name else ""
    return path if path and os.path.isfile(path) else ""


def _paper_passages_text(lead_key: str, citation: str, data: dict) -> str:
    """One paper's passages from POST /graph/papers/passages, under its citation
    and the path of its local file (for what the index lacks: equations)."""
    out = [f"### {_cite(f'{lead_key}__', citation)}"]
    for key in [lead_key] + [k for k in data.get("keys") or [] if k != lead_key]:
        if path := _local_file(key):
            out[0] += f"\nLocal file: {path}"
            break
    n_chunks = data.get("n_chunks")
    blocks = []
    for i, p in enumerate(data.get("passages") or [], start=1):
        if not isinstance(p, dict) or not isinstance(p.get("content"), str):
            continue
        file_path = p.get("file_path") or f"{p.get('key') or ''}__"
        head = [f"[{i}] {_label({'file_path': file_path})}"]
        key = _zotero_key(file_path) or lead_key
        if section := _section(p.get("content_headings"), key, citation):
            head.append(section)
        if isinstance(p.get("chunk_order_index"), int):
            head.append(
                f"chunk {p['chunk_order_index'] + 1}"
                + (f" of {n_chunks}" if n_chunks else "")
            )
        blocks.append(" · ".join(head) + "\n" + _strip_header(p["content"]).strip())
    if blocks:
        out += _fit(blocks, MAX_CHARS - 3000)
    elif data.get("keys") == []:
        out.append(
            "The server holds no indexed copy of this paper. Call find_paper to "
            "check its keys."
        )
    else:
        out.append(_NO_PASSAGE)
    return "\n\n".join(out)


@mcp.tool(structured_output=False)
def search_library(
    query: str,
    mode: str = "mix",
    top_k: int = 10,
    answer: bool = False,
    include_graph: bool = False,
    passages: int = DEFAULT_PASSAGES,
) -> str:
    """Search the user's research library (Zotero papers indexed in LightRAG).

    Default (answer=False): returns the RETRIEVED CONTEXT, with no LLM synthesis:
    compiled citations [n], then the reranked source passages, each headed by its
    [n], first author and year, and section. Synthesize from it yourself: the
    passages are the primary source, and skipping answer generation makes this
    considerably faster. Text that two Zotero copies of a paper share is shown
    once, headed by both numbers ("[1][4]"). Past the proxy's size cap the
    lowest-ranked passages are left out, and the result says so.
    include_graph=True appends the knowledge-graph entities and relations, mainly
    useful for inspecting the index itself. answer=True: additionally has the
    LightRAG server's LLM write a synthesized answer; its own reference list is
    replaced by compiled citations. Slower; use it only when self-contained prose
    is required.

    Reading the passages: rest a claim about a paper on a passage that states it,
    and quote that sentence with its [n]. A title block or abstract that lists a
    paper's error sources, test cases or methods is not its finding. Passages of
    several papers on one topic sit side by side, so check the [n] before you
    attribute a sentence. To check what one paper says, use paper_passages; to
    learn whether a paper is in the library, use find_paper, because a ranked
    search cannot show that a paper is absent. Display equations are not in the
    index (see paper_passages).

    passages: how many passages to retrieve, default 8, 1 to 12. Each is about
    4 KB, so the default result is about 30 KB. top_k does not change the result
    size; passages does.
    mode: 'mix' (default, KG + vectors — best), 'hybrid', 'local' (entity-centric),
          'global' (theme/relationship-centric), or 'naive' (plain vector search).
          Any other value falls back to 'mix'.
    top_k: the knowledge-graph entities and relations retrieved, default 10,
    clamped to the proxy's configured maximum. Larger values fill the server's
    token budget with them and can starve document-chunk retrieval, leaving few or
    no chunks and no citations.
    Parallel calls share the server's LLM and reranker concurrency limits; beyond
    them they queue and can time out.
    Use this for questions about the *content* of the papers (methods, findings,
    who studied what, how concepts relate)."""
    mode = mode if mode in VALID_MODES else "mix"
    top_k = _top_k(top_k)
    if not answer:
        passages = max(1, min(int(passages), MAX_PASSAGES))
        return _context(query, mode, top_k, passages, include_graph)
    data = _post(
        "/query",
        {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "user_prompt": "Do not generate a References section; "
            "references are appended programmatically.",
        },
    )
    answer = _strip_llm_refs(data.get("response") or str(data))
    refs = data.get("references") or []
    key = (query, mode, top_k)
    if refs:
        _REF_MEMO[key] = refs
    else:
        # Cached answers (on some LightRAG versions) and chunk-starved retrievals
        # return no references, and /query/data re-runs retrieval to rebuild them.
        # Retrieval is often the expensive half of a query (reranking included),
        # so reuse this session's references for an identical query, and cap the
        # fallback so it can never double the worst case.
        refs = _REF_MEMO.get(key) or []
        if not refs:
            try:
                d2 = _post(
                    "/query/data",
                    {"query": query, "mode": mode, "top_k": top_k},
                    timeout=REF_FALLBACK_TIMEOUT,
                )
                refs = (d2.get("data") or {}).get("references") or []
                if refs:
                    _REF_MEMO[key] = refs
            except Exception:
                refs = []
    return answer + _compiled_refs(refs, "Sources (compiled from retrieval)")


@mcp.tool(structured_output=False)
def paper_passages(paper: str, question: str) -> str:
    """Passages of ONE paper that bear on a question: use it to check what a cited
    paper actually says.

    paper: the paper's Zotero key, DOI, or words that all occur in its title,
    authors or year (e.g. "Vaswani 2017 attention"). question: what to find in it;
    to check a claim, write the sentence you expect the paper to contain, in the
    paper's own terms.

    Returns the paper's citation and up to 3 of its passages ranked against the
    question; the opening chunk (title block and abstract) competes like any
    other. Re-asking with a paraphrase usually returns the same chunks: to reach
    another part of the paper, name a section or a distinctive term from it. If no
    passage states the claim, search_library(mode="naive") with the claim sentence
    alone searches every paper. When the paper is not indexed, or several papers or
    only near matches fit `paper`, it lists them instead of searching: call it again
    with one key or DOI. Quote the sentence you rely on; an abstract's list of
    methods or error sources is not the paper's finding.

    Not in the index: display equations. The corpus builder's text-layer converter
    drops them from born-digital PDFs (scans read by the OCR model keep them as
    LaTeX), so only references to them ("eq. (5)") remain, and tables often arrive
    scrambled. Asking again cannot reach them. For an equation's coefficients or a
    table's values, read the file named under the citation ("Local file:", when
    this host has the Zotero storage folder): `pdftotext -layout FILE out.txt`,
    where an equation's number ends its line, so
    `grep -n -B8 -A2 '(6)\\s*$' out.txt` shows equation (6)."""
    indexed, rest, truncated, near, unchecked = _lookup(paper)
    if indexed is None:
        if len(rest) != 1:
            return _lookup_text(paper, None, rest, False, near) + (
                "\n\npaper_passages needs exactly one paper: call it again with one "
                "Zotero key or DOI from the list."
                if rest or near
                else ""
            )
        keys = [rest[0][0]]
    elif len(indexed) != 1:
        return _lookup_text(paper, indexed, rest, truncated, near, unchecked) + (
            "\n\npaper_passages needs exactly one indexed paper: call it again with "
            "one Zotero key or DOI from the list."
            if indexed or near or unchecked
            else ""
        )
    else:
        keys = list(indexed[0].get("keys") or [])
    lead_key = _lead_key(keys)
    server_citation = indexed[0].get("citation") or "" if indexed else ""
    if indexed:
        data = _post_or_absent(
            "/graph/papers/passages",
            {
                "keys": [lead_key] + [k for k in keys if k != lead_key][:9],
                "question": question,
                "top_n": 3,
            },
        )
        if data is not None:
            citation = "" if _entry(lead_key) else server_citation
            return _paper_passages_text(lead_key, citation, data)
    # A server without the passages route: a /query that names the paper by its
    # first author and year, keeping only this paper's chunks.
    family, year = _first_author_year(lead_key, server_citation)
    lead = _query_lead(keys, lead_key, family, year)
    data = _post(
        "/query",
        {
            "query": f"{lead}: {question}" if lead else question,
            "mode": "naive",
            "top_k": 10,
            "only_need_context": True,
        },
    )
    ctx = data.get("response") or data.get("context") or str(data)
    references = data.get("references") or []
    mine_ids, citation = set(), ""
    for r in references:
        if _zotero_key(r.get("file_path", "")) in keys:
            mine_ids.add(str(r.get("reference_id")))
            citation = citation or r.get("citation") or ""
    labels = {str(r.get("reference_id")): _label(r) for r in references}
    mine = [c for c in _chunks(ctx) if str(c.get("reference_id")) in mine_ids]
    out = [f"### {_cite(f'{lead_key}__', citation)}"]
    if mine:
        out.append(
            "Its passages that the question reached, best first (name a section or a "
            "distinctive term to reach others):"
        )
        out += _fit(_passages(mine, labels, _sources(references)), MAX_CHARS - 3000)
    else:
        out.append(
            "No passage of this paper ranked for the question. Rephrase it in the "
            "paper's own terms, or write the sentence you expect it to contain."
        )
    others = [r for r in references if str(r.get("reference_id")) not in mine_ids]
    if others:
        out.append(
            "### Other papers the question matched (no passages shown)\n"
            + "\n".join(_other_line(r) for r in others[:5])
        )
    return "\n\n".join(out)


@mcp.tool(structured_output=False)
def find_paper(paper: str) -> str:
    """Lookup: is a paper in the knowledge base, only in Zotero, or in neither?

    paper: a Zotero key, a DOI, or words that must all occur in the paper's title,
    authors or year (e.g. "Vaswani 2017 attention"). Hyphens, apostrophes,
    accents, quote marks and the "et al" after a name are ignored, and a BibTeX key
    (vaswaniAttentionAllYou2017) is split into its words. Lists the indexed
    papers that match (their text is searchable), then the Zotero items that
    match but are NOT indexed (no tool here can read them). When no item has
    every word, it lists near matches (the year off by one, or the same author)
    marked as such: check one before citing it. A long list is cut, and a cut
    list does not show absence. An empty answer means only that no item has
    every word; a wrong year or a topic word missing from the title gives it
    too, so it does not show the paper is missing. About 1-3 s alone, longer
    while searches run; no retrieval and no LLM."""
    return _lookup_text(paper, *_lookup(paper))


@mcp.tool(structured_output=False)
def add_note(text: str, description: str = "") -> str:
    """Insert a text note/snippet into the knowledge base (it will be chunked,
    embedded, and folded into the knowledge graph on the next processing cycle).

    LightRAG names each document after its source and refuses a name it already
    holds, so every note gets a unique name: note-<UTC time>-<hash of the text>.
    An optional short description goes in front of it, as in
    "reading notes (note-20260101T120000Z-1a2b3c4d)"; citations show that name."""
    name = (
        f"note-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
        f"{hashlib.sha1(text.encode()).hexdigest()[:8]}"
    )
    # LightRAG keeps only the last path segment of a source and refuses control
    # characters, so path separators and control characters become spaces.
    label = re.sub(r"[\s/\\\x00-\x1f\x7f]+", " ", description).strip()
    data = _post(
        "/documents/text",
        {"text": text, "file_source": f"{label} ({name})" if label else name},
    )
    return str(data)


@mcp.tool(structured_output=False)
def health() -> str:
    """Check that the LightRAG server is reachable and report its status."""
    try:
        r = httpx.get(f"{BASE}/health", headers=HEADERS, timeout=HEALTH_TIMEOUT)
        return f"{r.status_code}: {r.text}"
    except Exception as e:
        return f"unreachable at {BASE}: {e}"


if __name__ == "__main__":
    mcp.run()
