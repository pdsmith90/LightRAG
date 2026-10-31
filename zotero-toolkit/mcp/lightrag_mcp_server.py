#!/usr/bin/env python3
"""
lightrag_mcp_server.py — expose a Zotero-backed LightRAG knowledge base to any
MCP client (Claude Code, Claude Desktop, other agents) over stdio.

It's a thin, self-contained proxy over the LightRAG Server REST API. Five tools:

    search_library(query, mode, top_k, answer=False, include_graph=False)
                                       -> compiled citations, then the retrieved
                                          passages (answer=True: the server's
                                          synthesized answer, with citations)
    paper_passages(paper, question)    -> the passages of ONE paper that answer a
                                          question, then the other papers it matched
    find_paper(paper)                  -> exact lookup: is a paper indexed, only in
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
    LIGHTRAG_MCP_TIMEOUT               seconds per query, default 300
    LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT  seconds for the /query/data fallback, default 90
    LIGHTRAG_MCP_HEALTH_TIMEOUT        seconds for health(), default 15
    LIGHTRAG_MCP_MAX_TOP_K             upper bound applied to top_k, default 40
    LIGHTRAG_MCP_MAX_CHARS             longest retrieved-context result; lower-ranked
                                       passages are left out beyond it, default
                                       48000 (Claude Code saves a tool result over
                                       50,000 characters to a file instead)

find_paper and paper_passages ask the server's GET /graph/papers/search (the
zlightrag fork's paper lookup) which papers are indexed. A stock LightRAG Server
has no such route; both then fall back to zotero_metadata.json alone.

Run:  python3 lightrag_mcp_server.py     (normally launched by the MCP client)
Deps: pip install -r requirements.txt    (mcp 2.x, httpx)
"""

from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import time
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

# Zotero metadata for deterministic citations. Answering LLMs fabricate the
# "### References" block (placeholder titles, wrong years), so we strip that and
# compile real citations from the server's structured `references` array
# (chunk file_paths are "<ZoteroKey>__<slug>.md").
META_PATH = os.path.expanduser(
    os.environ.get("ZOTERO_METADATA") or "~/Zotero/zotero_metadata.json"
)
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


def _first_author_year(key: str, server_citation: str = "") -> tuple[str, str]:
    """(first author's family name, year) of a corpus document, for passage
    labels and query leads: structured creators first, then the server's
    citation ("Family, Given; ... (Year)."), then the plain author string."""
    m = _metadata().get(key) if key else None
    m = m if isinstance(m, dict) else {}
    family = ""
    creators = m.get("creators") or []
    if creators and isinstance(creators[0], (list, tuple)) and creators[0]:
        family = str(creators[0][0] or "").strip()
    if not family and (c := re.match(r"\s*([^,;.(]+)", server_citation or "")):
        family = c.group(1).strip()
    if not family and m.get("authors"):
        family = str(m["authors"][0]).strip()
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


def _passage(chunk: dict, labels: dict) -> str:
    rid = str(chunk.get("reference_id", "?"))
    head = f"[{rid}] {labels.get(rid, '')}".rstrip()
    if chunk.get("content_headings"):
        head += f" · {chunk['content_headings']}"
    return f"{head}\n{chunk['content'].strip()}"


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
    shown = _fit([_passage(c, labels) for c in chunks], budget)
    out = f"{head}\n\n" + "\n\n".join(shown)
    if left := len(chunks) - len(shown):
        out += (
            f"\n\n[{left} lower-ranked passage(s) left out to keep this result under "
            f"{MAX_CHARS} characters. Ask a narrower question, or use paper_passages "
            "for one paper.]"
        )
    return f"{out}\n\n{graph}" if graph else out


def _context(query: str, mode: str, top_k: int, include_graph: bool = False) -> str:
    """Retrieval only — no LLM answer."""
    data = _post(
        "/query",
        {"query": query, "mode": mode, "top_k": top_k, "only_need_context": True},
    )
    return _render_context(data, include_graph)


def _indexed_papers(q: str) -> list[dict] | None:
    """Indexed papers matching q, best match first, from the server's paper lookup;
    None when the server has no such route (a stock LightRAG Server)."""
    path = "/graph/papers/search"
    try:
        r = httpx.get(
            f"{BASE}{path}",
            params={"q": q, "limit": PAPER_SEARCH_LIMIT},
            headers=HEADERS,
            timeout=QUERY_TIMEOUT,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        papers = r.json()
    except httpx.HTTPError as e:
        raise _tool_error(path, e) from e
    except ValueError as e:
        raise ToolError(f"LightRAG {path} returned no JSON: {e}") from e
    return (
        [p for p in papers if isinstance(p, dict)] if isinstance(papers, list) else []
    )


def _metadata_matches(q: str) -> list[tuple[str, dict]]:
    """Corpus metadata entries matching q the way the server's paper lookup does:
    an exact Zotero key or DOI, else every word of q in the title, authors, year,
    DOI or key; most title words first, then the newest."""
    key_q, doi_q, words = q.strip().upper(), _norm_doi(q), q.lower().split()
    exact, loose = [], []
    for key, m in _metadata().items():
        if not isinstance(m, dict):
            continue
        doi = _norm_doi(m.get("doi") or "")
        if key == key_q or (doi_q and doi == doi_q):
            exact.append((key, m))
            continue
        title = str(m.get("title") or "").lower()
        authors = " ".join(str(a) for a in m.get("authors") or []).lower()
        hay = " ".join([title, authors, str(m.get("year") or ""), doi, key.lower()])
        if words and all(w in hay for w in words):
            loose.append((key, m))
    if exact:
        return exact

    def rank(item):
        title, year = (
            str(item[1].get("title") or "").lower(),
            str(item[1].get("year") or ""),
        )
        return -sum(w in title for w in words), -int(year) if year.isdigit() else 0

    return sorted(loose, key=rank)


def _lookup(paper: str) -> tuple[list[dict] | None, list[tuple[str, dict]], bool]:
    """(indexed papers, or None when the server cannot say; Zotero items that match
    but are not indexed, one per DOI; whether the server cut its list at its limit).

    An item is reported as not indexed only after the server, asked for its exact
    key, does not know it either."""
    indexed = _indexed_papers(paper)
    local = _metadata_matches(paper)
    if indexed is None:
        return None, local, False
    if len(indexed) >= PAPER_SEARCH_LIMIT:
        return indexed, [], True
    keys = {k for p in indexed for k in p.get("keys") or []}
    dois = {_norm_doi(p.get("doi") or "") for p in indexed} - {""}
    rest = []
    for key, m in local:
        doi = _norm_doi(m.get("doi") or "")
        if key in keys or (doi and doi in dois):
            continue
        if len(rest) < 10:
            found = [
                p for p in _indexed_papers(key) or [] if key in (p.get("keys") or [])
            ]
            if found:
                indexed += found
                keys |= {k for p in found for k in p.get("keys") or []}
                dois |= {_norm_doi(p.get("doi") or "") for p in found} - {""}
                continue
        if doi:
            dois.add(doi)  # list one Zotero copy of a paper, not every attachment
        rest.append((key, m))
    return indexed, rest, False


def _zotero_line(key: str, m: dict) -> str:
    family, year = _first_author_year(key)
    if len(m.get("creators") or m.get("authors") or []) > 1:
        family += " et al."
    doi = f" · doi:{m['doi']}" if m.get("doi") else ""
    return f"- {key} · {family} ({year or 'n.d.'}). {m.get('title') or ''}{doi}"


def _lookup_text(paper: str, indexed, rest, truncated: bool) -> str:
    def listing(lines: list[str]) -> list[str]:
        more = [f"  … and {len(lines) - 10} more"] if len(lines) > 10 else []
        return (lines[:10] + more) or ["  (none)"]

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
            *listing(zotero),
        ]
        return "\n".join(out)
    rows = [
        f"- {', '.join(p.get('keys') or [])} · {p.get('citation') or p.get('title') or p.get('paper')}"
        + (f" · doi:{p['doi']}" if p.get("doi") else "")
        for p in indexed
    ]
    out += [f"\nIndexed, so its text is searchable ({len(rows)}):", *listing(rows)]
    if truncated:
        out.append(
            f"\n{PAPER_SEARCH_LIMIT} or more indexed papers match, so the papers that "
            "are only in Zotero are not listed: add title words, or pass a key or DOI."
        )
    else:
        out += [
            f"\nIn Zotero but NOT indexed, so no tool here can read its text ({len(zotero)}):",
            *listing(zotero),
        ]
        if not rows and not zotero:
            out.append(
                "\nNeither the index nor the corpus metadata (the Zotero items with an "
                "attached file) has a match. Check the spelling, use fewer words, or "
                "pass the DOI."
            )
    return "\n".join(out)


@mcp.tool()
def search_library(
    query: str,
    mode: str = "mix",
    top_k: int = 10,
    answer: bool = False,
    include_graph: bool = False,
) -> str:
    """Search the user's research library (Zotero papers indexed in LightRAG).

    Default (answer=False): returns the RETRIEVED CONTEXT, with no LLM synthesis:
    compiled citations [n], then the reranked source passages, each headed by its
    [n], first author and year, and section. Synthesize from it yourself: the
    passages are the primary source, and skipping answer generation makes this
    considerably faster. Past the proxy's size cap the lowest-ranked passages are
    left out, and the result says so. include_graph=True appends the knowledge-graph
    entities and relations, mainly useful for inspecting the index itself.
    answer=True: additionally has the LightRAG server's LLM write a synthesized
    answer; its own reference list is replaced by compiled citations. Slower; use
    it only when self-contained prose is required.

    Reading the passages: rest a claim about a paper on a passage that states it,
    and quote that sentence with its [n]. A title block or abstract that lists a
    paper's error sources, test cases or methods is not its finding. Passages of
    several papers on one topic sit side by side, so check the [n] before you
    attribute a sentence. To check what one paper says, use paper_passages; to
    learn whether a paper is in the library, use find_paper, because a ranked
    search cannot show that a paper is absent.

    mode: 'mix' (default, KG + vectors — best), 'hybrid', 'local' (entity-centric),
          'global' (theme/relationship-centric), or 'naive' (plain vector search).
          Any other value falls back to 'mix'.
    top_k: default 10, clamped to the proxy's configured maximum. Larger values
    fill the server's token budget with KG entities and relations and can starve
    document-chunk retrieval, leaving few or no chunks and no citations.
    Parallel calls share the server's LLM and reranker concurrency limits; beyond
    them they queue and can time out.
    Use this for questions about the *content* of the papers (methods, findings,
    who studied what, how concepts relate)."""
    mode = mode if mode in VALID_MODES else "mix"
    top_k = _top_k(top_k)
    if not answer:
        return _context(query, mode, top_k, include_graph)
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


@mcp.tool()
def paper_passages(paper: str, question: str) -> str:
    """Passages of ONE paper that bear on a question: use it to check what a cited
    paper actually says.

    paper: the paper's Zotero key, DOI, or words that all occur in its title,
    authors or year (e.g. "Vaswani 2017 attention"). question: what to find in it;
    to check a claim, write the sentence you expect the paper to contain, in the
    paper's own terms.

    Returns the paper's citation and its passages that best match the question
    (a server may return only a few passages of one paper per query: ask about
    another part of the paper to reach others), then the other papers the
    question matched, which helps when a statement really comes from a different
    paper. When the paper is not indexed, or several papers match `paper`, it says
    so and lists them instead of searching: call it again with one key or DOI.
    Quote the sentence you rely on; an abstract's list of methods or error sources
    is not the paper's finding."""
    indexed, rest, truncated = _lookup(paper)
    if indexed is None:
        if len(rest) != 1:
            return _lookup_text(paper, None, rest, False) + (
                "\n\npaper_passages needs exactly one paper: call it again with one "
                "Zotero key or DOI from the list."
                if rest
                else ""
            )
        keys, server_citation = {rest[0][0]}, ""
    elif len(indexed) != 1:
        return _lookup_text(paper, indexed, rest, truncated) + (
            "\n\npaper_passages needs exactly one indexed paper: call it again with "
            "one Zotero key or DOI from the list."
            if indexed
            else ""
        )
    else:
        keys, server_citation = set(indexed[0].get("keys") or []), ""
    lead_key = sorted(keys)[0]
    family, year = _first_author_year(
        lead_key, indexed[0].get("citation") or "" if indexed else ""
    )
    title = str((_metadata().get(lead_key) or {}).get("title") or "")
    if indexed:
        title = title or indexed[0].get("title") or ""
    # Naming the paper (author, year, title) lets the server's metadata and
    # named-work legs bring its chunks forward.
    lead = " ".join(p for p in (family, year, title) if p)
    data = _post(
        "/query",
        {
            "query": f"{lead}: {question}",
            "mode": "naive",
            "top_k": 10,
            "only_need_context": True,
        },
    )
    ctx = data.get("response") or data.get("context") or str(data)
    references = data.get("references") or []
    mine_ids = set()
    for r in references:
        if _zotero_key(r.get("file_path", "")) in keys:
            mine_ids.add(str(r.get("reference_id")))
            server_citation = server_citation or r.get("citation") or ""
    labels = {str(r.get("reference_id")): _label(r) for r in references}
    mine = [c for c in _chunks(ctx) if str(c.get("reference_id")) in mine_ids]
    out = [f"### {_cite(f'{lead_key}__', server_citation)}"]
    if mine:
        out.append(
            "Its passages that the question reached, best first (ask about another "
            "part of the paper to reach others):"
        )
        out += _fit([_passage(c, labels) for c in mine], MAX_CHARS - 3000)
    else:
        out.append(
            "No passage of this paper ranked for the question. Rephrase it in the "
            "paper's own terms, or write the sentence you expect it to contain."
        )
    others = [r for r in references if str(r.get("reference_id")) not in mine_ids]
    return "\n\n".join(out) + _compiled_refs(
        others, "Other papers the question matched (no passages shown)"
    )


@mcp.tool()
def find_paper(paper: str) -> str:
    """Exact lookup: is a paper in the knowledge base, only in Zotero, or in neither?

    paper: a Zotero key, a DOI, or words that must all occur in the paper's title,
    authors or year (e.g. "Vaswani 2017 attention"). Lists the indexed papers that
    match (their text is searchable), then the Zotero items that match but are NOT
    indexed (no tool here can read them). Unlike search_library, an empty answer
    here does show absence: from the index, and from the corpus metadata (the
    Zotero items with an attached file). Fast: no retrieval and no LLM."""
    return _lookup_text(paper, *_lookup(paper))


@mcp.tool()
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


@mcp.tool()
def health() -> str:
    """Check that the LightRAG server is reachable and report its status."""
    try:
        r = httpx.get(f"{BASE}/health", headers=HEADERS, timeout=HEALTH_TIMEOUT)
        return f"{r.status_code}: {r.text}"
    except Exception as e:
        return f"unreachable at {BASE}: {e}"


if __name__ == "__main__":
    mcp.run()
