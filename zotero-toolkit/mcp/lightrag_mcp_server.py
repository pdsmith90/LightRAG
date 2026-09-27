#!/usr/bin/env python3
"""
lightrag_mcp_server.py — expose a Zotero-backed LightRAG knowledge base to any
MCP client (Claude Code, Claude Desktop, other agents) over stdio.

It's a thin, self-contained proxy over the LightRAG Server REST API. Four tools:

    search_library(query, mode, top_k, answer=False)
                                       -> compiled citations, then the retrieved
                                          context (answer=True: the server's
                                          synthesized answer, with citations)
    get_sources(query, mode, top_k)    -> the raw retrieved context (entities/
                                           relations/chunks) for citation/inspection
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

# Zotero metadata for deterministic citations. Answering LLMs fabricate the
# "### References" block (placeholder titles, wrong years), so we strip that and
# compile real citations from the server's structured `references` array
# (chunk file_paths are "<ZoteroKey>__<slug>.md").
META_PATH = os.path.expanduser(
    os.environ.get("ZOTERO_METADATA") or "~/Zotero/zotero_metadata.json"
)
try:
    with open(META_PATH, encoding="utf-8") as _f:
        _META = json.load(_f)
except Exception as _e:
    _META = {}
    # stderr, never stdout: stdout carries the MCP protocol.
    print(
        f"lightrag_mcp_server: no citation metadata loaded from "
        f"{os.path.abspath(META_PATH)} ({type(_e).__name__}); citations fall "
        f"back to file names",
        file=sys.stderr,
    )

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


def _cite(file_path: str) -> str:
    name = os.path.basename(file_path or "")
    if not re.match(r"^[A-Z0-9]{8}__", name):
        return name or "(untitled source)"  # notes / non-corpus sources
    key = name[:8]
    m = _META.get(key)
    if m:
        authors = ", ".join(m.get("authors") or []) or None
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
        f"- [{r.get('reference_id', '?')}] {_cite(r.get('file_path', ''))}"
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
        # mcp >= 2.1 shows the client only "Error executing tool <name>" for an
        # unexpected exception; a ToolError's message reaches it on every 2.x.
        msg = f"LightRAG {path} failed: {e}"
        # The status line does not say why; LightRAG's body does (a 409 names the
        # document that already exists, or the scan or deletion blocking inserts).
        if isinstance(e, httpx.HTTPStatusError) and (
            detail := _error_detail(e.response)
        ):
            msg += f"\nServer response: {detail}"
        raise ToolError(msg) from e
    try:
        return r.json()
    except Exception:
        return {"response": r.text}


def _context(query: str, mode: str, top_k: int) -> str:
    """Retrieval only — no LLM answer. Compiled citations FIRST, then the reranked
    entities/relations/passages, so a client that truncates long tool results
    keeps the citations."""
    data = _post(
        "/query",
        {"query": query, "mode": mode, "top_k": top_k, "only_need_context": True},
    )
    ctx = data.get("response") or data.get("context") or str(data)
    refs = _compiled_refs(
        data.get("references") or [], "Retrieved documents (compiled citations)"
    ).strip()
    return f"{refs}\n\n{ctx}" if refs else ctx


@mcp.tool()
def search_library(
    query: str, mode: str = "mix", top_k: int = 10, answer: bool = False
) -> str:
    """Search the user's research library (Zotero papers indexed in LightRAG).

    Default (answer=False): returns the RETRIEVED CONTEXT — compiled citations, then
    the reranked entities, relations and source passages — with no LLM synthesis.
    Synthesize from it yourself: the passages are the primary source, and skipping
    answer generation makes this considerably faster.
    answer=True: additionally has the LightRAG server's LLM write a synthesized
    answer; its own reference list is replaced by compiled citations. Slower; use
    it only when self-contained prose is required.

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
        return _context(query, mode, top_k)
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
def get_sources(query: str, mode: str = "mix", top_k: int = 10) -> str:
    """Return the raw retrieved context for a query (entities, relations, and source
    chunks with their document titles / Zotero keys) WITHOUT a synthesized answer.
    Use this when you need to cite specific papers or inspect what the KB retrieved."""
    mode = mode if mode in VALID_MODES else "mix"
    top_k = _top_k(top_k)
    data = _post(
        "/query",
        {"query": query, "mode": mode, "top_k": top_k, "only_need_context": True},
    )
    ctx = data.get("response") or data.get("context") or str(data)
    return ctx + _compiled_refs(
        data.get("references") or [], "Retrieved documents (compiled citations)"
    )


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
