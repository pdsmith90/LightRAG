"""The library's citation structure: related papers, papers by topic, what
connects two papers, and a paper map for the graph view.

The structure is a PostgreSQL schema built outside this repository (papers,
attachment keys, authors, venues, a four-level topic hierarchy, citation edges
between library papers and bibliographic-coupling pairs; ZLIGHTRAG.md lists the
tables). ZOTERO_STRUCT_SCHEMA names it; unset, every entry point raises
:class:`Unavailable` and nothing reads the database. It is read through the
document-status storage's PostgreSQL pool, so it needs PGDocStatusStorage and
does not depend on the graph storage.

Rules for callers:

- The schema is read whole into memory on first use (about 10,000 papers and
  170,000 links load in a few seconds) and read again when a rebuild is seen:
  at most every _CHECK_INTERVAL seconds the tables' OIDs and file nodes are
  compared, and a drop-and-reload or TRUNCATE changes them. An in-place UPDATE
  is not seen until the server restarts; the layer is rebuilt, never edited.
- Which papers are in the knowledge base comes from the document-status table:
  processed documents of the server's workspace, mapped to attachment keys with
  :func:`lightrag.zotero_citations.zotero_key`, read at most every _DOCS_TTL
  seconds. A view the schema may carry for that join is not used.
- A paper is named by an attachment key, a DOI, or a schema paper id
  (``doi:<doi>`` / ``zot:<item key>``). Request text is only ever compared with
  loaded values or bound as a parameter; the schema name, the one identifier in
  the SQL, is checked against _IDENT_RE.

Fork-local (no upstream counterpart).
"""

from __future__ import annotations

import asyncio
import heapq
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

from lightrag import zotero_citations
from lightrag.base import DocStatus
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import logger
from lightrag.zotero_papers import MAX_PAPERS, _KEY_RE, fold, norm_doi

SCHEMA_ENV = "ZOTERO_STRUCT_SCHEMA"
# An unquoted lower-case PostgreSQL identifier.
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_TABLES = (
    "papers",
    "venues",
    "paper_attachments",
    "authors",
    "paper_authors",
    "topics",
    "paper_topics",
    "cites",
    "coupled",
)
_CHECK_INTERVAL = 60.0
_DOCS_TTL = 300.0
# how long /health waits for the catalog check before answering "not available"
_AVAILABLE_TIMEOUT = 3.0

RELATIONS = ("cites", "cited_by", "coupled", "coauthor")
LEVELS = ("domain", "field", "subfield", "topic")
MAX_LIMIT = 200
MAX_PATHS = 5
MAX_HOPS = 8

# related_papers: score = _W_CITE for a citation either way (+ _W_BOTH when it
# runs both ways) + _W_COUPLED * coupling score + _W_AUTHOR per shared author
# (at most _AUTHOR_CAP of them).
_W_CITE = 1.0
_W_BOTH = 0.25
_W_COUPLED = 2.0
_W_AUTHOR = 0.25
_AUTHOR_CAP = 4
# connect_papers: entering a paper that is not the target costs
# 1 + _HUB_PENALTY * log2(1 + its links), so a path through a paper with 600
# links costs about as much as two more ordinary steps.
_HUB_PENALTY = 0.5
# paper_map: a two-hop paper reached through u gains strength(u) * w(u, v)
# / log2(2 + links of u). Links shown: every link of a chosen paper, the
# _EDGES_PER_PAPER strongest of every other shown paper, and at most
# _EDGE_CAP_FACTOR x the node limit in all.
_AUTHORS_SHOWN = 10
_EDGES_PER_PAPER = 6
_EDGE_CAP_FACTOR = 4


class Unavailable(Exception):
    """The structure cannot be read; ``reason`` is one of ``off``,
    ``bad_schema_name``, ``not_postgres``, ``no_schema``."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


_SQL = {
    "fingerprint": """-- zotero_struct:fingerprint
SELECT c.relname, c.oid::bigint AS oid, pg_relation_filenode(c.oid)::bigint AS filenode
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind = 'r'""",
    "papers": """-- zotero_struct:papers
SELECT p.paper_id, p.doi, p.title, p.year, p.cited_by_count, p.n_refs_openalex,
       v.display_name AS venue
FROM "{s}".papers p LEFT JOIN "{s}".venues v ON v.venue_id = p.venue_id""",
    "attachments": """-- zotero_struct:attachments
SELECT attachment_key, paper_id FROM "{s}".paper_attachments WHERE paper_id IS NOT NULL""",
    "authors": """-- zotero_struct:authors
SELECT pa.paper_id, pa.author_id, a.display_name
FROM "{s}".paper_authors pa LEFT JOIN "{s}".authors a ON a.author_id = pa.author_id
WHERE mod(hashtext(pa.paper_id) & 2147483647, $1) = $2
ORDER BY pa.paper_id, pa.position""",
    "topics": """-- zotero_struct:topics
SELECT topic_id, display_name, level, parent_id FROM "{s}".topics""",
    "paper_topics": """-- zotero_struct:paper_topics
SELECT paper_id, topic_id, is_primary FROM "{s}".paper_topics
WHERE mod(hashtext(paper_id) & 2147483647, $1) = $2""",
    "cites": """-- zotero_struct:cites
SELECT citing_paper_id, cited_paper_id FROM "{s}".cites
WHERE mod(hashtext(citing_paper_id) & 2147483647, $1) = $2""",
    "coupled": """-- zotero_struct:coupled
SELECT paper_a, paper_b, shared_refs, score FROM "{s}".coupled
WHERE mod(hashtext(paper_a) & 2147483647, $1) = $2""",
    # The server's own workspace and processed documents only.
    "documents": """-- zotero_struct:documents
SELECT file_path FROM LIGHTRAG_DOC_STATUS WHERE workspace = $1 AND status = $2""",
}


@dataclass
class _Struct:
    """The schema in memory; papers are numbered 0..n-1. Read-only once built."""

    ids: list[str] = field(default_factory=list)
    doi: list[str] = field(default_factory=list)
    title: list[str] = field(default_factory=list)
    year: list[int | None] = field(default_factory=list)
    venue: list[str] = field(default_factory=list)
    cited_by_count: list[int | None] = field(default_factory=list)
    n_refs: list[int | None] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)
    by_doi: dict[str, int] = field(default_factory=dict)
    by_key: dict[str, int] = field(default_factory=dict)
    keys: list[list[str]] = field(default_factory=list)
    authors: list[list[tuple[str, str]]] = field(default_factory=list)
    papers_of_author: dict[str, list[int]] = field(default_factory=dict)
    cites_out: list[set[int]] = field(default_factory=list)
    cites_in: list[set[int]] = field(default_factory=list)
    coupled: list[dict[int, tuple[int, float]]] = field(default_factory=list)
    adj: list[set[int]] = field(default_factory=list)
    # topic id -> (name, level, parent id)
    topics: dict[str, tuple[str, str, str | None]] = field(default_factory=dict)
    # paper -> its primary topic id ('' without one); its topic ids with ancestors
    primary: list[str] = field(default_factory=list)
    under: list[set[str]] = field(default_factory=list)
    primary_under: list[set[str]] = field(default_factory=list)
    # topic id (any level) -> papers with a row under it / whose primary is under it
    n_papers: dict[str, int] = field(default_factory=dict)
    n_primary: dict[str, int] = field(default_factory=dict)
    fingerprint: tuple = ()

    def ancestors(self, topic_id: str) -> list[str]:
        out, seen = [], set()
        while topic_id and topic_id in self.topics and topic_id not in seen:
            seen.add(topic_id)
            out.append(topic_id)
            topic_id = self.topics[topic_id][2] or ""
        return out

    def level_of(self, idx: int, level: str) -> str:
        """Name of the paper's primary topic at ``level`` ('' without one)."""
        for tid in self.ancestors(self.primary[idx]):
            if self.topics[tid][1] == level:
                return self.topics[tid][0]
        return ""


def _build(rows: dict[str, list[dict]], fingerprint: tuple) -> _Struct:
    st = _Struct(fingerprint=fingerprint)
    for r in rows["papers"]:
        idx = len(st.ids)
        pid = r["paper_id"]
        st.ids.append(pid)
        st.index[pid] = idx
        doi = norm_doi(r.get("doi") or "")
        st.doi.append(doi)
        if doi:
            st.by_doi.setdefault(doi, idx)
        st.title.append(re.sub(r"\s+", " ", r.get("title") or "").strip())
        st.year.append(r.get("year"))
        st.venue.append(r.get("venue") or "")
        st.cited_by_count.append(r.get("cited_by_count"))
        st.n_refs.append(r.get("n_refs_openalex"))
    n = len(st.ids)
    st.keys = [[] for _ in range(n)]
    st.authors = [[] for _ in range(n)]
    st.cites_out = [set() for _ in range(n)]
    st.cites_in = [set() for _ in range(n)]
    st.coupled = [{} for _ in range(n)]
    st.adj = [set() for _ in range(n)]
    st.primary = [""] * n
    st.under = [set() for _ in range(n)]
    st.primary_under = [set() for _ in range(n)]
    for r in rows["attachments"]:
        idx = st.index.get(r["paper_id"])
        key = (r.get("attachment_key") or "").upper()
        if idx is not None and key:
            st.by_key[key] = idx
            st.keys[idx].append(key)
    for ks in st.keys:
        ks.sort()
    for r in rows["authors"]:
        idx = st.index.get(r["paper_id"])
        aid = r.get("author_id") or ""
        if idx is None or not aid:
            continue
        st.authors[idx].append((aid, r.get("display_name") or ""))
        st.papers_of_author.setdefault(aid, []).append(idx)
    for r in rows["topics"]:
        st.topics[r["topic_id"]] = (
            r.get("display_name") or r["topic_id"],
            r.get("level") or "",
            r.get("parent_id"),
        )
    for r in rows["paper_topics"]:
        idx = st.index.get(r["paper_id"])
        if idx is None or r["topic_id"] not in st.topics:
            continue
        chain = st.ancestors(r["topic_id"])
        st.under[idx].update(chain)
        if r.get("is_primary"):
            st.primary[idx] = r["topic_id"]
            st.primary_under[idx].update(chain)
    for idx in range(n):
        for tid in st.under[idx]:
            st.n_papers[tid] = st.n_papers.get(tid, 0) + 1
        for tid in st.primary_under[idx]:
            st.n_primary[tid] = st.n_primary.get(tid, 0) + 1
    for r in rows["cites"]:
        a, b = st.index.get(r["citing_paper_id"]), st.index.get(r["cited_paper_id"])
        if a is None or b is None or a == b:
            continue
        st.cites_out[a].add(b)
        st.cites_in[b].add(a)
        st.adj[a].add(b)
        st.adj[b].add(a)
    for r in rows["coupled"]:
        a, b = st.index.get(r["paper_a"]), st.index.get(r["paper_b"])
        if a is None or b is None or a == b:
            continue
        link = (int(r.get("shared_refs") or 0), float(r.get("score") or 0.0))
        st.coupled[a][b] = link
        st.coupled[b][a] = link
        st.adj[a].add(b)
        st.adj[b].add(a)
    return st


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_state: dict = {
    "struct": None,
    "owner": None,  # (id(db), schema) the struct was read from
    "checked": 0.0,
    "failure": None,  # (reason, message) of a failed read, kept until the next check
    "avail_checked": 0.0,  # when available() last asked the catalog
    "lock": None,
    "loop": None,
    "docs": None,  # (owner, workspace, read at, keys)
    "known": None,  # (schema, available, reason): the last answer, for /health
}


def schema_name() -> str:
    """ZOTERO_STRUCT_SCHEMA, stripped; '' when unset."""
    return (os.environ.get(SCHEMA_ENV) or "").strip()


def _db(rag):
    """The document-status storage's PostgreSQL pool and workspace."""
    schema = schema_name()
    if not schema:
        raise Unavailable("off", f"{SCHEMA_ENV} is not set")
    if not _IDENT_RE.match(schema):
        raise Unavailable(
            "bad_schema_name",
            f"{SCHEMA_ENV} must be a lower-case SQL identifier (letters, digits, _)",
        )
    doc_status = getattr(rag, "doc_status", None)
    db = getattr(doc_status, "db", None)
    if db is None or not callable(getattr(db, "query", None)):
        raise Unavailable(
            "not_postgres", "the document-status storage is not PostgreSQL"
        )
    return db, schema, getattr(doc_status, "workspace", None) or "default"


def _lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    if _state["lock"] is None or _state["loop"] is not loop:
        _state.update(lock=asyncio.Lock(), loop=loop)
    return _state["lock"]


async def _fingerprint(db, schema: str) -> tuple:
    rows = await db.query(_SQL["fingerprint"], [schema], multirows=True) or []
    found = {r["relname"]: (int(r["oid"]), int(r["filenode"] or 0)) for r in rows}
    missing = [t for t in _TABLES if t not in found]
    if missing:
        raise Unavailable(
            "no_schema",
            f"schema '{schema}' is missing or incomplete (no table "
            + ", ".join(missing)
            + ")",
        )
    return tuple(sorted((t, found[t]) for t in _TABLES))


_LOADED = (
    "papers",
    "attachments",
    "authors",
    "topics",
    "paper_topics",
    "cites",
    "coupled",
)
# The big tables are read in _CHUNKS slices (rows of one paper stay together),
# so no single fetch holds the event loop for long while its rows are decoded;
# the loop runs between slices and the conversion into the structure happens in
# a worker thread.
_CHUNKED = ("authors", "paper_topics", "cites", "coupled")
_CHUNKS = 8


async def _read_tables(db, schema: str, fp: tuple) -> tuple[tuple, dict]:
    """The tables' rows, and the fingerprint they were read under. A table that
    is gone or unreadable (dropped between the fingerprint and the read) is
    tried once more under a fresh fingerprint, then reported as unavailable."""
    for attempt in (1, 2):
        try:
            rows = {}
            for name in _LOADED:
                sql = _SQL[name].format(s=schema)
                if name not in _CHUNKED:
                    rows[name] = await db.query(sql, multirows=True) or []
                    continue
                rows[name] = []
                for k in range(_CHUNKS):
                    rows[name] += (
                        await db.query(sql, [_CHUNKS, k], multirows=True) or []
                    )
                    await asyncio.sleep(0)
            return fp, rows
        except Exception as e:
            logger.warning(
                f"zotero_struct: reading '{schema}' failed ({attempt}/2): {e}"
            )
            if attempt == 2:
                raise Unavailable(
                    "no_schema",
                    f"schema '{schema}' could not be read (a table changed or went "
                    "missing while it was being loaded)",
                ) from e
            fp = await _fingerprint(db, schema)
    raise AssertionError("unreachable")


async def _structure(rag) -> _Struct:
    """The loaded structure, read again when its tables changed. A structure is
    kept only when every table was read: a failed read leaves none, and the
    failure stands until the next check, one _CHECK_INTERVAL later."""
    db, schema, _ = _db(rag)
    owner = (id(db), schema)
    try:
        async with _lock():
            now = time.monotonic()
            st = _state["struct"]
            if _state["owner"] == owner and now - _state["checked"] < _CHECK_INTERVAL:
                if st is not None:
                    return st
                if _state["failure"] is not None:
                    raise Unavailable(*_state["failure"])
            fp = await _fingerprint(db, schema)
            if st is not None and _state["owner"] == owner and st.fingerprint == fp:
                _state.update(checked=now, failure=None)
                return st
            started = time.monotonic()
            try:
                fp, rows = await _read_tables(db, schema, fp)
            except Unavailable as e:
                _state.update(
                    struct=None, owner=owner, checked=now, failure=(e.reason, str(e))
                )
                raise
            st = await asyncio.to_thread(_build, rows, fp)
            _state.update(struct=st, owner=owner, checked=now, failure=None)
            logger.info(
                f"zotero_struct: read {len(st.ids)} papers, {len(rows['cites'])} "
                f"citations and {len(rows['coupled'])} coupling pairs from "
                f"'{schema}' in {time.monotonic() - started:.1f}s"
            )
    except Unavailable as e:
        _state["known"] = (schema, False, e.reason)
        raise
    _state["known"] = (schema, True, None)
    return st


async def _kb_keys(rag) -> set[str]:
    """Attachment keys of the processed documents of the server's workspace."""
    db, _, workspace = _db(rag)
    cached = _state["docs"]
    now = time.monotonic()
    if (
        cached
        and cached[0] is db
        and cached[1] == workspace
        and now - cached[2] < _DOCS_TTL
    ):
        return cached[3]
    rows = await db.query(
        _SQL["documents"], [workspace, DocStatus.PROCESSED.value], multirows=True
    )
    keys = {k for r in rows or [] if (k := zotero_citations.zotero_key(r["file_path"]))}
    _state["docs"] = (db, workspace, now, keys)
    return keys


async def available(rag) -> bool:
    """Whether the structure can be read (cheap: the table check alone, run at
    most once per _CHECK_INTERVAL, and given up after _AVAILABLE_TIMEOUT so a
    slow pool cannot hold /health). Never raises."""
    schema = schema_name()
    if not schema:
        return False
    known = _state["known"]
    now = time.monotonic()
    if (
        known
        and known[0] == schema
        and now - max(_state["checked"], _state["avail_checked"]) < _CHECK_INTERVAL
    ):
        return known[1]
    _state["avail_checked"] = now
    try:
        db, schema, _ = _db(rag)
        await asyncio.wait_for(_fingerprint(db, schema), _AVAILABLE_TIMEOUT)
    except Unavailable as e:
        _state["known"] = (schema, False, e.reason)
        return False
    except Exception as e:
        logger.warning(
            f"zotero_struct: availability check failed: {e or type(e).__name__}"
        )
        _state["known"] = (schema, False, "error")
        return False
    _state["known"] = (schema, True, None)
    return True


def _reset_for_tests() -> None:
    _state.update(
        struct=None,
        owner=None,
        checked=0.0,
        failure=None,
        avail_checked=0.0,
        lock=None,
        loop=None,
        docs=None,
        known=None,
    )


# ---------------------------------------------------------------------------
# papers as results
# ---------------------------------------------------------------------------


def _resolve(st: _Struct, text: str) -> int | None:
    """The paper an attachment key, a DOI or a schema paper id names."""
    text = (text or "").strip()
    if text in st.index:
        return st.index[text]
    if text.lower().startswith("doi:") and text.lower() in st.index:
        return st.index[text.lower()]
    doi = norm_doi(text)
    if doi:
        return st.by_doi.get(doi)
    key = text.upper()
    return st.by_key.get(key) if _KEY_RE.match(key) else None


def _kb_key(st: _Struct, idx: int, kb: set[str]) -> str | None:
    return next((k for k in st.keys[idx] if k in kb), None)


def _node_id(st: _Struct, idx: int, kb: set[str]) -> str:
    """How the fork names a paper: its DOI, else an attachment key (one in the
    knowledge base first), else the schema id."""
    if st.doi[idx]:
        return st.doi[idx]
    return _kb_key(st, idx, kb) or (st.keys[idx][0] if st.keys[idx] else st.ids[idx])


def _family(st: _Struct, idx: int) -> str:
    """First author's family name: the Zotero metadata's first creator when an
    attachment has one, else the last word of the schema's first author."""
    meta = zotero_citations._load()
    for key in st.keys[idx]:
        entry = meta.get(key)
        creators = entry.get("creators") if isinstance(entry, dict) else None
        if creators and isinstance(creators[0], (list, tuple)) and creators[0]:
            family = str(creators[0][0] or "").strip()
            if family:
                return family
    if st.authors[idx]:
        words = st.authors[idx][0][1].split()
        if words:
            return words[-1]
    return ""


def _short(st: _Struct, idx: int) -> str:
    """'Family Year', else the title's first words."""
    who = " ".join(p for p in (_family(st, idx), str(st.year[idx] or "")) if p).strip()
    if who:
        return who
    words = st.title[idx].split()
    return " ".join(words[:5]) + ("…" if len(words) > 5 else "") or st.ids[idx]


def _lite(st: _Struct, idx: int, kb: set[str]) -> dict:
    """A paper as the agent routes list it."""
    key = _kb_key(st, idx, kb)
    names = [name for _, name in st.authors[idx]]
    return {
        "paper_id": st.ids[idx],
        "doi": st.doi[idx] or None,
        "label": _short(st, idx),
        "title": st.title[idx],
        "year": st.year[idx],
        "venue": st.venue[idx] or None,
        "authors": names[:3],
        "n_authors": len(names),
        "field": st.level_of(idx, "field") or None,
        "topic": st.level_of(idx, "topic") or None,
        "in_kb": key is not None,
        "key": key,
        "keys": st.keys[idx],
        "cited_by_library": len(st.cites_in[idx]),
        "cites_library": len(st.cites_out[idx]),
        "cited_by_count": st.cited_by_count[idx],
    }


def _link_weight(st: _Struct, a: int, b: int) -> float:
    """A link's strength for the map and its edge width: _W_CITE when either
    paper cites the other (once, also when both do), plus _W_COUPLED times the
    coupling score."""
    w = _W_CITE if (b in st.cites_out[a] or a in st.cites_out[b]) else 0.0
    if b in st.coupled[a]:
        w += _W_COUPLED * st.coupled[a][b][1]
    return w


def _relation(st: _Struct, a: int, b: int) -> list[str]:
    """How paper a relates to paper b, read from a."""
    out = []
    if b in st.cites_out[a]:
        out.append("cites")
    if b in st.cites_in[a]:
        out.append("cited_by")
    if b in st.coupled[a]:
        out.append("coupled")
    return out


# ---------------------------------------------------------------------------
# related papers
# ---------------------------------------------------------------------------


def _related(
    st: _Struct, kb: set[str], idx: int, relations: list[str], limit: int
) -> dict:
    found: dict[int, dict] = {}

    def entry(j: int) -> dict:
        return found.setdefault(
            j,
            {"reasons": [], "shared_refs": None, "coupling_score": None, "shared": []},
        )

    totals = dict.fromkeys(relations, 0)
    if "cites" in relations:
        for j in st.cites_out[idx]:
            entry(j)["reasons"].append("cites")
        totals["cites"] = len(st.cites_out[idx])
    if "cited_by" in relations:
        for j in st.cites_in[idx]:
            entry(j)["reasons"].append("cited_by")
        totals["cited_by"] = len(st.cites_in[idx])
    if "coupled" in relations:
        for j, (shared, score) in st.coupled[idx].items():
            e = entry(j)
            e["reasons"].append("coupled")
            e["shared_refs"], e["coupling_score"] = shared, round(score, 4)
        totals["coupled"] = len(st.coupled[idx])
    if "coauthor" in relations:
        names = dict(st.authors[idx])
        coauthored = set()
        for aid, name in names.items():
            for j in st.papers_of_author.get(aid, []):
                if j != idx:
                    entry(j)["shared"].append(name)
                    coauthored.add(j)
        for j in coauthored:
            entry(j)["reasons"].append("coauthor")
        totals["coauthor"] = len(coauthored)

    def score(j: int) -> float:
        e = found[j]
        cited = ("cites" in e["reasons"]) + ("cited_by" in e["reasons"])
        s = (_W_CITE if cited else 0.0) + (_W_BOTH if cited == 2 else 0.0)
        s += _W_COUPLED * (e["coupling_score"] or 0.0)
        s += _W_AUTHOR * min(len(e["shared"]), _AUTHOR_CAP)
        return round(s, 4)

    ranked = sorted(
        found,
        key=lambda j: (-score(j), -len(st.cites_in[j]), -(st.year[j] or 0), st.ids[j]),
    )
    related = []
    for j in ranked[:limit]:
        e = found[j]
        item = {**_lite(st, j, kb), "score": score(j), "reasons": e["reasons"]}
        if e["coupling_score"] is not None:
            item["shared_refs"] = e["shared_refs"]
            item["coupling_score"] = e["coupling_score"]
        if e["shared"]:
            item["shared_authors"] = sorted(set(e["shared"]))
        related.append(item)
    totals["papers"] = len(found)
    totals["in_kb"] = sum(1 for j in found if _kb_key(st, j, kb))
    return {
        "paper": _lite(st, idx, kb),
        "relations": relations,
        "totals": totals,
        "ranking": (
            f"score = {_W_CITE} for a citation either way (+{_W_BOTH} both ways) "
            f"+ {_W_COUPLED} x coupling score + {_W_AUTHOR} per shared author "
            f"(at most {_AUTHOR_CAP}); ties: most cited by library papers, newest"
        ),
        "related": related,
        "truncated": len(found) > limit,
    }


async def related_papers(
    rag, paper: str, relations: list[str] | None = None, limit: int = 20
) -> dict:
    """Papers linked to one paper: those it cites, those citing it, its
    coupling partners and papers sharing an author, best first.

    ``reason: "not_in_structure"`` when no paper of the structure matches
    ``paper``. Raises :class:`Unavailable`."""
    st = await _structure(rag)
    kb = await _kb_keys(rag)
    idx = _resolve(st, paper)
    if idx is None:
        return {"paper": None, "asked": paper, "reason": "not_in_structure"}
    rel = [r for r in RELATIONS if r in (relations or RELATIONS)]
    return await asyncio.to_thread(_related, st, kb, idx, rel, limit)


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------


def _topic_row(st: _Struct, tid: str) -> dict:
    name, level, _ = st.topics[tid]
    chain = [st.topics[t][0] for t in reversed(st.ancestors(tid))]
    return {
        "topic_id": tid,
        "name": name,
        "level": level,
        "path": " > ".join(chain),
        "n_papers": st.n_papers.get(tid, 0),
        "n_primary": st.n_primary.get(tid, 0),
    }


def _find_topics(st: _Struct, q: str, level: str | None) -> list[str]:
    """Topic ids whose name holds every word of ``q`` (folded), exact names
    first, then names starting with it, then by papers whose primary topic is
    under them; without ``q`` every topic (of ``level``) by that count."""
    words = fold(q).split()
    exact = fold(q).strip()
    hits = []
    for tid, (name, lvl, _) in st.topics.items():
        if level and lvl != level:
            continue
        folded = fold(name)
        if all(w in folded for w in words):
            hits.append(
                (
                    folded != exact,
                    not folded.startswith(exact),
                    -st.n_primary.get(tid, 0),
                    -st.n_papers.get(tid, 0),
                    tid,
                )
            )
    return [h[-1] for h in sorted(hits)]


def _topics(st: _Struct, q: str, level: str | None, limit: int) -> dict:
    tids = _find_topics(st, q, level)
    return {
        "total": len(tids),
        "topics": [_topic_row(st, t) for t in tids[:limit]],
    }


async def list_topics(
    rag, q: str = "", level: str | None = None, limit: int = 20
) -> dict:
    """Topics at any level whose name holds every word of ``q``, with how many
    papers have a topic under each and how many have their primary topic
    under it. Raises :class:`Unavailable`."""
    st = await _structure(rag)
    return await asyncio.to_thread(_topics, st, q or "", level, limit)


def _by_topic(
    st: _Struct, kb: set[str], topic: str, limit: int, primary_only: bool
) -> dict:
    topic = (topic or "").strip()
    alternatives: list[str] = []
    if topic in st.topics:
        tid = topic
    else:
        tids = _find_topics(st, topic, None)
        if not tids:
            return {"topic": None, "reason": "no_topic", "papers": [], "total": 0}
        tid, alternatives = tids[0], tids[1:6]
    members = [
        i
        for i in range(len(st.ids))
        if tid in st.primary_under[i] or (not primary_only and tid in st.under[i])
    ]
    members.sort(
        key=lambda i: (
            tid not in st.primary_under[i],
            -len(st.cites_in[i]),
            -(st.cited_by_count[i] or 0),
            -(st.year[i] or 0),
            st.ids[i],
        )
    )
    papers = []
    for i in members[:limit]:
        papers.append({**_lite(st, i, kb), "primary": tid in st.primary_under[i]})
    return {
        "topic": _topic_row(st, tid),
        "alternatives": [_topic_row(st, t) for t in alternatives],
        "primary_only": primary_only,
        "total": len(members),
        "in_kb": sum(1 for i in members if _kb_key(st, i, kb)),
        "ranking": (
            "papers whose primary topic is under the topic first, then most cited "
            "by library papers, then most cited overall (OpenAlex), then newest"
        ),
        "papers": papers,
        "truncated": len(members) > limit,
    }


async def papers_by_topic(
    rag, topic: str, limit: int = 20, primary_only: bool = False
) -> dict:
    """The papers under one topic: ``topic`` is a topic id at any level
    (``topic:T…``, ``subfield:…``, ``field:…``, ``domain:…``) or words of its
    name (the best match is used; ``alternatives`` lists the next ones). A
    paper counts when one of its topics is the topic or below it;
    ``primary_only`` keeps papers whose primary topic is. ``reason:
    "no_topic"`` when nothing matches. Raises :class:`Unavailable`."""
    st = await _structure(rag)
    kb = await _kb_keys(rag)
    return await asyncio.to_thread(_by_topic, st, kb, topic, limit, primary_only)


# ---------------------------------------------------------------------------
# what connects two papers
# ---------------------------------------------------------------------------


def _cheapest_path(
    st: _Struct, a: int, b: int, banned: set[int], banned_pairs: set, max_hops: int
) -> list[int] | None:
    """Cheapest path a -> b of at most ``max_hops`` steps over citations (either
    way) and coupling, where entering a paper other than b costs
    1 + _HUB_PENALTY * log2(1 + its links); papers in ``banned`` and the pairs in
    ``banned_pairs`` are not used.

    The search runs over (paper, steps used) states, so a paper reached cheaply
    by a long chain does not hide a costlier route that gets there in fewer
    steps: whenever a path within the limit exists, the cheapest one is found.
    A state is skipped when a settled one (cheaper or equal, since states leave
    the heap in cost order) reached the same paper in no more steps."""
    best = {(a, 0): 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(0.0, 0, a)]
    fewest: dict[int, int] = {}  # paper -> fewest steps among its settled states
    while heap:
        d, h, u = heapq.heappop(heap)
        if best[(u, h)] < d or fewest.get(u, max_hops + 1) <= h:
            continue
        fewest[u] = h
        if u == b:
            state, path = (u, h), [u]
            while state != (a, 0):
                state = prev[state]
                path.append(state[0])
            return path[::-1]
        if h >= max_hops:
            continue
        for v in st.adj[u]:
            if v in banned or (min(u, v), max(u, v)) in banned_pairs:
                continue
            if fewest.get(v, max_hops + 1) <= h + 1:
                continue
            cost = 1.0 if v == b else 1.0 + _HUB_PENALTY * math.log2(1 + len(st.adj[v]))
            nd = d + cost
            if nd < best.get((v, h + 1), math.inf):
                best[(v, h + 1)] = nd
                prev[(v, h + 1)] = (u, h)
                heapq.heappush(heap, (nd, h + 1, v))
    return None


def _unique_labels(st: _Struct, papers: list[int]) -> dict[int, str]:
    """'Family Year' for each paper; papers with the same label get a, b, ...
    after it (in id order), so no two papers of one answer read alike."""
    labels = {v: _short(st, v) for v in papers}
    same: dict[str, list[int]] = {}
    for v in labels:
        same.setdefault(labels[v], []).append(v)
    for label, vs in same.items():
        if len(vs) > 1:
            for n, v in enumerate(sorted(vs, key=lambda i: st.ids[i])):
                labels[v] = f"{label}{chr(ord('a') + n) if n < 26 else n}"
    return labels


def _connect(
    st: _Struct, kb: set[str], a: int, b: int, max_paths: int, max_hops: int
) -> dict:
    out: dict = {"a": _lite(st, a, kb), "b": _lite(st, b, kb)}
    if a == b:
        out.update(reason="same_paper", paths=[], links={})
        return out
    b_authors = {aid for aid, _ in st.authors[b]}
    links: dict = {
        "relation": _relation(st, a, b),
        "shared_authors": sorted(
            {name for aid, name in st.authors[a] if aid in b_authors}
        ),
    }
    for level in ("topic", "subfield", "field"):
        name = st.level_of(a, level)
        links[f"same_{level}"] = (
            name if name and name == st.level_of(b, level) else None
        )
    links["cited_by_both"] = len(st.cites_in[a] & st.cites_in[b])
    links["cite_both"] = len(st.cites_out[a] & st.cites_out[b])
    if b in st.coupled[a]:
        links["shared_refs"], links["coupling_score"] = (
            st.coupled[a][b][0],
            round(st.coupled[a][b][1], 4),
        )
    out["links"] = links
    found, banned, banned_pairs = [], set(), set()
    for _ in range(max_paths):
        path = _cheapest_path(st, a, b, banned, banned_pairs, max_hops)
        if path is None:
            break
        inner = path[1:-1]
        if not inner:
            banned_pairs.add((min(a, b), max(a, b)))
        banned.update(inner)
        found.append(path)
    # one label per paper for the whole answer, so two papers never read alike
    labels = _unique_labels(st, [a, b] + [i for path in found for i in path[1:-1]])
    out["a"]["label"], out["b"]["label"] = labels[a], labels[b]
    paths = []
    for path in found:
        inner = path[1:-1]
        cost = (
            sum(1.0 + _HUB_PENALTY * math.log2(1 + len(st.adj[v])) for v in inner) + 1.0
        )
        steps = []
        for u, v in zip(path, path[1:]):
            step = {
                "from": labels[u],
                "relation": _relation(st, u, v),
                "to": labels[v],
            }
            if v in st.coupled[u]:
                step["shared_refs"] = st.coupled[u][v][0]
            steps.append(step)
        papers = []
        for i in inner:
            papers.append({**_lite(st, i, kb), "label": labels[i]})
        paths.append(
            {
                "hops": len(path) - 1,
                "cost": round(cost, 3),
                "papers": papers,
                "steps": steps,
            }
        )
    out["paths"] = paths
    out["max_hops"] = max_hops
    out["ranking"] = (
        f"cheapest first; each paper passed through costs 1 + {_HUB_PENALTY} x "
        "log2(1 + its links), and later paths avoid the papers of earlier ones"
    )
    if not paths:
        out["reason"] = "not_connected"
        out["message"] = (
            f"No route of at most {max_hops} steps over citations (either "
            "direction) and bibliographic coupling links the two papers"
        )
    return out


async def connect_papers(
    rag, a: str, b: str, max_paths: int = 3, max_hops: int = 6
) -> dict:
    """What connects two papers: up to ``max_paths`` paths over citations
    (either direction) and coupling, cheapest first, hubs penalised, each step
    labelled; and the direct ``links`` between them (citation, coupling,
    shared authors, same primary topic, subfield or field, papers citing both,
    library papers both cite). ``reason``: ``not_in_structure`` (``missing``
    names what matched no paper), ``same_paper``, ``not_connected``. Raises
    :class:`Unavailable`."""
    st = await _structure(rag)
    kb = await _kb_keys(rag)
    ia, ib = _resolve(st, a), _resolve(st, b)
    if ia is None or ib is None:
        missing = [t for t, i in ((a, ia), (b, ib)) if i is None]
        return {"reason": "not_in_structure", "missing": missing, "paths": []}
    return await asyncio.to_thread(_connect, st, kb, ia, ib, max_paths, max_hops)


# ---------------------------------------------------------------------------
# paper map
# ---------------------------------------------------------------------------


def _map(
    st: _Struct, seeds: list[int], hops: int, max_nodes: int
) -> tuple[list[int], dict[int, int], dict]:
    """Shown papers (seeds first, then one-hop then two-hop papers, strongest
    first), each paper's hop, and the counts before the cap."""
    seed_set = set(seeds)
    strength1: dict[int, float] = {}
    for s in seeds:
        for v in st.adj[s]:
            if v not in seed_set:
                strength1[v] = strength1.get(v, 0.0) + _link_weight(st, s, v)
    strength2: dict[int, float] = {}
    if hops >= 2:
        for u, su in strength1.items():
            damp = math.log2(2 + len(st.adj[u]))
            for v in st.adj[u]:
                if v not in seed_set and v not in strength1:
                    strength2[v] = (
                        strength2.get(v, 0.0) + su * _link_weight(st, u, v) / damp
                    )

    def order(strength):
        return sorted(
            strength,
            key=lambda v: (-strength[v], -len(st.cites_in[v]), st.ids[v]),
        )

    shown = list(dict.fromkeys(seeds))
    shown += order(strength1)
    shown += order(strength2)
    hop = {s: 0 for s in seeds}
    hop.update({v: 1 for v in strength1})
    hop.update({v: 2 for v in strength2})
    counts = {"n_hop1": len(strength1), "n_hop2": len(strength2)}
    return shown[:max_nodes], hop, counts


def _seed_summary(st: _Struct, kb: set[str], idx: int) -> dict:
    key = _kb_key(st, idx, kb)
    keys = st.keys[idx]
    return {
        "paper": _node_id(st, idx, kb),
        "doi": st.doi[idx],
        "keys": keys,
        "title": st.title[idx],
        "citation": zotero_citations.citation_short(f"{key or keys[0]}__")
        if keys
        else _short(st, idx),
        "year": str(st.year[idx] or ""),
        "n_refs": st.n_refs[idx],
        "n_cited": len(st.cites_out[idx]),
        "n_cited_in_kb": sum(1 for j in st.cites_out[idx] if _kb_key(st, j, kb)),
    }


def _orient(st: _Struct, a: int, b: int) -> tuple[int, int]:
    """A pair as an edge runs: from the citing paper, else from the first id."""
    if b in st.cites_out[a] and a not in st.cites_out[b]:
        return a, b
    if a in st.cites_out[b] and b not in st.cites_out[a]:
        return b, a
    return (a, b) if st.ids[a] <= st.ids[b] else (b, a)


def _thin_links(st: _Struct, shown: list[int], seeds: set[int], max_nodes: int) -> dict:
    """The links among the shown papers (pairs of paper numbers) to draw:
    every link of a chosen paper; of every other paper its _EDGES_PER_PAPER
    strongest (a link kept for either end counts once); then at most
    _EDGE_CAP_FACTOR x ``max_nodes`` in all, the weakest links of papers that
    were not chosen going first. Strength is _link_weight, ties go to the
    papers' ids. ``found`` counts all the links among the shown papers."""
    inside = set(shown)
    weight: dict[tuple[int, int], float] = {}
    for u in shown:
        for v in st.adj[u]:
            if v in inside and u < v:
                weight[(u, v)] = _link_weight(st, u, v)

    def rank(pair):
        return (-weight[pair], *sorted((st.ids[pair[0]], st.ids[pair[1]])))

    chosen = {p for p in weight if p[0] in seeds or p[1] in seeds}
    own: dict[int, list] = {}
    for pair in weight:
        for v in pair:
            if v not in seeds:
                own.setdefault(v, []).append(pair)
    kept = set(chosen)
    for pairs in own.values():
        kept.update(sorted(pairs, key=rank)[:_EDGES_PER_PAPER])
    ordered = sorted(kept, key=lambda p: (p not in chosen, *rank(p)))
    kept = sorted(ordered[: _EDGE_CAP_FACTOR * max_nodes], key=rank)
    return {"found": len(weight), "kept": kept}


def _map_graph(
    st: _Struct, kb: set[str], asked: list[str], hops: int, max_nodes: int
) -> dict:
    view: dict = {"mode": "map", "hops": hops}
    empty = KnowledgeGraph().model_dump()
    summaries, seeds = [], []
    for text in dict.fromkeys(asked):
        idx = _resolve(st, text)
        if idx is None:
            summaries.append({"paper": text, "reason": "not_in_structure"})
        elif idx not in seeds:
            seeds.append(idx)
            summaries.append(_seed_summary(st, kb, idx))
    view["papers"] = summaries
    if len(summaries) == 1:
        view = {**summaries[0], **view}
    if not seeds:
        return {**empty, "paper": {**view, "reason": "not_in_structure"}}
    shown, hop, counts = _map(st, seeds, hops, max_nodes)
    view.update(counts)
    view["shown_hop1"] = sum(1 for v in shown if hop[v] == 1)
    view["shown_hop2"] = sum(1 for v in shown if hop[v] == 2)
    view["ranking"] = (
        "the chosen papers, then one-hop papers by link strength to them "
        f"({_W_CITE} when either cites the other + {_W_COUPLED} x coupling score), "
        "then two-hop papers by the strength of the one-hop papers they link "
        "to, damped by log2(2 + links) of each; links: all of a chosen paper's, "
        f"the {_EDGES_PER_PAPER} strongest of each other paper, at most "
        f"{_EDGE_CAP_FACTOR} x the node limit"
    )

    labels = _unique_labels(st, shown)
    links = _thin_links(st, shown, set(seeds), max_nodes)
    view["n_links"], view["shown_links"] = links["found"], len(links["kept"])
    kg = KnowledgeGraph(
        is_truncated=len(shown) < len(seeds) + counts["n_hop1"] + counts["n_hop2"]
        or len(links["kept"]) < links["found"]
    )
    node_ids = {v: _node_id(st, v, kb) for v in shown}
    for v in shown:
        names = [name for _, name in st.authors[v]]
        props = {
            "entity_type": st.level_of(v, "field") or "Unclassified",
            "paper_id": st.ids[v],
            "title": st.title[v],
            "year": st.year[v],
            "venue": st.venue[v] or None,
            "doi": st.doi[v] or None,
            "authors": "; ".join(names[:_AUTHORS_SHOWN])
            + (" et al." if len(names) > _AUTHORS_SHOWN else ""),
            "topic": st.level_of(v, "topic") or None,
            "subfield": st.level_of(v, "subfield") or None,
            "hop": hop[v],
            "cited_by_library": len(st.cites_in[v]),
            "cites_library": len(st.cites_out[v]),
            "coupled_with": len(st.coupled[v]),
            "cited_by_count": st.cited_by_count[v],
            "in_kb": _kb_key(st, v, kb) is not None,
            "attachment_key": _kb_key(st, v, kb)
            or (st.keys[v][0] if st.keys[v] else None),
        }
        kg.nodes.append(
            KnowledgeGraphNode(
                id=node_ids[v],
                labels=[labels[v]],
                properties={k: p for k, p in props.items() if p is not None},
            )
        )
    for a, b in links["kept"]:
        u, v = _orient(st, a, b)
        cites_uv, cites_vu = v in st.cites_out[u], u in st.cites_out[v]
        words, props = [], {}
        if cites_uv and cites_vu:
            words.append("cite each other")
        elif cites_uv:
            words.append("cites")
        if v in st.coupled[u]:
            words.append("coupled")
            props["shared_refs"] = st.coupled[u][v][0]
            props["coupling_score"] = round(st.coupled[u][v][1], 4)
        props.update(
            keywords=", ".join(words),
            weight=round(_link_weight(st, u, v), 4),
            description=f"{labels[u]} {' and '.join(words)} {labels[v]}"
            if cites_uv or cites_vu
            else f"{labels[u]} and {labels[v]} share {props['shared_refs']} references",
        )
        if cites_uv or cites_vu:
            props["direction"] = (
                "both" if cites_uv and cites_vu else "source cites target"
            )
        kg.edges.append(
            KnowledgeGraphEdge(
                id=f"{node_ids[u]}-{node_ids[v]}",
                type="DIRECTED" if cites_uv or cites_vu else "UNDIRECTED",
                source=node_ids[u],
                target=node_ids[v],
                properties=props,
            )
        )
    if not kg.edges:
        view["reason"] = "no_links"
    return {**kg.model_dump(), "paper": view}


async def paper_map(
    rag,
    papers: str | list[str],
    hops: int = 1,
    max_nodes: int = 1000,
    encode: bool = False,
) -> dict | bytes:
    """A graph of papers around one or more chosen papers, in the ``/graphs``
    shape, plus ``paper``: the chosen papers' summaries and the view's counts.

    Nodes are papers one or ``hops`` = 2 steps away over citations and
    coupling, capped at ``max_nodes`` (seeds kept, then the strongest
    neighbours; ``paper.ranking`` says how). A node's ``entity_type`` is the
    field of its primary topic, its label "Family Year". An edge runs from the
    citing paper to the cited one; a coupling-only edge is ``UNDIRECTED``.
    Links are thinned (see :func:`_thin_links`): ``paper.n_links`` counts those
    among the shown papers, ``paper.shown_links`` the ones drawn, and
    ``is_truncated`` is true when papers or links were dropped.
    ``paper.reason``: ``too_many_papers`` (over MAX_PAPERS),
    ``not_in_structure`` (no chosen paper is in it), ``no_links``.
    With ``encode`` the answer is JSON bytes made in the worker thread, so a
    large graph is never serialised on the event loop. Raises
    :class:`Unavailable`, also for more than MAX_PAPERS papers."""
    _db(rag)  # off, bad name or no PostgreSQL: unavailable whatever the request
    finish = (
        (
            lambda r: json.dumps(
                r, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
        )
        if encode
        else (lambda r: r)
    )
    asked = [papers] if isinstance(papers, str) else list(papers)
    if len(set(asked)) > MAX_PAPERS:
        return finish(
            {
                **KnowledgeGraph().model_dump(),
                "paper": {
                    "mode": "map",
                    "hops": hops,
                    "papers": [],
                    "reason": "too_many_papers",
                },
            }
        )
    st = await _structure(rag)
    kb = await _kb_keys(rag)
    return await asyncio.to_thread(
        lambda: finish(_map_graph(st, kb, asked, hops, max_nodes))
    )
