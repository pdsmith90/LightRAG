"""Papers of a Zotero-backed corpus as views of the knowledge graph.

The WebUI's paper filter shows the part of the graph one paper contributed, or
the part it shares with the library papers it cites. A paper is named by its
DOI or by the Zotero storage key of one of its attachments; every processed
document whose corpus file is ``<KEY>__<slug>.md`` for one of the paper's keys
belongs to it (a key stands for every attachment with the same DOI).

What a document contributed is read from LightRAG's per-document anchors:
``full_entities`` / ``full_relations`` list the names its extraction produced,
and a name counts only while ``entity_chunks`` / ``relation_chunks`` still cite
one of the document's chunks -- a prune or a merge can leave a name in the
anchors that no longer draws on the document. Only key lookups, so it works on
every storage backend.

Which library papers a paper cites comes from ``zotero_references.json``
(``zotero-toolkit/corpus/build_references.py``: OpenAlex reference lists kept
to library DOIs), read from the server's working directory or the path in
ZOTERO_REFERENCES and reloaded when it changes. Without it the shared view is
empty and says why.

Fork-local (no upstream counterpart).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

from lightrag import zotero_citations
from lightrag.base import DocStatus
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import make_relation_chunk_key, parse_relation_chunk_key

logger = logging.getLogger("lightrag")

REFERENCES_PATH = os.path.abspath(
    os.environ.get("ZOTERO_REFERENCES", "zotero_references.json")
)

_STAT_INTERVAL = 5.0
# How long the list of processed documents is reused: the pipeline adds a few
# documents an hour, and a search runs on every keystroke.
_DOCS_TTL = 300.0
# A shared node lists at most this many of the references it is shared with.
_SHARED_WITH_MAX = 8

_lock = threading.Lock()
_refs: dict = {}
_refs_mtime: float | None = None
_refs_next_stat = 0.0
_refs_warned = False

_docs_cache: dict = {"at": 0.0, "owner": None, "by_key": {}}

_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


def norm_doi(text: str) -> str:
    """Lowercase DOI without a resolver prefix or ``doi:``; '' when not a DOI."""
    d = (text or "").strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d if d.startswith("10.") and "/" in d else ""


def _load_references() -> dict:
    """Citing DOI -> {"n_refs", "cited"}, reloaded when the file changes."""
    global _refs, _refs_mtime, _refs_next_stat, _refs_warned
    now = time.monotonic()
    with _lock:
        if now < _refs_next_stat:
            return _refs
        _refs_next_stat = now + _STAT_INTERVAL
        try:
            mtime = os.path.getmtime(REFERENCES_PATH)
        except OSError as e:
            if not _refs_warned:
                logger.warning(
                    f"zotero_papers: references unavailable at {REFERENCES_PATH} ({e}); "
                    "the paper-and-references graph view is empty"
                )
                _refs_warned = True
            return _refs
        if mtime == _refs_mtime:
            return _refs
        try:
            with open(REFERENCES_PATH, encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as e:
            if not _refs_warned:
                logger.warning(f"zotero_papers: cannot parse {REFERENCES_PATH}: {e}")
                _refs_warned = True
            return _refs
        if not isinstance(loaded, dict):
            return _refs
        _refs, _refs_mtime, _refs_warned = loaded, mtime, False
        logger.info(f"zotero_papers: loaded {len(_refs)} reference lists")
        return _refs


def _keys_by_doi(meta: dict) -> dict[str, list[str]]:
    by_doi: dict[str, list[str]] = {}
    for key, entry in meta.items():
        if isinstance(entry, dict):
            doi = norm_doi(entry.get("doi") or "")
            if doi:
                by_doi.setdefault(doi, []).append(key)
    return by_doi


def resolve_paper(paper: str) -> tuple[str, list[str]]:
    """``(doi, storage keys)`` of the paper a DOI or a storage key names; the
    DOI is '' for a key without one, and both are empty for anything else."""
    meta = zotero_citations._load()
    doi = norm_doi(paper)
    if not doi:
        key = (paper or "").strip().upper()
        if not _KEY_RE.match(key):
            return "", []
        entry = meta.get(key)
        doi = norm_doi(entry.get("doi") or "") if isinstance(entry, dict) else ""
        if not doi:
            return "", [key]
    return doi, _keys_by_doi(meta).get(doi, [])


async def _kb_documents(rag) -> dict[str, list[tuple[str, set[str]]]]:
    """Storage key -> [(doc id, chunk ids)] of every processed document."""
    now = time.monotonic()
    cache = _docs_cache
    if cache["owner"] is rag.doc_status and now - cache["at"] < _DOCS_TTL:
        return cache["by_key"]
    docs = await rag.doc_status.get_docs_by_statuses([DocStatus.PROCESSED])
    by_key: dict[str, list[tuple[str, set[str]]]] = {}
    for doc_id, status in docs.items():
        key = zotero_citations.zotero_key(status.file_path or "")
        if key:
            by_key.setdefault(key, []).append((doc_id, set(status.chunks_list or [])))
    cache.update(at=now, owner=rag.doc_status, by_key=by_key)
    return by_key


def _paper_summary(
    doi: str, keys: list[str], kb: dict, refs: dict, meta: dict, by_doi: dict
) -> dict:
    key = next((k for k in keys if k in kb), keys[0])
    entry = meta.get(key) if isinstance(meta.get(key), dict) else {}
    listed = refs.get(doi) if doi else None
    cited = (listed or {}).get("cited") or []
    in_kb = [c for c in cited if any(k in kb for k in by_doi.get(c, []))]
    return {
        "paper": doi or key,
        "doi": doi,
        "keys": keys,
        "title": zotero_citations._plain_title(entry) if entry else "",
        "citation": zotero_citations.citation_short(f"{key}__"),
        "year": (entry or {}).get("year", ""),
        "n_refs": (listed or {}).get("n_refs"),
        "n_cited": len(cited) if listed else None,
        "n_cited_in_kb": len(in_kb) if listed else None,
    }


async def search_papers(rag, query: str, limit: int = 20) -> list[dict]:
    """Papers in the knowledge base whose DOI, storage key, title, authors or
    year match ``query`` (every word must occur), best match first."""
    words = (query or "").lower().split()
    if not words:
        return []
    kb = await _kb_documents(rag)
    meta = zotero_citations._load()
    doi_q = norm_doi(query)
    key_q = query.strip().upper()
    best: dict[str, tuple] = {}  # paper (DOI, else key) -> best rank
    keys_of: dict[str, list[str]] = {}
    for key in kb:
        entry = meta.get(key)
        if not isinstance(entry, dict):
            continue
        doi = norm_doi(entry.get("doi") or "")
        paper = doi or key
        title = zotero_citations._plain_title(entry).lower()
        hay = " ".join(
            [
                title,
                " ".join(entry.get("authors") or []).lower(),
                entry.get("year") or "",
                doi,
                key.lower(),
            ]
        )
        exact = (doi_q and doi == doi_q) or key == key_q
        if not exact and not all(w in hay for w in words):
            continue
        year = entry.get("year") or ""
        rank = (
            not exact,
            -sum(w in title for w in words),
            -int(year) if year.isdigit() else 0,
        )
        best[paper] = min(best.get(paper, rank), rank)
        keys_of.setdefault(paper, []).append(key)
    refs = _load_references()
    by_doi = _keys_by_doi(meta)
    papers = sorted(best, key=best.get)[:limit]
    return [
        _paper_summary(norm_doi(p), sorted(keys_of[p]), kb, refs, meta, by_doi)
        for p in papers
    ]


async def _contributions(
    rag, docs: list[tuple[str, set[str]]], entities: bool
) -> dict[str, set[str]]:
    """Entity name (or relation pair key) -> the chunk ids, among ``docs``',
    that it still cites."""
    anchors, chunks_store, field = (
        (rag.full_entities, rag.entity_chunks, "entity_names")
        if entities
        else (rag.full_relations, rag.relation_chunks, "relation_pairs")
    )
    rows = await anchors.get_by_ids([doc_id for doc_id, _ in docs])
    candidates: dict[str, set[str]] = {}  # name -> chunks of the docs listing it
    for (_, chunks), row in zip(docs, rows):
        for item in (row or {}).get(field) or []:
            name = item if entities else make_relation_chunk_key(item[0], item[1])
            candidates.setdefault(name, set()).update(chunks)
    names = list(candidates)
    found = await chunks_store.get_by_ids(names) if names else []
    out = {}
    for name, row in zip(names, found):
        hit = set((row or {}).get("chunk_ids") or []) & candidates[name]
        if hit:
            out[name] = hit
    return out


async def paper_graph(
    rag, paper: str, mode: str = "paper", max_nodes: int = 1000
) -> dict:
    """The graph a paper contributed (``mode="paper"``), or the part of it it
    shares with the library papers it cites (``mode="shared"``).

    Nodes are the entities the paper's documents still draw on; in the shared
    view only those that at least one cited paper's documents draw on too,
    each listing those papers in its ``shared_with`` property. Edges are the
    relations the shown documents contributed between two shown nodes. Over
    ``max_nodes``, the nodes shared with the most papers and then the best
    connected are kept. ``paper`` in the result summarises the paper and its
    references; an empty graph carries ``paper["reason"]``: ``not_in_kb``,
    ``no_references_file``, ``no_reference_list`` (no DOI, or OpenAlex lists no
    references) or ``no_references_in_kb``.
    """
    doi, keys = resolve_paper(paper)
    kb = await _kb_documents(rag)
    refs = _load_references()
    meta = zotero_citations._load()
    by_doi = _keys_by_doi(meta)
    paper_docs = [d for k in keys for d in kb.get(k, [])]
    summary = (
        _paper_summary(doi, keys, kb, refs, meta, by_doi) if keys else {"paper": paper}
    )
    summary.update(mode=mode, references_available=bool(refs), cited_sharing=0)
    if not paper_docs:
        summary["reason"] = "not_in_kb"
        return {**KnowledgeGraph().model_dump(), "paper": summary}

    # works: "" is the paper itself, else a cited paper's DOI
    works: dict[str, list[tuple[str, set[str]]]] = {"": paper_docs}
    if mode == "shared":
        for cited in ((refs.get(doi) or {}).get("cited") or []) if doi else []:
            cited_docs = [d for k in by_doi.get(cited, []) for d in kb.get(k, [])]
            if cited_docs and cited not in works:
                works[cited] = cited_docs
        if len(works) == 1:
            if not refs:
                summary["reason"] = "no_references_file"
            elif doi and doi in refs:
                summary["reason"] = "no_references_in_kb"
            else:
                summary["reason"] = "no_reference_list"
            return {**KnowledgeGraph().model_dump(), "paper": summary}

    all_docs = [d for docs in works.values() for d in docs]
    work_of_chunk = {
        c: w for w, docs in works.items() for _, chunks in docs for c in chunks
    }
    entity_hits = await _contributions(rag, all_docs, entities=True)
    members = {
        name: {work_of_chunk[c] for c in chunks} for name, chunks in entity_hits.items()
    }
    if mode == "shared":
        nodes = {n: w - {""} for n, w in members.items() if "" in w and len(w) > 1}
        summary["cited_sharing"] = len(set().union(*nodes.values())) if nodes else 0
        edge_docs = all_docs
    else:
        nodes = {n: set() for n, w in members.items() if "" in w}
        edge_docs = paper_docs
    relation_hits = await _contributions(rag, edge_docs, entities=False)
    pairs = []
    for rel_key in relation_hits:
        src, tgt = parse_relation_chunk_key(rel_key)
        if src in nodes and tgt in nodes:
            pairs.append((src, tgt))

    truncated = len(nodes) > max_nodes
    if truncated:
        degree: dict[str, int] = {}
        for src, tgt in pairs:
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1
        keep = sorted(nodes, key=lambda n: (-len(nodes[n]), -degree.get(n, 0), n))
        nodes = {n: nodes[n] for n in keep[:max_nodes]}
        pairs = [(s, t) for s, t in pairs if s in nodes and t in nodes]

    node_props = await rag.chunk_entity_relation_graph.get_nodes_batch(list(nodes))
    edge_props = await rag.chunk_entity_relation_graph.get_edges_batch(
        [{"src": s, "tgt": t} for s, t in pairs]
    )
    kg = KnowledgeGraph(is_truncated=truncated)
    for name, shared in nodes.items():
        props = node_props.get(name)
        if not props:
            continue
        props = dict(props)
        if shared:
            cites = sorted(
                zotero_citations.citation_short(f"{by_doi[w][0]}__", 80) for w in shared
            )
            more = len(cites) - _SHARED_WITH_MAX
            props["shared_with"] = "; ".join(cites[:_SHARED_WITH_MAX]) + (
                f"; and {more} more" if more > 0 else ""
            )
        kg.nodes.append(KnowledgeGraphNode(id=name, labels=[name], properties=props))
    shown = {n.id for n in kg.nodes}
    for src, tgt in pairs:
        props = edge_props.get((src, tgt))
        if props is None or src not in shown or tgt not in shown:
            continue
        kg.edges.append(
            KnowledgeGraphEdge(
                id=f"{src}-{tgt}",
                type="DIRECTED",
                source=src,
                target=tgt,
                properties=props,
            )
        )
    return {**kg.model_dump(), "paper": summary}
