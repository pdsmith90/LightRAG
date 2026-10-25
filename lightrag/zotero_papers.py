"""Papers of a Zotero-backed corpus as views of the knowledge graph.

The WebUI's paper filter shows the part of the graph a few papers contributed,
the part each shares with the library papers it cites, or the part at least two
of them share. A paper is named by its DOI or by the Zotero storage key of one
of its attachments; every processed document whose corpus file is
``<KEY>__<slug>.md`` for one of the paper's keys belongs to it (a key stands for
every attachment with the same DOI).

What a document contributed is read from LightRAG's per-document anchors:
``full_entities`` / ``full_relations`` list the names its extraction produced,
and a name counts only while ``entity_chunks`` / ``relation_chunks`` still cite
one of the document's chunks -- a prune or a merge can leave a name in the
anchors that no longer draws on the document. Those are key lookups, so they
work on every storage backend.

A paper's own relations leave a third of its entities unconnected, so a view
can add context: every relation the graph holds between the entities shown,
and entities the papers did not contribute that link several shown ones. On the
Postgres/AGE graph storage that is one plain-SQL query on the label tables;
other backends go through the generic graph storage calls.

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
# A view takes at most this many papers.
MAX_PAPERS = 10
# A linking entity touches at least _LINK_MIN shown entities; a view gets at most
# max(1, shown // _LINK_DIVISOR) of them, the ones touching the most first.
_LINK_MIN = 2
_LINK_DIVISOR = 4

_CONTEXT_EDGE = "Relation from other documents in the knowledge base"
_CONTEXT_NODE = "Not from the selected papers; linked to {k} shown entities"

# Every relation among a set of entities, and the entities outside it touching
# several of them, read straight from the AGE label tables: through the Cypher
# graph API the edge properties cost about 11 ms per pair on a 1M-edge graph.
# The names join the vertices the way get_nodes_batch does (entity_idx_node_id).
_AGE_NAMES = """WITH input AS (SELECT DISTINCT unnest($1::text[]) AS name),
v AS MATERIALIZED (
  SELECT b.id, i.name FROM "{g}".base b JOIN input i
    ON ag_catalog.agtype_access_operator(VARIADIC ARRAY[b.properties, '"entity_id"'::agtype])
       = (to_json(i.name)::text)::agtype)"""
_AGE_EDGES_SQL = (
    "-- zotero_papers:edges\n"
    + _AGE_NAMES
    + """
SELECT s.name AS src, t.name AS tgt, d.properties
FROM "{g}"."DIRECTED" d JOIN v s ON s.id = d.start_id JOIN v t ON t.id = d.end_id
WHERE s.id <> t.id"""
)
_AGE_LINKING_SQL = (
    "-- zotero_papers:linking\n"
    + _AGE_NAMES
    + """,
inc AS (
  SELECT d.end_id AS other, d.start_id AS inner_id FROM "{g}"."DIRECTED" d JOIN v ON v.id = d.start_id
  UNION ALL
  SELECT d.start_id, d.end_id FROM "{g}"."DIRECTED" d JOIN v ON v.id = d.end_id),
cand AS (
  SELECT other, count(DISTINCT inner_id) AS links FROM inc
  WHERE other NOT IN (SELECT id FROM v)
  GROUP BY other HAVING count(DISTINCT inner_id) >= $2)
SELECT name, links FROM (
  SELECT ag_catalog.agtype_access_operator(VARIADIC ARRAY[b.properties, '"entity_id"'::agtype])::text AS name,
         c.links
  FROM cand c JOIN "{g}".base b ON b.id = c.other) x
ORDER BY links DESC, name COLLATE "C" LIMIT $3"""
)

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


def resolve_paper(paper: str, by_doi: dict | None = None) -> tuple[str, list[str]]:
    """``(doi, storage keys)`` of the paper a DOI or a storage key names; the
    DOI is '' for a key without one, and both are empty for anything else.
    ``by_doi`` is the metadata's ``_keys_by_doi``, built here when omitted."""
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
    return doi, (_keys_by_doi(meta) if by_doi is None else by_doi).get(doi, [])


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


def _cite_key(keys: list[str], kb: dict) -> str:
    """The storage key a paper is cited by: its first one in the knowledge base."""
    return next((k for k in keys if k in kb), keys[0])


def _paper_summary(
    doi: str, keys: list[str], kb: dict, refs: dict, meta: dict, by_doi: dict
) -> dict:
    key = _cite_key(keys, kb)
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
    rag, docs: list[tuple[str, set[str]]], entities: bool, keep=None
) -> dict[str, set[str]]:
    """Entity name (or relation pair key) -> the chunk ids, among ``docs``',
    that it still cites. ``keep(name)`` false skips a name before its lookup."""
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
    names = [n for n in candidates if keep is None or keep(n)]
    found = await chunks_store.get_by_ids(names) if names else []
    out = {}
    for name, row in zip(names, found):
        hit = set((row or {}).get("chunk_ids") or []) & candidates[name]
        if hit:
            out[name] = hit
    return out


def _is_age(graph) -> bool:
    """The Postgres/AGE graph storage, duck-typed as entity_name_fold does."""
    return getattr(graph, "graph_name", None) is not None and hasattr(graph, "_query")


async def _graph_edges(graph, names: list[str], only=None) -> dict[str, dict]:
    """Relation key -> properties of each graph edge between two of ``names``;
    with ``only``, of those relation keys alone."""
    if _is_age(graph):
        rows = await graph._query(
            _AGE_EDGES_SQL.format(g=graph.graph_name), params={"names": names}
        )
        found: dict[str, dict] = {}
        for row in rows:
            key = make_relation_chunk_key(row["src"], row["tgt"])
            if only is not None and key not in only:
                continue
            # a pair stored both ways shows its reverse edge, as get_edges_batch does
            if key not in found or row["src"] > row["tgt"]:
                props = row["properties"]
                found[key] = json.loads(props) if isinstance(props, str) else props
        return found
    if only is not None:  # today's call for the papers' own relations
        pairs = [parse_relation_chunk_key(k) for k in only]
    else:
        inside = set(names)
        neighbours = await graph.get_nodes_edges_batch(names)
        pairs = sorted(
            {
                tuple(sorted(e))
                for edges in neighbours.values()
                for e in edges
                if e[0] != e[1] and e[0] in inside and e[1] in inside
            }
        )
    props = await graph.get_edges_batch([{"src": s, "tgt": t} for s, t in pairs])
    return {
        make_relation_chunk_key(s, t): props[(s, t)]
        for s, t in pairs
        if props.get((s, t)) is not None
    }


async def _linking_nodes(graph, names: list[str], budget: int) -> dict[str, int]:
    """Up to ``budget`` entities outside ``names`` that touch at least _LINK_MIN
    of them -> how many they touch; the ones touching the most first, then by
    name."""
    if _is_age(graph):
        rows = await graph._query(
            _AGE_LINKING_SQL.format(g=graph.graph_name),
            params={"names": names, "min_links": _LINK_MIN, "limit": budget},
        )
        return {row["name"]: int(row["links"]) for row in rows}
    inside = set(names)
    touched: dict[str, set[str]] = {}
    for name, edges in (await graph.get_nodes_edges_batch(names)).items():
        for src, tgt in edges:
            other = tgt if src == name else src
            if other not in inside:
                touched.setdefault(other, set()).add(name)
    ranked = sorted(
        (o for o, by in touched.items() if len(by) >= _LINK_MIN),
        key=lambda o: (-len(touched[o]), o),
    )
    return {o: len(touched[o]) for o in ranked[:budget]}


def _cite_list(works, key_of: dict[str, str]) -> str:
    cites = sorted(zotero_citations.citation_short(f"{key_of[w]}__", 80) for w in works)
    more = len(cites) - _SHARED_WITH_MAX
    return "; ".join(cites[:_SHARED_WITH_MAX]) + (
        f"; and {more} more" if more > 0 else ""
    )


def _view(view: dict, summaries: list[dict], reason: str | None = None) -> dict:
    """The result's ``paper``: one paper keeps today's summary fields at the top
    level; ``papers`` lists every requested paper's summary."""
    paper = dict(summaries[0]) if len(summaries) == 1 else {}
    paper.update(view, papers=summaries)
    if reason:
        paper["reason"] = reason
    return paper


async def paper_graph(
    rag,
    papers: str | list[str],
    mode: str = "paper",
    max_nodes: int = 1000,
    context: str = "own",
) -> dict:
    """The graph a few papers contributed (``mode="paper"``), the part each
    shares with the library papers it cites (``mode="shared"``), or the part at
    least two of them contributed (``mode="common"``).

    Nodes are the entities the papers' documents still draw on. In the shared
    view only those that a cited paper's documents draw on too, each listing
    those papers in its ``shared_with`` property; a selected paper cited by
    another counts as a reference. With several papers each node names the ones
    that contributed it in ``contributed_by``. Over ``max_nodes``, the nodes
    shared with the most papers and then the best connected are kept.

    Edges are the relations the shown documents contributed between two shown
    nodes (``context="own"``). ``context="kb"`` adds every other relation the
    graph holds between two shown nodes, and ``"linked"`` also entities the
    papers did not contribute that touch several shown ones (none when the view
    was truncated), with their relations to shown nodes. What the papers did not
    contribute carries a ``paper_context`` property.

    ``paper`` in the result: ``papers``, the summary of each requested paper (a
    paper not in the knowledge base carries ``reason: "not_in_kb"``), and the
    view's counts; for a single paper its summary fields too, as before. An
    empty graph carries ``paper["reason"]``: ``too_many_papers`` (over
    MAX_PAPERS), ``not_in_kb``, ``need_two_papers`` / ``nothing_in_common``
    (common view), or, in the shared view, ``no_references_file``,
    ``no_reference_list`` (no DOI, or OpenAlex lists no references) or
    ``no_references_in_kb``.
    """
    meta = zotero_citations._load()
    by_doi = _keys_by_doi(meta)
    resolved: dict[str, tuple[str, list[str], str]] = {}  # id -> (DOI, keys, as asked)
    for asked in [papers] if isinstance(papers, str) else papers:
        doi, keys = resolve_paper(asked, by_doi)
        resolved.setdefault(doi or (keys[0] if keys else asked), (doi, keys, asked))
    refs = _load_references()
    view = dict(
        mode=mode,
        context=context,
        references_available=bool(refs),
        cited_sharing=0,
        n_linking=0,
        n_context_edges=0,
    )
    empty = KnowledgeGraph().model_dump()
    if len(resolved) > MAX_PAPERS:
        return {**empty, "paper": _view(view, [], "too_many_papers")}

    kb = await _kb_documents(rag)
    summaries, selected = [], {}  # selected: paper id -> its documents
    for paper, (doi, keys, asked) in resolved.items():
        summary = (
            _paper_summary(doi, keys, kb, refs, meta, by_doi)
            if keys
            else {"paper": asked}
        )
        docs = [d for k in keys for d in kb.get(k, [])]
        if docs:
            selected[paper] = docs
        else:
            summary["reason"] = "not_in_kb"
        summaries.append(summary)
    if not selected:
        return {**empty, "paper": _view(view, summaries, "not_in_kb")}
    if mode == "common" and len(selected) < 2:
        return {**empty, "paper": _view(view, summaries, "need_two_papers")}

    # works: the selected papers, and in the shared view the papers they cite
    works: dict[str, list[tuple[str, set[str]]]] = dict(selected)
    # cited by the key their summary (the chip) uses
    key_of = {p: _cite_key(resolved[p][1], kb) for p in selected}
    cites: dict[str, set[str]] = {}
    if mode == "shared":
        for paper in selected:
            doi = resolved[paper][0]
            cites[paper] = set()
            for cited in ((refs.get(doi) or {}).get("cited") or []) if doi else []:
                cited_docs = [d for k in by_doi.get(cited, []) for d in kb.get(k, [])]
                if cited_docs:
                    works.setdefault(cited, cited_docs)
                    key_of.setdefault(cited, by_doi[cited][0])
                    cites[paper].add(cited)
        if not any(cites.values()):
            if not refs:
                reason = "no_references_file"
            elif any(resolved[p][0] in refs for p in selected):
                reason = "no_references_in_kb"
            else:
                reason = "no_reference_list"
            return {**empty, "paper": _view(view, summaries, reason)}

    work_of_chunk = {
        c: w for w, docs in works.items() for _, chunks in docs for c in chunks
    }
    hits = await _contributions(
        rag, [d for docs in selected.values() for d in docs], entities=True
    )
    if mode == "shared":
        # only a name a selected paper contributed can be shown: the cited
        # papers' anchors are checked for those names alone
        cited_docs = [d for w, docs in works.items() if w not in selected for d in docs]
        cited_hits = await _contributions(
            rag, cited_docs, entities=True, keep=hits.__contains__
        )
        for name, chunks in cited_hits.items():
            hits[name] |= chunks
    members = {
        name: {work_of_chunk[c] for c in chunks} for name, chunks in hits.items()
    }
    chosen = set(selected)
    if mode == "shared":
        nodes = {}  # name -> the cited papers it is shared with
        for name, ws in members.items():
            shared = set().union(*(ws & cites[p] for p in ws & chosen))
            if shared:
                nodes[name] = shared
        view["cited_sharing"] = len(set().union(*nodes.values())) if nodes else 0
    else:  # name -> the selected papers that contributed it
        nodes = {n: ws & chosen for n, ws in members.items() if ws & chosen}
        if mode == "common":
            nodes = {n: ws for n, ws in nodes.items() if len(ws) > 1}
            if not nodes:
                return {**empty, "paper": _view(view, summaries, "nothing_in_common")}

    own = await _contributions(
        rag,
        [d for docs in works.values() for d in docs],
        entities=False,
        keep=lambda k: all(e in nodes for e in parse_relation_chunk_key(k)),
    )
    truncated = len(nodes) > max_nodes
    if truncated:
        degree: dict[str, int] = {}
        for src, tgt in map(parse_relation_chunk_key, own):
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1
        keep = sorted(nodes, key=lambda n: (-len(nodes[n]), -degree.get(n, 0), n))
        nodes = {n: nodes[n] for n in keep[:max_nodes]}
    own_keys = {k for k in own if all(e in nodes for e in parse_relation_chunk_key(k))}

    graph = rag.chunk_entity_relation_graph
    linking: dict[str, int] = {}
    if context == "linked" and not truncated:
        budget = min(max(1, len(nodes) // _LINK_DIVISOR), max_nodes - len(nodes))
        if budget > 0:
            linking = await _linking_nodes(graph, list(nodes), budget)
    names = list(nodes) + list(linking)
    edges = await _graph_edges(
        graph, names, only=own_keys if context == "own" else None
    )
    node_props = await graph.get_nodes_batch(names)

    kg = KnowledgeGraph(is_truncated=truncated)
    for name in names:
        props = node_props.get(name)
        if not props:
            continue
        props = dict(props)
        if name in linking:
            props["paper_context"] = _CONTEXT_NODE.format(k=linking[name])
        else:
            if mode == "shared":
                props["shared_with"] = _cite_list(nodes[name], key_of)
            if len(selected) > 1:
                props["contributed_by"] = _cite_list(members[name] & chosen, key_of)
        kg.nodes.append(KnowledgeGraphNode(id=name, labels=[name], properties=props))
    shown = {n.id for n in kg.nodes}
    view["n_linking"] = len(shown & set(linking))
    for key in sorted(edges):
        src, tgt = parse_relation_chunk_key(key)
        if src not in shown or tgt not in shown:
            continue
        props = edges[key]
        if key not in own_keys:
            if src in linking and tgt in linking:
                continue
            props = {**props, "paper_context": _CONTEXT_EDGE}
            view["n_context_edges"] += 1
        kg.edges.append(
            KnowledgeGraphEdge(
                id=f"{src}-{tgt}",
                type="DIRECTED",
                source=src,
                target=tgt,
                properties=props,
            )
        )
    return {**kg.model_dump(), "paper": _view(view, summaries)}
