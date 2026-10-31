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

Paper search folds typography on both sides (:func:`fold`) and reads a BibTeX
key as its words; :func:`paper_passages` answers a question from one paper's
own chunks, reranked against the bare question.

Fork-local (no upstream counterpart).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import unicodedata

import numpy as np

from lightrag import zotero_citations
from lightrag.base import DocStatus
from lightrag.chunk_schema import format_parent_headings
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import (
    apply_rerank_if_enabled,
    filter_bibliography_chunks,
    make_relation_chunk_key,
    parse_relation_chunk_key,
)

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

# pending: (doc_status, task) of the one refresh in flight
_docs_cache: dict = {"at": 0.0, "owner": None, "by_key": {}, "pending": None}

_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")

# A keys= search takes at most this many storage keys.
MAX_SEARCH_KEYS = 50
# paper_passages: a work of at most _PASSAGE_ALL chunks offers every chunk to the
# reranker; a longer one the _PASSAGE_LEG best by vector similarity and the
# _PASSAGE_LEG best by full-text rank (at most _PASSAGE_MAX together).
MAX_PASSAGE_KEYS = 10
_PASSAGE_ALL = 60
_PASSAGE_LEG = 40
_PASSAGE_MAX = 80

# fold(): letters NFKD leaves whole, dashes, and apostrophes / primes (deleted).
_FOLD_TABLE = str.maketrans(
    {
        "\u0131": "i",
        "\u0142": "l",
        "\u0141": "L",
        "\u00f8": "o",
        "\u00d8": "O",
        "\u0111": "d",
        "\u0110": "D",
        "\u00f0": "d",
        "\u00d0": "D",
        "\u00df": "ss",
        "\u1e9e": "SS",
        "\u00e6": "ae",
        "\u00c6": "AE",
        "\u0153": "oe",
        "\u0152": "OE",
        "\u00fe": "th",
        "\u00de": "TH",
        **{chr(c): "-" for c in range(0x2010, 0x2016)},
        "\u2212": "-",
        "\u00ad": "-",
        **{c: None for c in "'\u2019\u2018\u02bc\u2032`"},
    }
)
# stripped from both ends of a query token: brackets, sentence punctuation, '*'
# (emphasis) and typographic double quotes and guillemets
_EDGE_PUNCT = '()[]{},.;:"!?*\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a'
_BIBKEY_RE = re.compile(r"^([a-z][a-z]*)((?:[A-Z][A-Za-z0-9]*)+)(\d{4})[a-z]?$")
# camel-case words, acronym runs ("ABCFollow" -> ABC, Follow) and digit runs
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_NON_ALNUM_RE = re.compile(r"[\W_]+")
_PART_SPLIT_RE = re.compile(r"[-/]")


def fold(text: str) -> str:
    """``text`` for matching: NFKD without combining marks, the letters NFKD
    keeps whole spelled in ASCII (ı ł ø đ ð ß æ œ þ), every dash and the soft
    hyphen as '-', apostrophes and primes deleted, lowercased."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.translate(_FOLD_TABLE).lower()


def _query_tokens(query: str) -> list[str]:
    """The whitespace-separated tokens of a paper search, each without its edge
    punctuation, and without the "et al" that follows a name in a citation
    ("Doe et al. 2011"). The MCP proxy carries the same rule (``_tokens``)."""
    raw = [t for t in (tok.strip(_EDGE_PUNCT) for tok in (query or "").split()) if t]
    tokens: list[str] = []
    i = 0
    while i < len(raw):
        if (
            tokens
            and i + 1 < len(raw)
            and fold(raw[i]) == "et"
            and fold(raw[i + 1]) == "al"
        ):
            i += 2
            continue
        tokens.append(raw[i])
        i += 1
    return tokens


def query_words(query: str) -> list[tuple[str, bool]]:
    """The words of a paper search, folded, as ``(word, is_title_word)``.

    Each token loses its leading and trailing punctuation (brackets, ``*``,
    straight and curly quotes, guillemets) and an "et al" after a name is
    dropped; a token with no letter or digit left (the " - " of "Author - Year -
    Title", an "&") is no word. A token shaped like a BibTeX key
    (``doePruningOrchardsAppleTrees2001``) becomes its family name, each
    camel-case title word and its year; the title words match the haystack with
    everything but letters and digits removed (``is_title_word``). A word
    matches as it is, without its hyphens, or by all its '-' and '/' parts
    (:func:`_word_in`) -- except in a DOI-shaped query, matched whole. The proxy
    carries the same rule; ``zotero-toolkit/tests/mcp/paper_search_cases.json``
    holds the cases both suites run."""
    words: list[tuple[str, bool]] = []
    for token in _query_tokens(query):
        m = _BIBKEY_RE.match(token)
        if m:
            family, title, year = m.groups()
            words.append((fold(family), False))
            words.extend((fold(w), True) for w in _CAMEL_RE.findall(title))
            words.append((year, False))
            continue
        word = fold(token).strip(_EDGE_PUNCT)
        if any(ch.isalnum() for ch in word):  # not a bare "-" between fields
            words.append((word, False))
    return words


def _forms(text: str) -> tuple[str, str, str]:
    """A haystack folded, without hyphens, and with only letters and digits."""
    folded = fold(text)
    return folded, folded.replace("-", ""), _NON_ALNUM_RE.sub("", folded)


def _word_in(
    word: str, title_word: bool, forms: tuple[str, str, str], loose: bool = True
) -> bool:
    """Whether a :func:`query_words` word occurs in a :func:`_forms` haystack:
    as it is, or (``loose``) without its hyphens in the hyphen-less haystack or
    by every part of it split on '-' or '/'."""
    folded, bare_hay, alnum_hay = forms
    if title_word:
        return word in alnum_hay
    if word in folded:
        return True
    if not loose:
        return False
    bare = word.replace("-", "")
    if bare and bare in bare_hay:
        return True
    parts = [p for p in _PART_SPLIT_RE.split(word) if p]
    return len(parts) > 1 and all(p in folded for p in parts)


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


async def _load_kb_documents(doc_status) -> dict[str, list[tuple[str, set[str]]]]:
    docs = await doc_status.get_docs_by_statuses([DocStatus.PROCESSED])
    by_key: dict[str, list[tuple[str, set[str]]]] = {}
    for doc_id, status in docs.items():
        key = zotero_citations.zotero_key(status.file_path or "")
        if key:
            by_key.setdefault(key, []).append((doc_id, set(status.chunks_list or [])))
    _docs_cache.update(at=time.monotonic(), owner=doc_status, by_key=by_key)
    return by_key


def _refresh_done(task: asyncio.Task) -> None:
    pending = _docs_cache.get("pending")
    if pending is not None and pending[1] is task:
        _docs_cache["pending"] = None
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            f"zotero_papers: reading the processed documents failed: {task.exception()}"
        )


async def _kb_documents(rag) -> dict[str, list[tuple[str, set[str]]]]:
    """Storage key -> [(doc id, chunk ids)] of every processed document.

    The list is read at most once per _DOCS_TTL, and concurrent callers share
    one read. Once it is older than that, callers get it at once while one
    refresh runs behind them; only the first read is waited for."""
    cache = _docs_cache
    owner = rag.doc_status
    known = cache["owner"] is owner
    if known and time.monotonic() - cache["at"] < _DOCS_TTL:
        return cache["by_key"]
    loop = asyncio.get_running_loop()
    pending = cache.get("pending")
    if pending is None or pending[0] is not owner or pending[1].get_loop() is not loop:
        task = loop.create_task(_load_kb_documents(owner))
        task.add_done_callback(_refresh_done)
        pending = cache["pending"] = (owner, task)
    if known:
        return cache["by_key"]
    return await asyncio.shield(pending[1])


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


_index_lock = threading.Lock()
# the folded haystacks of the indexed papers, rebuilt when the processed-document
# list or the metadata is replaced
_index: dict = {"kb": None, "meta": None, "rows": []}


def _search_index(kb: dict, meta: dict) -> list[tuple]:
    """(key, DOI, paper, year, title forms, haystack forms) per indexed key."""
    with _index_lock:
        if _index["kb"] is kb and _index["meta"] is meta:
            return _index["rows"]
        rows = []
        for key in kb:
            entry = meta.get(key)
            entry = entry if isinstance(entry, dict) else {}
            doi = norm_doi(entry.get("doi") or "")
            title = zotero_citations._plain_title(entry) if entry else ""
            year = str(entry.get("year") or "")
            hay = " ".join(
                [title, " ".join(entry.get("authors") or []), year, doi, key.lower()]
            )
            rows.append((key, doi, doi or key, year, _forms(title), _forms(hay)))
        _index.update(kb=kb, meta=meta, rows=rows)
        return rows


def _search(kb: dict, query: str, words: list, wanted: set, limit: int) -> list[dict]:
    meta = zotero_citations._load()
    rows = _search_index(kb, meta)
    doi_q = norm_doi(query)
    key_q = query.strip().upper()
    chosen = {row[2] for row in rows if row[0] in wanted}
    best: dict[str, tuple] = {}  # paper (DOI, else key) -> best rank
    keys_of: dict[str, list[str]] = {}
    for key, doi, paper, year, title_forms, hay_forms in rows:
        if wanted and paper not in chosen:
            continue
        named = bool(doi_q and doi == doi_q) or key == key_q
        # a DOI is matched whole: its parts ("10.1", "x") are in most haystacks
        if (
            words
            and not named
            and not all(_word_in(w, t, hay_forms, not doi_q) for w, t in words)
        ):
            continue
        rank = (
            not (named or key in wanted),
            -sum(_word_in(w, t, title_forms) for w, t in words),
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


async def search_papers(
    rag, query: str, limit: int = 20, keys: list[str] | None = None
) -> list[dict]:
    """Papers in the knowledge base whose DOI, storage key, title, authors or
    year match ``query`` (every word must occur, both sides folded, a BibTeX
    key read as its words), best match first. With ``keys`` (storage keys),
    only the papers carrying one of them, and ``query`` may be empty."""
    words = query_words(query)
    wanted = {k.strip().upper() for k in keys or [] if k.strip()}
    if not words and not wanted:
        return []
    kb = await _kb_documents(rag)
    # the scan and the summaries are plain Python over every indexed paper
    return await asyncio.to_thread(_search, kb, query or "", words, wanted, limit)


def _cosines(query_vector, vectors: dict) -> dict[str, float]:
    ids = list(vectors)
    if not ids:
        return {}
    matrix = np.asarray([vectors[i] for i in ids], dtype=np.float64)
    q = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    dots = matrix @ q
    sims = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)
    return {i: float(s) for i, s in zip(ids, sims)}


async def _vector_scores(
    rag, question: str, ids: list[str], limit: int | None = None
) -> dict[str, float]:
    """Chunk id -> cosine similarity to the question, for the ``ids`` that have
    a stored vector. The ranking covers the given chunks alone (a vector-index
    query cannot be kept to one work). A storage with ``rank_ids_by_vector``
    (Postgres) ranks them where they are stored and returns only the ``limit``
    nearest, so the cost of the call does not grow with the length of the work;
    another storage returns every vector by id and all the scores."""
    vdb = rag.chunks_vdb
    embedded = await vdb.embedding_func(
        [question], context="query", _priority=DEFAULT_QUERY_PRIORITY
    )
    rank = getattr(vdb, "rank_ids_by_vector", None)
    if rank is not None:
        return await rank(ids, embedded[0], limit or len(ids))
    vectors = await vdb.get_vectors_by_ids(ids)
    return await asyncio.to_thread(_cosines, embedded[0], vectors)


def _passage_candidates(ids, rows, key_of, drop_bibliography) -> list[dict]:
    """The rows as reranker candidates: reference lists out when the filter is
    on, and a text that two copies of the paper share only once."""
    candidates, seen = [], set()
    for chunk_id, row in zip(ids, rows):
        if not isinstance(row, dict):
            continue
        content = row.get("content") or ""
        if content in seen:
            continue
        seen.add(content)
        candidates.append(
            {
                "key": key_of[chunk_id],
                "file_path": row.get("file_path") or "",
                "chunk_id": chunk_id,
                "chunk_order_index": int(row.get("chunk_order_index") or 0),
                "content": content,
                "content_headings": format_parent_headings(row) or None,
            }
        )
    if drop_bibliography:
        candidates = filter_bibliography_chunks(candidates)
    return candidates


async def paper_passages(rag, keys: list[str], question: str, top_n: int = 3) -> dict:
    """The ``top_n`` passages of one work that best answer ``question``.

    ``keys`` are the storage keys of the work's copies. The answer's ``keys``
    are those in the knowledge base and ``n_chunks`` is the number of chunks of
    the work's longest copy (what ``chunk_order_index`` counts within).
    A work of at most _PASSAGE_ALL chunks offers all of them; a longer one the
    _PASSAGE_LEG best by cosine similarity of their stored vectors to the
    question and the _PASSAGE_LEG best by the text storage's
    ``rank_chunks_by_text`` (Postgres full-text rank; skipped on a storage
    without it), both over the work's own chunk ids. Reference lists drop out
    under DROP_BIBLIOGRAPHY_CHUNKS, and the server's reranker orders the rest
    against the bare question; the first chunk is a candidate like any other.
    Without a reranker, or when it fails, vector similarity orders them."""
    kb = await _kb_documents(rag)
    indexed = [k for k in dict.fromkeys(keys) if k in kb]
    key_of: dict[str, str] = {}
    for key in indexed:
        for _, chunks in kb[key]:
            for chunk_id in chunks:
                key_of.setdefault(chunk_id, key)
    ids = sorted(key_of)
    # the paper's length is that of its longest copy: copies are not added up,
    # so a passage's "chunk i" is at most n_chunks
    n_chunks = max((len(c) for k in indexed for _, c in kb[k]), default=0)
    out = {"keys": indexed, "n_chunks": n_chunks, "passages": []}
    if not ids:
        return out

    if len(ids) <= _PASSAGE_ALL:
        chosen = ids
    else:
        scores = await _vector_scores(rag, question, ids, _PASSAGE_LEG)
        by_vector = sorted(scores, key=lambda c: -scores[c])[:_PASSAGE_LEG]
        rank_text = getattr(rag.text_chunks, "rank_chunks_by_text", None)
        by_text = await rank_text(ids, question, _PASSAGE_LEG) if rank_text else []
        chosen = list(dict.fromkeys(by_vector + list(by_text)))[:_PASSAGE_MAX]
    rows = await rag.text_chunks.get_by_ids(chosen)
    candidates = await asyncio.to_thread(
        _passage_candidates,
        chosen,
        rows,
        key_of,
        bool(getattr(rag, "drop_bibliography_chunks", False)),
    )
    if not candidates:
        return out

    ranked: list[tuple[dict, float]] | None = None
    rerank_func = getattr(rag, "rerank_model_func", None)
    how = "no reranker configured"
    if rerank_func is not None:
        reranked = await apply_rerank_if_enabled(
            query=question,
            retrieved_docs=candidates,
            global_config={"rerank_model_func": rerank_func},
            enable_rerank=True,
            top_n=top_n,
        )
        if reranked and all("rerank_score" in c for c in reranked):
            ranked = sorted(
                ((c, float(c["rerank_score"])) for c in reranked),
                key=lambda pair: -pair[1],
            )
            how = "reranked"
        else:
            how = "reranking failed"
    if ranked is None:
        # the first scores cover only the vector leg's leaders, not the text leg's
        scores = await _vector_scores(
            rag, question, [c["chunk_id"] for c in candidates]
        )
        ranked = sorted(
            ((c, scores.get(c["chunk_id"], -1.0)) for c in candidates),
            key=lambda pair: -pair[1],
        )
        how += ", in vector order"
    level = logging.WARNING if "failed" in how else logging.INFO
    logger.log(
        level,
        f"Paper passages: {len(candidates)} candidates of {len(ids)} chunks "
        f"({', '.join(indexed)}), {how}",
    )
    out["passages"] = [
        {
            "key": c["key"],
            "file_path": c["file_path"],
            "chunk_id": c["chunk_id"],
            "chunk_order_index": c["chunk_order_index"],
            "content": c["content"],
            "content_headings": c["content_headings"],
            "score": score,
        }
        for c, score in ranked[:top_n]
    ]
    return out


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
