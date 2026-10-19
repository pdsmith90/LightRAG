"""The WebUI's paper filter: the graph one paper contributed, and the part it
shares with the library papers it cites (lightrag/zotero_papers.py).

A paper is named by DOI or by a storage key; its documents are the processed
ones whose corpus file starts with one of its keys. Its entities and relations
come from the per-document anchors, kept only while the chunk-tracking stores
still cite one of its chunks. References come from zotero_references.json.
Fixtures are invented.
"""

import json

import pytest

import lightrag.zotero_citations as zc
import lightrag.zotero_papers as zp
from lightrag.base import DocStatus
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline

META = {
    "PPPP0001": {
        "title": "Mascon leakage over Antarctica",
        "authors": ["Okafor Chidi"],
        "year": "2019",
        "doi": "10.1/P",
    },
    "PPPP0002": {
        "title": "Mascon leakage over Antarctica",
        "authors": ["Okafor Chidi"],
        "year": "2019",
        "doi": "https://doi.org/10.1/p",
    },
    "RRRR0001": {
        "title": "Gravity recovery basics",
        "authors": ["Lindqvist Maja"],
        "year": "2004",
        "doi": "10.1/r1",
    },
    "RRRR0002": {
        "title": "Not ingested",
        "authors": ["Varga Tamás"],
        "year": "2001",
        "doi": "10.1/r2",
    },
    "UUUU0001": {
        "title": "Unrelated leakage study",
        "authors": ["Nakamura Ren"],
        "year": "2021",
        "doi": "10.1/u",
    },
    "KKKK0001": {
        "title": "A report without a DOI",
        "authors": ["Lee Min"],
        "year": "1999",
    },
}

REFERENCES = {
    "10.1/p": {"n_refs": 40, "cited": ["10.1/r1", "10.1/r2"]},
    "10.1/r1": {"n_refs": 3, "cited": []},
    "10.1/u": {"n_refs": 9, "cited": ["10.1/p"]},
}

# doc id -> (corpus file, chunk ids)
DOCS = {
    "doc-p1": ("PPPP0001__Mascon_leakage.md", ["p1-c0", "p1-c1"]),
    "doc-p2": ("PPPP0002__Mascon_leakage.md", ["p2-c0"]),
    "doc-r1": ("RRRR0001__Gravity_recovery.md", ["r1-c0"]),
    "doc-u": ("UUUU0001__Unrelated.md", ["u-c0"]),
    "doc-k": ("KKKK0001__A_report.md", ["k-c0"]),
}

FULL_ENTITIES = {
    # "Stale" is still listed but no longer cites doc-p1; "Gone" was pruned
    "doc-p1": ["GRACE", "Mascon", "Leakage", "Stale", "Gone"],
    "doc-p2": ["Filter"],
    "doc-r1": ["GRACE", "Mascon", "Other"],
    "doc-u": ["Leakage"],
    "doc-k": ["Report"],
}
ENTITY_CHUNKS = {
    "GRACE": ["p1-c0", "r1-c0"],
    "Mascon": ["p1-c1", "r1-c0"],
    "Leakage": ["p1-c1", "u-c0"],
    "Stale": ["u-c0"],
    "Filter": ["p2-c0"],
    "Other": ["r1-c0"],
    "Report": ["k-c0"],
}
FULL_RELATIONS = {
    "doc-p1": [["GRACE", "Mascon"], ["Leakage", "Mascon"], ["GRACE", "Gone"]],
    "doc-p2": [["Filter", "Mascon"]],
    "doc-r1": [["GRACE", "Mascon"], ["GRACE", "Other"]],
    "doc-u": [],
}
RELATION_CHUNKS = {
    make_relation_chunk_key("GRACE", "Mascon"): ["p1-c0", "r1-c0"],
    make_relation_chunk_key("Leakage", "Mascon"): ["p1-c1"],
    make_relation_chunk_key("Filter", "Mascon"): ["p2-c0"],
    make_relation_chunk_key("GRACE", "Other"): ["r1-c0"],
}


class _Status:
    def __init__(self, file_path, chunks):
        self.file_path, self.chunks_list = file_path, chunks


class _DocStatus:
    def __init__(self):
        self.calls = 0

    async def get_docs_by_statuses(self, statuses):
        assert statuses == [DocStatus.PROCESSED]
        self.calls += 1
        return {d: _Status(f, c) for d, (f, c) in DOCS.items()}


class _KV:
    def __init__(self, rows):
        self.rows = rows

    async def get_by_ids(self, ids):
        return [self.rows.get(i) for i in ids]


class _Graph:
    async def get_nodes_batch(self, names):
        return {
            n: {"entity_id": n, "entity_type": "concept"} for n in names if n != "Gone"
        }

    async def get_edges_batch(self, pairs):
        return {(p["src"], p["tgt"]): {"keywords": "relates"} for p in pairs}


class _Rag:
    max_graph_nodes = 1000

    def __init__(self):
        self.doc_status = _DocStatus()
        self.full_entities = _KV(
            {d: {"entity_names": n} for d, n in FULL_ENTITIES.items()}
        )
        self.entity_chunks = _KV(
            {n: {"chunk_ids": c} for n, c in ENTITY_CHUNKS.items()}
        )
        self.full_relations = _KV(
            {d: {"relation_pairs": p} for d, p in FULL_RELATIONS.items()}
        )
        self.relation_chunks = _KV(
            {k: {"chunk_ids": c} for k, c in RELATION_CHUNKS.items()}
        )
        self.chunk_entity_relation_graph = _Graph()


def _reset_refs(path):
    zp.REFERENCES_PATH = path
    zp._refs, zp._refs_mtime, zp._refs_next_stat, zp._refs_warned = {}, None, 0.0, False
    zp._docs_cache.update(at=0.0, owner=None, by_key={})


@pytest.fixture
def library(tmp_path):
    meta_path = tmp_path / "zotero_metadata.json"
    meta_path.write_text(json.dumps(META), encoding="utf-8")
    refs_path = tmp_path / "zotero_references.json"
    refs_path.write_text(json.dumps(REFERENCES), encoding="utf-8")
    saved_meta, saved_refs = zc.METADATA_PATH, zp.REFERENCES_PATH
    zc.METADATA_PATH = str(meta_path)
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    _reset_refs(str(refs_path))
    yield refs_path
    zc.METADATA_PATH = saved_meta
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    _reset_refs(saved_refs)


def _nodes(result):
    return sorted(n["id"] for n in result["nodes"])


def _edges(result):
    return sorted(tuple(sorted((e["source"], e["target"]))) for e in result["edges"])


async def test_paper_view_keeps_what_the_paper_still_contributes(library):
    result = await zp.paper_graph(_Rag(), "10.1/P")
    # both attachments of the DOI count; Stale and Gone are anchors only
    assert _nodes(result) == ["Filter", "GRACE", "Leakage", "Mascon"]
    assert _edges(result) == [
        ("Filter", "Mascon"),
        ("GRACE", "Mascon"),
        ("Leakage", "Mascon"),
    ]
    assert result["paper"]["keys"] == ["PPPP0001", "PPPP0002"]
    assert result["paper"]["n_cited_in_kb"] == 1
    assert not result["is_truncated"]
    assert all("shared_with" not in n["properties"] for n in result["nodes"])


async def test_a_storage_key_names_every_attachment_of_its_doi(library):
    by_key = await zp.paper_graph(_Rag(), "pppp0002")
    by_doi = await zp.paper_graph(_Rag(), "https://doi.org/10.1/p")
    assert _nodes(by_key) == _nodes(by_doi) == ["Filter", "GRACE", "Leakage", "Mascon"]


async def test_shared_view_keeps_nodes_a_cited_paper_also_draws_on(library):
    result = await zp.paper_graph(_Rag(), "10.1/p", mode="shared")
    # Leakage is shared only with a paper p does not cite; Other only with r1
    assert _nodes(result) == ["GRACE", "Mascon"]
    assert _edges(result) == [("GRACE", "Mascon")]
    for node in result["nodes"]:
        assert (
            node["properties"]["shared_with"]
            == "Lindqvist Maja. (2004). Gravity recovery basics"
        )
    assert result["paper"]["cited_sharing"] == 1


@pytest.mark.parametrize(
    "paper, reason",
    [
        ("10.1/r1", "no_references_in_kb"),
        ("KKKK0001", "no_reference_list"),
        ("10.1/r2", "not_in_kb"),
        ("not a paper", "not_in_kb"),
    ],
)
async def test_empty_shared_view_says_why(library, paper, reason):
    result = await zp.paper_graph(_Rag(), paper, mode="shared")
    assert result["nodes"] == [] and result["edges"] == []
    assert result["paper"]["reason"] == reason


async def test_shared_view_without_a_references_file(library):
    library.unlink()
    _reset_refs(str(library))
    result = await zp.paper_graph(_Rag(), "10.1/p", mode="shared")
    assert result["nodes"] == []
    assert result["paper"]["references_available"] is False
    assert result["paper"]["reason"] == "no_references_file"
    # the paper view does not need the file
    assert _nodes(await zp.paper_graph(_Rag(), "10.1/p")) == [
        "Filter",
        "GRACE",
        "Leakage",
        "Mascon",
    ]


async def test_truncation_keeps_the_best_connected_nodes(library):
    result = await zp.paper_graph(_Rag(), "10.1/p", max_nodes=2)
    assert result["is_truncated"]
    # Mascon has three edges; the one-edge nodes tie and go by name
    assert _nodes(result) == ["Filter", "Mascon"]
    assert _edges(result) == [("Filter", "Mascon")]


async def test_search_finds_papers_in_the_knowledge_base_only(library):
    rag = _Rag()
    hits = await zp.search_papers(rag, "leakage")
    # one result per DOI, the title match first; r2 is not ingested
    assert [h["paper"] for h in hits] == ["10.1/u", "10.1/p"]
    assert hits[1]["keys"] == ["PPPP0001", "PPPP0002"]
    assert hits[1]["n_refs"] == 40 and hits[1]["n_cited_in_kb"] == 1
    assert [h["paper"] for h in await zp.search_papers(rag, "10.1/R1")] == ["10.1/r1"]
    assert [h["paper"] for h in await zp.search_papers(rag, "kkkk0001")] == ["KKKK0001"]
    assert await zp.search_papers(rag, "not ingested") == []
    assert await zp.search_papers(rag, "okafor 2019 antarctica") != []
    # the processed-document list is reused between searches
    assert rag.doc_status.calls == 1
