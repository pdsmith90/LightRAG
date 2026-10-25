"""The WebUI's paper filter: the graph a few papers contributed, the part each
shares with the library papers it cites, or the part they have in common, with
or without context from the rest of the graph (lightrag/zotero_papers.py).

A paper is named by DOI or by a storage key; its documents are the processed
ones whose corpus file starts with one of its keys. Its entities and relations
come from the per-document anchors, kept only while the chunk-tracking stores
still cite one of its chunks. References come from zotero_references.json.
Fixtures are invented.
"""

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lightrag.zotero_citations as zc
import lightrag.zotero_papers as zp
from lightrag.base import DocStatus
from lightrag.utils import make_relation_chunk_key

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

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
# The graph's directed edges: the relations above (GRACE-Mascon stored in both
# directions), Leakage-GRACE that no document of these papers contributed, and
# entities outside every paper: Hub3 touches three of p's entities, Hub2 two,
# Lone one; Hub3-Hub2 joins two outsiders.
EDGES = [
    ("GRACE", "Mascon"),
    ("Mascon", "GRACE"),
    ("Leakage", "Mascon"),
    ("Filter", "Mascon"),
    ("GRACE", "Other"),
    ("Leakage", "GRACE"),
    ("Hub3", "Filter"),
    ("GRACE", "Hub3"),
    ("Hub3", "Leakage"),
    ("Hub2", "GRACE"),
    ("Mascon", "Hub2"),
    ("Lone", "GRACE"),
    ("Hub3", "Hub2"),
]


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
        self.asked: list[str] = []

    async def get_by_ids(self, ids):
        self.asked.extend(ids)
        return [self.rows.get(i) for i in ids]


def _props(src, tgt):
    return {"keywords": f"{src} -> {tgt}"}


class _Graph:
    async def get_nodes_batch(self, names):
        return {
            n: {"entity_id": n, "entity_type": "concept"} for n in names if n != "Gone"
        }

    async def get_edges_batch(self, pairs):
        # as the Postgres storage does: the forward edge, then the reverse one
        # over it, when a pair is stored both ways
        out = {}
        for p in pairs:
            s, t = p["src"], p["tgt"]
            for src, tgt in ((s, t), (t, s)):
                if (src, tgt) in EDGES:
                    out[(s, t)] = _props(src, tgt)
        return out

    async def get_nodes_edges_batch(self, names):
        return {n: [e for e in EDGES if n in e] for n in names}


class _AgeGraph(_Graph):
    """The Postgres/AGE storage as zotero_papers sees it: its two tagged SQL
    queries answered from EDGES, the Cypher batch calls off limits."""

    graph_name = "test_graph"

    async def _query(self, sql, params=None):
        # PGGraphStorage._query binds list(params.values()) as $1, $2, ...
        values = list(params.values())
        assert "unnest($1::text[])" in sql
        names = set(values[0])
        if sql.startswith("-- zotero_papers:edges"):
            return [
                {"src": s, "tgt": t, "properties": json.dumps(_props(s, t))}
                for s, t in EDGES
                if s in names and t in names and s != t
            ]
        assert sql.startswith("-- zotero_papers:linking"), sql
        assert ">= $2" in sql and "LIMIT $3" in sql
        min_links, limit = values[1], values[2]
        touched: dict[str, set[str]] = {}
        for edge in EDGES:
            for inner, other in (edge, edge[::-1]):
                if inner in names and other not in names:
                    touched.setdefault(other, set()).add(inner)
        rows = sorted((o, len(by)) for o, by in touched.items() if len(by) >= min_links)
        rows.sort(key=lambda r: -r[1])
        return [{"name": o, "links": k} for o, k in rows[:limit]]

    async def get_edges_batch(self, pairs):
        raise AssertionError("the AGE path reads edges with SQL, not Cypher")

    async def get_nodes_edges_batch(self, names):
        raise AssertionError("the AGE path reads edges with SQL, not Cypher")


class _Rag:
    max_graph_nodes = 1000

    def __init__(self, graph=None):
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
        self.chunk_entity_relation_graph = graph or _Graph()


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


def _node(result, name):
    return next(n for n in result["nodes"] if n["id"] == name)


def _cite(*keys):
    return "; ".join(sorted(zc.citation_short(f"{k}__", 80) for k in keys))


async def test_several_papers_show_their_union_and_who_contributed(library):
    result = await zp.paper_graph(_Rag(), ["10.1/p", "10.1/u"])
    # Stale stays out: u's chunk cites it, but u's anchors do not list it
    assert _nodes(result) == ["Filter", "GRACE", "Leakage", "Mascon"]
    assert _node(result, "Leakage")["properties"]["contributed_by"] == _cite(
        "PPPP0001", "UUUU0001"
    )
    assert _node(result, "Filter")["properties"]["contributed_by"] == _cite("PPPP0001")
    assert [p["paper"] for p in result["paper"]["papers"]] == ["10.1/p", "10.1/u"]
    # the per-paper summaries stay in `papers`
    assert "citation" not in result["paper"]
    single = await zp.paper_graph(_Rag(), ["10.1/p"])
    assert all("contributed_by" not in n["properties"] for n in single["nodes"])


async def test_ids_of_one_paper_count_once(library):
    result = await zp.paper_graph(
        _Rag(), ["PPPP0001", "10.1/p", "https://doi.org/10.1/P"]
    )
    assert [p["paper"] for p in result["paper"]["papers"]] == ["10.1/p"]
    assert _nodes(result) == ["Filter", "GRACE", "Leakage", "Mascon"]
    # one paper: its summary fields stay at the top, as before
    assert result["paper"]["keys"] == ["PPPP0001", "PPPP0002"]


async def test_papers_not_in_the_kb_are_named_and_the_rest_shown(library):
    result = await zp.paper_graph(_Rag(), ["10.1/p", "10.1/r2", "not a paper"])
    assert _nodes(result) == ["Filter", "GRACE", "Leakage", "Mascon"]
    assert [p.get("reason") for p in result["paper"]["papers"]] == [
        None,
        "not_in_kb",
        "not_in_kb",
    ]
    assert "reason" not in result["paper"]
    nothing = await zp.paper_graph(_Rag(), ["10.1/r2", "not a paper"])
    assert nothing["nodes"] == [] and nothing["paper"]["reason"] == "not_in_kb"


async def test_common_view_keeps_what_two_papers_contributed(library):
    result = await zp.paper_graph(_Rag(), ["10.1/p", "10.1/r1"], mode="common")
    assert _nodes(result) == ["GRACE", "Mascon"]
    assert _edges(result) == [("GRACE", "Mascon")]
    assert _node(result, "GRACE")["properties"]["contributed_by"] == _cite(
        "PPPP0001", "RRRR0001"
    )


@pytest.mark.parametrize(
    "papers, reason",
    [
        (["10.1/p"], "need_two_papers"),
        (["10.1/p", "10.1/r2"], "need_two_papers"),
        (["10.1/r1", "10.1/u"], "nothing_in_common"),
    ],
)
async def test_empty_common_view_says_why(library, papers, reason):
    result = await zp.paper_graph(_Rag(), papers, mode="common")
    assert result["nodes"] == [] and result["paper"]["reason"] == reason


async def test_shared_view_of_several_papers(library):
    result = await zp.paper_graph(_Rag(), ["10.1/p", "10.1/u"], mode="shared")
    # p shares GRACE and Mascon with r1, which it cites; u cites p, so the
    # Leakage both contributed counts as shared with p
    assert _nodes(result) == ["GRACE", "Leakage", "Mascon"]
    assert _edges(result) == [("GRACE", "Mascon"), ("Leakage", "Mascon")]
    assert _node(result, "GRACE")["properties"]["shared_with"] == _cite("RRRR0001")
    assert _node(result, "Leakage")["properties"]["shared_with"] == _cite("PPPP0001")
    assert result["paper"]["cited_sharing"] == 2
    empty = await zp.paper_graph(_Rag(), ["10.1/r1", "KKKK0001"], mode="shared")
    assert empty["paper"]["reason"] == "no_references_in_kb"


async def test_shared_view_looks_up_only_what_a_selected_paper_contributed(library):
    rag = _Rag()
    result = await zp.paper_graph(rag, "10.1/p", mode="shared")
    assert _nodes(result) == ["GRACE", "Mascon"]
    # r1's anchors list Other, which p never contributed
    assert "Other" not in rag.entity_chunks.asked
    assert make_relation_chunk_key("GRACE", "Other") not in rag.relation_chunks.asked


async def test_kb_context_adds_the_graphs_other_relations_marked(library):
    result = await zp.paper_graph(_Rag(), "10.1/p", context="kb")
    assert _nodes(result) == ["Filter", "GRACE", "Leakage", "Mascon"]
    # GRACE-Mascon, stored in both directions, is one edge
    assert _edges(result) == [
        ("Filter", "Mascon"),
        ("GRACE", "Leakage"),
        ("GRACE", "Mascon"),
        ("Leakage", "Mascon"),
    ]
    marked = [
        (e["source"], e["target"])
        for e in result["edges"]
        if "paper_context" in e["properties"]
    ]
    assert marked == [("GRACE", "Leakage")]
    assert result["paper"]["n_context_edges"] == 1


async def test_linked_context_adds_the_entity_linking_the_most(library):
    result = await zp.paper_graph(_Rag(), "10.1/p", context="linked")
    # four shown entities leave room for one: Hub3 touches three, Hub2 two
    assert _nodes(result) == ["Filter", "GRACE", "Hub3", "Leakage", "Mascon"]
    assert (
        _node(result, "Hub3")["properties"]["paper_context"]
        == "Not from the selected papers; linked to 3 shown entities"
    )
    hub = [e for e in result["edges"] if "Hub3" in (e["source"], e["target"])]
    assert len(hub) == 3 and all("paper_context" in e["properties"] for e in hub)
    assert result["paper"]["n_linking"] == 1


async def test_linking_entities_are_not_joined_to_each_other(library, monkeypatch):
    monkeypatch.setattr(zp, "_LINK_DIVISOR", 2)  # room for two
    result = await zp.paper_graph(_Rag(), "10.1/p", context="linked")
    assert {"Hub2", "Hub3"} <= set(_nodes(result))
    assert "Lone" not in _nodes(result)
    assert ("Hub2", "Hub3") not in _edges(result)


async def test_a_truncated_view_gets_no_linking_entities(library):
    result = await zp.paper_graph(_Rag(), "10.1/p", max_nodes=2, context="linked")
    assert result["is_truncated"]
    assert _nodes(result) == ["Filter", "Mascon"]
    assert result["paper"]["n_linking"] == 0


async def test_too_many_papers_are_refused_before_any_lookup(library):
    rag = _Rag()
    papers = [f"10.1/x{i}" for i in range(zp.MAX_PAPERS + 1)]
    result = await zp.paper_graph(rag, papers)
    assert result["nodes"] == [] and result["paper"]["reason"] == "too_many_papers"
    assert result["paper"]["papers"] == [] and rag.doc_status.calls == 0


@pytest.mark.parametrize("graph", [_Graph, _AgeGraph])
async def test_a_pair_stored_both_ways_shows_its_reverse_edge(library, graph):
    result = await zp.paper_graph(_Rag(graph()), "10.1/p")
    edge = next(
        e for e in result["edges"] if {e["source"], e["target"]} == {"GRACE", "Mascon"}
    )
    assert edge["properties"]["keywords"] == "Mascon -> GRACE"


@pytest.mark.parametrize("context", ["own", "kb", "linked"])
@pytest.mark.parametrize(
    "mode, papers",
    [
        ("paper", ["10.1/p"]),
        ("paper", ["10.1/p", "10.1/u"]),
        ("shared", ["10.1/p"]),
        ("common", ["10.1/p", "10.1/r1"]),
    ],
)
async def test_the_age_sql_path_gives_the_same_graph(library, mode, papers, context):
    generic = await zp.paper_graph(_Rag(), papers, mode, context=context)
    age = await zp.paper_graph(_Rag(_AgeGraph()), papers, mode, context=context)
    assert age == generic


def test_route_passes_every_paper_and_the_options(monkeypatch):
    seen = {}

    async def fake(rag, papers, mode, max_nodes, context):
        seen.update(papers=papers, mode=mode, max_nodes=max_nodes, context=context)
        return {"nodes": [], "edges": [], "is_truncated": False, "paper": {}}

    monkeypatch.setattr(zp, "paper_graph", fake)
    app = FastAPI()
    rag = SimpleNamespace(max_graph_nodes=500)
    app.include_router(_graph_routes.create_graph_routes(rag, api_key="test-key"))
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}
    url = "/graphs/paper?paper=10.1/p&paper=UUUU0001&mode=common&context=kb"
    assert client.get(url + "&max_nodes=900", headers=headers).status_code == 200
    assert seen == {
        "papers": ["10.1/p", "UUUU0001"],
        "mode": "common",
        "max_nodes": 500,
        "context": "kb",
    }
    assert client.get("/graphs/paper?paper=10.1/p", headers=headers).status_code == 200
    assert seen["papers"] == ["10.1/p"] and seen["context"] == "own"
    bad = client.get("/graphs/paper?paper=10.1/p&context=all", headers=headers)
    assert bad.status_code == 422
