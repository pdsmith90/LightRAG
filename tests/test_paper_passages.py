"""The best passages of one named paper for a question (POST /graph/papers/passages).

A paper is one work: every copy filed under its Zotero storage keys. Its chunks
are the candidates -- all of them for a short paper, else the best by vector
similarity to the question and the best by full-text rank, both kept to the
work's own chunks -- and the server's reranker orders them against the bare
question. The first chunk competes like any other. Reference lists stay out
when the bibliography filter is on. Without a reranker, or when it fails, the
vector order decides. Fixtures are invented.
"""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lightrag.zotero_papers as zp
from lightrag.base import DocStatus

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

DIM = 8
BIB = "\n".join(
    f"Doe, J. ({1990 + i}). A title number {i}. J. Clay, {10 + i}(2), 45-67. doi:10.9/x{i}"
    for i in range(6)
)


def _unit(i):
    v = np.zeros(DIM)
    v[i % DIM] = 1.0
    return v


class _Status:
    def __init__(self, file_path, chunks):
        self.file_path, self.chunks_list = file_path, chunks


class _DocStatus:
    def __init__(self, docs):
        self.docs = docs

    async def get_docs_by_statuses(self, statuses):
        assert statuses == [DocStatus.PROCESSED]
        return {d: _Status(f, c) for d, (f, c) in self.docs.items()}


class _Chunks:
    """text_chunks: rows by id; ``lexical`` (if set) answers rank_chunks_by_text."""

    def __init__(self, rows, lexical=None):
        self.rows = rows
        self.asked: list[list[str]] = []
        if lexical is not None:
            self.lexical = lexical
            self.lexical_calls = []

            async def rank_chunks_by_text(ids, query, limit):
                self.lexical_calls.append((sorted(ids), query, limit))
                return [i for i in self.lexical(query) if i in set(ids)][:limit]

            self.rank_chunks_by_text = rank_chunks_by_text

    async def get_by_ids(self, ids):
        self.asked.append(list(ids))
        return [self.rows.get(i) for i in ids]


class _Vectors:
    def __init__(self, vectors, query_vector):
        self.vectors = vectors
        self.query_vector = query_vector
        self.asked: list[list[str]] = []
        self.embedded: list[str] = []

        async def embedding_func(texts, **kwargs):
            self.embedded.extend(texts)
            return np.array([self.query_vector for _ in texts])

        self.embedding_func = embedding_func

    async def get_vectors_by_ids(self, ids):
        self.asked.append(list(ids))
        return {i: list(self.vectors[i]) for i in ids if i in self.vectors}


def _row(doc, idx, content, key):
    return {
        "content": content,
        "chunk_order_index": idx,
        "full_doc_id": doc,
        "file_path": f"{key}__Paper.md",
        "heading": {"parent_headings": [f"Section {idx}"]} if idx else {},
    }


def _library(n_chunks, *, copies=1, contents=None, lexical=None, vector_of=None):
    """One work of ``n_chunks`` chunks per copy (keys PPPP0001, PPPP0002...)."""
    docs, rows, vectors = {}, {}, {}
    for c in range(copies):
        key = f"PPPP000{c + 1}"
        ids = [f"p{c}-c{i}" for i in range(n_chunks)]
        docs[f"doc-p{c}"] = (f"{key}__Paper.md", ids)
        for i, cid in enumerate(ids):
            text = contents[i] if contents else f"passage {i} of the paper"
            rows[cid] = _row(f"doc-p{c}", i, text, key)
            vectors[cid] = (vector_of or _unit)(i)
    docs["doc-o"] = ("OOOO0001__Other.md", ["o-c0"])
    rows["o-c0"] = _row("doc-o", 0, "another paper", "OOOO0001")
    vectors["o-c0"] = _unit(0)
    return docs, rows, vectors


def _rag(
    docs, rows, vectors, *, query_vector=None, rerank=None, lexical=None, bib=False
):
    return SimpleNamespace(
        doc_status=_DocStatus(docs),
        text_chunks=_Chunks(rows, lexical),
        chunks_vdb=_Vectors(
            vectors, _unit(0) if query_vector is None else query_vector
        ),
        rerank_model_func=rerank,
        drop_bibliography_chunks=bib,
    )


class _Reranker:
    """Scores a document by the number in "passage N" (higher N, better);
    ``best`` names one text to put first."""

    def __init__(self, best=None, fail=False):
        self.best, self.fail = best, fail
        self.calls = []

    async def __call__(self, query, documents, top_n=None):
        self.calls.append((query, list(documents), top_n))
        if self.fail:
            raise RuntimeError("reranker down")

        def score(text):
            if text == self.best:
                return 1.0
            digits = "".join(ch for ch in text if ch.isdigit())
            return int(digits) / 1000 if digits else 0.0

        order = sorted(range(len(documents)), key=lambda i: -score(documents[i]))
        return [{"index": i, "relevance_score": score(documents[i])} for i in order][
            :top_n
        ]


@pytest.fixture(autouse=True)
def _fresh_cache():
    zp._docs_cache.update(at=0.0, owner=None, by_key={}, pending=None)
    yield
    zp._docs_cache.update(at=0.0, owner=None, by_key={}, pending=None)


async def test_a_short_paper_reranks_all_its_chunks_against_the_bare_question():
    rerank = _Reranker()
    rag = _rag(*_library(5), rerank=rerank)
    out = await zp.paper_passages(
        rag, ["PPPP0001", "ZZZZ0009"], "why is the sky blue?", 2
    )
    assert out["keys"] == ["PPPP0001"]
    assert out["n_chunks"] == 5
    assert [p["chunk_id"] for p in out["passages"]] == ["p0-c4", "p0-c3"]
    first = out["passages"][0]
    assert first == {
        "key": "PPPP0001",
        "file_path": "PPPP0001__Paper.md",
        "chunk_id": "p0-c4",
        "chunk_order_index": 4,
        "content": "passage 4 of the paper",
        "content_headings": "Section 4",
        "score": pytest.approx(0.004),
    }
    query, documents, _ = rerank.calls[0]
    assert query == "why is the sky blue?"
    # every chunk of the work competed, the first one too; no other paper's
    assert sorted(documents) == sorted(f"passage {i} of the paper" for i in range(5))
    # a short paper needs no vector ranking
    assert rag.chunks_vdb.asked == []


async def test_the_first_chunk_wins_when_the_reranker_prefers_it():
    rerank = _Reranker(best="passage 0 of the paper")
    out = await zp.paper_passages(
        _rag(*_library(5), rerank=rerank), ["PPPP0001"], "q", 3
    )
    assert [p["chunk_order_index"] for p in out["passages"]] == [0, 4, 3]
    assert out["passages"][0]["content_headings"] is None


async def test_no_indexed_key_answers_empty():
    out = await zp.paper_passages(
        _rag(*_library(3), rerank=_Reranker()), ["ZZZZ0009"], "q", 3
    )
    assert out == {"keys": [], "n_chunks": 0, "passages": []}


async def test_reference_lists_stay_out_when_the_filter_is_on():
    contents = ["passage 1 intro", BIB, "passage 2 method"]
    lib = _library(3, contents=contents)
    rerank = _Reranker(best=BIB)
    out = await zp.paper_passages(
        _rag(*lib, rerank=rerank, bib=True), ["PPPP0001"], "q", 3
    )
    assert BIB not in [p["content"] for p in out["passages"]]
    assert BIB not in rerank.calls[0][1]
    out = await zp.paper_passages(
        _rag(*lib, rerank=_Reranker(best=BIB)), ["PPPP0001"], "q", 3
    )
    assert out["passages"][0]["content"] == BIB


async def test_without_a_reranker_the_vector_order_decides():
    # the question points at chunk 2's direction
    rag = _rag(*_library(5), query_vector=_unit(2))
    out = await zp.paper_passages(rag, ["PPPP0001"], "q", 2)
    assert out["passages"][0]["chunk_id"] == "p0-c2"
    assert out["passages"][0]["score"] == pytest.approx(1.0)
    assert out["passages"][1]["score"] == pytest.approx(0.0)


async def test_a_failing_reranker_falls_back_to_the_vector_order():
    rerank = _Reranker(fail=True)
    rag = _rag(*_library(5), query_vector=_unit(3), rerank=rerank)
    out = await zp.paper_passages(rag, ["PPPP0001"], "q", 1)
    assert rerank.calls and [p["chunk_id"] for p in out["passages"]] == ["p0-c3"]


def _spread(i):
    """Chunk i's vector leans toward axis 0 the lower i is."""
    v = np.zeros(DIM)
    v[0], v[1] = 1.0 / (1 + i), 1.0
    return v


async def test_a_long_paper_takes_the_vector_and_text_leaders_of_its_own_chunks():
    n = 100
    # the text leg ranks the highest-numbered chunks first
    lexical = lambda q: [f"p0-c{i}" for i in reversed(range(n))] + ["o-c0"]  # noqa: E731
    rerank = _Reranker()
    rag = _rag(*_library(n, vector_of=_spread), rerank=rerank, lexical=lexical)
    out = await zp.paper_passages(rag, ["PPPP0001"], "the question", 3)
    assert out["n_chunks"] == n
    documents = rerank.calls[0][1]
    nums = sorted(int(d.split()[1]) for d in documents)
    # vector leaders 0..39, text leaders 60..99
    assert nums == list(range(40)) + list(range(60, 100))
    assert len(documents) <= 80
    # both legs were kept to the work's chunks, and got the bare question
    ((ids, query, limit),) = rag.text_chunks.lexical_calls
    assert ids == sorted(f"p0-c{i}" for i in range(n)) and query == "the question"
    assert limit == 40
    assert rag.chunks_vdb.asked == [sorted(f"p0-c{i}" for i in range(n))]
    assert rag.chunks_vdb.embedded == ["the question"]
    assert [p["chunk_id"] for p in out["passages"]] == ["p0-c99", "p0-c98", "p0-c97"]
    # only the candidates' rows were read
    assert sum(len(a) for a in rag.text_chunks.asked) == 80


async def test_a_long_paper_on_a_storage_without_text_rank_uses_the_vector_leg():
    rerank = _Reranker()
    rag = _rag(*_library(100, vector_of=_spread), rerank=rerank)
    await zp.paper_passages(rag, ["PPPP0001"], "q", 3)
    nums = sorted(int(d.split()[1]) for d in rerank.calls[0][1])
    assert nums == list(range(40))


async def test_a_storage_that_ranks_by_id_ships_only_the_leaders():
    """The vector leg of a long work is ranked where the vectors are stored: the
    storage answers with its best _PASSAGE_LEG ids, not every vector of the work."""
    n = 2000
    rerank = _Reranker()
    rag = _rag(*_library(n, vector_of=_spread), rerank=rerank)
    vdb = rag.chunks_vdb
    asked = []

    async def rank_ids_by_vector(ids, query_embedding, limit):
        asked.append((len(ids), len(query_embedding), limit))
        sims = {i: float(np.dot(vdb.vectors[i], query_embedding)) for i in ids}
        best = sorted(sims, key=lambda i: -sims[i])[:limit]
        return {i: sims[i] for i in best}

    async def fetch_everything(ids):
        raise AssertionError("every vector of the work was fetched")

    vdb.rank_ids_by_vector = rank_ids_by_vector
    vdb.get_vectors_by_ids = fetch_everything
    out = await zp.paper_passages(rag, ["PPPP0001"], "the question", 3)
    assert asked == [(n, DIM, 40)]
    nums = sorted(int(d.split()[1]) for d in rerank.calls[0][1])
    assert nums == list(range(40))
    assert out["n_chunks"] == n and len(out["passages"]) == 3


async def test_copies_of_one_paper_count_once():
    rerank = _Reranker()
    rag = _rag(*_library(4, copies=2), rerank=rerank)
    out = await zp.paper_passages(rag, ["PPPP0001", "PPPP0002"], "q", 8)
    # n_chunks is the length of the paper (its longest copy), so "chunk i of n"
    # holds; the copies are not added up
    assert out["keys"] == ["PPPP0001", "PPPP0002"] and out["n_chunks"] == 4
    # the same text from two copies reaches the reranker once
    assert len(rerank.calls[0][1]) == 4
    assert len(out["passages"]) == 4
    assert {p["key"] for p in out["passages"]} <= {"PPPP0001", "PPPP0002"}


async def test_n_chunks_is_the_longest_copy_not_the_sum():
    docs, rows, vectors = _library(4, copies=2)
    # the second copy was split into more chunks
    docs["doc-p1"] = ("PPPP0002__Paper.md", [f"p1-c{i}" for i in range(6)])
    for i in range(4, 6):
        rows[f"p1-c{i}"] = _row("doc-p1", i, f"passage {i} of the paper", "PPPP0002")
        vectors[f"p1-c{i}"] = _unit(i)
    out = await zp.paper_passages(
        _rag(docs, rows, vectors), ["PPPP0001", "PPPP0002"], "q", 3
    )
    assert out["n_chunks"] == 6
    assert all(p["chunk_order_index"] < out["n_chunks"] for p in out["passages"])


# --- the route ---------------------------------------------------------------


def _client(rag):
    app = FastAPI()
    app.include_router(_graph_routes.create_graph_routes(rag, api_key="k"))
    return TestClient(app), {"X-API-Key": "k"}


def test_route_answers_with_passages_and_the_marker():
    client, h = _client(_rag(*_library(5), rerank=_Reranker()))
    r = client.post(
        "/graph/papers/passages",
        json={"keys": ["pppp0001"], "question": "  what?  "},
        headers=h,
    )
    assert r.status_code == 200 and r.headers["X-Paper-Search"] == "2"
    body = r.json()
    assert body["keys"] == ["PPPP0001"] and body["n_chunks"] == 5
    assert len(body["passages"]) == 3  # top_n defaults to 3


@pytest.mark.parametrize(
    "body",
    [
        {"keys": [], "question": "q"},
        {"keys": [f"PPPP{i:04d}" for i in range(11)], "question": "q"},
        {"keys": ["PPPP0001"], "question": "   "},
        {"keys": ["PPPP0001"]},
        {"keys": ["PPPP0001"], "question": "q", "top_n": 0},
        {"keys": ["PPPP0001"], "question": "q", "top_n": 9},
        {"keys": ["not a key"], "question": "q"},
    ],
)
def test_route_refuses_a_bad_body(body):
    client, h = _client(_rag(*_library(2), rerank=_Reranker()))
    assert (
        client.post("/graph/papers/passages", json=body, headers=h).status_code == 422
    )


def test_route_hides_an_internal_error():
    rag = _rag(*_library(2), rerank=_Reranker())

    async def broken(ids):
        raise RuntimeError("db down at a private host")

    rag.text_chunks.get_by_ids = broken
    client, h = _client(rag)
    r = client.post(
        "/graph/papers/passages",
        json={"keys": ["PPPP0001"], "question": "q"},
        headers=h,
    )
    assert r.status_code == 500 and r.headers["X-Paper-Search"] == "2"
    assert "private host" not in r.text


# --- the Postgres legs -------------------------------------------------------


async def test_postgres_text_rank_is_kept_to_the_given_chunks():
    from lightrag.kg.postgres_impl import PGKVStorage, SQL_TEMPLATES

    seen = {}

    class _DB:
        async def query(self, sql, params=None, multirows=False, **kw):
            seen.update(sql=sql, params=params, multirows=multirows)
            return [{"id": "c7"}, {"id": "c2"}]

    store = SimpleNamespace(
        namespace="text_chunks", workspace="ws", db=_DB(), _lexical_index_ok=True
    )
    got = await PGKVStorage.rank_chunks_by_text(
        store, ["c2", "c7", "c9"], "glaze firing", 40
    )
    assert got == ["c7", "c2"]
    assert seen["sql"] == SQL_TEMPLATES["rank_chunks_by_text_text_chunks"]
    assert seen["params"] == ["ws", ["c2", "c7", "c9"], "glaze firing", 40]
    assert seen["multirows"] is True
    assert "id = ANY($2)" in seen["sql"] and "LIMIT $4" in seen["sql"]
    assert await PGKVStorage.rank_chunks_by_text(store, [], "q", 40) == []
    assert await PGKVStorage.rank_chunks_by_text(store, ["c2"], "  ", 40) == []


async def test_postgres_vector_rank_returns_only_the_nearest_of_the_given_chunks():
    from lightrag.kg.postgres_impl import PGVectorStorage

    seen = {}

    class _DB:
        vector_index_type = "HNSW_HALFVEC"

        async def query(self, sql, params=None, multirows=False, **kw):
            seen.update(sql=sql, params=params, multirows=multirows)
            return [{"id": "c7", "distance": 0.25}, {"id": "c2", "distance": 0.5}]

    store = SimpleNamespace(
        workspace="ws",
        table_name="LIGHTRAG_VDB_CHUNKS",
        db=_DB(),
        _pending_vector_docs={},
        _pending_vector_deletes=set(),
    )
    q = np.array([1.0, 0.0])
    got = await PGVectorStorage.rank_ids_by_vector(store, ["c2", "c7", "c9"], q, 40)
    assert got == {"c7": 0.75, "c2": 0.5} and list(got) == ["c7", "c2"]
    sql, params = seen["sql"], seen["params"]
    assert params[:2] == ["ws", ["c2", "c7", "c9"]] and params[3] == 40
    assert params[2] is q and seen["multirows"] is True
    # ranked among the given ids only and exactly (an index scan would filter
    # after the fact), cut in SQL, in the storage's vector type
    assert "id = ANY($2)" in sql and "MATERIALIZED" in sql
    assert "LIMIT $4" in sql and "$3::halfvec" in sql
    assert await PGVectorStorage.rank_ids_by_vector(store, [], q, 40) == {}


async def test_postgres_vector_rank_reads_unflushed_vectors_from_the_buffer():
    from lightrag.kg.postgres_impl import PGVectorStorage

    class _DB:
        vector_index_type = "HNSW"

        async def query(self, sql, params=None, multirows=False, **kw):
            assert params[1] == ["c1"]  # a buffered chunk is not asked of SQL
            return [{"id": "c1", "distance": 0.5}]

    async def get_vectors_by_ids(ids):
        assert ids == ["c2"]
        return {"c2": [1.0, 0.0]}

    store = SimpleNamespace(
        workspace="ws",
        table_name="T",
        db=_DB(),
        _pending_vector_docs={"c2": object()},
        _pending_vector_deletes={"c3"},
        get_vectors_by_ids=get_vectors_by_ids,
    )
    got = await PGVectorStorage.rank_ids_by_vector(
        store, ["c1", "c2", "c3"], np.array([1.0, 0.0]), 40
    )
    assert got == {"c2": 1.0, "c1": 0.5}
