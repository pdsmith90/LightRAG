"""Query-side knobs added for a citation-heavy paper corpus (fork).

- LL_KEYWORDS_FALLBACK: the entity leg reuses the high-level keywords when the
  extractor returned no low-level ones, instead of being skipped.
- SIDECAR_RELATIONS=none: a table/equation sidecar entity gets no fan-out edges.
- LEXICAL_DF_CAP_PCT: the lexical leg's document-frequency cap is a setting.
- Pending chunk vectors are embedded at query priority, so a query does not wait
  behind the pipeline's embedding backlog.
- QueryParam.min_rerank_score overrides the server's floor for one request.
"""

import asyncio
import datetime

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_QUERY_PRIORITY
from lightrag.kg.postgres_impl import PGKVStorage, PGVectorStorage, _PendingPGVectorDoc
from lightrag.namespace import NameSpace
from lightrag.operate import _fallback_low_level_keywords, _sidecar_fanout_enabled

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# LL_KEYWORDS_FALLBACK
# ---------------------------------------------------------------------------


def test_fallback_is_off_by_default_and_only_fills_an_empty_list():
    hl = ["flicker noise", "least squares"]
    assert _fallback_low_level_keywords(hl, [], "mix", {}) == []
    assert (
        _fallback_low_level_keywords(hl, [], "mix", {"ll_keywords_fallback": False})
        == []
    )
    assert _fallback_low_level_keywords(
        hl, ["c20"], "mix", {"ll_keywords_fallback": True}
    ) == ["c20"]
    assert (
        _fallback_low_level_keywords([], [], "mix", {"ll_keywords_fallback": True})
        == []
    )


def test_fallback_copies_the_high_level_keywords_for_entity_modes_only():
    hl = ["flicker noise", "least squares"]
    on = {"ll_keywords_fallback": True}
    for mode in ("local", "hybrid", "mix"):
        got = _fallback_low_level_keywords(hl, [], mode, on)
        assert got == hl and got is not hl
    assert _fallback_low_level_keywords(hl, [], "global", on) == []
    assert _fallback_low_level_keywords(hl, [], "naive", on) == []


# ---------------------------------------------------------------------------
# SIDECAR_RELATIONS
# ---------------------------------------------------------------------------


def test_sidecar_fanout_is_on_unless_none():
    assert _sidecar_fanout_enabled({}) is True
    assert _sidecar_fanout_enabled({"sidecar_relations": "all"}) is True
    assert _sidecar_fanout_enabled({"sidecar_relations": "none"}) is False


# ---------------------------------------------------------------------------
# LEXICAL_DF_CAP_PCT and get_chunks_for_works (fake database)
# ---------------------------------------------------------------------------


class _FakeDB:
    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    async def query(self, sql, params=None, multirows=False):
        self.calls.append((sql, params))
        if "pg_indexes" in sql:
            return {"?column?": 1}
        return [
            {
                "id": "c1",
                "full_doc_id": "d1",
                "file_path": "AAAA0001__x.md",
                "content": "c",
                "chunk_order_index": 0,
                "score": 1,
                "work": "AAAA0001",
            }
        ]


def _kv(db, namespace=NameSpace.KV_STORE_TEXT_CHUNKS, global_config=None):
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = namespace
    storage.workspace = "ws"
    storage.db = db
    if global_config is not None:
        storage.global_config = global_config
    return storage


@pytest.mark.asyncio
async def test_lexical_search_passes_the_df_cap_default_and_setting():
    db = _FakeDB()
    await _kv(db).lexical_search("okafor 2019", 6)
    assert db.calls[-1][1] == ["ws", "okafor 2019", 6, 2.0]
    assert "$4::float / 100.0" in db.calls[-1][0]
    await _kv(db, global_config={"lexical_df_cap_pct": 5}).lexical_search(
        "okafor 2019", 6
    )
    assert db.calls[-1][1] == ["ws", "okafor 2019", 6, 5.0]


@pytest.mark.asyncio
async def test_chunks_for_works_queries_text_chunks_only():
    db = _FakeDB()
    rows = await _kv(db).get_chunks_for_works(["AAAA0001", "AAAA0002"], "residuals", 2)
    assert rows[0]["work"] == "AAAA0001"
    sql, params = db.calls[-1]
    assert params == ["ws", ["AAAA0001", "AAAA0002"], "residuals", 2]
    assert "split_part(s.file_path, '__', 1)" in sql and "row_number()" in sql
    assert await _kv(db).get_chunks_for_works([], "q", 2) == []
    assert await _kv(db).get_chunks_for_works(["AAAA0001"], "q", 0) == []
    assert (
        await _kv(db, NameSpace.KV_STORE_FULL_DOCS).get_chunks_for_works(
            ["AAAA0001"], "q", 2
        )
        == []
    )
    assert len(db.calls) == 1


# ---------------------------------------------------------------------------
# Pending vectors are embedded at query priority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_embedding_of_pending_vectors_carries_query_priority():
    calls: list[dict] = []

    async def embed(texts, **kwargs):
        calls.append(kwargs)
        return np.ones((len(texts), 2), dtype=np.float32)

    pending = _PendingPGVectorDoc(
        item={"content": "a pending chunk"}, created_at=datetime.datetime.now()
    )
    storage = PGVectorStorage.__new__(PGVectorStorage)
    storage.workspace = "ws"
    storage._flush_lock = asyncio.Lock()
    storage._pending_vector_deletes = set()
    storage._pending_vector_docs = {"c1": pending}
    storage._max_batch_size = 8
    storage.embedding_func = embed

    got = await storage.get_vectors_by_ids(["c1"])

    assert got == {"c1": [1.0, 1.0]}
    assert calls == [{"context": "document", "_priority": DEFAULT_QUERY_PRIORITY}]
    assert pending.vector is not None  # cached for the flush


# ---------------------------------------------------------------------------
# QueryParam.min_rerank_score
# ---------------------------------------------------------------------------


def test_min_rerank_score_defaults_to_none_and_is_a_trailing_field():
    param = QueryParam()
    assert param.min_rerank_score is None
    assert QueryParam(min_rerank_score=0.1).min_rerank_score == 0.1


@pytest.mark.asyncio
async def test_request_floor_replaces_the_server_floor_for_one_query():
    from lightrag.utils import Tokenizer, TokenizerInterface, process_chunks_unified

    class _Chars(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [0] * len(content)

        def decode(self, tokens: list[int]) -> str:
            return "x" * len(tokens)

    chunks = [
        {"content": "kept", "file_path": "a.md", "chunk_id": "a", "rerank_score": 0.9},
        {"content": "weak", "file_path": "b.md", "chunk_id": "b", "rerank_score": 0.1},
    ]
    global_config = {
        "tokenizer": Tokenizer("chars", _Chars()),
        "min_rerank_score": 0.05,
    }

    async def run(floor):
        param = QueryParam(enable_rerank=True, chunk_top_k=None, min_rerank_score=floor)
        kept = await process_chunks_unified(
            query="q",
            unique_chunks=[dict(c) for c in chunks],
            query_param=param,
            global_config=global_config,
            chunk_token_limit=10_000,
        )
        return [c["chunk_id"] for c in kept]

    assert await run(None) == ["a", "b"]  # the server's 0.05 applies
    assert await run(0.5) == ["a"]  # the request's floor wins
    assert await run(0.0) == ["a", "b"]  # an explicit 0 disables the floor
