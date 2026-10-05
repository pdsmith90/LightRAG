"""LEXICAL_CHUNK_TOP_K: full-text matches join mix and naive candidates.

Dense retrieval misses queries made of names, years and acronyms. The lexical
leg asks the text-chunk storage for full-text matches, drops reference lists
before cutting to ``lexical_chunk_top_k`` (when the bibliography filter is on),
and interleaves the rest with the vector chunks ahead of reranking. Both this
switch and ``max_chunks_per_doc`` partition the query-answer cache, keyed only
when on so existing entries keep hitting.
"""

import pytest

import lightrag.operate as operate
from lightrag.base import QueryContextResult, QueryParam
from lightrag.kg.postgres_impl import PGKVStorage
from lightrag.namespace import NameSpace
from lightrag.operate import (
    _get_lexical_context,
    _interleave_chunks,
    _merge_all_chunks,
    kg_query,
    naive_query,
)
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline

BIBLIOGRAPHY = """LUHN H. P. 1957. A statistical approach to mechanized encoding and searching of literary information. IBM J. Res. Dev. 1, 309–317.
MARON M. E. & KUHNS J. L. 1960. On relevance, probabilistic indexing and information retrieval. J. ACM 7, 216–244.
SALTON G., WONG A. & YANG C. S. 1975. A vector space model for automatic indexing. Commun. ACM 18, 613–620.
ROBERTSON S. E. 1977. The probability ranking principle in IR. J. Doc. 33, 294–304.
SALTON G. & McGILL M. J. 1983. Introduction to Modern Information Retrieval. McGraw-Hill, New York."""


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _TextChunks:
    """A text-chunk storage with a scripted ``lexical_search``."""

    def __init__(self, global_config, rows=()):
        self.global_config = global_config
        self.rows = list(rows)
        self.requests: list[tuple[str, int]] = []

    async def lexical_search(self, query, top_k):
        self.requests.append((query, top_k))
        return self.rows[:top_k]


def _row(chunk_id, content=None, file_path="lex.md"):
    return {
        "id": chunk_id,
        "content": content or f"lexical {chunk_id}",
        "file_path": file_path,
        "score": 1.0,
    }


@pytest.mark.asyncio
async def test_off_by_default_and_never_asks_the_storage():
    storage = _TextChunks({}, [_row("l1")])
    assert await _get_lexical_context("okafor 2019", storage) == []
    assert storage.requests == []


@pytest.mark.asyncio
async def test_a_storage_without_lexical_search_is_skipped(lightrag_log_records):
    class _Plain:
        global_config = {"lexical_chunk_top_k": 4}

    operate._lexical_unsupported_warned = False
    assert await _get_lexical_context("okafor 2019", _Plain()) == []
    assert any("has no lexical_search" in r.getMessage() for r in lightrag_log_records)


@pytest.mark.asyncio
async def test_reference_lists_are_dropped_before_the_cut():
    rows = [
        _row("bib1", BIBLIOGRAPHY),
        _row("l1"),
        _row("bib2", BIBLIOGRAPHY),
        _row("l2"),
        _row("l3"),
    ]
    storage = _TextChunks(
        {"lexical_chunk_top_k": 2, "drop_bibliography_chunks": True}, rows
    )

    chunks = await _get_lexical_context("luhn 1957", storage)

    assert storage.requests == [("luhn 1957", 6)]  # over-fetched three-fold
    assert [c["chunk_id"] for c in chunks] == ["l1", "l2"]
    assert {c["source_type"] for c in chunks} == {"lexical"}


def test_interleave_alternates_and_drops_repeated_ids():
    vec = [{"chunk_id": "v1"}, {"chunk_id": "shared"}, {"chunk_id": "v3"}]
    lex = [{"chunk_id": "shared"}, {"chunk_id": "l2"}]
    assert [c["chunk_id"] for c in _interleave_chunks(vec, lex)] == [
        "v1",
        "shared",
        "l2",
        "v3",
    ]


@pytest.mark.asyncio
async def test_mix_merge_puts_lexical_chunks_beside_the_vector_ones():
    vec = [{"chunk_id": "v1", "content": "v1", "file_path": "v.md"}]
    lex = [
        {
            "chunk_id": "l1",
            "content": "l1",
            "file_path": "l.md",
            "source_type": "lexical",
        }
    ]
    merged = await _merge_all_chunks([], [], vec, lexical_chunks=lex)
    assert [c["chunk_id"] for c in merged] == ["v1", "l1"]


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [{"id": "v1", "content": "vector prose", "file_path": "v.md"}]


def _config(model, **extra) -> dict:
    return {
        "tokenizer": Tokenizer("fake", _FakeTokenizerImpl()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        **extra,
    }


@pytest.mark.asyncio
async def test_naive_context_holds_the_lexical_chunks():
    async def unused_model(*_args, **_kwargs):
        raise AssertionError("only_need_context must not call the model")

    config = _config(unused_model, lexical_chunk_top_k=2)
    storage = _TextChunks(
        config, [_row("l1", "Okafor Chidi 2019 adaptive median filter")]
    )
    result = await naive_query(
        "okafor 2019",
        _ChunksVDB(),
        QueryParam(mode="naive", enable_rerank=False, only_need_context=True),
        config,
        text_chunks_db=storage,
    )
    assert [c["chunk_id"] for c in result.raw_data["data"]["chunks"]] == ["v1", "l1"]


@pytest.mark.asyncio
async def test_naive_query_with_no_vector_hits_still_answers_from_lexical_ones():
    class _EmptyVDB(_ChunksVDB):
        async def query(self, *_args, **_kwargs):
            return []

    async def unused_model(*_args, **_kwargs):
        raise AssertionError("only_need_context must not call the model")

    config = _config(unused_model, lexical_chunk_top_k=2)
    result = await naive_query(
        "okafor 2019",
        _EmptyVDB(),
        QueryParam(mode="naive", enable_rerank=False, only_need_context=True),
        config,
        text_chunks_db=_TextChunks(config, [_row("l1")]),
    )
    assert [c["chunk_id"] for c in result.raw_data["data"]["chunks"]] == ["l1"]


# ---------------------------------------------------------------------------
# query-answer cache key
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.store = {}

    async def get_by_id(self, key):
        return self.store.get(key)

    async def upsert(self, entries):
        self.store.update(entries)

    def answer_keys(self):
        return [key for key in self.store if ":query:" in key]


class _Model:
    def __init__(self):
        self.calls = 0

    async def __call__(self, *_args, **_kwargs):
        self.calls += 1
        return f"answer-{self.calls}"


@pytest.fixture
def stub_kg_context(monkeypatch):
    async def fake_keywords(*_args, **_kwargs):
        return "", "retrieval"

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(context="context", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["lexical_chunk_top_k", "max_chunks_per_doc"])
async def test_kg_query_option_on_partitions_the_answer_cache(stub_kg_context, option):
    model, cache = _Model(), _Cache()

    async def run(config):
        return await kg_query(
            "which filter removes noise?",
            None,
            None,
            None,
            _TextChunks(config),
            QueryParam(mode="local", enable_rerank=False, ll_keywords=["filter"]),
            config,
            hashing_kv=cache,
        )

    await run(_config(model))
    await run(_config(model, **{option: 0}))  # off: the existing key keeps hitting
    await run(_config(model, **{option: 3}))
    assert model.calls == 2
    assert len(cache.answer_keys()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["lexical_chunk_top_k", "max_chunks_per_doc"])
async def test_naive_query_option_on_partitions_the_answer_cache(option):
    model, cache = _Model(), _Cache()

    async def run(config):
        return await naive_query(
            "which filter removes noise?",
            _ChunksVDB(),
            QueryParam(mode="naive", enable_rerank=False),
            config,
            hashing_kv=cache,
            text_chunks_db=_TextChunks(config),
        )

    await run(_config(model))
    await run(_config(model, **{option: 0}))
    await run(_config(model, **{option: 3}))
    assert model.calls == 2
    assert len(cache.answer_keys()) == 2


# ---------------------------------------------------------------------------
# PGKVStorage.lexical_search (fake database)
# ---------------------------------------------------------------------------


class _FakeDB:
    def __init__(self, index_present=True):
        self.index_present = index_present
        self.calls: list[tuple[str, list]] = []

    async def query(self, sql, params=None, multirows=False):
        self.calls.append((sql, params))
        if "pg_indexes" in sql:
            return {"?column?": 1} if self.index_present else None
        return [
            {
                "id": "c1",
                "full_doc_id": "d1",
                "file_path": "x.md",
                "content": "c",
                "chunk_order_index": 0,
                "score": 3,
            }
        ]


def _storage(db, namespace=NameSpace.KV_STORE_TEXT_CHUNKS):
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = namespace
    storage.workspace = "ws"
    storage.db = db
    return storage


@pytest.mark.asyncio
async def test_storage_searches_with_workspace_query_and_limit():
    db = _FakeDB()
    rows = await _storage(db).lexical_search("okafor 2019", 6)
    assert db.calls[-1][1] == ["ws", "okafor 2019", 6, 2.0]
    assert rows[0]["score"] == 3.0 and isinstance(rows[0]["score"], float)


@pytest.mark.asyncio
async def test_storage_without_the_index_warns_once_and_finds_nothing(
    lightrag_log_records,
):
    db = _FakeDB(index_present=False)
    storage = _storage(db)
    assert await storage.lexical_search("okafor 2019", 6) == []
    assert await storage.lexical_search("okafor 2019", 6) == []
    assert len(db.calls) == 1  # one index check, no search
    assert (
        sum(
            "lexical chunk search is off" in r.getMessage()
            for r in lightrag_log_records
        )
        == 1
    )


@pytest.mark.asyncio
async def test_storage_ignores_other_namespaces_and_empty_queries():
    db = _FakeDB()
    assert await _storage(db, NameSpace.KV_STORE_FULL_DOCS).lexical_search("x", 3) == []
    assert await _storage(db).lexical_search("   ", 3) == []
    assert await _storage(db).lexical_search("x", 0) == []
    assert db.calls == []
