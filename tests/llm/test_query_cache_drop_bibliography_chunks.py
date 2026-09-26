"""``drop_bibliography_chunks`` must partition the query-answer cache.

The filter changes which chunks reach the answer prompt, so an answer generated
with it off is not interchangeable with one generated with it on: without a key
component, enabling the filter would keep serving every previously cached
answer built from reference-list context. The component is only present when
the filter is on, so entries written with it off -- including every entry
written before the option existed -- keep their key and keep hitting.

``kg_query`` keys on ``text_chunks_db.global_config`` (the dict its chunk
processing reads); ``naive_query`` keys on the ``global_config`` it is handed.
"""

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import Tokenizer


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _Cache:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.store = {}

    async def get_by_id(self, key):
        return self.store.get(key)

    async def upsert(self, entries):
        self.store.update(entries)

    def answer_keys(self) -> list[str]:
        return [key for key in self.store if ":query:" in key]


class _TextChunks:
    def __init__(self, global_config):
        self.global_config = global_config


class _Model:
    def __init__(self):
        self.calls = 0
        self.system_prompts: list[str] = []

    async def __call__(self, *_args, **kwargs):
        self.calls += 1
        self.system_prompts.append(kwargs.get("system_prompt") or "")
        return f"answer-{self.calls}"


PROSE = (
    "Dense retrievers outperform sparse ones on open-domain question answering "
    "(Karpukhin et al., 2020); see doi:10.18653/v1/2020.emnlp-main.550."
)
BIBLIOGRAPHY = """LUHN H. P. 1957. A statistical approach to mechanized encoding and searching of literary information. IBM J. Res. Dev. 1, 309–317.
MARON M. E. & KUHNS J. L. 1960. On relevance, probabilistic indexing and information retrieval. J. ACM 7, 216–244.
SALTON G., WONG A. & YANG C. S. 1975. A vector space model for automatic indexing. Commun. ACM 18, 613–620.
ROBERTSON S. E. 1977. The probability ranking principle in IR. J. Doc. 33, 294–304.
SALTON G. & McGILL M. J. 1983. Introduction to Modern Information Retrieval. McGraw-Hill, New York."""


def _config(model, **extra) -> dict:
    return {
        "tokenizer": Tokenizer("fake", _FakeTokenizerImpl()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        **extra,
    }


# ---------------------------------------------------------------------------
# kg_query
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_kg_context(monkeypatch):
    async def fake_keywords(*_args, **_kwargs):
        return "", "retrieval"

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(context="context", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)


async def _kg(config, cache):
    return await kg_query(
        "which retriever works best?",
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="local", enable_rerank=False, ll_keywords=["retrieval"]),
        config,
        hashing_kv=cache,
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_filter_on_partitions_answer_cache(stub_kg_context):
    model = _Model()
    cache = _Cache()

    first = await _kg(_config(model), cache)
    second = await _kg(_config(model, drop_bibliography_chunks=True), cache)

    assert (first.content, second.content) == ("answer-1", "answer-2")
    assert model.calls == 2
    assert len(cache.answer_keys()) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_filter_off_keeps_the_existing_key(stub_kg_context):
    """Absent (every pre-existing entry) and False must produce one key."""
    model = _Model()
    cache = _Cache()

    first = await _kg(_config(model), cache)
    second = await _kg(_config(model, drop_bibliography_chunks=False), cache)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(cache.answer_keys()) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_keys_on_the_storage_snapshot(stub_kg_context):
    """The key follows ``text_chunks_db.global_config``, which is what chunk
    processing reads in KG modes, not the per-call ``global_config``."""
    model = _Model()
    cache = _Cache()
    query_param = QueryParam(
        mode="local", enable_rerank=False, ll_keywords=["retrieval"]
    )

    async def run(snapshot_flag: bool, call_flag: bool):
        return await kg_query(
            "which retriever works best?",
            None,
            None,
            None,
            _TextChunks(_config(model, drop_bibliography_chunks=snapshot_flag)),
            query_param,
            _config(model, drop_bibliography_chunks=call_flag),
            hashing_kv=cache,
        )

    await run(snapshot_flag=False, call_flag=True)
    await run(snapshot_flag=True, call_flag=True)

    assert model.calls == 2
    assert len(cache.answer_keys()) == 2


# ---------------------------------------------------------------------------
# naive_query (end to end through process_chunks_unified)
# ---------------------------------------------------------------------------


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {"id": "c-prose", "content": PROSE, "file_path": "a.pdf"},
            {"id": "c-refs", "content": BIBLIOGRAPHY, "file_path": "a.pdf"},
        ]


async def _naive(config, cache):
    return await naive_query(
        "which retriever works best?",
        _ChunksVDB(),
        QueryParam(mode="naive", enable_rerank=False),
        config,
        hashing_kv=cache,
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_filter_on_removes_references_and_misses_old_entry():
    model = _Model()
    cache = _Cache()

    off = await _naive(_config(model), cache)
    on = await _naive(_config(model, drop_bibliography_chunks=True), cache)

    assert (off.content, on.content) == ("answer-1", "answer-2")
    assert len(cache.answer_keys()) == 2
    off_prompt, on_prompt = model.system_prompts
    assert "Dense retrievers outperform" in off_prompt
    assert "Dense retrievers outperform" in on_prompt
    assert "probability ranking principle" in off_prompt
    assert "probability ranking principle" not in on_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_filter_off_keeps_the_existing_key():
    model = _Model()
    cache = _Cache()

    first = await _naive(_config(model), cache)
    second = await _naive(_config(model, drop_bibliography_chunks=False), cache)

    assert first.content == second.content == "answer-1"
    assert model.calls == 1
    assert len(cache.answer_keys()) == 1
