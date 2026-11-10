"""``adelete_by_doc_id(delete_llm_cache=True)`` keeps every cache row a chunk of
another document still lists.

An extraction cache row is keyed on the prompt, not on the chunk, so two
documents that share a chunk's text share one row: the second document's chunk
hits it and lists its key in its own ``llm_cache_list``. Deleting the first
document with its cache must not delete that row, or the surviving chunk's
reference dangles and its next rebuild or re-ingest finds nothing cached. Rows
only the deleted document listed are still deleted. Fixtures are invented.
"""

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

SHARED = "Atlas Harbor shelters the Borealis fleet."


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


def _input_text(prompt: str) -> str:
    marker = "---Input Text---\n```\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    return prompt[start : prompt.find("\n```", start)]


class _ExtractLLM:
    """Extraction LLM that records the input text of every call; any
    non-extraction call (a summary) fails the test."""

    def __init__(self):
        self.texts: list[str] = []

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **_):
        text = _input_text(prompt)
        if not text:
            raise AssertionError(f"unexpected non-extraction LLM call: {prompt[:80]}")
        self.texts.append(text)
        rows = []
        if "Atlas" in text:
            rows.append(
                "entity<|#|>Atlas Harbor<|#|>location<|#|>Atlas Harbor is a harbor."
            )
            rows.append(
                "entity<|#|>Borealis Fleet<|#|>organization<|#|>Borealis Fleet is a fleet."
            )
            rows.append(
                "relation<|#|>Atlas Harbor<|#|>Borealis Fleet<|#|>shelter"
                "<|#|>Atlas Harbor shelters the Borealis Fleet."
            )
        for word in ("Cygnus", "Draco"):
            if word in text:
                rows.append(
                    f"entity<|#|>{word} Station<|#|>location<|#|>{word} Station is a station."
                )
        rows.append("<|COMPLETE|>")
        return "\n".join(rows)


async def _make_rag(tmp_path, llm):
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"delete_shared_cache_{uuid4().hex}",
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=0,
        chunk_token_size=4000,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


FIRST = f"{SHARED}\n\nCygnus Station is in the first copy only."
SECOND = f"{SHARED}\n\nDraco Station is in the second copy only."


async def _insert(rag, text, doc_id):
    await rag.ainsert(
        text,
        split_by_character="\n\n",
        split_by_character_only=True,
        ids=[doc_id],
        file_paths=[f"{doc_id}.txt"],
    )


async def _cache_lists(rag, doc_id):
    chunk_ids = (await rag.doc_status.get_by_id(doc_id))["chunks_list"]
    rows = await rag.text_chunks.get_by_ids(chunk_ids)
    return [row["llm_cache_list"] for row in rows]


async def _existing(rag, keys):
    return [
        k for k, row in zip(keys, await rag.llm_response_cache.get_by_ids(keys)) if row
    ]


@pytest.mark.asyncio
async def test_deleting_the_writer_with_its_cache_keeps_rows_the_survivor_lists(
    tmp_path,
):
    llm = _ExtractLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, FIRST, "doc-first")
        await _insert(rag, SECOND, "doc-second")
        # the shared chunk hit the first copy's row: only the Draco chunk called
        assert len(llm.texts) == 3
        first_shared_keys, first_only_keys = await _cache_lists(rag, "doc-first")
        second_shared_keys, _ = await _cache_lists(rag, "doc-second")
        assert first_shared_keys and first_shared_keys == second_shared_keys

        result = await rag.adelete_by_doc_id("doc-first", delete_llm_cache=True)
        assert result.status == "success", result.message
        assert f"kept {len(first_shared_keys)}" in result.message

        # the survivor's rows are all still there; the deleted copy's own are gone
        assert await _existing(rag, second_shared_keys) == second_shared_keys
        assert await _existing(rag, first_only_keys) == []

        # re-ingesting the deleted copy finds the shared extraction cached:
        # only its own chunk, whose row was deleted, calls the LLM again
        llm.texts.clear()
        await _insert(rag, FIRST, "doc-first")
        assert len(llm.texts) == 1
        assert "Cygnus" in llm.texts[0]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_deleting_both_copies_with_their_cache_leaves_no_row(tmp_path):
    llm = _ExtractLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await _insert(rag, FIRST, "doc-first")
        await _insert(rag, SECOND, "doc-second")
        all_keys = [
            key
            for doc_id in ("doc-first", "doc-second")
            for keys in await _cache_lists(rag, doc_id)
            for key in keys
        ]
        assert len(set(all_keys)) == 3

        for doc_id in ("doc-first", "doc-second"):
            result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            assert result.status == "success", result.message

        # the second delete owns the shared rows once the first copy is gone
        assert await _existing(rag, sorted(set(all_keys))) == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_storage_that_cannot_answer_deletes_every_id_as_before(monkeypatch):
    import lightrag.lightrag as lightrag_module

    warnings = []
    monkeypatch.setattr(
        lightrag_module.logger, "warning", lambda msg, *a: warnings.append(msg % a)
    )
    rag = SimpleNamespace(text_chunks=object())
    deletable, kept = await LightRAG._split_shared_llm_cache_ids(
        rag, ["k1", "k2"], ["c1"]
    )
    assert (deletable, kept) == (["k1", "k2"], [])
    assert len(warnings) == 1
    assert "cannot report which LLM cache rows" in warnings[0]


@pytest.mark.asyncio
async def test_the_split_keeps_what_the_storage_reports_listed_elsewhere():
    seen = {}

    class _Chunks:
        async def cache_ids_listed_by_other_chunks(self, cache_ids, exclude):
            seen.update(cache_ids=cache_ids, exclude=exclude)
            return {"k2"}

    rag = SimpleNamespace(text_chunks=_Chunks())
    deletable, kept = await LightRAG._split_shared_llm_cache_ids(
        rag, ["k1", "k2", "k3"], ["c1", "c2"]
    )
    assert (deletable, kept) == (["k1", "k3"], ["k2"])
    assert seen == {"cache_ids": ["k1", "k2", "k3"], "exclude": ["c1", "c2"]}


@pytest.mark.asyncio
async def test_postgres_lookup_matches_list_elements_outside_the_excluded_chunks():
    from lightrag.kg.postgres_impl import PGKVStorage, SQL_TEMPLATES

    seen = {}

    class _DB:
        async def query(self, sql, params=None, multirows=False, **kw):
            seen.update(sql=sql, params=params, multirows=multirows)
            # asyncpg hands jsonb back as text; a list is accepted too
            return [
                {"llm_cache_list": '["k2", "k-other"]'},
                {"llm_cache_list": ["k3"]},
                {"llm_cache_list": "not json"},
            ]

    store = SimpleNamespace(namespace="text_chunks", workspace="ws", db=_DB())
    got = await PGKVStorage.cache_ids_listed_by_other_chunks(
        store, ["k3", "k1", "k2", "k1"], ["c1"]
    )
    assert got == {"k2", "k3"}
    assert seen["sql"] == SQL_TEMPLATES["cache_ids_listed_by_other_chunks_text_chunks"]
    assert seen["params"] == ["ws", ["k1", "k2", "k3"], ["c1"]]
    assert seen["multirows"] is True
    sql = " ".join(seen["sql"].split())
    assert "FROM LIGHTRAG_DOC_CHUNKS" in sql
    assert "workspace = $1" in sql
    assert "llm_cache_list ?| $2::text[]" in sql
    assert "NOT (id = ANY($3::text[]))" in sql

    assert (
        await PGKVStorage.cache_ids_listed_by_other_chunks(store, [], ["c1"]) == set()
    )
    other = SimpleNamespace(namespace="llm_response_cache", workspace="ws", db=_DB())
    assert (
        await PGKVStorage.cache_ids_listed_by_other_chunks(other, ["k1"], []) == set()
    )
