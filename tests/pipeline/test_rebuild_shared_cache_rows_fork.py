"""A KG rebuild reads each surviving chunk's cached extraction through that
chunk's own ``llm_cache_list``, whoever wrote the row.

An extraction cache row is keyed on the prompt (chunk text, heading context,
system prompt, LLM identity), not on the chunk, so two documents that share a
chunk's text share one row. The row's ``chunk_id`` names only the chunk that
first wrote it; a later chunk that hits it records the key in its own
``llm_cache_list`` and leaves the row alone. Deleting the document that wrote
the row must not cost the surviving copy its cached extraction: the rebuild
groups rows by the surviving chunk that references them. Fixtures are invented.
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.operate import _get_cached_extraction_results
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
    """Extraction LLM that numbers its calls, so a description says which
    call produced it; any non-extraction call (a summary) fails the test."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **_):
        text = _input_text(prompt)
        if not text:
            raise AssertionError(f"unexpected non-extraction LLM call: {prompt[:80]}")
        self.calls += 1
        n = self.calls
        rows = []
        if "Atlas" in text:
            rows.append(
                f"entity<|#|>Atlas Harbor<|#|>location<|#|>Atlas Harbor, extraction {n}."
            )
            rows.append(
                f"entity<|#|>Borealis Fleet<|#|>organization<|#|>Borealis Fleet, extraction {n}."
            )
            rows.append(
                "relation<|#|>Atlas Harbor<|#|>Borealis Fleet<|#|>shelter"
                f"<|#|>Atlas Harbor shelters the Borealis Fleet, extraction {n}."
            )
        for word in ("Cygnus", "Draco"):
            if word in text:
                rows.append(
                    f"entity<|#|>{word} Station<|#|>location<|#|>{word} Station, extraction {n}."
                )
        rows.append("<|COMPLETE|>")
        return "\n".join(rows)


async def _make_rag(tmp_path, llm):
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"shared_cache_{uuid4().hex}",
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


async def _chunks_of(rag, doc_id):
    return (await rag.doc_status.get_by_id(doc_id))["chunks_list"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delete_kwargs",
    [{}, {"keep_summaries": True}, {"delete_llm_cache": True}],
    ids=["plain", "keep_summaries", "delete_llm_cache"],
)
async def test_deleting_the_copy_that_wrote_shared_rows_rebuilds_from_the_survivor(
    tmp_path, delete_kwargs
):
    llm = _ExtractLLM()
    rag = await _make_rag(tmp_path, llm)
    try:
        await rag.ainsert(
            f"{SHARED}\n\nCygnus Station is in the first copy only.",
            split_by_character="\n\n",
            split_by_character_only=True,
            ids=["doc-first"],
            file_paths=["first.txt"],
        )
        calls_after_first = llm.calls
        await rag.ainsert(
            f"{SHARED}\n\nDraco Station is in the second copy only.",
            split_by_character="\n\n",
            split_by_character_only=True,
            ids=["doc-second"],
            file_paths=["second.txt"],
        )
        # the shared chunk hit the first copy's row; only the other chunk called
        assert llm.calls == calls_after_first + 1

        first_shared = (await _chunks_of(rag, "doc-first"))[0]
        second_shared = (await _chunks_of(rag, "doc-second"))[0]
        assert first_shared != second_shared
        first_keys = (await rag.text_chunks.get_by_id(first_shared))["llm_cache_list"]
        second_keys = (await rag.text_chunks.get_by_id(second_shared))["llm_cache_list"]
        assert first_keys == second_keys
        # the row still names the first copy's chunk: a hit does not re-own it
        row = await rag.llm_response_cache.get_by_id(first_keys[0])
        assert row["chunk_id"] == first_shared

        calls_before_delete = llm.calls
        result = await rag.adelete_by_doc_id("doc-first", **delete_kwargs)
        assert result.status == "success", result.message
        # the survivor's cached extraction is enough: no LLM call
        assert llm.calls == calls_before_delete

        node = await rag.chunk_entity_relation_graph.get_node("Atlas Harbor")
        assert node is not None
        assert node["description"] == "Atlas Harbor, extraction 1."
        assert node["source_id"] == second_shared
        assert node["file_path"] == "second.txt"
        edge = await rag.chunk_entity_relation_graph.get_edge(
            "Atlas Harbor", "Borealis Fleet"
        )
        assert edge is not None
        assert edge["description"] == (
            "Atlas Harbor shelters the Borealis Fleet, extraction 1."
        )
        assert edge["keywords"] == "shelter"
        assert edge["source_id"] == second_shared
        assert edge["file_path"] == "second.txt"
    finally:
        await rag.finalize_storages()


class _KV:
    def __init__(self, data):
        self.data = dict(data)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]


def _row(chunk_id, text, create_time, cache_type="extract"):
    return {
        "cache_type": cache_type,
        "chunk_id": chunk_id,
        "return": text,
        "create_time": create_time,
    }


@pytest.mark.asyncio
async def test_a_row_shared_by_two_requested_chunks_counts_for_each():
    # c-old wrote both rows and is gone; c1 and c2 hit them. Before, a row
    # counted only for the chunk its chunk_id named, so neither survivor got it.
    text_chunks = _KV(
        {
            "c1": {"content": "same", "llm_cache_list": ["k-initial", "k-glean"]},
            "c2": {"content": "same", "llm_cache_list": ["k-initial", "k-glean"]},
        }
    )
    cache = _KV(
        {
            "k-initial": _row("c-old", "initial", 10),
            "k-glean": _row("c-old", "glean", 11),
        }
    )
    results, chunk_data = await _get_cached_extraction_results(
        cache, {"c1", "c2"}, text_chunks_storage=text_chunks
    )
    assert results == {
        "c1": [("initial", 10), ("glean", 11)],
        "c2": [("initial", 10), ("glean", 11)],
    }
    assert set(chunk_data) == {"c1", "c2"}


@pytest.mark.asyncio
async def test_rows_sort_by_create_time_whoever_wrote_them():
    # c1's initial extraction hit an older chunk's row; its gleaning was a fresh
    # call it wrote itself. The list order is the reverse of create_time.
    text_chunks = _KV(
        {
            "c1": {"llm_cache_list": ["k-glean", "k-initial"]},
            "c2": {"llm_cache_list": ["k-own"]},
        }
    )
    cache = _KV(
        {
            "k-initial": _row("c-old", "initial", 5),
            "k-glean": _row("c1", "glean", 20),
            "k-own": _row("c2", "own", 7),
        }
    )
    results, _ = await _get_cached_extraction_results(
        cache, {"c1", "c2"}, text_chunks_storage=text_chunks
    )
    assert results["c1"] == [("initial", 5), ("glean", 20)]
    # chunks are ordered by their earliest row, as before
    assert list(results) == ["c1", "c2"]


@pytest.mark.asyncio
async def test_only_extract_rows_listed_by_a_requested_chunk_are_returned():
    text_chunks = _KV(
        {
            # a duplicated key, a multimodal analysis row, a key whose row is gone
            "c1": {"llm_cache_list": ["k1", "k1", "k-analysis", "k-missing"]},
            # not requested: its rows must not leak into the result
            "c9": {"llm_cache_list": ["k9"]},
        }
    )
    cache = _KV(
        {
            "k1": _row("c-old", "r1", 1),
            "k-analysis": _row(None, "analysis", 2, cache_type="analysis"),
            "k9": _row("c1", "r9", 3),
        }
    )
    results, _ = await _get_cached_extraction_results(
        cache, {"c1"}, text_chunks_storage=text_chunks
    )
    # k9 names c1 as its writer, but c1 does not reference it: not c1's
    assert results == {"c1": [("r1", 1)]}
