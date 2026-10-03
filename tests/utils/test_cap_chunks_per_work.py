"""MAX_CHUNKS_PER_DOC: at most N chunks of one work in a query's final context.

The cap runs in ``process_chunks_unified`` after reranking and the rerank-score
floor, before ``chunk_top_k``, so the slots a capped work would have taken go to
the next works in rerank order -- which needs the reranker to return every
candidate, not just the first ``chunk_top_k``. A work is one document, or every
copy of one paper filed under several Zotero items (``zotero_citations.work_key``).
"""

import json
import os
import tempfile

import pytest

import lightrag.zotero_citations as zotero_citations
from lightrag.base import QueryParam
from lightrag.utils import cap_chunks_per_work, process_chunks_unified

pytestmark = pytest.mark.offline

# Synthetic storage keys; two copies of one paper, plus two unrelated prefaces.
METADATA = {
    "ZZCOPY01": {
        "title": "Spatial filtering of urban noise maps",
        "doi": "10.5555/a",
    },
    "ZZCOPY02": {
        "title": "Spatial <i>filtering</i> of urban noise maps",
        "doi": "10.5555/b",
    },
    "ZZPREF01": {"title": "Preface", "doi": "10.5555/c"},
    "ZZPREF02": {"title": "Preface", "doi": "10.5555/d"},
}

_saved = None


def setup_module(module=None):
    global _saved
    _saved = {
        name: getattr(zotero_citations, name)
        for name in ("METADATA_PATH", "_meta", "_mtime", "_next_stat", "_warned")
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(METADATA, f)
    zotero_citations.METADATA_PATH = path
    zotero_citations._meta = {}
    zotero_citations._mtime = None
    zotero_citations._next_stat = 0.0
    zotero_citations._warned = False


def teardown_module(module=None):
    os.unlink(zotero_citations.METADATA_PATH)
    for name, value in _saved.items():
        setattr(zotero_citations, name, value)


def _chunk(chunk_id, file_path):
    return {
        "chunk_id": chunk_id,
        "content": f"text of {chunk_id}",
        "file_path": file_path,
    }


def test_cap_keeps_rank_order_and_n_chunks_per_document():
    chunks = [
        _chunk("a1", "a.md"),
        _chunk("a2", "a.md"),
        _chunk("b1", "b.md"),
        _chunk("a3", "a.md"),
        _chunk("c1", "c.md"),
        _chunk("b2", "b.md"),
        _chunk("b3", "b.md"),
    ]
    kept = cap_chunks_per_work(chunks, 2)
    assert [c["chunk_id"] for c in kept] == ["a1", "a2", "b1", "c1", "b2"]
    assert len(chunks) == 7  # input not mutated


def test_cap_of_zero_is_a_no_op():
    chunks = [_chunk("a1", "a.md"), _chunk("a2", "a.md")]
    assert cap_chunks_per_work(chunks, 0) is chunks


def test_copies_of_one_paper_count_as_one_work():
    chunks = [
        _chunk("x1", "ZZCOPY01__Spatial_filtering.md"),
        _chunk("y1", "ZZCOPY02__Spatial_filtering.md"),  # same title, other DOI
        _chunk("x2", "ZZCOPY01__Spatial_filtering.md"),
    ]
    assert [c["chunk_id"] for c in cap_chunks_per_work(chunks, 1)] == ["x1"]


def test_short_generic_titles_do_not_merge_works():
    chunks = [
        _chunk("p1", "ZZPREF01__Preface.md"),
        _chunk("q1", "ZZPREF02__Preface.md"),
    ]
    assert [c["chunk_id"] for c in cap_chunks_per_work(chunks, 1)] == ["p1", "q1"]


class _Reranker:
    """Keeps the incoming order and records the top_n it is asked for."""

    def __init__(self):
        self.top_n: list = []

    async def __call__(self, query, documents, top_n=None):
        self.top_n.append(top_n)
        scored = [
            {"index": i, "relevance_score": 1.0 - i / 100}
            for i in range(len(documents))
        ]
        return scored[:top_n] if top_n else scored


def _ranked_candidates():
    # Document a dominates the top of the ranking; d and e sit below chunk_top_k.
    return [
        _chunk("a1", "a.md"),
        _chunk("a2", "a.md"),
        _chunk("a3", "a.md"),
        _chunk("a4", "a.md"),
        _chunk("b1", "b.md"),
        _chunk("d1", "d.md"),
        _chunk("e1", "e.md"),
    ]


async def _process(max_chunks_per_doc):
    reranker = _Reranker()
    config = {"rerank_model_func": reranker, "min_rerank_score": 0.0}
    if max_chunks_per_doc is not None:
        config["max_chunks_per_doc"] = max_chunks_per_doc
    result = await process_chunks_unified(
        query="spatial filtering",
        unique_chunks=_ranked_candidates(),
        query_param=QueryParam(enable_rerank=True, chunk_top_k=4),
        global_config=config,
    )
    return reranker, [c["chunk_id"] for c in result]


@pytest.mark.asyncio
async def test_capped_slots_are_refilled_from_below_chunk_top_k():
    reranker, kept = await _process(1)
    assert reranker.top_n == [7]  # every candidate scored and returned
    assert kept == ["a1", "b1", "d1", "e1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, 0])
async def test_cap_off_leaves_rerank_and_chunk_top_k_as_they_were(flag):
    reranker, kept = await _process(flag)
    assert reranker.top_n == [4]
    assert kept == ["a1", "a2", "a3", "a4"]
