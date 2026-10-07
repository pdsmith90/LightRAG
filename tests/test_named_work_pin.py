"""NAMED_WORK_PIN: the chunks of the works a query names by author and year ride
through the rerank floor, the chunk_top_k cut and the token budget, first in the
context. The papers citing a work often match a question better than the work
itself, so without the pin the named work's chunks are the first to go.
Fixtures are invented.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.utils import pin_named_work_chunks, process_chunks_unified

pytestmark = pytest.mark.offline


def _chunk(cid: str, path: str, score: float, pinned: bool = False) -> dict:
    return {
        "chunk_id": cid,
        "content": f"text of {cid}",
        "file_path": path,
        "rerank_score": score,
        **({"pinned": True} if pinned else {}),
    }


CHUNKS = [
    _chunk("c1", "AAAA0009__citer.md", 0.9),
    _chunk("p1", "AAAA0001__named.md", 0.5, True),
    _chunk("p2", "AAAA0001__named.md", 0.4, True),
    _chunk("p3", "AAAA0001__named.md", 0.3, True),
    _chunk("q1", "AAAA0002__other.md", 0.2, True),
    _chunk("r1", "AAAA0003__third.md", 0.1, True),
]


def _ids(chunks):
    return [c["chunk_id"] for c in chunks]


def test_pin_moves_the_best_named_works_first_and_caps_them():
    out = pin_named_work_chunks(CHUNKS, max_per_work=2, max_works=2)
    assert _ids(out) == ["p1", "p2", "q1", "c1", "p3", "r1"]
    assert _ids(pin_named_work_chunks(CHUNKS, max_per_work=1, max_works=1)) == [
        "p1",
        "c1",
        "p2",
        "p3",
        "q1",
        "r1",
    ]


def test_pin_is_a_no_op_without_pinned_chunks():
    plain = [_chunk("a", "AAAA0001__x.md", 0.2), _chunk("b", "AAAA0002__y.md", 0.1)]
    assert pin_named_work_chunks(plain, max_per_work=2) is plain


# The citers outscore the named work: below the floor and past the cut, its one
# chunk is the first to go unless pinned.
CITERS_FIRST = [
    _chunk("c1", "AAAA0009__citer.md", 0.9),
    _chunk("c2", "AAAA0008__citer.md", 0.7),
    _chunk("p1", "AAAA0001__named.md", 0.5, True),
    _chunk("c3", "AAAA0007__citer.md", 0.65),
]


@pytest.mark.asyncio
async def test_unified_processing_keeps_pinned_chunks_through_floor_and_cut():
    param = QueryParam(mode="mix", enable_rerank=True, chunk_top_k=2)
    config = {"named_work_pin": True, "min_rerank_score": 0.6, "tokenizer": None}
    out = await process_chunks_unified("a question", list(CITERS_FIRST), param, config)
    assert _ids(out) == ["p1", "c1"]
    assert [c["id"] for c in out] == ["DC1", "DC2"]


@pytest.mark.asyncio
async def test_unified_processing_ignores_the_flag_when_the_option_is_off():
    param = QueryParam(mode="mix", enable_rerank=True, chunk_top_k=2)
    config = {"min_rerank_score": 0.6, "tokenizer": None}
    out = await process_chunks_unified("a question", list(CITERS_FIRST), param, config)
    assert _ids(out) == ["c1", "c2"]
