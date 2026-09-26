"""In ``naive`` mode, ``drop_bibliography_chunks`` frees the context slots that
reference-list chunks held but does not refill them.

``_get_vector_context`` asks the vector store for exactly ``chunk_top_k``
candidates, and ``naive`` has no other source, so prose ranked below
``chunk_top_k`` never becomes a candidate: the filter leaves a smaller context,
the same way ``min_rerank_score`` does. This is an accepted residue, recorded in
docs/design/BibliographyChunkFilter.md; a change that over-fetches to refill the
slots must update that note together with this test.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import naive_query
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


BIBLIOGRAPHY = """LUHN H. P. 1957. A statistical approach to mechanized encoding and searching of literary information. IBM J. Res. Dev. 1, 309–317.
MARON M. E. & KUHNS J. L. 1960. On relevance, probabilistic indexing and information retrieval. J. ACM 7, 216–244.
SALTON G., WONG A. & YANG C. S. 1975. A vector space model for automatic indexing. Commun. ACM 18, 613–620.
ROBERTSON S. E. 1977. The probability ranking principle in IR. J. Doc. 33, 294–304.
SALTON G. & McGILL M. J. 1983. Introduction to Modern Information Retrieval. McGraw-Hill, New York."""


class _RankedChunksVDB:
    """Three reference lists outrank five prose chunks; honours ``top_k``."""

    cosine_better_than_threshold = 0.0

    def __init__(self):
        self.top_k_requests: list[int] = []
        self.ranked = [
            {"id": f"bib{i}", "content": BIBLIOGRAPHY, "file_path": f"b{i}.pdf"}
            for i in range(3)
        ] + [
            {
                "id": f"prose{i}",
                "content": f"Prose paragraph {i}.",
                "file_path": "p.pdf",
            }
            for i in range(5)
        ]

    async def query(self, _query, top_k, query_embedding=None):
        self.top_k_requests.append(top_k)
        return self.ranked[:top_k]


async def _context_chunk_ids(drop_bibliography_chunks: bool, vdb) -> list[str]:
    async def unused_model(*_args, **_kwargs):
        raise AssertionError("only_need_context must not call the model")

    result = await naive_query(
        "which retriever works best?",
        vdb,
        QueryParam(
            mode="naive", chunk_top_k=5, enable_rerank=False, only_need_context=True
        ),
        {
            "tokenizer": Tokenizer("fake", _FakeTokenizerImpl()),
            "role_llm_funcs": {"query": unused_model},
            "addon_params": {"language": "en"},
            "drop_bibliography_chunks": drop_bibliography_chunks,
        },
    )
    return [chunk["chunk_id"] for chunk in result.raw_data["data"]["chunks"]]


@pytest.mark.asyncio
async def test_naive_filter_shrinks_context_without_refilling_it():
    vdb = _RankedChunksVDB()

    off = await _context_chunk_ids(False, vdb)
    on = await _context_chunk_ids(True, vdb)

    assert off == ["bib0", "bib1", "bib2", "prose0", "prose1"]
    # The three freed slots stay empty: prose2..prose4 are never fetched.
    assert on == ["prose0", "prose1"]
    assert vdb.top_k_requests == [5, 5]
