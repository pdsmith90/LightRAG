"""The compiled reference block and the budget ceiling on /query and /query/stream.

The answering LLM is given only corpus file paths, so the reference section it
writes is unreliable; the query routes strip it and append a ``### Sources``
block compiled from the retrieval result and zotero_metadata.json. With
ENABLE_QUERY_BUDGET_CEILING the configured retrieval budgets are also
per-request maxima. These tests drive the real route handlers with a fake
``rag.aquery_llm`` and a temporary metadata file -- no network, no database.

The ceiling tests compare against the ``QueryParam`` defaults, which read TOP_K,
CHUNK_TOP_K and MAX_*_TOKENS from the environment and from a .env file in the
working directory, so they hold under whatever values the server is configured
with.

pytest-compatible; also runs as a script:
python tests/api/routes/test_query_reference_block.py
"""

import asyncio
import importlib
import json
import os
import sys
import tempfile

import lightrag.zotero_citations as zotero_citations
from lightrag.base import QueryParam

try:
    import pytest
except ImportError:  # the script runner below needs no pytest
    pytest = None
else:
    pytestmark = pytest.mark.offline

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_qr = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _qr.QueryRequest
create_query_routes = _qr.create_query_routes

# Synthetic storage keys; the bibliographic data is public.
METADATA = {
    "ZZTEST01": {
        "authors": ["Gutenberg B.", "Richter C. F."],
        "year": "1944",
        "title": "Frequency of earthquakes in California",
        "publication": "Bulletin of the Seismological Society of America",
        "doi": "10.1785/BSSA0340040185",
    },
    "ZZTEST02": {
        "authors": [
            "Hersbach H.",
            "Bell B.",
            "Berrisford P.",
            "Hirahara S.",
            "Horányi A.",
        ],
        "year": "2020",
        "title": "The ERA5 global reanalysis",
        "publication": "Quarterly Journal of the Royal Meteorological Society",
        "doi": "10.1002/qj.3803",
    },
}

# Retrieval result in reference_id order; the third key is not in the metadata.
REFERENCES = [
    {"reference_id": "1", "file_path": "ZZTEST01__Gutenberg_1944_Frequency.md"},
    {"reference_id": "2", "file_path": "ZZTEST02__Hersbach_2020_ERA5.md"},
    {"reference_id": "3", "file_path": "ZZTEST03__Unindexed_field_report.md"},
]

SOURCES_BLOCK = (
    "\n\n### Sources\n\n"
    "- [1] Gutenberg B., Richter C. F. (1944). Frequency of earthquakes in "
    "California. Bulletin of the Seismological Society of America. "
    "https://doi.org/10.1785/BSSA0340040185\n"
    "- [2] Hersbach H., Bell B., Berrisford P. et al. (2020). The ERA5 global "
    "reanalysis. Quarterly Journal of the Royal Meteorological Society. "
    "https://doi.org/10.1002/qj.3803\n"
    "- [3] Unindexed field report\n"
)

# Cites only [2]; every retrieved source must still be listed.
PROSE = (
    "Reanalysis near-surface temperatures agree with station observations "
    "once the difference in elevation is accounted for [2]."
)
INVENTED = (
    "\n\n### References\n\n- [1] Smith J. (2019). An invented paper. doi:10.0000/x\n"
)

_saved_citation_state = None


def setup_module(module=None):
    global _saved_citation_state
    _saved_citation_state = {
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
    for name, value in _saved_citation_state.items():
        setattr(zotero_citations, name, value)


class _FakeRag:
    """Stands in for LightRAG at the ``aquery_llm`` boundary the routes call."""

    def __init__(self, *, content=None, chunks=None, references=(), **flags):
        self.content = content
        self.chunks = chunks
        self.references = references
        self.flags = flags
        self.params = []

    async def aquery_llm(self, query, param=None, progress_callback=None):
        self.params.append(param)
        if self.chunks is not None:

            async def _stream():
                for chunk in self.chunks:
                    yield chunk

            llm_response = {"is_streaming": True, "response_iterator": _stream()}
        else:
            llm_response = {"is_streaming": False, "content": self.content}
        llm_response.update(self.flags)
        return {
            "llm_response": llm_response,
            "data": {"references": [dict(r) for r in self.references], "chunks": []},
        }


def _call(rag, path, ceiling=False, **body):
    """Run one route handler; /query returns its JSON, /query/stream its lines.

    ``ceiling`` is the router's ENABLE_QUERY_BUDGET_CEILING switch.
    """
    router = create_query_routes(rag, enable_query_budget_ceiling=ceiling)
    endpoint = next(r.endpoint for r in router.routes if r.path == path)
    request = QueryRequest(query="How accurate are reanalysis temperatures?", **body)

    async def _run():
        response = await endpoint(request)
        if path == "/query":
            return response.model_dump()
        return [json.loads(line) async for line in response.body_iterator]

    return asyncio.run(_run())


def _streamed_text(lines):
    return "".join(line["response"] for line in lines if "response" in line)


BUDGET_FIELDS = (
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
)


def _configured():
    """The server's values: what an omitted budget field resolves to."""
    defaults = QueryParam()
    return {name: getattr(defaults, name) for name in BUDGET_FIELDS}


def _budget(param):
    return {name: getattr(param, name) for name in BUDGET_FIELDS}


def test_query_replaces_invented_references_with_every_retrieved_source():
    rag = _FakeRag(content=PROSE + INVENTED, references=REFERENCES)

    body = _call(rag, "/query")

    assert body["response"] == PROSE + SOURCES_BLOCK
    assert body["llm_generated"] is True
    assert [r["reference_id"] for r in body["references"]] == ["1", "2", "3"]


def test_stream_catches_a_heading_split_across_chunks():
    # The first chunk is longer than the stripper's holdback, so some prose is
    # sent before the heading arrives -- and none of the heading is.
    chunks = [PROSE, "\n\n##", "# Refer", "ences\n\n- [1] Smith J. (2019). Invented.\n"]
    rag = _FakeRag(chunks=chunks, references=REFERENCES)

    lines = _call(rag, "/query/stream", stream=True)

    assert lines[0] == {"references": REFERENCES}
    assert _streamed_text(lines) == PROSE + SOURCES_BLOCK
    sent = [line["response"] for line in lines if "response" in line]
    assert PROSE.startswith(sent[0]) and sent[0] != PROSE
    assert not any("Refer" in s or "Smith" in s for s in sent)


def test_stream_cache_hit_is_rewritten_on_its_single_line():
    # A cached answer comes back as a plain string even when stream=true.
    rag = _FakeRag(content=PROSE + INVENTED, references=REFERENCES)

    lines = _call(rag, "/query/stream", stream=True)

    assert len(lines) == 1
    assert lines[0]["response"] == PROSE + SOURCES_BLOCK
    assert lines[0]["llm_generated"] is True
    assert lines[0]["references"] == REFERENCES


def test_invented_section_is_stripped_when_retrieval_returned_nothing():
    for heading in ("### References", "### Sources"):
        answer = (
            PROSE + f"\n\n{heading}\n\n- [2] SmoothFilter (Reference: Contextual)\n"
        )

        body = _call(_FakeRag(content=answer), "/query")
        lines = _call(_FakeRag(chunks=[answer]), "/query/stream", stream=True)

        assert body["response"] == PROSE, heading
        assert _streamed_text(lines) == PROSE, heading


def test_sources_of_error_prose_is_not_mistaken_for_a_reference_section():
    prose = (
        PROSE
        + "\n\n### Sources of error\n\nStation moves and instrument changes dominate."
    )
    rag = _FakeRag(content=prose + INVENTED, references=REFERENCES)

    body = _call(rag, "/query")

    assert body["response"] == prose + SOURCES_BLOCK

    bare = _call(_FakeRag(content=prose), "/query")
    assert bare["response"] == prose


def test_only_need_context_output_passes_through_untouched():
    context = "-----Document Chunks-----\n" + INVENTED
    for path in ("/query", "/query/stream"):
        rag = _FakeRag(content=context, references=REFERENCES, llm_generated=False)

        result = _call(rag, path, only_need_context=True)

        text = result["response"] if path == "/query" else result[0]["response"]
        assert text == context, path


def test_ceiling_on_lowers_budgets_above_the_configured_values():
    above = {name: value + 1 for name, value in _configured().items()}
    for path in ("/query", "/query/stream"):
        rag = _FakeRag(content=PROSE)

        _call(rag, path, ceiling=True, **above)

        (param,) = rag.params
        assert _budget(param) == _configured(), path


def test_ceiling_on_honours_budgets_at_or_below_the_configured_values():
    below = {name: max(1, value // 2) for name, value in _configured().items()}
    for body in (below, _configured(), {}):
        for path in ("/query", "/query/stream"):
            rag = _FakeRag(content=PROSE)

            _call(rag, path, ceiling=True, **body)

            (param,) = rag.params
            assert _budget(param) == {**_configured(), **body}, (path, body)


def test_ceiling_off_passes_budgets_above_the_configured_values_through():
    above = {name: value + 1 for name, value in _configured().items()}
    for path in ("/query", "/query/stream"):
        rag = _FakeRag(content=PROSE)

        _call(rag, path, **above)

        (param,) = rag.params
        assert _budget(param) == above, path


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    setup_module()
    try:
        for t in tests:
            t()
            print("ok  ", t.__name__)
    finally:
        teardown_module()
    print(f"{len(tests)} tests passed")
