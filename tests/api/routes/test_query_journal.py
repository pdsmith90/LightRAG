"""QUERY_JOURNAL_FILE: one JSON line per /query and /query/stream request.

Each line records the prompt, the settings retrieval was given (after the budget
ceiling), the keywords it searched with, the answer exactly as the client
received it (sources block included) and the sources, so queries made through
the web UI can be reviewed later. These tests drive the real route handlers with
a fake ``rag.aquery_llm`` -- no network, no database.

pytest-compatible; also runs as a script:
python tests/api/routes/test_query_journal.py
"""

import asyncio
import importlib
import json
import os
import sys
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from lightrag.base import QueryParam
from lightrag.zotero_citations import format_reference_block

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

QUERY = "How accurate are reanalysis temperatures?"
PROSE = (
    "Reanalysis temperatures agree with stations once elevation is accounted for [1]."
)
REFERENCES = [
    {"reference_id": "1", "file_path": "ZZTEST01__Field_report_one.md"},
    {"reference_id": "2", "file_path": "ZZTEST02__Field_report_two.md"},
]
KEYWORDS = {"high_level": ["reanalysis accuracy"], "low_level": ["temperature"]}


class _FakeRag:
    """Stands in for LightRAG at the ``aquery_llm`` boundary the routes call."""

    def __init__(self, *, content=None, chunks=None, fail=None, break_stream=None):
        self.content = content
        self.chunks = chunks
        self.fail = fail
        self.break_stream = break_stream

    async def aquery_llm(self, query, param=None, progress_callback=None):
        if self.fail:
            raise self.fail
        if self.chunks is not None:

            async def _stream():
                for chunk in self.chunks:
                    yield chunk
                if self.break_stream:
                    raise self.break_stream

            llm_response = {"is_streaming": True, "response_iterator": _stream()}
        else:
            llm_response = {"is_streaming": False, "content": self.content}
        return {
            "llm_response": llm_response,
            "data": {"references": [dict(r) for r in REFERENCES], "chunks": []},
            "metadata": {"keywords": KEYWORDS},
        }


def _journal_path():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)  # the route creates it on first write
    return path


def _entries(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _call(rag, path, journal, ceiling=False, **body):
    """Run one route handler; /query returns its JSON, /query/stream its lines."""
    router = create_query_routes(
        rag, enable_query_budget_ceiling=ceiling, query_journal_file=journal
    )
    endpoint = next(r.endpoint for r in router.routes if r.path == path)
    request = QueryRequest(query=QUERY, **body)

    async def _run():
        response = await endpoint(request)
        if path == "/query":
            return response.model_dump()
        return [json.loads(line) async for line in response.body_iterator]

    return asyncio.run(_run())


def test_query_answer_is_journaled_as_delivered():
    journal = _journal_path()
    configured = QueryParam().top_k
    try:
        body = _call(
            _FakeRag(content=PROSE),
            "/query",
            journal,
            ceiling=True,
            top_k=configured + 5,
        )

        (entry,) = _entries(journal)
        assert entry["endpoint"] == "/query"
        assert entry["query"] == QUERY
        assert entry["response"] == body["response"]
        assert entry["response"] == PROSE + format_reference_block(REFERENCES)
        assert entry["params"]["top_k"] == configured  # as applied, after the ceiling
        assert entry["params"]["mode"] == "mix"
        assert entry["keywords"] == KEYWORDS
        assert entry["references"] == REFERENCES
        assert entry["complete"] is True and entry["error"] is None
        assert entry["client"] is None  # no HTTP request when called directly
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_streamed_answer_is_journaled_once_complete():
    journal = _journal_path()
    rag = _FakeRag(chunks=[PROSE[:20], PROSE[20:]])
    try:
        lines = _call(rag, "/query/stream", journal, stream=True)

        streamed = "".join(line.get("response", "") for line in lines)
        (entry,) = _entries(journal)
        assert entry["endpoint"] == "/query/stream"
        assert entry["response"] == streamed
        assert streamed == PROSE + format_reference_block(REFERENCES)
        assert entry["references"] == REFERENCES
        assert entry["keywords"] == KEYWORDS
        assert entry["complete"] is True
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_sources_are_journaled_even_when_the_client_asked_for_none():
    journal = _journal_path()
    try:
        _call(
            _FakeRag(chunks=[PROSE]),
            "/query/stream",
            journal,
            stream=True,
            include_references=False,
        )

        (entry,) = _entries(journal)
        assert entry["references"] == REFERENCES
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_a_client_that_disconnects_leaves_an_incomplete_entry():
    journal = _journal_path()
    rag = _FakeRag(chunks=[PROSE[:20], PROSE[20:]])
    router = create_query_routes(rag, query_journal_file=journal)
    endpoint = next(r.endpoint for r in router.routes if r.path == "/query/stream")

    async def _run():
        # No references line and no sources rewrite, so chunks pass verbatim.
        request = QueryRequest(query=QUERY, stream=True, include_references=False)
        response = await endpoint(request)
        it = response.body_iterator
        await it.__anext__()  # the first chunk of the answer
        await it.aclose()  # the client went away

    try:
        asyncio.run(_run())

        (entry,) = _entries(journal)
        assert entry["complete"] is False
        assert entry["response"] == PROSE[:20]
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_a_disconnect_still_cancels_the_running_query():
    # include_progress runs the query as a task; closing the response must
    # still cancel it promptly, journal or not.
    journal = _journal_path()
    seen = {"cancelled": False}

    class _SlowRag:
        async def aquery_llm(self, query, param=None, progress_callback=None):
            await progress_callback("extracting_keywords")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["cancelled"] = True
                raise

    router = create_query_routes(_SlowRag(), query_journal_file=journal)
    endpoint = next(r.endpoint for r in router.routes if r.path == "/query/stream")

    async def _run():
        request = QueryRequest(query=QUERY, stream=True, include_progress=True)
        response = await endpoint(request)
        it = response.body_iterator
        await it.__anext__()  # the progress line
        await it.aclose()  # the client went away
        return seen["cancelled"]

    try:
        assert asyncio.run(_run()) is True

        (entry,) = _entries(journal)
        assert entry["complete"] is False
        assert entry["keywords"] is None  # the query never produced a result
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_an_error_mid_stream_is_journaled_as_incomplete():
    journal = _journal_path()
    try:
        lines = _call(
            _FakeRag(chunks=[PROSE], break_stream=RuntimeError("stream broke")),
            "/query/stream",
            journal,
            stream=True,
        )

        assert {"error": "stream broke"} in lines
        (entry,) = _entries(journal)
        assert entry["error"] == "stream broke"
        assert entry["complete"] is False
        assert entry["response"].startswith(PROSE)
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_a_failed_query_is_journaled_with_its_error():
    journal = _journal_path()
    try:
        for path in ("/query", "/query/stream"):
            try:
                _call(_FakeRag(fail=RuntimeError("LLM unavailable")), path, journal)
            except HTTPException as e:
                assert e.status_code == 500
            else:
                raise AssertionError(f"{path} did not fail")

        entries = _entries(journal)
        assert [e["endpoint"] for e in entries] == ["/query", "/query/stream"]
        assert all(e["error"] == "LLM unavailable" for e in entries)
        assert all(e["complete"] is False and e["response"] == "" for e in entries)
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


def test_no_journal_file_means_nothing_is_written():
    journal = _journal_path()
    for path in ("/query", "/query/stream"):
        _call(_FakeRag(content=PROSE), path, None)
    assert not os.path.exists(journal)


def test_an_unwritable_journal_does_not_fail_the_query():
    journal = os.path.join(tempfile.gettempdir(), "no-such-dir-for-journal", "q.jsonl")
    body = _call(_FakeRag(content=PROSE), "/query", journal)
    lines = _call(_FakeRag(chunks=[PROSE]), "/query/stream", journal, stream=True)
    assert body["response"].startswith(PROSE)
    assert "".join(line.get("response", "") for line in lines).startswith(PROSE)


def test_the_http_request_supplies_client_and_user_agent():
    journal = _journal_path()
    app = FastAPI()
    app.include_router(
        create_query_routes(_FakeRag(chunks=[PROSE]), query_journal_file=journal)
    )
    try:
        r = TestClient(app).post(
            "/query/stream",
            json={"query": QUERY, "stream": True},
            headers={"User-Agent": "journal-test/1.0"},
        )

        assert r.status_code == 200
        (entry,) = _entries(journal)
        assert entry["user_agent"] == "journal-test/1.0"
        assert entry["client"]  # TestClient's peer, e.g. "testclient"
        assert entry["response"].startswith(PROSE)
    finally:
        if os.path.exists(journal):
            os.unlink(journal)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
