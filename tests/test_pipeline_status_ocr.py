"""GET /documents/pipeline_status interleaves Zotero corpus OCR progress.

The nightly corpus build can spend hours on one scanned PDF and writes nothing to
the pipeline until that document is done, so the endpoint folds what the build
leaves on disk (via ``zotero_ocr_progress``) into ``history_messages``.
``latest_message`` is left exactly as the pipeline set it; the web UI renders
``history_messages`` alone.

These tests pin the rules that make that safe:

- the OCR line is written into the SHARED history list, once per change, so it
  takes its place in time order -- later pipeline messages land below it -- and
  is not a per-request tail that always sits at the bottom,
- an unchanged line (same document, same page) is not written again by the next
  poll, so polling the dialog cannot fill the log with duplicates,
- a poll with nothing in flight writes nothing, and a broken helper costs the
  line, never the request.

Fixtures are a real (in-process) ``pipeline_status`` namespace and a fake
``get_ocr_progress_lines``; no Zotero, PDF, OCR service or bundle directory is
involved.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import SimpleNamespace

import pytest

# Importing the router loads ``lightrag.api.config`` which parses sys.argv via
# argparse. Stash argv so pytest's CLI flags don't trip the parser.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

OCR_LINE = "OCR in progress: x.pdf: page 1 of 2 (50%)"
NEXT_LINE = "OCR in progress: x.pdf: page 2 of 2 (100%)"
EXISTING_LINE = "Starting document processing"
PIPELINE_LINE = "Processing file 2/2: b.pdf"


@contextlib.asynccontextmanager
async def _status_session(
    monkeypatch, *, ocr_lines=(), helper=None, busy=False, latest_message=""
):
    """The endpoint bound to one real pipeline_status, for one or more polls.

    Every poll in the session sees the history its predecessors left behind,
    which is what the interleaving tests are about. Yields
    ``(endpoint, pipeline_status)`` so a test can read the shared list and log a
    pipeline message between polls.
    """
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        get_namespace_data,
        initialize_pipeline_status,
        initialize_share_data,
    )

    monkeypatch.setattr(
        _document_routes,
        "get_ocr_progress_lines",
        helper if helper is not None else (lambda *args, **kwargs: list(ocr_lines)),
    )

    initialize_share_data()
    try:
        await initialize_pipeline_status(workspace="")
        pipeline_status = await get_namespace_data("pipeline_status", workspace="")
        pipeline_status["busy"] = busy
        pipeline_status["latest_message"] = latest_message
        pipeline_status["history_messages"][:] = [EXISTING_LINE]

        router = _document_routes.create_document_routes(
            SimpleNamespace(workspace=""),
            SimpleNamespace(),
            api_key="test-key",
        )
        # The route's auth dependency only runs through FastAPI's resolver, so
        # calling the endpoint itself needs no client and no API key.
        endpoint = next(
            route.endpoint
            for route in router.routes
            if getattr(route, "path", None) == "/documents/pipeline_status"
        )
        yield endpoint, pipeline_status
    finally:
        finalize_share_data()


async def test_ocr_line_is_written_into_the_shared_history_once(monkeypatch):
    async with _status_session(monkeypatch, ocr_lines=[OCR_LINE]) as (
        endpoint,
        pipeline_status,
    ):
        response = await endpoint()

    assert response.history_messages == [EXISTING_LINE, OCR_LINE]
    # The shared list every other writer appends to, not a response-only tail.
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE, OCR_LINE]


async def test_an_unchanged_line_is_not_written_by_a_second_poll(monkeypatch):
    async with _status_session(monkeypatch, ocr_lines=[OCR_LINE]) as (
        endpoint,
        pipeline_status,
    ):
        first = await endpoint()
        second = await endpoint()

    assert first.history_messages == [EXISTING_LINE, OCR_LINE]
    assert second.history_messages == [EXISTING_LINE, OCR_LINE]
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE, OCR_LINE]


async def test_a_changed_line_is_written_where_it_happened(monkeypatch):
    """page 1, then a pipeline message, then page 2: time order, not a tail."""
    answers = [OCR_LINE]
    async with _status_session(
        monkeypatch, helper=lambda *args, **kwargs: list(answers)
    ) as (endpoint, pipeline_status):
        await endpoint()
        from lightrag.kg.shared_storage import append_pipeline_history

        append_pipeline_history(pipeline_status, PIPELINE_LINE)
        answers[:] = [NEXT_LINE]
        response = await endpoint()

        expected = [EXISTING_LINE, OCR_LINE, PIPELINE_LINE, NEXT_LINE]
        assert list(pipeline_status["history_messages"]) == expected
        assert response.history_messages == expected


async def test_a_quiet_poll_writes_nothing_and_the_next_line_still_lands(
    monkeypatch,
):
    answers = [OCR_LINE]
    async with _status_session(
        monkeypatch, helper=lambda *args, **kwargs: list(answers)
    ) as (endpoint, pipeline_status):
        await endpoint()
        answers[:] = []  # OCR finished, or its document went quiet
        quiet = await endpoint()
        answers[:] = [NEXT_LINE]  # a later run reports a new page
        resumed = await endpoint()

    assert quiet.history_messages == [EXISTING_LINE, OCR_LINE]
    assert resumed.history_messages == [EXISTING_LINE, OCR_LINE, NEXT_LINE]
    assert list(pipeline_status["history_messages"]) == [
        EXISTING_LINE,
        OCR_LINE,
        NEXT_LINE,
    ]


async def test_idle_pipeline_does_not_override_latest_message(monkeypatch):
    async with _status_session(monkeypatch, ocr_lines=[OCR_LINE]) as (
        endpoint,
        pipeline_status,
    ):
        response = await endpoint()

    # The OCR line lands in history only; nothing is promoted to latest_message.
    assert response.history_messages == [EXISTING_LINE, OCR_LINE]
    assert response.latest_message == ""
    assert pipeline_status["latest_message"] == ""


async def test_busy_pipeline_keeps_its_ingest_message_on_top(monkeypatch):
    ingest = "Processing file 3/9: a.pdf"
    async with _status_session(
        monkeypatch, ocr_lines=[OCR_LINE], busy=True, latest_message=ingest
    ) as (endpoint, pipeline_status):
        response = await endpoint()

    # The OCR line is still visible, but it does not hide what is running.
    assert OCR_LINE in response.history_messages
    assert response.latest_message == ingest
    assert pipeline_status["latest_message"] == ingest


async def test_empty_ocr_lines_write_nothing(monkeypatch):
    async with _status_session(monkeypatch, ocr_lines=[]) as (
        endpoint,
        pipeline_status,
    ):
        response = await endpoint()

    assert response.history_messages == [EXISTING_LINE]
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE]
    assert response.latest_message == ""


async def test_a_broken_helper_never_breaks_the_endpoint(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("helper bug")

    async with _status_session(monkeypatch, helper=_boom) as (
        endpoint,
        pipeline_status,
    ):
        response = await endpoint()

    assert response.history_messages == [EXISTING_LINE]
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE]
    assert response.latest_message == ""


async def test_the_response_list_is_a_copy_not_the_shared_history(monkeypatch):
    """No caller may alias the shared list through the response."""
    async with _status_session(monkeypatch, ocr_lines=[OCR_LINE]) as (
        endpoint,
        pipeline_status,
    ):
        response = await endpoint()

        assert response.history_messages is not pipeline_status["history_messages"]
