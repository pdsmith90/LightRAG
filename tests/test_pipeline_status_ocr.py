"""GET /documents/pipeline_status surfaces Zotero corpus OCR progress.

The nightly corpus build can spend hours on one scanned PDF and writes nothing to
the pipeline until that document is done, so the endpoint folds what the build
leaves on disk (via ``zotero_ocr_progress``) into the response: the lines are
appended to ``history_messages`` only. ``latest_message`` is left exactly as the
pipeline set it; the web UI renders ``history_messages`` alone.

These tests pin the two things that make that safe:

- the lines go into the per-request COPY of the status entry, never into the
  shared Manager history list (polling the dialog must not grow it), and
- an active ingest message is not hidden behind an OCR line, because no OCR line
  is ever promoted over ``latest_message``.

Fixtures are a real (in-process) ``pipeline_status`` namespace and a fake
``get_ocr_progress_lines``; no Zotero, PDF, OCR service or bundle directory is
involved.
"""

from __future__ import annotations

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
EXISTING_LINE = "Starting document processing"


async def _call_pipeline_status(
    monkeypatch, *, busy, latest_message, ocr_lines, helper=None
):
    """Run the endpoint against a real pipeline_status.

    Returns ``(response, pipeline_status)`` so a test can assert the SHARED dict
    was left alone as well as what the response carried.
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
        return await endpoint(), pipeline_status
    finally:
        finalize_share_data()


async def test_ocr_line_lands_in_history_and_never_in_the_shared_list(monkeypatch):
    response, pipeline_status = await _call_pipeline_status(
        monkeypatch, busy=False, latest_message="", ocr_lines=[OCR_LINE]
    )

    assert response.history_messages == [EXISTING_LINE, OCR_LINE]
    # The copy was mutated, not the Manager list every other writer shares.
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE]


async def test_idle_pipeline_does_not_override_latest_message(monkeypatch):
    response, pipeline_status = await _call_pipeline_status(
        monkeypatch, busy=False, latest_message="", ocr_lines=[OCR_LINE]
    )

    # The OCR line lands in history only; nothing is promoted to latest_message.
    assert response.history_messages == [EXISTING_LINE, OCR_LINE]
    assert response.latest_message == ""
    assert pipeline_status["latest_message"] == ""


async def test_busy_pipeline_keeps_its_ingest_message_on_top(monkeypatch):
    ingest = "Processing file 3/9: a.pdf"
    response, pipeline_status = await _call_pipeline_status(
        monkeypatch, busy=True, latest_message=ingest, ocr_lines=[OCR_LINE]
    )

    # The OCR line is still visible, but it does not hide what is running.
    assert OCR_LINE in response.history_messages
    assert response.latest_message == ingest
    assert pipeline_status["latest_message"] == ingest
    assert list(pipeline_status["history_messages"]) == [EXISTING_LINE]


async def test_empty_ocr_lines_change_nothing(monkeypatch):
    response, _ = await _call_pipeline_status(
        monkeypatch, busy=False, latest_message="", ocr_lines=[]
    )

    assert response.history_messages == [EXISTING_LINE]
    assert response.latest_message == ""


async def test_a_broken_helper_never_breaks_the_endpoint(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("helper bug")

    response, _ = await _call_pipeline_status(
        monkeypatch, busy=False, latest_message="", ocr_lines=[], helper=_boom
    )

    assert response.history_messages == [EXISTING_LINE]
    assert response.latest_message == ""


async def test_existing_history_is_not_mutated_in_place(monkeypatch):
    """The response list must be a NEW list, so no caller can alias the shared one."""
    response, pipeline_status = await _call_pipeline_status(
        monkeypatch, busy=False, latest_message="", ocr_lines=[OCR_LINE]
    )

    assert response.history_messages is not pipeline_status["history_messages"]
    assert len(pipeline_status["history_messages"]) == 1
