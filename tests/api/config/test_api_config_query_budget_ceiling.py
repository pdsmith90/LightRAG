"""``ENABLE_QUERY_BUDGET_CEILING``: parsing, wiring into the query routes, /health.

The clamp itself is covered in ``tests/api/routes/test_query_budget_ceiling.py``.
These tests pin the path from the environment to it: the switch defaults to off,
``create_app`` hands the parsed value to ``create_query_routes``, and an
authenticated ``/health`` reports it next to the other query settings.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.api.config import parse_args

pytestmark = pytest.mark.offline

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "ENABLE_QUERY_BUDGET_CEILING",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    # Import first: config loads .env at module import time. Clearing before
    # this import would let the loader immediately repopulate the variables.
    import lightrag.api.config as config

    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    # Part of the minimal viable server config since create_app began
    # refusing to start without a named embedding model.
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3:latest")

    # monkeypatch restores the previous global config afterwards, so later
    # tests that read global_args are not left with an unparsed one.
    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_defaults_to_off():
    assert parse_args().enable_query_budget_ceiling is False


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("false", False), ("True", True), ("False", False)],
)
def test_parses_as_a_bool(monkeypatch, value, expected):
    monkeypatch.setenv("ENABLE_QUERY_BUDGET_CEILING", value)

    assert parse_args().enable_query_budget_ceiling is expected


# ---------------------------------------------------------------------------
# Wiring: create_app -> create_query_routes, and /health
# ---------------------------------------------------------------------------


class _FakeLightRAG:
    """Minimal stand-in implementing the async surface /health touches."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _build_client(
    monkeypatch, query_route_calls: list[dict], *, drop_switch: bool = False
) -> TestClient:
    """Build a /health-capable app with backend I/O mocked out.

    ``create_query_routes`` is replaced by a recorder so the test sees exactly
    what ``create_app`` passed it. ``drop_switch`` removes the attribute from
    the parsed args, the shape of a Namespace built before the switch existed.
    """
    from lightrag.api.config import initialize_config

    args = parse_args()
    if drop_switch:
        del args.enable_query_budget_ceiling
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server
    import lightrag.api.utils_api as utils_api

    def _record_query_routes(*_args, **kwargs):
        query_route_calls.append(kwargs)
        return APIRouter()

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "create_query_routes", _record_query_routes)
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )
    # Fully open mode, so /health returns the configuration block.
    monkeypatch.setattr(utils_api, "auth_configured", False)

    return TestClient(lightrag_server.create_app(args))


@pytest.mark.parametrize("value,expected", [(None, False), ("true", True)])
def test_create_app_passes_the_switch_to_the_query_routes_and_health(
    monkeypatch, value, expected
):
    if value is not None:
        monkeypatch.setenv("ENABLE_QUERY_BUDGET_CEILING", value)
    query_route_calls: list[dict] = []

    client = _build_client(monkeypatch, query_route_calls)

    (kwargs,) = query_route_calls
    assert kwargs["enable_query_budget_ceiling"] is expected

    response = client.get("/health")
    assert response.status_code == 200
    configuration = response.json()["configuration"]
    assert configuration["enable_query_budget_ceiling"] is expected


def test_create_app_treats_args_without_the_switch_as_off(monkeypatch):
    """Callers that build args themselves need not know about the switch."""
    query_route_calls: list[dict] = []

    client = _build_client(monkeypatch, query_route_calls, drop_switch=True)

    (kwargs,) = query_route_calls
    assert kwargs["enable_query_budget_ceiling"] is False

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["configuration"]["enable_query_budget_ceiling"] is False
