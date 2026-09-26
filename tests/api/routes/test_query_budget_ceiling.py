"""``ENABLE_QUERY_BUDGET_CEILING``: server retrieval defaults as per-request maxima.

Off (the default), ``top_k``, ``chunk_top_k`` and the three ``max_*_tokens``
budgets reach ``QueryParam`` exactly as the client sent them; the server values
are defaults only. On, a value above the server default is lowered to it, a
value at or below it is honoured, and an omitted value still gets the default.

The clamp lives in ``QueryRequest.to_query_params``, the one conversion that
``/query``, ``/query/stream`` and ``/query/data`` share, so these tests pin both
the conversion and that every route hands it the flag it was built with.
"""

from __future__ import annotations

import importlib
import logging
import sys
import typing

import pytest

# lightrag.api.config parses argv at import time, so the import has to be guarded
# the way the rest of the API tests guard it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

QueryRequest = _query_routes.QueryRequest
create_query_routes = _query_routes.create_query_routes

from lightrag.base import QueryParam  # noqa: E402

pytestmark = pytest.mark.offline

# Listed here rather than imported, so the test pins the documented coverage
# instead of trusting the module's own list.
BUDGET_FIELDS = (
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
)
QUERY_ROUTES = ("/query", "/query/stream", "/query/data")
QUERY_TEXT = "which retrieval settings apply here"


def _defaults() -> dict[str, int]:
    """The server-configured values: what an omitted field resolves to."""
    defaults = QueryParam()
    return {name: getattr(defaults, name) for name in BUDGET_FIELDS}


def _above() -> dict[str, int]:
    return {name: value + 1 for name, value in _defaults().items()}


def _below() -> dict[str, int]:
    return {name: max(1, value // 2) for name, value in _defaults().items()}


def _budget(param: QueryParam) -> dict[str, int]:
    return {name: getattr(param, name) for name in BUDGET_FIELDS}


def _convert(apply_budget_ceiling: bool | None = None, **budget) -> QueryParam:
    request = QueryRequest(query=QUERY_TEXT, **budget)
    if apply_budget_ceiling is None:
        return request.to_query_params(False)
    return request.to_query_params(False, apply_budget_ceiling=apply_budget_ceiling)


# ---------------------------------------------------------------------------
# Flag off: today's behaviour, the server values are defaults only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apply_budget_ceiling", [None, False])
def test_off_passes_values_above_the_server_default_through(apply_budget_ceiling):
    param = _convert(apply_budget_ceiling, **_above())
    assert _budget(param) == _above()


def test_off_omitted_values_get_the_server_default():
    assert _budget(_convert()) == _defaults()


# ---------------------------------------------------------------------------
# Flag on: a ceiling, not an override.
# ---------------------------------------------------------------------------


def test_on_lowers_values_above_the_server_default():
    param = _convert(True, **_above())
    assert _budget(param) == _defaults()


@pytest.mark.parametrize("budget", [_below, _defaults], ids=["below", "equal"])
def test_on_honours_values_at_or_below_the_server_default(budget):
    param = _convert(True, **budget())
    assert _budget(param) == budget()


def test_on_omitted_values_get_the_server_default():
    assert _budget(_convert(True)) == _defaults()


@pytest.mark.parametrize("field", BUDGET_FIELDS)
def test_on_clamps_each_field_independently(field):
    """Only the knob that is over its own default moves; the rest are kept."""
    body = _below()
    body[field] = _defaults()[field] + 1
    expected = _below()
    expected[field] = _defaults()[field]

    assert _budget(_convert(True, **body)) == expected


def test_on_leaves_the_rest_of_the_request_alone():
    param = QueryRequest(
        query=QUERY_TEXT,
        mode="local",
        response_type="Bullet Points",
        enable_rerank=False,
        **_above(),
    ).to_query_params(True, apply_budget_ceiling=True)

    assert param.mode == "local"
    assert param.response_type == "Bullet Points"
    assert param.enable_rerank is False
    assert param.stream is True


# ---------------------------------------------------------------------------
# Logging: one line per clamped request, and no query text in it.
# ---------------------------------------------------------------------------


@pytest.fixture
def ceiling_lines(lightrag_log_records):
    """Returns a reader for the budget-ceiling lines logged during the test.

    Records come from ``lightrag_log_records``, so each one counts once on every
    pytest version. The line is INFO and ``create_app`` sets the logger's level
    from LOG_LEVEL for the whole process, so the level is pinned here.
    """
    lightrag_logger = logging.getLogger("lightrag")
    previous_level = lightrag_logger.level
    lightrag_logger.setLevel(logging.INFO)
    try:
        yield lambda: [
            record.getMessage()
            for record in lightrag_log_records
            if "budget ceiling" in record.getMessage()
        ]
    finally:
        lightrag_logger.setLevel(previous_level)


def test_on_logs_one_line_per_clamped_request(ceiling_lines):
    _convert(True, **_above())

    (line,) = ceiling_lines()
    for name in BUDGET_FIELDS:
        assert f"{name} {_above()[name]}->{_defaults()[name]}" in line
    assert QUERY_TEXT not in line


@pytest.mark.parametrize(
    "apply_budget_ceiling,budget",
    [(True, _below), (True, _defaults), (False, _above)],
    ids=["on-below", "on-equal", "off-above"],
)
def test_nothing_is_logged_when_nothing_is_lowered(
    ceiling_lines, apply_budget_ceiling, budget
):
    _convert(apply_budget_ceiling, **budget())

    assert ceiling_lines() == []


# ---------------------------------------------------------------------------
# Every query route funnels through the conversion with the router's flag.
# ---------------------------------------------------------------------------


class _RecordingRag:
    """Captures the QueryParam each route hands to the RAG instance."""

    def __init__(self):
        self.params: list[QueryParam] = []

    async def aquery_llm(self, _query, param, **_kwargs):
        self.params.append(param)
        return {"llm_response": {"content": "answer"}, "data": {"references": []}}

    async def aquery_data(self, _query, param):
        self.params.append(param)
        return {"status": "success", "message": "", "data": {}, "metadata": {}}


def _endpoint(router, path):
    return next(route.endpoint for route in router.routes if route.path == path)


def test_the_listed_routes_are_every_route_that_takes_a_query_request():
    """A new route accepting QueryRequest must be added to QUERY_ROUTES."""
    router = create_query_routes(_RecordingRag())
    taking_request = {
        route.path
        for route in router.routes
        if QueryRequest in typing.get_type_hints(route.endpoint).values()
    }
    assert taking_request == set(QUERY_ROUTES)


@pytest.mark.parametrize("path", QUERY_ROUTES)
@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
async def test_every_query_route_applies_the_flag_it_was_built_with(path, enabled):
    rag = _RecordingRag()
    router = create_query_routes(rag, enable_query_budget_ceiling=enabled)

    await _endpoint(router, path)(QueryRequest(query=QUERY_TEXT, **_above()))

    (param,) = rag.params
    assert _budget(param) == (_defaults() if enabled else _above())


@pytest.mark.parametrize("path", QUERY_ROUTES)
async def test_routes_default_to_the_ceiling_off(path):
    """Callers that build the router without the flag keep today's behaviour."""
    rag = _RecordingRag()
    router = create_query_routes(rag)

    await _endpoint(router, path)(QueryRequest(query=QUERY_TEXT, **_above()))

    (param,) = rag.params
    assert _budget(param) == _above()
