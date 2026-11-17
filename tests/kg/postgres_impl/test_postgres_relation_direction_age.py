"""The extraction contract's inverse names on a live PostgreSQL + Apache AGE graph
(lightrag/kg_contract.py).

AGE keeps the direction an edge is written in, and its upsert replaces an edge
of the pair in either direction. With ``relationship_inverses`` the merge and the
entity rename write each pair in sorted order, so every DIRECTED edge reads
"start keywords end": the merge of "GRACE part_of Constellation" is
Constellation -has_part-> GRACE, and a rename that flips a pair rewrites both the
arrow and the keyword. Without the key the rename writes the pair in the order
``get_node_edges`` gives it, as before. Names are invented.

Opt in with ``--run-integration`` and the ``POSTGRES_*`` variables pointing at a
server that has AGE preloaded, e.g. the ``apache/age`` image. The module skips
itself when ``POSTGRES_PASSWORD`` is unset or the server has no AGE extension.
"""

import os
import uuid

import pytest

from lightrag import utils_graph
from lightrag.kg_contract import RELATIONSHIP_INVERSES_V31, inverse_table
from tests.extraction.test_kg_contract_direction import (
    CONTRACT_WITH_INVERSES,
    _chunk,
    _merge,
)
from tests.extraction.test_rename_tracking_migration_ordering import _VectorStorage

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

if not os.environ.get("POSTGRES_PASSWORD"):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping PostgreSQL AGE tests",
        allow_module_level=True,
    )

TABLE = inverse_table(RELATIONSHIP_INVERSES_V31)


@pytest.fixture
async def graph(monkeypatch):
    """A PGGraphStorage on a fresh AGE graph, dropped after the test."""
    from lightrag.kg.postgres_impl import PGGraphStorage
    from lightrag.kg.shared_storage import initialize_share_data

    monkeypatch.setenv("POSTGRES_WORKSPACE", "")
    token = uuid.uuid4().hex[:8]
    initialize_share_data()
    storage = PGGraphStorage(
        namespace="test_relation_direction_age",
        workspace=f"direction{token}",
        global_config={},
        embedding_func=None,
    )
    try:
        await storage.initialize()
    except Exception:
        if storage.db is not None and not await storage.db.query(
            "SELECT 1 AS ok FROM pg_extension WHERE extname = 'age'"
        ):
            await storage.finalize()
            pytest.skip("Apache AGE is not installed on the configured server")
        raise
    if token not in storage.graph_name:
        graph_name = storage.graph_name
        await storage.finalize()
        pytest.skip(f"shared PG pool resolved graph {graph_name!r}, not this test's")
    try:
        yield storage
    finally:
        await storage.db.query(
            "SELECT ag_catalog.drop_graph($1::name, true)", [storage.graph_name]
        )
        await storage.finalize()


async def _directed(graph):
    """(start, end, keywords) of every DIRECTED edge."""
    return sorted(
        (e["source"].strip('"'), e["target"].strip('"'), e.get("keywords"))
        for e in await graph.get_all_edges()
    )


async def _rename(graph, old, new, table):
    await utils_graph.aedit_entity(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        old,
        {"entity_name": new},
        relation_inverse_table=table,
    )


@pytest.mark.asyncio
async def test_the_merge_writes_the_arrow_the_keyword_reads(graph):
    await _merge(
        [
            _chunk(("GRACE", "part_of", "Constellation"), chunk="chunk-1"),
            _chunk(("Constellation", "has_property", "GRACE"), chunk="chunk-2"),
            _chunk(("Accelerometer", "part_of", "GRACE"), chunk="chunk-3"),
        ],
        CONTRACT_WITH_INVERSES,
        graph,
    )
    assert await _directed(graph) == [
        ("Accelerometer", "GRACE", "part_of"),
        ("Constellation", "GRACE", "has_part,has_property"),
    ]


@pytest.mark.asyncio
async def test_a_rename_that_flips_the_pair_turns_arrow_and_keyword(graph):
    await _merge(
        [_chunk(("Accelerometer", "part_of", "GRACE"))], CONTRACT_WITH_INVERSES, graph
    )
    await _rename(graph, "Accelerometer", "Zond Accelerometer", TABLE)
    assert await _directed(graph) == [("GRACE", "Zond Accelerometer", "has_part")]


@pytest.mark.asyncio
async def test_without_inverses_the_rename_writes_as_before(graph):
    await _merge(
        [_chunk(("Accelerometer", "part_of", "GRACE"))], CONTRACT_WITH_INVERSES, graph
    )
    await _rename(graph, "Accelerometer", "Zond Accelerometer", None)
    # get_node_edges gives (renamed node, other): the arrow follows it
    assert await _directed(graph) == [("Zond Accelerometer", "GRACE", "part_of")]
