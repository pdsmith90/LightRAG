"""PGGraphStorage.remove_edges / remove_nodes against a live PostgreSQL + Apache AGE server.

The offline tests in ``test_postgres_graph_batch.py`` pin the statement shapes
with a fake connection; this module proves the plain-SQL DELETEs remove exactly
the requested edges, or vertices with their edges, on a real AGE graph: both
stored directions, every edge label, several chunks, and the rest left alone.

Opt in with ``--run-integration`` and the ``POSTGRES_*`` variables pointing at a
server that has AGE preloaded, e.g. the ``apache/age`` image. The module skips
itself when ``POSTGRES_PASSWORD`` is unset or the server has no AGE extension.
"""

import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

if not os.environ.get("POSTGRES_PASSWORD"):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping PostgreSQL AGE tests",
        allow_module_level=True,
    )


async def _age_installed(db) -> bool:
    row = await db.query("SELECT 1 AS ok FROM pg_extension WHERE extname = 'age'")
    return bool(row)


@pytest.fixture
async def graph(monkeypatch):
    """A PGGraphStorage on a fresh AGE graph, dropped after the test.

    The workspace is mixed case on purpose: AGE keeps the graph name's case, so
    any plain-SQL statement must quote it.
    """
    from lightrag.kg.postgres_impl import PGGraphStorage
    from lightrag.kg.shared_storage import initialize_share_data

    # POSTGRES_WORKSPACE (shell, .env or config.ini) outranks the constructor's
    # workspace, and teardown drops the graph. Pin it empty rather than unset:
    # load_dotenv(override=False) would refill an unset variable from .env.
    monkeypatch.setenv("POSTGRES_WORKSPACE", "")
    token = uuid.uuid4().hex[:8]
    initialize_share_data()
    storage = PGGraphStorage(
        # A test-only namespace, so the graph can never be a deployment's.
        namespace="test_remove_edges_age",
        workspace=f"RemoveEdges{token}",
        global_config={},
        embedding_func=None,
    )
    try:
        await storage.initialize()
    except Exception:
        if storage.db is not None and not await _age_installed(storage.db):
            await storage.finalize()
            pytest.skip("Apache AGE is not installed on the configured server")
        raise
    # The pool is a process-wide singleton that keeps the workspace it was
    # created with. If another test still holds one with a workspace set, the
    # graph resolved here is not this test's: skip before writing to or
    # dropping it.
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


def _node(entity_id: str) -> dict:
    return {"entity_id": entity_id, "entity_type": "TEST", "description": entity_id}


def _edge() -> dict:
    return {"weight": "1.0", "description": "e", "keywords": "k", "source_id": "c"}


async def _edges(storage) -> list[tuple[str, str, str]]:
    """Every stored edge as (start entity_id, end entity_id, label table)."""
    g = f'"{storage.graph_name}"'
    entity_id = (
        "ag_catalog.agtype_access_operator("
        "VARIADIC ARRAY[{v}.properties, '\"entity_id\"'::ag_catalog.agtype])::text"
    )
    rows = await storage.db.query(
        f"SELECT {entity_id.format(v='a')} AS src, {entity_id.format(v='b')} AS tgt, "
        "c.relname::text AS label "
        f"FROM {g}._ag_label_edge e "
        f"JOIN {g}._ag_label_vertex a ON a.id = e.start_id "
        f"JOIN {g}._ag_label_vertex b ON b.id = e.end_id "
        "JOIN pg_class c ON c.oid = e.tableoid",
        multirows=True,
        with_age=True,  # graphid's = operator lives in ag_catalog
        graph_name=storage.graph_name,
    )
    return sorted((r["src"], r["tgt"], r["label"]) for r in rows or [])


async def _vertex_count(storage) -> int:
    row = await storage.db.query(
        f'SELECT count(*) AS n FROM "{storage.graph_name}"._ag_label_vertex'
    )
    return int(row["n"])


@pytest.mark.asyncio
async def test_remove_edges_deletes_exactly_the_requested_pairs(graph):
    names = ["A", "B", "C", "D", "E", 'say "hi"', "back\\slash", "北京"]
    for name in names:
        await graph.upsert_node(name, _node(name))
    stored = [
        ("A", "B"),
        ("C", "A"),
        ("A", "D"),
        ("B", "C"),
        ("D", "B"),
        ("E", "A"),
        ('say "hi"', "back\\slash"),
        ("back\\slash", "E"),
        ("北京", "A"),
    ]
    await graph.upsert_edges_batch([(src, tgt, _edge()) for src, tgt in stored])
    # A second edge label between A and B: the pattern (a)-[r]-(b) matches any
    # label, so removal must reach it too.
    await graph._query(
        f"SELECT * FROM cypher('{graph.graph_name}', $$"
        'MATCH (b:base {entity_id: "B"}), (a:base {entity_id: "A"}) '
        "CREATE (b)-[:OTHER]->(a)"
        "$$) AS (r agtype)",
        readonly=False,
    )
    vertices_before = await _vertex_count(graph)
    assert len(await _edges(graph)) == len(stored) + 1

    # Two pairs per chunk, so the request spans three transactions.
    graph._max_delete_records_per_batch = 2
    await graph.remove_edges(
        [
            ("A", "B"),  # stored orientation, plus the OTHER edge B->A
            ("A", "C"),  # stored as C->A
            ("back\\slash", 'say "hi"'),  # stored the other way round
            ("A", "北京"),  # stored the other way round
            ("D", "E"),  # no such edge: a no-op
        ]
    )

    assert await _edges(graph) == sorted(
        [
            ("A", "D", "DIRECTED"),
            ("B", "C", "DIRECTED"),
            ("D", "B", "DIRECTED"),
            ("E", "A", "DIRECTED"),
            ("back\\slash", "E", "DIRECTED"),
        ]
    )
    assert await _vertex_count(graph) == vertices_before == len(names)


@pytest.mark.asyncio
async def test_remove_nodes_deletes_the_vertices_and_every_edge_touching_them(graph):
    names = ["A", "B", "C", "D", "E", 'say "hi"', "back\\slash", "北京"]
    for name in names:
        await graph.upsert_node(name, _node(name))
    stored = [
        ("A", "B"),
        ("C", "A"),
        ("A", "D"),
        ("B", "C"),
        ("D", "B"),
        ("E", "A"),
        ('say "hi"', "back\\slash"),
        ("back\\slash", "E"),
        ("北京", "A"),
    ]
    await graph.upsert_edges_batch([(src, tgt, _edge()) for src, tgt in stored])
    # A second edge label pointing INTO a removed vertex: DETACH DELETE removed
    # edges of every label at either end, so the SQL must reach it too.
    await graph._query(
        f"SELECT * FROM cypher('{graph.graph_name}', $$"
        'MATCH (b:base {entity_id: "B"}), (a:base {entity_id: "A"}) '
        "CREATE (b)-[:OTHER]->(a)"
        "$$) AS (r agtype)",
        readonly=False,
    )
    assert len(await _edges(graph)) == len(stored) + 1

    # Two ids per chunk, so the request spans two chunks of one transaction.
    graph._max_delete_records_per_batch = 2
    await graph.remove_nodes(["A", 'say "hi"', "北京", "missing"])

    assert await _edges(graph) == sorted(
        [
            ("B", "C", "DIRECTED"),
            ("D", "B", "DIRECTED"),
            ("back\\slash", "E", "DIRECTED"),
        ]
    )
    assert await _vertex_count(graph) == len(names) - 3
    # Cypher reads see the SQL removal. (get_node_edges is the Cypher read
    # here; has_node and the batch reads interpolate the graph name unquoted
    # and cannot reach this mixed-case graph.)
    assert sorted(await graph.get_node_edges("B")) == [("B", "C"), ("B", "D")]
    assert await graph.get_node_edges("E") == [("E", "back\\slash")]

    await graph.delete_node("back\\slash")

    assert await _edges(graph) == [("B", "C", "DIRECTED"), ("D", "B", "DIRECTED")]
    assert await _vertex_count(graph) == len(names) - 4
    assert await graph.get_node_edges("E") == []
