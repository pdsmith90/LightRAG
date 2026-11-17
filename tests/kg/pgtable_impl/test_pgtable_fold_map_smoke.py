"""Real-PostgreSQL check of ENTITY_NAME_FOLD's map build on PGTableGraphStorage.

The offline fold tests mock ``_fetch``; this runs the table query itself: every
node of the storage's workspace and namespace with its entity type and its
degree (each canonical edge row counted once for each endpoint), and nothing
from another workspace.

Same markers and guard as ``test_pgtable_smoke.py``: skipped unless pytest runs
with ``--run-integration`` and ``POSTGRES_PASSWORD`` is set.
"""

import os
import uuid

import pytest

import lightrag.entity_name_fold as fold_mod
import lightrag.entity_name_guard as guard
from lightrag.entity_name_fold import canonical_name, load_fold_lists

pytestmark = [pytest.mark.pg_smoke, pytest.mark.integration]

if not os.environ.get("POSTGRES_PASSWORD", ""):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping pg_smoke tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="session", autouse=True)
def init_shared_storage():
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (frozenset(), frozenset()))
    monkeypatch.delenv("ENTITY_NAME_FOLD_LISTS", raising=False)
    monkeypatch.delenv("ENTITY_NAME_ALIASES", raising=False)
    fold_mod.disarm()
    load_fold_lists(None)
    yield
    fold_mod.disarm()
    load_fold_lists(None)


async def _storage(workspace: str):
    from lightrag.kg.pgtable_impl import PGTableGraphStorage

    storage = PGTableGraphStorage(
        namespace="graph",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


def _node(entity_id: str, entity_type: str) -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "description": f"fold smoke node {entity_id}",
        "source_id": "fold-smoke-chunk",
        "file_path": "fold-smoke.txt",
    }


_EDGE = {
    "weight": 1.0,
    "description": "fold smoke edge",
    "keywords": "uses",
    "source_id": "fold-smoke-chunk",
    "file_path": "fold-smoke.txt",
}


@pytest.fixture
async def stores():
    """Two storages in separate workspaces; the second must not leak into the
    first one's map."""
    suffix = uuid.uuid4().hex[:8]
    first = await _storage(f"foldsmoke_{suffix}")
    other = await _storage(f"foldsmoke_other_{suffix}")
    try:
        yield first, other
    finally:
        for storage in (first, other):
            await storage.drop()
            await storage.finalize()


@pytest.mark.asyncio
async def test_the_map_reads_names_types_and_degrees_of_one_workspace(stores):
    first, other = stores
    await first.upsert_node("Imaginary Filter", _node("Imaginary Filter", "method"))
    await first.upsert_node("imaginary filter", _node("imaginary filter", "method"))
    await first.upsert_node("Quux Basin", _node("Quux Basin", "location"))
    await first.upsert_node("Zorp Probe", _node("Zorp Probe", "mission"))
    # canonical order is the storage's business; pass the pairs either way
    await first.upsert_edge("Quux Basin", "Imaginary Filter", _EDGE)
    await first.upsert_edge("Imaginary Filter", "Zorp Probe", _EDGE)
    await other.upsert_node("Glorbon Model", _node("Glorbon Model", "method"))
    await other.upsert_edge("Glorbon Model", "Quux Basin", _EDGE)

    rows = await fold_mod._fold_rows_reader(first)()
    assert sorted((r["name"], r["type"], int(r["degree"])) for r in rows) == [
        ("Imaginary Filter", "method", 2),
        ("Quux Basin", "location", 1),
        ("Zorp Probe", "mission", 1),
        ("imaginary filter", "method", 0),
    ]

    assert await fold_mod.build_fold_map(first) == 3
    assert fold_mod.is_armed()
    # the best-connected spelling is the canonical one
    assert canonical_name("imaginary-filter") == "Imaginary Filter"
    # a name only the other workspace holds is not in the map: it registers as new
    assert canonical_name("glorbon model", register=False) == "glorbon model"
