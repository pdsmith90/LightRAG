"""Real-PostgreSQL check of the extraction contract's inverse names on
PGTableGraphStorage (lightrag/kg_contract.py).

The table storage keeps each pair once, ordered by Python ``min``/``max`` (code
points), the order the merge sorts by. With ``relationship_inverses`` a stored
row (src_id, tgt_id, keywords) reads "src_id keywords tgt_id", including for a
non-ASCII name that a linguistic collation would order the other way round.

Same markers and guard as ``test_pgtable_smoke.py``: skipped unless pytest runs
with ``--run-integration`` and ``POSTGRES_PASSWORD`` is set. Names are invented.
"""

import os
import uuid

import pytest

from tests.extraction.test_kg_contract_direction import (
    CONTRACT,
    CONTRACT_WITH_INVERSES,
    _chunk,
    _merge,
)

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


@pytest.fixture
async def store():
    from lightrag.kg.pgtable_impl import PGTableGraphStorage

    storage = PGTableGraphStorage(
        namespace="graph",
        workspace=f"directionsmoke_{uuid.uuid4().hex[:8]}",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
    )
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.drop()
        await storage.finalize()


async def _rows(storage):
    rows = await storage._fetch(
        """SELECT src_id, tgt_id, properties->>'keywords' AS keywords
           FROM lightrag_graph_edges WHERE workspace = $1 AND namespace = $2""",
        storage.workspace,
        storage.namespace,
    )
    return sorted((r["src_id"], r["tgt_id"], r["keywords"]) for r in rows)


@pytest.mark.asyncio
async def test_stored_rows_read_in_their_own_order(store):
    await _merge(
        [
            _chunk(
                ("GRACE", "part_of", "Constellation"),
                ("Ångström Probe", "measures", "Zeta Field"),
                chunk="chunk-1",
            ),
            _chunk(("Constellation", "has_property", "GRACE"), chunk="chunk-2"),
        ],
        CONTRACT_WITH_INVERSES,
        store,
    )
    assert await _rows(store) == [
        ("Constellation", "GRACE", "has_part,has_property"),
        # 'Z' < 'Å' by code point: the probe is the target of the stored row
        ("Zeta Field", "Ångström Probe", "measured_by"),
    ]


@pytest.mark.asyncio
async def test_without_inverses_the_rows_are_as_before(store):
    await _merge(
        [_chunk(("Ångström Probe", "measures", "Zeta Field"))], CONTRACT, store
    )
    assert await _rows(store) == [("Zeta Field", "Ångström Probe", "measures")]
