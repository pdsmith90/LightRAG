"""Real-PostgreSQL check of the text-chunk column that records the extraction
contract per cache key (``llm_cache_contract``, lightrag/kg_contract.py).

A row written without the mapping stores NULL and reads back with no such key,
exactly as before the column existed; a row with it reads back the dict, and
``update_chunk_cache_list`` round-trips it. Same markers and guard as
``tests/kg/pgtable_impl/test_pgtable_smoke.py``: skipped unless pytest runs with
``--run-integration`` and ``POSTGRES_PASSWORD`` is set. Names are invented.
"""

import os
import uuid

import pytest

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
async def chunks():
    from lightrag.kg.postgres_impl import PGKVStorage
    from lightrag.namespace import NameSpace

    storage = PGKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace=f"contractsmoke_{uuid.uuid4().hex[:8]}",
        global_config={"embedding_batch_num": 10},
        embedding_func=None,
    )
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.drop()
        await storage.finalize()


def _row(**extra):
    return {
        "tokens": 5,
        "chunk_order_index": 0,
        "full_doc_id": "doc-1",
        "content": "Zeta System holds the Alpha Unit.",
        "file_path": "one.txt",
        "llm_cache_list": ["k1"],
        **extra,
    }


async def test_the_column_round_trips(chunks):
    from lightrag.utils import update_chunk_cache_list

    await chunks.upsert({"c1": _row(), "c2": _row(llm_cache_contract={"k1": "kgc-a"})})
    plain, contracted = await chunks.get_by_ids(["c1", "c2"])
    assert "llm_cache_contract" not in plain
    assert contracted["llm_cache_contract"] == {"k1": "kgc-a"}
    assert "llm_cache_contract" not in await chunks.get_by_id("c1")

    assert await update_chunk_cache_list("c1", chunks, ["k2"], cache_contract="kgc-b")
    row = await chunks.get_by_id("c1")
    assert row["llm_cache_list"] == ["k1", "k2"]
    assert row["llm_cache_contract"] == {"k2": "kgc-b"}

    # a run without a contract takes its key back out
    assert await update_chunk_cache_list("c1", chunks, ["k2"])
    assert (await chunks.get_by_id("c1"))["llm_cache_contract"] == {}

    # a fresh row (re-ingest) has none again
    await chunks.upsert({"c2": _row()})
    assert "llm_cache_contract" not in await chunks.get_by_id("c2")
