"""Real-PostgreSQL checks of the text-chunk column that holds each chunk's
accepted extraction records (``kg_records``, lightrag/kg_contract.py).

- On an existing table without the column, nothing names it until the
  extraction contract or a hook needs it; then start-up adds it, and the
  records round-trip; a later extraction with neither clears them.
- A session holding a lock on the table makes the bounded add fail, and
  start-up is refused when the feature is on (and goes ahead when it is off).
- The general chunk statements (the ones the base branch also issues) work on
  a table that has the column and rows that use it.

Same markers and guard as ``tests/kg/pgtable_impl/test_pgtable_smoke.py``:
skipped unless pytest runs with ``--run-integration`` and ``POSTGRES_PASSWORD``
is set. Run it against a throwaway database: it drops and adds the column.
Names are invented.
"""

import asyncio
import os
import uuid

import numpy as np
import pytest

pytestmark = [pytest.mark.pg_smoke, pytest.mark.integration]

if not os.environ.get("POSTGRES_PASSWORD", ""):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping pg_smoke tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module", autouse=True)
def init_shared_storage():
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)


RECORDS = {
    "format": 1,
    "entities": [{"entity_name": "Zeta System", "description": "Z."}],
    "relations": [
        {
            "src_id": "Zeta System",
            "tgt_id": "Alpha Unit",
            "keywords": "part_of",
            "reverse_keywords": "has_part",
            "description": "Z in A.",
        }
    ],
}


def _row(**extra):
    return {
        "tokens": 5,
        "chunk_order_index": 0,
        "full_doc_id": "doc-1",
        "content": "Zeta System holds the Alpha Unit.",
        "file_path": "one.txt",
        "llm_cache_list": [],
        **extra,
    }


async def _connect():
    import asyncpg

    return await asyncpg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ.get("POSTGRES_DATABASE", "postgres"),
    )


async def _has_column(conn) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'lightrag_doc_chunks' AND column_name = 'kg_records'"
        )
    )


@pytest.fixture
async def chunks():
    """A text-chunk storage on a table WITHOUT the column, as on a database
    made by the base branch."""
    from lightrag.kg.postgres_impl import PGKVStorage
    from lightrag.namespace import NameSpace

    storage = PGKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace=f"kgrecords_{uuid.uuid4().hex[:8]}",
        # the LightRAG instances below use the same process-wide pool
        global_config={
            "embedding_batch_num": 10,
            "vector_storage": "NanoVectorDBStorage",
        },
        embedding_func=None,
    )
    await storage.initialize()
    conn = await _connect()
    try:
        await conn.execute(
            "ALTER TABLE LIGHTRAG_DOC_CHUNKS DROP COLUMN IF EXISTS kg_records"
        )
    finally:
        await conn.close()
    # the start-up probe, run again on the old table
    await storage.db._migrate_text_chunks_add_heading_sidecar()
    try:
        yield storage
    finally:
        await storage.drop()
        await storage.finalize()


async def test_the_column_is_added_when_needed_and_round_trips(chunks):
    from lightrag.utils import (
        clear_chunk_kg_records,
        ensure_chunk_kg_records_field,
        read_chunk_kg_records,
        write_chunk_kg_records,
    )

    assert chunks.db.chunk_kg_records_column is False
    await chunks.upsert({"c1": _row(), "c2": _row()})
    # with the column absent nothing names it: reads, clears and general
    # statements all work
    assert await read_chunk_kg_records(chunks, ["c1"], {}) == {}
    await clear_chunk_kg_records(chunks, ["c1"])
    assert (await chunks.get_by_id("c1"))["content"].startswith("Zeta")

    await ensure_chunk_kg_records_field(chunks)
    conn = await _connect()
    try:
        assert await _has_column(conn)
    finally:
        await conn.close()
    await write_chunk_kg_records(chunks, "c1", RECORDS)
    assert await read_chunk_kg_records(chunks, ["c1", "c2"], {}) == {"c1": RECORDS}
    # the general read and upsert leave the column alone
    row = await chunks.get_by_id("c1")
    assert "kg_records" not in row
    await chunks.upsert({"c1": {**row, "llm_cache_list": ["k1"]}})
    assert await read_chunk_kg_records(chunks, ["c1"], {}) == {"c1": RECORDS}
    with pytest.raises(RuntimeError, match="not found"):
        await write_chunk_kg_records(chunks, "c9", RECORDS)
    await clear_chunk_kg_records(chunks, ["c1", "c2"])
    assert await read_chunk_kg_records(chunks, ["c1", "c2"], {}) == {}


async def test_general_chunk_statements_work_on_rows_with_records(chunks):
    """The base branch's chunk statements (unchanged here) on the migrated
    table, with a row that holds records: read, re-write and delete."""
    from lightrag.utils import ensure_chunk_kg_records_field, write_chunk_kg_records

    await ensure_chunk_kg_records_field(chunks)
    await chunks.upsert({"c1": _row()})
    await write_chunk_kg_records(chunks, "c1", RECORDS)
    [row] = await chunks.get_by_ids(["c1"])
    assert row["content"].startswith("Zeta") and "kg_records" not in row
    await chunks.upsert({"c1": {**row, "content": "Changed."}})
    assert (await chunks.get_by_id("c1"))["content"] == "Changed."
    assert await chunks.filter_keys({"c1", "c2"}) == {"c2"}
    await chunks.delete(["c1"])
    assert await chunks.get_by_id("c1") is None


async def test_extraction_without_contract_clears_records_of_its_chunks(chunks):
    """A chunk row PostgreSQL kept across a re-run (its upsert does not touch
    the column) loses its records when it is extracted with neither contract
    nor hook."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import resolve_entity_extraction_prompt_profile
    from lightrag.utils import (
        ensure_chunk_kg_records_field,
        read_chunk_kg_records,
        write_chunk_kg_records,
    )

    await ensure_chunk_kg_records_field(chunks)
    await chunks.upsert({"c1": _row()})
    await write_chunk_kg_records(chunks, "c1", RECORDS)

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        return '{"entities": [], "relationships": []}'

    config = {
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "llm_model_max_async": 1,
        "entity_extraction_use_json": True,
        "_entity_extraction_prompt_profile": resolve_entity_extraction_prompt_profile(
            {}, True
        ),
    }
    await extract_entities({"c1": _row()}, config, text_chunks_storage=chunks)
    assert await read_chunk_kg_records(chunks, ["c1"], {}) == {}


async def _embed(texts):
    return np.ones((len(texts), 8))


def _rag(tmp_path, validator):
    from lightrag import LightRAG
    from lightrag.kg_contract import kg_extraction_validator
    from lightrag.utils import EmbeddingFunc

    async def llm(*_args, **_kwargs):
        return "A summary."

    return LightRAG(
        working_dir=str(tmp_path / uuid.uuid4().hex),
        workspace=f"kgrecords_{uuid.uuid4().hex[:8]}",
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        kv_storage="PGKVStorage",
        **({"kg_extraction_validator": kg_extraction_validator} if validator else {}),
    )


@pytest.mark.parametrize("feature_on", [True, False], ids=["hook", "neither"])
async def test_a_held_lock_refuses_start_up_only_when_the_column_is_needed(
    chunks, tmp_path, monkeypatch, feature_on
):
    """Another session holds a lock on the chunk table (an open transaction
    that read it). With the hook on, every bounded attempt to add the column
    times out and start-up is refused with a clear message; with neither, the
    column is not needed and start-up goes ahead without naming it."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    monkeypatch.setattr(PostgreSQLDB, "KG_RECORDS_COLUMN_LOCK_TIMEOUT", "300ms")
    monkeypatch.setattr(PostgreSQLDB, "KG_RECORDS_COLUMN_RETRY_DELAY_S", 0.05)
    holder = await _connect()
    tx = holder.transaction()
    await tx.start()
    await holder.fetch("SELECT id FROM LIGHTRAG_DOC_CHUNKS LIMIT 1")
    rag = _rag(tmp_path, feature_on)
    try:
        if feature_on:
            with pytest.raises(RuntimeError, match="Cannot add the column kg_records"):
                await asyncio.wait_for(rag.initialize_storages(), 60)
        else:
            await asyncio.wait_for(rag.initialize_storages(), 60)
            assert rag.text_chunks.db.chunk_kg_records_column is False
    finally:
        await tx.rollback()
        await holder.close()
        await rag.finalize_storages()
    conn = await _connect()
    try:
        assert await _has_column(conn) is False
    finally:
        await conn.close()
