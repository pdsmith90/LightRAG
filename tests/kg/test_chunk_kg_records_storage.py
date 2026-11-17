"""Extraction contract: where each chunk's accepted records live
(``kg_records``, lightrag/utils.py ``write_chunk_kg_records`` and the helpers
next to it).

- JSON (verified here; Redis and OpenSearch replace the document the same way,
  read only): the field is in the chunk document; a row read and written back
  keeps it, the pipeline's upsert of a re-created chunk drops it.
- MongoDB: the chunk upsert is a ``$set``, so it unsets the field when the row
  written does not carry it.
- PostgreSQL: a column only the three dedicated statements name; while the
  start-up probe has not found it, nothing names it.

Names are invented.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.utils import (
    CHUNK_KG_RECORDS_FIELD,
    clear_chunk_kg_records,
    read_chunk_kg_records,
    update_chunk_cache_list,
    write_chunk_kg_records,
)

pytestmark = pytest.mark.offline

RECORDS = {"format": 1, "entities": [], "relations": []}


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


@pytest.fixture
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def test_json_keeps_the_records_in_the_chunk_document(tmp_path, _shared_data):
    chunks = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await chunks.initialize()
    try:
        await chunks.upsert({"c1": _row()})
        await write_chunk_kg_records(chunks, "c1", RECORDS)
        assert (await chunks.get_by_id("c1"))[CHUNK_KG_RECORDS_FIELD] == RECORDS

        # a read-modify-write of the row (the cache references) keeps them
        assert await update_chunk_cache_list("c1", chunks, ["k1"])
        row = await chunks.get_by_id("c1")
        assert row["llm_cache_list"] == ["k1"]
        rows = {"c1": row}
        assert await read_chunk_kg_records(chunks, ["c1", "c2"], rows) == {
            "c1": RECORDS
        }

        # the pipeline re-creating the chunk for a new extraction drops them
        await chunks.upsert({"c1": _row()})
        assert CHUNK_KG_RECORDS_FIELD not in await chunks.get_by_id("c1")

        # a missing row is an error, never a silent skip
        with pytest.raises(RuntimeError, match="not found"):
            await write_chunk_kg_records(chunks, "c2", RECORDS)
        # no clear statement exists for a document store
        await clear_chunk_kg_records(chunks, ["c1"])
    finally:
        await chunks.finalize()


def _mongo_kv(namespace):
    from lightrag.kg.mongo_impl import MongoKVStorage

    storage = MongoKVStorage.__new__(MongoKVStorage)
    storage.namespace = namespace
    storage.workspace = "ws"
    storage._data = MagicMock()
    storage._max_upsert_payload_bytes = 16 * 1024 * 1024
    storage._max_upsert_records_per_batch = 128
    return storage


@pytest.mark.parametrize("namespace", ["text_chunks", "full_docs"])
async def test_mongo_unsets_the_records_a_chunk_row_does_not_carry(namespace):
    pytest.importorskip("pymongo")
    storage = _mongo_kv(namespace)
    calls = []

    async def fake_bulk(collection, ops, **kwargs):
        calls.extend(op for op, _bytes, _logid in ops)

    with patch(
        "lightrag.kg.mongo_impl._run_batched_bulk_write",
        new=AsyncMock(side_effect=fake_bulk),
    ):
        await storage.upsert(
            {"c1": _row(), "c2": _row(**{CHUNK_KG_RECORDS_FIELD: RECORDS})}
        )
    fresh, carried = (op._doc for op in calls)
    if namespace == "text_chunks":
        assert fresh["$unset"] == {CHUNK_KG_RECORDS_FIELD: ""}
    else:
        assert "$unset" not in fresh
    assert "$unset" not in carried
    assert carried["$set"][CHUNK_KG_RECORDS_FIELD] == RECORDS


def _pg_chunks(column: bool):
    from lightrag.kg.postgres_impl import PGKVStorage

    db = MagicMock()
    db.chunk_kg_records_column = column
    db.query = AsyncMock(return_value=[{"id": "c1"}])
    db.execute = AsyncMock()
    db.ensure_chunk_kg_records_column = AsyncMock()
    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = NameSpace.KV_STORE_TEXT_CHUNKS
    storage.workspace = "ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = db
    storage.__post_init__()
    return storage


async def test_postgres_names_the_column_only_when_it_exists():
    storage = _pg_chunks(column=False)
    assert await read_chunk_kg_records(storage, ["c1"], {"c1": _row()}) == {}
    await clear_chunk_kg_records(storage, ["c1"])
    storage.db.query.assert_not_awaited()
    storage.db.execute.assert_not_awaited()

    storage = _pg_chunks(column=True)
    storage.db.query = AsyncMock(
        return_value=[{"id": "c1", "kg_records": '{"format": 1}'}]
    )
    assert await read_chunk_kg_records(storage, ["c1", "c2"], {}) == {
        "c1": {"format": 1}
    }
    sql, params = storage.db.query.await_args.args
    assert "kg_records IS NOT NULL" in sql and params == ["ws", ["c1", "c2"]]
    await clear_chunk_kg_records(storage, ["c1"])
    sql, params = storage.db.execute.await_args.args
    assert sql.split()[:5] == [
        "UPDATE",
        "LIGHTRAG_DOC_CHUNKS",
        "SET",
        "kg_records",
        "=",
    ]
    assert params == {"workspace": "ws", "ids": ["c1"]}


async def test_postgres_write_adds_the_column_first_and_fails_on_a_missing_row():
    storage = _pg_chunks(column=False)
    await write_chunk_kg_records(storage, "c1", RECORDS)
    storage.db.ensure_chunk_kg_records_column.assert_awaited_once()
    sql, params = storage.db.query.await_args.args
    assert "SET kg_records = $3::jsonb" in sql and params[:2] == ["ws", "c1"]

    storage.db.query = AsyncMock(return_value=[])
    with pytest.raises(RuntimeError, match="not found"):
        await write_chunk_kg_records(storage, "c9", RECORDS)


def test_postgres_general_chunk_statements_leave_the_column_alone():
    from lightrag.kg.postgres_impl import SQL_TEMPLATES

    general = [
        sql
        for name, sql in SQL_TEMPLATES.items()
        if "LIGHTRAG_DOC_CHUNKS" in sql and "kg_records" not in name
    ]
    assert general and not any("kg_records" in sql for sql in general)
