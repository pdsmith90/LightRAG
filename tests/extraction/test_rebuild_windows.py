"""Fork (windowed rebuild): a rebuild over many chunks loads, rebuilds and releases them
window by window (``kg_rebuild_window_chunks``) instead of holding every referenced chunk
at once, and the outcome is the one a single window gives."""

import asyncio

import pytest

import lightrag.operate as operate
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import _split_rebuild_windows, rebuild_knowledge_from_chunks

pytestmark = pytest.mark.offline

_REAL_LOAD = operate._load_chunk_extract_rows


@pytest.fixture(autouse=True)
def _shared_data():
    initialize_share_data()


class _KV:
    def __init__(self, data):
        self.data = dict(data)

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)


class _VDB:
    def __init__(self):
        self.flushes = 0

    async def index_done_callback(self):
        self.flushes += 1


def _stores(chunks):
    """Every chunk but the one named ``gone`` has a cached extraction."""
    text_chunks = {c: {"content": c, "llm_cache_list": [f"k-{c}"]} for c in chunks}
    llm_cache = {
        f"k-{c}": {
            "cache_type": "extract",
            "chunk_id": c,
            "return": "r",
            "create_time": 1,
        }
        for c in chunks
        if c != "gone"
    }
    return _KV(text_chunks), _KV(llm_cache)


ENTITIES = {"E1": ["c1", "c2"], "E2": ["c2", "c3"], "E3": ["c4"], "E4": ["c5", "gone"]}
RELATIONSHIPS = {("A", "B"): ["c1", "c6"], ("B", "C"): ["c7"]}
CHUNKS = ["c1", "c2", "c3", "c4", "c5", "gone", "c6", "c7"]


async def _run(monkeypatch, window):
    loads: list[int] = []
    rebuilt: dict = {}

    async def spy_load(llm_response_cache, chunk_ids, text_chunks_storage):
        loads.append(len(chunk_ids))
        return await _REAL_LOAD(
            llm_response_cache, chunk_ids, text_chunks_storage=text_chunks_storage
        )

    async def fake_entity(**kwargs):
        # The window the item is in must have loaded the item's chunks.
        loaded = set(kwargs["chunk_data_by_id"])
        assert set(kwargs["chunk_ids"]) <= loaded
        rebuilt[kwargs["entity_name"]] = kwargs["chunk_ids"]
        return False

    async def fake_relationship(**kwargs):
        loaded = set(kwargs["chunk_data_by_id"])
        assert set(kwargs["chunk_ids"]) <= loaded
        rebuilt[(kwargs["src"], kwargs["tgt"])] = kwargs["chunk_ids"]
        return False

    monkeypatch.setattr(operate, "_load_chunk_extract_rows", spy_load)
    monkeypatch.setattr(operate, "_rebuild_single_entity", fake_entity)
    monkeypatch.setattr(operate, "_rebuild_single_relationship", fake_relationship)

    text_chunks, llm_cache = _stores(CHUNKS)
    entities_vdb, relationships_vdb = _VDB(), _VDB()
    status = {"latest_message": "", "history_messages": []}
    report = await rebuild_knowledge_from_chunks(
        entities_to_rebuild=dict(ENTITIES),
        relationships_to_rebuild=dict(RELATIONSHIPS),
        knowledge_graph_inst=object(),
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 2, "kg_rebuild_window_chunks": window},
        pipeline_status=status,
        pipeline_status_lock=asyncio.Lock(),
    )
    return (
        report,
        loads,
        rebuilt,
        status["history_messages"],
        entities_vdb,
        relationships_vdb,
    )


def test_windows_keep_every_object_whole_and_bound_the_union():
    windows = _split_rebuild_windows(ENTITIES, RELATIONSHIPS, 3)
    unions = [{c for ids in {**e, **r}.values() for c in ids} for e, r in windows]
    assert all(len(u) <= 3 for u in unions)
    # Every object is in exactly one window, in order.
    assert [name for e, _ in windows for name in e] == list(ENTITIES)
    assert [key for _, r in windows for key in r] == list(RELATIONSHIPS)
    assert len(windows) > 1


def test_one_object_over_the_bound_is_a_window_of_its_own():
    big = {"BIG": [f"c{i}" for i in range(10)], "SMALL": ["c1"]}
    windows = _split_rebuild_windows(big, {}, 3)
    assert [list(e) for e, _ in windows] == [["BIG"], ["SMALL"]]


def test_no_bound_is_one_window():
    assert _split_rebuild_windows(ENTITIES, RELATIONSHIPS, 0) == [
        (ENTITIES, RELATIONSHIPS)
    ]


@pytest.mark.asyncio
async def test_a_windowed_rebuild_has_the_outcome_of_a_single_window(monkeypatch):
    one = await _run(monkeypatch, 0)
    many = await _run(monkeypatch, 3)
    report_one, loads_one, rebuilt_one, history_one, evdb_one, rvdb_one = one
    report_many, loads_many, rebuilt_many, history_many, evdb_many, rvdb_many = many

    # Same objects rebuilt from the same chunks; same report.
    assert rebuilt_many == rebuilt_one == {**ENTITIES, **RELATIONSHIPS}
    assert report_many.referenced_chunk_ids == report_one.referenced_chunk_ids
    assert report_many.referenced_chunk_ids == set(CHUNKS)
    assert (
        report_many.missing_cache_chunk_ids
        == report_one.missing_cache_chunk_ids
        == {"gone"}
    )

    # One window loads everything at once and never flushes (the caller does);
    # the windows each load at most the bound and flush the vector buffers.
    assert loads_one == [len(CHUNKS)]
    assert len(loads_many) > 1
    assert max(loads_many) <= 3
    assert evdb_one.flushes == rvdb_one.flushes == 0
    assert evdb_many.flushes == rvdb_many.flushes == len(loads_many)

    # The summaries of one window stay as they were; the windows are numbered.
    assert any(
        m.startswith("Starting parallel rebuild of 4 entities and 2 relationships")
        for m in history_one
    )
    assert not any("window" in m for m in history_one)
    assert sum(m.startswith("Starting parallel rebuild") for m in history_many) == len(
        loads_many
    )
    assert any("(window 1 of" in m for m in history_many)
    assert (
        sum(
            m.startswith("KG rebuild completed: 4 entities and 2 relationships")
            for m in history_many
        )
        == 1
    )


@pytest.mark.asyncio
async def test_windows_without_any_cache_still_rebuild_every_object(monkeypatch):
    # Upstream's rebuild keeps an object no surviving record names (its
    # provenance is repaired), so a window without cached extraction is
    # rebuilt like the others instead of being skipped.
    rebuilt: dict = {}

    async def fake_entity(**kwargs):
        rebuilt[kwargs["entity_name"]] = kwargs["chunk_ids"]
        return True

    monkeypatch.setattr(operate, "_rebuild_single_entity", fake_entity)
    text_chunks, llm_cache = _stores(["gone"])
    report = await rebuild_knowledge_from_chunks(
        entities_to_rebuild={"E1": ["gone"], "E2": ["gone2"]},
        relationships_to_rebuild={},
        knowledge_graph_inst=object(),
        entities_vdb=_VDB(),
        relationships_vdb=_VDB(),
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 1, "kg_rebuild_window_chunks": 1},
    )
    assert rebuilt == {"E1": ["gone"], "E2": ["gone2"]}
    assert report.referenced_chunk_ids == {"gone", "gone2"}
    assert report.missing_cache_chunk_ids == {"gone", "gone2"}
    assert set(report.degraded_entities) == {"E1", "E2"}


@pytest.mark.asyncio
async def test_relation_windows_run_before_entity_windows(monkeypatch):
    # An entity known only through its edges reads edges: with several windows
    # each holds one kind, and every relation window runs before an entity one.
    order: list[str] = []

    async def fake_entity(**kwargs):
        order.append("entity")
        return False

    async def fake_relationship(**kwargs):
        order.append("relation")
        return False

    monkeypatch.setattr(operate, "_rebuild_single_entity", fake_entity)
    monkeypatch.setattr(operate, "_rebuild_single_relationship", fake_relationship)
    text_chunks, llm_cache = _stores(CHUNKS)
    await rebuild_knowledge_from_chunks(
        entities_to_rebuild=dict(ENTITIES),
        relationships_to_rebuild=dict(RELATIONSHIPS),
        knowledge_graph_inst=object(),
        entities_vdb=_VDB(),
        relationships_vdb=_VDB(),
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 2, "kg_rebuild_window_chunks": 3},
    )
    assert order == ["relation"] * len(RELATIONSHIPS) + ["entity"] * len(ENTITIES)


@pytest.mark.asyncio
async def test_a_chunk_in_a_relation_and_an_entity_window_is_extracted_once(
    monkeypatch,
):
    # c1 and c2 are referenced by a relation (window 1) and by E1 (window 2); neither
    # has an extraction cache, so each window that met them would re-extract them.
    # The memo hands window 2 the records window 1 got: the extraction function sees
    # each chunk once, and the entity rebuild reads the records of that one call.
    extracted: list[str] = []
    seen_by_entity: dict[str, dict] = {}

    async def fake_extract_entities(chunks, global_config, **kwargs):
        extracted.extend(chunks)
        return [
            ({f"N-{chunk_id}": [{"description": f"from {chunk_id}"}]}, {})
            for chunk_id in chunks
        ]

    async def fake_entity(**kwargs):
        seen_by_entity[kwargs["entity_name"]] = {
            chunk_id: kwargs["chunk_entities"].get(chunk_id)
            for chunk_id in kwargs["chunk_ids"]
        }
        return False

    async def fake_relationship(**kwargs):
        return False

    monkeypatch.setattr(operate, "extract_entities", fake_extract_entities)
    monkeypatch.setattr(operate, "_rebuild_single_entity", fake_entity)
    monkeypatch.setattr(operate, "_rebuild_single_relationship", fake_relationship)

    chunks = ["c1", "c2", "c3", "c4"]
    text_chunks = _KV(
        {c: {"content": c, "full_doc_id": "doc", "llm_cache_list": []} for c in chunks}
    )
    report = await rebuild_knowledge_from_chunks(
        entities_to_rebuild={"E1": ["c1", "c2"], "E2": ["c3", "c4"]},
        relationships_to_rebuild={("A", "B"): ["c1", "c2"]},
        knowledge_graph_inst=object(),
        entities_vdb=_VDB(),
        relationships_vdb=_VDB(),
        text_chunks_storage=text_chunks,
        llm_response_cache=_KV({}),
        global_config={"llm_model_max_async": 2, "kg_rebuild_window_chunks": 2},
    )

    # Three windows: the relation's, then one per entity.
    assert sorted(extracted) == chunks
    assert report.reextracted_chunk_ids == set(chunks)
    assert seen_by_entity["E1"] == {
        "c1": {"N-c1": [{"description": "from c1"}]},
        "c2": {"N-c2": [{"description": "from c2"}]},
    }
    assert set(seen_by_entity["E2"]) == {"c3", "c4"}
