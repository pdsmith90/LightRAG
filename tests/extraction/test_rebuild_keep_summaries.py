"""KEEP SUMMARIES (fork): a deletion that the same document's re-ingestion will
follow can rebuild the entities and relations it shared without the LLM.

Where the surviving fragments join without a summary, the rebuild is exact as
before. Where they would need an LLM summary, the current description stays and
only the provenance (source chunks, file paths, type, keywords, weight) is
rewritten from the survivors. The re-ingested document's merge reads the kept
summary as one fragment and adds its new fragments to it, as it would to a fresh
summary, so the deletion's LLM pass is skipped, not moved. The cost: the kept
summary still carries the deleted document's old contribution until that entity
is next summarized. Fixtures are invented.
"""

from contextlib import asynccontextmanager

import pytest

import lightrag.operate as operate_module
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    KGRebuildReport,
    _handle_entity_relation_summary,
    _rebuild_single_entity,
    _rebuild_single_relationship,
)
from lightrag.utils import Tokenizer, load_json, make_relation_chunk_key, write_json
from tests.extraction.test_rebuild_synthesis import (
    DummyGraph,
    DummyKV,
    DummyVector,
    _DummyTokenizer,
)
from tests.pipeline.test_purge_primitive import (
    _KV,
    _DocStatus,
    _Graph,
    _make_rag,
    _node,
    _status,
)

pytestmark = pytest.mark.offline

SEP = GRAPH_FIELD_SEP


def _config(**overrides):
    # No LLM function at all: any summary call would fail the test.
    config = {
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 3,
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
        "source_ids_limit_method": "KEEP",
    }
    config.update(overrides)
    return config


@pytest.mark.asyncio
async def test_a_summary_that_needs_the_llm_returns_the_kept_text():
    # three fragments reach force_llm_summary_on_merge: the LLM would summarize
    text, used = await _handle_entity_relation_summary(
        "Entity", "QUUX", ["a", "b", "c"], SEP, _config(), llm_fallback="kept"
    )
    assert (text, used) == ("kept", False)
    # fragments too long for one pass would go through the map phase: kept too
    text, used = await _handle_entity_relation_summary(
        "Entity",
        "QUUX",
        ["alpha" * 4, "bravo" * 4, "charlie" * 4],
        SEP,
        _config(summary_context_size=30, force_llm_summary_on_merge=10),
        llm_fallback="kept",
    )
    assert (text, used) == ("kept", False)
    # below the thresholds the fragments still join, exactly as without it
    text, used = await _handle_entity_relation_summary(
        "Entity", "QUUX", ["a", "b"], SEP, _config(), llm_fallback="kept"
    )
    assert (text, used) == (SEP.join(["a", "b"]), False)


@pytest.mark.asyncio
async def test_an_entity_keeps_its_summary_and_takes_the_survivors_provenance():
    graph = DummyGraph()
    await graph.upsert_node(
        "QUUX",
        {
            "description": "Current summary of Quux",
            "entity_type": "METHOD",
            "source_id": SEP.join(["c0", "c1", "c2", "c3"]),
            "file_path": SEP.join(["gone.md", "a.md", "b.md", "c.md"]),
        },
    )
    chunk_entities = {
        f"c{i}": {
            "QUUX": [
                {
                    "entity_type": "METHOD",
                    "description": f"Quux fragment {i}",
                    "file_path": f"{name}.md",
                }
            ]
        }
        for i, name in ((1, "a"), (2, "b"), (3, "c"))
    }
    await _rebuild_single_entity(
        knowledge_graph_inst=graph,
        entities_vdb=DummyVector(),
        entity_name="QUUX",
        chunk_ids=["c1", "c2", "c3"],
        chunk_entities=chunk_entities,
        llm_response_cache=DummyKV(),
        global_config=_config(),
        keep_summaries=True,
    )
    node = await graph.get_node("QUUX")
    assert node["description"] == "Current summary of Quux"
    assert node["source_id"] == SEP.join(["c1", "c2", "c3"])
    assert node["file_path"] == SEP.join(["a.md", "b.md", "c.md"])

    # an entity whose survivors join without the LLM is rebuilt exactly
    await graph.upsert_node(
        "BLORP",
        {"description": "Old", "entity_type": "METHOD", "source_id": "c0"},
    )
    await _rebuild_single_entity(
        knowledge_graph_inst=graph,
        entities_vdb=DummyVector(),
        entity_name="BLORP",
        chunk_ids=["c1"],
        chunk_entities={
            "c1": {"BLORP": [{"entity_type": "METHOD", "description": "Fresh"}]}
        },
        llm_response_cache=DummyKV(),
        global_config=_config(),
        keep_summaries=True,
    )
    assert (await graph.get_node("BLORP"))["description"] == "Fresh"


@pytest.mark.asyncio
async def test_a_relation_keeps_its_summary_and_takes_the_survivors_keywords():
    graph = DummyGraph()
    await graph.upsert_node("A", {"entity_type": "ENT", "description": "Node A"})
    await graph.upsert_node("B", {"entity_type": "ENT", "description": "Node B"})
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "Current relation summary",
            "keywords": "gone",
            "weight": 4.0,
            "source_id": SEP.join(["c0", "c1", "c2", "c3"]),
            "file_path": "gone.md",
        },
    )
    chunk_relationships = {
        f"c{i}": {("A", "B"): [{"keywords": f"kw{i}", "description": f"rel {i}"}]}
        for i in (1, 2, 3)
    }
    await _rebuild_single_relationship(
        knowledge_graph_inst=graph,
        relationships_vdb=DummyVector(),
        entities_vdb=DummyVector(),
        src="A",
        tgt="B",
        chunk_ids=["c1", "c2", "c3"],
        chunk_relationships=chunk_relationships,
        llm_response_cache=DummyKV(),
        global_config=_config(),
        keep_summaries=True,
    )
    edge = await graph.get_edge("A", "B")
    assert edge["description"] == "Current relation summary"
    assert edge["keywords"] == "kw1, kw2, kw3"
    assert edge["source_id"] == SEP.join(["c1", "c2", "c3"])


@pytest.mark.asyncio
async def test_the_purge_passes_keep_summaries_to_the_rebuild(monkeypatch):
    @asynccontextmanager
    async def _noop_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(operate_module, "get_storage_keyed_lock", _noop_lock)
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1", "c2"])})
    rag = _make_rag(
        graph=graph,
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1", "c2"]}}),
        text_chunks=_KV({"c2": {"content": "x", "llm_cache_list": []}}),
    )
    seen = []

    async def spy(*args, **kwargs):
        seen.append(kwargs.get("keep_summaries"))
        return KGRebuildReport()

    monkeypatch.setattr(operate_module, "rebuild_knowledge_from_chunks", spy)
    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", spy)
    for keep in (True, False):
        status, lock = _status()
        await rag._purge_kg_contributions(
            "d1",
            ["c1"],
            candidate_entities=["ALICE"],
            candidate_relations=[],
            patch_only=True,
            keep_summaries=keep,
            pipeline_status=status,
            pipeline_status_lock=lock,
        )
    assert seen == [True, False]


@pytest.mark.asyncio
async def test_the_rebuild_reports_which_summaries_it_kept():
    graph = DummyGraph()
    await graph.upsert_node(
        "QUUX", {"description": "Kept", "entity_type": "METHOD", "source_id": "c0"}
    )
    await graph.upsert_node(
        "BLORP", {"description": "Old", "entity_type": "METHOD", "source_id": "c0"}
    )
    kept: set = set()
    many = {
        f"c{i}": {"QUUX": [{"entity_type": "METHOD", "description": f"q{i}"}]}
        for i in (1, 2, 3)
    }
    for name, chunk_ids, chunk_entities in (
        ("QUUX", ["c1", "c2", "c3"], many),
        ("BLORP", ["c1"], {"c1": {"BLORP": [{"description": "Fresh"}]}}),
    ):
        await _rebuild_single_entity(
            knowledge_graph_inst=graph,
            entities_vdb=DummyVector(),
            entity_name=name,
            chunk_ids=chunk_ids,
            chunk_entities=chunk_entities,
            llm_response_cache=DummyKV(),
            global_config=_config(),
            keep_summaries=True,
            kept=kept,
        )
    # only the entity whose summary would have needed the LLM is reported
    assert kept == {"QUUX"}


def _resummary_rag(tmp_path, statuses):
    rag = _make_rag(
        graph=_Graph(nodes={"QUUX": _node("QUUX", ["c1", "c9"])}),
        entity_chunks=_KV({"QUUX": {"chunk_ids": ["c1", "c9"], "count": 2}}),
        relation_chunks=_KV(
            {make_relation_chunk_key("BLORP", "QUUX"): {"chunk_ids": ["c9"]}}
        ),
        doc_status=_DocStatus(statuses),
    )
    rag.working_dir = str(tmp_path)
    return rag


@pytest.mark.asyncio
async def test_kept_summaries_are_rebuilt_once_the_document_is_back(
    tmp_path, monkeypatch
):
    report = KGRebuildReport()
    report.kept_entities.add("QUUX")
    report.kept_relationships.add(("BLORP", "QUUX"))
    rag = _resummary_rag(tmp_path, {"d1": {"status": "processing"}})
    rag._record_kept_summaries("d1", report)
    registry = load_json(rag._kept_summaries_path())
    assert registry["d1"]["entities"] == ["QUUX"]
    assert registry["d1"]["relations"] == [["BLORP", "QUUX"]]

    calls = []

    async def spy(**kwargs):
        calls.append(kwargs)
        return KGRebuildReport()

    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", spy)
    status, lock = _status()
    # not processed yet: nothing rebuilt, the entry waits
    await rag._resummarize_kept_summaries(status, lock)
    assert calls == [] and "d1" in load_json(rag._kept_summaries_path())

    # back in the KB: rebuilt with the LLM from every current chunk, entry dropped
    rag.doc_status.data["d1"]["status"] = "processed"
    await rag._resummarize_kept_summaries(status, lock)
    assert len(calls) == 1
    assert calls[0]["entities_to_rebuild"] == {"QUUX": ["c1", "c9"]}
    assert calls[0]["relationships_to_rebuild"] == {("BLORP", "QUUX"): ["c9"]}
    assert not calls[0].get("keep_summaries")
    assert load_json(rag._kept_summaries_path()) == {}


@pytest.mark.asyncio
async def test_a_failed_resummary_keeps_the_entry_for_the_next_batch(
    tmp_path, monkeypatch
):
    report = KGRebuildReport()
    report.kept_entities.add("QUUX")
    rag = _resummary_rag(tmp_path, {"d1": {"status": "processed"}})
    rag._record_kept_summaries("d1", report)

    async def boom(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", boom)
    status, lock = _status()
    await rag._resummarize_kept_summaries(status, lock)  # logs, never raises
    assert "d1" in load_json(rag._kept_summaries_path())


@pytest.mark.asyncio
async def test_a_document_that_never_returns_gets_its_summaries_rebuilt_anyway(
    tmp_path, monkeypatch
):
    report = KGRebuildReport()
    report.kept_entities.add("QUUX")
    rag = _resummary_rag(tmp_path, {})  # the document is gone from doc_status
    rag._record_kept_summaries("d1", report)
    calls = []

    async def spy(**kwargs):
        calls.append(kwargs)
        return KGRebuildReport()

    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", spy)
    status, lock = _status()
    await rag._resummarize_kept_summaries(status, lock)
    assert calls == []  # recent: it may still come back
    registry = load_json(rag._kept_summaries_path())
    registry["d1"]["kept_at"] -= rag._KEPT_SUMMARIES_MAX_AGE_S + 1
    write_json(registry, rag._kept_summaries_path())
    await rag._resummarize_kept_summaries(status, lock)
    assert calls and calls[0]["entities_to_rebuild"] == {"QUUX": ["c1", "c9"]}
    assert load_json(rag._kept_summaries_path()) == {}
