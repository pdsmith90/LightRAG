"""KEEP SUMMARIES (fork): a deletion that the same document's re-ingestion will
follow can rebuild the entities and relations it shared without the LLM.

Where the surviving fragments join without a summary, the rebuild is exact as
before. Where they would need an LLM summary, the current description stays and
only the provenance (source chunks, file paths, type, keywords, weight) is
rewritten from the survivors: the re-ingested document's merge reads the kept
summary as one fragment and summarizes it with its own new fragments, so the
deletion's LLM pass is skipped rather than moved. Fixtures are invented.
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
from lightrag.utils import Tokenizer
from tests.extraction.test_rebuild_synthesis import (
    DummyGraph,
    DummyKV,
    DummyVector,
    _DummyTokenizer,
)
from tests.pipeline.test_purge_primitive import _KV, _Graph, _make_rag, _node, _status

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
