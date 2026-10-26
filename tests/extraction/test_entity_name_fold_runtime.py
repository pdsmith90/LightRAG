"""ENTITY_NAME_FOLD at runtime: what the server does around the map.

- The pipeline waits for the startup map build before it folds extraction
  results or rebuilds from cached records (bounded), so nothing runs unfolded.
- A missing or malformed lists file arms the map with the default lists; a bad
  alias file arms it without aliases; the graph query is retried. The outcome is
  kept for /health.
- Spellings the legacy key folded together still meet (legacy fallback); an
  alias resolves through the map and chains of aliases are followed.
- A key new to the map takes a chunk's best-ranked spelling.
- A rebuild of a merged canonical finds the records cached under its variants'
  spellings and rebuilds its edges.

All names below are invented.
"""

import asyncio
import copy
import logging
import time
from contextlib import asynccontextmanager

import pytest

import lightrag.entity_name_fold as fold_mod
import lightrag.entity_name_guard as guard
import lightrag.operate as operate_module
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.entity_name_fold import (
    alias_chunk_result,
    canonical_name,
    fold_chunk_result,
    fold_key,
    legacy_fold_key,
    load_aliases,
    load_fold_lists,
    load_fold_map,
)
from lightrag.operate import KGRebuildReport, rebuild_knowledge_from_chunks
from lightrag.utils import Tokenizer, TokenizerInterface
from tests.pipeline.test_purge_primitive import (
    _KV,
    _DocStatus,
    _Graph,
    _make_rag,
    _node,
    _status,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (frozenset(), frozenset()))
    monkeypatch.delenv("ENTITY_NAME_FOLD_LISTS", raising=False)
    monkeypatch.delenv("ENTITY_NAME_ALIASES", raising=False)
    monkeypatch.setattr(fold_mod, "_QUERY_RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    fold_mod.disarm()
    load_fold_lists(None)
    yield
    fold_mod.disarm()
    load_fold_lists(None)


@pytest.fixture
def _noop_keyed_lock(monkeypatch):
    @asynccontextmanager
    async def _noop_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(operate_module, "get_storage_keyed_lock", _noop_lock)


NODES = [
    ("Glorbon Model", "model", 12),
    ("Quux Basin", "location", 7),
    ("Zorp Analysis Kit", "software", 40),
]
ALIASES = [("Zorp Toolkit", "Glorbon Model")]


class _FoldGraph:
    graph_name = "g"

    def __init__(self, nodes, fail=0):
        self.nodes = nodes
        self.fail = fail
        self.calls = 0

    async def _query(self, sql):
        self.calls += 1
        if self.calls <= self.fail:
            raise ConnectionError("graph not reachable yet")
        return [{"name": n, "type": t, "degree": d} for n, t, d in self.nodes]


def _armed_line(text):
    return next(ln for ln in text.splitlines() if "ENTITY_NAME_FOLD armed" in ln)


# ------------------------------------------------------- lists and alias files


async def test_a_missing_lists_file_arms_with_the_default_lists(
    tmp_path, monkeypatch, caplog
):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("ENTITY_NAME_FOLD_LISTS", str(missing))
    with caplog.at_level(logging.INFO, logger="lightrag"):
        assert await fold_mod.build_fold_map(_FoldGraph(NODES)) == 3
    assert fold_mod.is_armed()
    assert any(
        r.levelno == logging.ERROR and "DEFAULT lists" in r.getMessage()
        for r in caplog.records
    )
    line = _armed_line(caplog.text)
    assert "3 name keys, 0 aliases, lists defaults (FALLBACK: FileNotFoundError" in line
    # a check that wants the installed file in the line must fail
    assert f"lists {missing}," not in line
    state = fold_mod.fold_state()
    assert state["state"] == "armed" and state["keys"] == 3
    assert state["lists"].startswith("defaults (FALLBACK: ")


async def test_a_malformed_lists_file_arms_with_the_default_lists(
    tmp_path, monkeypatch, caplog
):
    bad = tmp_path / "fold_lists.json"
    bad.write_text('{"no_fold": "zorbels"}', encoding="utf-8")  # not a list
    monkeypatch.setenv("ENTITY_NAME_FOLD_LISTS", str(bad))
    with caplog.at_level(logging.INFO, logger="lightrag"):
        assert await fold_mod.build_fold_map(_FoldGraph(NODES)) == 3
    assert "lists defaults (FALLBACK: ValueError: " in _armed_line(caplog.text)
    assert fold_mod.NO_FOLD == fold_mod._DEFAULT_NO_FOLD


async def test_a_missing_alias_file_arms_without_aliases(tmp_path, monkeypatch, caplog):
    load_aliases(ALIASES)  # left over from an earlier load
    monkeypatch.setenv("ENTITY_NAME_ALIASES", str(tmp_path / "missing.csv"))
    with caplog.at_level(logging.INFO, logger="lightrag"):
        assert await fold_mod.build_fold_map(_FoldGraph(NODES)) == 3
    assert fold_mod.is_armed()
    assert any(
        r.levelno == logging.ERROR and "alias file" in r.getMessage()
        for r in caplog.records
    )
    assert "3 name keys, 0 aliases, lists defaults," in _armed_line(caplog.text)
    assert canonical_name("Zorp Toolkit") == "Zorp Toolkit"
    assert fold_mod.fold_state()["aliases"] == 0


def test_short_and_empty_alias_rows_are_skipped_and_counted(tmp_path, caplog):
    path = tmp_path / "aliases.csv"
    path.write_text(
        "variant,canonical\nZorp Toolkit,Glorbon Model\nFlarn Basin\n,Quux Basin\n"
        "Snarf Engine,\n\nFlarn Basin,Quux Basin\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert load_aliases(path) == 2
    assert "skipped 3 rows with fewer than two fields or an empty field" in caplog.text
    assert fold_mod._ALIASES[fold_key("Flarn Basin")] == "Quux Basin"


def test_an_alias_file_with_a_bom_does_not_install_its_header(tmp_path):
    path = tmp_path / "aliases.csv"
    path.write_bytes(
        "variant,canonical\nZorp Toolkit,Glorbon Model\n".encode("utf-8-sig")
    )
    assert load_aliases(path) == 1
    assert fold_key("variant") not in fold_mod._ALIASES
    assert fold_mod._ALIASES == {fold_key("Zorp Toolkit"): "Glorbon Model"}


async def test_a_bad_alias_row_never_keeps_the_map_off(tmp_path, monkeypatch):
    path = tmp_path / "aliases.csv"
    path.write_text("Zorp Toolkit,Glorbon Model\nFlarn Basin\n", encoding="utf-8")
    monkeypatch.setenv("ENTITY_NAME_ALIASES", str(path))
    assert await fold_mod.build_fold_map(_FoldGraph(NODES)) == 3
    assert canonical_name("zorp toolkit") == "Glorbon Model"


async def test_the_graph_query_is_retried(caplog):
    graph = _FoldGraph(NODES, fail=2)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert await fold_mod.build_fold_map(graph) == 3
    assert graph.calls == 3
    assert "graph query failed (attempt 1/4" in caplog.text


async def test_a_query_that_keeps_failing_records_the_failure():
    graph = _FoldGraph(NODES, fail=10)
    with pytest.raises(ConnectionError):
        await fold_mod.build_fold_map(graph)
    assert graph.calls == 4
    assert not fold_mod.is_armed()
    assert fold_mod.fold_state()["state"].startswith("failed:ConnectionError")


async def test_the_state_reads_pending_while_the_map_is_built(monkeypatch):
    seen = []

    def compute(rows):
        seen.append(fold_mod.fold_state()["state"])
        return {}, {}

    monkeypatch.setattr(fold_mod, "_compute_fold_map", compute)
    assert fold_mod.fold_state() == {"state": "off"}
    await fold_mod.build_fold_map(_FoldGraph(NODES))
    assert seen == ["pending"]
    assert fold_mod.fold_state() == {
        "state": "armed",
        "lists": "defaults",
        "aliases": 0,
        "keys": 0,
    }


async def test_an_unsupported_graph_records_the_failure():
    assert await fold_mod.build_fold_map(object()) is None
    assert fold_mod.fold_state()["state"].startswith("failed:")


async def test_a_relative_path_is_read_as_absolute(tmp_path, monkeypatch, caplog):
    (tmp_path / "aliases.csv").write_text(
        "Zorp Toolkit,Glorbon Model\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTITY_NAME_ALIASES", "aliases.csv")
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert await fold_mod.build_fold_map(_FoldGraph(NODES)) == 3
    assert f"using {tmp_path / 'aliases.csv'}" in caplog.text
    assert fold_mod.fold_state()["aliases"] == 1


# ------------------------------------------------------------- legacy fallback


def test_a_spelling_the_legacy_key_folded_still_meets_its_node():
    # the graph holds only the all-capitals spelling; its Titlecase plural has
    # another key now, but the legacy key joined them
    assert legacy_fold_key("Zorbles") == legacy_fold_key("ZORBLES")
    assert fold_key("Zorbles") != fold_key("ZORBLES")
    load_fold_map([("ZORBLES", "concept", 5)])
    assert canonical_name("Zorbles", register=False) == "ZORBLES"
    assert fold_key("Zorbles") not in fold_mod._MAP  # register=False
    assert canonical_name("Zorbles") == "ZORBLES"
    # registered: the singular, which shares the plural's key, follows
    assert fold_mod._MAP[fold_key("Zorbles")] == "ZORBLES"
    assert canonical_name("Zorble") == "ZORBLES"


def test_the_key_map_wins_over_the_legacy_fallback():
    load_fold_map([("ZORBLES", "concept", 5), ("Zorble", "concept", 3)])
    assert canonical_name("Zorbles") == "Zorble"
    n, _e = alias_chunk_result({"Zorbles": [{"entity_name": "Zorbles"}]}, {})
    # the rebuild files the record under both nodes it may belong to
    assert set(n) == {"Zorbles", "Zorble", "ZORBLES"}
    assert n["ZORBLES"][0]["entity_name"] == "ZORBLES"


def test_the_legacy_fallback_reaches_relation_endpoints_in_the_rebuild():
    load_fold_map(
        [("ZORBLES", "concept", 5), ("Zorble", "concept", 3), ("Quux Basin", "x", 1)]
    )
    _n, e = alias_chunk_result(
        {}, {("Zorbles", "Quux Basin"): [{"src_id": "Zorbles", "tgt_id": "Quux Basin"}]}
    )
    assert set(e) == {
        ("Zorbles", "Quux Basin"),
        ("Zorble", "Quux Basin"),
        ("ZORBLES", "Quux Basin"),
    }
    assert e[("ZORBLES", "Quux Basin")][0]["src_id"] == "ZORBLES"


@pytest.mark.parametrize("caps_degree, expected", [(5, "Zorble"), (50, "ZORBLES")])
def test_the_legacy_canonical_is_that_of_the_best_ranked_member(caps_degree, expected):
    # "ZORBLES" and "zorbles" share a legacy key but not a key; the
    # best-connected of the two decides, through its own key's canonical
    load_fold_map(
        [
            ("ZORBLES", "concept", caps_degree),
            ("zorbles", "concept", 9),
            ("Zorble", "concept", 20),  # the canonical of "zorbles"'s key
        ]
    )
    assert fold_mod._MAP[fold_key("zorbles")] == "Zorble"
    assert fold_mod._LEGACY[legacy_fold_key("Zorbles")] == expected


def test_symbolic_names_never_take_the_legacy_fallback():
    assert legacy_fold_key("Flarn′") == legacy_fold_key("Flarn")
    load_fold_map([("Flarn", "concept", 4)])
    assert canonical_name("Flarn′") == "Flarn′"
    assert legacy_fold_key("Flarn′") in fold_mod._LEGACY
    # nor does a symbolic graph name become one
    fold_mod.disarm()
    load_fold_map([("Flarn′", "symbol", 4)])
    assert fold_mod._LEGACY == {}
    assert canonical_name("Flarn") == "Flarn"


def test_a_new_name_keeps_its_legacy_key_for_later_spellings():
    load_fold_map([("Quux Basin", "location", 7)])
    assert canonical_name("ZORBLES") == "ZORBLES"  # new: registered
    assert canonical_name("Zorbles") == "ZORBLES"  # legacy key of the new name


# ----------------------------------------------------- aliases through the map


def test_an_alias_resolves_through_the_map():
    load_aliases([("The Flarn", "Flarn")])
    load_fold_map([("Flarns", "concept", 4)])  # the map spells the key "Flarns"
    assert fold_key("Flarn") == fold_key("Flarns")
    assert canonical_name("The Flarn") == "Flarns"
    assert canonical_name("Flarn") == "Flarns"


def test_an_alias_whose_canonical_is_not_in_the_map_registers_it():
    load_aliases([("The Flarn", "Flarn")])
    load_fold_map([("Quux Basin", "location", 7)])
    assert canonical_name("The Flarn", register=False) == "Flarn"
    assert fold_key("Flarn") not in fold_mod._MAP
    assert canonical_name("The Flarn") == "Flarn"
    assert canonical_name("flarns") == "Flarn"  # the same key now meets it


def test_alias_chains_are_followed_and_loops_dropped(caplog):
    rows = [
        ("Zorp Toolkit", "Snarf Engine"),
        ("snarf-engine kit", "Glorbon Model"),  # unrelated, no chain
        ("Snarf Engines", "Quux Basin"),  # Snarf Engine's key -> Quux Basin
        ("Blorpic Index", "Wibble Rate"),
        ("Wibble Rates", "Blorpic Index"),  # loops back
    ]
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        assert load_aliases(rows) == 3
    assert "1 aliases name a canonical that is itself aliased" in caplog.text
    assert "dropped 2 aliases whose chain of canonicals loops" in caplog.text
    assert fold_mod._ALIASES[fold_key("Zorp Toolkit")] == "Quux Basin"
    assert fold_key("Blorpic Index") not in fold_mod._ALIASES
    assert fold_key("Wibble Rate") not in fold_mod._ALIASES
    load_fold_map(NODES)
    assert canonical_name("zorp toolkit") == "Quux Basin"


# ------------------------------------------------ best spelling of a new key


def test_a_new_key_takes_the_chunks_best_ranked_spelling():
    load_fold_map(NODES)
    nodes = {
        "Zorp Models": [{"entity_name": "Zorp Models", "entity_type": "model"}],
        " Zorp Gadget": [{"entity_name": " Zorp Gadget", "entity_type": "device"}],
        "Zorp Model": [{"entity_name": "Zorp Model", "entity_type": "model"}],
    }
    edges = {
        ("Zorp Models", "Zorp Gadget"): [
            {"src_id": "Zorp Models", "tgt_id": "Zorp Gadget"}
        ]
    }
    n, e = fold_chunk_result(nodes, edges)
    # singular before plural, a clean spelling before stray whitespace -- not
    # the first name seen
    assert set(n) == {"Zorp Model", "Zorp Gadget"}
    assert set(e) == {("Zorp Model", "Zorp Gadget")}
    assert canonical_name("zorp models") == "Zorp Model"


def test_a_known_key_keeps_the_graphs_spelling():
    load_fold_map(NODES)
    n, _e = fold_chunk_result(
        {
            "Glorbon Models": [{"entity_name": "Glorbon Models"}],
            "glorbon model": [{"entity_name": "glorbon model"}],
        },
        {},
    )
    assert set(n) == {"Glorbon Model"}


# ----------------------------------------------------------- the rebuild path


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _MemGraph:
    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src, tgt):
        return (src, tgt) in self.edges or (tgt, src) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


class _MemVdb:
    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass


def _entity(name, description, chunk_id):
    return {
        "entity_name": name,
        "entity_type": "model",
        "description": description,
        "source_id": chunk_id,
        "file_path": f"{chunk_id}.md",
        "timestamp": 1,
    }


def _relation(src, tgt, description, chunk_id):
    return {
        "src_id": src,
        "tgt_id": tgt,
        "weight": 1.0,
        "description": description,
        "keywords": "flow",
        "source_id": chunk_id,
        "file_path": f"{chunk_id}.md",
        "timestamp": 1,
    }


# What the extraction cache holds for each chunk: the spellings the LLM wrote
# before the variants were merged into "Glorbon Model".
CACHED = {
    "c1": (
        {
            "Glorbon Models": [_entity("Glorbon Models", "Glorbon fragment one", "c1")],
            "Quux Basin": [_entity("Quux Basin", "Quux fragment one", "c1")],
        },
        {
            ("Glorbon Models", "Quux Basin"): [
                _relation("Glorbon Models", "Quux Basin", "flows into", "c1")
            ]
        },
    ),
    "c2": (
        {
            "Zorp Toolkit": [_entity("Zorp Toolkit", "Glorbon fragment two", "c2")],
            "Quux Basin": [_entity("Quux Basin", "Quux fragment two", "c2")],
        },
        {
            ("Zorp Toolkit", "Quux Basin"): [
                _relation("Zorp Toolkit", "Quux Basin", "drains", "c2")
            ]
        },
    ),
    "c3": (
        {"Glorbon Model": [_entity("Glorbon Model", "Glorbon fragment three", "c3")]},
        {},
    ),
}


async def _merged_graph():
    graph = _MemGraph()
    for name, chunks in (("Glorbon Model", "c1 c2 c3"), ("Quux Basin", "c1 c2")):
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "model",
                "description": f"merged summary of {name}",
                "source_id": GRAPH_FIELD_SEP.join(chunks.split()),
                "file_path": "merged.md",
            },
        )
    await graph.upsert_edge(
        "Glorbon Model",
        "Quux Basin",
        {
            "description": "merged relation summary",
            "keywords": "flow",
            "weight": 2.0,
            "source_id": GRAPH_FIELD_SEP.join(["c1", "c2"]),
            "file_path": "merged.md",
        },
    )
    return graph


async def _rebuild_merged(monkeypatch):
    async def cached(llm_response_cache, chunk_ids, text_chunks_storage):
        return {c: [(c, 1)] for c in sorted(chunk_ids) if c in CACHED}, {}

    async def parsed(*, chunk_id, **_kwargs):
        return copy.deepcopy(CACHED[chunk_id])

    monkeypatch.setattr(operate_module, "_get_cached_extraction_results", cached)
    monkeypatch.setattr(operate_module, "_rebuild_from_extraction_result", parsed)
    graph = await _merged_graph()
    await rebuild_knowledge_from_chunks(
        entities_to_rebuild={"Glorbon Model": ["c1", "c2", "c3"]},
        relationships_to_rebuild={("Glorbon Model", "Quux Basin"): ["c1", "c2"]},
        knowledge_graph_inst=graph,
        entities_vdb=_MemVdb(),
        relationships_vdb=_MemVdb(),
        text_chunks_storage=_KV(),
        llm_response_cache=_KV(),
        global_config={
            "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
            "summary_context_size": 1_000_000,
            "summary_max_tokens": 1_000_000,
            "force_llm_summary_on_merge": 6,
            "source_ids_limit_method": "KEEP",
            "max_source_ids_per_entity": 10,
            "max_source_ids_per_relation": 10,
            "max_file_paths": 10,
            "file_path_more_placeholder": "...",
            "llm_model_max_async": 1,
        },
    )
    return graph


async def test_an_armed_rebuild_uses_the_records_of_merged_variants(
    monkeypatch, _noop_keyed_lock
):
    load_aliases(ALIASES)
    load_fold_map(NODES)
    graph = await _rebuild_merged(monkeypatch)
    node = graph.nodes["Glorbon Model"]
    for fragment in ("one", "two", "three"):
        assert f"Glorbon fragment {fragment}" in node["description"]
    edge = await graph.get_edge("Glorbon Model", "Quux Basin")
    assert "flows into" in edge["description"] and "drains" in edge["description"]
    assert edge["source_id"].split(GRAPH_FIELD_SEP) == ["c1", "c2"]


async def test_an_unarmed_rebuild_misses_them(monkeypatch, _noop_keyed_lock):
    # the reason the pipeline waits for the map: unarmed, only the records under
    # the exact canonical spelling are found and the edge is not rebuilt
    graph = await _rebuild_merged(monkeypatch)
    node = graph.nodes["Glorbon Model"]
    assert "Glorbon fragment three" in node["description"]
    assert "Glorbon fragment one" not in node["description"]
    edge = await graph.get_edge("Glorbon Model", "Quux Basin")
    assert "flows into" not in edge["description"]


# ------------------------------------------------- waiting for the startup map


def _pending_build():
    """A build task that arms the map (with ALIASES) once ``gate`` is set."""
    gate = asyncio.Event()

    async def build():
        await gate.wait()
        load_aliases(ALIASES)
        load_fold_map(NODES)

    return gate, asyncio.create_task(build())


def _copies_seen():
    nodes, _edges = alias_chunk_result({"Zorp Toolkit": [{"entity_name": "x"}]}, {})
    return sorted(nodes)


async def _until_waiting(task):
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert not task.done()


async def test_a_purge_rebuild_waits_for_the_map(monkeypatch, _noop_keyed_lock):
    graph = _Graph(nodes={"Glorbon Model": _node("Glorbon Model", ["c1", "c2"])})
    rag = _make_rag(
        graph=graph,
        entity_chunks=_KV({"Glorbon Model": {"chunk_ids": ["c1", "c2"]}}),
        text_chunks=_KV({"c2": {"content": "x", "llm_cache_list": []}}),
    )
    seen = []

    async def spy(*args, **kwargs):
        seen.append(_copies_seen())
        return KGRebuildReport()

    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", spy)
    gate, rag._fold_map_task = _pending_build()
    rag._fold_map_deadline = time.monotonic() + 30
    status, lock = _status()
    purge = asyncio.create_task(
        rag._purge_kg_contributions(
            "d1",
            ["c1"],
            candidate_entities=["Glorbon Model"],
            candidate_relations=[],
            patch_only=True,
            pipeline_status=status,
            pipeline_status_lock=lock,
        )
    )
    await _until_waiting(purge)
    assert seen == []
    gate.set()
    await purge
    assert seen == [["Glorbon Model", "Zorp Toolkit"]]


async def test_the_kept_summary_rebuild_waits_for_the_map(tmp_path, monkeypatch):
    rag = _make_rag(
        graph=_Graph(nodes={"Glorbon Model": _node("Glorbon Model", ["c1"])}),
        entity_chunks=_KV({"Glorbon Model": {"chunk_ids": ["c1"], "count": 1}}),
        doc_status=_DocStatus({"d1": {"status": "processed"}}),
    )
    rag.working_dir = str(tmp_path)
    report = KGRebuildReport()
    report.kept_entities.add("Glorbon Model")
    rag._record_kept_summaries("d1", report)
    seen = []

    async def spy(**kwargs):
        seen.append(_copies_seen())
        return KGRebuildReport()

    monkeypatch.setattr("lightrag.lightrag.rebuild_knowledge_from_chunks", spy)
    gate, rag._fold_map_task = _pending_build()
    rag._fold_map_deadline = time.monotonic() + 30
    status, lock = _status()
    job = asyncio.create_task(rag._resummarize_kept_summaries(status, lock))
    await _until_waiting(job)
    gate.set()
    await job
    assert seen == [["Glorbon Model", "Zorp Toolkit"]]


async def test_extraction_waits_for_the_map(monkeypatch):
    rag = _make_rag(graph=_Graph(nodes={}))
    seen = []

    async def spy(chunk, **kwargs):
        seen.append(fold_mod.is_armed())
        return []

    monkeypatch.setattr("lightrag.lightrag.extract_entities", spy)
    gate, rag._fold_map_task = _pending_build()
    rag._fold_map_deadline = time.monotonic() + 30
    job = asyncio.create_task(rag._process_extract_entities({}))
    await _until_waiting(job)
    gate.set()
    await job
    assert seen == [True]
    await rag._process_extract_entities({})  # armed: no wait
    assert seen == [True, True]


async def test_the_wait_is_bounded(monkeypatch, caplog):
    rag = _make_rag(graph=_Graph(nodes={}))
    seen = []

    async def spy(chunk, **kwargs):
        seen.append(fold_mod.is_armed())
        return []

    monkeypatch.setattr("lightrag.lightrag.extract_entities", spy)
    gate, rag._fold_map_task = _pending_build()
    rag._fold_map_deadline = time.monotonic() + 0.05
    try:
        with caplog.at_level(logging.ERROR, logger="lightrag"):
            await rag._process_extract_entities({})
            await rag._process_extract_entities({})
        assert seen == [False, False]
        assert caplog.text.count("still not armed") == 1
    finally:
        rag._fold_map_task.cancel()


async def test_no_build_task_means_no_wait(monkeypatch):
    rag = _make_rag(graph=_Graph(nodes={}))

    async def spy(chunk, **kwargs):
        return ["done"]

    monkeypatch.setattr("lightrag.lightrag.extract_entities", spy)
    assert await rag._process_extract_entities({}) == ["done"]
