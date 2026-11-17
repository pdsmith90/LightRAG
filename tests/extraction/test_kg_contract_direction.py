"""Extraction contract: relation direction in the graph (lightrag/kg_contract.py).

The graph stores each pair of entities once, in sorted order. With
``relationship_inverses`` in the profile, every write path puts a relation into
that order and, when that flips it, stores the inverse names of its directed
types, so a stored edge (src, tgt) with keyword k always reads "src k tgt".
Without the key every write is what it was. The LLM never sees an inverse name.
All names below are invented.
"""

import asyncio
import json
import shutil
from uuid import uuid4

import numpy as np
import pytest

import lightrag.entity_name_fold as fold_mod
import lightrag.entity_name_guard as guard
from lightrag import LightRAG
from lightrag.entity_name_fold import fold_chunk_result, load_fold_map
from lightrag.kg_contract import (
    RELATIONSHIP_INVERSES_V31,
    RELATIONSHIP_TYPES_V31,
    inverse_table,
    add_reverse_keywords,
    invert_keywords,
    orient_relation,
    oriented_record,
    profile_status,
    schema_for_profile,
)
from lightrag.operate import merge_nodes_and_edges
from lightrag.prompt import (
    load_entity_extraction_prompt_profile,
    resolve_entity_extraction_prompt_profile,
    validate_entity_extraction_prompt_profile_for_mode,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

from tests.extraction.test_kg_contract import (
    CONTRACT_SAMPLE,
    PLAIN_SAMPLE,
    _sent_prompts,
)

pytestmark = pytest.mark.offline

TABLE = inverse_table(RELATIONSHIP_INVERSES_V31)


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """PROMPT_DIR holding both sample profiles (as in test_kg_contract.py)."""
    (tmp_path / "entity_type").mkdir()
    shutil.copy(CONTRACT_SAMPLE, tmp_path / "entity_type" / "contract.yml")
    shutil.copy(PLAIN_SAMPLE, tmp_path / "entity_type" / "plain.yml")
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "9999999")
    return tmp_path


# ------------------------------------------------------------ the rewrite


def test_the_sample_profile_declares_the_v31_inverses():
    profile = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    assert profile["relationship_inverses"] == RELATIONSHIP_INVERSES_V31
    symmetric = set(RELATIONSHIP_TYPES_V31) - set(RELATIONSHIP_INVERSES_V31)
    assert symmetric == {"same_as", "compared_with", "related_to"}


def test_inverse_table_goes_both_ways_and_is_none_without_inverses():
    assert TABLE["part_of"] == "has_part" and TABLE["has_part"] == "part_of"
    assert TABLE["depends_on"] == "determines" and TABLE["determines"] == "depends_on"
    assert inverse_table(None) is None and inverse_table({}) is None


@pytest.mark.parametrize(
    "keywords,expected",
    [
        ("part_of", "has_part"),
        ("has_part", "part_of"),
        ("same_as", "same_as"),
        ("part_of,has_property", "has_part,property_of"),
        ("part_of, related_to , free text", "has_part, related_to , free text"),
        ("Part_Of,partof", "Part_Of,partof"),  # only exact names are types
        ("", ""),
    ],
)
def test_invert_keywords_rewrites_each_known_type(keywords, expected):
    assert invert_keywords(keywords, TABLE) == expected


def test_orient_relation_sorts_and_inverts_a_flipped_pair():
    assert orient_relation("GRACE", "Constellation", "part_of", TABLE) == (
        "Constellation",
        "GRACE",
        "has_part",
    )
    assert orient_relation("Constellation", "GRACE", "has_part", TABLE) == (
        "Constellation",
        "GRACE",
        "has_part",
    )
    # without a table the pair is sorted and the keywords stay
    assert orient_relation("GRACE", "Constellation", "part_of", None) == (
        "Constellation",
        "GRACE",
        "part_of",
    )


def test_a_record_carries_its_reverse_reading_and_is_oriented_by_it():
    """The extraction merge and the rebuild orient a relation record through
    ``oriented_record`` alone; the reverse reading is resolved before, with the
    table, by ``add_reverse_keywords``."""
    shared = {"src_id": "x", "tgt_id": "y", "keywords": "uses", "weight": 1.0}
    edges = {
        ("B", "A"): [shared],
        ("A", "C"): [{"keywords": "part_of"}],
        ("D", "E"): [shared],  # one object under two pairs (a hook may do it)
    }
    add_reverse_keywords(edges, TABLE)
    assert shared["src_id"] == "x"  # copies, the input is left alone
    assert edges[("D", "E")][0]["src_id"] == "D"
    [record] = edges[("B", "A")]
    assert record == {
        "src_id": "B",
        "tgt_id": "A",
        "keywords": "uses",
        "weight": 1.0,
        "reverse_keywords": "used_by",
    }
    assert oriented_record(record) == {
        "src_id": "A",
        "tgt_id": "B",
        "keywords": "used_by",
        "weight": 1.0,
    }
    assert record["keywords"] == "uses"  # the input is not modified
    # in order: only the reverse reading goes
    assert oriented_record(edges[("A", "C")][0]) == {
        "src_id": "A",
        "tgt_id": "C",
        "keywords": "part_of",
    }
    # without a table the reverse reading is the same keywords
    edges = {("B", "A"): [{"keywords": "uses"}]}
    add_reverse_keywords(edges, None)
    assert oriented_record(edges[("B", "A")][0])["keywords"] == "uses"
    # a record without one (no contract, no hook) is returned as it is
    plain = {"src_id": "B", "tgt_id": "A", "keywords": "uses"}
    assert oriented_record(plain) is plain


# ------------------------------------------------------------ the profile key


def _write_profile(path, **fields):
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"relationship_inverses": {"part_of": "has_part"}}, "needs 'relationship_types'"),
        ({"relationship_types": {"part_of": "x"}, "relationship_inverses": "has_part"}, "non-empty mapping"),
        ({"relationship_types": {"part_of": "x"}, "relationship_inverses": {}}, "non-empty mapping"),
        ({"relationship_types": {"part_of": "x"}, "relationship_inverses": {"is_a": "has_kind"}}, "not a type"),
        ({"relationship_types": {"part_of": "x"}, "relationship_inverses": {"part_of": "Has Part"}}, "snake case"),
        ({"relationship_types": {"part_of": "x"}, "relationship_inverses": {"part_of": 3}}, "snake case"),
        ({"relationship_types": {"part_of": "x", "uses": "y"}, "relationship_inverses": {"part_of": "uses"}}, "also a type"),
        ({"relationship_types": {"part_of": "x", "uses": "y"}, "relationship_inverses": {"part_of": "has", "uses": "has"}}, "more than one type"),
    ],
)  # fmt: skip
def test_the_loader_rejects_malformed_inverses(tmp_path, fields, message):
    path = _write_profile(tmp_path / "p.yml", **fields)
    with pytest.raises(ValueError, match=message):
        load_entity_extraction_prompt_profile(path)


def test_the_inverses_reach_the_cached_profile(prompt_dir):
    resolved = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "contract.yml"}, True
    )
    validated = validate_entity_extraction_prompt_profile_for_mode(resolved, True)
    assert validated["relationship_inverses"] == RELATIONSHIP_INVERSES_V31


def test_no_table_without_the_key(prompt_dir):
    plain = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "plain.yml"}, True
    )
    assert inverse_table(plain.get("relationship_inverses")) is None


async def test_the_llm_never_sees_an_inverse_name(prompt_dir):
    """The contract profile with and without the inverses key sends the same
    prompts, and the schema keeps the forward list."""
    no_inverses = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    del no_inverses["relationship_inverses"]
    (prompt_dir / "entity_type" / "noinv.yml").write_text(
        json.dumps(no_inverses), encoding="utf-8"
    )
    with_key = await _sent_prompts(True, {"entity_type_prompt_file": "contract.yml"})
    without = await _sent_prompts(True, {"entity_type_prompt_file": "noinv.yml"})
    assert with_key == without
    sent = "\n".join(p for pair in with_key for p in pair)
    for inverse in RELATIONSHIP_INVERSES_V31.values():
        assert inverse not in sent, inverse
    schema = schema_for_profile(CONTRACT_SAMPLE)
    rel_enum = schema["properties"]["relationships"]["items"]["properties"]
    assert rel_enum["keywords"]["enum"] == list(RELATIONSHIP_TYPES_V31)
    assert schema == schema_for_profile(prompt_dir / "entity_type" / "noinv.yml")


def test_profile_status_counts_types_and_inverses():
    assert profile_status(None) == {"relationship_types": 0, "relationship_inverses": 0}
    assert profile_status(load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)) == {
        "relationship_types": 17,
        "relationship_inverses": 14,
    }
    assert profile_status(load_entity_extraction_prompt_profile(PLAIN_SAMPLE)) == {
        "relationship_types": 0,
        "relationship_inverses": 0,
    }


# ------------------------------------------------------------ merge, unit level


class _RecordingGraph:
    """A graph that keys edges by the pair it was handed and records each write."""

    def __init__(self):
        self.nodes, self.edges, self.writes = {}, {}, []

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src, tgt):
        return (src, tgt) in self.edges or (tgt, src) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges.pop((tgt, src), None)
        self.edges[(src, tgt)] = dict(edge_data)
        self.writes.append((src, tgt, edge_data.get("keywords")))


def _edge_record(src, tgt, keywords, chunk="chunk-1"):
    return {
        "src_id": src,
        "tgt_id": tgt,
        "keywords": keywords,
        "description": f"{src} {keywords} {tgt}.",
        "source_id": chunk,
        "weight": 1.0,
        "file_path": "doc.txt",
    }


def _node_record(name, chunk="chunk-1"):
    return {
        "entity_name": name,
        "entity_type": "Concept",
        "description": f"{name}.",
        "source_id": chunk,
        "file_path": "doc.txt",
    }


async def _merge(chunk_results, profile, graph=None):
    """merge_nodes_and_edges with ``profile`` as the cached prompt profile (None:
    a config without one, as custom wiring builds it). Under a contract each
    relation record first gets its reverse reading, as extraction gives it
    before the records reach the merge."""
    graph = graph or _RecordingGraph()
    if profile and profile.get("relationship_types"):
        table = inverse_table(profile.get("relationship_inverses"))
        for _nodes, edges in chunk_results:
            add_reverse_keywords(edges, table)
    global_config = {
        "tokenizer": Tokenizer("characters", _Characters()),
        "llm_model_max_async": 1,
        "max_source_ids_per_entity": 100,
        "max_source_ids_per_relation": 100,
        "source_ids_limit_method": "KEEP",
        "max_file_paths": 100,
        "file_path_more_placeholder": "more",
        "force_llm_summary_on_merge": 100,
        "summary_max_tokens": 100_000,
        "summary_context_size": 100_000,
        "workspace": f"direction-{uuid4().hex}",
    }
    if profile is not None:
        global_config["_entity_extraction_prompt_profile"] = profile
    await merge_nodes_and_edges(
        chunk_results=chunk_results,
        knowledge_graph_inst=graph,
        entity_vdb=None,
        relationships_vdb=None,
        global_config=global_config,
        pipeline_status={"history_messages": []},
        pipeline_status_lock=asyncio.Lock(),
    )
    return graph


def _chunk(*relations, chunk="chunk-1"):
    names = {n for src, _k, tgt in relations for n in (src, tgt)}
    nodes = {n: [_node_record(n, chunk)] for n in names}
    edges = {}
    for src, keywords, tgt in relations:
        edges.setdefault((src, tgt), []).append(_edge_record(src, tgt, keywords, chunk))
    return nodes, edges


CONTRACT = {"relationship_types": dict(RELATIONSHIP_TYPES_V31)}
CONTRACT_WITH_INVERSES = {
    **CONTRACT,
    "relationship_inverses": dict(RELATIONSHIP_INVERSES_V31),
}


@pytest.fixture
def _shared_storage():
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data()


@pytest.mark.usefixtures("_shared_storage")
@pytest.mark.parametrize(
    "src,tgt", [("GRACE", "Constellation"), ("Constellation", "GRACE")]
)
async def test_merge_keeps_the_direction_for_both_endpoint_orders(src, tgt):
    keywords = "part_of"
    graph = await _merge([_chunk((src, keywords, tgt))], CONTRACT_WITH_INVERSES)
    stored = "has_part" if src > tgt else "part_of"
    assert graph.writes == [("Constellation", "GRACE", stored)]
    assert graph.edges[("Constellation", "GRACE")]["keywords"] == stored


@pytest.mark.usefixtures("_shared_storage")
async def test_two_chunks_stating_opposite_directions_merge_readably():
    """A part_of B (flipped by the sort) and B has_property A: both read
    "Constellation k GRACE" in the stored order."""
    graph = await _merge(
        [
            _chunk(("GRACE", "part_of", "Constellation"), chunk="chunk-1"),
            _chunk(("Constellation", "has_property", "GRACE"), chunk="chunk-2"),
        ],
        CONTRACT_WITH_INVERSES,
    )
    assert graph.writes == [("Constellation", "GRACE", "has_part,has_property")]


@pytest.mark.usefixtures("_shared_storage")
async def test_a_multi_keyword_record_rewrites_each_known_type():
    graph = await _merge(
        [_chunk(("Zeta", "uses, same_as, made up", "Alpha"))], CONTRACT_WITH_INVERSES
    )
    [(src, tgt, keywords)] = graph.writes
    assert (src, tgt) == ("Alpha", "Zeta")
    assert set(keywords.split(",")) == {"used_by", "same_as", "made up"}


@pytest.mark.usefixtures("_shared_storage")
async def test_a_later_document_merges_onto_the_stored_reading():
    """Zeta uses Alpha is stored (Alpha, Zeta) used_by; a later document's
    Zeta uses Alpha, flipped the same way, adds nothing new."""
    graph = await _merge([_chunk(("Zeta", "uses", "Alpha"))], CONTRACT_WITH_INVERSES)
    await _merge(
        [_chunk(("Zeta", "uses", "Alpha"), chunk="chunk-2")],
        CONTRACT_WITH_INVERSES,
        graph,
    )
    assert graph.writes == [("Alpha", "Zeta", "used_by"), ("Alpha", "Zeta", "used_by")]
    assert graph.edges[("Alpha", "Zeta")]["source_id"].count("chunk-") == 2


@pytest.mark.usefixtures("_shared_storage")
@pytest.mark.parametrize(
    "profile", [None, {}, CONTRACT], ids=["none", "plain", "contract"]
)
async def test_without_the_key_the_merge_writes_what_it_wrote_before(profile):
    """Sorted pair, the extracted keywords, as on the base branch."""
    graph = await _merge(
        [
            _chunk(("GRACE", "part_of", "Constellation"), chunk="chunk-1"),
            _chunk(("Constellation", "has_property", "GRACE"), chunk="chunk-2"),
        ],
        profile,
    )
    assert graph.writes == [("Constellation", "GRACE", "has_property,part_of")]


@pytest.fixture
def _armed_fold(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (frozenset(), frozenset()))
    fold_mod.disarm()
    load_fold_map([("Imaginary Filter", "method", 40), ("Quux Basin", "location", 7)])
    yield
    fold_mod.disarm()


@pytest.mark.usefixtures("_shared_storage", "_armed_fold")
async def test_the_name_fold_renames_before_the_pair_is_oriented():
    """'imaginary filter' sorts after 'Quux Basin', its canonical spelling
    before it: the orientation follows the folded names."""
    nodes, edges = _chunk(("imaginary filter", "applied_to", "Quux Basin"))
    add_reverse_keywords(edges, TABLE)  # extraction: before the fold
    folded = fold_chunk_result(nodes, edges)
    assert set(folded[1]) == {("Imaginary Filter", "Quux Basin")}
    graph = await _merge([folded], CONTRACT_WITH_INVERSES)
    assert graph.writes == [("Imaginary Filter", "Quux Basin", "applied_to")]

    nodes, edges = _chunk(("Quux Basin", "located_in", "imaginary filter"))
    add_reverse_keywords(edges, TABLE)
    folded = fold_chunk_result(nodes, edges)
    graph = await _merge([folded], CONTRACT_WITH_INVERSES)
    assert graph.writes == [("Imaginary Filter", "Quux Basin", "location_of")]


# ------------------------------------------------------------ end to end


def _input_text(prompt: str) -> str:
    marker = "---Input Text---\n```\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    return prompt[start : prompt.find("\n```", start)]


# chunk text -> the relations the stub LLM extracts from it (source, type, target)
STATEMENTS = {
    "GRACE is one of the satellites of the Constellation.": [
        ("GRACE", "part_of", "Constellation")
    ],
    "The Constellation is described by its member GRACE.": [
        ("Constellation", "has_property", "GRACE")
    ],
}


async def _stub_llm(prompt, system_prompt=None, history_messages=None, **_):
    text = _input_text(prompt)
    if not text:
        return "A summary."
    relations = [
        r for sentence, rs in STATEMENTS.items() if sentence in text for r in rs
    ]
    names = sorted({n for s, _k, t in relations for n in (s, t)})
    return json.dumps(
        {
            "entities": [
                {"name": n, "type": "Mission", "description": f"{n} is named."}
                for n in names
            ],
            "relationships": [
                {
                    "source": s,
                    "target": t,
                    "keywords": k,
                    "description": f"{s} {k} {t}.",
                }
                for s, k, t in relations
            ],
        }
    )


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


@pytest.fixture
def contract_profiles(prompt_dir):
    """contract.yml has the inverses; noinv.yml is the same without them."""
    no_inverses = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    del no_inverses["relationship_inverses"]
    (prompt_dir / "entity_type" / "noinv.yml").write_text(
        json.dumps(no_inverses), encoding="utf-8"
    )
    return prompt_dir


async def _make_rag(tmp_path, profile_file):
    addon = {"entity_type_prompt_file": profile_file} if profile_file else {}
    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace=f"direction_{uuid4().hex}",
        llm_model_func=_stub_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=0,
        entity_extraction_use_json=True,
        chunk_token_size=4000,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        addon_params=addon,
    )
    await rag.initialize_storages()
    writes = []
    graph = rag.chunk_entity_relation_graph
    original = graph.upsert_edge

    async def recording_upsert_edge(src, tgt, edge_data):
        writes.append((src, tgt, edge_data.get("keywords")))
        return await original(src, tgt, edge_data)

    graph.upsert_edge = recording_upsert_edge
    return rag, writes


async def _keywords(rag, a, b):
    edge = await rag.chunk_entity_relation_graph.get_edge(a, b)
    return edge and edge.get("keywords")


@pytest.mark.parametrize(
    "profile_file,merged,rebuilt",
    [
        ("contract.yml", "has_part,has_property", "has_part"),
        ("noinv.yml", "has_property,part_of", "part_of"),
        (None, "has_property,part_of", "part_of"),
    ],
    ids=["inverses", "contract-without-inverses", "default"],
)
async def test_insert_then_delete_rebuilds_in_the_stored_reading(
    contract_profiles, tmp_path, profile_file, merged, rebuilt
):
    """Two documents state the relation in opposite directions; deleting one
    rebuilds the edge from the other's cached extraction."""
    rag, writes = await _make_rag(tmp_path, profile_file)
    try:
        sentences = list(STATEMENTS)
        await rag.ainsert(sentences[0], ids=["doc-part"], file_paths=["part.txt"])
        await rag.ainsert(sentences[1], ids=["doc-prop"], file_paths=["prop.txt"])
        assert await _keywords(rag, "Constellation", "GRACE") == merged
        assert {(s, t) for s, t, _k in writes} == {("Constellation", "GRACE")}

        writes.clear()
        result = await rag.adelete_by_doc_id("doc-prop")
        assert result.status == "success"
        assert (await _keywords(rag, "Constellation", "GRACE")).replace(" ", "") == (
            rebuilt
        )
        assert [(s, t) for s, t, _k in writes] == [("Constellation", "GRACE")]
    finally:
        await rag.finalize_storages()


async def _two_entities(rag, *names):
    for name in names:
        await rag.acreate_entity(
            name, {"description": f"{name} exists.", "entity_type": "Concept"}
        )


@pytest.mark.parametrize("profile_file", ["contract.yml", "noinv.yml"])
async def test_create_and_edit_relation_in_the_callers_order(
    contract_profiles, tmp_path, profile_file
):
    rag, writes = await _make_rag(tmp_path, profile_file)
    inverses = profile_file == "contract.yml"
    try:
        await _two_entities(rag, "Alpha", "Zeta")
        await rag.acreate_relation(
            "Zeta", "Alpha", {"description": "Zeta uses Alpha.", "keywords": "uses"}
        )
        if inverses:
            assert writes == [("Alpha", "Zeta", "used_by")]
        else:  # as before: the caller's order and keywords
            assert writes == [("Zeta", "Alpha", "uses")]

        writes.clear()
        await rag.aedit_relation("Zeta", "Alpha", {"keywords": "part_of"})
        expected = "has_part" if inverses else "part_of"
        assert writes == [("Alpha", "Zeta", expected)]
        assert await _keywords(rag, "Alpha", "Zeta") == expected
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("profile_file", ["contract.yml", "noinv.yml"])
async def test_rename_that_flips_the_pair(contract_profiles, tmp_path, profile_file):
    """Alpha part_of Mid is stored (Alpha, Mid); renaming Alpha to Zed moves
    the pair to (Mid, Zed), where it reads Mid has_part Zed."""
    rag, writes = await _make_rag(tmp_path, profile_file)
    inverses = profile_file == "contract.yml"
    try:
        await _two_entities(rag, "Alpha", "Mid")
        await rag.acreate_relation(
            "Alpha", "Mid", {"description": "Alpha is in Mid.", "keywords": "part_of"}
        )
        writes.clear()
        await rag.aedit_entity("Alpha", {"entity_name": "Zed"})
        if inverses:
            assert writes == [("Mid", "Zed", "has_part")]
            assert await _keywords(rag, "Mid", "Zed") == "has_part"
        else:  # as before: the order get_node_edges gave, keywords unchanged
            assert writes == [("Zed", "Mid", "part_of")]
            assert await _keywords(rag, "Mid", "Zed") == "part_of"
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("profile_file", ["contract.yml", "noinv.yml"])
async def test_entity_merge_that_flips_and_joins_edges(
    contract_profiles, tmp_path, profile_file
):
    """Alpha part_of Mid and Mid uses Beta; merging Alpha and Beta into Zed
    puts both on (Mid, Zed): Mid has_part Zed and Mid uses Zed."""
    rag, writes = await _make_rag(tmp_path, profile_file)
    inverses = profile_file == "contract.yml"
    try:
        await _two_entities(rag, "Alpha", "Beta", "Mid")
        await rag.acreate_relation(
            "Alpha", "Mid", {"description": "Alpha is in Mid.", "keywords": "part_of"}
        )
        await rag.acreate_relation(
            "Mid", "Beta", {"description": "Mid uses Beta.", "keywords": "uses"}
        )
        writes.clear()
        await rag.amerge_entities(["Alpha", "Beta"], "Zed")
        [(src, tgt, keywords)] = writes
        if inverses:
            assert (src, tgt) == ("Mid", "Zed")
            assert set(keywords.split(",")) == {"has_part", "uses"}
        else:  # as before: the keywords as stored, joined
            assert set(keywords.split(",")) == {"part_of", "uses"}
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("profile_file", ["contract.yml", "noinv.yml"])
async def test_custom_kg_relations_in_the_callers_order(
    contract_profiles, tmp_path, profile_file
):
    rag, _writes = await _make_rag(tmp_path, profile_file)
    try:
        await rag.ainsert_custom_kg(
            {
                "chunks": [
                    {
                        "content": "Zeta and Alpha.",
                        "chunk_order_index": 0,
                        "source_id": "c1",
                    }
                ],
                "entities": [],
                "relationships": [
                    {
                        "src_id": "Zeta",
                        "tgt_id": "Alpha",
                        "description": "Zeta is in Alpha.",
                        "keywords": "part_of",
                        "source_id": "c1",
                    }
                ],
            }
        )
        expected = "has_part" if profile_file == "contract.yml" else "part_of"
        assert await _keywords(rag, "Alpha", "Zeta") == expected
    finally:
        await rag.finalize_storages()
