"""ENTITY_NAME_FOLD: names that differ only in case, spacing or punctuation are
mapped to the spelling the graph already uses for them.

The map is filled from the graph's nodes (best-connected spelling first) and is
armed when complete; extraction results are renamed on their way out of
extract_entities, and the knowledge rebuild files cached records under both the
original and the canonical name. Apparatus names never become canonical. Off,
nothing changes. All names below are invented.
"""

from unittest.mock import AsyncMock

import pytest

import lightrag.entity_name_fold as fold_mod
import lightrag.entity_name_guard as guard
from lightrag.entity_name_fold import (
    alias_chunk_result,
    canonical_name,
    fold_chunk_result,
    fold_key,
    load_fold_map,
)
from lightrag.operate import extract_entities
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

NODES = [
    ("Imaginary Filter", "method", 40),
    ("imaginary filter", "method", 3),
    ("ZORP-FO", "mission", 25),
    ("Table 2", "other", 90),  # apparatus: never canonical
    ("TIDE", "instrument", 5),
    ("Quux Basin", "location", 7),
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (frozenset(), frozenset()))
    fold_mod.disarm()
    yield
    fold_mod.disarm()


def test_fold_key_rules():
    assert (
        fold_key("Imaginary Filter")
        == fold_key("imaginary-filter")
        == "imaginaryfilter"
    )
    assert fold_key("CO") is None and fold_key("Sar") is None  # under four characters
    assert fold_key("TIDE") != fold_key("tide")  # short acronym vs word kept apart
    assert fold_key("ZORPFOX") == fold_key("Zorpfox")  # longer: case does not matter
    assert fold_key("Model 2") != fold_key("Model 3")


def test_disarmed_passes_names_through():
    nodes = {"imaginary filter": [{"entity_name": "imaginary filter"}]}
    assert fold_chunk_result(nodes, {}) == (nodes, {})
    assert canonical_name("imaginary filter") == "imaginary filter"


def test_best_connected_spelling_wins_and_apparatus_is_skipped():
    assert load_fold_map(NODES) == 4  # Table 2 skipped; two spellings share a key
    assert canonical_name("Imaginary-Filter") == "Imaginary Filter"
    assert canonical_name("imaginary filter") == "Imaginary Filter"
    assert canonical_name("Zorp-FO") == "ZORP-FO"
    assert canonical_name("table 2") == "table 2"  # not mapped onto the apparatus node
    assert canonical_name("tide") == "tide"  # the acronym TIDE stays separate


def test_new_key_registers_first_spelling():
    load_fold_map(NODES)
    assert canonical_name("Blorft Index") == "Blorft Index"
    assert canonical_name("blorft-index") == "Blorft Index"
    assert canonical_name("Wibble Rate", register=False) == "Wibble Rate"
    assert canonical_name("wibble rate") == "wibble rate"  # the peek did not register


def test_chunk_result_is_renamed_and_merged():
    load_fold_map(NODES)
    nodes = {
        "imaginary filter": [{"entity_name": "imaginary filter", "description": "a"}],
        "Imaginary-Filter": [{"entity_name": "Imaginary-Filter", "description": "b"}],
        "Quux basin": [{"entity_name": "Quux basin", "description": "c"}],
    }
    edges = {
        ("imaginary filter", "Quux basin"): [
            {"src_id": "imaginary filter", "tgt_id": "Quux basin"}
        ],
        ("imaginary filter", "Imaginary-Filter"): [
            {"src_id": "imaginary filter", "tgt_id": "Imaginary-Filter"}
        ],
    }
    n, e = fold_chunk_result(nodes, edges)
    assert set(n) == {"Imaginary Filter", "Quux Basin"}
    assert [r["description"] for r in n["Imaginary Filter"]] == ["a", "b"]
    assert all(r["entity_name"] == "Imaginary Filter" for r in n["Imaginary Filter"])
    assert set(e) == {("Imaginary Filter", "Quux Basin")}  # the self-loop is gone
    [rec] = e[("Imaginary Filter", "Quux Basin")]
    assert (rec["src_id"], rec["tgt_id"]) == ("Imaginary Filter", "Quux Basin")


def test_rebuild_alias_keeps_both_spellings():
    load_fold_map(NODES)
    nodes = {"imaginary filter": [{"entity_name": "imaginary filter"}]}
    edges = {
        ("imaginary filter", "Quux basin"): [
            {"src_id": "imaginary filter", "tgt_id": "Quux basin"}
        ]
    }
    n, e = alias_chunk_result(nodes, edges)
    assert set(n) == {"imaginary filter", "Imaginary Filter"}
    assert n["imaginary filter"][0]["entity_name"] == "imaginary filter"
    assert n["Imaginary Filter"][0]["entity_name"] == "Imaginary Filter"
    assert set(e) == {
        ("imaginary filter", "Quux basin"),
        ("Imaginary Filter", "Quux Basin"),
    }


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


async def test_extract_entities_returns_canonical_names(monkeypatch):
    load_fold_map(NODES)
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    llm = AsyncMock(
        return_value=(
            "entity<|#|>imaginary-filter<|#|>Method<|#|>A filter for imaginary data."
            "\nentity<|#|>Quux basin<|#|>Location<|#|>A basin under study."
            "\nrelation<|#|>imaginary-filter<|#|>Quux basin<|#|>applied to<|#|>The filter is applied to the basin."
            "\n<|COMPLETE|>"
        )
    )
    global_config = {
        "llm_model_func": llm,
        "role_llm_funcs": {"extract": llm, "keyword": llm, "query": llm, "vlm": llm},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _CharTokenizer()),
        "llm_model_max_async": 1,
    }
    chunks = {
        "chunk-001": {
            "tokens": 20,
            "content": "The imaginary filter is applied to the Quux basin.",
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }
    [(nodes, edges)] = await extract_entities(
        chunks=chunks, global_config=global_config
    )
    assert set(nodes) == {"Imaginary Filter", "Quux Basin"}
    assert set(edges) == {("Imaginary Filter", "Quux Basin")}


async def test_build_fold_map_reads_the_graph():
    class _Graph:
        graph_name = "g"

        async def _query(self, sql):
            assert '"g".base' in sql
            return [{"name": n, "type": t, "degree": d} for n, t, d in NODES]

    assert await fold_mod.build_fold_map(_Graph()) == 4
    assert fold_mod.is_armed()


async def test_build_fold_map_needs_a_supported_graph():
    assert await fold_mod.build_fold_map(object()) is None
    assert not fold_mod.is_armed()
