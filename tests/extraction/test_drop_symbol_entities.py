"""DROP_SYMBOL_ENTITIES: an extracted entity named by a lone symbol or variable is
dropped, and so is every extracted relation that names one.

The extraction LLM turns formula variables ("x", "R", "θ") into entities whatever
the prompt says. With the switch on, ``extract_entities`` arms the filter for the
process and the text and JSON parsers drop those records; a relation goes too, or
the merge would recreate its symbol endpoint as an untyped node. Off, nothing
changes.
"""

import json
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.operate import (
    _configure_symbol_entity_filter,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    _process_json_extraction_result,
    extract_entities,
    is_symbol_entity_name,
)
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _disarm():
    yield
    operate._DROP_SYMBOL_ENTITIES = False


@pytest.mark.parametrize(
    "name", ["x", "R", "θ", "7", "a,", "x.", "12", "t•", "X̂", " p "]
)
def test_symbol_names(name):
    assert is_symbol_entity_name(name)


@pytest.mark.parametrize(
    "name", ["Io", "J2", "pH", "L1", "GM", "ΔT", "x_1", "Acme Corp"]
)
def test_names_that_stay(name):
    assert not is_symbol_entity_name(name)


def _entity(name):
    return ["entity", name, "quantity", f"{name} as used in the text."]


def _relation(src, tgt):
    return ["relation", src, tgt, "related", f"{src} relates to {tgt}."]


def test_armed_filter_drops_symbol_records():
    assert _configure_symbol_entity_filter({"drop_symbol_entities": True}) is True
    assert _handle_single_entity_extraction(_entity("x"), "chunk-1", 0) is None
    kept = _handle_single_entity_extraction(_entity("J2"), "chunk-1", 0)
    assert kept["entity_name"] == "J2"
    assert (
        _handle_single_relationship_extraction(
            _relation("x", "Acme Corp"), "chunk-1", 0
        )
        is None
    )
    assert (
        _handle_single_relationship_extraction(
            _relation("Acme Corp", "θ"), "chunk-1", 0
        )
        is None
    )
    edge = _handle_single_relationship_extraction(
        _relation("J2", "Acme Corp"), "chunk-1", 0
    )
    assert (edge["src_id"], edge["tgt_id"]) == ("J2", "Acme Corp")


def test_disarmed_filter_keeps_symbol_records():
    assert _configure_symbol_entity_filter({}) is False
    kept = _handle_single_entity_extraction(_entity("x"), "chunk-1", 0)
    assert kept["entity_name"] == "x"
    assert (
        _handle_single_relationship_extraction(
            _relation("x", "Acme Corp"), "chunk-1", 0
        )
        is not None
    )


async def test_json_result_drops_symbol_entities_and_their_relations():
    _configure_symbol_entity_filter({"drop_symbol_entities": True})
    result = json.dumps(
        {
            "entities": [
                {"name": n, "type": "quantity", "description": f"{n} in the text."}
                for n in ("x", "J2", "Acme Corp")
            ],
            "relationships": [
                {
                    "source": s,
                    "target": t,
                    "keywords": "related",
                    "description": f"{s} relates to {t}.",
                }
                for s, t in (("x", "Acme Corp"), ("J2", "Acme Corp"))
            ],
        }
    )
    nodes, edges = await _process_json_extraction_result(result, "chunk-1", 0)
    assert set(nodes) == {"J2", "Acme Corp"}
    assert set(edges) == {("J2", "Acme Corp")}


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


async def test_extract_entities_arms_the_filter_from_global_config(monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    llm = AsyncMock(
        return_value=(
            "entity<|#|>x<|#|>Quantity<|#|>The coordinate along the baseline."
            "\nentity<|#|>Acme Corp<|#|>Mission<|#|>A company that makes everything."
            "\nrelation<|#|>x<|#|>Acme Corp<|#|>measured by<|#|>Acme Corp measures x."
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
        "drop_symbol_entities": True,
    }
    chunks = {
        "chunk-001": {
            "tokens": 20,
            "content": "x along the baseline.",
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }

    [(nodes, edges)] = await extract_entities(
        chunks=chunks, global_config=global_config
    )

    assert set(nodes) == {"Acme Corp"}
    assert not edges
