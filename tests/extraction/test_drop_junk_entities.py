"""DROP_JUNK_ENTITIES: extracted entities that are bibliography or document
apparatus are dropped, and so is every extracted relation that names one.

A paper corpus feeds the extraction LLM reference lists, in-text citations and
captions; the model turns cited authors, citations, journal names, numbered
figure/table/equation labels and placeholder words into entities whatever the
prompt says. With the switch on, the text and JSON parsers drop those records --
judging an entity on its name and on the type the LLM wrote, before the type
allow-list maps it -- and drop the relations naming them, or the merge would
recreate the endpoint as an untyped node. Off, nothing changes.

All names below are invented.
"""

import json
from unittest.mock import AsyncMock

import pytest

import lightrag.entity_name_guard as guard
import lightrag.operate as operate
from lightrag.entity_name_guard import fold, junk_entity_class
from lightrag.operate import (
    _configure_entity_type_allowlist,
    _configure_junk_entity_filter,
    _process_extraction_result,
    _process_json_extraction_result,
    extract_entities,
)
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

SURNAMES = frozenset({"doe", "roe", "moe", "quill"})
VENUES = frozenset({fold("Journal of Invented Studies"), fold("Imaginary Problems")})


@pytest.fixture(autouse=True)
def _library(monkeypatch):
    monkeypatch.setattr(guard, "_library_lists", lambda: (SURNAMES, VENUES))
    yield
    operate._DROP_JUNK_ENTITIES = False
    operate._ENTITY_TYPE_ALLOWLIST = None


@pytest.mark.parametrize(
    "name, raw_type, cls",
    [
        ("Doe, J.", "other", "person"),
        ("Quill, A. B.", "organization", "person"),
        ("J. Doe", "other", "person"),
        ("Doe J", "unknown", "person"),
        ("Jane Doe", "person", "person"),
        ("Gauss", "person", "person"),
        ("Roe et al. (2001)", "other", "citation"),
        ("Roe _et al._ (2001)", "content", "citation"),
        ("Roe and Moe (1999)", "concept", "citation"),
        ("Roe, 2003", "other", "citation"),
        ("ROE AND MOE 2001", "other", "citation"),
        ("Roe and Moe", "other", "citation"),
        ("Ann. Imag. Lett.", "content", "citation"),
        ("Journal of Invented Studies", "content", "citation"),
        (
            "Journal of Invented Studies",
            "other",
            "citation",
        ),  # a journal word in the title
        ("Imaginary Problems", "journal", "citation"),
        ("https://example.org/data", "other", "citation"),
        ("10.9999/abc.123", "other", "citation"),
        ("Table 2", "other", "label"),
        ("Fig. 3a", "concept", "label"),
        ("Eq. (12)", "method", "label"),
        ("Lemma B.I", "other", "label"),
        ("Appendix E", "content", "label"),
        ("Concept", "concept", "generic"),
        ("Table Name", "other", "generic"),
        ("1999", "other", "numeric"),
    ],
)
def test_apparatus_names(name, raw_type, cls):
    assert junk_entity_class(name, raw_type) == cls


@pytest.mark.parametrize(
    "name, raw_type",
    [
        ("Equator", "location"),
        ("Equinox", "concept"),
        ("Equatorial Plasma Bubble", "concept"),
        ("Equation of state", "concept"),
        ("Tablet", "other"),
        ("J. Zed", "other"),  # surname not in the library
        ("Zed and Ned", "other"),
        ("Hayford (1909)", "model"),  # an ellipsoid named after its year
        ("Hayford 1909", "other"),  # bare year: product names look like this
        ("Journal of Invented Studies", "concept"),
        ("Imaginary Problems", "other"),  # a journal title that also names a field
        ("Doe and Roe", "dataset"),  # a data set named after its authors
        ("Example Archive (https://example.org)", "dataset"),
        ("N. Imaginaria", "location"),
        ("Satellite", "artifact"),
        ("2011 Imaginary earthquake", "event"),
        ("Kalman filter", "method"),
        ("x", "quantity"),  # one character belongs to DROP_SYMBOL_ENTITIES
    ],
)
def test_names_that_stay(name, raw_type):
    assert junk_entity_class(name, raw_type) is None


@pytest.mark.parametrize(
    "name, raw_type, cls",
    [
        # a real name with its citation appended is judged by the name
        ("IMAGIMODEL-00 (Roe et al. 2002)", "method", None),
        ("Imaginary Threshold Retracker (Moe et al., 2006)", "method", None),
        ("Thermal models [Doe, 2003]", "other", None),
        ("Table 2 (Roe et al., 2001)", "other", "label"),
        ("Doe, J. (2004)", "other", "person"),
        # a list of surnames before the year is the citation itself
        ("Doe and Roe (2002)", "other", "citation"),
        ("van Moe & Le Roe (1976a)", "other", "citation"),
    ],
)
def test_trailing_citation(name, raw_type, cls):
    assert junk_entity_class(name, raw_type) == cls


def test_untyped_endpoint_uses_the_name_alone():
    assert junk_entity_class("Doe, J.", None) == "person"
    assert junk_entity_class("Ann. Imag. Lett.", None) == "citation"
    assert junk_entity_class("Journal of Invented Studies", None) == "citation"
    # without a type, a bare title or a surname pair may be a field or a data set
    assert junk_entity_class("Imaginary Problems", None) is None
    assert junk_entity_class("Doe and Roe", None) is None


def _json_result(entities, relationships):
    return json.dumps(
        {
            "entities": [
                {"name": n, "type": t, "description": f"{n} in the text."}
                for n, t in entities
            ],
            "relationships": [
                {
                    "source": s,
                    "target": t,
                    "keywords": "related",
                    "description": f"{s} relates to {t}.",
                }
                for s, t in relationships
            ],
        }
    )


async def test_json_result_drops_apparatus_and_their_relations():
    assert _configure_junk_entity_filter({"drop_junk_entities": True}) is True
    result = _json_result(
        [
            ("Jane Doe", "Person"),
            ("Table 2", "Other"),
            ("Imaginary Basin", "Location"),
            ("Kalman filter", "Method"),
        ],
        [
            ("Jane Doe", "Kalman filter"),  # endpoint dropped by its type
            ("Table 2", "Imaginary Basin"),
            ("Roe et al. (2001)", "Imaginary Basin"),  # junk by name, no entity record
            ("Kalman filter", "Imaginary Basin"),
        ],
    )
    nodes, edges = await _process_json_extraction_result(result, "chunk-1", 0)
    assert set(nodes) == {"Imaginary Basin", "Kalman filter"}
    assert set(edges) == {("Kalman filter", "Imaginary Basin")}
    assert all("_raw_type" not in r for rs in nodes.values() for r in rs)


async def test_person_type_counts_before_the_allow_list_maps_it():
    _configure_entity_type_allowlist(
        {"entity_type_strict": True}, "- Location: places\n- Method: techniques\n"
    )
    _configure_junk_entity_filter({"drop_junk_entities": True})
    result = _json_result(
        [("Jane Doe", "Person"), ("Imaginary Basin", "Location")],
        [("Jane Doe", "Imaginary Basin")],
    )
    nodes, edges = await _process_json_extraction_result(result, "chunk-1", 0)
    assert set(nodes) == {"Imaginary Basin"}
    assert not edges


async def test_text_result_drops_apparatus_and_their_relations():
    _configure_junk_entity_filter({"drop_junk_entities": True})
    result = (
        "entity<|#|>Doe, J.<|#|>Other<|#|>An author of a cited paper."
        "\nentity<|#|>Imaginary Basin<|#|>Location<|#|>A basin under study."
        "\nentity<|#|>Concept<|#|>Concept<|#|>A placeholder."
        "\nrelation<|#|>Doe, J.<|#|>Imaginary Basin<|#|>studied<|#|>Doe studied the basin."
        "\nrelation<|#|>Concept<|#|>Imaginary Basin<|#|>about<|#|>A placeholder edge."
        "\n<|COMPLETE|>"
    )
    nodes, edges = await _process_extraction_result(result, "chunk-1", 0)
    assert set(nodes) == {"Imaginary Basin"}
    assert not edges
    assert all("_raw_type" not in r for rs in nodes.values() for r in rs)


async def test_disarmed_filter_keeps_everything():
    assert _configure_junk_entity_filter({}) is False
    result = _json_result(
        [("Jane Doe", "Person"), ("Table 2", "Other")], [("Jane Doe", "Table 2")]
    )
    nodes, edges = await _process_json_extraction_result(result, "chunk-1", 0)
    assert set(nodes) == {"Jane Doe", "Table 2"}
    assert set(edges) == {("Jane Doe", "Table 2")}
    assert all("_raw_type" not in r for rs in nodes.values() for r in rs)


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


async def test_extract_entities_arms_the_filter_from_global_config(monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    llm = AsyncMock(
        return_value=(
            "entity<|#|>Roe et al. (2001)<|#|>Other<|#|>A cited study."
            "\nentity<|#|>Imaginary Basin<|#|>Location<|#|>A basin under study."
            "\nrelation<|#|>Roe et al. (2001)<|#|>Imaginary Basin<|#|>studied<|#|>The study covers the basin."
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
        "drop_junk_entities": True,
    }
    chunks = {
        "chunk-001": {
            "tokens": 20,
            "content": "Roe et al. studied the basin.",
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }

    [(nodes, edges)] = await extract_entities(
        chunks=chunks, global_config=global_config
    )

    assert set(nodes) == {"Imaginary Basin"}
    assert not edges
