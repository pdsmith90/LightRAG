"""ENTITY_TYPE_STRICT: an extracted type the guidance does not list is stored as
``other``.

The extraction LLM invents types freely (hundreds of distinct values on a paper
corpus). With the switch on, ``extract_entities`` arms an allow-list parsed from
the active guidance's ``- Type: ...`` lines; the shared type validator maps
anything else to ``other``. Off, or with a guidance that lists no types, nothing
changes.
"""

import pytest

import lightrag.operate as operate
from lightrag.operate import (
    _configure_entity_type_allowlist,
    _normalize_and_validate_entity_type,
    entity_types_in_guidance,
)
from lightrag.prompt import PROMPTS

pytestmark = pytest.mark.offline

GUIDANCE = """Classify each entity using one of the following types. If no type fits, use `Other`.

  - Concept: physical phenomena, processes, theories (e.g. hydrostatic equilibrium)
  - Quantity: named parameters and observables (e.g. degree-1 terms)
  - Method: algorithms and processing techniques
  * Natural Object: minerals, celestial bodies
  - `Mission`: satellites and constellations

  Never extract people: authors, editors, cited researchers.
"""


@pytest.fixture(autouse=True)
def _disarm():
    yield
    operate._ENTITY_TYPE_ALLOWLIST = None


def test_guidance_type_lines_are_parsed_and_normalised():
    assert entity_types_in_guidance(GUIDANCE) == {
        "concept",
        "quantity",
        "method",
        "naturalobject",
        "mission",
    }
    assert "person" in entity_types_in_guidance(
        PROMPTS["default_entity_types_guidance"]
    )
    assert entity_types_in_guidance("") == frozenset()


def test_strict_mode_maps_unlisted_types_to_other():
    allow = _configure_entity_type_allowlist({"entity_type_strict": True}, GUIDANCE)
    assert "other" in allow
    assert _normalize_and_validate_entity_type("Method", "t") == "method"
    assert _normalize_and_validate_entity_type("Natural Object", "t") == "naturalobject"
    assert _normalize_and_validate_entity_type("Other", "t") == "other"
    assert _normalize_and_validate_entity_type("person", "t") == "other"
    assert _normalize_and_validate_entity_type("journal", "t") == "other"
    # Reserved names are still rejected outright, not mapped.
    assert _normalize_and_validate_entity_type("__proto__", "t") is None


def test_off_or_without_type_lines_leaves_types_as_written(lightrag_log_records):
    assert (
        _configure_entity_type_allowlist({"entity_type_strict": False}, GUIDANCE)
        is None
    )
    assert _normalize_and_validate_entity_type("journal", "t") == "journal"
    assert (
        _configure_entity_type_allowlist(
            {"entity_type_strict": True}, "Classify entities sensibly."
        )
        is None
    )
    assert _normalize_and_validate_entity_type("journal", "t") == "journal"
    assert any(
        "ENTITY_TYPE_STRICT ignored" in r.getMessage() for r in lightrag_log_records
    )
