"""Extraction contract (lightrag/kg_contract.py).

- A profile without ``relationship_types`` sends exactly the prompts it sent
  before the contract existed: the system, user and gleaning prompts are the
  untouched templates formatted with the profile.
- A profile with it gets the relation list in its guidance and the contract's
  wording in the JSON system prompt; text mode refuses it.
- ``python -m lightrag.kg_contract --schema`` prints the request schema of a
  profile.
- The validator drops notation names, renames or drops glued names, drops
  relations to ungrounded undeclared endpoints and the self-loops a rename
  creates, and keeps grounded undeclared endpoints.
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lightrag import kg_contract
from lightrag.kg_contract import (
    RELATIONSHIP_TYPES_V31,
    contract_system_prompt,
    glued_action,
    kg_extraction_validator,
    relationship_guidance,
    validate_chunk,
    variable_like,
)
from lightrag.operate import entity_types_in_guidance, extract_entities
from lightrag.prompt import (
    PROMPTS,
    get_default_entity_extraction_prompt_profile,
    load_entity_extraction_prompt_profile,
    resolve_entity_extraction_prompt_profile,
    validate_entity_extraction_prompt_profile_for_mode,
)
from lightrag.utils import Tokenizer, TokenizerInterface, sanitize_text_for_encoding

pytestmark = pytest.mark.offline

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "prompts" / "samples"
CONTRACT_SAMPLE = SAMPLES / "entity_type_prompt.kg_contract.sample.yml"
PLAIN_SAMPLE = SAMPLES / "entity_type_prompt.sample.yml"

CHUNK_TEXT = "The Glorbon mission measured the gravity field of Quux."


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """PROMPT_DIR holding both sample profiles."""
    (tmp_path / "entity_type").mkdir()
    shutil.copy(CONTRACT_SAMPLE, tmp_path / "entity_type" / "contract.yml")
    shutil.copy(PLAIN_SAMPLE, tmp_path / "entity_type" / "plain.yml")
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "9999999")
    return tmp_path


async def _sent_prompts(
    use_json: bool, addon_params: dict, cached_profile: bool = False
) -> list[tuple]:
    """(user prompt, system prompt) of every LLM call one chunk's extraction
    makes, with one gleaning pass. ``cached_profile`` hands extraction the
    profile the way ``LightRAG`` caches it (``_entity_extraction_prompt_profile``,
    resolved and validated once) instead of letting it resolve its own."""
    reply = '{"entities": [], "relationships": []}' if use_json else "<|COMPLETE|>"
    llm = AsyncMock(return_value=reply)
    global_config = {
        "llm_model_func": llm,
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 1,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": addon_params,
        "tokenizer": Tokenizer("dummy", _CharTokenizer()),
        "llm_model_max_async": 1,
        "entity_extraction_use_json": use_json,
    }
    if cached_profile:
        global_config["_entity_extraction_prompt_profile"] = (
            validate_entity_extraction_prompt_profile_for_mode(
                resolve_entity_extraction_prompt_profile(addon_params, use_json),
                use_json,
                addon_params.get("entity_type_prompt_file"),
            )
        )
    chunks = {
        "chunk-1": {
            "tokens": len(CHUNK_TEXT),
            "content": CHUNK_TEXT,
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
        }
    }
    await extract_entities(chunks=chunks, global_config=global_config)
    return [(c.args[0], c.kwargs.get("system_prompt")) for c in llm.call_args_list]


def _expected_prompts(use_json: bool, profile: dict) -> list[tuple]:
    """The prompts as the templates give them, formatted the way extraction
    formatted them before the contract existed."""
    context = dict(
        entity_types_guidance=profile["entity_types_guidance"],
        language="English",
        max_total_records=100,
        max_entity_records=40,
    )
    if use_json:
        context["examples"] = "\n".join(profile["entity_extraction_json_examples"])
        system = PROMPTS["entity_extraction_json_system_prompt"].format(**context)
        user = PROMPTS["entity_extraction_json_user_prompt"].format(
            **context, input_text=CHUNK_TEXT, heading_context_block=""
        )
        glean = PROMPTS["entity_continue_extraction_json_user_prompt"].format(**context)
    else:
        delimiters = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )
        context.update(delimiters)
        context["examples"] = "\n".join(profile["entity_extraction_examples"]).format(
            **delimiters,
            entity_types_guidance=context["entity_types_guidance"],
            language="English",
        )
        system = PROMPTS["entity_extraction_system_prompt"].format(**context)
        user = PROMPTS["entity_extraction_user_prompt"].format(
            **context, input_text=CHUNK_TEXT, heading_context_block=""
        )
        glean = PROMPTS["entity_continue_extraction_user_prompt"].format(
            **context, input_text=CHUNK_TEXT
        )
    s = sanitize_text_for_encoding
    return [(s(user), s(system)), (s(glean), s(system))]


# ------------------------------------------------------------ default unchanged


@pytest.mark.parametrize("cached_profile", [False, True], ids=["resolved", "cached"])
@pytest.mark.parametrize("use_json", [True, False])
@pytest.mark.parametrize("profile_file", [None, "plain.yml"])
async def test_a_profile_without_relationship_types_sends_the_untouched_prompts(
    prompt_dir, use_json, profile_file, cached_profile
):
    addon = {"entity_type_prompt_file": profile_file} if profile_file else {}
    profile = (
        load_entity_extraction_prompt_profile(PLAIN_SAMPLE)
        if profile_file
        else get_default_entity_extraction_prompt_profile()
    )
    assert "relationship_types" not in profile
    sent = await _sent_prompts(use_json, addon, cached_profile)
    assert sent == _expected_prompts(use_json, profile)
    assert "Relationship types." not in sent[0][1]


def test_a_profile_without_the_key_resolves_as_before(prompt_dir):
    resolved = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "plain.yml"}, True
    )
    assert set(resolved) == {
        "entity_types_guidance",
        "entity_extraction_examples",
        "entity_extraction_json_examples",
    }
    assert (
        resolved["entity_types_guidance"]
        == load_entity_extraction_prompt_profile(PLAIN_SAMPLE)["entity_types_guidance"]
    )
    validated = validate_entity_extraction_prompt_profile_for_mode(resolved, True)
    assert "relationship_types" not in validated


# ------------------------------------------------------------ contract on


def test_the_sample_profile_declares_the_v31_list():
    profile = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    assert profile["relationship_types"] == RELATIONSHIP_TYPES_V31
    assert list(profile["relationship_types"]) == list(RELATIONSHIP_TYPES_V31)
    assert "Relationship types." not in profile["entity_types_guidance"]


def test_the_relation_list_is_appended_to_the_guidance(prompt_dir):
    resolved = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "contract.yml"}, True
    )
    plain = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    guidance = resolved["entity_types_guidance"]
    assert guidance == (
        plain["entity_types_guidance"]
        + "\n\n"
        + relationship_guidance(RELATIONSHIP_TYPES_V31)
    )
    assert "  part_of = SOURCE is a component" in guidance
    assert guidance.endswith("a forced type is worse than related_to.")
    # the relation lines are never read as entity types (ENTITY_TYPE_STRICT)
    assert entity_types_in_guidance(guidance) == entity_types_in_guidance(
        plain["entity_types_guidance"]
    )
    assert resolved["relationship_types"] == RELATIONSHIP_TYPES_V31
    validated = validate_entity_extraction_prompt_profile_for_mode(resolved, True)
    assert validated["relationship_types"] == RELATIONSHIP_TYPES_V31


def test_the_direction_check_names_related_to_only_when_it_is_listed():
    text = relationship_guidance({"is_a": "SOURCE is a kind of TARGET"})
    assert "  is_a = SOURCE is a kind of TARGET" in text
    assert "related_to" not in text


async def test_the_contract_profile_sends_the_contract_wording(prompt_dir):
    sent = await _sent_prompts(True, {"entity_type_prompt_file": "contract.yml"})
    resolved = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "contract.yml"}, True
    )
    (user, system), (glean, glean_system) = sent
    context = dict(
        entity_types_guidance=resolved["entity_types_guidance"],
        examples="\n".join(resolved["entity_extraction_json_examples"]),
        language="English",
        max_total_records=100,
        max_entity_records=40,
    )
    expected_system = contract_system_prompt(
        PROMPTS["entity_extraction_json_system_prompt"]
    ).format(**context)
    assert system == glean_system == sanitize_text_for_encoding(expected_system)
    assert "title case" not in system
    assert "Relationships are **directed**" in system
    assert "exactly one relationship type from the `Relationship types` list" in system
    assert "List an entity only when the text states at least one relationship" in (
        system
    )
    assert "undirected" not in system
    # the user prompt repeats the guidance, relation list included
    assert "  same_as = SOURCE and TARGET name the same thing" in user
    # the gleaning prompt carries no rule the contract replaces
    assert glean == sanitize_text_for_encoding(
        PROMPTS["entity_continue_extraction_json_user_prompt"].format(**context)
    )


def test_text_mode_refuses_a_contract_profile(prompt_dir):
    (prompt_dir / "entity_type" / "text.yml").write_text(
        "entity_types_guidance: '- Thing: x'\n"
        "entity_extraction_examples: ['entity{tuple_delimiter}<entity_name>']\n"
        "relationship_types:\n  part_of: SOURCE is in TARGET\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="needs JSON entity extraction"):
        resolve_entity_extraction_prompt_profile(
            {"entity_type_prompt_file": "text.yml"}, False
        )


def test_a_changed_base_template_fails_instead_of_half_editing():
    template = PROMPTS["entity_extraction_json_system_prompt"].replace(
        "**undirected**", "undirected"
    )
    with pytest.raises(ValueError, match="system prompt changed"):
        contract_system_prompt(template)


@pytest.mark.parametrize(
    "value,message",
    [
        ("is_a", "non-empty mapping"),
        ({}, "non-empty mapping"),
        ({"Is A": "x"}, "lower-case snake case"),
        ({"- part_of": "x"}, "lower-case snake case"),
        ({"is_a": ""}, "non-empty definition"),
        ({"is_a": ["x"]}, "non-empty definition"),
    ],
)
def test_the_loader_rejects_a_malformed_relation_list(tmp_path, value, message):
    path = tmp_path / "profile.yml"
    path.write_text(
        json.dumps(
            {"entity_types_guidance": "- Thing: x", "relationship_types": value}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_entity_extraction_prompt_profile(path)


def test_the_loader_collapses_whitespace_in_definitions(tmp_path):
    path = tmp_path / "profile.yml"
    path.write_text(
        "relationship_types:\n  is_a: |\n    SOURCE is a kind\n    of TARGET\n",
        encoding="utf-8",
    )
    profile = load_entity_extraction_prompt_profile(path)
    assert profile["relationship_types"] == {"is_a": "SOURCE is a kind of TARGET"}


# ------------------------------------------------------------ schema command

EXPECTED_V31_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "Concept",
                            "Quantity",
                            "Method",
                            "Mission",
                            "Instrument",
                            "Dataset",
                            "Software",
                            "Organization",
                            "Location",
                            "Event",
                            "Other",
                        ],
                    },
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "keywords": {
                        "type": "string",
                        "enum": [
                            "is_a",
                            "part_of",
                            "same_as",
                            "measures",
                            "estimates",
                            "uses",
                            "applied_to",
                            "produces",
                            "derived_from",
                            "operates",
                            "located_in",
                            "affects",
                            "corrects",
                            "compared_with",
                            "depends_on",
                            "has_property",
                            "related_to",
                        ],
                    },
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "keywords", "description"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


def test_the_schema_command_prints_the_v31_schema(capsys):
    assert kg_contract.main(["--schema", "--profile", str(CONTRACT_SAMPLE)]) == 0
    assert json.loads(capsys.readouterr().out) == EXPECTED_V31_SCHEMA


def test_the_schema_command_runs_as_a_module():
    out = subprocess.run(
        [sys.executable, "-m", "lightrag.kg_contract", "--schema", "--profile"]
        + [str(CONTRACT_SAMPLE)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=True,
    )
    assert json.loads(out.stdout) == EXPECTED_V31_SCHEMA


def test_the_schema_command_adds_other_once(tmp_path, capsys):
    path = tmp_path / "profile.yml"
    path.write_text(
        "entity_types_guidance: |\n  - Thing: a\n  - Natural Object: b\n  - other: c\n"
        "relationship_types:\n  part_of: SOURCE is in TARGET\n",
        encoding="utf-8",
    )
    assert kg_contract.main(["--schema", "--profile", str(path)]) == 0
    schema = json.loads(capsys.readouterr().out)
    entity_enum = schema["properties"]["entities"]["items"]["properties"]["type"]
    assert entity_enum["enum"] == ["Thing", "Natural Object", "other"]


def test_the_schema_command_needs_a_relation_list(capsys):
    assert kg_contract.main(["--schema", "--profile", str(PLAIN_SAMPLE)]) == 2
    assert "declares no relationship_types" in capsys.readouterr().err


def test_the_schema_command_reports_a_missing_profile_file(tmp_path, capsys):
    missing = tmp_path / "absent.yml"
    assert kg_contract.main(["--schema", "--profile", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does not exist" in captured.err and str(missing) in captured.err


def test_the_schema_command_takes_the_entity_type_prompt_file_value(
    prompt_dir, capsys, monkeypatch, tmp_path
):
    """A bare file name is looked up in PROMPT_DIR/entity_type, as the server
    looks up ENTITY_TYPE_PROMPT_FILE, from any current directory; stderr says
    which file was used and stdout stays the schema."""
    monkeypatch.chdir(tmp_path)
    assert kg_contract.main(["--schema", "--profile", "contract.yml"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == EXPECTED_V31_SCHEMA
    used = prompt_dir / "entity_type" / "contract.yml"
    assert f"profile: {used} (ENTITY_TYPE_PROMPT_FILE" in captured.err


def test_the_schema_command_takes_a_path_and_says_so(capsys):
    assert kg_contract.main(["--schema", "--profile", str(CONTRACT_SAMPLE)]) == 0
    assert f"profile: {CONTRACT_SAMPLE} (path)" in capsys.readouterr().err


def test_the_schema_command_names_both_places_for_a_missing_name(
    prompt_dir, capsys, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    assert kg_contract.main(["--schema", "--profile", "absent.yml"]) == 2
    err = capsys.readouterr().err
    assert str(prompt_dir / "entity_type" / "absent.yml") in err
    assert "nor in the current directory" in err


def test_the_schema_command_needs_the_schema_flag():
    with pytest.raises(SystemExit) as exc:
        kg_contract.main(["--profile", str(CONTRACT_SAMPLE)])
    assert exc.value.code == 2


# ------------------------------------------------------------ validator


@pytest.mark.parametrize(
    "name", ["x_i", "a_n", "T_2n", "q_n(x)", "f(x)", "p0(x)", "x1*", "xy′", "x^2"]
)
def test_notation_is_variable_like(name):
    assert variable_like(name)


@pytest.mark.parametrize(
    "name",
    ["L1", "J2", "k2", "C20", "F10.7", "P(Y)", "GRACE-FO", "scale height", "x"],
)
def test_real_names_are_not_variable_like(name):
    assert not variable_like(name)


@pytest.mark.parametrize(
    "name,text,expected",
    [
        ("MetOp", "MetOp carries the sensor.", ("keep", None)),
        ("ScaleHeight", "the scale height grows", ("rename", "scale height")),
        ("scale_height", "the Scale-Height grows", ("rename", "Scale-Height")),
        ("NormalDistribution", "nothing here", ("rename", "normal distribution")),
        ("PhiTau", "nothing here", ("drop", None)),
        ("p_y_2", "nothing here", ("drop", None)),
        ("DoeEtAl", "nothing here", ("drop", None)),
        ("Quux Basin", "nothing here", ("keep", None)),
    ],
)
def test_glued_names(name, text, expected):
    assert glued_action(name, text) == expected


def _node(name, etype="concept"):
    return [{"entity_name": name, "entity_type": etype, "description": f"{name}."}]


def _edge(src, tgt):
    return [{"src_id": src, "tgt_id": tgt, "keywords": "uses", "description": "x"}]


TEXT = (
    "The scale height of the thermosphere sets the drag on Glorbon. The state "
    "vector x_k is estimated, f(x) is minimized, and terrestrial water storage "
    "changes over Quux."
)


def _run(nodes, edges):
    return validate_chunk(TEXT, nodes, edges)


def test_notation_names_and_their_relations_are_dropped():
    nodes = {"x_k": _node("x_k"), "Glorbon": _node("Glorbon", "mission")}
    edges = {("x_k", "Glorbon"): _edge("x_k", "Glorbon")}
    out_nodes, out_edges, stats = _run(nodes, edges)
    assert set(out_nodes) == {"Glorbon"}
    assert out_edges == {}
    assert stats["variable"] == 1 and stats["endpoint_removed"] == 1


def test_a_glued_name_is_renamed_to_the_text_spelling_with_its_relations():
    nodes = {"ScaleHeight": _node("ScaleHeight"), "thermosphere": _node("thermosphere")}
    edges = {("ScaleHeight", "thermosphere"): _edge("ScaleHeight", "thermosphere")}
    out_nodes, out_edges, stats = _run(nodes, edges)
    assert set(out_nodes) == {"scale height", "thermosphere"}
    assert out_nodes["scale height"][0]["entity_name"] == "scale height"
    [record] = out_edges[("scale height", "thermosphere")]
    assert (record["src_id"], record["tgt_id"]) == ("scale height", "thermosphere")
    assert stats["glued_renamed"] == 1


def test_a_glued_notation_name_is_dropped_with_its_relations():
    nodes = {"PhiTau": _node("PhiTau"), "Glorbon": _node("Glorbon")}
    edges = {("Glorbon", "PhiTau"): _edge("Glorbon", "PhiTau")}
    out_nodes, out_edges, stats = _run(nodes, edges)
    assert set(out_nodes) == {"Glorbon"} and out_edges == {}
    assert stats["glued_dropped"] == 1 and stats["endpoint_removed"] == 1


def test_an_ungrounded_undeclared_endpoint_drops_its_relation():
    nodes = {"Glorbon": _node("Glorbon")}
    edges = {("Glorbon", "Zorp Probe"): _edge("Glorbon", "Zorp Probe")}
    _n, out_edges, stats = _run(nodes, edges)
    assert out_edges == {} and stats["undeclared_dropped"] == 1


def test_a_notation_undeclared_endpoint_drops_its_relation_even_in_the_text():
    nodes = {"Glorbon": _node("Glorbon")}
    edges = {("Glorbon", "f(x)"): _edge("Glorbon", "f(x)")}
    _n, out_edges, stats = _run(nodes, edges)
    assert out_edges == {} and stats["undeclared_dropped"] == 1


def test_a_grounded_undeclared_endpoint_is_kept():
    nodes = {"Glorbon": _node("Glorbon")}
    edges = {
        ("Glorbon", "Terrestrial Water Storage"): _edge(
            "Glorbon", "Terrestrial Water Storage"
        )
    }
    out_nodes, out_edges, stats = _run(nodes, edges)
    assert set(out_edges) == {("Glorbon", "Terrestrial Water Storage")}
    assert set(out_nodes) == {"Glorbon"}  # the merge creates the endpoint
    assert stats["undeclared_kept"] == 1


def test_a_rename_that_joins_both_endpoints_drops_the_self_loop():
    nodes = {"ScaleHeight": _node("ScaleHeight"), "scale height": _node("scale height")}
    edges = {("ScaleHeight", "scale height"): _edge("ScaleHeight", "scale height")}
    out_nodes, out_edges, stats = _run(nodes, edges)
    assert list(out_nodes) == ["scale height"]
    assert len(out_nodes["scale height"]) == 2  # both records, one name
    assert out_edges == {} and stats["self_loop"] == 1


def test_grounding_reads_the_chunk_text_as_extraction_normalizes_names():
    """Extraction removes the curly apostrophe from a name (Earth's -> Earths);
    the endpoint is still found in the chunk text, normalized the same way."""
    text = "Glorbon maps Earth\u2019s gravity field."
    nodes = {"Glorbon": _node("Glorbon")}
    edges = {
        ("Glorbon", "Earths gravity field"): _edge("Glorbon", "Earths gravity field")
    }
    _n, out_edges, stats = validate_chunk(text, nodes, edges)
    assert set(out_edges) == {("Glorbon", "Earths gravity field")}
    assert stats["undeclared_kept"] == 1 and stats["undeclared_dropped"] == 0


def test_relation_direction_is_kept():
    nodes = {"Glorbon": _node("Glorbon"), "Quux": _node("Quux", "location")}
    edges = {("Quux", "Glorbon"): _edge("Quux", "Glorbon")}
    _n, out_edges, _s = _run(nodes, edges)
    assert set(out_edges) == {("Quux", "Glorbon")}


def test_the_validator_leaves_its_inputs_alone_and_returns_a_pair():
    nodes = {"ScaleHeight": _node("ScaleHeight"), "x_k": _node("x_k")}
    edges = {("ScaleHeight", "x_k"): _edge("ScaleHeight", "x_k")}
    before = copy.deepcopy((nodes, edges))
    result = kg_extraction_validator("chunk-1", TEXT, nodes, edges)
    assert isinstance(result, tuple) and len(result) == 2
    assert all(isinstance(part, dict) for part in result)
    assert set(result[0]) == {"scale height"} and result[1] == {}
    assert (nodes, edges) == before


async def test_the_validator_runs_as_the_extraction_hook(monkeypatch):
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "9999999")
    reply = json.dumps(
        {
            "entities": [
                {"name": "ScaleHeight", "type": "Quantity", "description": "A height."},
                {"name": "thermosphere", "type": "Location", "description": "A layer."},
            ],
            "relationships": [
                {
                    "source": "ScaleHeight",
                    "target": "thermosphere",
                    "keywords": "has_property",
                    "description": "The layer has a scale height.",
                }
            ],
        }
    )
    llm = AsyncMock(return_value=reply)
    content = "The scale height of the thermosphere grows with temperature."
    global_config = {
        "llm_model_func": llm,
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _CharTokenizer()),
        "llm_model_max_async": 1,
        "entity_extraction_use_json": True,
        "kg_extraction_validator": kg_extraction_validator,
    }
    chunks = {
        "chunk-1": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
        }
    }
    [(nodes, edges)] = await extract_entities(
        chunks=chunks, global_config=global_config
    )
    assert set(nodes) == {"scale height", "thermosphere"}
    assert set(edges) == {("scale height", "thermosphere")}
