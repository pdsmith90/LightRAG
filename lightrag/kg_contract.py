"""Fork (extraction contract): a closed list of relation types, the prompt wording
that goes with it, the JSON schema for the request layer, and an optional
validator for the extraction hook.

The contract is switched on by the entity-type profile (``ENTITY_TYPE_PROMPT_FILE``):
a profile that declares ``relationship_types`` -- a mapping of relation type name
to its definition, in prompt order -- gets

- the relation list appended to its entity-type guidance, as ``name = definition``
  lines (never ``- name:`` bullets, which ENTITY_TYPE_STRICT would read as entity
  types), with a read-it-back direction check;
- the JSON extraction system prompt with ``SYSTEM_PROMPT_EDITS`` applied: the
  naming rule defers to the profile instead of asking for title case, ``keywords``
  takes exactly one type from the list, relations are directed, and an entity is
  listed only when a relation for it is stated.

A profile without the key leaves every prompt exactly as it was. The contract
needs JSON extraction (``ENTITY_EXTRACTION_USE_JSON=true``); the text-mode
prompts are not edited. The gleaning turn is sent with the contract system
prompt and the unchanged gleaning template.

Direction in the graph. The graph stores each pair of entities once, in sorted
order (``src < tgt`` by code point, Python's ``sorted``): the extraction merge
and the rebuild after a deletion write each pair that way, and the plain-table
PostgreSQL storage enforces it, so "GRACE part_of Constellation" is stored as
(Constellation, GRACE) with keywords ``part_of``. Without more,
the stored ``keywords`` no longer say which node is the part; the direction
survives only in the description text. A profile that also declares
``relationship_inverses`` -- a mapping of a directed type name to its inverse
name; a type the mapping does not name is symmetric -- keeps it: every write
path puts the pair into sorted order and, when that flips an extracted
relation "A t B", stores the inverse name of ``t`` instead (``orient_relation``),
so a stored edge (src, tgt) with a declared type or inverse name k reads
"src k tgt". That holds for every keyword of an LLM-extracted relation, which
the parser checks against the list; keywords given by a caller or made by the
code (sidecar links) that are not on it are stored as given. The LLM never sees
the inverse names: the prompt, the guidance and the schema keep the forward
list. See ``ZLIGHTRAG.md`` for the write paths and their limits.

The schema the extraction requests are constrained by is built by a front
process, not by this code, so ``python -m lightrag.kg_contract --schema --profile
PATH`` prints it for a profile: the entity types its guidance lists (read the
way ENTITY_TYPE_STRICT reads them) plus ``Other``, and the profile's relation
types, both as enums. Nothing on
this side can make the front apply it, so the parser checks every extracted
relation's keywords against the list (``check_relation_keywords``) and counts
what it had to change.

``kg_extraction_validator`` (wired by ``KG_CONTRACT_VALIDATOR`` in the API
server) applies the checks the parser does not make: it drops entity names that
are mathematical notation of three or more characters, renames or drops
identifiers glued from several words, drops a relation whose endpoint is
neither a declared entity nor present in the chunk text, and drops self-loops
that a rename creates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightrag.utils import logger, normalize_entity_name

# The relation list v3.1: each definition says what SOURCE and TARGET must be, and
# `related_to` is the right answer when nothing else fits. The sample profile
# prompts/samples/entity_type_prompt.kg_contract.sample.yml declares exactly this.
RELATIONSHIP_TYPES_V31: dict[str, str] = {
    "is_a": "SOURCE is a kind or an instance of TARGET (the specific thing is the source)",
    "part_of": "SOURCE is a component, member or subgroup of TARGET, or is contained in it (the smaller thing is the source)",
    "same_as": "SOURCE and TARGET name the same thing: an acronym and its expansion, a synonym",
    "measures": "SOURCE is an instrument, mission, network or survey that observes or records TARGET",
    "estimates": "SOURCE is a method, model or analysis that computes, retrieves or predicts TARGET, a quantity or field",
    "uses": "SOURCE is a method, model, study, system or organization that takes TARGET as input, data, tool or assumption",
    "applied_to": "SOURCE is a method, model or analysis and TARGET is the region, object, problem or data it is applied to",
    "produces": "SOURCE generates TARGET as its output: a data product, a result, a signal",
    "derived_from": "SOURCE is computed, adapted, extended or generalized from TARGET (the newer or derived thing is the source)",
    "operates": "SOURCE is an organization that builds, runs, maintains or distributes TARGET",
    "located_in": "SOURCE lies in, occurs in or is observed at TARGET, a region, layer, body or site",
    "affects": "SOURCE causes, drives, changes, biases or limits TARGET",
    "corrects": "SOURCE is a method, model or filter that removes or reduces TARGET, an error or unwanted signal",
    "compared_with": "SOURCE is compared with, validated against or benchmarked against TARGET",
    "depends_on": "SOURCE is a quantity, result or property whose value is determined by TARGET",
    "has_property": "TARGET is a quantity, parameter or metric that describes or evaluates SOURCE",
    "related_to": "the text links the two and no type above fits exactly",
}

# A relation type name is lower-case snake case: it is an enum value in the
# request schema, and it must never look like a `- Type:` line of the guidance.
RELATIONSHIP_TYPE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def normalize_relationship_types(value: Any, source: str) -> dict[str, str]:
    """Validate a profile's ``relationship_types`` and return it as a plain dict.

    ``source`` names the profile in the error message."""
    field = f"{source} field 'relationship_types'"
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"{field} must be a non-empty mapping of relation type name to definition."
        )
    types: dict[str, str] = {}
    for name, definition in value.items():
        if not isinstance(name, str) or not RELATIONSHIP_TYPE_NAME_RE.match(name):
            raise ValueError(
                f"{field}: relation type name {name!r} must be lower-case snake case "
                "(letters, digits and underscores, starting with a letter)."
            )
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(
                f"{field}: relation type {name!r} needs a non-empty definition."
            )
        types[name] = " ".join(definition.split())
    return types


# The inverse names of the v3.1 list; the sample profile declares exactly this.
# same_as, compared_with and related_to are symmetric and have none.
RELATIONSHIP_INVERSES_V31: dict[str, str] = {
    "is_a": "has_kind",
    "part_of": "has_part",
    "measures": "measured_by",
    "estimates": "estimated_by",
    "uses": "used_by",
    "applied_to": "target_of",
    "produces": "produced_by",
    "derived_from": "source_of",
    "operates": "operated_by",
    "located_in": "location_of",
    "affects": "affected_by",
    "corrects": "corrected_by",
    "depends_on": "determines",
    "has_property": "property_of",
}


def normalize_relationship_inverses(
    value: Any, types: Mapping[str, str] | None, source: str
) -> dict[str, str]:
    """Validate a profile's ``relationship_inverses`` against its
    ``relationship_types`` and return it as a plain dict.

    Each key is a declared type, each value its inverse name: lower-case snake
    case, not a declared type, and used once. ``source`` names the profile in
    the error message."""
    field = f"{source} field 'relationship_inverses'"
    if not types:
        raise ValueError(f"{field} needs 'relationship_types' in the same profile.")
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            f"{field} must be a non-empty mapping of relation type name to its "
            "inverse name."
        )
    inverses: dict[str, str] = {}
    for name, inverse in value.items():
        if name not in types:
            raise ValueError(
                f"{field}: {name!r} is not a type in 'relationship_types'."
            )
        if not isinstance(inverse, str) or not RELATIONSHIP_TYPE_NAME_RE.match(inverse):
            raise ValueError(
                f"{field}: the inverse of {name!r}, {inverse!r}, must be lower-case "
                "snake case (letters, digits and underscores, starting with a letter)."
            )
        if inverse in types:
            raise ValueError(
                f"{field}: the inverse of {name!r}, {inverse!r}, is also a type in "
                "'relationship_types'."
            )
        if inverse in inverses.values():
            raise ValueError(
                f"{field}: {inverse!r} is the inverse of more than one type."
            )
        inverses[name] = inverse
    return inverses


def inverse_table(inverses: Mapping[str, str] | None) -> dict[str, str] | None:
    """Both directions of a ``relationship_inverses`` mapping (``part_of`` <->
    ``has_part``), or None when there is none: a stored keyword may already be
    an inverse name that a later re-keying flips back."""
    if not inverses:
        return None
    table = dict(inverses)
    table.update((inverse, name) for name, inverse in inverses.items())
    return table


_KEYWORD_TOKEN = re.compile(r"[^,]+")


def invert_keywords(keywords: str, table: Mapping[str, str]) -> str:
    """``keywords`` read the other way round: each comma-separated keyword that
    is a directed type or an inverse name in ``table`` becomes its counterpart;
    symmetric types, unknown keywords, separators and spacing stay as they are."""

    def swap(match: re.Match) -> str:
        token = match.group(0)
        name = token.strip()
        if name not in table:
            return token
        start = token.index(name)
        return token[:start] + table[name] + token[start + len(name) :]

    return _KEYWORD_TOKEN.sub(swap, keywords)


def orient_relation(
    src: str, tgt: str, keywords: str, table: Mapping[str, str] | None
) -> tuple[str, str, str]:
    """A relation whose ``keywords`` read "src keywords tgt", in the graph's
    storage order: ``(min, max, keywords)``, the keywords inverted when the pair
    was flipped. Without a table the pair is sorted and the keywords are
    returned unchanged (the behaviour without ``relationship_inverses``)."""
    if src <= tgt:
        return src, tgt, keywords
    if table and keywords:
        keywords = invert_keywords(keywords, table)
    return tgt, src, keywords


def orient_edge_records(
    edge_key: tuple[str, str], records: list[dict], table: Mapping[str, str] | None
) -> list[dict]:
    """The parsed relation records of ``edge_key`` (as extracted: source,
    target) for the sorted key: when sorting flips the pair, copies with
    ``src_id``/``tgt_id`` swapped and ``keywords`` inverted. The records are
    returned as they are when there is no table or no flip."""
    src, tgt = edge_key
    if not table or src <= tgt:
        return records
    flipped = []
    for record in records:
        record = {**record, "src_id": tgt, "tgt_id": src}
        if record.get("keywords"):
            record["keywords"] = invert_keywords(str(record["keywords"]), table)
        flipped.append(record)
    return flipped


def _active_profile(global_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The profile ``global_config`` carries (the one ``LightRAG`` caches as
    ``_entity_extraction_prompt_profile``). A config without a cached profile
    (tests, custom wiring) resolves it the way ``extract_entities`` does; one
    without a profile file has no contract keys."""
    profile = global_config.get("_entity_extraction_prompt_profile")
    if profile is None:
        addon_params = global_config.get("addon_params") or {}
        if not addon_params.get("entity_type_prompt_file"):
            return {}
        from lightrag.prompt import resolve_entity_extraction_prompt_profile

        profile = resolve_entity_extraction_prompt_profile(
            addon_params, global_config.get("entity_extraction_use_json", False)
        )
    return profile


def relation_inverse_table(global_config: Mapping[str, Any]) -> dict[str, str] | None:
    """The inverse table of the profile ``global_config`` carries, or None when
    it declares no ``relationship_inverses``."""
    return inverse_table(_active_profile(global_config).get("relationship_inverses"))


def relation_types(global_config: Mapping[str, Any]) -> dict[str, str] | None:
    """The ``relationship_types`` of the profile ``global_config`` carries, or
    None when it declares none (the contract is off)."""
    return _active_profile(global_config).get("relationship_types") or None


# ---- the relation list at parse time ----------------------------------------------

# What the keyword check counts: every relation it checks, and each change it
# makes. A keyword that is exactly one declared type is not counted as a change.
RELATION_KEYWORD_COUNTS = (
    "checked",
    "normalized",
    "multi_type",
    "related_to",
    "dropped",
)


def check_relation_keywords(
    keywords: str, types: Mapping[str, str]
) -> tuple[str | None, str | None]:
    """One LLM-extracted relation's ``keywords`` checked against the declared
    relation list: ``(keywords to store, or None to drop the relation; the
    change made, or None when the field is exactly one declared type)``.

    A comma-separated keyword matches a declared type when it equals the type
    after trimming and case-folding (``Part_Of`` is ``part_of``; ``part of`` and
    ``member of`` match nothing). A field of one keyword that matches is
    rewritten to the declared spelling (``normalized``). A field of several
    keywords keeps the declared types among them, in their order, and loses the
    rest (``multi_type``); several spellings of one type are ``normalized``. A
    field with no declared type becomes ``related_to`` when the list declares
    it, and is otherwise dropped with its relation (``dropped``)."""
    tokens = [token.strip() for token in keywords.split(",") if token.strip()]
    declared_by_fold = {name.casefold(): name for name in types}
    declared = list(
        dict.fromkeys(
            declared_by_fold[token.casefold()]
            for token in tokens
            if token.casefold() in declared_by_fold
        )
    )
    if not declared:
        if "related_to" in types:
            return "related_to", "related_to"
        return None, "dropped"
    if len(declared) > 1 or any(t.casefold() not in declared_by_fold for t in tokens):
        return ",".join(declared), "multi_type"
    if keywords == declared[0]:
        return keywords, None
    return declared[0], "normalized"


def check_chunk_relation_keywords(
    maybe_edges: dict, types: Mapping[str, str], counts: Counter | None = None
) -> dict:
    """``check_relation_keywords`` applied to every record of one parsed
    extraction result (``(source, target) -> [edge records]``). Returns a new
    dict without the dropped records (and without a pair left with none); the
    records are copied when their keywords change. ``counts`` gains one
    ``checked`` per record and one per change (``RELATION_KEYWORD_COUNTS``)."""
    checked: dict = {}
    for edge_key, records in maybe_edges.items():
        kept = []
        for record in records:
            keywords, change = check_relation_keywords(
                str(record.get("keywords") or ""), types
            )
            if counts is not None:
                counts["checked"] += 1
                if change:
                    counts[change] += 1
            if keywords is None:
                continue
            if keywords != record.get("keywords"):
                record = {**record, "keywords": keywords}
            kept.append(record)
        if kept:
            checked[edge_key] = kept
    return checked


# Running totals of this process for /health, by where the check ran: new LLM
# output ("extraction") or cached LLM output re-parsed by a knowledge rebuild
# ("rebuild"), which re-counts records an extraction already counted.
_RELATION_KEYWORD_TOTALS: dict[str, Counter] = {
    "extraction": Counter(),
    "rebuild": Counter(),
}


def record_relation_keyword_counts(stage: str, counts: Mapping[str, int]) -> None:
    """Add one document's (or one rebuild's) keyword-check counts to the
    running totals of ``stage`` (``extraction`` or ``rebuild``)."""
    _RELATION_KEYWORD_TOTALS[stage].update(counts)


def relation_keyword_totals() -> dict[str, dict[str, int]]:
    """For ``/health``: the keyword-check counts of this process so far."""
    return {
        stage: {name: totals.get(name, 0) for name in RELATION_KEYWORD_COUNTS}
        for stage, totals in _RELATION_KEYWORD_TOTALS.items()
    }


def relation_keyword_changes(counts: Mapping[str, int]) -> str | None:
    """A log line's worth of the changes in ``counts``, or None when the check
    changed nothing."""
    changes = {name: counts.get(name, 0) for name in RELATION_KEYWORD_COUNTS[1:]}
    if not any(changes.values()):
        return None
    detail = ", ".join(f"{name}={n}" for name, n in changes.items())
    return (
        f"{sum(changes.values())} of {counts.get('checked', 0)} relation keywords "
        f"were not exactly one declared relationship type ({detail})"
    )


def profile_status(profile: Mapping[str, Any] | None) -> dict[str, int]:
    """For ``/health``: how many relation types and inverse names the active
    entity-type profile declares (0 and 0: the contract is off)."""
    profile = profile or {}
    return {
        "relationship_types": len(profile.get("relationship_types") or ()),
        "relationship_inverses": len(profile.get("relationship_inverses") or ()),
    }


def relationship_guidance(types: Mapping[str, str]) -> str:
    """The relation list for the entity-type guidance. Lines are ``name =
    definition``, never ``- name: ...``: ENTITY_TYPE_STRICT reads ``- Name:``
    bullets as entity types."""
    lines = [
        "Relationship types. The `keywords` field of a relationship takes exactly one of:"
    ]
    lines += [f"  {name} = {text}" for name, text in types.items()]
    check = (
        "Check each relationship by reading it back as SOURCE, type, TARGET. If that "
        "sentence is false or backwards, swap the two names or choose another type."
    )
    if "related_to" in types:
        check += (
            " If no type fits exactly, use related_to: a forced type is worse than "
            "related_to."
        )
    lines.append(check)
    return "\n".join(lines)


def guidance_with_relationship_types(guidance: str, types: Mapping[str, str]) -> str:
    """The entity-type guidance with the relation list appended."""
    return f"{guidance.rstrip()}\n\n{relationship_guidance(types)}"


# (old, new, expected count) on PROMPTS["entity_extraction_json_system_prompt"].
# A count that does not match means the base template changed: fail instead of
# sending a half-edited prompt.
SYSTEM_PROMPT_EDITS: tuple[tuple[str, str, int], ...] = (
    (
        "If the entity name is case-insensitive, capitalize the first letter of each "
        "significant word (title case). Ensure **consistent naming** across the entire "
        "extraction process.",
        "Follow the naming rule in the `---Entity Types---` section. Use the spelling "
        "the input text uses and never glue several words into one identifier. Ensure "
        "**consistent naming** across the entire extraction process.",
        1,
    ),
    (
        " Capitalize the first letter of each significant word (title case) if the "
        "name is case-insensitive.",
        "",
        2,
    ),
    (
        "`keywords`: One or more high-level keywords summarizing the overarching "
        "nature, concepts, or themes of the relationship, separated by commas.",
        "`keywords`: exactly one relationship type from the `Relationship types` list "
        "in the `---Entity Types---` section.",
        1,
    ),
    (
        "Output fewer records if fewer high-value items are present. Do not try to "
        "fill the limit.",
        "Output fewer records if fewer high-value items are present. Do not try to "
        "fill the limit. List an entity only when the text states at least one "
        "relationship for it; never invent a relationship to keep an entity.",
        1,
    ),
    (
        "Treat all relationships as **undirected** unless explicitly stated otherwise. "
        "Swapping the source and target entities for an undirected relationship does "
        "not constitute a new relationship.",
        "Relationships are **directed**: order `source` and `target` as the "
        "relationship type's definition does (the part is the source of `part_of`). "
        "State each pair once.",
        1,
    ),
)


def contract_system_prompt(base_template: str) -> str:
    """The JSON extraction system prompt template with ``SYSTEM_PROMPT_EDITS``
    applied. Raises ValueError when the base template no longer holds a passage
    the expected number of times."""
    out = base_template
    for old, new, count in SYSTEM_PROMPT_EDITS:
        found = out.count(old)
        if found != count:
            raise ValueError(
                "extraction contract: the JSON extraction system prompt changed; "
                f"expected {count} of {old[:60]!r}, found {found}"
            )
        out = out.replace(old, new)
    return out


# ---- request schema ---------------------------------------------------------------


def extraction_schema(
    entity_types: list[str], relationship_types: list[str]
) -> dict[str, Any]:
    """The JSON schema of one extraction response with closed type lists."""
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": list(entity_types)},
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
                            "enum": list(relationship_types),
                        },
                        "description": {"type": "string"},
                    },
                    "required": ["source", "target", "keywords", "description"],
                },
            },
        },
        "required": ["entities", "relationships"],
    }


def schema_for_profile(profile_path: str | Path) -> dict[str, Any]:
    """The extraction schema for an entity-type profile file: the types its
    guidance lists (as ENTITY_TYPE_STRICT reads them, in order) plus ``Other``,
    and its ``relationship_types``."""
    from lightrag.operate import entity_type_names_in_guidance
    from lightrag.prompt import PROMPTS, load_entity_extraction_prompt_profile

    profile = load_entity_extraction_prompt_profile(profile_path)
    types = profile.get("relationship_types")
    if not types:
        raise ValueError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' declares no relationship_types."
        )
    guidance = profile.get(
        "entity_types_guidance", PROMPTS["default_entity_types_guidance"]
    )
    entity_types = entity_type_names_in_guidance(guidance)
    if not any(t.replace(" ", "").lower() == "other" for t in entity_types):
        entity_types.append("Other")
    return extraction_schema(entity_types, list(types))


# ---- name checks the parser does not make -----------------------------------------

_TEX = re.compile(r"[\\^{}=]")
# x_i, f_k, a_n, T_2n, q_n(x): one to three letters, a subscript, an optional argument
_SUBSCRIPT = re.compile(
    r"^[A-Za-zα-ωΑ-Ω]{1,3}_\{?[A-Za-z0-9]{1,8}\}?(\([a-z0-9, ]{1,8}\))?$"
)
# f(x), p0(x), x(v), g,0(t): a short token applied to lower-case single-letter arguments
_FUNCTION = re.compile(r"^[^\s()]{1,6}\(\s*[a-z](\s*,\s*[a-z0-9])*\s*\)$")
# z*, x', A†
_MARKED = re.compile(r"^[A-Za-zα-ωΑ-Ω]{1,3}\d?[*'′″†˜~]+$")


def variable_like(name: str) -> bool:
    """Mathematical notation of three or more characters. Deliberately narrow:
    ``L1``, ``J2``, ``k2``, ``C20``, ``F10.7`` and ``P(Y)`` are real entities and
    none of them match. Shorter names are left to DROP_SYMBOL_ENTITIES."""
    n = name.strip()
    if len(n) < 3:
        return False
    if _TEX.search(n) or _FUNCTION.match(n):
        return True
    return " " not in n and bool(_SUBSCRIPT.match(n) or _MARKED.match(n))


_CAMEL = re.compile(r"^(?:[A-Z][a-z0-9]+){2,}[A-Za-z0-9]*$")
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def glued_parts(name: str) -> list[str] | None:
    """The words of a glued identifier (``NevadaCity``, ``error_term_variance``),
    else None."""
    n = name.strip()
    if " " in n or len(n) < 5:
        return None
    if "_" in n:
        parts = [p for p in n.split("_") if p]
        words = all(re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*|\d+", p) for p in parts)
        return parts if len(parts) >= 2 and words else None
    if _CAMEL.match(n):
        return [p for p in _CAMEL_SPLIT.split(n) if p]
    return None


def _spelling_in_text(parts: list[str], text: str) -> str | None:
    """The text's own spelling of the words ``parts``, joined by spaces or
    hyphens, if present."""
    m = re.search(
        r"(?<!\w)" + r"[\s\-–—/]+".join(re.escape(p) for p in parts) + r"(?!\w)",
        text,
        re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", m.group(0)) if m else None


_NOTATION_WORDS = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta kappa lambda mu nu xi pi rho "
    "sigma tau phi chi psi omega hat tilde bar plus minus".split()
)


def glued_action(name: str, text: str) -> tuple[str, str | None]:
    """What to do with one entity name: ``('keep', None)``, ``('rename', new
    name)`` or ``('drop', None)``.

    A glued identifier the text itself contains (``MetOp``, ``PyTorch``) is kept.
    One whose words the text contains is renamed to the text's spelling.
    Otherwise it is notation or a made-up identifier when a part is one or two
    characters, a Greek-letter or accent word, or carries a digit (``p_y_2``,
    ``PjPlus1``, ``PhiTau``, ``DoeEtAl2015``), and is dropped; a plain glued
    concept (``NormalDistribution``) is split into lower-case words."""
    parts = glued_parts(name)
    if not parts or name.casefold() in text.casefold():
        return "keep", None
    spelled = _spelling_in_text(parts, text)
    if spelled:
        return "rename", spelled
    if "etal" in name.casefold() or any(
        len(p) <= 2 or p.casefold() in _NOTATION_WORDS or any(c.isdigit() for c in p)
        for p in parts
    ):
        return "drop", None
    return "rename", " ".join(p if p.isupper() else p.lower() for p in parts)


def validate_chunk(
    chunk_text: str, maybe_nodes: dict, maybe_edges: dict
) -> tuple[dict, dict, dict]:
    """Apply the contract's checks to one chunk's parsed records (the parser's
    shapes: name -> [node records], (source, target) -> [edge records]).
    Returns new dicts and a count per action; the inputs are not modified.

    The chunk text is normalized the way extraction normalizes names
    (``normalize_entity_name``) before names are looked up in it, case-folded.

    A relation whose endpoint is not a declared entity is dropped only when that
    name is not in the chunk text either, or is notation. A grounded undeclared
    endpoint is kept, as without the validator (the merge creates the node):
    dropping all of them removes about a fifth of the relations on a paper corpus
    and leaves more entities without any relation."""
    stats = {
        "entities_in": len(maybe_nodes),
        "relations_in": len(maybe_edges),
        "variable": 0,
        "glued_renamed": 0,
        "glued_dropped": 0,
        "undeclared_kept": 0,
        "undeclared_dropped": 0,
        "endpoint_removed": 0,
        "self_loop": 0,
    }
    text = normalize_entity_name(chunk_text)
    folded_text = text.casefold()
    rename: dict[str, str] = {}
    removed: set[str] = set()
    nodes: dict = {}
    for name, records in maybe_nodes.items():
        if variable_like(name):
            stats["variable"] += 1
            removed.add(name)
            continue
        action, new_name = glued_action(name, text)
        if action == "drop":
            stats["glued_dropped"] += 1
            removed.add(name)
            continue
        if action == "rename":
            new_name = normalize_entity_name(new_name) or name
            stats["glued_renamed"] += 1
            rename[name] = new_name
            records = [dict(r, entity_name=new_name) for r in records]
        nodes.setdefault(new_name or name, []).extend(records)
    edges: dict = {}
    for (src, tgt), records in maybe_edges.items():
        if src in removed or tgt in removed:
            stats["endpoint_removed"] += 1
            continue
        src, tgt = rename.get(src, src), rename.get(tgt, tgt)
        if src == tgt:
            stats["self_loop"] += 1
            continue
        undeclared = [x for x in (src, tgt) if x not in nodes]
        if undeclared:
            if any(
                variable_like(x) or x.casefold() not in folded_text for x in undeclared
            ):
                stats["undeclared_dropped"] += 1
                continue
            stats["undeclared_kept"] += 1
        edges.setdefault((src, tgt), []).extend(
            dict(r, src_id=src, tgt_id=tgt) for r in records
        )
    stats["entities_out"], stats["relations_out"] = len(nodes), len(edges)
    return nodes, edges, stats


def kg_extraction_validator(
    chunk_key: str, chunk_text: str, maybe_nodes: dict, maybe_edges: dict
) -> tuple[dict, dict]:
    """``LightRAG.kg_extraction_validator``: runs after the parser's own filters
    and before the merge."""
    nodes, edges, stats = validate_chunk(chunk_text, maybe_nodes, maybe_edges)
    if (stats["entities_out"], stats["relations_out"]) != (
        stats["entities_in"],
        stats["relations_in"],
    ) or stats["glued_renamed"]:
        logger.debug(f"KG_CONTRACT_VALIDATOR {chunk_key}: {stats}")
    return nodes, edges


# ---- command line -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lightrag.kg_contract",
        description="Print the extraction JSON schema for an entity-type profile.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON schema of one extraction response",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="path of an entity-type profile (YAML) that declares relationship_types",
    )
    args = parser.parse_args(argv)
    if not args.schema:
        parser.error("nothing to do: pass --schema")
    try:
        schema = schema_for_profile(args.profile)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
