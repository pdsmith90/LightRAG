"""Extraction contract: what a chunk's extraction stored, the rebuild after a
deletion reproduces (lightrag/kg_contract.py, lightrag/operate.py).

- The extraction-quality hook (``kg_extraction_validator``) runs again on each
  chunk's re-parsed records, with the text it saw at extraction, so a name it
  renamed is rebuilt from the surviving document only, as without a hook.
- With ``relationship_inverses``, a gleaning turn that restates a first-pass
  relation the other way round with the same type corrects its direction; one
  of another type is a second relation. The rebuild merges a chunk's cached
  passes the same way.
- With ``relationship_types``, every LLM-extracted relation's keywords are
  checked against the list at parse time (first pass, gleaning, the rebuild's
  re-parse), the changes counted, logged per document and totalled for
  ``/health``; edges the code makes (sidecar links) are not checked.

All names are invented.
"""

import json
import logging
import shutil
from collections import Counter
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG, kg_contract
from lightrag.kg_contract import (
    RELATIONSHIP_INVERSES_V31,
    check_chunk_relation_keywords,
    check_relation_keywords,
    inverse_table,
    kg_extraction_validator,
    relation_keyword_changes,
    relation_keyword_totals,
)
from lightrag.operate import _merge_extraction_pass, extract_entities
from lightrag.prompt import load_entity_extraction_prompt_profile
from lightrag.utils import EmbeddingFunc, Tokenizer, logger

from tests.extraction.test_kg_contract import CONTRACT_SAMPLE

pytestmark = pytest.mark.offline

TABLE = inverse_table(RELATIONSHIP_INVERSES_V31)
NO_REPLY = json.dumps({"entities": [], "relationships": []})


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """contract.yml: the sample (types and inverses); noinv.yml: without the
    inverses; norelated.yml: without related_to."""
    (tmp_path / "entity_type").mkdir()
    shutil.copy(CONTRACT_SAMPLE, tmp_path / "entity_type" / "contract.yml")
    profile = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)
    noinv = {k: v for k, v in profile.items() if k != "relationship_inverses"}
    (tmp_path / "entity_type" / "noinv.yml").write_text(
        json.dumps(noinv), encoding="utf-8"
    )
    norelated = dict(profile)
    norelated["relationship_types"] = {
        k: v for k, v in profile["relationship_types"].items() if k != "related_to"
    }
    (tmp_path / "entity_type" / "norelated.yml").write_text(
        json.dumps(norelated), encoding="utf-8"
    )
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "9999999")
    return tmp_path


def _input_text(prompt: str) -> str:
    marker = "---Input Text---\n```\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    return prompt[start : prompt.find("\n```", start)]


def _reply(entities=(), relations=()) -> str:
    return json.dumps(
        {
            "entities": [
                {"name": n, "type": t, "description": d} for n, t, d in entities
            ],
            "relationships": [
                {"source": s, "target": o, "keywords": k, "description": d}
                for s, o, k, d in relations
            ],
        }
    )


def _llm(first, glean=None):
    """A stub extraction LLM: ``first(text)`` and ``glean(text)`` give the
    first-pass and gleaning replies for a chunk's text; anything else is a
    summary request."""

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        if history_messages:
            text = _input_text(history_messages[0]["content"])
            return glean(text) if glean else NO_REPLY
        text = _input_text(prompt)
        return first(text) if text else "A summary."

    return llm


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _make_rag(tmp_path, llm, *, profile=None, validator=None, gleaning=0):
    rag = LightRAG(
        working_dir=str(tmp_path / f"rag_{uuid4().hex}"),
        workspace=f"kgc_{uuid4().hex}",
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=gleaning,
        entity_extraction_use_json=True,
        chunk_token_size=4000,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        addon_params={"entity_type_prompt_file": profile} if profile else {},
        **({"kg_extraction_validator": validator} if validator else {}),
    )
    await rag.initialize_storages()
    return rag


def _strip(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ("created_at", "timestamp")}


def _graph(rag) -> dict:
    graph = rag.chunk_entity_relation_graph._graph
    return {
        "nodes": {n: _strip(dict(d)) for n, d in graph.nodes(data=True)},
        "edges": {
            tuple(sorted((u, v))): _strip(dict(d)) for u, v, d in graph.edges(data=True)
        },
    }


async def _run(tmp_path, llm, steps, **rag_kwargs) -> dict:
    """Insert (doc id, text) pairs, then delete the doc ids in ``deletes``;
    the graph afterwards."""
    rag = await _make_rag(tmp_path, llm, **rag_kwargs)
    try:
        for kind, doc_id, arg in steps:
            if kind == "insert":
                await rag.ainsert(arg, ids=[doc_id], file_paths=[f"{doc_id}.txt"])
            else:
                result = await rag.adelete_by_doc_id(doc_id, keep_summaries=arg)
                assert result.status == "success"
        return _graph(rag)
    finally:
        await rag.finalize_storages()


# ------------------------------------------------------------ F3-01: the hook in the rebuild

SCALE_DOCS = {
    "d1": "The scale height of the thermosphere is large. Report one.",
    "d2": "The scale height of the thermosphere is small. Report two.",
}


def _scale_height_llm(name: str):
    """The LLM names the quantity ``name`` in both documents."""

    def first(text):
        tag = "one" if "Report one" in text else "two"
        return _reply(
            [
                (name, "Quantity", f"A height of the layer ({tag})."),
                ("thermosphere", "Location", f"A layer ({tag})."),
            ],
            [(name, "thermosphere", "located_in", f"The height of the layer ({tag}).")],
        )

    return _llm(first)


@pytest.mark.parametrize("keep_summaries", [False, True], ids=["delete", "kept"])
async def test_a_validator_rename_is_rebuilt_from_the_surviving_document(
    tmp_path, keep_summaries
):
    """The LLM writes ``ScaleHeight``; the contract validator renames it to the
    text's ``scale height``. Deleting the second document leaves the edge and
    both nodes with the first document's chunk and text only: the graph equals
    that of the same deletion where the LLM wrote ``scale height`` and no hook
    ran."""
    steps = [
        ("insert", "d1", SCALE_DOCS["d1"]),
        ("insert", "d2", SCALE_DOCS["d2"]),
        ("delete", "d2", keep_summaries),
    ]
    validated = await _run(
        tmp_path,
        _scale_height_llm("ScaleHeight"),
        steps,
        validator=kg_extraction_validator,
    )
    plain = await _run(tmp_path, _scale_height_llm("scale height"), steps)

    edge = validated["edges"][("scale height", "thermosphere")]
    assert edge["source_id"] == "d1-chunk-000"
    assert edge["description"] == "The height of the layer (one)."
    node = validated["nodes"]["scale height"]
    assert node["source_id"] == "d1-chunk-000"
    assert "(two)" not in node["description"]
    assert validated == plain


class _RenamingValidator:
    """Any hook, not the contract's: async, stateful, renames ``Probe-X`` to
    ``probe x`` and records the text it saw for each chunk."""

    def __init__(self):
        self.texts: dict[str, list[str]] = {}

    async def __call__(self, chunk_key, chunk_text, maybe_nodes, maybe_edges):
        self.texts.setdefault(chunk_key, []).append(chunk_text)

        def name(n):
            return "probe x" if n == "Probe-X" else n

        nodes = {}
        for n, records in maybe_nodes.items():
            nodes.setdefault(name(n), []).extend(
                dict(r, entity_name=name(n)) for r in records
            )
        edges = {}
        for (s, t), records in maybe_edges.items():
            edges.setdefault((name(s), name(t)), []).extend(
                dict(r, src_id=name(s), tgt_id=name(t)) for r in records
            )
        return nodes, edges

    def __deepcopy__(self, memo):
        return self


async def test_any_validator_sees_the_extraction_text_again_in_the_rebuild(
    tmp_path,
):
    """A user hook gets each surviving chunk's records again in the rebuild,
    with the text it got at extraction (markup stripped as for the prompt)."""
    docs = {
        "d1": 'Probe-X watches <cite refid="r1">the Quux Belt</cite>. Report one.',
        "d2": "Probe-X watches the Quux Belt. Report two.",
    }

    def first(text):
        tag = "one" if "Report one" in text else "two"
        return _reply(
            [
                ("Probe-X", "Instrument", f"A probe ({tag})."),
                ("Quux Belt", "Location", f"A belt ({tag})."),
            ],
            [("Probe-X", "Quux Belt", "measures", f"The probe watches ({tag}).")],
        )

    validator = _RenamingValidator()
    graph = await _run(
        tmp_path,
        _llm(first),
        [
            ("insert", "d1", docs["d1"]),
            ("insert", "d2", docs["d2"]),
            ("delete", "d2", False),
        ],
        validator=validator,
    )
    texts = validator.texts["d1-chunk-000"]
    assert len(texts) == 2 and texts[0] == texts[1]
    assert "<cite" not in texts[0] and "the Quux Belt" in texts[0]
    edge = graph["edges"][("Quux Belt", "probe x")]
    assert edge["source_id"] == "d1-chunk-000"
    assert edge["description"] == "The probe watches (one)."
    assert graph["nodes"]["probe x"]["description"] == "A probe (one)."


# ------------------------------------------------------------ F3-02: a gleaning correction

CORRECTION_TEXT = "Alpha Unit is a component of the Zeta System."
UNITS = [
    ("Alpha Unit", "Instrument", "A unit."),
    ("Zeta System", "Mission", "A system."),
]


def _correction_llm(first_relation, glean_relation):
    """First pass and gleaning for CORRECTION_TEXT; any other text states
    ``Alpha Unit part_of Zeta System`` in the first pass only."""

    def first(text):
        if text == CORRECTION_TEXT:
            return _reply(UNITS, [first_relation])
        return _reply(
            UNITS,
            [("Alpha Unit", "Zeta System", "part_of", "Alpha in Zeta (other).")],
        )

    def glean(text):
        if text == CORRECTION_TEXT:
            return _reply((), [glean_relation])
        return NO_REPLY

    return _llm(first, glean)


WRONG = ("Zeta System", "Alpha Unit", "part_of", "Zeta part of Alpha.")
CORRECTED = (
    "Alpha Unit",
    "Zeta System",
    "part_of",
    "Corrected: Alpha Unit is a component of the Zeta System.",
)


async def test_a_gleaning_turn_corrects_the_direction(prompt_dir, tmp_path):
    graph = await _run(
        tmp_path,
        _correction_llm(WRONG, CORRECTED),
        [("insert", "d1", CORRECTION_TEXT)],
        profile="contract.yml",
        gleaning=1,
    )
    edge = graph["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "part_of"
    assert edge["description"] == CORRECTED[3]


async def test_a_reversed_gleaning_relation_of_another_type_is_kept_beside(
    prompt_dir, tmp_path
):
    graph = await _run(
        tmp_path,
        _correction_llm(
            ("Alpha Unit", "Zeta System", "part_of", "Alpha in Zeta."),
            ("Zeta System", "Alpha Unit", "uses", "Zeta uses Alpha."),
        ),
        [("insert", "d1", CORRECTION_TEXT)],
        profile="contract.yml",
        gleaning=1,
    )
    edge = graph["edges"][("Alpha Unit", "Zeta System")]
    assert set(edge["keywords"].split(",")) == {"part_of", "used_by"}
    assert set(edge["description"].split("<SEP>")) == {
        "Alpha in Zeta.",
        "Zeta uses Alpha.",
    }


async def test_without_inverses_both_directions_merge_as_before(prompt_dir, tmp_path):
    graph = await _run(
        tmp_path,
        _correction_llm(WRONG, CORRECTED),
        [("insert", "d1", CORRECTION_TEXT)],
        profile="noinv.yml",
        gleaning=1,
    )
    edge = graph["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "part_of"
    assert set(edge["description"].split("<SEP>")) == {WRONG[3], CORRECTED[3]}


@pytest.mark.parametrize(
    "glean_relation",
    [CORRECTED, ("Alpha Unit", "Zeta System", "affects", "Alpha affects Zeta.")],
    ids=["same-type", "other-type"],
)
async def test_the_rebuild_reproduces_a_chunk_with_a_gleaning_correction(
    prompt_dir, tmp_path, glean_relation
):
    """A second document states the relation too; deleting it rebuilds the
    edge and nodes from the first one's cached passes, and the graph equals the
    one that document alone produces."""
    llm = _correction_llm(WRONG, glean_relation)
    kwargs = dict(profile="contract.yml", gleaning=1)
    alone = await _run(tmp_path, llm, [("insert", "d1", CORRECTION_TEXT)], **kwargs)
    rebuilt = await _run(
        tmp_path,
        llm,
        [
            ("insert", "d1", CORRECTION_TEXT),
            ("insert", "d2", "Alpha Unit sits in the Zeta System."),
            ("delete", "d2", False),
        ],
        **kwargs,
    )
    for graph in (alone, rebuilt):
        edge = graph["edges"][("Alpha Unit", "Zeta System")]
        # Extraction joins keywords with "," and the rebuild with ", ", and the
        # two order equal-length description fragments differently.
        edge["keywords"] = ",".join(
            sorted(k.strip() for k in edge["keywords"].split(","))
        )
        edge["description"] = "<SEP>".join(sorted(edge["description"].split("<SEP>")))
    assert rebuilt == alone
    edge = rebuilt["edges"][("Alpha Unit", "Zeta System")]
    if glean_relation is CORRECTED:
        assert (edge["keywords"], edge["description"]) == ("part_of", CORRECTED[3])
    else:
        assert edge["keywords"] == "affects,has_part"
        assert set(edge["description"].split("<SEP>")) == {WRONG[3], glean_relation[3]}


def _edge(src, tgt, keywords, description="d"):
    return {
        "src_id": src,
        "tgt_id": tgt,
        "keywords": keywords,
        "description": description,
        "source_id": "chunk-1",
    }


async def test_merge_pass_rules():
    first = {
        ("B", "A"): [_edge("B", "A", "part_of"), _edge("B", "A", "uses", "dd")],
        ("C", "D"): [_edge("C", "D", "part_of")],
        ("E", "F"): [_edge("E", "F", "part_of")],
    }
    glean = {
        # restates part_of reversed: the B->A part_of record goes, uses stays
        ("A", "B"): [_edge("A", "B", "part_of")],
        # states both orders itself: corrects nothing
        ("D", "C"): [_edge("D", "C", "part_of")],
        ("C", "D"): [_edge("C", "D", "part_of")],
        # another type reversed: kept beside
        ("F", "E"): [_edge("F", "E", "affects")],
    }
    nodes, edges = {}, {k: list(v) for k, v in first.items()}
    await _merge_extraction_pass(nodes, edges, {}, glean, TABLE)
    assert edges[("B", "A")] == [_edge("B", "A", "uses", "dd")]
    assert edges[("A", "B")] == glean[("A", "B")]
    assert edges[("C", "D")] == first[("C", "D")]
    assert edges[("D", "C")] == glean[("D", "C")]
    assert edges[("E", "F")] == first[("E", "F")]
    assert edges[("F", "E")] == glean[("F", "E")]

    # the only record of a corrected pair takes the pair with it
    edges = {("B", "A"): [_edge("B", "A", "part_of")]}
    await _merge_extraction_pass(
        {}, edges, {}, {("A", "B"): [_edge("A", "B", "part_of")]}, TABLE
    )
    assert list(edges) == [("A", "B")]

    # without the table: the merge extraction always made
    edges = {k: list(v) for k, v in first.items()}
    await _merge_extraction_pass({}, edges, {}, glean, None)
    assert edges[("B", "A")] == first[("B", "A")]


# ------------------------------------------------------------ F3-03: the keyword check

TYPES = load_entity_extraction_prompt_profile(CONTRACT_SAMPLE)["relationship_types"]
WITHOUT_RELATED = {k: v for k, v in TYPES.items() if k != "related_to"}


@pytest.mark.parametrize(
    "keywords,types,expected",
    [
        ("part_of", TYPES, ("part_of", None)),
        ("Part_Of", TYPES, ("part_of", "normalized")),
        ("  PART_OF ", TYPES, ("part_of", "normalized")),
        ("part_of, part_of", TYPES, ("part_of", "normalized")),
        ("part_of, uses", TYPES, ("part_of,uses", "multi_type")),
        ("Uses,member of", TYPES, ("uses", "multi_type")),
        ("member of", TYPES, ("related_to", "related_to")),
        ("part of", TYPES, ("related_to", "related_to")),
        ("", TYPES, ("related_to", "related_to")),
        ("member of", WITHOUT_RELATED, (None, "dropped")),
        (
            "has_part",
            TYPES,
            ("related_to", "related_to"),
        ),  # inverse names are not on the list
    ],
)
def test_check_relation_keywords(keywords, types, expected):
    assert check_relation_keywords(keywords, types) == expected


def test_check_chunk_relation_keywords_counts_and_drops():
    counts = Counter()
    edges = {
        ("A", "B"): [_edge("A", "B", "Part_Of"), _edge("A", "B", "free words")],
        ("C", "D"): [_edge("C", "D", "free words")],
        ("E", "F"): [_edge("E", "F", "uses")],
    }
    checked = check_chunk_relation_keywords(edges, WITHOUT_RELATED, counts)
    assert checked == {
        ("A", "B"): [_edge("A", "B", "part_of")],
        ("E", "F"): [edges[("E", "F")][0]],
    }
    assert edges[("A", "B")][0]["keywords"] == "Part_Of"  # input left alone
    assert counts == Counter(checked=4, normalized=1, dropped=2)
    assert relation_keyword_changes(counts) == (
        "3 of 4 relation keywords were not exactly one declared relationship type "
        "(normalized=1, multi_type=0, related_to=0, dropped=2)"
    )
    assert relation_keyword_changes(Counter(checked=4)) is None


@pytest.fixture
def warnings_seen(monkeypatch, caplog):
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger="lightrag")
    return caplog


def _totals_delta(before):
    after = relation_keyword_totals()
    return {
        stage: {k: after[stage][k] - before[stage][k] for k in after[stage]}
        for stage in after
    }


OFF_LIST_TEXT = "GRACE is one of the satellites of the Constellation."


def _off_list_llm():
    def first(text):
        return _reply(
            [
                ("GRACE", "Mission", "A mission."),
                ("Constellation", "Mission", "A group."),
                ("Ground Net", "Organization", "A network."),
            ],
            [
                # flipped into storage order: must read has_part once normalized
                ("GRACE", "Constellation", "Part_Of", "GRACE in the group."),
                ("Ground Net", "GRACE", "member of", "The net tracks GRACE."),
            ],
        )

    def glean(text):
        return _reply(
            (),
            [("Ground Net", "Constellation", "operates, runs", "The net runs it.")],
        )

    return _llm(first, glean)


async def test_extraction_checks_keywords_and_reports_the_document(
    prompt_dir, tmp_path, warnings_seen
):
    before = relation_keyword_totals()
    graph = await _run(
        tmp_path,
        _off_list_llm(),
        [("insert", "d1", OFF_LIST_TEXT)],
        profile="contract.yml",
        gleaning=1,
    )
    assert graph["edges"][("Constellation", "GRACE")]["keywords"] == "has_part"
    assert graph["edges"][("GRACE", "Ground Net")]["keywords"] == "related_to"
    assert graph["edges"][("Constellation", "Ground Net")]["keywords"] == "operated_by"
    delta = _totals_delta(before)
    assert delta["extraction"] == {
        "checked": 3,
        "normalized": 1,
        "multi_type": 1,
        "related_to": 1,
        "dropped": 0,
    }
    assert not any(delta["rebuild"].values())
    [warning] = [
        r.getMessage()
        for r in warnings_seen.records
        if "relation keywords" in r.getMessage()
    ]
    assert "3 of 3 relation keywords" in warning and "d1.txt" in warning


async def test_without_related_to_an_off_list_relation_is_dropped(prompt_dir, tmp_path):
    graph = await _run(
        tmp_path,
        _off_list_llm(),
        [("insert", "d1", OFF_LIST_TEXT)],
        profile="norelated.yml",
        gleaning=1,
    )
    assert ("GRACE", "Ground Net") not in graph["edges"]
    assert graph["edges"][("Constellation", "GRACE")]["keywords"] == "has_part"


async def test_the_rebuild_re_parse_checks_keywords(
    prompt_dir, tmp_path, warnings_seen
):
    def first(text):
        tag = "one" if "one" in text else "two"
        return _reply(
            [
                ("GRACE", "Mission", f"A mission ({tag})."),
                ("Constellation", "Mission", "A group."),
            ],
            [("GRACE", "Constellation", "Part_Of", f"GRACE in the group ({tag}).")],
        )

    before = relation_keyword_totals()
    graph = await _run(
        tmp_path,
        _llm(first),
        [
            ("insert", "d1", "GRACE is in the Constellation, report one."),
            ("insert", "d2", "GRACE is in the Constellation, report two."),
            ("delete", "d2", False),
        ],
        profile="contract.yml",
    )
    edge = graph["edges"][("Constellation", "GRACE")]
    assert edge["keywords"] == "has_part"
    assert edge["description"] == "GRACE in the group (one)."
    delta = _totals_delta(before)
    assert delta["extraction"]["normalized"] == 2
    assert delta["rebuild"]["normalized"] == 1 and delta["rebuild"]["checked"] == 1
    assert any(
        "KG rebuild" in r.getMessage() and "normalized=1" in r.getMessage()
        for r in warnings_seen.records
    )


async def test_sidecar_links_are_not_checked(prompt_dir):
    """The fan-out edges the code makes for a table chunk keep their free-text
    keywords even under a list without related_to."""
    reply = _reply(
        [("Quux Basin", "Location", "A basin.")],
        [("Quux Basin", "Table One", "free words", "Dropped.")],
    )

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        return reply

    from lightrag.prompt import (
        resolve_entity_extraction_prompt_profile,
        validate_entity_extraction_prompt_profile_for_mode,
    )

    addon = {"entity_type_prompt_file": "norelated.yml"}
    global_config = {
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": addon,
        "llm_model_max_async": 1,
        "entity_extraction_use_json": True,
        "_entity_extraction_prompt_profile": validate_entity_extraction_prompt_profile_for_mode(
            resolve_entity_extraction_prompt_profile(addon, True), True, "norelated.yml"
        ),
    }
    chunks = {
        "chunk-1": {
            "tokens": 10,
            "content": "Table One lists the Quux Basin.",
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
            "sidecar": {"type": "table", "id": "tbl-1"},
        }
    }
    [(nodes, edges)] = await extract_entities(
        chunks=chunks, global_config=global_config
    )
    assert ("Quux Basin", "Table One") not in edges
    assert (
        edges[("tbl-1", "Quux Basin")][0]["keywords"] == "associated with, contained in"
    )


def test_health_totals_start_with_every_count():
    totals = relation_keyword_totals()
    assert set(totals) == {"extraction", "rebuild"}
    for stage in totals.values():
        assert set(stage) == set(kg_contract.RELATION_KEYWORD_COUNTS)
