"""Extraction contract: the knowledge rebuild that follows a deletion re-parses
each surviving chunk's cached extraction the way that chunk was post-processed
when it was extracted (lightrag/operate.py, lightrag/kg_contract.py).

- A chunk extracted without any contract or hook (every chunk extracted before
  this rule) gets the plain re-parse even after the hook or the relation list
  is switched on, so the graph is rebuilt exactly as without them.
- A chunk extracted under another contract or hook cannot be reproduced; an
  entity or relation for which it then yields no record keeps its stored
  description and gets its provenance repaired, and the deletion says so.
- A hook that raises in the rebuild, or a chunk without its text, is handled
  the same way: no unjudged records reach the graph, no stale chunk id stays.
- With the name fold armed and relationship_inverses declared, the rebuild
  folds after the gleaning merge, as extraction does.

Each test drives a real LightRAG instance (JSON and NetworkX storages) through
inserts, a reopen with another configuration and a deletion. All names are
invented.
"""

import json
from uuid import uuid4

import pytest

from lightrag import LightRAG, entity_name_fold
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.kg_contract import kg_extraction_validator
from lightrag.utils import EmbeddingFunc, Tokenizer

from tests.extraction import test_kg_contract_rebuild as _rebuild_tests
from tests.extraction.test_kg_contract_rebuild import (
    NO_REPLY,
    _Characters,
    _embed,
    _graph,
    _llm,
    _reply,
)

# The profiles fixture of the rebuild tests: contract.yml (the sample, types
# and inverses), noinv.yml, norelated.yml (without related_to).
prompt_dir = _rebuild_tests.prompt_dir

pytestmark = pytest.mark.offline

TAGS = {"Report one": "A", "Report two": "B", "Report three": "C"}

DOCS = {
    "A": (
        "Report one. The scale height of the thermosphere is large. The Zeta "
        "System holds the Alpha Unit. The Mid Tool uses the Beta Set."
    ),
    "B": (
        "Report two. The scale height of the thermosphere is small. The Zeta "
        "System holds the Alpha Unit. The Mid Tool uses the Beta Set."
    ),
    "C": "Report three. The Zeta System holds the Alpha Unit.",
}


def _tag(text: str) -> str:
    return next(tag for marker, tag in TAGS.items() if marker in text)


def _first(text: str) -> str:
    """The LLM glues ``scale height`` into ``ScaleHeight`` (the contract
    validator renames it back) and writes free keywords for Zeta/Alpha."""
    tag = _tag(text)
    entities, relations = [], []
    if "scale height" in text:
        entities += [
            ("ScaleHeight", "Concept", f"SH {tag}."),
            ("thermosphere", "Location", f"T {tag}."),
        ]
        relations.append(("ScaleHeight", "thermosphere", "located_in", f"SH-T {tag}."))
    if "Zeta" in text:
        entities += [
            ("Zeta System", "Concept", f"Z {tag}."),
            ("Alpha Unit", "Concept", f"A {tag}."),
        ]
        relations.append(("Zeta System", "Alpha Unit", "contains, holds", f"ZA {tag}."))
    if "Mid Tool" in text:
        entities += [
            ("Mid Tool", "Method", f"M {tag}."),
            ("Beta Set", "Data", f"B {tag}."),
        ]
        relations.append(("Mid Tool", "Beta Set", "uses", f"MB {tag}."))
    return _reply(entities, relations)


async def _open(path, workspace, llm, *, profile=None, validator=None, gleaning=0):
    rag = LightRAG(
        working_dir=str(path),
        workspace=workspace,
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


async def _insert(rag, *doc_ids):
    for doc_id in doc_ids:
        await rag.ainsert(DOCS[doc_id], ids=[doc_id], file_paths=[f"{doc_id}.txt"])


async def _tracking(rag) -> dict:
    graph = _graph(rag)
    entity_rows = await rag.entity_chunks.get_by_ids(sorted(graph["nodes"]))
    return {
        "entities": {
            name: (row or {}).get("chunk_ids")
            for name, row in zip(sorted(graph["nodes"]), entity_rows)
        },
    }


async def _history(rag) -> list[str]:
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    return list(status.get("history_messages") or [])


async def _build_then_delete(tmp_path, before: dict, after: dict, docs=("A", "B")):
    """Insert ``docs`` under the configuration ``before``, reopen the same
    storage under ``after`` and delete B: the graph before and after the
    deletion, the deletion result and the pipeline history of the deletion."""
    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    rag = await _open(path, workspace, _llm(_first), **before)
    try:
        await _insert(rag, *docs)
        built = _graph(rag)
    finally:
        await rag.finalize_storages()
    rag = await _open(path, workspace, _llm(_first), **after)
    try:
        result = await rag.adelete_by_doc_id("B")
        return {
            "built": built,
            "graph": _graph(rag),
            "tracking": await _tracking(rag),
            "result": result,
            "history": await _history(rag),
        }
    finally:
        await rag.finalize_storages()


# ------------------------------------- chunks extracted without the contract or hook


async def test_switching_the_validator_on_rebuilds_old_chunks_as_they_were(tmp_path):
    """COLD1-01: a graph built without the hook, the hook switched on, a
    document deleted: every item is rebuilt from the surviving document's
    plain extraction, as without the hook."""
    hooked = await _build_then_delete(
        tmp_path, {}, {"validator": kg_extraction_validator}
    )
    plain = await _build_then_delete(tmp_path, {}, {})
    assert hooked["result"].status == "success"
    assert "unreproducible" not in hooked["result"].message
    assert hooked["graph"] == plain["graph"]
    edge = hooked["graph"]["edges"][("ScaleHeight", "thermosphere")]
    assert edge["source_id"] == "A-chunk-000"
    assert edge["description"] == "SH-T A."
    assert hooked["graph"]["nodes"]["ScaleHeight"]["description"] == "SH A."


@pytest.mark.parametrize("profile", ["contract.yml", "norelated.yml"])
async def test_declaring_the_relation_list_rebuilds_old_chunks_as_they_were(
    prompt_dir, tmp_path, profile
):
    """COLD1-03: cached extractions written before ``relationship_types`` was
    declared keep their free keywords in the rebuild, with or without
    ``related_to`` on the list."""
    contract = await _build_then_delete(tmp_path, {}, {"profile": profile})
    plain = await _build_then_delete(tmp_path, {}, {})
    assert "unreproducible" not in contract["result"].message
    assert contract["graph"] == plain["graph"]
    edge = contract["graph"]["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "contains, holds"
    assert edge["source_id"] == "A-chunk-000"
    assert edge["description"] == "ZA A."


# ------------------------------------------ chunks extracted under another configuration


async def test_switching_the_validator_off_repairs_what_it_renamed(tmp_path):
    """A graph built with the hook, the hook switched off, a document deleted:
    the surviving chunk's plain re-parse has no record for the names the hook
    gave, so those items keep their stored descriptions and lose the deleted
    document's chunk id; the deletion says it completed with unreproducible
    items."""
    out = await _build_then_delete(tmp_path, {"validator": kg_extraction_validator}, {})
    result, built, graph = out["result"], out["built"], out["graph"]
    assert result.status == "success"
    assert "completed with unreproducible items" in result.message
    assert "1 surviving chunk(s)" in result.message
    assert any("cannot be reproduced" in m for m in out["history"])

    key = ("scale height", "thermosphere")
    assert graph["edges"][key]["source_id"] == "A-chunk-000"
    assert graph["edges"][key]["description"] == built["edges"][key]["description"]
    node = graph["nodes"]["scale height"]
    assert node["source_id"] == "A-chunk-000"
    assert node["description"] == built["nodes"]["scale height"]["description"]
    assert out["tracking"]["entities"]["scale height"] == ["A-chunk-000"]
    # a name the hook left alone is rebuilt from the plain re-parse
    assert graph["nodes"]["thermosphere"]["description"] == "T A."


async def test_changing_the_relation_list_counts_the_old_chunks(prompt_dir, tmp_path):
    """A graph built with one relation list, then another: the surviving chunk
    is re-parsed plainly and counted; nothing keeps the deleted chunk id."""
    out = await _build_then_delete(
        tmp_path, {"profile": "contract.yml"}, {"profile": "norelated.yml"}
    )
    assert "completed with unreproducible items" in out["result"].message
    for data in (*out["graph"]["nodes"].values(), *out["graph"]["edges"].values()):
        assert "B-chunk-000" not in data["source_id"]


# ------------------------------------------- the hook fails or has no text in the rebuild


class _FlakyValidator:
    """The contract validator, raising for one chunk once switched on."""

    def __init__(self):
        self.fail_on = None

    def __call__(self, chunk_key, chunk_text, maybe_nodes, maybe_edges):
        if chunk_key == self.fail_on:
            raise RuntimeError("transient hook failure")
        return kg_extraction_validator(chunk_key, chunk_text, maybe_nodes, maybe_edges)

    def __deepcopy__(self, memo):
        return self


async def _hook_trouble(tmp_path, trouble: str) -> dict:
    validator = _FlakyValidator()
    rag = await _open(
        tmp_path / f"rag_{uuid4().hex}",
        f"kgc_{uuid4().hex}",
        _llm(_first),
        validator=validator,
    )
    try:
        await _insert(rag, "A", "B", "C")
        built = _graph(rag)
        if trouble == "raises":
            validator.fail_on = "A-chunk-000"
        else:
            row = await rag.text_chunks.get_by_id("A-chunk-000")
            await rag.text_chunks.upsert({"A-chunk-000": {**row, "content": None}})
        result = await rag.adelete_by_doc_id("B")
        return {
            "built": built,
            "graph": _graph(rag),
            "tracking": await _tracking(rag),
            "result": result,
        }
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("trouble", ["raises", "no-text"])
async def test_a_failing_hook_in_the_rebuild_repairs_provenance(tmp_path, trouble):
    """COLD1-02: the hook raises on (or cannot run for) a surviving chunk. The
    deletion completes and says so; no item keeps the deleted chunk id or text
    of a record the hook did not judge; the items the chunk touches keep their
    stored descriptions with their provenance repaired."""
    out = await _hook_trouble(tmp_path, trouble)
    result, built, graph = out["result"], out["built"], out["graph"]
    assert result.status == "success"
    assert "completed with unreproducible items" in result.message
    for data in (*graph["nodes"].values(), *graph["edges"].values()):
        assert "B-chunk-000" not in data["source_id"]

    key = ("Beta Set", "Mid Tool")  # A and B only
    assert graph["edges"][key]["source_id"] == "A-chunk-000"
    assert graph["edges"][key]["description"] == built["edges"][key]["description"]
    assert out["tracking"]["entities"]["Mid Tool"] == ["A-chunk-000"]
    assert graph["nodes"]["Mid Tool"]["source_id"] == "A-chunk-000"

    node = graph["nodes"]["Alpha Unit"]  # A, B and C
    assert node["source_id"].split("<SEP>") == ["A-chunk-000", "C-chunk-000"]
    assert node["description"] == built["nodes"]["Alpha Unit"]["description"]


# ------------------------------------------------- fold, inverse names and the gleaning

FOLD_TEXT = "Report one. The Alpha Unit is a component of the ZETA SYSTEM."


def _fold_llm():
    """Doc A spells ``ZETA SYSTEM`` in the first pass and ``Zeta System`` in
    the gleaning, which restates the pair the other way round; any other text
    states the pair once."""

    def first(text):
        if text == FOLD_TEXT:
            return _reply(
                [
                    ("ZETA SYSTEM", "Concept", "Upper spelling."),
                    ("Alpha Unit", "Concept", "A unit."),
                ],
                [
                    (
                        "ZETA SYSTEM",
                        "Alpha Unit",
                        "part_of",
                        "first pass upper spelling, ZETA part_of Alpha.",
                    )
                ],
            )
        return _reply(
            [("Zeta System", "Concept", "A system."), ("Alpha Unit", "Concept", "A.")],
            [("Alpha Unit", "Zeta System", "part_of", "Alpha in Zeta (other).")],
        )

    def glean(text):
        if text == FOLD_TEXT:
            return _reply(
                (),
                [
                    (
                        "Alpha Unit",
                        "Zeta System",
                        "part_of",
                        "gleaning: Alpha part_of Zeta.",
                    )
                ],
            )
        return NO_REPLY

    return _llm(first, glean)


def _normalized(graph: dict) -> dict:
    # Extraction joins keywords with "," and the rebuild with ", "; the two
    # order description fragments of equal timestamps differently.
    for edge in graph["edges"].values():
        edge["keywords"] = ",".join(
            sorted(k.strip() for k in edge["keywords"].split(","))
        )
        edge["description"] = "<SEP>".join(sorted(edge["description"].split("<SEP>")))
    return graph


async def _fold_run(tmp_path, steps, validator):
    rag = await _open(
        tmp_path / f"rag_{uuid4().hex}",
        f"kgc_{uuid4().hex}",
        _fold_llm(),
        profile="contract.yml",
        validator=validator,
        gleaning=1,
    )
    entity_name_fold.load_fold_map([("Zeta System", "concept", 3)])
    try:
        for kind, doc_id, text in steps:
            if kind == "insert":
                await rag.ainsert(text, ids=[doc_id], file_paths=[f"{doc_id}.txt"])
            else:
                assert (await rag.adelete_by_doc_id(doc_id)).status == "success"
        return _normalized(_graph(rag))
    finally:
        entity_name_fold.disarm()
        await rag.finalize_storages()


@pytest.mark.parametrize(
    "validator", [None, kg_extraction_validator], ids=["no-hook", "hook"]
)
async def test_the_rebuild_folds_after_the_gleaning_merge(
    prompt_dir, tmp_path, validator
):
    """COLD1-04: with the fold armed and relationship_inverses declared, a
    chunk whose two passes spell an endpoint differently is rebuilt to what its
    extraction stored: both relations, as the document alone gives them."""
    alone = await _fold_run(tmp_path, [("insert", "A", FOLD_TEXT)], validator)
    rebuilt = await _fold_run(
        tmp_path,
        [
            ("insert", "A", FOLD_TEXT),
            ("insert", "B", "Report two. Alpha Unit sits in the Zeta System."),
            ("delete", "B", None),
        ],
        validator,
    )
    edge = alone["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "has_part,part_of"
    assert rebuilt == alone


# ------------------------------------------------------------- inverse names, JSON lists


def _keywords_llm(keywords):
    def first(text):
        tag = _tag(text)
        return json.dumps(
            {
                "entities": [
                    {"name": "Constellation", "type": "Mission", "description": "G."},
                    {"name": "GRACE", "type": "Mission", "description": f"M {tag}."},
                ],
                "relationships": [
                    {
                        "source": "Constellation",
                        "target": "GRACE",
                        "keywords": keywords,
                        "description": f"The group holds GRACE ({tag}).",
                    }
                ],
            }
        )

    return _llm(first)


@pytest.mark.parametrize(
    "keywords,stored",
    [
        # "Constellation has_part GRACE" is "GRACE part_of Constellation", which
        # reads has_part in storage order (Constellation, GRACE)
        ("has_part", "has_part"),
        # a JSON list is read as its elements
        (["has_part"], "has_part"),
        (["Part_Of"], "part_of"),
    ],
)
async def test_inverse_names_and_keyword_lists_are_read_under_the_contract(
    prompt_dir, tmp_path, keywords, stored
):
    """COLD1-06 and COLD1-10: an inverse name is the relation read the other
    way round, a JSON list is its elements; the rebuild re-reads them the
    same way."""
    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    rag = await _open(path, workspace, _keywords_llm(keywords), profile="contract.yml")
    try:
        await rag.ainsert("Report one. GRACE flies.", ids=["A"], file_paths=["A.txt"])
        alone = _graph(rag)
        await rag.ainsert("Report two. GRACE flies.", ids=["B"], file_paths=["B.txt"])
        assert (await rag.adelete_by_doc_id("B")).status == "success"
        rebuilt = _graph(rag)
    finally:
        await rag.finalize_storages()
    for graph in (alone, rebuilt):
        edge = graph["edges"][("Constellation", "GRACE")]
        assert edge["keywords"] == stored
        assert edge["source_id"] == "A-chunk-000"


# ------------------------------------------------------- the fingerprint and its use


def test_the_contract_fingerprint():
    from functools import partial

    from lightrag.kg_contract import (
        RELATIONSHIP_INVERSES_V31,
        RELATIONSHIP_TYPES_V31,
        contract_fingerprint,
        validator_identity,
    )

    types, inverses = RELATIONSHIP_TYPES_V31, RELATIONSHIP_INVERSES_V31
    assert contract_fingerprint(None, None, None) is None
    assert contract_fingerprint({}, {}, None) is None
    base = contract_fingerprint(types, inverses, None)
    assert base.startswith("kgc-") and len(base) == 20
    # stable, and independent of the definitions' text and of the order
    assert (
        contract_fingerprint(
            dict(reversed(list({k: "x" for k in types}.items()))), inverses, None
        )
        == base
    )
    others = {
        contract_fingerprint(types, None, None),
        contract_fingerprint(
            {k: v for k, v in types.items() if k != "related_to"}, inverses, None
        ),
        contract_fingerprint(types, inverses, kg_extraction_validator),
        contract_fingerprint(None, None, kg_extraction_validator),
    }
    assert base not in others and len(others) == 4

    def hook(chunk_key, chunk_text, nodes, edges):
        return nodes, edges

    assert validator_identity(hook).endswith(
        "test_the_contract_fingerprint.<locals>.hook"
    )
    assert validator_identity(partial(hook)) == validator_identity(hook)
    assert validator_identity(_FlakyValidator()).endswith("._FlakyValidator")
    hook.kg_validator_version = 2
    assert validator_identity(hook).endswith(".hook/v2")
    assert "kg_extraction_validator/v" in validator_identity(kg_extraction_validator)


@pytest.mark.parametrize(
    "made_under,mode",
    [
        ({}, "baseline"),  # before fingerprints, or no contract and no hook
        ({"k1": "kgc-now", "k2": "kgc-now"}, "contract"),
        ({"k1": "kgc-old", "k2": "kgc-old"}, "unreproducible"),
        ({"k1": "kgc-now"}, "unreproducible"),  # COLD1-11: two runs, two configs
        ({"k1": "kgc-now", "k2": "kgc-old"}, "unreproducible"),
    ],
)
def test_how_a_chunk_is_re_parsed(made_under, mode):
    from lightrag.operate import _cached_extraction_mode

    results = [("r1", 1, "k1"), ("r2", 2, "k2")]
    row = {"llm_cache_list": ["k1", "k2"], "llm_cache_contract": made_under}
    assert _cached_extraction_mode(row, results, "kgc-now") == mode
    if not made_under:
        assert _cached_extraction_mode(row, results, None) == "baseline"
        assert _cached_extraction_mode(None, [("r1", 1)], "kgc-now") == "baseline"
    elif mode == "contract":
        # the contract switched off after the chunk was extracted under it
        assert _cached_extraction_mode(row, results, None) == "unreproducible"


async def test_extraction_records_the_fingerprint_only_under_a_contract(
    prompt_dir, tmp_path
):
    from lightrag.kg_contract import contract_fingerprint
    from lightrag.prompt import load_entity_extraction_prompt_profile

    for profile, validator in (
        (None, None),
        ("contract.yml", None),
        (None, kg_extraction_validator),
    ):
        rag = await _open(
            tmp_path / f"rag_{uuid4().hex}",
            f"kgc_{uuid4().hex}",
            _llm(_first),
            profile=profile,
            validator=validator,
        )
        try:
            await _insert(rag, "C")
            row = await rag.text_chunks.get_by_id("C-chunk-000")
        finally:
            await rag.finalize_storages()
        if profile is None and validator is None:
            assert "llm_cache_contract" not in row
            continue
        types = inverses = None
        if profile:
            loaded = load_entity_extraction_prompt_profile(
                prompt_dir / "entity_type" / profile
            )
            types, inverses = (
                loaded["relationship_types"],
                loaded["relationship_inverses"],
            )
        expected = contract_fingerprint(types, inverses, validator)
        assert row["llm_cache_contract"] == {
            key: expected for key in row["llm_cache_list"]
        }


def test_the_postgres_column_reads_like_the_other_backends():
    from lightrag.kg.postgres_impl import _dump_cache_contract, _parse_cache_contract

    row = {"id": "c1", "llm_cache_contract": None}
    _parse_cache_contract(row)
    assert row == {"id": "c1"}  # a NULL column is no key, as before the column
    row = {"id": "c1", "llm_cache_contract": '{"k1": "kgc-a"}'}
    _parse_cache_contract(row)
    assert row == {"id": "c1", "llm_cache_contract": {"k1": "kgc-a"}}
    assert _dump_cache_contract({"id": "c1"}) is None
    assert _dump_cache_contract(row) == '{"k1": "kgc-a"}'
