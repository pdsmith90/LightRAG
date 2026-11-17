"""Extraction contract: the knowledge rebuild that follows a deletion reads each
surviving chunk's stored accepted records (lightrag/operate.py,
lightrag/kg_contract.py), whatever the configuration is at that time.

- A chunk extracted with the contract or a hook stored the records extraction
  handed the merge; the rebuild reads them and re-runs none of the
  post-processing, so changing the relation list, the inverse names, the hook
  (or its rules), switching the contract off or the JSON mode changes nothing
  in what a deletion rebuilds.
- A chunk extracted with neither (every chunk extracted before the contract)
  stored none and is re-parsed from its cached LLM output, as on the base
  branch, even when the contract or hook is on later; an entity or relation fed
  by both kinds of chunk gets each chunk's contribution its way.
- A chunk extracted again with neither loses its stored records.
- The stored records carry the names the merge used, after the name fold, so
  a rebuild finds them whatever the fold map is then (off, not armed, another
  canonical, another alias file); cached results are still filed under both
  spellings.

Each test drives a real LightRAG instance (JSON and NetworkX storages) through
inserts, a reopen with another configuration and a deletion. All names are
invented.
"""

import json
import sys
from uuid import uuid4

import pytest

from lightrag import LightRAG, entity_name_fold, kg_contract
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.kg_contract import kg_extraction_validator
from lightrag.utils import (
    CHUNK_KG_RECORDS_FIELD,
    EmbeddingFunc,
    Tokenizer,
    TruncatedResponse,
)

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


async def _open(
    path,
    workspace,
    llm,
    *,
    profile=None,
    validator=None,
    gleaning=0,
    use_json=True,
    **extra,
):
    rag = LightRAG(
        working_dir=str(path),
        workspace=workspace,
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=gleaning,
        entity_extraction_use_json=use_json,
        chunk_token_size=4000,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        addon_params={"entity_type_prompt_file": profile} if profile else {},
        **({"kg_extraction_validator": validator} if validator else {}),
        **extra,
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
    assert contract["graph"] == plain["graph"]
    edge = contract["graph"]["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "contains, holds"
    assert edge["source_id"] == "A-chunk-000"
    assert edge["description"] == "ZA A."


# ------------------------------- chunks with stored records: the configuration changes


def _flipped(text: str) -> str:
    """COLD3-01: ``Zeta System part_of Alpha Unit`` and ``Mid Tool uses Beta
    Set``, both pairs sorting the other way round; ``ScaleHeight`` for the hook
    to rename."""
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
        relations.append(("Zeta System", "Alpha Unit", "part_of", f"ZA {tag}."))
    if "Mid Tool" in text:
        entities += [
            ("Mid Tool", "Method", f"M {tag}."),
            ("Beta Set", "Data", f"B {tag}."),
        ]
        relations.append(("Mid Tool", "Beta Set", "uses", f"MB {tag}."))
    return _reply(entities, relations)


@pytest.fixture
def flipped_llm(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_first", _flipped)


@pytest.fixture
def other_profile(prompt_dir):
    """other.yml: another relation list (no ``affects``) and other inverse
    names for part_of and uses."""
    from lightrag.prompt import load_entity_extraction_prompt_profile

    profile = load_entity_extraction_prompt_profile(
        prompt_dir / "entity_type" / "contract.yml"
    )
    profile["relationship_types"] = {
        k: v for k, v in profile["relationship_types"].items() if k != "affects"
    }
    profile["relationship_inverses"] = {
        k: {"part_of": "contains_part", "uses": "employed_by"}.get(k, v)
        for k, v in profile["relationship_inverses"].items()
        if k != "affects"
    }
    (prompt_dir / "entity_type" / "other.yml").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    return "other.yml"


class _OtherValidator:
    """Another hook: renames every name to upper case."""

    def __call__(self, chunk_key, chunk_text, maybe_nodes, maybe_edges):
        nodes = {
            n.upper(): [dict(r, entity_name=n.upper()) for r in records]
            for n, records in maybe_nodes.items()
        }
        edges = {
            (s.upper(), t.upper()): [
                dict(r, src_id=s.upper(), tgt_id=t.upper()) for r in records
            ]
            for (s, t), records in maybe_edges.items()
        }
        return nodes, edges

    def __deepcopy__(self, memo):
        return self


def _raising_validator(chunk_key, chunk_text, maybe_nodes, maybe_edges):
    raise RuntimeError("hook down")


def _broken_rules(monkeypatch):
    """A code update that changes every rule: the rules version (where there is
    one), the keyword check and the inverse names."""
    monkeypatch.setattr(kg_contract, "CONTRACT_RULES_VERSION", 2, raising=False)
    monkeypatch.setattr(kg_contract, "VALIDATOR_RULES_VERSION", 2, raising=False)

    def check(maybe_edges, *args, **kwargs):
        return {
            key: [dict(r, keywords="changed_rule") for r in records]
            for key, records in maybe_edges.items()
        }

    import lightrag.operate as operate

    monkeypatch.setattr(operate, "check_chunk_relation_keywords", check)
    monkeypatch.setattr(kg_contract, "invert_keywords", lambda k, t: "changed_rule")


BUILT_UNDER = {
    "contract": {"profile": "contract.yml"},
    "contract+hook": {"profile": "contract.yml", "validator": kg_extraction_validator},
}
AFTER = {
    "same": None,
    "rules-version": "rules",
    "hook-swapped": {"profile": "contract.yml", "validator": _OtherValidator()},
    "hook-removed": {"profile": "contract.yml"},
    "hook-added": {"profile": "contract.yml", "validator": kg_extraction_validator},
    "hook-raises": {"profile": "contract.yml", "validator": _raising_validator},
    "list-and-inverses-changed": {"profile": "other.yml"},
    "inverses-removed": {"profile": "noinv.yml"},
    "contract-off": {},
    "text-mode": {"use_json": False},
}


@pytest.mark.usefixtures("flipped_llm")
@pytest.mark.parametrize("after", list(AFTER))
@pytest.mark.parametrize("built_under", list(BUILT_UNDER))
async def test_a_deletion_rebuilds_what_extraction_stored_whatever_the_config(
    prompt_dir, other_profile, tmp_path, monkeypatch, built_under, after
):
    """COLD3-01, R1: a graph built under the contract (with or without the
    hook), the configuration changed, a document deleted: the graph equals the
    one the same deletion gives under the unchanged configuration, and the
    flipped pairs keep their stored direction (``has_part``, ``used_by``)."""
    before = BUILT_UNDER[built_under]
    control = await _build_then_delete(tmp_path, before, before)
    change = AFTER[after]
    if change == "rules":
        orig_open = sys.modules[__name__]._open
        calls = []

        async def _open_then_break(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                _broken_rules(monkeypatch)
            return await orig_open(*args, **kwargs)

        monkeypatch.setattr(sys.modules[__name__], "_open", _open_then_break)
        change = before
    elif change is None:
        change = before
    out = await _build_then_delete(tmp_path, before, change)
    assert out["result"].status == "success"
    edges = control["graph"]["edges"]
    if "hook" in built_under:
        assert edges[("scale height", "thermosphere")]["source_id"] == "A-chunk-000"
    assert edges[("Alpha Unit", "Zeta System")]["keywords"] == "has_part"
    assert edges[("Beta Set", "Mid Tool")]["keywords"] == "used_by"
    assert out["graph"] == control["graph"]
    assert out["tracking"] == control["tracking"]


# ------------------------------------------------------------- mixed graphs


async def test_an_item_fed_by_a_legacy_chunk_and_a_stored_one(prompt_dir, tmp_path):
    """Document A extracted without the contract (no stored records), C and B
    with it; the inverse names are then dropped from the profile and B is
    deleted. The items A and C share are rebuilt from A's cached extraction
    (its keywords as written) and C's stored records (oriented as extraction
    oriented them): the graph of A and C alone."""

    async def run(docs, delete=None):
        path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
        rag = await _open(path, workspace, _llm(_flipped))
        try:
            await _insert(rag, "A")
        finally:
            await rag.finalize_storages()
        rag = await _open(path, workspace, _llm(_flipped), profile="contract.yml")
        try:
            await _insert(rag, *docs)
        finally:
            await rag.finalize_storages()
        rag = await _open(path, workspace, _llm(_flipped), profile="noinv.yml")
        try:
            if delete:
                assert (await rag.adelete_by_doc_id(delete)).status == "success"
            rows = await rag.text_chunks.get_by_ids(["A-chunk-000", "C-chunk-000"])
            assert [CHUNK_KG_RECORDS_FIELD in row for row in rows] == [False, True]
            return _normalized(_graph(rag))
        finally:
            await rag.finalize_storages()

    alone = await run(["C"])
    rebuilt = await run(["C", "B"], delete="B")
    edge = rebuilt["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "has_part,part_of"  # A as written, C oriented
    assert edge["source_id"].split("<SEP>") == ["A-chunk-000", "C-chunk-000"]
    assert rebuilt == alone


# ------------------------------------------------- a chunk extracted again without


async def test_a_chunk_extracted_again_without_the_contract_loses_its_records(
    prompt_dir, tmp_path
):
    """Document A extracted under the contract, then processed again with
    neither contract nor hook: its chunk keeps no records, and a deletion
    rebuilds it by the plain re-parse, as a graph built without the contract
    from the start."""
    from lightrag.base import DocStatus

    async def run(first_profile):
        path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
        rag = await _open(path, workspace, _llm(_flipped), profile=first_profile)
        try:
            await _insert(rag, "A", "B")
        finally:
            await rag.finalize_storages()
        rag = await _open(path, workspace, _llm(_flipped))
        try:
            status = await rag.doc_status.get_by_id("A")
            await rag.doc_status.upsert(
                {"A": {**status, "status": DocStatus.PROCESSING}}
            )
            await rag.apipeline_process_enqueue_documents()
            assert (await rag.doc_status.get_by_id("A"))["status"] == (
                DocStatus.PROCESSED
            )
            row = await rag.text_chunks.get_by_id("A-chunk-000")
            assert CHUNK_KG_RECORDS_FIELD not in row
            assert (await rag.adelete_by_doc_id("B")).status == "success"
            return _graph(rag)
        finally:
            await rag.finalize_storages()

    again = await run("contract.yml")
    plain = await run(None)
    edge = again["edges"][("Alpha Unit", "Zeta System")]
    assert edge["keywords"] == "part_of"  # the plain re-parse: as the LLM wrote it
    assert edge["source_id"] == "A-chunk-000"
    assert again == plain


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


class _ChunkStore:
    """Text-chunk storage stub with the PostgreSQL-only entry points."""

    def __init__(self, fail_write=False):
        self.rows, self.cleared, self.written = {}, [], {}
        self.fail_write = fail_write

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def upsert(self, rows):
        self.rows.update(rows)

    async def clear_kg_records(self, chunk_ids):
        self.cleared.append(list(chunk_ids))

    async def write_kg_records(self, chunk_id, records):
        if self.fail_write:
            raise RuntimeError("column write failed")
        self.written[chunk_id] = records


def _extract_config(validator):
    from lightrag.prompt import resolve_entity_extraction_prompt_profile

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        return _flipped("Report one. The Zeta System holds the Alpha Unit.")

    return {
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "llm_model_max_async": 1,
        "entity_extraction_use_json": True,
        "kg_extraction_validator": validator,
        "_entity_extraction_prompt_profile": resolve_entity_extraction_prompt_profile(
            {}, True
        ),
    }


CHUNKS = {
    "c1": {
        "tokens": 10,
        "content": "The Zeta System holds the Alpha Unit.",
        "full_doc_id": "doc-1",
        "chunk_order_index": 0,
    }
}


@pytest.mark.parametrize("validator", [None, kg_extraction_validator])
async def test_extraction_clears_earlier_records_then_stores_its_own(validator):
    """R6: before a document's chunks are extracted, records an earlier run
    left are cleared (a no-op except on PostgreSQL); with a hook (or the
    contract) each chunk's accepted records are written, without neither."""
    from lightrag.operate import extract_entities

    store = _ChunkStore()
    await extract_entities(
        CHUNKS, _extract_config(validator), text_chunks_storage=store
    )
    assert store.cleared == [["c1"]]
    if validator is None:
        assert store.written == {}
    else:
        records = store.written["c1"]
        assert records["format"] == 1
        assert {r["entity_name"] for r in records["entities"]} == {
            "Zeta System",
            "Alpha Unit",
        }
        [relation] = records["relations"]
        assert (relation["src_id"], relation["tgt_id"]) == ("Zeta System", "Alpha Unit")
        assert relation["reverse_keywords"] == relation["keywords"] == "part_of"


async def test_a_failed_records_write_fails_the_chunk():
    from lightrag.operate import extract_entities

    store = _ChunkStore(fail_write=True)
    with pytest.raises(RuntimeError, match="column write failed"):
        await extract_entities(
            CHUNKS, _extract_config(kg_extraction_validator), text_chunks_storage=store
        )


# ------------------------------------------------ kept summaries, the rollback policy


async def test_a_kept_summary_gets_the_document_once_after_its_re_ingest(
    prompt_dir, tmp_path
):
    """COLD3-09, R9: a keep_summaries deletion keeps the summaries a rebuild
    would need the LLM for and registers them; the hook is switched on before
    the deletion. After the document's re-ingest the kept items are rebuilt from
    the stored records, so the graph equals the one built before the deletion
    and no description holds the document's text twice."""
    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    kw = dict(profile="contract.yml", force_llm_summary_on_merge=2)
    rag = await _open(path, workspace, _llm(_flipped), **kw)
    try:
        await _insert(rag, "A", "B", "C")
        built = _graph(rag)
    finally:
        await rag.finalize_storages()
    rag = await _open(
        path, workspace, _llm(_flipped), validator=_OtherValidator(), **kw
    )
    try:
        result = await rag.adelete_by_doc_id("B", keep_summaries=True)
        assert result.status == "success"
        registry = json.loads((path / "kg_kept_summaries.json").read_text())
        assert "Alpha Unit" in registry["B"]["entities"]
    finally:
        await rag.finalize_storages()
    rag = await _open(path, workspace, _llm(_flipped), **kw)
    try:
        await _insert(rag, "B")
        again = _graph(rag)
        assert not (path / "kg_kept_summaries.json").exists() or not json.loads(
            (path / "kg_kept_summaries.json").read_text()
        )
    finally:
        await rag.finalize_storages()
    for data in (*again["nodes"].values(), *again["edges"].values()):
        assert data["description"].count(" B.") <= 1
    assert again["edges"][("Alpha Unit", "Zeta System")]["keywords"] == "has_part"

    def provenance_sorted(graph):
        # the re-ingested document's chunk now comes last
        for data in (*graph["nodes"].values(), *graph["edges"].values()):
            for field in ("source_id", "file_path"):
                data[field] = sorted(data[field].split("<SEP>"))
        return graph

    assert provenance_sorted(again) == provenance_sorted(built)


async def test_the_rollback_policy_reads_stored_records_without_the_cache(
    prompt_dir, tmp_path
):
    """Under ``rollback``, a chunk with stored records needs no extraction
    cache: its items are rebuilt from the records, not structurally repaired."""
    from lightrag.operate import rebuild_knowledge_from_chunks

    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    rag = await _open(path, workspace, _llm(_flipped), profile="contract.yml")
    try:
        await _insert(rag, "A", "B")
        await rag.llm_response_cache.drop()
        report = await rebuild_knowledge_from_chunks(
            entities_to_rebuild={"Alpha Unit": ["A-chunk-000"]},
            relationships_to_rebuild={("Alpha Unit", "Zeta System"): ["A-chunk-000"]},
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            text_chunks_storage=rag.text_chunks,
            llm_response_cache=rag.llm_response_cache,
            global_config=rag._build_global_config(),
            rebuild_policy="rollback",
        )
        graph = _graph(rag)
    finally:
        await rag.finalize_storages()
    assert not report.has_warnings
    edge = graph["edges"][("Alpha Unit", "Zeta System")]
    assert (edge["keywords"], edge["description"]) == ("has_part", "ZA A.")
    assert graph["nodes"]["Alpha Unit"]["description"] == "A A."


# ---------------------------------------------- a gleaning correction, also truncated

CORRECTION_TEXT = "Report one. Alpha Unit is a component of the Zeta System."
WRONG = ("Zeta System", "Alpha Unit", "part_of", "Zeta part of Alpha.")
CORRECTED = ("Alpha Unit", "Zeta System", "part_of", "Corrected: Alpha in Zeta.")


def _correction_llm(truncated: bool):
    units = [
        ("Alpha Unit", "Concept", "A unit."),
        ("Zeta System", "Concept", "A system."),
    ]

    def first(text):
        if text == CORRECTION_TEXT:
            return _reply(units, [WRONG])
        return _reply(
            units, [("Alpha Unit", "Zeta System", "part_of", "Alpha in Zeta (B).")]
        )

    def glean(text):
        if text != CORRECTION_TEXT:
            return NO_REPLY
        reply = _reply((), [CORRECTED])
        # A truncated reply is parsed and merged but never cached.
        return TruncatedResponse(reply) if truncated else reply

    return _llm(first, glean)


@pytest.mark.parametrize("truncated", [False, True], ids=["cached", "truncated"])
@pytest.mark.parametrize("after", ["same", "contract-off"])
async def test_a_gleaning_direction_correction_survives_a_deletion(
    prompt_dir, tmp_path, truncated, after
):
    """COLD3-05: the gleaning turn corrects the first pass's direction; a
    deletion of another document rebuilds the edge as extraction stored it,
    also when the gleaning reply was truncated (and so not cached)."""

    async def run(docs, delete):
        path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
        llm = _correction_llm(truncated)
        rag = await _open(path, workspace, llm, profile="contract.yml", gleaning=1)
        try:
            for doc_id, text in docs:
                await rag.ainsert(text, ids=[doc_id], file_paths=[f"{doc_id}.txt"])
        finally:
            await rag.finalize_storages()
        profile = "contract.yml" if after == "same" else None
        rag = await _open(path, workspace, llm, profile=profile, gleaning=1)
        try:
            if delete:
                assert (await rag.adelete_by_doc_id(delete)).status == "success"
            return _normalized(_graph(rag))
        finally:
            await rag.finalize_storages()

    alone = await run([("A", CORRECTION_TEXT)], None)
    rebuilt = await run(
        [("A", CORRECTION_TEXT), ("B", "Report two. Alpha Unit sits in Zeta.")], "B"
    )
    edge = alone["edges"][("Alpha Unit", "Zeta System")]
    assert (edge["keywords"], edge["description"]) == ("part_of", CORRECTED[3])
    assert rebuilt == alone


# ------------------------------------------- the name fold at extraction and rebuild

# The fold map extraction runs under: the key of "Soil Moisture" spells
# "Soil moisture" (the best-connected member), "Water Cycle" "Water cycle".
FOLD_ROWS = [
    ("Soil moisture", "concept", 10),
    ("soil moisture", "concept", 1),
    ("Water cycle", "concept", 3),
]
# The same graph read again after the degrees changed: the key's canonical is
# now the lower-case spelling.
MOVED_ROWS = [
    ("Soil moisture", "concept", 1),
    ("soil moisture", "concept", 10),
    ("Water cycle", "concept", 3),
]
FOLD_DOCS = {
    "A": "Report one. Soil moisture is part of the water cycle.",
    "B": "Report two. Soil moisture is part of the water cycle.",
    "C": "Report three. Soil moisture is part of the water cycle.",
}
FOLD_TAGS = {"A": "alpha-doc", "B": "bravo-doc", "C": "charlie-doc"}


def _fold_names_llm(case):
    """``plain``: the LLM writes ``Soil Moisture`` and ``Water Cycle``, both
    folded. ``flip``: it writes ``soil moisture``, which sorts after ``Water
    Cycle`` while the folded pair sorts the other way round. ``join``: it also
    writes ``soil moistures`` and relates the two spellings, which the fold
    joins into one name."""

    def first(text):
        tag = FOLD_TAGS[_tag(text)]
        soil = "soil moisture" if case == "flip" else "Soil Moisture"
        entities = [
            (soil, "Concept", f"SM {tag}."),
            ("Water Cycle", "Concept", f"WC {tag}."),
        ]
        relations = [(soil, "Water Cycle", "part_of", f"SM in WC {tag}.")]
        if case == "join":
            entities.append(("soil moistures", "Concept", f"SMs {tag}."))
            relations.append(
                ("Soil Moisture", "soil moistures", "same_as", f"SM is SMs {tag}.")
            )
        return _reply(entities, relations)

    return _llm(first)


def _arm(state):
    """The fold map in force: ``FOLD_ROWS`` at extraction; at rebuild one of
    the states a restart can leave it in."""
    entity_name_fold.disarm()
    if state in ("extraction", "same"):
        entity_name_fold.load_fold_map(FOLD_ROWS)
    elif state == "moved":
        entity_name_fold.load_fold_map(MOVED_ROWS)
    elif state in ("aliased", "extraction-aliased"):
        entity_name_fold.load_aliases([("Soil Moisture", "Moisture of soil")])
        entity_name_fold.load_fold_map(FOLD_ROWS)
    # "off" and "unarmed": no map


def _never_armed(rag):
    """ENTITY_NAME_FOLD on, but the startup map build has not finished within
    its wait: the deletion logs the error and goes on unfolded."""
    import asyncio
    import time

    rag._fold_map_task = asyncio.get_running_loop().create_future()
    rag._fold_map_deadline = time.monotonic()
    return rag._fold_map_task


async def _fold_rebuild(
    tmp_path, llm, inserts, delete, state, legacy=(), extraction="extraction"
):
    """Insert ``inserts`` (those in ``legacy`` without the contract, the others
    with it) under the map ``extraction``, reopen under ``state`` and delete
    ``delete``: the graph, normalized."""
    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    try:
        for doc_id in inserts:
            profile = None if doc_id in legacy else "contract.yml"
            rag = await _open(path, workspace, llm, profile=profile)
            _arm(extraction)  # opening the storages disarms the map
            try:
                await rag.ainsert(
                    FOLD_DOCS[doc_id], ids=[doc_id], file_paths=[f"{doc_id}.txt"]
                )
            finally:
                await rag.finalize_storages()
        rag = await _open(path, workspace, llm, profile="contract.yml")
        _arm(state)
        pending = _never_armed(rag) if state == "unarmed" else None
        try:
            if delete:
                assert (await rag.adelete_by_doc_id(delete)).status == "success"
            graph = _normalized(_graph(rag))
            for node in graph["nodes"].values():
                node["description"] = "<SEP>".join(
                    sorted(node["description"].split("<SEP>"))
                )
            return graph
        finally:
            if pending is not None:
                pending.cancel()
            await rag.finalize_storages()
    finally:
        entity_name_fold.disarm()


@pytest.mark.parametrize("state", ["same", "off", "unarmed", "moved", "aliased"])
@pytest.mark.parametrize("case", ["plain", "flip", "join"])
async def test_a_deletion_rebuilds_folded_names_whatever_the_fold_is_now(
    prompt_dir, tmp_path, case, state
):
    """COLD4-02: the records are stored under the names the merge used (after
    ENTITY_NAME_FOLD), so a deletion rebuilds the folded node and its edge from
    them whatever the fold map is at rebuild time: the same, switched off, not
    armed within its wait, a key's canonical moved, or an alias added. The
    graph is that of the surviving document alone, with nothing of the deleted
    one. A pair the fold flips keeps the direction the merge gave it; a pair
    the fold joins stays dropped."""
    llm = _fold_names_llm(case)
    alone = await _fold_rebuild(tmp_path, llm, ["A"], None, "same")
    rebuilt = await _fold_rebuild(tmp_path, llm, ["A", "B"], "B", state)

    assert set(alone["nodes"]) == {"Soil moisture", "Water cycle"}
    assert set(alone["edges"]) == {("Soil moisture", "Water cycle")}
    edge = alone["edges"][("Soil moisture", "Water cycle")]
    assert (edge["keywords"], edge["source_id"]) == ("part_of", "A-chunk-000")
    if case == "join":
        node = alone["nodes"]["Soil moisture"]
        assert node["description"] == "SM alpha-doc.<SEP>SMs alpha-doc."
    assert "bravo-doc" not in repr(rebuilt)
    assert rebuilt == alone


async def test_a_deletion_after_an_alias_is_removed(prompt_dir, tmp_path):
    """An alias file in force at extraction joins ``Soil Moisture`` into
    ``Moisture of soil``; the deletion runs after it was removed. The records
    name the node the merge wrote, so the rebuild still finds them."""
    llm = _fold_names_llm("plain")
    alone = await _fold_rebuild(
        tmp_path, llm, ["A"], None, "aliased", extraction="extraction-aliased"
    )
    rebuilt = await _fold_rebuild(
        tmp_path, llm, ["A", "B"], "B", "same", extraction="extraction-aliased"
    )
    assert set(alone["edges"]) == {("Moisture of soil", "Water cycle")}
    assert "bravo-doc" not in repr(rebuilt)
    assert rebuilt == alone


async def test_the_stored_records_carry_the_names_the_merge_used(prompt_dir, tmp_path):
    """The records stored with a chunk are the ones the merge consumed: after
    the fold, the joined relation absent."""
    path, workspace = tmp_path / f"rag_{uuid4().hex}", f"kgc_{uuid4().hex}"
    rag = await _open(path, workspace, _fold_names_llm("join"), profile="contract.yml")
    _arm("extraction")
    try:
        await rag.ainsert(FOLD_DOCS["A"], ids=["A"], file_paths=["A.txt"])
        row = await rag.text_chunks.get_by_id("A-chunk-000")
    finally:
        entity_name_fold.disarm()
        await rag.finalize_storages()
    records = row[CHUNK_KG_RECORDS_FIELD]
    assert sorted(r["entity_name"] for r in records["entities"]) == [
        "Soil moisture",
        "Soil moisture",
        "Water cycle",
    ]
    assert [(r["src_id"], r["tgt_id"]) for r in records["relations"]] == [
        ("Soil moisture", "Water cycle")
    ]


@pytest.mark.parametrize("state", ["same", "off"])
async def test_an_item_fed_by_a_legacy_chunk_and_a_folded_stored_one(
    prompt_dir, tmp_path, state
):
    """Document A extracted without the contract (no records), C and B with
    it, all under the fold; B is deleted. Each chunk contributes its own way:
    C's stored records under the names the merge used, whatever the fold is
    now; A's re-parsed cache under the LLM's spelling, found through the fold
    map of the rebuild as on the base branch, so with the fold off A's text is
    not found (its chunk id stays). B's text is gone either way."""
    llm = _fold_names_llm("plain")
    alone = await _fold_rebuild(tmp_path, llm, ["A", "C"], None, "same", legacy="A")
    rebuilt = await _fold_rebuild(
        tmp_path, llm, ["A", "C", "B"], "B", state, legacy="A"
    )
    assert "bravo-doc" not in repr(rebuilt)
    if state == "same":
        assert rebuilt == alone
        return
    node = rebuilt["nodes"]["Soil moisture"]
    assert node["description"] == "SM charlie-doc."
    assert node["source_id"].split("<SEP>") == ["A-chunk-000", "C-chunk-000"]
    edge = rebuilt["edges"][("Soil moisture", "Water cycle")]
    assert (edge["keywords"], edge["description"]) == (
        "part_of",
        "SM in WC charlie-doc.",
    )


def _pass_through(chunk_key, chunk_text, maybe_nodes, maybe_edges):
    return maybe_nodes, maybe_edges


async def test_extraction_stores_exactly_what_it_returns_for_the_merge():
    """Under the fold, the records written for a chunk are the chunk's result
    as extract_entities returns it to the merge: names folded, the order the
    fold flips kept as the folded pair, the relation the fold joins absent.
    The merge and the rebuild orient the same records the same way."""
    from lightrag.kg_contract import accepted_records, oriented_record
    from lightrag.operate import extract_entities

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        return _reply(
            [
                ("Soil Moisture", "Concept", "SM."),
                ("soil moistures", "Concept", "SMs."),
                ("Water Cycle", "Concept", "WC."),
            ],
            [
                ("soil moisture", "Water Cycle", "part_of", "SM in WC."),
                ("Soil Moisture", "soil moistures", "same_as", "joined."),
            ],
        )

    config = _extract_config(_pass_through)
    config["role_llm_funcs"] = {r: llm for r in ("extract", "keyword", "query", "vlm")}
    store = _ChunkStore()
    entity_name_fold.load_fold_map(FOLD_ROWS)
    try:
        [(nodes, edges)] = await extract_entities(
            CHUNKS, config, text_chunks_storage=store
        )
    finally:
        entity_name_fold.disarm()
    stored = store.written["c1"]
    assert stored == accepted_records(nodes, edges)
    assert sorted(nodes) == ["Soil moisture", "Water cycle"]
    [relation] = stored["relations"]
    assert (relation["src_id"], relation["tgt_id"]) == ("Soil moisture", "Water cycle")
    assert oriented_record(relation)["keywords"] == "part_of"
