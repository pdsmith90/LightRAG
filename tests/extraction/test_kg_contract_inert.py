"""Extraction contract: with no contract key in the profile, no validator and no
new setting, the prompts sent and the graph written are those of the base branch.

The comparison is against digests recorded from the base branch's own tree
(``git archive zlightrag``, this module's ``_record`` run there with that tree
first on ``sys.path``), not against templates or code of this branch, so a
change to a base template or to a write path shows up here. The workload covers
both extraction modes, no profile and a profile without the contract keys, the
entity name fold unarmed and armed (with a document that spells a name the fold
maps), one gleaning turn that re-outputs entities and relations of the first
pass with a longer, a shorter and an equal-length description (the merge's
"keep the longer description" branch), repeats the first pass's pairs reversed
(same and other type) and adds new names, keywords off any list (``Part_Of``, a
free phrase), a glued name, three documents sharing entities, a deletion and a
kept-summary deletion (the rebuilds from cached extractions), the re-ingest of
the second one (with the kept-summary rebuild after it) and the graph after
each step. All names are invented. The graph workload runs on a clock of its
own (``_one_second_clock``), because the order of a name's descriptions follows
the second each was extracted in.

To re-record after an intended change on the base branch, run
``python tests/extraction/test_kg_contract_inert.py`` with the base branch's
tree first on PYTHONPATH and paste the printed digests.
"""

import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.offline

PLAIN_SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "samples"
    / "entity_type_prompt.sample.yml"
)

PLAIN_GUIDANCE = (
    "- Concept: an idea, quantity or system\n"
    "- Method: a technique or tool\n"
    "- Data: a data set\n"
    "- Location: a place or layer"
)

DOC_1 = (
    "Zeta System holds the Alpha Unit. Mid Tool uses the Beta Set. "
    "The scale height of the thermosphere is large."
)
DOC_2 = "Zeta System holds the Alpha Unit again. Mid Tool uses the Beta Set again."
# Spells Zeta System the way the armed fold maps (FOLD_ROWS).
DOC_3 = "ZETA SYSTEM holds the Alpha Unit, third report."
FOLD_ROWS = [("Zeta System", "concept", 5), ("Alpha Unit", "concept", 3)]

ENTITIES = [
    ("Zeta System", "Concept"),
    ("Alpha Unit", "Concept"),
    ("Mid Tool", "Method"),
    ("Beta Set", "Data"),
    ("ScaleHeight", "Concept"),
    ("thermosphere", "Location"),
]
FIRST = [
    ("Zeta System", "Alpha Unit", "part_of", "Zeta part of Alpha"),
    ("Mid Tool", "Beta Set", "Part_Of, member of", "Mid in Beta"),
    ("ScaleHeight", "thermosphere", "located_in", "Scale height in the layer"),
]
GLEAN = [
    ("Alpha Unit", "Zeta System", "part_of", "Corrected: Alpha Unit is part of Zeta"),
    ("Beta Set", "Mid Tool", "uses", "Beta Set is used with Mid Tool"),
]


def _glean_same_keys(tag: str):
    """Gleaning records for the first pass's own names and pairs: a longer, a
    shorter and an equal-length description (FIRST's are ``f"{d} ({tag})."``),
    and a relation between new names."""
    entities = [
        (
            "Zeta System",
            "Concept",
            f"Zeta System, described at much greater length ({tag}).",
        ),
        ("Alpha Unit", "Concept", "AU."),
        ("Mid Tool", "Method", f"Mid Tool [{tag}]."),
        ("Gamma Probe", "Method", f"A new probe ({tag})."),
        # declared, so no rebuild has to make it from its edges (whose file
        # paths the base branch collects in a set, in hash order)
        ("ALPHA UNIT", "Concept", f"Upper spelling of the unit ({tag})."),
    ]
    relations = [
        (
            "Zeta System",
            "Alpha Unit",
            "part_of",
            f"Zeta part of Alpha, restated at length ({tag})",
        ),
        ("Mid Tool", "Beta Set", "uses", "Mid"),
        ("ScaleHeight", "thermosphere", "located_in", "Scale height in the layes"),
        # the armed fold maps this spelling to Alpha Unit
        (
            "Gamma Probe",
            "ALPHA UNIT",
            "measures",
            f"The probe measures the unit ({tag})",
        ),
    ]
    return entities, relations


def _input_text(prompt: str) -> str:
    marker = "---Input Text---\n```\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    return prompt[start : prompt.find("\n```", start)]


def _reply(relations, use_json: bool, tag: str, entities=True) -> str:
    if entities is True:
        ents = [(n, t, f"{n} ({tag}).") for n, t in ENTITIES]
    else:
        ents = list(entities or [])
    if use_json:
        return json.dumps(
            {
                "entities": [
                    {"name": n, "type": t, "description": d} for n, t, d in ents
                ],
                "relationships": [
                    {
                        "source": s,
                        "target": o,
                        "keywords": k,
                        "description": f"{d} ({tag}).",
                    }
                    for s, o, k, d in relations
                ],
            }
        )
    d = "<|#|>"
    rows = [f"entity{d}{n}{d}{t}{d}{x}" for n, t, x in ents]
    rows += [f"relation{d}{s}{d}{o}{d}{k}{d}{x} ({tag})." for s, o, k, x in relations]
    return "\n".join(rows + ["<|COMPLETE|>"])


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


def _plain(records: dict) -> dict:
    return {
        "|".join(k) if isinstance(k, tuple) else k: [
            {f: v for f, v in r.items() if f != "timestamp"} for r in recs
        ]
        for k, recs in sorted(records.items())
    }


async def _extraction_prompts(use_json: bool, addon: dict) -> dict:
    """The prompts one chunk's extraction sends, and its parsed results."""
    from lightrag.operate import extract_entities
    from lightrag.utils import Tokenizer

    glean_entities, glean_relations = _glean_same_keys("one")
    replies = [
        _reply(FIRST, use_json, "one"),
        _reply(GLEAN + glean_relations, use_json, "glean", glean_entities),
    ]
    llm = AsyncMock(side_effect=replies)
    global_config = {
        "llm_model_func": llm,
        "role_llm_funcs": {r: llm for r in ("extract", "keyword", "query", "vlm")},
        "entity_extract_max_gleaning": 1,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": addon,
        "tokenizer": Tokenizer("characters", _Characters()),
        "llm_model_max_async": 1,
        "entity_extraction_use_json": use_json,
    }
    chunks = {
        "chunk-1": {
            "tokens": len(DOC_1),
            "content": DOC_1,
            "full_doc_id": "doc-1",
            "chunk_order_index": 0,
        }
    }
    results = await extract_entities(chunks=chunks, global_config=global_config)
    return {
        "sent": [
            (c.args[0], c.kwargs.get("system_prompt")) for c in llm.call_args_list
        ],
        "results": [{"nodes": _plain(n), "edges": _plain(e)} for n, e in results],
    }


def _strip(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ("created_at", "timestamp")}


def _graph(rag) -> dict:
    graph = rag.chunk_entity_relation_graph._graph
    return {
        "nodes": sorted(
            (n, json.dumps(_strip(dict(d)), sort_keys=True))
            for n, d in graph.nodes(data=True)
        ),
        "edges": sorted(
            (*sorted((u, v)), json.dumps(_strip(dict(d)), sort_keys=True))
            for u, v, d in graph.edges(data=True)
        ),
    }


def _doc_tag(text: str) -> str:
    if "third" in text:
        return "three"
    return "two" if "again" in text else "one"


def _first_reply(text: str, use_json: bool, tag: str) -> str:
    if tag == "three":
        return _reply(
            [("ZETA SYSTEM", "Alpha Unit", "part_of", "Upper Zeta holds Alpha")],
            use_json,
            tag,
            [
                ("ZETA SYSTEM", "Concept", "The system, upper spelling (three)."),
                ("Alpha Unit", "Concept", "Alpha Unit (three)."),
            ],
        )
    return _reply(FIRST, use_json, tag)


async def _graph_workload(
    work_dir: str, use_json: bool, addon: dict, workspace: str, fold: bool = False
) -> dict:
    """Insert three documents, delete the second, delete the third keeping
    summaries and re-insert it: every LLM prompt, every edge and node write,
    the graph after each step, and the KV and vector records at the end."""
    from lightrag import LightRAG, entity_name_fold
    from lightrag.utils import EmbeddingFunc, Tokenizer

    prompts = []

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        prompts.append((prompt, system_prompt, bool(history_messages)))
        text = _input_text(prompt)
        tag = _doc_tag(text)
        if history_messages:
            first = _input_text(history_messages[0]["content"])
            tag = _doc_tag(first)
            if tag == "two":
                return _reply(GLEAN, use_json, f"glean {tag}", False)
            glean_entities, glean_relations = _glean_same_keys(tag)
            return _reply(
                GLEAN + glean_relations, use_json, f"glean {tag}", glean_entities
            )
        if text:
            return _first_reply(text, use_json, tag)
        return "A summary."

    rag = LightRAG(
        working_dir=os.path.join(work_dir, "rag"),
        workspace=workspace,
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=1,
        entity_extraction_use_json=use_json,
        chunk_token_size=4000,
        chunk_overlap_token_size=0,
        max_parallel_insert=1,
        addon_params=addon,
        # Summaries from the second fragment on, so the kept-summary deletion
        # keeps some and the re-ingest rebuilds them.
        force_llm_summary_on_merge=2,
    )
    await rag.initialize_storages()
    if fold:
        entity_name_fold.load_fold_map(FOLD_ROWS)
    writes = []
    graph = rag.chunk_entity_relation_graph
    upsert_edge, upsert_node = graph.upsert_edge, graph.upsert_node

    async def recording_upsert_edge(src, tgt, edge_data):
        writes.append(("edge", src, tgt, json.dumps(_strip(edge_data), sort_keys=True)))
        return await upsert_edge(src, tgt, edge_data)

    async def recording_upsert_node(name, node_data):
        writes.append(("node", name, json.dumps(_strip(node_data), sort_keys=True)))
        return await upsert_node(name, node_data)

    graph.upsert_edge, graph.upsert_node = recording_upsert_edge, recording_upsert_node
    steps = {}
    try:
        await rag.ainsert(DOC_1, ids=["doc-1"], file_paths=["one.txt"])
        await rag.ainsert(DOC_2, ids=["doc-2"], file_paths=["two.txt"])
        await rag.ainsert(DOC_3, ids=["doc-3"], file_paths=["three.txt"])
        steps["inserted"] = _graph(rag)
        result = await rag.adelete_by_doc_id("doc-2")
        steps["delete_status"] = (result.status, result.message)
        steps["deleted"] = _graph(rag)
        result = await rag.adelete_by_doc_id("doc-3", keep_summaries=True)
        steps["kept_status"] = (result.status, result.message)
        steps["kept"] = _graph(rag)
        kept = json.loads(Path(rag._kept_summaries_path()).read_text())
        steps["kept_registry"] = {
            doc_id: {k: v for k, v in entry.items() if k != "kept_at"}
            for doc_id, entry in kept.items()
        }
        await rag.ainsert(DOC_3, ids=["doc-3"], file_paths=["three.txt"])
        steps["reinserted"] = _graph(rag)
        steps["records"] = await _records(rag)
    finally:
        entity_name_fold.disarm()
        await rag.finalize_storages()
    return {"prompts": sorted(prompts), "writes": sorted(writes), "steps": steps}


_VOLATILE = {
    "created_at",
    "timestamp",
    "create_time",
    "update_time",
    "_id",
    "__created_at__",
    "__write_seq__",
}


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


async def _records(rag) -> dict:
    """Every KV row and vector record the workload left, without timestamps."""
    out = {}
    for name in (
        "text_chunks",
        "llm_response_cache",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
    ):
        data = getattr(rag, name)._data
        out[name] = _stable({key: dict(data[key]) for key in sorted(data.keys())})
    for name in ("entities_vdb", "relationships_vdb", "chunks_vdb"):
        client = await getattr(rag, name)._get_client()
        out[name] = sorted(
            json.dumps(
                _stable({k: v for k, v in row.items() if k != "__vector__"}),
                sort_keys=True,
            )
            for row in client._NanoVectorDB__storage["data"]
        )
    return out


CASES = [
    (use_json, profile, fold)
    for use_json in (True, False)
    for profile in (None, "plain.yml")
    for fold in (False, True)
]


def _case_id(use_json, profile, fold=False):
    return f"{'json' if use_json else 'text'}-{profile or 'default'}" + (
        "-fold" if fold else ""
    )


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# A whole second, and what the workload's clock adds at every reading.
_CLOCK_START = 1_700_000_000
_CLOCK_STEP = 1e-6


@contextlib.contextmanager
def _one_second_clock():
    """Run the graph workload inside one clock second.

    The merge lists a name's descriptions by the whole second each was extracted
    in, and by length within a second. That order is part of the summary prompts,
    so of the cache keys and of the digest. A run that crosses a second boundary
    between two extractions therefore writes other prompts than one that does
    not, on the base branch as on this one, and a digest recorded on a fast
    machine fails now and then on a slow one. The workload brings its own clock
    instead: it still advances, by a microsecond at every reading.
    """
    real, readings = time.time, itertools.count()
    time.time = lambda: _CLOCK_START + next(readings) * _CLOCK_STEP
    try:
        yield
        assert next(readings) * _CLOCK_STEP < 0.5, "the workload left its second"
    finally:
        time.time = real


async def _run_case(work_dir: str, use_json: bool, profile, fold: bool) -> dict:
    prompt_dir = Path(work_dir) / "prompts"
    (prompt_dir / "entity_type").mkdir(parents=True, exist_ok=True)
    # The plain sample with its own guidance (the sample's equals the default).
    plain = yaml.safe_load(PLAIN_SAMPLE.read_text(encoding="utf-8"))
    plain["entity_types_guidance"] = PLAIN_GUIDANCE
    (prompt_dir / "entity_type" / "plain.yml").write_text(
        json.dumps(plain), encoding="utf-8"
    )
    os.environ["PROMPT_DIR"] = str(prompt_dir)
    os.environ["MAX_EXTRACT_INPUT_TOKENS"] = "9999999"
    addon = {"entity_type_prompt_file": profile} if profile else {}
    from lightrag import entity_name_fold

    if fold:
        entity_name_fold.load_fold_map(FOLD_ROWS)
    try:
        extraction = await _extraction_prompts(use_json, addon)
    finally:
        entity_name_fold.disarm()
    with _one_second_clock():
        graph = await _graph_workload(
            os.path.join(work_dir, "g"),
            use_json,
            addon,
            f"inert_{_case_id(use_json, profile, fold)}".replace(".", "_").replace(
                "-", "_"
            ),
            fold,
        )
    return {"extraction": _digest(extraction), "graph": _digest(graph)}


# Recorded from zlightrag a2bd9d148 (see the module docstring); the graph digests
# re-recorded 2026-11-27 after the merge of upstream 47038cf16, whose only effect
# on this workload is that an edge's weight counts once per chunk (upstream's
# rule on insert and in the rebuild): the prompts, writes and records of all eight
# cases equal those of the pre-merge tree once the weights are masked.
GOLDEN = {
    "json-default": {
        "extraction": "5044ed3760d32cd045b980574313583446bbf674922bf72ff05b99c6067f61a0",
        "graph": "795011a7e8d5e084328b3e729a6b4c43c56f3a1b78901a74a0395a56e676625b",
    },
    "json-default-fold": {
        "extraction": "fb26b0c232fce8c84a8798176f62e6d266a6ffb5808c8a9bcd3964a5a8fae986",
        "graph": "2f32510fad86e63e4178e6b19a0d96db2ac59a2c9fb81132813df33d94321655",
    },
    "json-plain.yml": {
        "extraction": "00a5d6ddd90686debe0c5d5b641b07a636743f283f4556f076c6efc67fe9308d",
        "graph": "a5d22b2e8d95dacec8a07cd2c07dc233f8eb98d5f881d5e861f933acd86f4364",
    },
    "json-plain.yml-fold": {
        "extraction": "1e7676a678847dec244ae3b90c2c80e1ff1d049b9ae2eda8597afec26d3feda1",
        "graph": "2a4633090d8141536885eb939a5614bd15588774d1a6613fdf4fd45a11fac99e",
    },
    "text-default": {
        "extraction": "26ead7a411f34f5c7cd815a507c8436519f5e770b2445d5f0c4c253e417e204c",
        "graph": "7be909ec93a171c15d88f45a2eb83c11cd7b97dc5e3ef70d2d6c496313206e88",
    },
    "text-default-fold": {
        "extraction": "5a5cf0bf19d22b995069bc8bba6942b81ea81ba530a8514ee29fa9fc32f5a89d",
        "graph": "9bf4f4f39388df8a7626e792745ee03cad49fa4a18082b5b9b2282b34cd6e390",
    },
    "text-plain.yml": {
        "extraction": "59ad105fa8307280463cebd4beed5b2982be056193bcb2c0a9314bcb42b4a6d3",
        "graph": "cc92b07bb59706b7fd8ea181b3bd3923d5ff087aec0e4d3e9c7e47537b69fc76",
    },
    "text-plain.yml-fold": {
        "extraction": "14ce7869473a5c729608285ea0841d0f8b754527a9ed79b0c127244836fbb605",
        "graph": "92dae5b63d954d7abbec52b12c559a4fa5706f2b4998b6ace18befbd19f67a70",
    },
}


@pytest.mark.parametrize(
    "use_json,profile,fold", CASES, ids=[_case_id(*c) for c in CASES]
)
async def test_prompts_and_graph_writes_equal_the_base_branch(
    tmp_path, monkeypatch, use_json, profile, fold
):
    monkeypatch.setenv("PROMPT_DIR", "")
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "")
    assert (
        await _run_case(str(tmp_path), use_json, profile, fold)
        == GOLDEN[_case_id(use_json, profile, fold)]
    )


async def test_the_graph_digest_does_not_depend_on_the_wall_clock(
    tmp_path, monkeypatch
):
    """On a slow machine two extractions fall into different clock seconds, which
    reorders the descriptions the merge hands to a summary. The workload runs on
    its own clock, so a wall clock that jumps 0.7 s at every reading gives the
    digest of one that stands still."""
    monkeypatch.setenv("PROMPT_DIR", "")
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "")
    start, readings = time.time(), itertools.count(1)
    monkeypatch.setattr(time, "time", lambda: start + 0.7 * next(readings))
    assert (
        await _run_case(str(tmp_path), False, "plain.yml", True)
        == GOLDEN["text-plain.yml-fold"]
    )


def _record() -> dict:
    out = {}
    for use_json, profile, fold in CASES:
        work_dir = tempfile.mkdtemp(dir=os.environ.get("INERT_SCRATCH"))
        try:
            out[_case_id(use_json, profile, fold)] = asyncio.run(
                _run_case(work_dir, use_json, profile, fold)
            )
        finally:
            shutil.rmtree(work_dir)
    return out


if __name__ == "__main__":
    json.dump(_record(), sys.stdout, indent=4, sort_keys=True)
    print()
