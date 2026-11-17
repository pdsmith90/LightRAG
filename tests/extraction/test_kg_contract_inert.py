"""Extraction contract: with no contract key in the profile, no validator and no
new setting, the prompts sent and the graph written are those of the base branch.

The comparison is against digests recorded from the base branch's own tree
(``git archive zlightrag``, this module's ``_record`` run there with that tree
first on ``sys.path``), not against templates or code of this branch, so a
change to a base template or to a write path shows up here. The workload covers
both extraction modes, no profile and a profile without the contract keys, one gleaning
turn whose relations repeat the first pass's pairs reversed (same and other
type), keywords off any list (``Part_Of``, a free phrase), a glued name, a
second document, its deletion (the rebuild from cached extractions) and the
graph after each step. All names are invented.

To re-record after an intended change on the base branch, run
``python tests/extraction/test_kg_contract_inert.py`` with the base branch's
tree first on PYTHONPATH and paste the printed digests.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
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


def _input_text(prompt: str) -> str:
    marker = "---Input Text---\n```\n"
    start = prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    return prompt[start : prompt.find("\n```", start)]


def _reply(relations, use_json: bool, tag: str, entities=True) -> str:
    ents = ENTITIES if entities else []
    if use_json:
        return json.dumps(
            {
                "entities": [
                    {"name": n, "type": t, "description": f"{n} ({tag})."}
                    for n, t in ents
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
    rows = [f"entity{d}{n}{d}{t}{d}{n} ({tag})." for n, t in ents]
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

    replies = [_reply(FIRST, use_json, "one"), _reply(GLEAN, use_json, "glean", False)]
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


async def _graph_workload(
    work_dir: str, use_json: bool, addon: dict, workspace: str
) -> dict:
    """Insert two documents and delete the second: every LLM prompt, every edge
    and node write, and the graph after each step."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer

    prompts = []

    async def llm(prompt, system_prompt=None, history_messages=None, **_):
        prompts.append((prompt, system_prompt, bool(history_messages)))
        text = _input_text(prompt)
        tag = "two" if "again" in text else "one"
        if history_messages:
            first = _input_text(history_messages[0]["content"])
            tag = "two" if "again" in first else "one"
            return _reply(GLEAN, use_json, f"glean {tag}", False)
        if text:
            return _reply(FIRST, use_json, tag)
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
    )
    await rag.initialize_storages()
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
        steps["inserted"] = _graph(rag)
        result = await rag.adelete_by_doc_id("doc-2")
        steps["delete_status"] = result.status
        steps["deleted"] = _graph(rag)
    finally:
        await rag.finalize_storages()
    return {"prompts": sorted(prompts), "writes": sorted(writes), "steps": steps}


CASES = [
    (use_json, profile) for use_json in (True, False) for profile in (None, "plain.yml")
]


def _case_id(use_json, profile):
    return f"{'json' if use_json else 'text'}-{profile or 'default'}"


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


async def _run_case(work_dir: str, use_json: bool, profile) -> dict:
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
    return {
        "extraction": _digest(await _extraction_prompts(use_json, addon)),
        "graph": _digest(
            await _graph_workload(
                os.path.join(work_dir, "g"),
                use_json,
                addon,
                f"inert_{_case_id(use_json, profile)}".replace(".", "_").replace(
                    "-", "_"
                ),
            )
        ),
    }


# Recorded from zlightrag a2bd9d148 (see the module docstring).
GOLDEN = {
    "json-default": {
        "extraction": "7c6199f0c7d100a890fdfb99dce17ac4dd5bad8e2220b4fa47ba0a72992e191f",
        "graph": "a99a5862c1abf1c03d0bc02201372728e61d74823b84273366882fc5011958cb",
    },
    "json-plain.yml": {
        "extraction": "6d20034ad7e5505250c1781e1ed3f7326ad684ef00ab0eace6861cd386d4528f",
        "graph": "9e7bcb8312ab6b5e917f47a4057ae028e9e78f47151793e911677a6bca632e1d",
    },
    "text-default": {
        "extraction": "658256505f16bf251db04ae635e2625cdf9c275e5e3a0e34f26414322d9b210d",
        "graph": "f5c738365cea843e81f4d38bef36056da4d455f0aba97936e7f6c93ab2cbf0dd",
    },
    "text-plain.yml": {
        "extraction": "7e3ee923ee1186dd409b8448ace027a43e0dfd52d1e07266d412e9667426a734",
        "graph": "5a3ffe2e01b7e361112df3f222395c4627c4e78f94682a7e7aa2cd25d820c309",
    },
}


@pytest.mark.parametrize("use_json,profile", CASES, ids=[_case_id(*c) for c in CASES])
async def test_prompts_and_graph_writes_equal_the_base_branch(
    tmp_path, monkeypatch, use_json, profile
):
    monkeypatch.setenv("PROMPT_DIR", "")
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "")
    assert (
        await _run_case(str(tmp_path), use_json, profile)
        == GOLDEN[_case_id(use_json, profile)]
    )


def _record() -> dict:
    out = {}
    for use_json, profile in CASES:
        work_dir = tempfile.mkdtemp(dir=os.environ.get("INERT_SCRATCH"))
        try:
            out[_case_id(use_json, profile)] = asyncio.run(
                _run_case(work_dir, use_json, profile)
            )
        finally:
            shutil.rmtree(work_dir)
    return out


if __name__ == "__main__":
    json.dump(_record(), sys.stdout, indent=4, sort_keys=True)
    print()
