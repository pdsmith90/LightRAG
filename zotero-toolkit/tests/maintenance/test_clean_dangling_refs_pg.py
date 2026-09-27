"""Integration test: clean_dangling_refs.py against PostgreSQL + Apache AGE.

The workspace state is written through lightrag-hku 1.5.7's own PostgreSQL storages
(KV maps, text chunks, doc status and the AGE graph), then one document is removed
behind LightRAG's back, the way an out-of-band deletion leaves it: the entities and
relations it shared with a surviving document still cite its chunks. The two vector
tables are created from v1.5.7's DDL with the embedding column typed REAL[] instead
of VECTOR, so the database needs Apache AGE but not pgvector; the tool never reads
that column.

Opt in by pointing CLEAN_DANGLING_TEST_DSN at a SCRATCH database with Apache AGE:

    CLEAN_DANGLING_TEST_DSN=postgresql://user:password@127.0.0.1:5432/dbname

The tests create uniquely named workspaces and remove their rows, graphs and backup
tables afterwards. Requires `pip install lightrag-hku==1.5.7`.
"""

import asyncio
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlsplit

import pytest

asyncpg = pytest.importorskip("asyncpg")

DSN = os.environ.get("CLEAN_DANGLING_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN,
    reason="set CLEAN_DANGLING_TEST_DSN to a scratch PostgreSQL + Apache AGE database",
)

TOOL = Path(__file__).resolve().parents[2] / "maintenance" / "clean_dangling_refs.py"
spec = importlib.util.spec_from_file_location("clean_dangling_refs", TOOL)
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)

SEP = "<SEP>"
MODEL, DIM = "toolkit-test-embed", 8
KV_TABLES = (
    "lightrag_doc_full",
    "lightrag_doc_chunks",
    "lightrag_doc_status",
    "lightrag_entity_chunks",
    "lightrag_relation_chunks",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_llm_cache",
)
DOCS = {
    "a": ("paper_a.md", ["Alpha opening paragraph.", "Alpha closing paragraph."]),
    "b": ("paper_b.md", ["Beta opening paragraph.", "Beta closing paragraph."]),
}
EXPECTED_DANGLING = {
    "lightrag_vdb_entity": 2,
    "lightrag_vdb_relation": 2,
    "lightrag_entity_chunks": 2,
    "lightrag_relation_chunks": 2,
    "vertices": 2,
    "edges": 2,
}
NONE_DANGLING = dict.fromkeys(EXPECTED_DANGLING, 0)


def pg_settings():
    u = urlsplit(DSN)
    return {
        "POSTGRES_HOST": u.hostname or "localhost",
        "POSTGRES_PORT": str(u.port or 5432),
        "POSTGRES_USER": unquote(u.username or "postgres"),
        "POSTGRES_PASSWORD": unquote(u.password or ""),
        "POSTGRES_DATABASE": u.path.lstrip("/") or "postgres",
    }


async def pg_connect():
    s = pg_settings()
    return await asyncpg.connect(
        host=s["POSTGRES_HOST"],
        port=int(s["POSTGRES_PORT"]),
        user=s["POSTGRES_USER"],
        password=s["POSTGRES_PASSWORD"] or None,
        database=s["POSTGRES_DATABASE"],
    )


def fetch(sql, *args):
    async def go():
        con = await pg_connect()
        try:
            return await con.fetch(sql, *args)
        finally:
            await con.close()

    return asyncio.run(go())


def execute(sql, *args):
    async def go():
        con = await pg_connect()
        try:
            return await con.execute(sql, *args)
        finally:
            await con.close()

    return asyncio.run(go())


async def _llm(*args, **kwargs):
    return ""


async def _embed(texts, **kwargs):
    import numpy as np

    return np.zeros((len(texts), DIM), dtype=np.float32)


async def build_workspace(ws, workdir, created_tables):
    """Write a two-document workspace through LightRAG, then drop document "a" out of band."""
    from lightrag import LightRAG
    from lightrag.base import DocStatus
    from lightrag.kg.postgres_impl import SQL_TEMPLATES, TABLES, PGVectorStorage
    from lightrag.namespace import NameSpace
    from lightrag.utils import EmbeddingFunc, compute_mdhash_id, make_relation_chunk_key

    ef = EmbeddingFunc(embedding_dim=DIM, func=_embed, model_name=MODEL)
    rag = LightRAG(
        working_dir=str(workdir),
        workspace=ws,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="NanoVectorDBStorage",
        llm_model_func=_llm,
        embedding_func=ef,
    )
    await rag.initialize_storages()
    con = await pg_connect()
    try:
        doc_ids, chunks, path_of = {}, {}, {}
        now = datetime.now().isoformat()
        for key, (path, parts) in DOCS.items():
            text = "\n\n".join(parts)
            doc = doc_ids[key] = compute_mdhash_id(text, prefix="doc-")
            ids = [
                f"{doc}-chunk-{i:03d}" for i in range(len(parts))
            ]  # LightRAG's text-pipeline ids
            for i, (cid, part) in enumerate(zip(ids, parts)):
                chunks[f"{key.upper()}{i}"] = cid
                path_of[cid] = path
                await rag.text_chunks.upsert(
                    {
                        cid: {
                            "content": part,
                            "tokens": 3,
                            "chunk_order_index": i,
                            "full_doc_id": doc,
                            "file_path": path,
                        }
                    }
                )
            await rag.full_docs.upsert({doc: {"content": text, "file_path": path}})
            await rag.doc_status.upsert(
                {
                    doc: {
                        "status": DocStatus.PROCESSED,
                        "content_summary": text[:40],
                        "content_length": len(text),
                        "chunks_count": len(ids),
                        "chunks_list": ids,
                        "file_path": path,
                        "created_at": now,
                        "updated_at": now,
                    }
                }
            )
        c = chunks
        c["CUSTOM"] = compute_mdhash_id(
            "custom kg chunk", prefix="chunk-"
        )  # a custom-KG id: never touched
        path_of[c["CUSTOM"]] = "custom_kg"
        entities = {
            "Shared": [c["A0"], c["B0"]],
            "OnlyA": [c["A1"]],
            "OnlyB": [c["B1"], c["CUSTOM"]],
        }
        relations = {
            ("Shared", "OnlyA"): [c["A1"]],
            ("Shared", "OnlyB"): [c["A0"], c["B1"]],
        }

        def paths(ids):
            return SEP.join(dict.fromkeys(path_of[i] for i in ids))

        graph = rag.chunk_entity_relation_graph
        for name, ids in entities.items():
            await graph.upsert_node(
                name,
                {
                    "entity_id": name,
                    "entity_type": "concept",
                    "description": f"{name} description",
                    "source_id": SEP.join(ids),
                    "file_path": paths(ids),
                    "created_at": 1,
                },
            )
            await rag.entity_chunks.upsert(
                {name: {"chunk_ids": ids, "count": len(ids)}}
            )
        for (src, tgt), ids in relations.items():
            await graph.upsert_edge(
                src,
                tgt,
                {
                    "weight": 1.0,
                    "description": f"{src} and {tgt}",
                    "keywords": "related",
                    "source_id": SEP.join(ids),
                    "file_path": paths(ids),
                    "created_at": 1,
                },
            )
            await rag.relation_chunks.upsert(
                {
                    make_relation_chunk_key(src, tgt): {
                        "chunk_ids": ids,
                        "count": len(ids),
                    }
                }
            )

        # Vector tables: v1.5.7's DDL and naming, embedding column as REAL[] (no pgvector).
        tables = {}
        for ns, base in (
            (NameSpace.VECTOR_STORE_ENTITIES, "LIGHTRAG_VDB_ENTITY"),
            (NameSpace.VECTOR_STORE_RELATIONSHIPS, "LIGHTRAG_VDB_RELATION"),
        ):
            name = PGVectorStorage(
                namespace=ns, workspace=ws, global_config=asdict(rag), embedding_func=ef
            ).table_name
            if not await con.fetchval(
                "select to_regclass($1) is not null", name.lower()
            ):
                ddl = (
                    TABLES[base]["ddl"]
                    .replace("VECTOR(dimension)", "REAL[]")
                    .replace(base, name)
                )
                await con.execute(ddl)
                created_tables.append(name.lower())
            tables[base] = name.lower()
        stamp = datetime.now()
        for name, ids in entities.items():
            await con.execute(
                SQL_TEMPLATES["upsert_entity"].format(
                    table_name=tables["LIGHTRAG_VDB_ENTITY"]
                ),
                ws,
                compute_mdhash_id(name, prefix="ent-"),
                name,
                f"{name} description",
                None,
                ids,
                paths(ids),
                stamp,
                stamp,
            )
        for (src, tgt), ids in relations.items():
            await con.execute(
                SQL_TEMPLATES["upsert_relationship"].format(
                    table_name=tables["LIGHTRAG_VDB_RELATION"]
                ),
                ws,
                compute_mdhash_id(src + tgt, prefix="rel-"),
                src,
                tgt,
                f"{src} and {tgt}",
                None,
                ids,
                paths(ids),
                stamp,
                stamp,
            )

        for storage in (
            rag.full_docs,
            rag.text_chunks,
            rag.doc_status,
            rag.entity_chunks,
            rag.relation_chunks,
            graph,
        ):
            await storage.index_done_callback()

        # The out-of-band removal: document "a" disappears, its citations stay behind.
        for table, col in (
            ("lightrag_doc_chunks", "full_doc_id"),
            ("lightrag_doc_status", "id"),
            ("lightrag_doc_full", "id"),
        ):
            await con.execute(
                f"delete from {table} where workspace=$1 and {col}=$2", ws, doc_ids["a"]
            )
        return {
            "ws": ws,
            "graph": graph.graph_name,
            "ent": tables["LIGHTRAG_VDB_ENTITY"],
            "rel": tables["LIGHTRAG_VDB_RELATION"],
            "chunks": c,
            "doc_a": doc_ids["a"],
        }
    finally:
        await con.close()
        await rag.finalize_storages()


async def cleanup(built, created_tables):
    con = await pg_connect()
    try:
        backups = [
            r["table_name"]
            for r in await con.fetch(
                "select table_name from information_schema.tables where table_schema = any(current_schemas(false))"
                " and table_name ~ '^lightrag_[a-z_]+_bak_dangling_[0-9_]+$'"
            )
        ]
        for kb in built:
            for table in KV_TABLES + (kb["ent"], kb["rel"]):
                if await con.fetchval("select to_regclass($1) is not null", table):
                    await con.execute(
                        f"delete from {table} where workspace=$1", kb["ws"]
                    )
            for table in backups:
                if await con.fetchval(
                    f"select exists (select 1 from {table} where workspace=$1)",
                    kb["ws"],
                ):
                    await con.execute(f"drop table {table}")
            if await con.fetchval(
                "select exists (select 1 from ag_catalog.ag_graph where name = $1::name)",
                kb["graph"],
            ):
                await con.execute(
                    f"select ag_catalog.drop_graph('{kb['graph']}', true)"
                )
        for table in created_tables:
            await con.execute(f"drop table if exists {table}")
    finally:
        await con.close()


@pytest.fixture(scope="module")
def kbs(tmp_path_factory):
    pytest.importorskip("lightrag")
    tag = uuid.uuid4().hex[:8]
    # Mixed case and a hyphen: LightRAG keeps the case in the AGE graph name.
    names = {"sweep": f"Toolkit-Sweep-{tag}", "commit": f"Toolkit-Commit-{tag}"}
    built, created_tables = [], []

    async def build_all():
        for key, ws in names.items():
            built.append(
                await build_workspace(ws, tmp_path_factory.mktemp(key), created_tables)
            )

    try:
        with mock.patch.dict(os.environ, pg_settings()):
            os.environ.pop("POSTGRES_WORKSPACE", None)
            asyncio.run(build_all())
        yield dict(zip(names, built))
    finally:
        asyncio.run(cleanup(built, created_tables))


class IdleStub:
    """/documents/pipeline_status on 127.0.0.1, recording the X-API-Key it receives."""

    def __init__(self, busy=False):
        self.busy, self.keys = busy, []

    def __enter__(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                stub.keys.append(self.headers.get("X-API-Key"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"busy": stub.busy}).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def run_tool(settings, *args):
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        **pg_settings(),
        **settings,
    }
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def dangling(proc):
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(
        ln for ln in proc.stdout.splitlines() if "rows with missing chunk refs:" in ln
    )
    return {
        k: int(v)
        for k, v in (kv.split("=") for kv in line.split(": ", 1)[1].split(", "))
    }


def entity_rows(table, ws):
    return {
        r["entity_name"]: (list(r["chunk_ids"]), r["file_path"])
        for r in fetch(
            f"select entity_name, chunk_ids, file_path from {table} where workspace=$1",
            ws,
        )
    }


def relation_rows(table, ws):
    return {
        f"{r['source_id']}->{r['target_id']}": (list(r["chunk_ids"]), r["file_path"])
        for r in fetch(
            f"select source_id, target_id, chunk_ids, file_path from {table} where workspace=$1",
            ws,
        )
    }


def map_chunks(table, ws):
    return {
        r["id"]: (json.loads(r["cj"]), r["count"])
        for r in fetch(
            f"select id, chunk_ids::text as cj, count from {table} where workspace=$1",
            ws,
        )
    }


def graph_props(graph):
    vertices = {}
    for r in fetch(
        f'select id::text as gid, properties::text as p from "{graph}"._ag_label_vertex'
    ):
        vertices[r["gid"]] = json.loads(r["p"])
    edges = {}
    for r in fetch(
        f"select start_id::text as s, end_id::text as e, properties::text as p"
        f' from "{graph}"._ag_label_edge'
    ):
        pair = tuple(
            sorted((vertices[r["s"]]["entity_id"], vertices[r["e"]]["entity_id"]))
        )
        edges[pair] = json.loads(r["p"])
    return {v["entity_id"]: v for v in vertices.values()}, edges


def test_graph_and_vector_tables_are_named_like_lightrag(kbs):
    for kb in kbs.values():
        assert clean.graph_name(kb["ws"]) == kb["graph"]
        assert kb["graph"] != kb["graph"].lower(), (
            "the fixture should exercise a mixed-case graph name"
        )
        suffix = clean.vdb_suffix(MODEL, DIM)
        assert (f"lightrag_vdb_entity_{suffix}", f"lightrag_vdb_relation_{suffix}") == (
            kb["ent"],
            kb["rel"],
        )


def test_sweep_counts_fixes_and_converges(kbs):
    kb = kbs["sweep"]
    c, ws = kb["chunks"], kb["ws"]
    settings = {"WORKSPACE": ws}  # no EMBEDDING_*: the tool detects the vector tables

    dry = run_tool(settings, "--sweep", "--dry-run")
    assert dangling(dry) == EXPECTED_DANGLING
    assert dry.stdout.startswith(
        f"workspace {ws!r}, graph {kb['graph']!r}, vector tables {kb['ent']} / {kb['rel']}"
    )

    with IdleStub(busy=True) as stub:
        busy = run_tool(
            {**settings, "LIGHTRAG_BASE_URL": stub.url, "LIGHTRAG_API_KEY": "test-key"},
            "--sweep",
        )
    assert busy.returncode == 3, busy.stdout + busy.stderr
    assert stub.keys == ["test-key"]
    assert dangling(run_tool(settings, "--sweep", "--dry-run")) == EXPECTED_DANGLING, (
        "busy sweep wrote"
    )

    with IdleStub(busy=False) as stub:
        swept = run_tool({**settings, "LIGHTRAG_BASE_URL": stub.url}, "--sweep")
    assert dangling(swept) == EXPECTED_DANGLING
    assert "DRY-RUN" not in swept.stdout
    assert (
        "rows left with NO chunks (delete separately if any): {'lightrag_vdb_entity': 1, "
        "'lightrag_vdb_relation': 1, 'lightrag_entity_chunks': 1, 'lightrag_relation_chunks': 1}"
    ) in swept.stdout

    assert dangling(run_tool(settings, "--sweep", "--dry-run")) == NONE_DANGLING

    ents = entity_rows(kb["ent"], ws)
    assert ents["Shared"] == ([c["B0"]], "paper_b.md")
    assert ents["OnlyA"][0] == []
    assert ents["OnlyB"] == ([c["B1"], c["CUSTOM"]], f"paper_b.md{SEP}custom_kg")
    rels = relation_rows(kb["rel"], ws)
    assert rels["Shared->OnlyB"] == ([c["B1"]], "paper_b.md")
    assert rels["Shared->OnlyA"][0] == []
    assert map_chunks("lightrag_entity_chunks", ws)["Shared"] == ([c["B0"]], 1)
    assert map_chunks("lightrag_entity_chunks", ws)["OnlyA"] == ([], 0)
    vertices, edges = graph_props(kb["graph"])
    assert (vertices["Shared"]["source_id"], vertices["Shared"]["file_path"]) == (
        c["B0"],
        "paper_b.md",
    )
    assert vertices["OnlyA"]["source_id"] == ""
    assert vertices["OnlyB"]["source_id"] == f"{c['B1']}{SEP}{c['CUSTOM']}"
    assert (
        edges[("OnlyB", "Shared")]["source_id"],
        edges[("OnlyB", "Shared")]["file_path"],
    ) == (c["B1"], "paper_b.md")


def test_scan_commit_backs_up_deletions_and_skips_rows_changed_since_scan(
    kbs, tmp_path
):
    kb = kbs["commit"]
    c, ws, graph = kb["chunks"], kb["ws"], kb["graph"]
    settings = {
        "POSTGRES_WORKSPACE": ws,
        "EMBEDDING_MODEL": MODEL,
        "EMBEDDING_DIM": str(DIM),
    }
    plan_path = tmp_path / "plan.json"

    scan = run_tool(settings, "--scan", "--plan", str(plan_path))
    assert scan.returncode == 0, scan.stdout + scan.stderr
    assert f"docs with missing chunk refs: 1 -> ['{kb['doc_a']}']" in scan.stdout
    plan = json.loads(plan_path.read_text())
    assert (plan["workspace"], plan["graph"]) == (ws, graph)

    def update_delete(items):
        return sum(not i["delete"] for i in items), sum(
            bool(i["delete"]) for i in items
        )

    for items in (
        plan["vdb"][kb["ent"]],
        plan["vdb"][kb["rel"]],
        plan["maps"]["lightrag_entity_chunks"],
        plan["maps"]["lightrag_relation_chunks"],
        plan["vertices"],
        plan["edges"],
    ):
        assert update_delete(items) == (1, 1)

    # Extraction merges a new chunk into "Shared" between --scan and --commit.
    execute(
        f"update {kb['ent']} set chunk_ids = array_append(chunk_ids, $2::varchar)"
        f" where workspace=$1 and entity_name='Shared'",
        ws,
        c["B1"],
    )

    with IdleStub(busy=False) as stub:
        commit = run_tool(
            {**settings, "LIGHTRAG_BASE_URL": stub.url},
            "--commit",
            "--plan",
            str(plan_path),
        )
    assert commit.returncode == 0, commit.stdout + commit.stderr
    assert "committed: {'updated': 5, 'deleted': 6, 'skipped': 1}" in commit.stdout
    assert not plan_path.exists() and plan_path.with_name("plan.json.done").exists()

    stamp = re.search(r"_bak_dangling_(\d{8}_\d{6})", commit.stdout).group(1)
    for table, expected in (
        ("lightrag_vdb_entity", {"OnlyA"}),
        ("lightrag_vdb_relation", {"Shared->OnlyA"}),
        ("lightrag_entity_chunks", {"OnlyA"}),
        ("lightrag_relation_chunks", {f"OnlyA{SEP}Shared"}),
    ):
        name_sql = {
            "lightrag_vdb_entity": "entity_name",
            "lightrag_vdb_relation": "source_id || '->' || target_id",
        }.get(table, "id")
        rows = fetch(
            f"select {name_sql} as name from {table}_bak_dangling_{stamp} where workspace=$1",
            ws,
        )
        assert {r["name"] for r in rows} == expected, table
    vbak = fetch(
        f'select properties::text as p from "{graph}".bak_dangling_ag_label_vertex_{stamp}'
    )
    assert [json.loads(r["p"])["entity_id"] for r in vbak] == ["OnlyA"]
    assert (
        len(fetch(f'select 1 from "{graph}".bak_dangling_ag_label_edge_{stamp}')) == 1
    )

    ents = entity_rows(kb["ent"], ws)
    assert set(ents) == {"Shared", "OnlyB"}
    assert ents["Shared"][0] == [c["A0"], c["B0"], c["B1"]], (
        "a row changed since the scan is left alone"
    )
    assert set(relation_rows(kb["rel"], ws)) == {"Shared->OnlyB"}
    assert map_chunks("lightrag_entity_chunks", ws)["Shared"] == ([c["B0"]], 1)
    vertices, edges = graph_props(graph)
    assert set(vertices) == {"Shared", "OnlyB"}
    assert set(edges) == {("OnlyB", "Shared")}
    assert vertices["Shared"]["source_id"] == c["B0"]

    # Table detection must not mistake the backup tables for vector tables.
    detected = run_tool({"POSTGRES_WORKSPACE": ws}, "--sweep", "--dry-run")
    assert dangling(detected) == {**NONE_DANGLING, "lightrag_vdb_entity": 1}
    assert f"vector tables {kb['ent']} / {kb['rel']}" in detected.stdout

    # The sweep converges the skipped row and keeps the chunk merged in meanwhile.
    assert dangling(run_tool(settings, "--sweep", "--dry-run")) == {
        **NONE_DANGLING,
        "lightrag_vdb_entity": 1,
    }
    with IdleStub(busy=False) as stub:
        dangling(run_tool({**settings, "LIGHTRAG_BASE_URL": stub.url}, "--sweep"))
    assert entity_rows(kb["ent"], ws)["Shared"] == ([c["B0"], c["B1"]], "paper_b.md")
    assert dangling(run_tool(settings, "--sweep", "--dry-run")) == NONE_DANGLING
