"""Unit tests for maintenance/clean_dangling_refs.py and run_when_idle.sh.

Offline: no database is contacted; the idle check talks to a stub HTTP server on
127.0.0.1. The PostgreSQL + AGE integration test is test_clean_dangling_refs_pg.py.
"""
import asyncio
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

MAINTENANCE = Path(__file__).resolve().parents[2] / "maintenance"

spec = importlib.util.spec_from_file_location("clean_dangling_refs", MAINTENANCE / "clean_dangling_refs.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def configure(argv, env=None):
    """Configure the module as the CLI would, from argv and a clean environment."""
    with mock.patch.dict(os.environ, env or {}, clear=True):
        clean.configure(clean.parse_args(argv))


class IdleStub:
    """/documents/pipeline_status on 127.0.0.1, recording the X-API-Key it receives."""

    def __init__(self, busy=False):
        self.busy, self.keys = busy, []

    def __enter__(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                stub.keys.append(self.headers.get("X-API-Key"))
                ok = self.path == "/documents/pipeline_status"
                self.send_response(200 if ok else 404)
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


class CleanDanglingRefsTests(unittest.TestCase):
    def setUp(self):
        configure(["--sweep"])

    def test_removed_chunk_from_reingested_processed_document(self):
        doc = "doc-" + "a" * 32
        good, stale = doc + "-chunk-000", doc + "-chunk-018"
        kept, affected, path = clean.split_refs(
            [good, stale], {doc}, {doc: "A__paper.md"}, {good}
        )
        self.assertEqual(kept, [good])
        self.assertEqual(affected, {doc})
        self.assertEqual(path, "A__paper.md")

    def test_chunk_still_present_without_status_is_not_discarded(self):
        doc = "doc-" + "b" * 32
        chunk = doc + "-chunk-000"
        kept, affected, path = clean.split_refs([chunk], set(), {}, {chunk})
        self.assertEqual((kept, affected, path), ([chunk], set(), None))

    def test_unknown_chunk_format_is_preserved(self):
        doc = "doc-" + "c" * 32
        multimedia = doc + "-mm-table-000"
        kept, affected, path = clean.split_refs([multimedia], set(), {}, set())
        self.assertEqual((kept, affected, path), ([multimedia], set(), None))

    def test_graph_rows_update_in_one_statement_with_compare_and_set(self):
        class FakeConnection:
            calls = []

            async def execute(self, sql, payload):
                self.calls.append((sql, payload))
                return "UPDATE 2"

        configure(["--sweep", "--workspace", "Team-KB"])
        con = FakeConnection()
        rows = [
            {"gid": "123", "old_props": '{"source_id":"old"}',
             "new_props": '{"source_id":"new"}'},
            {"gid": "456", "old_props": '{"source_id":"old2"}',
             "new_props": '{"source_id":"new2"}'},
        ]
        asyncio.run(clean.update_graph_rows(con, "_ag_label_vertex", rows))
        self.assertEqual(len(con.calls), 1)
        sql, payload = con.calls[0]
        self.assertIn("jsonb_to_recordset", sql)
        self.assertIn("g.properties::text = u.old_props", sql)
        # AGE keeps the graph name's case, so the schema must be quoted
        self.assertIn('"Team_KB_chunk_entity_relation"._ag_label_vertex', sql)
        self.assertEqual(json.loads(payload), rows)

    def test_busy_pipeline_refuses_writes(self):
        async def unexpected_connect():
            self.fail("busy pipeline must be refused before connecting")

        with mock.patch.object(clean, "pipeline_idle", return_value=False), \
             mock.patch.object(clean, "connect", unexpected_connect):
            self.assertEqual(asyncio.run(clean.sweep(dry=False)), 3)

    def test_busy_pipeline_refuses_commit(self):
        async def unexpected_connect():
            self.fail("busy pipeline must be refused before connecting")

        with tempfile.TemporaryDirectory() as tmp:
            plan = os.path.join(tmp, "plan.json")
            with open(plan, "w") as fh:
                json.dump({"workspace": "default", "graph": "chunk_entity_relation"}, fh)
            configure(["--commit", "--plan", plan])
            with mock.patch.object(clean, "pipeline_idle", return_value=False), \
                 mock.patch.object(clean, "connect", unexpected_connect):
                self.assertEqual(asyncio.run(clean.commit(force=False)), 3)
            self.assertTrue(os.path.exists(plan), "a refused commit keeps its plan")

    def test_commit_refuses_plan_scanned_in_another_workspace(self):
        def unexpected_idle_check():
            self.fail("a foreign plan must be refused before anything else")

        with tempfile.TemporaryDirectory() as tmp:
            plan = os.path.join(tmp, "plan.json")
            with open(plan, "w") as fh:
                json.dump({"workspace": "papers", "graph": "papers_chunk_entity_relation"}, fh)
            configure(["--commit", "--plan", plan], {"POSTGRES_WORKSPACE": "notes"})
            with mock.patch.object(clean, "pipeline_idle", unexpected_idle_check):
                self.assertEqual(asyncio.run(clean.commit(force=True)), 2)

    def test_graph_name_follows_lightrag(self):
        # PGGraphStorage._get_workspace_graph_name, clipped to PostgreSQL's 63-byte names
        for ws in (None, "", "  ", "default", " Default "):
            self.assertEqual(clean.graph_name(ws), "chunk_entity_relation")
        self.assertEqual(clean.graph_name("papers"), "papers_chunk_entity_relation")
        self.assertEqual(clean.graph_name(" Team-KB.v2 "), "Team_KB_v2_chunk_entity_relation")
        self.assertEqual(clean.graph_name("x" * 60), ("x" * 60 + "_chunk_entity_relation")[:63])

    def test_vector_tables_named_like_lightrag_or_left_for_detection(self):
        configure(["--sweep"], {"EMBEDDING_MODEL": "BAAI/bge-m3", "EMBEDDING_DIM": "1024"})
        self.assertEqual((clean.T_ENT, clean.T_REL),
                         ("lightrag_vdb_entity_baai_bge_m3_1024d", "lightrag_vdb_relation_baai_bge_m3_1024d"))
        configure(["--sweep", "--embedding-model", "nomic-embed-text:latest", "--embedding-dim", "768"])
        self.assertEqual(clean.T_ENT, "lightrag_vdb_entity_nomic_embed_text_latest_768d")
        # without both, the server's table cannot be named here: connect() detects it
        configure(["--sweep"], {"EMBEDDING_MODEL": "bge-m3"})
        self.assertEqual((clean.T_ENT, clean.T_REL), (None, None))

    def test_vector_table_detection_prefers_the_table_holding_the_workspace(self):
        class FakeConnection:
            def __init__(self, tables, with_rows):
                self.tables, self.with_rows = tables, with_rows

            async def fetch(self, sql):
                return [{"table_name": t} for t in self.tables]

            async def fetchval(self, sql, ws):
                return any(f" {t} " in sql for t in self.with_rows)

        def detect(tables, with_rows):
            configure(["--sweep", "--workspace", "papers"])
            asyncio.run(clean.resolve_vdb_tables(FakeConnection(tables, with_rows)))
            return clean.T_ENT, clean.T_REL

        both = ["lightrag_vdb_entity", "lightrag_vdb_entity_bge_m3_1024d"]
        self.assertEqual(detect(both, ["lightrag_vdb_entity_bge_m3_1024d"]),
                         ("lightrag_vdb_entity_bge_m3_1024d", "lightrag_vdb_relation_bge_m3_1024d"))
        # no EMBEDDING_MODEL on the server: LightRAG's tables carry no suffix
        self.assertEqual(detect(both, ["lightrag_vdb_entity"]), ("lightrag_vdb_entity", "lightrag_vdb_relation"))
        self.assertEqual(detect(["lightrag_vdb_entity_bge_m3_1024d"], []),
                         ("lightrag_vdb_entity_bge_m3_1024d", "lightrag_vdb_relation_bge_m3_1024d"))
        for tables, with_rows in ((both, both), (both, []), ([], [])):
            with self.assertRaises(SystemExit) as cm:
                detect(tables, with_rows)
            self.assertIn("EMBEDDING_MODEL", str(cm.exception.code))

    def test_settings_come_from_flags_then_environment_then_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = os.path.join(tmp, ".env")
            with open(env_file, "w") as fh:
                fh.write("# LightRAG .env\n"
                         "export POSTGRES_HOST=db.example.com\n"
                         "POSTGRES_PASSWORD='not-a-secret #1'\n"
                         "PORT=9700  # server port\n"
                         "WORKSPACE=papers\n"
                         "POSTGRES_USER=\n")
            configure(["--sweep", "--env-file", env_file], {"POSTGRES_PORT": "6543"})
            with mock.patch.dict(os.environ, {"POSTGRES_PORT": "6543"}, clear=True):
                self.assertEqual(clean.setting("POSTGRES_HOST"), "db.example.com")
                self.assertEqual(clean.setting("POSTGRES_PASSWORD"), "not-a-secret #1")
                self.assertEqual(clean.setting("POSTGRES_PORT"), "6543")
                self.assertEqual(clean.setting("POSTGRES_USER", "postgres"), "postgres")
            self.assertEqual(clean.API_URL, "http://localhost:9700")
            self.assertEqual((clean.WS, clean.GRAPH), ("papers", "papers_chunk_entity_relation"))

            # POSTGRES_WORKSPACE overrides WORKSPACE, as in LightRAG; the flag overrides both
            configure(["--sweep", "--env-file", env_file], {"POSTGRES_WORKSPACE": "pg"})
            self.assertEqual(clean.WS, "pg")
            configure(["--sweep", "--env-file", env_file, "--workspace", "cli",
                       "--api-url", "http://127.0.0.1:9621/"], {"POSTGRES_WORKSPACE": "pg"})
            self.assertEqual((clean.WS, clean.API_URL), ("cli", "http://127.0.0.1:9621"))
        configure(["--sweep"])
        self.assertEqual((clean.WS, clean.GRAPH, clean.API_URL),
                         ("default", "chunk_entity_relation", "http://localhost:9621"))

    def test_idle_check_sends_api_key_header(self):
        with IdleStub(busy=False) as stub:
            configure(["--sweep"], {"LIGHTRAG_BASE_URL": stub.url})
            with mock.patch.dict(os.environ, {"LIGHTRAG_API_KEY": "test-key"}):
                self.assertTrue(clean.pipeline_idle())
            stub.busy = True
            with mock.patch.dict(os.environ, {"LIGHTRAG_API_KEY": ""}):
                self.assertFalse(clean.pipeline_idle())
        self.assertEqual(stub.keys, ["test-key", None])

    def test_unreachable_server_is_an_error_not_idle(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        configure(["--sweep", "--api-url", f"http://127.0.0.1:{port}"])
        with self.assertRaises(SystemExit) as cm:
            clean.pipeline_idle()
        self.assertIn("--force", str(cm.exception.code))


@unittest.skipUnless(shutil.which("bash"), "run_when_idle.sh needs bash")
class RunWhenIdleTests(unittest.TestCase):
    """run_when_idle.sh against a stand-in interpreter that reports busy N times.

    The stand-in answers the wrapper's `-c 'import asyncpg'` check without recording a
    call; STUB_NO_ASYNCPG makes that check fail as a python without asyncpg would.
    """

    def run_wrapper(self, busy, *args, **env):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        stub = os.path.join(tmp, "python")
        with open(stub, "w") as fh:
            fh.write('#!/usr/bin/env bash\n'
                     'if [ "$1" = -c ]; then\n'
                     '  [ -z "${STUB_NO_ASYNCPG:-}" ] && exit 0\n'
                     '  echo "ModuleNotFoundError: No module named \'asyncpg\'" >&2; exit 1\n'
                     'fi\n'
                     'echo "$*" >> "$STUB_DIR/calls"\n'
                     'n=$(wc -l < "$STUB_DIR/calls")\n'
                     '[ "$n" -le "$STUB_BUSY" ] && { echo "pipeline busy"; exit 3; }\n'
                     'echo "swept"\n')
        os.chmod(stub, 0o755)
        full_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHON": stub,
                    "STUB_DIR": tmp, "STUB_BUSY": str(busy), "IDLE_RETRY_INTERVAL": "0", **env}
        proc = subprocess.run(["bash", str(MAINTENANCE / "run_when_idle.sh"), *args],
                              env=full_env, capture_output=True, text=True, timeout=60)
        calls_file = os.path.join(tmp, "calls")
        calls = []
        if os.path.exists(calls_file):
            with open(calls_file) as fh:
                calls = fh.read().splitlines()
        return proc, calls

    def test_retries_while_busy_then_returns_the_tool_status(self):
        proc, calls = self.run_wrapper(2)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(calls, [f"{MAINTENANCE / 'clean_dangling_refs.py'} --sweep"] * 3)
        self.assertIn("swept", proc.stdout)

    def test_passes_arguments_through(self):
        proc, calls = self.run_wrapper(0, "--commit", "--plan", "p.json")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(calls, [f"{MAINTENANCE / 'clean_dangling_refs.py'} --commit --plan p.json"])

    def test_adds_sweep_when_no_mode_flag_is_given(self):
        # The documented `run_when_idle.sh --env-file ...` form must sweep, not die on
        # argparse's "one of the arguments ... is required" (exit 2, never retried).
        proc, calls = self.run_wrapper(0, "--env-file", "x.env")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(calls, [f"{MAINTENANCE / 'clean_dangling_refs.py'} --sweep --env-file x.env"])

    def test_leaves_an_explicit_mode_alone(self):
        for args in (["--scan"], ["--commit"], ["--show"], ["--sweep", "--dry-run"]):
            with self.subTest(args=args):
                proc, calls = self.run_wrapper(0, "--env-file", "x.env", *args)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(calls, [f"{MAINTENANCE / 'clean_dangling_refs.py'} --env-file x.env "
                                         + " ".join(args)])

    def test_gives_up_with_exit_3(self):
        proc, calls = self.run_wrapper(100, IDLE_MAX_WAIT="0")
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("giving up", proc.stdout)

    def test_refuses_an_interpreter_without_asyncpg(self):
        # cron and systemd do not activate the virtual environment asyncpg was installed
        # into, so a bare python3 there is the system one. The wrapper must say which
        # interpreter to set instead of failing every run with a ModuleNotFoundError.
        proc, calls = self.run_wrapper(0, STUB_NO_ASYNCPG="1")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(calls, [])
        self.assertIn("No module named 'asyncpg'", proc.stdout)
        self.assertIn("set PYTHON", proc.stdout)

    def test_refuses_a_missing_interpreter(self):
        proc, calls = self.run_wrapper(0, PYTHON="/nonexistent/venv/bin/python")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertEqual(calls, [])
        self.assertIn("/nonexistent/venv/bin/python", proc.stdout)
        self.assertIn("set PYTHON", proc.stdout)

    def test_runs_the_tool_with_an_interpreter_that_has_asyncpg(self):
        # This test module imports asyncpg, so its own interpreter passes the real check.
        # --show without a plan file exits 2 before any connection is attempted.
        plan = os.path.join(tempfile.mkdtemp(), "missing_plan.json")
        self.addCleanup(shutil.rmtree, os.path.dirname(plan))
        proc, _ = self.run_wrapper(0, "--show", "--plan", plan, PYTHON=sys.executable)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("rc=2 after 1 attempt(s)", proc.stdout)


if __name__ == "__main__":
    unittest.main()
