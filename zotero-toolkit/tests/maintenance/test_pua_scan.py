"""Unit tests for maintenance/pua_scan.py. Offline: asyncpg is replaced by a fake."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MAINTENANCE = Path(__file__).resolve().parents[2] / "maintenance"

spec = importlib.util.spec_from_file_location("pua_scan", MAINTENANCE / "pua_scan.py")
pua_scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pua_scan)


def row(file_path, status, chunks, length, pua):
    return {
        "file_path": file_path,
        "status": status,
        "chunks_count": chunks,
        "content_length": length,
        "pua": pua,
    }


ROWS = [
    row("TEST0001__scanned_encyclopedia.md", "processed", 6000, 3_000_000, 2_850_000),
    row("TEST0002__partly_garbled_book.md", "processed", 1900, 1_200_000, 850_000),
    row("TEST0003__handbook_with_greek.md", "processed", 1000, 3_000_000, 110_000),
    row("TEST0004__one_symbol.md", "processed", 20, 60_000, 1),
]


class FakeConnection:
    def __init__(self, rows):
        self.rows, self.queries, self.closed = rows, [], False

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        return self.rows

    async def close(self):
        self.closed = True


def run(argv, rows=ROWS, env=None):
    con = FakeConnection(rows)

    async def connect(**kwargs):
        con.kwargs = kwargs
        return con

    with (
        mock.patch.dict(os.environ, env or {}, clear=True),
        mock.patch.object(pua_scan.asyncpg, "connect", connect),
        mock.patch("sys.stdout") as out,
    ):
        rc = pua_scan.main(argv)
    text = "".join(c.args[0] for c in out.write.call_args_list)
    return rc, text, con


class ClassifyTests(unittest.TestCase):
    def test_default_bar_reports_the_garbled_scans_only(self):
        hits = pua_scan.classify(ROWS, 0.30)
        self.assertEqual(
            [h[1] for h in hits],
            ["TEST0001__scanned_encyclopedia.md", "TEST0002__partly_garbled_book.md"],
        )
        self.assertAlmostEqual(hits[0][0], 0.95)

    def test_null_counts_do_not_divide_by_zero(self):
        hits = pua_scan.classify(
            [row("TEST0005__empty.md", "processed", 0, None, None)], 0.3
        )
        self.assertEqual(hits, [])


class CliTests(unittest.TestCase):
    def test_reports_offenders_and_exits_1(self):
        rc, text, con = run(["--workspace", "demo"])
        self.assertEqual(rc, 1)
        self.assertIn("WARN 95.0% PUA", text)
        self.assertIn("TEST0001__scanned_encyclopedia.md", text)
        self.assertNotIn("TEST0003__handbook_with_greek.md", text)
        self.assertIn("2 glyph-code document(s) at >= 30% (4 documents contain", text)
        self.assertEqual(con.queries[0][1], ("demo",))
        self.assertTrue(con.closed)

    def test_clean_workspace_exits_0(self):
        rc, text, _ = run([], rows=ROWS[2:])
        self.assertEqual(rc, 0)
        self.assertIn("0 glyph-code document(s)", text)

    def test_min_ratio_flag(self):
        rc, text, _ = run(["--min-ratio", "0.9"])
        self.assertEqual(rc, 1)
        self.assertIn("1 glyph-code document(s) at >= 90%", text)

    def test_settings_precedence_flag_environment_env_file(self):
        with tempfile.TemporaryDirectory() as td:
            env_file = os.path.join(td, ".env")
            with open(env_file, "w") as fh:
                fh.write("POSTGRES_WORKSPACE=from_file\nPOSTGRES_HOST=db.example\n")
            rc, text, con = run(["--env-file", env_file])
            self.assertIn("workspace 'from_file'", text)
            self.assertEqual(con.kwargs["host"], "db.example")
            rc, text, con = run(
                ["--env-file", env_file], env={"POSTGRES_WORKSPACE": "from_env"}
            )
            self.assertIn("workspace 'from_env'", text)
            rc, text, con = run(
                ["--env-file", env_file, "--workspace", "from_flag"],
                env={"POSTGRES_WORKSPACE": "from_env"},
            )
            self.assertIn("workspace 'from_flag'", text)

    def test_database_error_exits_2(self):
        async def connect(**kwargs):
            raise OSError("connection refused")

        with (
            mock.patch.object(pua_scan.asyncpg, "connect", connect),
            mock.patch("sys.stderr"),
        ):
            self.assertEqual(pua_scan.main([]), 2)


if __name__ == "__main__":
    unittest.main()
