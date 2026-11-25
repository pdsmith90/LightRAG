import glob
import gzip
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

# The modules under test sit in citegraph/ and import one another by name.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "citegraph")
)

import s2_api  # noqa: E402
import s2_citations_sync as sync  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "input", "citations-sample.jsonl.gz")
KEY = "S2SECRETKEY-0123456789"
R1, R2, R3 = "2026-11-17", "2026-11-24", "2026-12-01"


def sample_lines():
    with gzip.open(SAMPLE, "rb") as f:
        return [ln.rstrip(b"\n") for ln in f if ln.strip()]


def gz(lines):
    return gzip.compress(b"\n".join(lines) + b"\n")


def rec(cid, citing, cited, infl=False, contexts=("ctx",), intents=(("background",),)):
    return {
        "citationid": cid,
        "citingcorpusid": citing,
        "citedcorpusid": cited,
        "isinfluential": infl,
        "contexts": None if contexts is None else list(contexts),
        "intents": None if intents is None else [list(i) for i in intents],
    }


def enc(r):
    return json.dumps(r).encode()


def url(kind, name):
    return f"https://s3.example.com/{kind}/{name}?X-Amz-Signature=SIGSIGSIG"


def rows(path):
    con = sqlite3.connect(path)
    try:
        return {r[0]: r for r in con.execute("SELECT * FROM citations")}
    finally:
        con.close()


def table(path, sql):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


class FakeApi:
    """Stands in for sync.Api: the release listing, the shard URLs, the diff chain."""

    def __init__(self, release=R1, shards=(), diffs=()):
        self.release, self.shards, self.diff_list = release, list(shards), list(diffs)
        self.files_calls = 0
        self.diffs_calls = 0

    def latest(self):
        return self.release

    def files(self, release):
        self.files_calls += 1
        return [url(f"citations/{release}", n) for n in self.shards]

    def diffs(self, frm, to):
        self.diffs_calls += 1
        return self.diff_list


class Served:
    """Patches sync.open_stream to serve byte blobs by URL; ``fail`` maps a URL
    basename to the number of attempts that raise first."""

    def __init__(self, blobs, fail=None):
        self.blobs, self.fail, self.opened = blobs, dict(fail or {}), []

    def __call__(self, u):
        name = sync.basename(u)
        self.opened.append(name)
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            raise urllib.error.URLError("connection reset")
        return io.BytesIO(self.blobs[name])


def in_process(job):
    try:
        with sync.open_stream(job["url"]) as s:
            return True, sync.process_shard(
                job["store"],
                job["release_id"],
                job["basename"],
                s,
                sync.read_ids(job["ids_file"]),
            )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


class Env(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = d.name
        self.store = os.path.join(d.name, "s2store", "citations.sqlite")
        self.ids_file = os.path.join(d.name, "ids.txt")
        self.lines = sample_lines()
        self.recs = [json.loads(x) for x in self.lines]
        out = mock.patch("sys.stdout", io.StringIO())
        out.start()
        self.addCleanup(out.stop)
        err = mock.patch("sys.stderr", io.StringIO())
        self.err = err.start()
        self.addCleanup(err.stop)

    def write_ids(self, ids, path=None):
        with open(path or self.ids_file, "w") as f:
            f.write("".join(f"{i}\n" for i in sorted(ids)))

    def shards3(self):
        n = len(self.lines)
        parts = [
            self.lines[: n // 3],
            self.lines[n // 3 : 2 * n // 3],
            self.lines[2 * n // 3 :],
        ]
        names = [f"{i}.jsonl.gz" for i in range(3)]
        return names, {nm: gz(p) for nm, p in zip(names, parts)}

    def library(self):
        """A small library: the cited ids of the first records."""
        return {
            self.recs[0]["citedcorpusid"],
            self.recs[10]["citingcorpusid"],
            self.recs[40]["citedcorpusid"],
        }

    def expected(self, L):
        return {
            r["citationid"]
            for r in self.recs
            if r["citingcorpusid"] in L or r["citedcorpusid"] in L
        }


class StreamFilterTest(Env):
    def test_sample_has_the_documented_shape(self):
        self.assertEqual(len(self.recs), 72)
        self.assertTrue(any(r["contexts"] is None for r in self.recs))
        self.assertTrue(any(r["contexts"] for r in self.recs))
        for r in self.recs:
            self.assertEqual(
                sorted(r),
                [
                    "citationid",
                    "citedcorpusid",
                    "citingcorpusid",
                    "contexts",
                    "intents",
                    "isinfluential",
                ],
            )

    def test_filter_on_the_real_sample_keeps_citing_or_cited(self):
        L = self.library()
        got = []
        seen, kept = sync.scan_lines(iter(self.lines), L, got.append)
        want = self.expected(L)
        self.assertEqual((seen, kept), (72, len(want)))
        self.assertTrue(0 < len(want) < 72)
        self.assertEqual({g[0] for g in got}, want)
        by_id = {g[0]: g for g in got}
        for r in self.recs:
            if r["citationid"] in want:
                g = by_id[r["citationid"]]
                self.assertEqual(
                    (g[1], g[2], g[3]),
                    (r["citingcorpusid"], r["citedcorpusid"], int(r["isinfluential"])),
                )
                # contexts: null stays NULL, a list is stored as JSON
                self.assertEqual(
                    g[4],
                    None
                    if r["contexts"] is None
                    else json.dumps(r["contexts"], ensure_ascii=False),
                )
                self.assertEqual(json.loads(g[4]) if g[4] else None, r["contexts"])

    def test_unresolved_cited_paper_is_null_in_the_sample_and_is_handled(self):
        nulls = [r for r in self.recs if r["citedcorpusid"] is None]
        self.assertTrue(
            nulls
        )  # S2 leaves the cited id null when it could not resolve the paper
        L = {nulls[0]["citingcorpusid"]}
        got = []
        sync.scan_lines(iter(self.lines), L, got.append)
        row = next(g for g in got if g[0] == nulls[0]["citationid"])
        self.assertIsNone(
            row[2]
        )  # kept (the citing paper is in the library), cited stored as NULL
        got = []
        sync.scan_lines(
            iter(self.lines), {123}, got.append
        )  # a null id is never "in" the library
        self.assertEqual(got, [])

    def test_both_directions_are_kept(self):
        a, b = 111, 222
        lines = [enc(rec(1, a, 9)), enc(rec(2, 9, b)), enc(rec(3, 7, 8))]
        got = []
        sync.scan_lines(iter(lines), {9}, got.append)
        self.assertEqual({g[0] for g in got}, {1, 2})

    def test_only_matching_lines_are_parsed(self):
        L = self.library()
        calls = []
        real = json.loads
        with mock.patch.object(
            sync.json, "loads", lambda s: (calls.append(1), real(s))[1]
        ):
            _, kept = sync.scan_lines(iter(self.lines), L, lambda row: None)
        self.assertEqual(len(calls), kept)

    def test_key_order_does_not_matter_and_blank_lines_skip(self):
        line = b'{"contexts":null,"citedcorpusid":9,"isinfluential":true,"intents":null,"citingcorpusid":5,"citationid":3}'
        got = []
        seen, kept = sync.scan_lines(iter([b"", line, b"  "]), {9}, got.append)
        self.assertEqual((seen, kept), (1, 1))
        self.assertEqual(got[0], (3, 5, 9, 1, None, None))

    def test_a_record_without_corpus_ids_stops_the_run(self):
        with self.assertRaises(sync.SyncError):
            sync.scan_lines(iter([b'{"citationid": 1}']), {1}, lambda r: None)

    def test_text_that_looks_like_a_field_does_not_fool_the_prefilter(self):
        r = rec(5, 1, 2, contexts=['x "citingcorpusid": 999, "citedcorpusid": 998 y'])
        got = []
        sync.scan_lines(iter([enc(r)]), {999}, got.append)
        self.assertEqual(got, [])  # the escaped quotes in the text do not match


class FullPassTest(Env):
    def run_full(self, api, L, served, spawn=in_process, parallel=2):
        self.write_ids(L)
        with mock.patch.object(sync, "open_stream", served):
            return sync.run_full(api, self.store, self.ids_file, parallel, spawn)

    def test_full_pass_keeps_exactly_the_filtered_rows(self):
        names, blobs = self.shards3()
        L = self.library()
        api = FakeApi(shards=names)
        self.assertEqual(self.run_full(api, L, Served(blobs)), 0)
        self.assertEqual(set(rows(self.store)), self.expected(L))
        meta = dict(table(self.store, "SELECT key, value FROM meta"))
        self.assertEqual(
            (meta["release_id"], meta["mode"], meta["full_complete"]), (R1, "full", "1")
        )
        self.assertEqual(meta["library_corpus_ids_hash"], sync.ids_hash(L))
        self.assertEqual(
            {r[0] for r in table(self.store, "SELECT corpusid FROM library_ids")}, L
        )
        self.assertEqual(len(table(self.store, "SELECT * FROM shards")), 3)
        self.assertEqual(glob.glob(os.path.join(self.dir, "s2store", "*.spool")), [])
        self.assertEqual(
            self.run_full(api, L, Served(blobs)), 0
        )  # a finished pass is not repeated

    def test_resume_per_shard_after_a_failed_shard(self):
        names, blobs = self.shards3()
        L = self.library()
        api = FakeApi(shards=names)
        served = Served(blobs, fail={"1.jsonl.gz": 99})
        self.assertEqual(self.run_full(api, L, served), 1)
        done = {r[1] for r in table(self.store, "SELECT * FROM shards")}
        self.assertEqual(done, {"0.jsonl.gz", "2.jsonl.gz"})
        self.assertEqual(
            served.opened.count("1.jsonl.gz"), 3
        )  # three tries, then it gives up
        self.assertGreaterEqual(
            api.files_calls, 3
        )  # a fresh URL listing before each retry (and the first)
        meta = dict(table(self.store, "SELECT key, value FROM meta"))
        self.assertEqual(meta["full_complete"], "0")
        self.assertEqual(table(self.store, "SELECT COUNT(*) FROM library_ids"), [(0,)])
        partial = set(rows(self.store))
        self.assertTrue(partial < self.expected(L))
        # the rerun streams only the missing shard
        again = Served(blobs)
        self.assertEqual(self.run_full(api, L, again), 0)
        self.assertEqual(again.opened, ["1.jsonl.gz"])
        self.assertEqual(set(rows(self.store)), self.expected(L))
        self.assertEqual(
            dict(table(self.store, "SELECT key, value FROM meta"))["full_complete"], "1"
        )

    def test_a_shard_that_dies_midway_leaves_nothing(self):
        names, blobs = self.shards3()
        L = self.library()

        class Dies(io.BytesIO):
            def read(self, n=-1):
                data = super().read(n)
                if self.tell() > 200:
                    raise ConnectionResetError("reset")
                return data

        def served(u):
            return (
                Dies(blobs[sync.basename(u)])
                if sync.basename(u) == "0.jsonl.gz"
                else io.BytesIO(blobs[sync.basename(u)])
            )

        api = FakeApi(shards=names)
        self.assertEqual(self.run_full(api, L, served), 1)
        self.assertNotIn(
            "0.jsonl.gz", {r[1] for r in table(self.store, "SELECT * FROM shards")}
        )
        ids0 = {json.loads(x)["citationid"] for x in self.lines[: len(self.lines) // 3]}
        self.assertFalse(
            ids0 & set(rows(self.store))
        )  # no row of the dead shard was committed
        self.assertEqual(glob.glob(os.path.join(self.dir, "s2store", "*.spool")), [])

    def test_retry_that_works_on_the_second_try(self):
        names, blobs = self.shards3()
        api = FakeApi(shards=names)
        served = Served(blobs, fail={"2.jsonl.gz": 1})
        self.assertEqual(self.run_full(api, self.library(), served), 0)
        self.assertEqual(served.opened.count("2.jsonl.gz"), 2)

    def test_a_different_latest_release_is_refused(self):
        names, blobs = self.shards3()
        self.assertEqual(
            self.run_full(FakeApi(shards=names), self.library(), Served(blobs)), 0
        )
        with self.assertRaises(sync.SyncError):
            self.run_full(
                FakeApi(release=R2, shards=names), self.library(), Served(blobs)
            )

    def test_changed_library_ids_mid_pass_are_refused(self):
        names, blobs = self.shards3()
        L = self.library()
        self.run_full(FakeApi(shards=names), L, Served(blobs, fail={"1.jsonl.gz": 99}))
        with self.assertRaises(sync.SyncError):
            self.run_full(FakeApi(shards=names), L | {123456789}, Served(blobs))

    def test_workers_run_under_nice_and_ionice_with_the_url_on_stdin(self):
        with mock.patch.object(sync.shutil, "which", lambda x: f"/usr/bin/{x}"):
            cmd = sync.worker_cmd()
        self.assertEqual(
            cmd[:6], ["nice", "-n", "19", "ionice", "-c3", sync.sys.executable]
        )
        self.assertEqual(cmd[-1], "--worker")
        job = {
            "url": url("citations", "0.jsonl.gz"),
            "store": "s",
            "release_id": R1,
            "basename": "0",
            "ids_file": "i",
        }
        done = mock.Mock(
            returncode=0, stdout='{"rows_seen": 5, "rows_kept": 2}\n', stderr=""
        )
        with mock.patch.object(sync.subprocess, "run", return_value=done) as run:
            ok, info = sync.spawn_worker(job)
        self.assertTrue(ok)
        self.assertEqual(info["rows_kept"], 2)
        args, kw = run.call_args
        self.assertNotIn(
            "SIGSIGSIG", " ".join(args[0])
        )  # not on the command line (ps shows it)
        self.assertIn("SIGSIGSIG", kw["input"])

    def test_worker_main_end_to_end(self):
        names, blobs = self.shards3()
        L = self.library()
        self.write_ids(L)
        job = {
            "store": self.store,
            "release_id": R1,
            "basename": "0.jsonl.gz",
            "url": url("c", "0.jsonl.gz"),
            "ids_file": self.ids_file,
        }
        with (
            mock.patch.object(sync, "open_stream", Served(blobs)),
            mock.patch("sys.stdin", io.StringIO(json.dumps(job))),
        ):
            rc = sync.worker_main()
        self.assertEqual(rc, 0)
        self.assertTrue(rows(self.store))
        # a failing worker reports without the URL
        with (
            mock.patch.object(
                sync, "open_stream", Served(blobs, fail={"0.jsonl.gz": 1})
            ),
            mock.patch("sys.stdin", io.StringIO(json.dumps(job))),
        ):
            self.assertEqual(sync.worker_main(), 1)
        self.assertNotIn("SIGSIGSIG", self.err.getvalue())


class DiffTest(Env):
    def seed(self, L):
        names, blobs = self.shards3()
        self.write_ids(L)
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            self.assertEqual(
                sync.run_full(
                    FakeApi(shards=names), self.store, self.ids_file, 1, in_process
                ),
                0,
            )
        self.in_store = sorted(rows(self.store))
        return self.in_store

    def files(self):
        L = self.library()
        ids = self.seed(L)
        x, y = ids[0], ids[1]  # an updated row, and a deleted one
        lib = next(iter(L))
        self.x, self.y, self.lib = x, y, lib
        old = rows(self.store)[x]
        blobs = {
            "uA": gz(
                [
                    enc(
                        rec(
                            x,
                            old[1],
                            old[2],
                            infl=True,
                            contexts=["new context"],
                            intents=[["result"]],
                        )
                    ),
                    enc(rec(900001, lib, 5000001)),
                    enc(rec(900002, 5000002, 5000003)),
                ]
            ),
            "dA": gz([enc({"citationid": y})]),
            "uB": gz([enc(rec(900003, 5000004, lib, contexts=None, intents=None))]),
            "dB": gz([enc({"citationid": x})]),
        }
        d1 = {
            "from_release": R1,
            "to_release": R2,
            "update_files": [url("d", "uA")],
            "delete_files": [url("d", "dA")],
        }
        d2 = {
            "from_release": R2,
            "to_release": R3,
            "update_files": [url("d", "uB")],
            "delete_files": [url("d", "dB")],
        }
        return blobs, d1, d2

    def test_diffs_apply_in_order_filtered_by_the_library(self):
        blobs, d1, d2 = self.files()
        api = FakeApi(release=R3, diffs=[d1, d2])
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            self.assertEqual(sync.run_diff(api, self.store, self.ids_file), 0)
        now = rows(self.store)
        self.assertNotIn(self.y, now)  # deleted by diff 1
        self.assertNotIn(
            self.x, now
        )  # updated by diff 1, deleted by diff 2: the order shows
        self.assertIn(900001, now)  # touches the library
        self.assertNotIn(900002, now)  # does not
        self.assertIn(900003, now)
        self.assertIsNone(now[900003][4])  # contexts null stays NULL
        meta = dict(table(self.store, "SELECT key, value FROM meta"))
        self.assertEqual((meta["release_id"], meta["mode"]), (R3, "diff"))
        self.assertEqual(len(self.in_store) - 2 + 2, len(now))

    def test_upsert_replaces_the_row_by_citationid(self):
        blobs, d1, d2 = self.files()
        api = FakeApi(release=R2, diffs=[d1])
        before = rows(self.store)[self.x]
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            sync.run_diff(api, self.store, self.ids_file)
        after = rows(self.store)[self.x]
        self.assertEqual((before[3], after[3]), (before[3], 1))
        self.assertEqual(json.loads(after[4]), ["new context"])
        self.assertEqual(json.loads(after[5]), [["result"]])
        self.assertEqual(
            dict(table(self.store, "SELECT key, value FROM meta"))["release_id"], R2
        )

    def test_a_delete_record_without_citationid_stops_and_the_release_advances_one_diff_at_a_time(
        self,
    ):
        blobs, d1, d2 = self.files()
        blobs["dB"] = gz([enc({"citedcorpusid": 1, "citingcorpusid": 2})])

        def opener(method, u, headers, body):
            return (
                {"release_id": R3}
                if u.endswith("/release/latest")
                else {"diffs": [d1, d2]}
            )

        client = s2_api.S2Client(
            KEY, opener=opener, clock=lambda: 0.0, sleep=lambda s: None
        )
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            rc = sync.main(
                ["--store", self.store, "--library-ids", self.ids_file, "--diff"],
                client=client,
            )
        self.assertEqual(rc, 1)
        self.assertIn("citationid", self.err.getvalue())
        self.assertNotIn(KEY, self.err.getvalue())
        meta = dict(table(self.store, "SELECT key, value FROM meta"))
        self.assertEqual(meta["release_id"], R2)  # diff 1 is in, diff 2 is not
        now = rows(self.store)
        self.assertIn(self.x, now)  # diff 2 deleted nothing
        self.assertNotIn(900003, now)
        self.assertNotIn(self.y, now)
        self.assertEqual(glob.glob(os.path.join(self.dir, "s2store", "*.spool")), [])

    def test_scan_deletes(self):
        self.assertEqual(
            sync.scan_deletes(
                iter([b'{"citationid": 4}', b"", b'{"citationid": 9, "x": 1}'])
            ),
            [4, 9],
        )
        for bad in (
            b'{"citingcorpusid": 4}',
            b'{"citationid": null}',
            b'{"citationid": "7"}',
            b'{"citationid": true}',
        ):
            with self.assertRaises(sync.SyncError):
                sync.scan_deletes(iter([bad]))

    def test_up_to_date_and_unfinished_store(self):
        self.seed(self.library())
        self.assertEqual(
            sync.run_diff(FakeApi(release=R1), self.store, self.ids_file), 0
        )
        con = sqlite3.connect(self.store)
        con.execute("UPDATE meta SET value = '0' WHERE key = 'full_complete'")
        con.commit()
        con.close()
        with self.assertRaises(sync.SyncError):
            sync.run_diff(FakeApi(release=R2), self.store, self.ids_file)

    def test_a_broken_diff_chain_is_refused(self):
        blobs, d1, d2 = self.files()
        with self.assertRaises(sync.SyncError):
            sync.run_diff(
                FakeApi(release=R3, diffs=[d2]), self.store, self.ids_file
            )  # gap at R1
        with self.assertRaises(sync.SyncError):
            sync.run_diff(
                FakeApi(release=R3, diffs=[d1]), self.store, self.ids_file
            )  # stops short

    def test_a_diff_file_that_fails_once_is_retried_with_a_fresh_listing(self):
        blobs, d1, d2 = self.files()
        api = FakeApi(release=R2, diffs=[d1])
        with mock.patch.object(sync, "open_stream", Served(blobs, fail={"uA": 1})):
            self.assertEqual(sync.run_diff(api, self.store, self.ids_file), 0)
        self.assertGreaterEqual(api.diffs_calls, 2)
        self.assertEqual(
            table(
                self.store, "SELECT COUNT(*) FROM citations WHERE citationid = 900001"
            ),
            [(1,)],
        )  # once


class NewIdsTest(Env):
    def test_grown_library_warns_and_backfill_restreams_only_for_the_new_ids(self):
        names, blobs = self.shards3()
        L = self.library()
        self.write_ids(L)
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            sync.run_full(
                FakeApi(shards=names), self.store, self.ids_file, 2, in_process
            )
        before = set(rows(self.store))
        extra = next(
            r["citingcorpusid"]
            for r in self.recs
            if r["citationid"] not in before and r["citingcorpusid"] not in L
        )
        L2 = L | {extra}
        self.write_ids(L2)
        api = FakeApi(shards=names)
        sync.run_diff(api, self.store, self.ids_file)  # same release: nothing to apply
        self.assertIn("1 new library id", self.err.getvalue())
        self.assertEqual(
            set(rows(self.store)), before
        )  # without the flag it only warns
        served = Served(blobs)
        with mock.patch.object(sync, "open_stream", served):
            self.assertEqual(
                sync.run_backfill(api, self.store, self.ids_file, 2, in_process), 0
            )
        self.assertEqual(sorted(served.opened), sorted(names))
        self.assertEqual(set(rows(self.store)), self.expected(L2))
        self.assertEqual(
            {r[0] for r in table(self.store, "SELECT corpusid FROM library_ids")}, L2
        )
        keys = {r[0] for r in table(self.store, "SELECT release_id FROM shards")}
        self.assertEqual(len(keys), 2)  # the full pass and the backfill pass
        meta = dict(table(self.store, "SELECT key, value FROM meta"))
        self.assertEqual(meta["library_corpus_ids_hash"], sync.ids_hash(L2))
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            self.assertEqual(
                sync.run_backfill(api, self.store, self.ids_file, 2, in_process), 0
            )  # nothing left


class KeyHandlingTest(Env):
    def test_datasets_calls_carry_the_key_only_as_a_header_to_the_api_host(self):
        seen = []

        def opener(method, u, headers, body):
            seen.append((method, u, dict(headers)))
            if u.endswith("/release/latest"):
                return {"release_id": R1}
            if "/dataset/citations" in u:
                return {"files": [url("c", "0.jsonl.gz")]}
            return {"diffs": []}

        c = s2_api.S2Client(KEY, opener=opener, clock=lambda: 0.0, sleep=lambda s: None)
        api = sync.Api(c)
        self.assertEqual(api.latest(), R1)
        self.assertEqual(len(api.files(R1)), 1)
        self.assertEqual(api.diffs(R1, R2), [])
        self.assertEqual(
            [u for _, u, _ in seen],
            [
                "https://api.semanticscholar.org/datasets/v1/release/latest",
                f"https://api.semanticscholar.org/datasets/v1/release/{R1}/dataset/citations",
                f"https://api.semanticscholar.org/datasets/v1/diffs/{R1}/to/{R2}/citations",
            ],
        )
        for _, u, h in seen:
            self.assertNotIn(KEY, u)
            self.assertEqual(h["x-api-key"], KEY)

    def test_a_rejected_key_stops_the_sync(self):
        def opener(method, u, headers, body):
            if "x-api-key" in headers:
                raise urllib.error.HTTPError(u, 403, "no", {}, None)
            return {"release_id": R1}

        c = s2_api.S2Client(
            KEY,
            opener=opener,
            clock=lambda: 0.0,
            sleep=lambda s: None,
            warn=lambda m: None,
        )
        with self.assertRaises(sync.SyncError):
            sync.Api(c).latest()

    def test_s3_urls_get_no_key_even_when_the_env_has_one(self):
        captured = []

        class Resp(io.BytesIO):
            headers = {"Content-Length": "123"}

            def __enter__(self):
                return self

        def fake_urlopen(req, timeout=None):
            captured.append(
                (
                    req.get_method(),
                    req.full_url,
                    {k.lower(): v for k, v in req.header_items()},
                )
            )
            return Resp(b"")

        u = url("citations", "0.jsonl.gz")
        with (
            mock.patch.dict(os.environ, {"S2_API_KEY": KEY}),
            mock.patch.object(urllib.request, "urlopen", fake_urlopen),
        ):
            sync.open_stream(u).close()
            self.assertEqual(sync.head_size(u), 123)
        self.assertEqual([c[0] for c in captured], ["GET", "HEAD"])
        for _, _, h in captured:
            self.assertNotIn("x-api-key", h)
            self.assertEqual(h["user-agent"], "zotero-citegraph/2")

    def test_head_refused_falls_back_to_a_one_byte_range(self):
        calls = []

        class Resp(io.BytesIO):
            headers = {"Content-Range": "bytes 0-0/9000"}

            def __enter__(self):
                return self

        def fake_urlopen(req, timeout=None):
            calls.append(
                (req.get_method(), {k.lower(): v for k, v in req.header_items()})
            )
            if req.get_method() == "HEAD":
                raise urllib.error.HTTPError(
                    req.full_url, 403, "signed for GET", {}, None
                )
            return Resp(b"x")

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertEqual(sync.head_size(url("c", "0")), 9000)
        self.assertEqual(calls[1][1]["range"], "bytes=0-0")
        self.assertNotIn("x-api-key", calls[1][1])


class MeasureAndCliTest(Env):
    def test_measure_lists_sizes_and_downloads_nothing(self):
        names, _ = self.shards3()
        api = FakeApi(shards=names)
        sizes = {
            url(f"citations/{R1}", n): (i + 1) * 2**30 for i, n in enumerate(names)
        }
        heads = []

        def head(u):
            heads.append(sync.basename(u))
            return sizes[u]

        def no_stream(u):
            raise AssertionError("downloaded")

        with mock.patch.object(sync, "open_stream", no_stream):
            out = sync.run_measure(api, self.store, head)
        self.assertEqual(heads, names)
        self.assertEqual(sum(out["shards"].values()), 6 * 2**30)
        self.assertIsNone(out["diffs"])  # no store yet
        shown = sys.stdout.getvalue()
        self.assertIn("6.0 GiB", shown)
        self.assertNotIn("SIGSIGSIG", shown)  # the pre-signed URL is never printed

    def test_measure_diffs_when_the_store_is_behind(self):
        names, blobs = self.shards3()
        self.write_ids(self.library())
        with mock.patch.object(sync, "open_stream", Served(blobs)):
            sync.run_full(
                FakeApi(shards=names), self.store, self.ids_file, 1, in_process
            )
        d = {
            "from_release": R1,
            "to_release": R2,
            "update_files": [url("d", "u")],
            "delete_files": [url("d", "x")],
        }
        out = sync.run_measure(
            FakeApi(release=R2, shards=names, diffs=[d]), self.store, lambda u: 1024**2
        )
        self.assertEqual(len(out["diffs"]), 2)

    def test_main_needs_a_key_and_valid_modes(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            self.assertEqual(sync.main(["--store", self.store, "--measure"]), 1)
        self.assertIn("S2_API_KEY", self.err.getvalue())
        self.assertNotIn(KEY, self.err.getvalue())
        for argv in (
            ["--store", self.store],
            ["--store", self.store, "--full", "--diff", "--library-ids", self.ids_file],
            ["--store", self.store, "--full"],
            ["--measure"],
        ):
            with self.assertRaises(SystemExit):
                sync.main(argv)


if __name__ == "__main__":
    unittest.main()
