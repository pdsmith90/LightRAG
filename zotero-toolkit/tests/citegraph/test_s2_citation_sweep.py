import collections
import contextlib
import datetime
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
import urllib.parse

# The modules under test sit in citegraph/ and import one another by name.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "citegraph")
)

import build_struct_layer as bsl  # noqa: E402
import s2_api  # noqa: E402
import s2_citation_sweep as sw  # noqa: E402
from test_s2_api import FakeOpener, FakeTime, client, http  # noqa: E402

PAPER_COLS = [
    "paper_id",
    "openalex_id",
    "s2_paper_id",
    "s2_corpus_id",
    "s2_citation_count",
]


def cite(citing, ctx=None, infl=False, **kw):
    rec = {
        "citingPaper": {"paperId": f"{citing:040x}", "corpusId": citing},
        "isInfluential": infl,
    }
    if ctx is not None:
        rec["contextsWithIntent"] = [{"context": c, "intents": i} for c, i in ctx]
    rec.update(kw)
    return rec


def page(data, nxt=None):
    out = {"offset": 0, "data": data}
    if nxt is not None:
        out["next"] = nxt
    return out


def target(n, count=3, **kw):
    return {
        "paper_id": f"z:{n}",
        "s2_paper_id": f"P{n}",
        "corpus_id": n,
        "count": count,
        **kw,
    }


def q(url):
    return urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)


class Env:
    """A temp store, a fake clock and a fake opener."""

    def __init__(self, tc, *answers, **kw):
        self.t = FakeTime()
        d = tempfile.TemporaryDirectory()
        tc.addCleanup(d.cleanup)
        self.dir = d.name
        self.store = os.path.join(d.name, "s2store", "api.sqlite")
        self.op = FakeOpener(*answers)
        self.stats = collections.Counter()
        self.client = s2_api.S2Client(
            "",
            opener=sw.counting_opener(self.op, self.stats),
            clock=self.t.clock,
            sleep=self.t.sleep,
            warn=lambda m: None,
            retries=sw.RETRIES,
            backoff0=s2_api.BACKOFF0,
            backoff_max=sw.BACKOFF_MAX,
            **kw,
        )
        self.con = sw.open_store(self.store)
        tc.addCleanup(self.con.close)
        self.logs = []

    def run(self, targets, library, **kw):
        sw.init_store(self.con, library)
        return sw.run_sweep(
            self.con,
            self.client,
            targets,
            library,
            stats=self.stats,
            clock=self.t.clock,
            log=self.logs.append,
            **kw,
        )

    def rows(
        self,
        sql="SELECT citationid, citing, cited, isinfluential, contexts, intents FROM citations ORDER BY 1",
    ):
        return self.con.execute(sql).fetchall()


class SelectTest(unittest.TestCase):
    def papers(self):
        def p(pid, oa, s2, cid, n):
            return dict(zip(PAPER_COLS, [pid, oa, s2, cid, n]))

        return [
            p("a", "W1", "S1", "11", "50"),  # cited by the library: target
            p("b", "W2", "S2", "12", "5"),  # not cited, has OpenAlex refs: not needed
            p("c", "", "S3", "13", "7"),  # rule (b): no openalex id
            p("d", "W4", "", "14", "9"),  # no s2 id
            p("e", "", "S5", "15", "0"),  # zero citations
            p("f", "W6", "S6", "16", "5"),  # cited: target, same count as c
            p("g", "", "S7", "", "4"),  # no corpus id: cannot be a cited end
            p("h", "W8", "S8", "18", ""),  # no count
        ]

    def test_rules_order_and_exclusions(self):
        lib, targets, info = sw.select_targets(self.papers(), {"a", "f", "d", "e", "h"})
        self.assertEqual(lib, {11, 12, 13, 14, 15, 16, 18})
        self.assertEqual(
            [t["paper_id"] for t in targets], ["f", "c", "a"]
        )  # count 5, 7, 50
        self.assertEqual(
            targets[0],
            {
                "paper_id": "f",
                "s2_paper_id": "S6",
                "corpus_id": 16,
                "count": 5,
                "citers": [],
            },
        )
        self.assertEqual(
            (
                info["no_s2_id"],
                info["zero_citations"],
                info["no_corpus_id"],
                info["not_needed"],
            ),
            (1, 2, 1, 1),
        )

    def test_equal_counts_ordered_by_paper_id(self):
        ps = [
            dict(zip(PAPER_COLS, ["z", "", "S1", "1", "3"])),
            dict(zip(PAPER_COLS, ["m", "", "S2", "2", "3"])),
        ]
        _, targets, _ = sw.select_targets(ps, set())
        self.assertEqual([t["paper_id"] for t in targets], ["m", "z"])

    def test_load_inputs_reads_the_builders_csv(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "papers.csv"), "w", newline="") as f:
                f.write(
                    ",".join(PAPER_COLS) + ",title\na,W1,S1,11,5,T\nb,W2,S2,12,5,U\n"
                )
            with open(os.path.join(d, "cites.csv"), "w", newline="") as f:
                f.write("citing_paper_id,cited_paper_id,source\nb,a,openalex\n")
            lib, targets, _ = sw.load_inputs(d)
        self.assertEqual(lib, {11, 12})
        self.assertEqual([t["paper_id"] for t in targets], ["a"])


class ConvertTest(unittest.TestCase):
    LIB = frozenset({1, 2, 3})

    def test_surrogate_id_formula(self):
        self.assertEqual(sw.surrogate_id(5, 7), -(5 * 2**31 + 7))
        self.assertLess(sw.surrogate_id(1, 1), 0)
        self.assertNotEqual(sw.surrogate_id(1, 2), sw.surrogate_id(2, 1))
        big = 2**31 - 1
        self.assertEqual(
            len({sw.surrogate_id(a, b) for a in (0, 1, big) for b in (0, 1, big)}), 9
        )

    def test_filter_keeps_library_citers_and_drops_self_and_unknown(self):
        self.assertEqual(sw.convert(cite(2), 1, self.LIB)["citing"], 2)
        self.assertIsNone(sw.convert(cite(1), 1, self.LIB))  # self-citation
        self.assertIsNone(sw.convert(cite(99), 1, self.LIB))  # not in the library
        self.assertIsNone(
            sw.convert({"citingPaper": {"paperId": "x", "corpusId": None}}, 1, self.LIB)
        )
        self.assertIsNone(sw.convert({"citingPaper": None}, 1, self.LIB))
        self.assertIsNone(sw.convert(None, 1, self.LIB))

    def test_corpus_id_comes_back_as_a_string(self):
        # live answer (2026-11-24): "citingPaper": {"paperId": "...", "corpusId": "290890461"}
        rec = {
            "citingPaper": {"paperId": "x", "corpusId": "2"},
            "contextsWithIntent": [],
            "contexts": ["c"],
            "intents": [],
        }
        self.assertEqual(sw.convert(rec, 1, self.LIB)["citing"], 2)

    def test_contexts_with_intent_conversion(self):
        r = sw.convert(
            cite(
                2,
                [("first", ["methodology"]), ("second", [])],
                infl=True,
                intents=["background"],
                contexts=["IGNORED"],
            ),
            1,
            self.LIB,
        )
        self.assertEqual(r["contexts"], ["first", "second"])
        self.assertEqual(
            r["intents"], [["methodology"], []]
        )  # the flat list is not copied onto each context
        self.assertEqual((r["infl"], r["kind"]), (1, "cwi"))

    def test_fallback_without_contexts_with_intent(self):
        r = sw.convert(
            cite(2, contexts=["a", "b"], intents=["background", "result"]), 1, self.LIB
        )
        self.assertEqual(
            (r["contexts"], r["intents"], r["kind"]), (["a", "b"], [[], []], "flat")
        )
        for rec in (cite(2, contexts=[], intents=["background"]), cite(2, ctx=[])):
            r = sw.convert(rec, 1, self.LIB)
            self.assertEqual(
                (r["contexts"], r["intents"], r["kind"], r["infl"]), ([], [], "none", 0)
            )


def serve(pages):
    """An opener that answers /paper/<id>/citations by (id, offset)."""

    def answer(method, url, headers, body):
        pid = urllib.parse.urlsplit(url).path.split("/")[-2]
        off = int(q(url)["offset"][0])
        a = pages[(pid, off)]
        if callable(a):
            a = a()
        if isinstance(a, Exception):
            raise a
        return a

    return answer


class PagingTest(unittest.TestCase):
    def test_follows_next_and_asks_for_the_right_fields(self):
        env = Env(
            self,
            serve(
                {
                    ("P1", 0): page(
                        [cite(2, [("x", ["background"])]), cite(50)], nxt=1000
                    ),
                    ("P1", 1000): page([cite(3, [("y", [])], infl=True)]),
                }
            ),
        )
        s = env.run([target(1, count=3)], {1, 2, 3})
        self.assertEqual(s["complete"], True)
        self.assertEqual(
            [r["url"].split("?")[0].rsplit("/", 2)[1:] for r in env.op.requests],
            [["P1", "citations"]] * 2,
        )
        qs = [q(r["url"]) for r in env.op.requests]
        self.assertEqual([x["offset"] for x in qs], [["0"], ["1000"]])
        self.assertEqual(qs[0]["limit"], ["1000"])
        self.assertEqual(
            qs[0]["fields"],
            ["corpusId,contexts,intents,contextsWithIntent,isInfluential"],
        )
        for r in env.op.requests:
            self.assertNotIn("x-api-key", r["headers"])
        self.assertEqual(sorted((r[1], r[2]) for r in env.rows()), [(2, 1), (3, 1)])
        sw_row = env.con.execute(
            "SELECT status, pages, rows_seen, rows_kept, truncated FROM swept"
        ).fetchone()
        self.assertEqual(sw_row, ("ok", 2, 3, 2, 0))

    def test_refused_offset_gives_truncated_with_what_was_read(self):
        env = Env(
            self, serve({("P1", 0): page([cite(2)], nxt=1000), ("P1", 1000): http(400)})
        )
        s = env.run([target(1, count=5000)], {1, 2})
        self.assertEqual(len(env.op.requests), 2)  # a 400 is not retried
        self.assertEqual(s["swept_truncated"], 1)
        self.assertEqual(
            env.con.execute(
                "SELECT status, truncated, rows_kept FROM swept"
            ).fetchone(),
            ("truncated", 1, 1),
        )
        self.assertEqual(len(env.rows()), 1)

    def test_next_stopping_early_is_truncated(self):
        env = Env(
            self, serve({("P1", 0): page([cite(2)] * 1)})
        )  # no next, 1 row for a count of 100
        env.run([target(1, count=100)], {1, 2})
        self.assertEqual(
            env.con.execute("SELECT status, truncated FROM swept").fetchone(),
            ("truncated", 1),
        )

    def test_non_advancing_next_does_not_loop(self):
        env = Env(self, serve({("P1", 0): page([cite(2)], nxt=0)}))
        env.run([target(1, count=1)], {1, 2})
        self.assertEqual(len(env.op.requests), 1)
        self.assertEqual(
            env.con.execute("SELECT status FROM swept").fetchone(), ("truncated",)
        )

    def test_not_found_first_page_is_recorded_and_not_a_failure(self):
        env = Env(self, serve({("P1", 0): http(404)}))
        s = env.run([target(1)], {1})
        self.assertEqual(
            (s["swept_notfound"], s["failed"], s["complete"]), (1, 0, True)
        )
        self.assertEqual(
            env.con.execute("SELECT status FROM swept").fetchone(), ("notfound",)
        )


class StoreTest(unittest.TestCase):
    def test_row_shape_and_meta(self):
        env = Env(
            self,
            serve(
                {("P1", 0): page([cite(2, [("ctx é", ["methodology"])], infl=True)])}
            ),
        )
        env.run([target(1, count=1)], {1, 2})
        self.assertEqual(
            env.rows(),
            [(sw.surrogate_id(2, 1), 2, 1, 1, '["ctx é"]', '[["methodology"]]')],
        )
        meta = dict(env.con.execute("SELECT key, value FROM meta"))
        self.assertEqual((meta["release_id"], meta["mode"]), ("graph-api", "api"))
        self.assertIn("updated_at", meta)
        self.assertEqual(
            {r[0] for r in env.con.execute("SELECT corpusid FROM library_ids")}, {1, 2}
        )

    def test_library_ids_replaced_on_each_run(self):
        env = Env(self, serve({}))
        sw.init_store(env.con, {1, 2, 3})
        sw.init_store(env.con, {2, 9})
        self.assertEqual(
            {r[0] for r in env.con.execute("SELECT corpusid FROM library_ids")}, {2, 9}
        )

    def test_one_commit_per_paper(self):
        env = Env(self, serve({(f"P{n}", 0): page([cite(9)]) for n in (1, 2, 3)}))
        sw.init_store(env.con, {1, 2, 3, 9})
        trace = []
        env.con.set_trace_callback(trace.append)
        sw.run_sweep(
            env.con,
            env.client,
            [target(n, count=1) for n in (1, 2, 3)],
            {1, 2, 3, 9},
            clock=env.t.clock,
            log=env.logs.append,
        )
        env.con.set_trace_callback(None)
        self.assertEqual(sum(1 for s in trace if s.startswith("COMMIT")), 3)
        self.assertEqual(sum(1 for s in trace if s.startswith("BEGIN")), 3)

    def test_mid_paper_failure_leaves_old_rows_and_marks_failed(self):
        ok = serve({("P1", 0): page([cite(2, [("old", [])])])})
        env = Env(self, ok)
        env.run([target(1, count=1)], {1, 2, 3})
        before = env.rows()
        self.assertEqual(len(before), 1)
        # second run: page 1 fine (a new citer), page 2 fails after the retries
        env.op.answers = [
            serve(
                {
                    ("P1", 0): page([cite(3, [("new", [])])], nxt=1000),
                    ("P1", 1000): http(503),
                }
            )
        ]
        s = env.run([target(1, count=2000)], {1, 2, 3}, refresh=True)
        self.assertEqual(env.rows(), before)  # untouched
        self.assertEqual((s["failed"], s["complete"]), (1, False))
        self.assertEqual(
            env.con.execute("SELECT status FROM swept").fetchone(), ("failed",)
        )
        self.assertEqual(
            env.con.execute("SELECT rows_kept FROM swept").fetchone(), (1,)
        )  # the old count stays
        self.assertIn(4.0, env.t.sleeps)
        self.assertEqual(
            sorted(x for x in env.t.sleeps if x >= 4), [4, 8, 16, 32, 64, 128, 256, 300]
        )  # 8 retries, cap 300
        # a failed paper is retried by the next run even without --refresh
        env.op.answers = [serve({("P1", 0): page([cite(3, [("new", [])])])})]
        s = env.run([target(1, count=1)], {1, 2, 3})
        self.assertEqual(s["swept_ok"], 1)
        self.assertEqual([r[1] for r in env.rows()], [3])  # old row replaced

    def test_rows_of_other_papers_are_not_deleted(self):
        env = Env(self, serve({("P1", 0): page([cite(2)]), ("P5", 0): page([cite(2)])}))
        env.run([target(1, count=1), target(5, count=1)], {1, 2, 5})
        env.op.answers = [serve({("P1", 0): page([])})]
        env.run([target(1, count=0)], {1, 2, 5}, refresh=True)
        self.assertEqual([(r[1], r[2]) for r in env.rows()], [(2, 5)])


class ResumeTest(unittest.TestCase):
    def test_skip_within_max_age_and_refresh(self):
        env = Env(self, serve({("P1", 0): page([cite(2)])}))
        env.run([target(1, count=1)], {1, 2})
        n = len(env.op.requests)
        s = env.run([target(1, count=1)], {1, 2})
        self.assertEqual((len(env.op.requests), s["skipped_fresh"]), (n, 1))
        s = env.run([target(1, count=1)], {1, 2}, refresh=True)
        self.assertEqual((len(env.op.requests), s["skipped_fresh"]), (n + 1, 0))
        # an old sweep is done again
        env.con.execute("UPDATE swept SET fetched_at = '2020-01-01T00:00:00Z'")
        env.run([target(1, count=1)], {1, 2})
        self.assertEqual(len(env.op.requests), n + 2)
        # --max-age-days widens the window
        env.con.execute("UPDATE swept SET fetched_at = '2020-01-01T00:00:00Z'")
        env.run([target(1, count=1)], {1, 2}, max_age_days=10**5)
        self.assertEqual(len(env.op.requests), n + 2)

    def test_truncated_is_skipped_failed_is_not(self):
        env = Env(self, serve({("P1", 0): page([cite(2)]), ("P2", 0): http(503)}))
        t = [target(1, count=500), target(2, count=1)]  # P1 truncated (1 of 500)
        env.run(t, {1, 2})
        n = len(env.op.requests)
        s = env.run(t, {1, 2})
        self.assertEqual(s["skipped_fresh"], 1)  # P1 only
        self.assertGreater(len(env.op.requests), n)  # P2 asked again


class StopTest(unittest.TestCase):
    def test_ten_failed_papers_in_a_row_exit_3_and_keep_committed(self):
        pages = {("P1", 0): page([cite(99)])}
        pages.update({(f"P{n}", 0): http(500) for n in range(2, 14)})
        env = Env(self, serve(pages))
        s = env.run([target(n, count=n) for n in range(1, 14)], {1, 99})
        self.assertEqual(
            (s["exit"], s["stopped"], s["failed"], s["complete"]),
            (3, "throttled", 10, False),
        )
        self.assertEqual(
            env.con.execute(
                "SELECT COUNT(*) FROM swept WHERE status='failed'"
            ).fetchone(),
            (10,),
        )
        self.assertEqual(
            env.con.execute("SELECT status FROM swept WHERE cited = 1").fetchone(),
            ("ok",),
        )
        self.assertEqual(
            env.con.execute("SELECT COUNT(*) FROM swept").fetchone(), (11,)
        )  # 12 and 13 never asked

    def test_a_success_resets_the_failure_run(self):
        pages = {(f"P{n}", 0): http(500) for n in range(1, 25)}
        for n in (10, 20):
            pages[(f"P{n}", 0)] = page([])
        env = Env(self, serve(pages))
        s = env.run([target(n, count=n) for n in range(1, 25)], {1})
        # runs of 9, 9 and 4 failures: 22 failed in all, but never 10 in a row
        self.assertEqual(
            (s["exit"], s["stopped"], s["failed"], s["complete"]), (0, None, 22, False)
        )
        self.assertEqual(
            env.con.execute("SELECT status FROM swept WHERE cited = 10").fetchone()[0],
            "truncated",
        )

    def test_max_calls_stops_cleanly(self):
        env = Env(
            self,
            serve({(f"P{n}", 0): page([cite(9)]) for n in range(1, 6)}),
            max_calls=3,
        )
        s = env.run([target(n, count=1) for n in range(1, 6)], {9, 1, 2, 3, 4, 5})
        self.assertEqual(
            (s["exit"], s["complete"], s["stopped"], s["swept_ok"]),
            (0, False, "max-calls", 3),
        )
        self.assertEqual(env.client.calls, 3)
        self.assertEqual(
            env.con.execute("SELECT COUNT(*) FROM swept").fetchone(), (3,)
        )  # no half paper recorded

    def test_max_hours_stops_cleanly(self):
        env = Env(self, serve({(f"P{n}", 0): page([cite(9)]) for n in range(1, 6)}))
        # each paper costs about 2 s of fake pacing; one hour is 3600 s
        s = env.run(
            [target(n, count=1) for n in range(1, 6)], {9}, max_hours=2.5 / 3600
        )
        self.assertEqual(
            (s["exit"], s["complete"], s["stopped"]), (0, False, "max-hours")
        )
        self.assertTrue(0 < s["swept_ok"] < 5)

    def test_sigterm_flag_stops_after_the_current_paper(self):
        env = Env(self, serve({(f"P{n}", 0): page([cite(9)]) for n in range(1, 6)}))
        stop = {"flag": False}
        orig = env.op

        def trip(method, url, headers, body):
            stop["flag"] = True  # raised during paper 1
            return orig(method, url, headers, body)

        env.client.opener = sw.counting_opener(trip, env.stats)
        s = env.run([target(n, count=1) for n in range(1, 6)], {9}, stop=stop)
        self.assertEqual(
            (s["stopped"], s["swept_ok"], s["exit"]), ("sigterm", 1, 0)
        )  # paper 1 was committed

    def test_progress_line_every_50_papers(self):
        env = Env(self, serve({(f"P{n}", 0): page([cite(9)]) for n in range(1, 121)}))
        env.run([target(n, count=1) for n in range(1, 121)], {9})
        lines = [m for m in env.logs if m.startswith("sweep: ")]
        self.assertEqual(len(lines), 2)
        for word in ("done", "left", "calls", "429s", "pace", "rows kept", "ETA"):
            self.assertIn(word, lines[0])


class RetryParamsTest(unittest.TestCase):
    def test_429_counted_and_defaults_of_the_builder_unchanged(self):
        env = Env(self, http(429), http(429), page([]))
        env.run([target(1, count=0)], {1})
        self.assertEqual(env.stats["429"], 2)
        # the builder's client keeps 6 retries and a 120 s cap
        op = FakeOpener(http(503))
        c, t = client(op)
        with self.assertRaises(s2_api.S2Error):
            c.get("/x", {})
        self.assertEqual((len(op.requests), max(t.sleeps)), (7, 120))

    def test_no_key_is_ever_sent_even_with_s2_api_key_set(self):
        env = Env(self, serve({("P1", 0): page([])}))
        with unittest.mock.patch.dict(os.environ, {"S2_API_KEY": "SECRETKEY"}):
            env.run([target(1, count=0)], {1})
        for r in env.op.requests:
            self.assertNotIn("x-api-key", r["headers"])
            self.assertNotIn("SECRETKEY", r["url"])


def iso(days_ago):
    d = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


class DueTest(unittest.TestCase):
    """Follow-up 1: a fresh target is due again when an in-library citer first appeared after its fetch."""

    def papers(self, extra=()):
        rows = [
            ("A", "W1", "S1", "1", "5"),
            ("B", "W2", "", "2", "0"),
            ("C", "W3", "S3", "3", "5"),
            ("D", "W4", "", "4", "0"),
        ]
        return [dict(zip(PAPER_COLS, r)) for r in rows + list(extra)]

    CITES = {"A": ["B"], "C": ["D"]}

    def swept_env(self):
        """A and C swept ok, fetched a day ago; every library id first seen two days ago."""
        pages = {("S1", 0): page([cite(2)]), ("S3", 0): page([cite(4)])}
        env = Env(self, serve(pages))
        lib, targets, _ = sw.select_targets(self.papers(), {"A", "C"}, self.CITES)
        self.assertEqual([t["citers"] for t in targets], [[2], [4]])
        env.run(targets, lib)
        env.con.execute("UPDATE swept SET fetched_at = ?", (iso(1),))
        env.con.execute("UPDATE library_seen SET first_seen = ?", (iso(2),))
        env.requests_before = len(env.op.requests)
        return env

    def test_new_citer_makes_exactly_its_cited_target_due(self):
        env = self.swept_env()
        new = [("E", "W5", "", "5", "0")]
        cites = {"A": ["B", "E"], "C": ["D"]}
        lib, targets, _ = sw.select_targets(self.papers(new), {"A", "C"}, cites)
        s = env.run(targets, lib)
        self.assertEqual(s["due"], {"new_citer": 1})
        asked = [
            r["url"].split("/paper/")[1].split("/")[0]
            for r in env.op.requests[env.requests_before :]
        ]
        self.assertEqual(asked, ["S1"])  # A only; C was not asked
        self.assertEqual(s["skipped_fresh"], 1)
        # the new id is first seen now, older ids kept their time
        seen = dict(env.con.execute("SELECT corpusid, first_seen FROM library_seen"))
        self.assertEqual(seen[1], iso(2))
        self.assertGreater(seen[5], iso(1))
        # A was fetched again after E appeared: nothing is due on the next run
        s = env.run(targets, lib)
        self.assertEqual((s["due"], s["skipped_fresh"]), ({}, 2))

    def test_an_old_citer_does_not_make_a_target_due(self):
        env = self.swept_env()
        # E is in cites.csv but was first seen before A was fetched
        env.con.execute("INSERT INTO library_seen VALUES (5, ?)", (iso(3),))
        lib, targets, _ = sw.select_targets(
            self.papers([("E", "W5", "", "5", "0")]),
            {"A", "C"},
            {"A": ["B", "E"], "C": ["D"]},
        )
        s = env.run(targets, lib)
        self.assertEqual(
            (s["due"], s["skipped_fresh"], len(env.op.requests)),
            ({}, 2, env.requests_before),
        )

    def test_first_seen_never_moves(self):
        env = Env(self, serve({}))
        sw.init_store(env.con, {1, 2})
        env.con.execute("UPDATE library_seen SET first_seen = '2000-01-01T00:00:00Z'")
        sw.init_store(env.con, {1, 2, 3})
        sw.init_store(env.con, {2, 3})  # 1 left L: its row stays too
        seen = dict(env.con.execute("SELECT corpusid, first_seen FROM library_seen"))
        self.assertEqual((seen[1], seen[2]), ("2000-01-01T00:00:00Z",) * 2)
        self.assertGreater(seen[3], "2020")
        self.assertEqual(
            {r[0] for r in env.con.execute("SELECT corpusid FROM library_ids")}, {2, 3}
        )  # replaced as before

    def test_reasons(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        t = {"citers": [9]}

        def f(status, fetched, seen=None, refresh=False):
            return sw.due_reason(t, status, fetched, seen or {}, now, 30, refresh)

        self.assertEqual(f(None, None), "never")
        self.assertEqual(f("failed", iso(1)), "failed")
        self.assertEqual(f("ok", iso(40)), "age")
        self.assertIsNone(f("truncated", iso(1)))
        self.assertIsNone(f("notfound", iso(1), {9: iso(2)}))
        self.assertEqual(f("ok", iso(1), {9: iso(0)}), "new_citer")
        self.assertEqual(f("ok", iso(1), {}, refresh=True), "refresh")
        self.assertIsNone(
            sw.due_reason({}, "ok", iso(1), {9: iso(0)}, now, 30)
        )  # no citers listed

    def test_report_counts_due_targets_by_reason(self):
        env = self.swept_env()
        lib, targets, _ = sw.select_targets(
            self.papers([("E", "W5", "", "5", "0")]),
            {"A", "C"},
            {"A": ["B", "E"], "C": ["D"]},
        )
        env.con.execute("INSERT INTO library_seen VALUES (5, ?)", (iso(0),))
        self.assertEqual(
            sw.report(env.store, targets)["targets_due"],
            {"never": 0, "failed": 0, "age": 0, "new_citer": 1},
        )
        env.con.execute("UPDATE swept SET status = 'failed' WHERE cited = 3")
        env.con.execute("UPDATE swept SET fetched_at = ? WHERE cited = 1", (iso(200),))
        self.assertEqual(
            sw.report(env.store, targets)["targets_due"],
            {"never": 0, "failed": 1, "age": 1, "new_citer": 0},
        )
        self.assertIsNone(sw.report(env.store)["targets_due"])


class ReportTest(unittest.TestCase):
    def test_report_counts(self):
        env = Env(
            self,
            serve(
                {
                    ("P1", 0): page([cite(2, [("c", [])]), cite(3)]),
                    ("P2", 0): page([cite(1)], nxt=1000),
                    ("P2", 1000): http(400),
                    ("P3", 0): http(500),
                }
            ),
        )
        env.run(
            [target(1, count=2), target(2, count=2000), target(3, count=1)], {1, 2, 3}
        )
        r = sw.report(env.store)
        self.assertEqual(r["papers_by_status"], {"ok": 1, "truncated": 1, "failed": 1})
        self.assertEqual((r["rows"], r["pairs"], r["pairs_with_context"]), (3, 3, 1))
        self.assertEqual(
            (
                r["truncated_papers"],
                r["truncated_cited"],
                r["failed_papers"],
                r["failed_cited"],
            ),
            (1, [2], 1, [3]),
        )
        self.assertEqual(r["release_id"], "graph-api")

    def test_report_is_read_only(self):
        env = Env(self, serve({("P1", 0): page([cite(2)])}))
        env.run([target(1, count=1)], {1, 2})
        before = os.stat(env.store).st_mtime_ns
        sw.report(env.store)
        self.assertEqual(os.stat(env.store).st_mtime_ns, before)

    def test_cli_report(self):
        env = Env(self, serve({("P1", 0): page([cite(2)])}))
        env.run([target(1, count=1)], {1, 2})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(sw.main(["--store", env.store, "--report"]), 0)
        self.assertEqual(json.loads(buf.getvalue())["rows"], 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(sw.main(["--store", env.store + ".none", "--report"]), 1)


class MainTest(unittest.TestCase):
    def write_out(self, d):
        os.makedirs(os.path.join(d, "out"))
        with open(os.path.join(d, "out", "papers.csv"), "w", newline="") as f:
            f.write(",".join(PAPER_COLS) + "\nz:1,W1,P1,1,1\nz:2,W2,P2,2,2\n")
        with open(os.path.join(d, "out", "cites.csv"), "w", newline="") as f:
            f.write("citing_paper_id,cited_paper_id\nz:2,z:1\nz:1,z:2\n")

    def test_main_runs_end_to_end_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_out(d)
            store = os.path.join(d, "s", "a.sqlite")
            op = FakeOpener(
                serve(
                    {
                        ("P1", 0): page([cite(2, [("c", ["result"])])]),
                        ("P2", 0): page([cite(1), cite(1)]),
                    }
                )
            )
            t = FakeTime()
            args = ["--out", os.path.join(d, "out"), "--store", store]
            buf = io.StringIO()
            with (
                contextlib.redirect_stdout(buf),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = sw.main(args, opener=op, clock=t.clock, sleep=t.sleep)
            self.assertEqual(code, 0)
            summary = json.loads(buf.getvalue())
            self.assertEqual(
                (summary["complete"], summary["swept_ok"], summary["rows_kept"]),
                (True, 2, 2),
            )
            self.assertEqual(
                (
                    summary["rows_from_contextsWithIntent"],
                    summary["rows_without_context"],
                ),
                (1, 2),
            )  # records, before merging a citer seen twice
            # the same run again resumes: nothing is asked
            n = len(op.requests)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                sw.main(args, opener=op, clock=t.clock, sleep=t.sleep)
            self.assertEqual(len(op.requests), n)
            # missing input is exit 1
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    sw.main(["--out", os.path.join(d, "none"), "--store", store]), 1
                )

    def test_main_exit_3_when_throttled(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "out"))
            with open(os.path.join(d, "out", "papers.csv"), "w", newline="") as f:
                f.write(
                    ",".join(PAPER_COLS)
                    + "\n"
                    + "".join(f"z:{n},,P{n},{n},5\n" for n in range(1, 13))
                )
            with open(os.path.join(d, "out", "cites.csv"), "w", newline="") as f:
                f.write("citing_paper_id,cited_paper_id\n")
            t = FakeTime()
            err = io.StringIO()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(err),
            ):
                code = sw.main(
                    [
                        "--out",
                        os.path.join(d, "out"),
                        "--store",
                        os.path.join(d, "a.sqlite"),
                    ],
                    opener=FakeOpener(http(429)),
                    clock=t.clock,
                    sleep=t.sleep,
                )
            self.assertEqual(code, 3)
            self.assertIn("throttled; resume later", err.getvalue())


class BuilderIntegrationTest(unittest.TestCase):
    """The point of the module: read_s2_store reads a sweep store unchanged."""

    def test_read_s2_store_reads_the_sweep_store(self):
        # library: corpus ids 1, 2, 3 -> papers zA, zB, zC; 77 is outside
        pages = {
            ("PA", 0): page(
                [
                    cite(
                        2,
                        [
                            ("B on A, first.", ["methodology"]),
                            ("B on A, second.", ["background", "result"]),
                        ],
                        infl=True,
                    ),
                    cite(3, contexts=["C on A flat."], intents=["background"]),
                    cite(77, [("outside", [])]),
                    cite(1, [("self", [])]),
                ]
            ),
            ("PB", 0): page(
                [cite(3, [("C on B.", [])]), cite(3, [("C on B again.", ["result"])])]
            ),
            ("PC", 0): page([cite(1)]),  # no context at all
        }
        env = Env(self, serve(pages))
        targets = [
            target(1, count=4, s2_paper_id="PA"),
            target(2, count=2, s2_paper_id="PB"),
            target(3, count=1, s2_paper_id="PC"),
        ]
        s = env.run(targets, {1, 2, 3})
        self.assertEqual(s["rows_kept"], 4)
        edges, info = bsl.read_s2_store(
            env.store, {1: "zA", 2: "zB", 3: "zC"}, library_cids={1, 2, 3}
        )
        self.assertEqual(
            (
                info["release_id"],
                info["mode"],
                info["rows"],
                info["library_ids_missing"],
            ),
            ("graph-api", "api", 4, 0),
        )
        self.assertEqual(
            set(edges), {("zB", "zA"), ("zC", "zA"), ("zC", "zB"), ("zA", "zC")}
        )
        ba = edges[("zB", "zA")]
        self.assertTrue(ba["influential"])
        self.assertEqual(ba["intents"], ["background", "methodology", "result"])
        self.assertEqual(
            ba["contexts"],
            [
                ("B on A, first.", ["methodology"]),
                ("B on A, second.", ["background", "result"]),
            ],
        )
        ca = edges[("zC", "zA")]
        self.assertFalse(ca["influential"])
        self.assertEqual(
            ca["contexts"], [("C on A flat.", [])]
        )  # flat intents were not copied onto the context
        self.assertEqual(ca["intents"], [])
        cb = edges[
            ("zC", "zB")
        ]  # the same citer twice in one answer is one row with both contexts
        self.assertEqual(
            cb["contexts"], [("C on B.", []), ("C on B again.", ["result"])]
        )
        self.assertEqual(cb["intents"], ["result"])
        ac = edges[("zA", "zC")]
        self.assertEqual((ac["contexts"], ac["intents"]), ([], []))


if __name__ == "__main__":
    unittest.main()
