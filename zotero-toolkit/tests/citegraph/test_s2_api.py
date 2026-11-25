import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.parse
from unittest import mock

# The modules under test sit in citegraph/ and import one another by name.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "citegraph")
)

import build_struct_layer as bsl  # noqa: E402
import s2_api  # noqa: E402

KEY = "S2SECRETKEY-0123456789"


class FakeTime:
    """A clock that sleep() advances: no test waits."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakeOpener:
    """Scripted answers; an Exception instance is raised, anything else returned.
    The last answer repeats. Every request is captured."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, method, url, headers, body):
        self.requests.append(
            {"method": method, "url": url, "headers": dict(headers), "body": body}
        )
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if callable(a):
            a = a(method, url, headers, body)
        if isinstance(a, Exception):
            raise a
        return a


def http(code, headers=None):
    e = urllib.error.HTTPError("https://x/", code, "err", headers or {}, None)
    e.close()  # nothing was opened: keeps the interpreter from warning when it is collected
    return e


def client(opener, key="", t=None, **kw):
    t = t or FakeTime()
    c = s2_api.S2Client(
        key,
        opener=opener,
        clock=t.clock,
        sleep=t.sleep,
        warn=kw.pop("warn", lambda m: None),
        **kw,
    )
    return c, t


def paper(n, doi=None, **kw):
    rec = {
        "paperId": f"{n:040x}",
        "corpusId": n,
        "externalIds": {"CorpusId": n, **({"DOI": doi} if doi else {})},
        "title": f"Paper {n}",
        "year": 2000,
        "referenceCount": 10,
        "citationCount": 5,
        "influentialCitationCount": 1,
        "tldr": {"model": "tldr@v2.0.0", "text": f"About {n}."},
    }
    rec.update(kw)
    return rec


class PacerTest(unittest.TestCase):
    def test_spacing_with_and_without_key(self):
        for key, gap in ((KEY, 1.1), ("", 2.0)):
            op = FakeOpener({"ok": 1})
            c, t = client(op, key)
            for _ in range(3):
                c.get("/paper/x", {})
            self.assertEqual(
                [round(x, 6) for x in t.sleeps], [gap, gap], key
            )  # none before the first request
            self.assertEqual(len(op.requests), 3)

    def test_slow_on_429_backoff_doubles_and_decay(self):
        op = FakeOpener(http(429), http(429), http(503), {"ok": 1})
        c, t = client(op, KEY)
        self.assertEqual(c.get("/x", {}), {"ok": 1})
        backoffs = [s for s in t.sleeps if s >= 4]
        self.assertEqual(backoffs, [4.0, 8.0, 16.0])
        self.assertAlmostEqual(c.pacer.interval, 1.1 * 1.5**3)
        # 20 successes in a row: one decay step toward the base
        op.answers = [{"ok": 1}]
        for _ in range(20):
            c.get("/x", {})
        self.assertAlmostEqual(c.pacer.interval, max(1.1, 1.1 * 1.5**3 / 1.5))
        for _ in range(20 * 4):
            c.get("/x", {})
        self.assertAlmostEqual(c.pacer.interval, 1.1)

    def test_pacer_capped_at_ten_and_backoff_at_120(self):
        op = FakeOpener(http(503))
        c, t = client(op, KEY)
        with self.assertRaises(s2_api.S2Error) as cm:
            c.get("/x", {})
        self.assertEqual(len(op.requests), 7)  # the request and 6 retries
        self.assertEqual([s for s in t.sleeps if s >= 4], [4, 8, 16, 32, 64, 120])
        self.assertLessEqual(c.pacer.interval, 10.0)
        self.assertNotIn(
            "http", str(cm.exception).lower().replace("http 503", "")
        )  # no URL in the message

    def test_retry_after_header_is_honoured_up_to_the_cap(self):
        op = FakeOpener(http(429, {"Retry-After": "30"}), {"ok": 1})
        c, t = client(op, KEY)
        c.get("/x", {})
        self.assertIn(30.0, t.sleeps)

    def test_network_error_is_retried(self):
        op = FakeOpener(urllib.error.URLError("down"), TimeoutError(), {"ok": 1})
        c, t = client(op)
        self.assertEqual(c.get("/x", {}), {"ok": 1})
        self.assertEqual(len(op.requests), 3)

    def test_404_is_not_retried(self):
        op = FakeOpener(http(404))
        c, _ = client(op)
        with self.assertRaises(s2_api.NotFound):
            c.get("/x", {})
        self.assertEqual(len(op.requests), 1)


class KeyTest(unittest.TestCase):
    def test_401_falls_back_without_the_key_and_warns(self):
        for code in (401, 403):
            warned = []
            op = FakeOpener(http(code), {"ok": 1})
            c, _ = client(op, KEY, warn=warned.append)
            self.assertEqual(c.get("/x", {}), {"ok": 1})
            self.assertTrue(c.key_rejected)
            self.assertEqual(len(warned), 1)
            self.assertNotIn(KEY, warned[0])
            self.assertIn("x-api-key", op.requests[0]["headers"])
            self.assertNotIn("x-api-key", op.requests[1]["headers"])
            c.get("/x", {})  # the rest of the run goes without the key, silently
            self.assertNotIn("x-api-key", op.requests[2]["headers"])
            self.assertEqual(len(warned), 1)
            self.assertEqual(c.pacer.base, s2_api.PACE_NOKEY)

    def test_401_without_a_key_is_an_error(self):
        op = FakeOpener(http(401))
        c, _ = client(op)
        with self.assertRaises(s2_api.S2Error):
            c.get("/x", {})
        self.assertEqual(len(op.requests), 1)

    def test_key_only_to_the_api_host(self):
        self.assertEqual(
            s2_api.s2_headers(s2_api.BASE + "/paper/x", KEY)["x-api-key"], KEY
        )
        for url in (
            "https://s3.amazonaws.com/bucket/citations/0.jsonl.gz?X-Amz-Signature=abc",
            "https://api.semanticscholar.org.evil.example/graph/v1/x",
            "https://evil.example/api.semanticscholar.org/x",
            "http://api.semanticscholar.org@evil.example/x",
        ):
            self.assertNotIn("x-api-key", s2_api.s2_headers(url, KEY), url)
        h = s2_api.s2_headers(s2_api.BASE + "/x", KEY)
        self.assertEqual(h["User-Agent"], "zotero-citegraph/2")

    def test_no_key_in_any_url_or_body_for_every_leg(self):
        def answer(method, url, headers, body):
            if "/paper/search/match" in url:
                return {
                    "data": [
                        paper(
                            7, title="A title", year=2001, authors=[{"name": "Ann Lee"}]
                        )
                    ]
                }
            ids = json.loads(body)["ids"]
            if "embedding" in url:
                return [
                    {
                        "paperId": i,
                        "embedding": {"model": "specter_v2", "vector": [0.5] * 768},
                    }
                    for i in ids
                ]
            return [paper(k) for k in range(len(ids))]

        op = FakeOpener(answer)
        c, _ = client(op, KEY)
        s2_api.papers_leg(c, ["DOI:10.1/a"], s2_api.new_papers_cache(), "2026-11-24")
        s2_api.embeddings_leg(c, ["p" * 40], {})
        s2_api.match_leg(
            c,
            [
                {
                    "paper_id": "zot:1",
                    "title": "A title",
                    "year": 2001,
                    "first_family": "Lee",
                }
            ],
            {},
            s2_api.new_papers_cache(),
            "2026-11-24",
        )
        self.assertEqual(len(op.requests), 3)
        for r in op.requests:
            self.assertNotIn(KEY, r["url"])
            self.assertNotIn("key", urllib.parse.urlsplit(r["url"]).query.lower())
            self.assertNotIn(KEY.encode(), r["body"] or b"")
            self.assertEqual(r["headers"]["x-api-key"], KEY)  # only in the header
            self.assertEqual(
                urllib.parse.urlsplit(r["url"]).hostname, "api.semanticscholar.org"
            )

    def test_openalex_key_is_a_header_and_fallback_flag(self):
        seen = []

        def fake(url, headers):
            seen.append((url, dict(headers)))
            if len(seen) == 1:
                raise urllib.error.HTTPError(url, 401, "no", {}, None)
            return {"results": []}

        bsl._STATE["oa_key_rejected"] = False
        with (
            mock.patch.object(bsl, "_urlopen_json", fake),
            mock.patch("sys.stderr", io.StringIO()) as err,
        ):
            bsl.get_json(bsl.API + "?a=1", "OAKEY")
            bsl.get_json(bsl.API + "?a=2", "OAKEY")
        self.assertEqual(seen[0][1]["api_key"], "OAKEY")
        self.assertNotIn("api_key", seen[1][1])
        self.assertNotIn("api_key", seen[2][1])
        self.assertTrue(bsl._STATE["oa_key_rejected"])
        self.assertNotIn("OAKEY", err.getvalue())
        self.assertTrue(all("OAKEY" not in u for u, _ in seen))
        bsl._STATE["oa_key_rejected"] = False


class BudgetTest(unittest.TestCase):
    def test_budget_stops_a_leg_cleanly(self):
        op = FakeOpener(
            lambda m, u, h, b: [paper(i) for i in range(len(json.loads(b)["ids"]))]
        )
        c, _ = client(op, KEY, max_calls=2)
        cache = s2_api.new_papers_cache()
        st = s2_api.papers_leg(
            c, [f"DOI:10.1/{i}" for i in range(1201)], cache, "2026-11-24"
        )
        self.assertFalse(st["complete"])
        self.assertTrue(st["budget_exhausted"])
        self.assertEqual(len(op.requests), 2)
        self.assertEqual(
            len(cache["by_id"]), 1000
        )  # the two batches that were paid for are kept
        # the next leg finds the budget gone and stops at once
        st2 = s2_api.embeddings_leg(c, ["a" * 40], {})
        self.assertFalse(st2["complete"])
        self.assertEqual(len(op.requests), 2)


class PapersLegTest(unittest.TestCase):
    def test_request_shape_and_alignment_with_nulls(self):
        recs = [paper(1, "10.1/a"), None, paper(3, "10.1/c")]
        op = FakeOpener(recs)
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        ids = ["DOI:10.1/a", "DOI:10.1/b", "DOI:10.1/c"]
        st = s2_api.papers_leg(c, ids, cache, "2026-11-24")
        self.assertTrue(st["complete"])
        r = op.requests[0]
        self.assertEqual(r["method"], "POST")
        self.assertEqual(
            json.loads(r["body"]), {"ids": ids}
        )  # fields are NOT in the body
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(r["url"]).query)
        self.assertEqual(q["fields"], [s2_api.PAPER_FIELDS])
        self.assertEqual(r["headers"]["Content-Type"], "application/json")
        self.assertEqual(cache["by_id"]["DOI:10.1/a"]["corpusId"], 1)
        self.assertIsNone(cache["by_id"]["DOI:10.1/b"])
        self.assertIn("DOI:10.1/b", cache["by_id"])  # a null is cached, not absent
        self.assertEqual(cache["by_id"]["DOI:10.1/c"]["tldr"]["text"], "About 3.")
        self.assertEqual(cache["fetched"]["DOI:10.1/b"], "2026-11-24")

    def test_misaligned_answer_is_rejected(self):
        op = FakeOpener([paper(1)])
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        with mock.patch("sys.stderr", io.StringIO()):
            st = s2_api.papers_leg(c, ["DOI:10.1/a", "DOI:10.1/b"], cache, "2026-11-24")
        self.assertFalse(st["complete"])
        self.assertEqual(st["calls_failed"], 1)
        self.assertEqual(cache["by_id"], {})

    def test_batches_of_500_and_cache_rules(self):
        op = FakeOpener(
            lambda m, u, h, b: [paper(i) for i in range(len(json.loads(b)["ids"]))]
        )
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        ids = [f"DOI:10.1/{i}" for i in range(1001)]
        s2_api.papers_leg(c, ids, cache, "2026-11-24")
        self.assertEqual(
            [len(json.loads(r["body"])["ids"]) for r in op.requests], [500, 500, 1]
        )
        n = len(op.requests)
        s2_api.papers_leg(c, ids, cache, "2026-11-24")
        self.assertEqual(len(op.requests), n)  # all cached
        cache["by_id"]["DOI:10.1/0"] = None
        cache["fetched"]["DOI:10.1/0"] = "2026-11-10"  # 14 days: too soon
        s2_api.papers_leg(c, ids, cache, "2026-11-24")
        self.assertEqual(len(op.requests), n)
        cache["fetched"]["DOI:10.1/0"] = "2026-10-01"  # 54 days: ask again
        s2_api.papers_leg(c, ids, cache, "2026-11-24")
        self.assertEqual(len(op.requests), n + 1)
        self.assertEqual(json.loads(op.requests[-1]["body"])["ids"], ["DOI:10.1/0"])

    def test_failed_batch_is_counted_not_cached(self):
        op = FakeOpener(http(400))
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        with mock.patch("sys.stderr", io.StringIO()):
            st = s2_api.papers_leg(c, ["DOI:10.1/a"], cache, "2026-11-24")
        self.assertEqual((st["complete"], st["calls_failed"]), (False, 1))
        self.assertEqual(cache["by_id"], {})


class EmbeddingsLegTest(unittest.TestCase):
    def test_batches_dimension_rounding_and_cache(self):
        def answer(method, url, headers, body):
            out = []
            for i, pid in enumerate(json.loads(body)["ids"]):
                if pid.startswith("n"):
                    out.append({"paperId": pid, "embedding": None})
                elif pid.startswith("s"):
                    out.append(
                        {
                            "paperId": pid,
                            "embedding": {"model": "specter_v2", "vector": [0.1] * 767},
                        }
                    )
                else:
                    out.append(
                        {
                            "paperId": pid,
                            "embedding": {
                                "model": "specter_v2",
                                "vector": [0.12345678] * 768,
                            },
                        }
                    )
            return out

        op = FakeOpener(answer)
        c, _ = client(op)
        cache = {}
        ids = [f"p{i:039d}" for i in range(250)] + ["n" * 40, "s" * 40]
        st = s2_api.embeddings_leg(c, ids, cache)
        self.assertEqual(
            [len(json.loads(r["body"])["ids"]) for r in op.requests], [100, 100, 52]
        )
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(op.requests[0]["url"]).query)
        self.assertEqual(q["fields"], ["paperId,embedding.specter_v2"])
        self.assertEqual((st["no_vector"], st["bad_dimension"]), (1, 1))
        self.assertEqual(len(cache), 250)
        v = cache[ids[0]]
        self.assertEqual(v["model"], "specter_v2")
        self.assertEqual(len(v["vector"]), 768)
        self.assertEqual(v["vector"][0], 0.123457)  # six decimals
        n = len(op.requests)
        s2_api.embeddings_leg(c, ids[:250], cache)
        self.assertEqual(len(op.requests), n)


def cand(title="Numerical Recipes in C", year=1992, family="Press", **kw):
    return {
        "paper_id": "zot:K1",
        "doi": "",
        "title": title,
        "year": year,
        "first_family": family,
        **kw,
    }


def s2rec(
    title="Numerical Recipes in C",
    year=1992,
    authors=("W. Press", "S. Teukolsky"),
    **kw,
):
    r = paper(
        5,
        title=title,
        year=year,
        authors=[{"authorId": str(i), "name": a} for i, a in enumerate(authors)],
    )
    r.update(kw)
    return r


class TitleMatchTest(unittest.TestCase):
    def test_rules(self):
        d = s2_api.decide_match
        self.assertEqual(d(cand(), s2rec()), "accepted")
        self.assertEqual(d(cand(), s2rec(year=1993)), "accepted")  # off by one is fine
        self.assertEqual(d(cand(), s2rec(year=1994)), "reject:year")  # off by two
        self.assertEqual(
            d(cand(), s2rec(authors=("A. Other", "B. Else"))), "reject:author"
        )
        self.assertEqual(
            d(cand(), s2rec(title="Something else entirely")), "reject:title"
        )
        self.assertEqual(d(cand(), None), "reject:none")
        # folded equality: case, punctuation, accents
        self.assertEqual(d(cand(title="Numérical recipes, in C!"), s2rec()), "accepted")
        # no Zotero year / no creators: those checks are skipped
        self.assertEqual(d(cand(year=None), s2rec(year=1850)), "accepted")
        self.assertEqual(d(cand(family=""), s2rec(authors=("A. Other",))), "accepted")
        # S2 has no year but Zotero does: cannot confirm
        self.assertEqual(d(cand(), s2rec(year=None)), "reject:year")
        # the author may be any of the S2 authors, compared on the last name, folded
        self.assertEqual(d(cand(family="Teukolský"), s2rec()), "accepted")
        self.assertEqual(d(cand(family="Pres"), s2rec()), "reject:author")

    def test_edition_suffix(self):
        d = s2_api.decide_match
        nr = cand(title="Numerical Recipes in C", year=1992)
        self.assertEqual(
            d(nr, s2rec(title="Numerical Recipes in C, 2nd Edition", year=1992)),
            "accepted",
        )
        self.assertEqual(
            d(
                nr,
                s2rec(
                    title="Numerical Recipes in C: Second edition, revised", year=1992
                ),
            ),
            "accepted",
        )
        self.assertEqual(
            d(nr, s2rec(title="Numerical Recipes in C, 3rd ed.", year=1992)), "accepted"
        )
        # a volume, a part, a bare number, a roman numeral or a year is a different work, not an edition
        base = "Treatise on Geophysics"
        for suffix in ("Volume 3", "Vol. II", "Part 2", "Pt 2", "2010", "III", "7"):
            self.assertEqual(
                d(
                    cand(title=base, year=2007, family="Schubert"),
                    s2rec(
                        title=f"{base}, {suffix}", year=2007, authors=("G. Schubert",)
                    ),
                ),
                "reject:title",
                suffix,
            )
        # any other word is not an edition suffix
        self.assertEqual(
            d(
                nr,
                s2rec(title="Numerical Recipes in C: The Art of Scientific Computing"),
            ),
            "reject:title",
        )
        # too short to trust a prefix: 1 word never, 2 words only with a year and a first creator that both agree
        self.assertEqual(
            d(
                cand(title="Geodesy", family="Torge"),
                s2rec(title="Geodesy 2nd edition", authors=("W. Torge",)),
            ),
            "reject:title",
        )
        pg = cand(title="Physical Geodesy", year=1967, family="Heiskanen")
        full = s2rec(
            title="Physical Geodesy, 2nd edition",
            year=1967,
            authors=("W. Heiskanen", "H. Moritz"),
        )
        self.assertEqual(d(pg, full), "accepted")
        self.assertEqual(
            d(cand(title="Physical Geodesy", year=1967, family=""), full),
            "reject:title",
        )  # no creator
        self.assertEqual(
            d(cand(title="Physical Geodesy", year=None, family="Heiskanen"), full),
            "reject:title",
        )  # no year
        self.assertEqual(
            d(
                pg,
                s2rec(
                    title="Physical Geodesy, 2nd edition",
                    year=1980,
                    authors=("W. Heiskanen",),
                ),
            ),
            "reject:title",
        )
        self.assertEqual(
            d(
                pg,
                s2rec(
                    title="Physical Geodesy, 2nd edition",
                    year=1967,
                    authors=("A. Other",),
                ),
            ),
            "reject:title",
        )
        # the 3-word rule settles the title only: year and author still decide
        self.assertEqual(
            d(nr, s2rec(title="Numerical Recipes in C, 3rd Edition", year=2007)),
            "reject:year",
        )
        self.assertTrue(s2_api.edition_suffix_only("A b c", "a B c 2nd ed"))
        self.assertFalse(s2_api.edition_suffix_only("a b c", "a b c"))
        self.assertFalse(s2_api.edition_suffix_only("a b c", "a b c volume 2"))

    def test_title_equality_keeps_digits(self):
        d = s2_api.decide_match
        self.assertEqual(
            d(
                cand(title="Ocean tides, Part 1", year=2001, family="Ray"),
                s2rec(title="Ocean tides, Part 2", year=2001, authors=("R. Ray",)),
            ),
            "reject:title",
        )
        self.assertEqual(
            d(
                cand(title="GRACE Level-2 Data Handbook RL05", year=None, family=""),
                s2rec(title="GRACE Level-2 Data Handbook RL06"),
            ),
            "reject:title",
        )
        self.assertEqual(
            d(
                cand(title="GRACE Handbook RL05", year=None, family=""),
                s2rec(title="GRACE handbook, RL05!"),
            ),
            "accepted",
        )
        self.assertEqual(s2_api.tfold("Kühn RL05"), "kuhnrl05")

    def test_token_jaccard_threshold(self):
        base = "one two three four five six seven eight nine ten"
        self.assertEqual(
            s2_api.decide_match(
                cand(title=base, family=""), s2rec(title=base + " eleven")
            ),
            "accepted",
        )  # 10/11
        self.assertEqual(
            s2_api.decide_match(
                cand(title=base, family=""), s2rec(title=base + " eleven twelve")
            ),
            "reject:title",  # 10/12
        )

    def test_leg_query_cache_and_404(self):
        def answer(method, url, headers, body):
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            if q["query"] == ["Unknown thing"]:
                return http(404)
            return {"data": [s2rec()]}

        op = FakeOpener(answer)
        c, _ = client(op)
        matches, pc = {}, s2_api.new_papers_cache()
        cands = [
            cand(),
            {
                "paper_id": "zot:K2",
                "doi": "",
                "title": "Unknown thing",
                "year": None,
                "first_family": "",
            },
            {
                "paper_id": "zot:K3",
                "doi": "",
                "title": "  ",
                "year": 2000,
                "first_family": "",
            },
        ]
        st = s2_api.match_leg(c, cands, matches, pc, "2026-11-24")
        self.assertTrue(st["complete"])
        self.assertEqual(len(op.requests), 2)  # the blank title is not asked
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(op.requests[0]["url"]).query)
        self.assertEqual(q["year"], ["1991-1993"])
        self.assertEqual(q["fields"], [s2_api.MATCH_FIELDS])
        self.assertEqual(q["query"], ["Numerical Recipes in C"])
        self.assertNotIn(
            "year",
            urllib.parse.parse_qs(urllib.parse.urlsplit(op.requests[1]["url"]).query),
        )
        by_paper = {e["paper_id"]: e for e in matches.values()}
        self.assertEqual(by_paper["zot:K1"]["decision"], "accepted")
        self.assertEqual(by_paper["zot:K2"]["decision"], "reject:none")
        self.assertEqual(by_paper["zot:K2"]["record"], None)
        self.assertIn(
            "CorpusId:5", pc["by_id"]
        )  # an accepted record is kept in the paper cache too
        # keyed by paper plus a hash of the query: a rerun asks nothing, a changed title is asked again
        s2_api.match_leg(c, cands, matches, pc, "2026-11-24")
        self.assertEqual(len(op.requests), 2)
        cands[0] = cand(title="Numerical Recipes in C, second edition")
        s2_api.match_leg(c, cands, matches, pc, "2026-11-24")
        self.assertEqual(len(op.requests), 3)
        self.assertEqual(len(matches), 3)
        self.assertEqual(len({k.split("#")[0] for k in matches}), 2)


class RobustnessTest(unittest.TestCase):
    def test_dropped_connection_and_bad_body_are_retried_then_become_s2error(self):
        import http.client

        op = FakeOpener(
            http.client.IncompleteRead(b"par"),
            http.client.RemoteDisconnected("closed"),
            json.JSONDecodeError("bad", "<html>", 0),
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
            {"ok": 1},
        )
        c, t = client(op)
        self.assertEqual(c.get("/x", {}), {"ok": 1})
        self.assertEqual(len(op.requests), 5)
        self.assertEqual([s for s in t.sleeps if s >= 4], [4.0, 8.0, 16.0, 32.0])
        op = FakeOpener(http.client.IncompleteRead(b""))
        c, _ = client(op)
        with self.assertRaises(s2_api.S2Error):
            c.get("/x", {})
        # a leg counts it as a failed batch instead of dying
        op = FakeOpener(json.JSONDecodeError("bad", "x", 0))
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        with mock.patch("sys.stderr", io.StringIO()):
            st = s2_api.papers_leg(c, ["DOI:10.1/a"], cache, "2026-11-24")
        self.assertEqual((st["complete"], st["calls_failed"]), (False, 1))

    def test_lone_surrogates_are_replaced_when_the_answer_is_parsed(self):
        rec = paper(
            1,
            "10.1/a",
            title="Bad \ud835 title",
            tldr={"model": "m", "text": "tl\ud800dr"},
        )
        op = FakeOpener([rec])
        c, _ = client(op)
        cache = s2_api.new_papers_cache()
        s2_api.papers_leg(c, ["DOI:10.1/a"], cache, "2026-11-24")
        got = cache["by_id"]["DOI:10.1/a"]
        self.assertEqual(got["title"], "Bad \ufffd title")
        self.assertEqual(got["tldr"]["text"], "tl\ufffddr")
        json.dumps(cache, ensure_ascii=False).encode(
            "utf-8"
        )  # a cache writer cannot raise on it
        self.assertEqual(
            s2_api.clean_obj({"a\ud835": ["x\udfff", 3, None]}),
            {"a\ufffd": ["x\ufffd", 3, None]},
        )
        self.assertEqual(
            s2_api.clean_text("Kühn 😀"), "Kühn 😀"
        )  # a proper pair is left alone

    def test_key_header_is_unredirected(self):
        import urllib.request as ur

        for header in ("x-api-key", "api_key"):
            req = s2_api.build_request(
                "GET", s2_api.BASE + "/x", {header: KEY, "Accept": "application/json"}
            )
            self.assertEqual(
                req.get_header(header.capitalize()), KEY
            )  # sent on the first request
            self.assertNotIn(header.capitalize(), req.headers)
            new = ur.HTTPRedirectHandler().redirect_request(
                req,
                io.BytesIO(b""),
                302,
                "Found",
                {},
                "https://elsewhere.example/landing",
            )
            self.assertIsNotNone(new)
            self.assertEqual(new.full_url, "https://elsewhere.example/landing")
            sent = {k.lower() for k in list(new.headers) + list(new.unredirected_hdrs)}
            self.assertNotIn(header, sent)
            self.assertIn("accept", sent)  # ordinary headers do follow
        post = s2_api.build_request(
            "POST",
            s2_api.BASE + "/x",
            s2_api.s2_headers(s2_api.BASE + "/x", KEY, True),
            b"{}",
        )
        self.assertEqual(post.get_method(), "POST")
        self.assertEqual(post.get_header("X-api-key"), KEY)
        self.assertEqual(post.get_header("Content-type"), "application/json")


class FoldAgreesWithBuilderTest(unittest.TestCase):
    def test_fold(self):
        for s in ("Kühn", "O'Brien-Smith", "Numérical, recipes 2nd", ""):
            self.assertEqual(s2_api.fold(s), bsl.fold(s))


if __name__ == "__main__":
    unittest.main()
