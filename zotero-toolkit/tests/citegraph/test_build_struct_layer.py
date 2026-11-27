import filecmp
import io
import json
import math
import os
import sys
import tempfile
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


def _s2_knows_nothing(method, url, headers, body):
    """Stand-in for the S2 network function in every test of this module."""
    if method == "POST":
        return [None] * len(json.loads(body)["ids"])
    e = urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    e.close()
    raise e


def setUpModule():
    # v1 tests run main() online with a fake OpenAlex; the S2 legs must not reach the network either
    p = mock.patch.object(s2_api, "_urlopen_json", _s2_knows_nothing)
    p.start()
    unittest.addModuleCleanup(p.stop)


def work(
    doi,
    wid,
    refs=(),
    authors=(),
    source=None,
    primary=None,
    topics=(),
    keywords=(),
    **kw,
):
    w = {
        "id": f"https://openalex.org/{wid}",
        "doi": f"https://doi.org/{doi}",
        "publication_year": 2000,
        "type": "article",
        "cited_by_count": 5,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "authorships": [
            {
                "author_position": "first",
                "author": {"id": f"https://openalex.org/{aid}", "display_name": nm},
            }
            for aid, nm in authors
        ],
        "primary_location": {
            "source": {
                "id": f"https://openalex.org/{source[0]}",
                "display_name": source[1],
            }
        }
        if source
        else None,
        "primary_topic": primary,
        "topics": list(topics),
        "keywords": list(keywords),
    }
    w.update(kw)
    return w


def topic(tid, name, sub, field, domain, score=0.9):
    def lv(kind, pair):
        return {"id": f"https://openalex.org/{kind}/{pair[0]}", "display_name": pair[1]}

    return {
        "id": f"https://openalex.org/{tid}",
        "display_name": name,
        "score": score,
        "subfield": lv("subfields", sub),
        "field": lv("fields", field),
        "domain": lv("domains", domain),
    }


def entry(**kw):
    base = {"title": "T", "source": "parent"}
    base.update(kw)
    return base


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def quiet_main(argv):
    with mock.patch("sys.stdout", io.StringIO()):
        return bsl.main(argv)


def col(tables, name, column):
    i = [c for c, _ in bsl.TABLES[name]["cols"]].index(column)
    return [r[i] for r in tables[name]]


class DoiTest(unittest.TestCase):
    def test_norm_doi(self):
        self.assertEqual(
            bsl.norm_doi("https://doi.org/10.1029/98JB02844"), "10.1029/98jb02844"
        )
        self.assertEqual(bsl.norm_doi("http://dx.doi.org/10.1/X"), "10.1/x")
        self.assertEqual(bsl.norm_doi("doi: 10.1/Y"), "10.1/y")
        self.assertEqual(bsl.norm_doi(" 10.1/Z "), "10.1/z")
        self.assertEqual(bsl.norm_doi("not a doi"), "")
        self.assertEqual(bsl.norm_doi(None), "")


class IdentityTest(unittest.TestCase):
    def test_papers_and_attachments(self):
        md = {
            "AAAAAAA1": entry(doi="10.1/X", item_key="K1"),
            "AAAAAAA2": entry(
                doi="https://doi.org/10.1/x",
                item_key="K1",
                year="1999",
                publication="J",
            ),
            "AAAAAAA3": entry(item_key="K3", title="No doi"),
            "AAAAAAA4": entry(title="No item key"),
        }
        tables, _ = bsl.assemble(md, {})
        self.assertEqual(
            col(tables, "papers", "paper_id"), ["doi:10.1/x", "zot:AAAAAAA4", "zot:K3"]
        )
        att = {r[0]: r for r in tables["paper_attachments"]}
        self.assertEqual(att["AAAAAAA1"][2], "doi:10.1/x")
        self.assertEqual(att["AAAAAAA2"][2], "doi:10.1/x")
        self.assertEqual(att["AAAAAAA3"], ["AAAAAAA3", "K3", "zot:K3"])
        self.assertEqual(att["AAAAAAA4"], ["AAAAAAA4", "", "zot:AAAAAAA4"])
        p = {r[0]: r for r in tables["papers"]}
        self.assertEqual(p["doi:10.1/x"][9], 2)  # n_attachments
        # the entry with more non-empty fields (AAAAAAA2) supplies the metadata
        self.assertEqual(p["doi:10.1/x"][4], 1999)
        self.assertEqual(p["doi:10.1/x"][5], "z:j")
        self.assertEqual(p["doi:10.1/x"][1], "10.1/x")
        self.assertIsNone(p["zot:K3"][1])

    def test_best_entry_tie_takes_smallest_key(self):
        md = {
            "BBBBBBBB": entry(doi="10.1/x", title="from B"),
            "AAAAAAAA": entry(doi="10.1/x", title="from A"),
        }
        tables, _ = bsl.assemble(md, {})
        self.assertEqual(tables["papers"][0][3], "from A")


class CitesTest(unittest.TestCase):
    def test_self_edge_duplicate_and_external_removed(self):
        md = {
            "AAAAAAA1": entry(doi="10.1/a"),
            "AAAAAAA2": entry(doi="10.1/b"),
            "AAAAAAA3": entry(doi="10.1/c"),
        }
        works = {
            "10.1/a": work("10.1/a", "W1", refs=["W2", "W1", "W2", "W9"]),
            "10.1/b": work("10.1/b", "W2", refs=[]),
        }
        tables, _ = bsl.assemble(md, works)
        self.assertEqual(tables["cites"], [["doi:10.1/a", "doi:10.1/b"]])
        p = {r[0]: r for r in tables["papers"]}
        self.assertEqual(p["doi:10.1/a"][8], 4)  # n_refs_openalex counts the raw list
        self.assertIsNone(p["doi:10.1/c"][2])


class CouplingTest(unittest.TestCase):
    def test_hand_computed_score(self):
        refs = {"p1": ["r1", "r2", "r3", "r4"], "p2": ["r1", "r2", "r3"], "p3": ["r5"]}
        rows = bsl.coupling(refs)
        w, wu = math.log(3 / 2), math.log(3)
        expect = 3 * w / math.sqrt((3 * w + wu) * (3 * w))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:3], ["p1", "p2", 3])
        self.assertAlmostEqual(rows[0][3], expect, places=6)

    def test_min_shared_and_df_bounds(self):
        two = {"p1": ["a", "b", "x"], "p2": ["a", "b", "y"], "p3": ["z"]}
        self.assertEqual(bsl.coupling(two), [])  # only 2 shared references
        three = {"p1": ["a", "b", "c"], "p2": ["a", "b", "c"], "p3": ["z"]}
        self.assertEqual(len(bsl.coupling(three)), 1)
        with mock.patch.object(bsl, "MAX_DF", 1):  # df 2 is now above the band
            self.assertEqual(bsl.coupling(three), [])
        with mock.patch.object(bsl, "MIN_DF", 3):  # df 2 is below the band
            self.assertEqual(bsl.coupling(three), [])

    def test_top_k_union_and_order(self):
        refs = {
            "A": ["x1", "x2", "x3", "x4", "z1", "z2", "z3"],
            "B": ["x1", "x2", "x3", "x4"],
            "C": ["y1", "y2", "y3", "y4", "z1", "z2", "z3"],
            "D": ["y1", "y2", "y3", "y4"],
        }
        full = bsl.coupling(refs)
        self.assertEqual([r[:2] for r in full], [["A", "B"], ["A", "C"], ["C", "D"]])
        with mock.patch.object(bsl, "TOP_K", 1):
            top1 = bsl.coupling(refs)
        self.assertEqual([r[:2] for r in top1], [["A", "B"], ["C", "D"]])
        self.assertTrue(all(r[0] < r[1] for r in full))

    def test_papers_without_reference_list_excluded_from_n(self):
        refs = {"p1": ["a", "b", "c"], "p2": ["a", "b", "c"], "p3": []}
        rows = bsl.coupling(refs)
        # N = 2, df = 2, idf = 0 for every shared reference: no usable score
        self.assertEqual(rows, [])


class AuthorTest(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(bsl.fold("Kühn"), "kuhn")
        self.assertEqual(bsl.fold("O'Brien-Smith"), "obriensmith")
        self.assertEqual(bsl.initials("Ogutu B."), "ob")
        self.assertEqual(bsl.initials("Jean-Pierre"), "jp")
        self.assertEqual(bsl.initials(""), "")

    def test_merge_rule(self):
        oa = work(
            "10.1/oa",
            "W1",
            authors=[
                ("A1", "Christian Hirt"),
                ("A2", "Michael Kuhn"),
                ("A3", "Yu Wang"),
                ("A4", "Yan Wang"),
            ],
        )
        md = {
            "AAAAAAA1": entry(doi="10.1/oa"),
            "AAAAAAA2": entry(
                item_key="K2",
                creators=[
                    ["Hirt", "Christian"],
                    ["Kühn", "Michael"],
                    ["Wang", "Y."],
                    ["Smith", "J."],
                ],
            ),
        }
        tables, aux = bsl.assemble(md, {"10.1/oa": oa})
        m = aux["author_merge"]
        self.assertEqual(
            (m["merged"], m["kept"], m["kept_ambiguous"], m["kept_no_match"]),
            (2, 2, 1, 1),
        )
        rows = [r for r in tables["paper_authors"] if r[0] == "zot:K2"]
        self.assertEqual(
            [(r[1], r[2], r[3]) for r in rows],
            [
                ("oa:A1", 1, "zotero"),
                ("oa:A2", 2, "zotero"),
                ("z:wang|y", 3, "zotero"),
                ("z:smith|j", 4, "zotero"),
            ],
        )
        oa_rows = [r for r in tables["paper_authors"] if r[0] == "doi:10.1/oa"]
        self.assertEqual(
            [(r[1], r[3]) for r in oa_rows],
            [(f"oa:A{i}", "openalex") for i in (1, 2, 3, 4)],
        )
        authors = {r[0]: r for r in tables["authors"]}
        self.assertEqual(authors["oa:A1"][3], 2)  # n_papers counts the merged paper
        self.assertEqual(authors["z:smith|j"][1], "J. Smith")
        self.assertIsNone(authors["z:smith|j"][2])

    def test_two_zotero_authors_merging_into_one_keep_first(self):
        oa = work("10.1/oa", "W1", authors=[("A1", "Christian Hirt")])
        md = {
            "AAAAAAA1": entry(doi="10.1/oa"),
            "AAAAAAA2": entry(
                item_key="K2", creators=[["Hirt", "C."], ["Hirt", "Christian Paul"]]
            ),
        }
        tables, _ = bsl.assemble(md, {"10.1/oa": oa})
        rows = [r for r in tables["paper_authors"] if r[0] == "zot:K2"]
        self.assertEqual([(r[1], r[2]) for r in rows], [("oa:A1", 1)])
        self.assertEqual(bsl.hard_checks(tables), [])

    def test_single_field_creator_name(self):
        md = {"AAAAAAA1": entry(item_key="K1", creators=[["", "Yun XIAO"]])}
        tables, _ = bsl.assemble(md, {})
        self.assertEqual(col(tables, "paper_authors", "author_id"), ["z:xiao|y"])


class VenueTest(unittest.TestCase):
    def test_fallback(self):
        md = {
            "AAAAAAA1": entry(doi="10.1/a", publication="Zotero Name"),
            "AAAAAAA2": entry(doi="10.1/b", publication="Journal of Geodesy"),
            "AAAAAAA3": entry(doi="10.1/c"),
            "AAAAAAA4": entry(item_key="K4", publication="  Journal  of Geodesy "),
        }
        works = {
            "10.1/a": work("10.1/a", "W1", source=("S7", "OpenAlex Name")),
            "10.1/c": work("10.1/c", "W3", source=None),
        }
        tables, _ = bsl.assemble(md, works)
        v = {r[0]: r for r in tables["papers"]}
        self.assertEqual(v["doi:10.1/a"][5], "oa:S7")
        self.assertEqual(v["doi:10.1/b"][5], "z:journalofgeodesy")
        self.assertIsNone(v["doi:10.1/c"][5])
        self.assertEqual(v["zot:K4"][5], "z:journalofgeodesy")
        venues = {r[0]: r for r in tables["venues"]}
        self.assertEqual(venues["oa:S7"], ["oa:S7", "OpenAlex Name", "S7", 1])
        self.assertEqual(venues["z:journalofgeodesy"][3], 2)
        self.assertEqual(len(venues), 2)


class TopicTest(unittest.TestCase):
    def test_hierarchy_and_paper_topics(self):
        t1 = topic(
            "T1",
            "Geoid",
            ("1908", "Geophysics"),
            ("19", "Earth Sci"),
            ("3", "Physical"),
            1.0,
        )
        t2 = topic(
            "T2",
            "Gravity",
            ("1908", "Geophysics"),
            ("19", "Earth Sci"),
            ("3", "Physical"),
            0.5,
        )
        w = work(
            "10.1/a",
            "W1",
            primary=t1,
            topics=[t1, t2],
            keywords=[
                {
                    "id": "https://openalex.org/keywords/geoid",
                    "display_name": "geoid",
                    "score": 0.6,
                }
            ],
        )
        tables, _ = bsl.assemble({"AAAAAAA1": entry(doi="10.1/a")}, {"10.1/a": w})
        self.assertEqual(
            tables["topics"],
            [
                ["domain:3", "Physical", "domain", None],
                ["field:19", "Earth Sci", "field", "domain:3"],
                ["subfield:1908", "Geophysics", "subfield", "field:19"],
                ["topic:T1", "Geoid", "topic", "subfield:1908"],
                ["topic:T2", "Gravity", "topic", "subfield:1908"],
            ],
        )
        self.assertEqual(
            tables["paper_topics"],
            [
                ["doi:10.1/a", "topic:T1", 1.0, True],
                ["doi:10.1/a", "topic:T2", 0.5, False],
            ],
        )
        self.assertEqual(tables["paper_keywords"], [["doi:10.1/a", "kw:geoid", 0.6]])
        self.assertEqual(tables["keywords"], [["kw:geoid", "geoid", 1]])
        self.assertEqual(bsl.hard_checks(tables), [])

    def test_primary_only(self):
        t1 = topic("T1", "Geoid", ("1908", "G"), ("19", "E"), ("3", "P"))
        w = work("10.1/a", "W1", primary=t1, topics=[])
        tables, _ = bsl.assemble({"AAAAAAA1": entry(doi="10.1/a")}, {"10.1/a": w})
        self.assertEqual(tables["paper_topics"][0][1:], ["topic:T1", 0.9, True])


class HttpTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(bsl.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def test_request_construction(self):
        calls = []

        def fake(url, headers):
            calls.append((url, dict(headers)))
            if url.startswith(bsl.API + "/doi:"):
                return work("10.1/a|b", "W9")
            return {"results": [work("10.1/a", "W1")]}

        store = {"works": {}, "unknown": []}
        with mock.patch.object(bsl, "_urlopen_json", fake):
            stats = bsl.harvest(
                ["10.1/a", "10.1/b", "10.1/a|b", "10.1/c,d"],
                store,
                "me@example.org",
                "SECRETKEY123",
            )
        self.assertEqual(
            (stats["batches"], stats["singles"], stats["failed"]), (1, 2, 0)
        )
        self.assertEqual(len(calls), 3)
        for url, headers in calls:
            self.assertEqual(headers["api_key"], "SECRETKEY123")
            self.assertNotIn("SECRETKEY123", url)
            self.assertNotIn("api_key", url)
        batch = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0][0]).query)
        self.assertEqual(batch["filter"], ["doi:10.1/a|10.1/b"])
        self.assertEqual(batch["per-page"], ["200"])
        self.assertEqual(batch["select"], [bsl.SELECT])
        # the odd DOIs went one at a time through /works/doi:<quoted>
        self.assertTrue(calls[1][0].startswith(bsl.API + "/doi:10.1%2Fa%7Cb?"))
        self.assertTrue(calls[2][0].startswith(bsl.API + "/doi:10.1%2Fc%2Cd?"))
        # a, found in the batch; b, asked and not returned
        self.assertIn("10.1/a", store["works"])
        self.assertIn("10.1/b", store["unknown"])

    def test_retry_then_success_and_failure_counted(self):
        attempts = []

        def flaky(url, headers):
            attempts.append(url)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(url, 429, "Too Many", {}, None)
            return {"results": []}

        with mock.patch.object(bsl, "_urlopen_json", flaky):
            self.assertEqual(bsl.get_json(bsl.API, "k"), {"results": []})
        self.assertEqual(len(attempts), 3)

        def dead(url, headers):
            raise urllib.error.HTTPError(url, 503, "Down", {}, None)

        store = {"works": {}, "unknown": []}
        with (
            mock.patch.object(bsl, "_urlopen_json", dead),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            stats = bsl.harvest(["10.1/a"], store, "", "")
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(store["unknown"], [])  # a failed request proves nothing

    def test_cached_dois_not_asked_again_unknown_are(self):
        calls = []

        def fake(url, headers):
            calls.append(url)
            return {"results": []}

        store = {"works": {"10.1/a": work("10.1/a", "W1")}, "unknown": ["10.1/u"]}
        with mock.patch.object(bsl, "_urlopen_json", fake):
            bsl.harvest(["10.1/a", "10.1/u"], store, "", "")
            self.assertEqual(len(calls), 1)
            self.assertIn(
                "10.1%2Fu".replace("%2F", "/"), urllib.parse.unquote(calls[0])
            )
            self.assertNotIn(
                "10.1/a", urllib.parse.unquote(calls[0]).split("filter=")[1]
            )
            bsl.harvest(["10.1/a"], store, "", "", refresh=True)
            self.assertEqual(len(calls), 2)


class CacheRunTest(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(bsl.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)

    def test_offline_rerun_is_byte_identical(self):
        t1 = topic("T1", "Geoid", ("1908", "G"), ("19", "E"), ("3", "P"))
        works = {
            "10.1/a": work(
                "10.1/a",
                "W1",
                refs=["W2", "R1", "R2", "R3"],
                authors=[("A1", "Ann Lee")],
                source=("S1", "Jour"),
                primary=t1,
                topics=[t1],
            ),
            "10.1/b": work(
                "10.1/b", "W2", refs=["R1", "R2", "R3"], authors=[("A1", "Ann Lee")]
            ),
        }

        def fake(url, headers):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if (
                "filter" not in q
            ):  # one outside work by id (R1-R3, unknown to the list filter): deleted
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            wanted = q["filter"][0][4:].split("|")
            return {"results": [works[d] for d in wanted if d in works]}

        with tempfile.TemporaryDirectory() as d:
            md = os.path.join(d, "md.json")
            ref = os.path.join(d, "ref.json")
            with open(md, "w") as f:
                json.dump(
                    {
                        "AAAAAAA1": entry(doi="10.1/a", year="2001"),
                        "AAAAAAA2": entry(doi="10.1/b", year="2002"),
                        "AAAAAAA3": entry(doi="10.1/missing"),
                    },
                    f,
                )
            with open(ref, "w") as f:
                json.dump(
                    {
                        "10.1/a": {"n_refs": 4, "cited": ["10.1/b"]},
                        "10.1/b": {"n_refs": 3, "cited": []},
                    },
                    f,
                )
            out1, out2 = os.path.join(d, "out1"), os.path.join(d, "out2")
            cache = os.path.join(d, "cache", "w.json")
            with mock.patch.object(bsl, "_urlopen_json", fake):
                rc = quiet_main(
                    [
                        "--metadata",
                        md,
                        "--out",
                        out1,
                        "--cache",
                        cache,
                        "--reference-check",
                        ref,
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(
                read(os.path.join(out1, "unmatched_dois.txt")), "10.1/missing\n"
            )

            def boom(url, headers):
                raise AssertionError("network used in offline mode")

            with mock.patch.object(bsl, "_urlopen_json", boom):
                rc = quiet_main(
                    ["--metadata", md, "--out", out2, "--cache", cache, "--offline"]
                )
            self.assertEqual(rc, 0)
            names = [f"{n}.csv" for n in bsl.TABLES] + [
                "unmatched_dois.txt",
                "schema.sql",
                "load.sql",
            ]
            for n in names:
                self.assertTrue(
                    filecmp.cmp(
                        os.path.join(out1, n), os.path.join(out2, n), shallow=False
                    ),
                    n,
                )
            # a rerun with the cache asks only for the unknown DOI
            seen = []

            def spy(url, headers):
                seen.append(url)
                return {"results": []}

            with mock.patch.object(bsl, "_urlopen_json", spy):
                quiet_main(["--metadata", md, "--out", out2, "--cache", cache])
            self.assertEqual(len(seen), 1)
            self.assertIn("10.1/missing", urllib.parse.unquote(seen[0]))
            qa = json.loads(read(os.path.join(out1, "qa.json")))
            self.assertEqual(qa["oracle"]["jaccard"], 1.0)
            self.assertIn("Jaccard", read(os.path.join(out1, "QA.md")))

    def test_oracle_below_threshold_exits_1(self):
        works = {
            "10.1/a": work("10.1/a", "W1", refs=["W2"]),
            "10.1/b": work("10.1/b", "W2"),
        }
        with tempfile.TemporaryDirectory() as d:
            md, ref = os.path.join(d, "md.json"), os.path.join(d, "ref.json")
            cache = os.path.join(d, "c.json")
            dump({"works": works, "unknown": []}, cache)
            dump({"AAAAAAA1": entry(doi="10.1/a"), "AAAAAAA2": entry(doi="10.1/b")}, md)
            dump({"10.1/a": {"n_refs": 1, "cited": []}}, ref)
            with mock.patch("sys.stderr", io.StringIO()) as err:
                rc = quiet_main(
                    [
                        "--metadata",
                        md,
                        "--out",
                        os.path.join(d, "o"),
                        "--cache",
                        cache,
                        "--offline",
                        "--reference-check",
                        ref,
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("ORACLE FAILED", err.getvalue())


class HardCheckTest(unittest.TestCase):
    def good(self):
        return {
            "venues": [],
            "topics": [
                ["domain:1", "D", "domain", None],
                ["field:2", "F", "field", "domain:1"],
            ],
            "papers": [
                ["doi:a", "a", None, "t", 2000, None, None, None, None, 1],
                ["doi:b", "b", None, "t", 2000, None, None, None, None, 1],
            ],
            "paper_attachments": [["K1", "I1", "doi:a"]],
            "authors": [],
            "paper_authors": [],
            "paper_topics": [],
            "keywords": [],
            "paper_keywords": [],
            "cites": [["doi:a", "doi:b"]],
            "coupled": [["doi:a", "doi:b", 3, 0.5]],
        }

    def test_good_fixture_passes(self):
        self.assertEqual(bsl.hard_checks(self.good()), [])

    def test_broken_fixture_trips_every_check(self):
        t = self.good()
        t["cites"] = [
            ["doi:a", "doi:a"],
            ["doi:a", "doi:zzz"],
        ]  # self edge, dangling FK
        t["coupled"] = [["doi:b", "doi:a", 3, 0.5]]  # a > b
        t["papers"].append(
            ["doi:a", "a", None, "t", 2000, "oa:S9", None, None, None, 1]
        )  # dup pk, bad venue
        t["topics"] = [
            ["domain:1", "D", "domain", "field:2"],
            ["field:2", "F", "field", "domain:1"],
            ["topic:3", "T", "topic", "subfield:9"],
        ]  # cycle, missing parent
        text = "\n".join(bsl.hard_checks(t))
        for needle in (
            "self-citation",
            "paper_a >= paper_b",
            "papers: 1 duplicate primary key",
            "papers.venue_id",
            "cites.cited_paper_id",
            "parent cycle",
            "topics.parent_id",
        ):
            self.assertIn(needle, text)


# --------------------------------------------------------------------------
# v2: citation graph, S2, outside works, generated SQL
# --------------------------------------------------------------------------

import csv
import re
import sys

import s2_citations_sync as sync

OAKEY, S2KEY = "OAKEY-abcdef123456", "S2KEY-abcdef123456"
SERVER_FINGERPRINT_SQL = """-- zotero_struct:fingerprint
SELECT c.relname, c.oid::bigint AS oid, pg_relation_filenode(c.oid)::bigint AS filenode
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind = 'r'"""
SERVER_TABLES = (
    "papers",
    "venues",
    "paper_attachments",
    "authors",
    "paper_authors",
    "topics",
    "paper_topics",
    "cites",
    "coupled",
)
SERVER_COLUMNS = {
    "papers": [
        "paper_id",
        "doi",
        "title",
        "year",
        "cited_by_count",
        "n_refs_openalex",
        "venue_id",
    ],
    "venues": ["venue_id", "display_name"],
    "paper_attachments": ["attachment_key", "paper_id"],
    "authors": ["author_id", "display_name"],
    "paper_authors": ["paper_id", "author_id", "position"],
    "topics": ["topic_id", "display_name", "level", "parent_id"],
    "paper_topics": ["paper_id", "topic_id", "is_primary"],
    "cites": ["citing_paper_id", "cited_paper_id"],
    "coupled": ["paper_a", "paper_b", "shared_refs", "score"],
}
# the server half of the schema contract, in the same checkout as this test (tests/citegraph/ -> repo root)
SERVER_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "lightrag",
    "zotero_struct.py",
)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class PgArrayTest(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(bsl.pg_array([]), "{}")
        self.assertEqual(
            bsl.pg_array(["background", "methodology"]), "{background,methodology}"
        )
        self.assertEqual(bsl.pg_array(["a,b"]), '{"a,b"}')
        self.assertEqual(bsl.pg_array(['say "hi"']), '{"say \\"hi\\""}')
        self.assertEqual(bsl.pg_array(["{x}"]), '{"{x}"}')
        self.assertEqual(bsl.pg_array(["back\\slash"]), '{"back\\\\slash"}')
        self.assertEqual(bsl.pg_array([""]), '{""}')
        self.assertEqual(bsl.pg_array(["NULL"]), '{"NULL"}')
        self.assertEqual(bsl.pg_array(["two words"]), '{"two words"}')

    def test_round_trip_and_csv_quoting(self):
        nasty = [
            "a,b",
            'q"uote',
            "{brace}",
            "back\\slash",
            "",
            "NULL",
            " lead",
            "plain",
            "}{",
            'x","y',
        ]
        lit = bsl.pg_array(nasty)
        self.assertEqual(bsl.parse_pg_array(lit), nasty)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            bsl.write_csv(path, ["k", "arr"], [["1", nasty], ["2", []], ["3", None]])
            rows = read_csv(path)
        self.assertEqual(
            bsl.parse_pg_array(rows[0]["arr"]), nasty
        )  # the CSV layer quotes the literal again
        self.assertEqual(rows[1]["arr"], "{}")
        self.assertEqual(rows[2]["arr"], "")  # NULL
        for bad in ("a,b", "{a", '{"a}', "{a b}", "{,}", '{"a"b}'):
            with self.assertRaises(ValueError, msg=bad):
                bsl.parse_pg_array(bad)


class ContextCleanTest(unittest.TestCase):
    def test_control_characters_and_cut(self):
        t, cut = bsl.clean_context("a\tb\nc\r\nd\x00e\x07f\ud800g  ")
        self.assertEqual((t, cut), ("a b c  def" + "g", False))
        t, cut = bsl.clean_context("x" * 2500)
        self.assertEqual((len(t), cut), (2000, True))
        self.assertEqual(bsl.clean_context("y" * 2000), ("y" * 2000, False))
        self.assertEqual(bsl.clean_context(None), ("", False))
        self.assertEqual(bsl.clean_context(" \x00 "), ("", False))
        self.assertEqual(
            bsl.clean_context("Kühn – “quoted”"), ("Kühn – “quoted”", False)
        )


class EmbeddingFormatTest(unittest.TestCase):
    def test_csv_value_is_pgvector_text(self):
        self.assertEqual(
            bsl._fmt((0.5, -1.25, 0.123457, 3.0)), "[0.5,-1.25,0.123457,3.0]"
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e.csv")
            bsl.write_csv(
                path,
                ["paper_id", "model", "embedding"],
                [["doi:a", "specter_v2", tuple([0.25] * 768)]],
            )
            row = read_csv(path)[0]
        vec = row["embedding"]
        self.assertTrue(vec.startswith("[0.25,") and vec.endswith("0.25]"))
        self.assertEqual(len(vec[1:-1].split(",")), 768)

    def test_dimension_is_a_hard_check(self):
        t = {
            "papers": [],
            "paper_embeddings": [["doi:a", "specter_v2", tuple([0.1] * 767)]],
        }
        self.assertIn("paper_embeddings", "\n".join(bsl.hard_checks_v2(t)))
        t["paper_embeddings"] = [["doi:a", "specter_v2", tuple([0.1] * 768)]]
        self.assertEqual(bsl.hard_checks_v2(t), [])
        t["paper_embeddings"] = [["doi:a", "specter_v2", tuple([float("nan")] * 768)]]
        self.assertTrue(bsl.hard_checks_v2(t))


class SchemaV2Test(unittest.TestCase):
    def test_v1_columns_kept_new_ones_at_the_end(self):
        for tname, cols in SERVER_COLUMNS.items():
            have = [c for c, _ in bsl.TABLES[tname]["cols"]]
            self.assertTrue(set(cols) <= set(have), tname)
        v1 = {
            "papers": [
                "paper_id",
                "doi",
                "openalex_id",
                "title",
                "year",
                "venue_id",
                "type",
                "cited_by_count",
                "n_refs_openalex",
                "n_attachments",
            ],
            "cites": ["citing_paper_id", "cited_paper_id"],
        }
        for tname, cols in v1.items():
            self.assertEqual(
                [c for c, _ in bsl.TABLES[tname]["cols"]][: len(cols)], cols
            )
        self.assertEqual(
            [c for c, _ in bsl.TABLES["papers"]["cols"]][10:],
            [
                "s2_paper_id",
                "s2_corpus_id",
                "s2_citation_count",
                "s2_influential_citation_count",
                "s2_reference_count",
                "tldr",
                "doi_source",
            ],
        )
        self.assertEqual(
            [c for c, _ in bsl.TABLES["cites"]["cols"]][2:],
            ["source", "is_influential", "intents", "n_contexts"],
        )
        types = dict(bsl.TABLES["cites"]["cols"])
        self.assertEqual(
            (types["is_influential"], types["intents"]), ("boolean", "text[]")
        )
        self.assertEqual(
            dict(bsl.TABLES["paper_embeddings"]["cols"])["embedding"], "vector(768)"
        )
        self.assertIn(
            'paper_a COLLATE "C" < paper_b COLLATE "C"', bsl.TABLES["coupled"]["checks"]
        )

    def test_ddl_of_the_new_parts(self):
        ddl = bsl.schema_sql("zs")
        for needle in (
            "source text NOT NULL",
            "intents text[]",
            "CHECK (source IN ('openalex', 's2', 'both'))",
            "CHECK (doi_source IN ('zotero', 's2_title'))",
            "FOREIGN KEY (citing_paper_id, cited_paper_id) REFERENCES zs.cites (citing_paper_id, cited_paper_id)",
            "PRIMARY KEY (citing_paper_id, cited_paper_id, ordinal)",
            "embedding vector(768)",
            "FOREIGN KEY (work_id) REFERENCES zs.external_works (work_id)",
            'CHECK (paper_a COLLATE "C" < paper_b COLLATE "C")',
        ):
            self.assertIn(needle, ddl)
        self.assertNotRegex(ddl, r"(?i)extension")


def s2rec_(n, doi=None, tldr=True, **kw):
    r = {
        "paperId": f"{n:040x}",
        "corpusId": n,
        "externalIds": {"CorpusId": n, **({"DOI": doi} if doi else {})},
        "title": f"S2 paper {n}",
        "year": 2000,
        "referenceCount": 7,
        "citationCount": 11,
        "influentialCitationCount": 2,
        "tldr": {"model": "tldr@v2.0.0", "text": f"TLDR {n}"} if tldr else None,
    }
    r.update(kw)
    return r


def paper_tables(*pids):
    """Minimal tables for the cites / external builders."""
    return {
        "papers": [[p, None, None, "t", 2000, None, None, None, None, 1] for p in pids]
    }


class CitesUnionTest(unittest.TestCase):
    def make_store(self, d, rows, lib_ids=()):
        path = os.path.join(d, "s2store", "citations.sqlite")
        con = sync.open_store(path)
        sync.commit_batch(
            con,
            rows,
            meta={"release_id": "2026-11-17", "mode": "full"},
            library_ids=lib_ids,
        )
        con.close()
        return path

    def test_union_source_tags_and_fields(self):
        c = json.dumps
        rows = [
            (
                1,
                10,
                20,
                1,
                c(['c "1", with comma', "second\nline {x}"]),
                c([["background"], ["methodology", "background"]]),
            ),
            (
                6,
                10,
                20,
                0,
                c(["x" * 2500, 'c "1", with comma']),
                c([["result"], ["background"]]),
            ),
            (2, 30, 10, 0, None, None),  # S2 row with no contexts
            (3, 10, 10, 1, c(["self"]), c([["background"]])),  # self-citation
            (
                4,
                999,
                10,
                1,
                c(["outside"]),
                c([["background"]]),
            ),  # citing paper not in the library
            (5, 10, None, 0, None, None),  # unresolved cited paper
        ]
        with tempfile.TemporaryDirectory() as d:
            path = self.make_store(d, rows)
            cid = {10: "A", 20: "B", 30: "C"}
            edges, info = bsl.read_s2_store(path, cid, [10, 20, 30])
        self.assertEqual(set(edges), {("A", "B"), ("C", "A")})
        e = edges[("A", "B")]
        self.assertTrue(e["influential"])
        self.assertEqual(e["intents"], ["background", "methodology", "result"])
        self.assertEqual(
            [t for t, _ in e["contexts"]],
            ['c "1", with comma', "second line {x}", "x" * 2000],
        )
        self.assertEqual(e["contexts"][0][1], ["background"])
        self.assertEqual(e["contexts"][1][1], ["background", "methodology"])
        self.assertEqual(e["cut"], 1)
        self.assertEqual(info["rows"], 6)
        self.assertEqual((info["release_id"], info["mode"]), ("2026-11-17", "full"))
        self.assertEqual(
            info["library_ids_missing"], 3
        )  # the store's library_ids table is empty

        tables = {"cites": [["A", "B"], ["D", "A"], ["D", "C"]]}
        qa = bsl.build_cites(tables, edges)
        self.assertEqual(
            tables["cites"],
            [
                ["A", "B", "both", True, ["background", "methodology", "result"], 3],
                [
                    "C",
                    "A",
                    "s2",
                    False,
                    [],
                    0,
                ],  # an S2 row without contexts: 0, not null
                ["D", "A", "openalex", None, None, None],  # no S2 row: nulls
                ["D", "C", "openalex", None, None, None],
            ],
        )
        self.assertEqual(qa["by_source"], {"openalex": 2, "s2": 1, "both": 1})
        self.assertEqual(
            (qa["with_context"], qa["total_contexts"], qa["cut_contexts"]), (1, 3, 1)
        )
        self.assertEqual(
            tables["cite_contexts"][:2],
            [
                ["A", "B", 1, 'c "1", with comma', ["background"]],
                ["A", "B", 2, "second line {x}", ["background", "methodology"]],
            ],
        )
        self.assertEqual([r[2] for r in tables["cite_contexts"]], [1, 2, 3])

    def test_duplicate_corpus_id_maps_neither(self):
        recs = {"A": s2rec_(10), "B": s2rec_(20), "G": s2rec_(60), "H": s2rec_(60)}
        unique, ambiguous, all_ids = bsl.corpus_map(recs)
        self.assertEqual(unique, {10: "A", 20: "B"})
        self.assertEqual(ambiguous, {60: ["G", "H"]})
        self.assertEqual(all_ids, [10, 20, 60])
        rows = [(1, 60, 10, 0, None, None), (2, 10, 20, 0, None, None)]
        with tempfile.TemporaryDirectory() as d:
            path = self.make_store(d, rows)
            edges, _ = bsl.read_s2_store(path, unique, all_ids)
        self.assertEqual(set(edges), {("A", "B")})  # nothing for the shared id

    def test_hard_checks_pass_on_the_built_tables_and_catch_a_dangling_context(self):
        t = {
            "venues": [],
            "topics": [],
            "paper_attachments": [],
            "authors": [],
            "paper_authors": [],
            "paper_topics": [],
            "keywords": [],
            "paper_keywords": [],
            "coupled": [],
            "papers": [
                [
                    "A",
                    "a",
                    None,
                    "t",
                    2000,
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "zotero",
                ],
                [
                    "B",
                    "b",
                    None,
                    "t",
                    2000,
                    None,
                    None,
                    None,
                    None,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
            ],
            "cites": [["A", "B", "both", True, ["x"], 1]],
            "cite_contexts": [["A", "B", 1, "ctx", ["x"]]],
        }
        self.assertEqual(bsl.hard_checks(t) + bsl.hard_checks_v2(t), [])
        t["cite_contexts"].append(["B", "A", 1, "ctx", []])  # no such cites row
        self.assertIn(
            "cite_contexts.citing_paper_id,cited_paper_id",
            "\n".join(bsl.hard_checks(t)),
        )
        t["cites"] = [
            ["A", "B", "openalex", True, None, None]
        ]  # S2 data on an OpenAlex-only edge
        t["cite_contexts"] = [["A", "B", 1, "bad\x00", []]]
        text = "\n".join(bsl.hard_checks_v2(t))
        self.assertIn("do not match their source", text)
        self.assertIn("control-character", text)


class ExternalWorksTest(unittest.TestCase):
    def aux(self):
        return {
            "oa_to_paper": {"W1": "pa", "W2": "pb"},
            "refs_by_paper": {
                "pa": [
                    "W2",
                    "W101",
                    "W102",
                    "W103",
                    "W1",
                ],  # W2 is a library paper, W1 is itself
                "pb": ["W101", "W102"],
                "pc": ["W101", "W103"],
                "pd": [],
            },
        }

    def test_threshold_counts_a_work_once_per_citing_paper(self):
        citers = bsl.external_citers(self.aux())
        self.assertEqual(
            citers,
            {"W101": ["pa", "pb", "pc"], "W102": ["pa", "pb"], "W103": ["pa", "pc"]},
        )
        aux = self.aux()
        aux["refs_by_paper"]["pa"] = [
            "W101",
            "W101",
            "W102",
        ]  # a list with a repeat is one citer
        aux["refs_by_paper"]["pc"] = ["W103"]
        self.assertEqual(
            bsl.external_citers(
                {
                    **aux,
                    "refs_by_paper": {
                        k: sorted(set(v)) for k, v in aux["refs_by_paper"].items()
                    },
                }
            ),
            {"W101": ["pa", "pb"], "W102": ["pa", "pb"]},
        )
        with mock.patch.object(bsl, "MIN_EXTERNAL_CITERS", 3):
            self.assertEqual(list(bsl.external_citers(self.aux())), ["W101"])

    def test_assemble_exposes_reference_lists_and_a_real_work_counts_once(self):
        md = {"AAAAAAA1": entry(doi="10.1/a"), "AAAAAAA2": entry(doi="10.1/b")}
        works = {
            "10.1/a": work("10.1/a", "W1", refs=["W2", "W101", "W101", "W102"]),
            "10.1/b": work("10.1/b", "W2", refs=["W101", "W102", "W9"]),
        }
        _, aux = bsl.assemble(md, works)
        self.assertEqual(
            bsl.external_citers(aux),
            {
                "W101": ["doi:10.1/a", "doi:10.1/b"],
                "W102": ["doi:10.1/a", "doi:10.1/b"],
            },
        )

    def test_tables_and_top_list(self):
        citers = {"W101": ["pa", "pb", "pc"], "W102": ["pa", "pb"]}
        ext = {
            "W101": {
                "id": "https://openalex.org/W101",
                "doi": "https://doi.org/10.9/X1",
                "title": None,
                "display_name": "Display One",
                "publication_year": 1990,
                "type": "article",
                "cited_by_count": 50,
                "authorships": [
                    {
                        "author": {
                            "id": "https://openalex.org/A1",
                            "display_name": "Zed Zee",
                        }
                    }
                ],
                "primary_location": {"source": {"display_name": "Jour X"}},
            }
        }
        tables = {}
        qa = bsl.build_external(tables, citers, ext)
        self.assertEqual(
            tables["external_works"],
            [
                [
                    "oa:W101",
                    "W101",
                    "10.9/x1",
                    "Display One",
                    1990,
                    "article",
                    50,
                    "Zed Zee",
                    "Jour X",
                    3,
                ],
                [
                    "oa:W102",
                    "W102",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    2,
                ],  # not fetched: id and count only
            ],
        )
        self.assertEqual(
            tables["cites_external"],
            [
                ["pa", "oa:W101"],
                ["pa", "oa:W102"],
                ["pb", "oa:W101"],
                ["pb", "oa:W102"],
                ["pc", "oa:W101"],
            ],
        )
        self.assertEqual(
            (qa["count"], qa["with_metadata"], qa["cites_external_rows"]), (2, 1, 5)
        )
        self.assertEqual(qa["top_cited_by_library"][0]["n_library_citers"], 3)

    def test_harvest_asks_only_for_missing_ids_50_at_a_time_and_trims_authorships(self):
        calls = []

        def fake(url, headers):
            calls.append((url, dict(headers)))
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            ids = q["filter"][0][len("openalex_id:") :].split("|")
            return {
                "results": [
                    {
                        "id": f"https://openalex.org/{i}",
                        "title": i,
                        "authorships": [
                            {
                                "author": {
                                    "id": "A",
                                    "display_name": "N",
                                    "orcid": "x",
                                },
                                "institutions": [{"id": "I"}] * 20,
                            }
                        ],
                    }
                    for i in ids
                    if i != "W7"
                ]
            }

        store = {"works": {"W0": {"id": "https://openalex.org/W0"}}, "unknown": []}
        ids = ["W0"] + [f"W{i}" for i in range(1, 121)]
        with (
            mock.patch.object(bsl, "_urlopen_json", fake),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
        ):
            st = bsl.harvest_external(ids, store, "me@example.org", OAKEY)
            self.assertEqual((st["batches"], st["failed"]), (3, 0))
            self.assertEqual(
                [
                    len(
                        urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["filter"][
                            0
                        ].split("|")
                    )
                    for u, _ in calls
                ],
                [50, 50, 20],
            )
            q = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0][0]).query)
            self.assertEqual(q["select"], [bsl.EXT_SELECT])
            self.assertEqual(q["mailto"], ["me@example.org"])
            for u, h in calls:
                self.assertNotIn(OAKEY, u)
                self.assertEqual(h["api_key"], OAKEY)
            self.assertEqual(store["unknown"], ["W7"])
            self.assertEqual(
                set(store["works"]["W1"]["authorships"][0]), {"author"}
            )  # no institutions kept
            self.assertEqual(
                store["works"]["W1"]["authorships"][0]["author"],
                {"id": "A", "display_name": "N"},
            )
            n = len(calls)
            bsl.harvest_external(
                ids, store, "", ""
            )  # a rerun: only ids not cached; the unknown one stays unknown
            self.assertEqual(len(calls), n)
            bsl.harvest_external(ids[:10], store, "", "", refresh=True)
            self.assertEqual(len(calls), n + 1)

    def test_resolve_unknown_sorts_deleted_merged_found_and_failed(self):
        calls = []

        def fake(url, headers):
            calls.append((url, dict(headers)))
            wid = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
            if wid == "W1":
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            if wid == "W4":
                raise urllib.error.HTTPError(url, 500, "boom", {}, None)
            new = {"W2": "W20", "W3": "W3"}[
                wid
            ]  # W2 was merged into W20; W3 the list filter missed
            return {
                "id": f"https://openalex.org/{new}",
                "title": new,
                "authorships": [
                    {
                        "author": {"id": "A", "display_name": "N", "orcid": "x"},
                        "institutions": [{"id": "I"}],
                    }
                ],
            }

        # W8 and W9 were settled by an earlier run: not asked again
        store = {
            "works": {},
            "unknown": ["W1", "W2", "W3", "W4", "W8", "W9"],
            "merged": {"W9": "W10"},
            "deleted": ["W8"],
        }
        with (
            mock.patch.object(bsl, "_urlopen_json", fake),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            st = bsl.resolve_external_unknown(store, "", OAKEY)
            self.assertEqual(
                st, {"asked": 4, "found": 1, "merged": 1, "deleted": 1, "failed": 1}
            )
            self.assertEqual(store["merged"], {"W9": "W10", "W2": "W20"})
            self.assertEqual(store["deleted"], ["W1", "W8"])
            self.assertEqual(
                store["unknown"], ["W4"]
            )  # the failed one is asked again next run
            self.assertEqual(sorted(store["works"]), ["W20", "W3"])
            self.assertEqual(
                store["works"]["W20"]["authorships"],
                [{"author": {"id": "A", "display_name": "N"}}],
            )
            asked = {urllib.parse.urlparse(u).path for u, _ in calls}
            self.assertEqual(
                asked, {"/works/W1", "/works/W2", "/works/W3", "/works/W4"}
            )
            for u, h in calls:
                self.assertNotIn(OAKEY, u)
                self.assertEqual(h["api_key"], OAKEY)
                self.assertEqual(
                    urllib.parse.parse_qs(urllib.parse.urlparse(u).query)["select"],
                    [bsl.EXT_SELECT],
                )
            n = len(calls)
            st = bsl.resolve_external_unknown(store, "", OAKEY)
            self.assertEqual((st["asked"], st["failed"]), (1, 1))
            self.assertEqual(
                {urllib.parse.urlparse(u).path for u, _ in calls[n:]}, {"/works/W4"}
            )

    def test_resolve_unknown_stops_after_ten_failures_in_a_row(self):
        def down(url, headers):
            raise urllib.error.HTTPError(url, 503, "down", {}, None)

        ids = [f"W{i}" for i in range(100, 115)]
        store = {"works": {}, "unknown": list(ids)}
        with (
            mock.patch.object(bsl, "_urlopen_json", down),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
            mock.patch("sys.stderr", io.StringIO()) as err,
        ):
            st = bsl.resolve_external_unknown(store, "", "")
        self.assertEqual((st["asked"], st["failed"]), (10, 10))
        self.assertEqual(
            store["unknown"], sorted(ids)
        )  # nothing settled: all are asked again next run
        self.assertEqual((store["merged"], store["deleted"]), ({}, []))
        self.assertIn("5 left for the next run", err.getvalue())

    def test_assemble_rewrites_merged_and_drops_deleted_references(self):
        md = {"AAAAAAA1": entry(doi="10.1/a"), "AAAAAAA2": entry(doi="10.1/b")}
        works = {
            "10.1/a": work("10.1/a", "W1", refs=["W901", "W902", "W101"]),
            "10.1/b": work("10.1/b", "W2", refs=["W902", "W904"]),
        }
        tables, aux = bsl.assemble(
            md, works
        )  # without the map: no citation, the deleted work looks shared
        self.assertEqual(tables["cites"], [])
        self.assertEqual(
            bsl.external_citers(aux), {"W902": ["doi:10.1/a", "doi:10.1/b"]}
        )
        tables, aux = bsl.assemble(
            md, works, None, {"W901": "W2", "W904": "W101"}, ["W902"]
        )
        self.assertEqual(
            tables["cites"], [["doi:10.1/a", "doi:10.1/b"]]
        )  # W901 was merged into library paper B
        self.assertEqual(
            aux["refs_by_paper"], {"doi:10.1/a": ["W101", "W2"], "doi:10.1/b": ["W101"]}
        )
        self.assertEqual(
            bsl.external_citers(aux), {"W101": ["doi:10.1/a", "doi:10.1/b"]}
        )


class LibraryAliasTest(unittest.TestCase):
    """Outside works that are library papers under another OpenAlex id."""

    def tables(self):
        def paper(pid, doi, title, year):
            return [pid, doi, None, title, year, None, None, None, None, 1]

        return {
            "papers": [
                paper("doi:10.1/g", "10.1/g", "Paper G", 2004),
                paper("zot:KJ", None, "Alternative methods to smooth the field", 1981),
                paper(
                    "zot:KR", None, "Introduction to the mathematics of inversion", 1977
                ),
                paper("zot:KN", None, "A handbook of geophysics, no authors", 1985),
                paper("zot:KX", None, "Same title in the library twice", 2000),
                paper("zot:KY", None, "Same title in the library twice", 2000),
                paper("zot:KS", None, "Introduction", 2001),
            ],
            "authors": [
                ["z:jekeli|c", "Christopher Jekeli", None, 1],
                ["z:twomey|s", "Sean Twomey", None, 1],
            ],
            "paper_authors": [
                ["zot:KJ", "z:jekeli|c", 1, "zotero"],
                ["zot:KR", "z:twomey|s", 1, "zotero"],
            ],
        }

    @staticmethod
    def ext(title, year, *names, doi=None):
        return {
            "doi": f"https://doi.org/{doi}" if doi else None,
            "title": title,
            "publication_year": year,
            "authorships": [{"author": {"display_name": n}} for n in names],
        }

    def aliases(self, works):
        return bsl.library_aliases(
            self.tables(), {w: ["pa", "pb"] for w in works}, works
        )

    def test_doi_title_author_and_year_rules(self):
        e = self.ext
        works = {
            "W1": e("Anything", 1950, doi="10.1/G"),  # the DOI is a library paper's
            "W2": e("Alternative Methods to Smooth the Field.", 1982, "C. Jekeli"),
            "W3": e("Alternative methods to smooth the field", 1990, "C. Jekeli"),
            "W4": e(
                "Introduction to the mathematics of inversion", 1977, "Brianne Hobbs"
            ),
            "W5": e("A handbook of geophysics - no authors", 1985),
            "W6": e("A handbook of geophysics, no authors", 1986),
            "W7": e("Same title in the library twice", 2000),
            "W8": e("Introduction", 2001),
        }
        aliases, qa = self.aliases(works)
        # W3: years too far apart; W4: a review, by another author; W6: no names, and not the same
        # year; W7: two library papers fit; W8: too short a title to match
        self.assertEqual(aliases, {"W1": "doi:10.1/g", "W2": "zot:KJ", "W5": "zot:KN"})
        self.assertEqual(
            (qa["by_doi"], qa["by_title"], qa["ambiguous_titles"]), (1, 2, 1)
        )

    def test_only_candidates_with_metadata_are_matched(self):
        works = {"W2": self.ext("Alternative methods to smooth the field", 1981)}
        aliases, _ = bsl.library_aliases(self.tables(), {"W9": ["pa", "pb"]}, works)
        self.assertEqual(aliases, {})

    def test_an_alias_is_a_citation_and_no_outside_work(self):
        md = {
            "AAAAAAA1": entry(doi="10.1/a"),
            "AAAAAAA2": entry(doi="10.1/b"),
            "AAAAAAA3": entry(item_key="KJ", title="A report OpenAlex lacks"),
        }
        works = {
            "10.1/a": work("10.1/a", "W1", refs=["W2", "W500", "W101"]),
            "10.1/b": work("10.1/b", "W2", refs=["W500", "W101"]),
        }
        tables, aux = bsl.assemble(
            md, works, aliases={"W500": "zot:KJ", "W2": "zot:KJ"}
        )
        self.assertEqual(
            sorted(map(tuple, tables["cites"])),
            [
                (
                    "doi:10.1/a",
                    "doi:10.1/b",
                ),  # W2 is paper B's own id: the alias is ignored
                ("doi:10.1/a", "zot:KJ"),
                ("doi:10.1/b", "zot:KJ"),
            ],
        )
        self.assertEqual(
            aux["alias_edges"], {("doi:10.1/a", "zot:KJ"), ("doi:10.1/b", "zot:KJ")}
        )
        self.assertEqual(list(bsl.external_citers(aux)), ["W101"])

    def test_the_reference_check_leaves_alias_edges_out(self):
        tables = {
            "cites": [["doi:10.1/a", "doi:10.1/b"], ["doi:10.1/a", "doi:10.1/g"]],
            "papers": [["doi:10.1/a", "10.1/a", "W1"], ["doi:10.1/b", "10.1/b", "W2"]],
        }
        ref = {"10.1/a": {"cited": ["10.1/b"]}}
        self.assertEqual(bsl.compare_reference(tables, ref)["edges_only_ours"], 1)
        kept = bsl.compare_reference(tables, ref, {("doi:10.1/a", "doi:10.1/g")})
        self.assertEqual((kept["edges_only_ours"], kept["jaccard"]), (0, 1.0))


class TitleMatchBuildTest(unittest.TestCase):
    def groups(self):
        md = {
            "AAAAAAA1": entry(doi="10.1/a", title="Paper A"),
            "AAAAAAA2": entry(
                item_key="KD",
                title="Numerical Recipes in C",
                year="1992",
                creators=[["Press", "William H."]],
            ),
            "AAAAAAA3": entry(
                item_key="KE",
                title="Collision of galaxies",
                year="2001",
                creators=[["Lee", "Ann"]],
            ),
            "AAAAAAA4": entry(doi="10.1/f", title="Quite different", year="2001"),
            "AAAAAAA5": entry(doi="10.1/known", title="Known to S2"),
        }
        return md, bsl.group_papers(md)

    def test_candidates(self):
        md, groups = self.groups()
        works = {"10.1/a": work("10.1/a", "W1")}
        s2_by_id = {"DOI:10.1/a": None, "DOI:10.1/f": None, "DOI:10.1/known": s2rec_(5)}
        cands = bsl.title_candidates(groups, works, s2_by_id)
        # no DOI -> candidate; DOI unknown to both -> candidate; known to OpenAlex or S2 -> not
        self.assertEqual(
            [c["paper_id"] for c in cands], ["doi:10.1/f", "zot:KD", "zot:KE"]
        )
        kd = cands[1]
        self.assertEqual(
            (kd["title"], kd["year"], kd["first_family"]),
            ("Numerical Recipes in C", 1992, "Press"),
        )
        # a DOI whose S2 status was never asked is not a candidate
        self.assertEqual(
            [c["paper_id"] for c in bsl.title_candidates(groups, works, {})],
            ["zot:KD", "zot:KE"],
        )

    def test_doi_assigned_conflict_and_paper_id_stays(self):
        md, groups = self.groups()
        cands = bsl.title_candidates(
            groups,
            {"10.1/a": work("10.1/a", "W1")},
            {"DOI:10.1/f": None, "DOI:10.1/known": s2rec_(5)},
        )
        matches = {}
        for c, rec in zip(
            cands,
            [
                None,  # F: asked, no match
                s2rec_(
                    40,
                    "10.1/DD",
                    title="Numerical Recipes in C",
                    year=1992,
                    authors=[{"name": "W. Press"}],
                ),
                s2rec_(
                    50,
                    "10.1/A",
                    title="Collision of galaxies",
                    year=2001,
                    authors=[{"name": "Ann Lee"}],
                ),
            ],
        ):
            matches[s2_api.match_key(c["paper_id"], c["title"], c["year"])] = {
                "record": rec
            }
        res = bsl.resolve_matches(cands, matches, ["10.1/a", "10.1/f", "10.1/known"])
        self.assertEqual(res["overrides"], {"zot:KD": "10.1/dd"})
        self.assertEqual(res["decisions"], {"reject:none": 1, "accepted": 2})
        self.assertEqual([c["paper_id"] for c in res["conflicts"]], ["zot:KE"])
        self.assertEqual(res["conflicts"][0]["doi"], "10.1/a")
        self.assertEqual(
            list(res["records"]), ["zot:KD"]
        )  # the conflicting match's S2 record is not attached
        self.assertEqual(len(res["examples"]), 1)
        # the new DOI flows through assemble like any DOI paper; paper_id stays zot:
        works = {
            "10.1/a": work("10.1/a", "W1"),
            "10.1/dd": work(
                "10.1/dd",
                "W4",
                refs=["W1"],
                authors=[("A9", "Bill Press")],
                source=("S3", "NR Press"),
            ),
        }
        tables, aux = bsl.assemble(md, works, res["overrides"])
        bsl.extend_papers(
            tables, {"zot:KD": res["records"]["zot:KD"]}, res["overrides"]
        )
        p = {r[0]: r for r in tables["papers"]}
        i = {c: k for k, (c, _) in enumerate(bsl.TABLES["papers"]["cols"])}
        kd = p["zot:KD"]
        self.assertEqual(
            (
                kd[i["doi"]],
                kd[i["doi_source"]],
                kd[i["openalex_id"]],
                kd[i["venue_id"]],
            ),
            ("10.1/dd", "s2_title", "W4", "oa:S3"),
        )
        self.assertEqual(kd[i["s2_corpus_id"]], 40)
        self.assertEqual(kd[i["tldr"]], "TLDR 40")
        self.assertEqual(p["doi:10.1/a"][i["doi_source"]], "zotero")
        self.assertIsNone(p["zot:KE"][i["doi_source"]])
        self.assertIn(
            ["zot:KD", "doi:10.1/a"], tables["cites"]
        )  # its OpenAlex references became edges
        self.assertIn(
            ("zot:KD", "oa:A9"), [(r[0], r[1]) for r in tables["paper_authors"]]
        )
        self.assertEqual(aux["doi_overrides"], {"zot:KD": "10.1/dd"})

    def test_first_assignment_wins_when_two_papers_find_the_same_doi(self):
        cands = [
            {
                "paper_id": "zot:K1",
                "doi": "",
                "title": "Same",
                "year": None,
                "first_family": "",
            },
            {
                "paper_id": "zot:K2",
                "doi": "",
                "title": "Same",
                "year": None,
                "first_family": "",
            },
        ]
        rec = s2rec_(1, "10.5/x", title="Same", authors=[])
        matches = {
            s2_api.match_key(c["paper_id"], "Same", None): {"record": rec}
            for c in cands
        }
        # same paper_id is part of the key, so give each its entry
        res = bsl.resolve_matches(cands, matches, [])
        self.assertEqual(res["overrides"], {"zot:K1": "10.5/x"})
        self.assertEqual([c["paper_id"] for c in res["conflicts"]], ["zot:K2"])


class SwapFingerprintTest(unittest.TestCase):
    """The server decides that the tables changed by the OID and filenode of the
    tables in the schema NAMED <schema>. ALTER SCHEMA ... RENAME keeps each table's
    OID, so the question is whether the tables behind the name change: they do."""

    class Catalog:
        def __init__(self):
            self.rel = []  # [nspname, relname, oid, filenode, relkind]
            self.next_oid = 16400

        def create(self, nsp, name, kind="r"):
            self.rel.append([nsp, name, self.next_oid, self.next_oid + 1000, kind])
            self.next_oid += 1

        def rename(self, old, new):
            if not any(r[0] == old for r in self.rel) and not self.exists(old):
                return
            for r in self.rel:
                if r[0] == old:
                    r[0] = new
            self.ns = getattr(self, "ns", set())

        def exists(self, nsp):
            return any(r[0] == nsp for r in self.rel)

    def fingerprint(self, cat, schema):
        # the server's fingerprint SQL: relkind 'r' in the schema named $1, then _TABLES
        found = {r[1]: (r[2], r[3]) for r in cat.rel if r[0] == schema and r[4] == "r"}
        missing = [t for t in SERVER_TABLES if t not in found]
        if missing:
            raise LookupError(missing)
        return tuple(sorted((t, found[t]) for t in SERVER_TABLES))

    def apply(self, cat, sql, schema):
        """What apply_<schema>.sql creates: its CREATE TABLE and CREATE VIEW statements."""
        for kind, nsp, name in re.findall(r"CREATE (TABLE|VIEW) (\w+)\.(\w+)", sql):
            cat.create(nsp, name, "r" if kind == "TABLE" else "v")

    def swap(self, cat, sql):
        sql = "\n".join(
            ln for ln in sql.splitlines() if not ln.lstrip().startswith("--")
        )  # psql skips comments
        assert "BEGIN;" in sql and sql.rstrip().endswith("COMMIT;")  # one transaction
        for m in re.finditer(
            r"DROP SCHEMA IF EXISTS (\w+) CASCADE|ALTER SCHEMA (\w+) RENAME TO (\w+)",
            sql,
        ):
            if m.group(1):
                cat.rel = [r for r in cat.rel if r[0] != m.group(1)]
            elif cat.exists(
                m.group(2)
            ):  # the DO block renames the live schema only if it exists
                cat.rename(m.group(2), m.group(3))

    def test_the_fingerprint_sql_is_the_servers(self):
        if not os.path.exists(SERVER_SOURCE):
            self.skipTest("lightrag/zotero_struct.py not in this checkout")
        norm = lambda t: re.sub(r"\s+", " ", t)  # noqa: E731
        self.assertIn(norm(SERVER_FINGERPRINT_SQL), norm(read(SERVER_SOURCE)))
        self.assertTrue(set(SERVER_TABLES) <= set(bsl.TABLES))

    def test_swap_changes_the_fingerprint_of_the_live_name(self):
        cat = self.Catalog()
        for t in bsl.TABLES:  # the live schema, as built the last time
            cat.create("zs", t)
        cat.create("zs", "paper_document", "v")
        before = self.fingerprint(cat, "zs")
        old_oids = {r[1]: r[2] for r in cat.rel if r[0] == "zs"}
        self.apply(cat, bsl.apply_sql("zs", "zotero"), "zs")
        self.assertEqual(
            self.fingerprint(cat, "zs"), before
        )  # building zs_next does not disturb the live one
        self.swap(cat, bsl.swap_sql("zs"))
        after = self.fingerprint(cat, "zs")
        self.assertNotEqual(after, before)
        for (t, (oid, fnode)), (_, (oid0, _f0)) in zip(after, before):
            self.assertNotEqual(
                oid, oid0, t
            )  # every table behind the name is a different table
        # the old tables keep their OIDs (a rename does not change them) under the _prev name
        self.assertEqual({r[1]: r[2] for r in cat.rel if r[0] == "zs_prev"}, old_oids)
        self.assertFalse(cat.exists("zs_next"))
        # a second round: prev is dropped, the live one becomes prev
        live = {r[2] for r in cat.rel if r[0] == "zs" and r[4] == "r"}
        self.apply(cat, bsl.apply_sql("zs", "zotero"), "zs")
        self.swap(cat, bsl.swap_sql("zs"))
        self.assertNotEqual(self.fingerprint(cat, "zs"), after)
        self.assertEqual(
            {r[2] for r in cat.rel if r[0] == "zs_prev" and r[4] == "r"}, live
        )
        self.assertFalse(
            set(old_oids.values()) & {r[2] for r in cat.rel}
        )  # the oldest generation is gone

    def test_first_swap_with_no_live_schema(self):
        cat = self.Catalog()
        with self.assertRaises(LookupError):
            self.fingerprint(cat, "zs")  # the server's no_schema state
        self.apply(cat, bsl.apply_sql("zs", "zotero"), "zs")
        self.swap(cat, bsl.swap_sql("zs"))
        self.assertEqual(len(self.fingerprint(cat, "zs")), len(SERVER_TABLES))
        self.assertFalse(cat.exists("zs_prev"))


class SqlGenerationTest(unittest.TestCase):
    def build(self, d, schema, workspace="zotero"):
        md = os.path.join(d, "md.json")
        dump({"AAAAAAA1": entry(doi="10.1/a")}, md)
        out = os.path.join(d, "out")
        rc = quiet_main(
            [
                "--metadata",
                md,
                "--out",
                out,
                "--cache",
                os.path.join(d, "cache", "w.json"),
                "--offline",
                "--schema",
                schema,
                "--workspace",
                workspace,
            ]
        )
        self.assertEqual(rc, 0)
        return out

    def test_every_sql_file_follows_the_schema_name(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.build(d, "citeg", "o'brien")
            names = sorted(f for f in os.listdir(out) if f.endswith(".sql"))
            self.assertEqual(
                names,
                [
                    "apply_citeg.sql",
                    "drop_prev_citeg.sql",
                    "load.sql",
                    "schema.sql",
                    "swap_citeg.sql",
                ],
            )
            sql = {n: read(os.path.join(out, n)) for n in names}
        for n, text in sql.items():
            self.assertNotIn("zstruct", text, n)
            self.assertNotRegex(text, r"(?i)(create|alter|drop)\s+extension", n)
        self.assertIn("CREATE SCHEMA citeg;", sql["schema.sql"])
        self.assertIn("\\copy citeg.papers (", sql["load.sql"])
        a = sql["apply_citeg.sql"]
        self.assertIn("DROP SCHEMA IF EXISTS citeg_next CASCADE;", a)
        self.assertIn("CREATE SCHEMA citeg_next;", a)
        self.assertIn("CREATE VIEW citeg_next.paper_document AS", a)
        self.assertIn(
            "d.workspace = 'o''brien' AND d.status = 'processed'", a
        )  # the 11-16 fix, quoted correctly
        self.assertIn("split_part(d.file_path, '__', 1) = a.attachment_key", a)
        self.assertIn('COLLATE "C"', a)
        self.assertEqual(a.count("BEGIN;"), 1)
        self.assertEqual(a.count("COMMIT;"), 1)
        for t in bsl.TABLES:
            self.assertIn(f"\\copy citeg_next.{t} (", a)
            self.assertIn(f"ANALYZE citeg_next.{t};", a)
            self.assertIn(f"CREATE TABLE citeg_next.{t} (", a)
        # nothing outside the new schema except the one table the view reads
        refs = set(
            re.findall(
                r"\b(?:FROM|JOIN|REFERENCES|ON|TABLE|VIEW|ANALYZE|copy)\s+(\w+)\.(\w+)",
                a,
            )
        )
        self.assertEqual({s for s, _ in refs} - {"citeg_next"}, {"public"})
        self.assertEqual({t for s, t in refs if s == "public"}, {"lightrag_doc_status"})
        self.assertNotRegex(a, r"lightrag_graph|lightrag_vdb")
        s = sql["swap_citeg.sql"]
        self.assertIn("DROP SCHEMA IF EXISTS citeg_prev CASCADE;", s)
        self.assertIn("ALTER SCHEMA citeg RENAME TO citeg_prev;", s)
        self.assertIn("ALTER SCHEMA citeg_next RENAME TO citeg;", s)
        self.assertLess(s.index("citeg_prev;"), s.index("citeg_next RENAME"))
        self.assertEqual((s.count("BEGIN;"), s.count("COMMIT;")), (1, 1))
        self.assertEqual(
            sql["drop_prev_citeg.sql"].count(
                "DROP SCHEMA IF EXISTS citeg_prev CASCADE;"
            ),
            1,
        )

    def test_default_schema_is_zstruct_and_bad_names_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = self.build(d, "zstruct")
            self.assertTrue(os.path.exists(os.path.join(out, "apply_zstruct.sql")))
            self.assertIn("zstruct_next", read(os.path.join(out, "swap_zstruct.sql")))
        for bad in ("Zs", "a-b", "x; DROP", "", "1abc"):
            with self.assertRaises(ValueError, msg=bad):
                bsl.check_schema_name(bad)
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            bsl.main(["--metadata", "m", "--out", "o", "--schema", "bad name"])


class World:
    """A small library with fake OpenAlex and S2 services, and an S2 citation store."""

    METADATA = {
        "AAAAAAA1": entry(
            doi="10.1/a",
            title="Paper A",
            year="2001",
            item_key="KA",
            creators=[["Lee", "Ann"]],
        ),
        "AAAAAAA2": entry(doi="10.1/b", title="Paper B", year="2002", item_key="KB"),
        "AAAAAAA3": entry(doi="10.1/c", title="Paper C", year="2003", item_key="KC"),
        "AAAAAAA4": entry(
            item_key="KD",
            title="Numerical Recipes in C",
            year="1992",
            creators=[["Press", "William H."]],
        ),
        "AAAAAAA5": entry(
            item_key="KE",
            title="Collision of galaxies",
            year="2001",
            creators=[["Lee", "Ann"]],
        ),
        "AAAAAAA6": entry(
            doi="10.1/f", title="Quite different", year="2001", item_key="KF"
        ),
        "AAAAAAA7": entry(doi="10.1/g", title="Paper G", year="2004", item_key="KG"),
        "AAAAAAA8": entry(doi="10.1/h", title="Paper H", year="2005", item_key="KH"),
    }

    def __init__(self, d):
        self.dir = d
        self.md = os.path.join(d, "md.json")
        dump(self.METADATA, self.md)
        self.out = os.path.join(d, "out")
        self.cache = os.path.join(d, "cache", "openalex_works.json")
        self.store = os.path.join(d, "s2store", "citations.sqlite")
        self.oa_urls, self.s2_requests = [], []
        t = topic("T1", "Geoid", ("1908", "G"), ("19", "E"), ("3", "P"))
        self.oa = {
            "10.1/a": work(
                "10.1/a",
                "W1",
                refs=["W2", "W101", "W102", "W103"],
                authors=[("A1", "Ann Lee")],
                primary=t,
                topics=[t],
            ),
            "10.1/b": work("10.1/b", "W2", refs=["W101", "W102"]),
            "10.1/dd": work(
                "10.1/dd",
                "W4",
                refs=["W1"],
                authors=[("A9", "William Press")],
                source=("S3", "NR Press"),
            ),
        }
        self.ext = {
            f"W{n}": {
                "id": f"https://openalex.org/W{n}",
                "doi": f"https://doi.org/10.9/x{n}",
                "title": f"External {n}",
                "publication_year": 1990,
                "type": "article",
                "cited_by_count": n,
                "authorships": [
                    {
                        "author": {
                            "id": "https://openalex.org/A7",
                            "display_name": "Zed Zee",
                        },
                        "institutions": [{"id": "I1"}],
                    }
                ],
                "primary_location": {"source": {"display_name": "Jour X"}},
            }
            for n in (101, 102, 103)
        }
        self.ext_single = {}
        self.s2_by_doi = {
            "10.1/a": s2rec_(10, "10.1/A"),
            "10.1/b": s2rec_(20, "10.1/B", tldr=False),
            "10.1/c": s2rec_(30, "10.1/c"),
            "10.1/g": s2rec_(60, "10.1/g", tldr=False),
            "10.1/h": s2rec_(60, "10.1/h", tldr=False),
        }
        self.s2_by_query = {
            "Numerical Recipes in C": s2rec_(
                40,
                "10.1/DD",
                title="Numerical Recipes in C",
                year=1992,
                authors=[{"authorId": "1", "name": "W. Press"}],
                matchScore=103.5,
            ),
            "Collision of galaxies": s2rec_(
                50,
                "10.1/A",
                title="Collision of galaxies",
                year=2001,
                authors=[{"authorId": "2", "name": "Ann Lee"}],
            ),
            "Quite different": s2rec_(
                70,
                None,
                title="Something else",
                year=2001,
                authors=[{"authorId": "3", "name": "Ann Lee"}],
            ),
        }
        self.no_embedding = {s2rec_(30)["paperId"]}  # S2 has no vector for C

    def oa_fake(self, url, headers):
        self.oa_urls.append((url, dict(headers)))
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/works") and "filter" in q:
            kind, _, vals = q["filter"][0].partition(":")
            wanted = vals.split("|")
            if kind == "doi":
                return {"results": [self.oa[d] for d in wanted if d in self.oa]}
            return {"results": [self.ext[w] for w in wanted if w in self.ext]}
        last = u.path.rsplit("/", 1)[-1]
        if last.startswith(
            "W"
        ):  # one outside work by id: ext_single holds what the list filter does not return
            if last not in self.ext_single:
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            return self.ext_single[last]
        doi = urllib.parse.unquote(u.path.rsplit("/doi:", 1)[1])
        if doi in self.oa:
            return self.oa[doi]
        raise urllib.error.HTTPError(url, 404, "nf", {}, None)

    def s2_fake(self, method, url, headers, body):
        self.s2_requests.append((method, url, dict(headers), body))
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/paper/search/match"):
            rec = self.s2_by_query.get(q["query"][0])
            if rec is None:
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            return {"data": [rec]}
        ids = json.loads(body)["ids"]
        if "embedding" in q["fields"][0]:
            return [
                None
                if i not in {r["paperId"] for r in self.recs()}
                else {
                    "paperId": i,
                    "embedding": None
                    if i in self.no_embedding
                    else {"model": "specter_v2", "vector": [0.5] * 768},
                }
                for i in ids
            ]
        return [self.s2_by_doi.get(i[4:]) for i in ids]

    def recs(self):
        return list(self.s2_by_doi.values()) + list(self.s2_by_query.values())

    def make_store(self):
        c = json.dumps
        rows = [
            (
                1,
                10,
                20,
                1,
                c(['c "1", with comma', "second\nline {x}"]),
                c([["background"], ["methodology", "background"]]),
            ),
            (
                6,
                10,
                20,
                0,
                c(["x" * 2500, 'c "1", with comma']),
                c([["result"], ["background"]]),
            ),
            (2, 30, 10, 0, None, None),
            (3, 10, 10, 1, c(["self"]), c([["background"]])),
            (4, 999, 10, 1, c(["outside"]), c([["background"]])),
            (7, 60, 10, 0, c(["ambiguous"]), c([["background"]])),
            (8, 10, None, 0, None, None),
        ]
        con = sync.open_store(self.store)
        sync.commit_batch(con, rows, meta={"release_id": "2026-11-17", "mode": "full"})
        con.close()

    def run(self, *extra, env=None, store=True):
        argv = [
            "--metadata",
            self.md,
            "--out",
            self.out,
            "--cache",
            self.cache,
            "--reference-check",
            self.refcheck(),
        ]
        if store:
            argv += ["--s2-store", self.store]
        argv += list(extra)
        env = {"OPENALEX_API_KEY": OAKEY, "S2_API_KEY": S2KEY, **(env or {})}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(bsl, "_urlopen_json", self.oa_fake),
            mock.patch.object(s2_api, "_urlopen_json", self.s2_fake),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
            mock.patch("sys.stderr", io.StringIO()) as err,
        ):
            rc = quiet_main(argv)
        self.err = err.getvalue()
        return rc

    def refcheck(self):
        path = os.path.join(self.dir, "ref.json")
        dump(
            {
                "10.1/a": {"n_refs": 4, "cited": ["10.1/b"]},
                "10.1/b": {"n_refs": 2, "cited": []},
            },
            path,
        )
        return path


class EndToEndV2Test(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.w = World(d.name)
        self.w.make_store()

    def test_full_run(self):
        w = self.w
        self.assertEqual(w.run(), 0, w.err)
        out = w.out
        papers = {r["paper_id"]: r for r in read_csv(os.path.join(out, "papers.csv"))}
        self.assertEqual(
            sorted(papers),
            [
                "doi:10.1/a",
                "doi:10.1/b",
                "doi:10.1/c",
                "doi:10.1/f",
                "doi:10.1/g",
                "doi:10.1/h",
                "zot:KD",
                "zot:KE",
            ],
        )
        a, d = papers["doi:10.1/a"], papers["zot:KD"]
        self.assertEqual(
            (
                a["s2_paper_id"],
                a["s2_corpus_id"],
                a["s2_citation_count"],
                a["s2_influential_citation_count"],
                a["s2_reference_count"],
                a["tldr"],
                a["doi_source"],
            ),
            (f"{10:040x}", "10", "11", "2", "7", "TLDR 10", "zotero"),
        )
        self.assertEqual(
            (
                d["doi"],
                d["doi_source"],
                d["s2_corpus_id"],
                d["openalex_id"],
                d["venue_id"],
            ),
            ("10.1/dd", "s2_title", "40", "W4", "oa:S3"),
        )
        self.assertEqual(papers["doi:10.1/b"]["tldr"], "")
        e = papers[
            "zot:KE"
        ]  # the match found 10.1/a, which belongs to another paper: not assigned, not attached
        self.assertEqual((e["doi"], e["doi_source"], e["s2_paper_id"]), ("", "", ""))
        self.assertEqual(papers["doi:10.1/f"]["s2_paper_id"], "")  # rejected on title
        cites = {
            (r["citing_paper_id"], r["cited_paper_id"]): r
            for r in read_csv(os.path.join(out, "cites.csv"))
        }
        self.assertEqual(
            sorted(cites),
            [
                ("doi:10.1/a", "doi:10.1/b"),
                ("doi:10.1/c", "doi:10.1/a"),
                ("zot:KD", "doi:10.1/a"),
            ],
        )
        ab = cites[("doi:10.1/a", "doi:10.1/b")]
        self.assertEqual(
            (ab["source"], ab["is_influential"], ab["n_contexts"]),
            ("both", "true", "3"),
        )
        self.assertEqual(
            bsl.parse_pg_array(ab["intents"]), ["background", "methodology", "result"]
        )
        ca = cites[("doi:10.1/c", "doi:10.1/a")]
        self.assertEqual(
            (ca["source"], ca["is_influential"], ca["intents"], ca["n_contexts"]),
            ("s2", "false", "{}", "0"),
        )
        da = cites[("zot:KD", "doi:10.1/a")]
        self.assertEqual(
            (da["source"], da["is_influential"], da["intents"], da["n_contexts"]),
            ("openalex", "", "", ""),
        )
        ctx = read_csv(os.path.join(out, "cite_contexts.csv"))
        self.assertEqual(
            [(r["ordinal"], r["context"][:12]) for r in ctx],
            [("1", 'c "1", with '), ("2", "second line "), ("3", "x" * 12)],
        )
        self.assertEqual(len(ctx[2]["context"]), 2000)
        ext = {
            r["work_id"]: r for r in read_csv(os.path.join(out, "external_works.csv"))
        }
        self.assertEqual(
            sorted(ext), ["oa:W101", "oa:W102"]
        )  # W103 is cited by one library paper only
        self.assertEqual(ext["oa:W101"]["n_library_citers"], "2")
        self.assertEqual(
            (
                ext["oa:W101"]["doi"],
                ext["oa:W101"]["first_author"],
                ext["oa:W101"]["venue"],
            ),
            ("10.9/x101", "Zed Zee", "Jour X"),
        )
        emb = read_csv(os.path.join(out, "paper_embeddings.csv"))
        self.assertEqual(
            sorted(r["paper_id"] for r in emb),
            ["doi:10.1/a", "doi:10.1/b", "doi:10.1/g", "doi:10.1/h", "zot:KD"],
        )
        self.assertTrue(
            emb[0]["embedding"].startswith("[0.5,0.5,")
            and len(emb[0]["embedding"][1:-1].split(",")) == 768
        )
        self.assertEqual(emb[0]["model"], "specter_v2")
        self.assertEqual(
            read(os.path.join(out, "library_corpus_ids.txt")), "10\n20\n30\n40\n60\n"
        )
        qa = json.loads(read(os.path.join(out, "qa.json")))
        self.assertTrue(qa["hard_checks"]["passed"], qa["hard_checks"])
        v = qa["v2"]
        self.assertEqual(
            v["s2_coverage"],
            {
                "papers": 8,
                "found_by_doi": 5,
                "found_by_title": 1,
                "with_tldr": 3,
                "with_embedding": 5,
            },
        )  # G and H share one S2 paper
        tm = v["title_match"]
        self.assertEqual(
            (tm["candidates"], tm["dois_assigned"], tm["doi_conflicts"]), (3, 1, 1)
        )
        self.assertEqual(tm["decisions"], {"accepted": 2, "reject:title": 1})
        self.assertEqual(len(tm["accepted_examples"]), 1)
        self.assertEqual(
            tm["accepted_examples"][0]["s2_title"], "Numerical Recipes in C"
        )
        self.assertEqual(v["cites"]["by_source"], {"openalex": 1, "s2": 1, "both": 1})
        self.assertEqual(
            (
                v["cites"]["with_context"],
                v["cites"]["total_contexts"],
                v["cites"]["cut_contexts"],
            ),
            (1, 3, 1),
        )
        self.assertEqual(v["ambiguous_corpus_ids"]["count"], 1)
        self.assertEqual(v["external_works"]["count"], 2)
        self.assertEqual(v["s2_store"]["rows"], 7)
        self.assertEqual(v["s2_store"]["release_id"], "2026-11-17")
        self.assertEqual(v["s2_store"]["library_ids_missing"], 5)
        for leg in (
            "openalex_works",
            "openalex_external",
            "s2_papers",
            "s2_embeddings",
            "s2_match",
        ):
            self.assertEqual(
                (v["legs"][leg]["ran"], v["legs"][leg]["complete"]), (True, True), leg
            )
        self.assertEqual(
            v["key_fallback"],
            {
                "openalex_key_set": True,
                "openalex_key_rejected": False,
                "s2_key_set": True,
                "s2_key_rejected": False,
            },
        )
        self.assertEqual(
            qa["oracle"]["jaccard"], 1.0
        )  # the oracle sees the OpenAlex edges only
        md = read(os.path.join(out, "QA.md"))
        for needle in (
            "Semantic Scholar coverage",
            "Title match",
            "Works outside the library",
            "S2 citation store",
            "Legs",
            "Keys",
        ):
            self.assertIn(needle, md)

    def test_outside_works_that_are_library_papers_become_citations(self):
        w = self.w
        # W101 carries the DOI of paper G (OpenAlex returns no record for 10.1/g); W102 is the
        # report KE (no DOI) by its title, year and author. A and B cite both.
        w.ext["W101"]["doi"] = "https://doi.org/10.1/G"
        w.ext["W102"].update(
            title="Collision of Galaxies.",
            publication_year=2001,
            authorships=[
                {"author": {"id": "https://openalex.org/A2", "display_name": "A. Lee"}}
            ],
        )
        self.assertEqual(w.run(), 0, w.err)
        ext = [
            r["work_id"] for r in read_csv(os.path.join(w.out, "external_works.csv"))
        ]
        self.assertEqual(ext, [])
        cites = {
            (r["citing_paper_id"], r["cited_paper_id"])
            for r in read_csv(os.path.join(w.out, "cites.csv"))
        }
        for citing in ("doi:10.1/a", "doi:10.1/b"):
            self.assertIn((citing, "doi:10.1/g"), cites)
            self.assertIn((citing, "zot:KE"), cites)
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        al = qa["v2"]["external_works"]["library_aliases"]
        self.assertEqual(
            (al["by_doi"], al["by_title"], al["library_citations_added"]), (1, 1, 4)
        )
        self.assertEqual(
            qa["v2"]["external_works"]["doi_equals_library_paper"]["count"], 0
        )
        # the reference file lacks A -> G and B -> G: alias edges are not compared
        self.assertEqual(qa["oracle"]["jaccard"], 1.0)
        self.assertIn(
            "library papers under another OpenAlex id",
            read(os.path.join(w.out, "QA.md")),
        )

    def test_external_threshold_in_a_run(self):
        w = self.w
        # W103 is cited by A only: below the threshold of 2 (W101 and W102 are cited by A and B)
        self.assertEqual(w.run(), 0, w.err)
        ext = [
            r["work_id"] for r in read_csv(os.path.join(w.out, "external_works.csv"))
        ]
        self.assertEqual(ext, ["oa:W101", "oa:W102"])
        links = {
            (r["citing_paper_id"], r["work_id"])
            for r in read_csv(os.path.join(w.out, "cites_external.csv"))
        }
        self.assertEqual(
            links,
            {(p, f"oa:W{n}") for p in ("doi:10.1/a", "doi:10.1/b") for n in (101, 102)},
        )

    def test_unknown_outside_ids_are_resolved_and_the_tables_rebuilt(self):
        w = self.w
        # A and B cite W901 (merged into library paper A: B's citation of A comes back, A's own is a self-citation),
        # W902 (deleted), W903 (live, but the list filter misses it) and W904 (merged into W109, outside the
        # library). Only ids at least two library papers cite are outside-work candidates, so only those are asked.
        w.oa["10.1/a"]["referenced_works"] += [
            f"https://openalex.org/W{n}" for n in (901, 902, 903, 904)
        ]
        w.oa["10.1/b"]["referenced_works"] += [
            f"https://openalex.org/W{n}" for n in (901, 902, 903, 904)
        ]
        w.ext_single = {
            "W901": w.oa["10.1/a"],
            "W903": {
                **w.ext["W101"],
                "id": "https://openalex.org/W903",
                "title": "Live 903",
            },
            "W904": {
                **w.ext["W101"],
                "id": "https://openalex.org/W109",
                "title": "Merged 109",
            },
        }
        ref = os.path.join(
            w.dir, "ref_merged.json"
        )  # Zotero's reference lists have B citing A too
        dump(
            {
                "10.1/a": {"n_refs": 4, "cited": ["10.1/b"]},
                "10.1/b": {"n_refs": 2, "cited": ["10.1/a"]},
            },
            ref,
        )
        w.refcheck = lambda: ref
        self.assertEqual(w.run(), 0, w.err)
        cites = {
            (r["citing_paper_id"], r["cited_paper_id"])
            for r in read_csv(os.path.join(w.out, "cites.csv"))
        }
        self.assertIn(("doi:10.1/b", "doi:10.1/a"), cites)
        ext = {
            r["work_id"]: r for r in read_csv(os.path.join(w.out, "external_works.csv"))
        }
        self.assertEqual(sorted(ext), ["oa:W101", "oa:W102", "oa:W109", "oa:W903"])
        self.assertEqual(
            (
                ext["oa:W903"]["title"],
                ext["oa:W109"]["title"],
                ext["oa:W109"]["n_library_citers"],
            ),
            ("Live 903", "Merged 109", "2"),
        )
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        self.assertEqual(
            qa["v2"]["external_works"]["ids_rewritten"],
            {"merged": 2, "deleted": 1, "still_unknown": 0},
        )
        self.assertTrue(qa["v2"]["legs"]["openalex_external"]["complete"])
        self.assertIn(
            "4 single lookups of unknown ids (1 found, 2 merged, 1 deleted, 0 failed)",
            qa["v2"]["legs"]["openalex_external"]["note"],
        )
        first = {
            f: read(os.path.join(w.out, f))
            for f in (
                "cites.csv",
                "coupled.csv",
                "external_works.csv",
                "cites_external.csv",
            )
        }
        n = len(w.oa_urls)
        w.out = os.path.join(w.dir, "out2")
        self.assertEqual(
            w.run("--offline"), 0, w.err
        )  # the map is in the cache: the same tables, no request
        self.assertEqual(len(w.oa_urls), n)
        for f, text in first.items():
            self.assertEqual(read(os.path.join(w.out, f)), text, f)

    def test_keys_are_headers_only_and_not_in_any_file(self):
        w = self.w
        self.assertEqual(w.run(), 0, w.err)
        self.assertTrue(w.oa_urls and w.s2_requests)
        for url, headers in w.oa_urls:
            self.assertNotIn(OAKEY, url)
            self.assertEqual(headers["api_key"], OAKEY)
            self.assertNotIn("x-api-key", headers)
        for method, url, headers, body in w.s2_requests:
            self.assertNotIn(S2KEY, url)
            self.assertNotIn(S2KEY.encode(), body or b"")
            self.assertEqual(headers["x-api-key"], S2KEY)
            self.assertEqual(headers["User-Agent"], "zotero-citegraph/2")
            self.assertEqual(
                urllib.parse.urlparse(url).hostname, "api.semanticscholar.org"
            )
        for root in (w.out, os.path.dirname(w.cache)):
            for dirpath, _, files in os.walk(root):
                for f in files:
                    with open(os.path.join(dirpath, f), "rb") as fh:
                        data = fh.read()
                    self.assertNotIn(OAKEY.encode(), data, f)
                    self.assertNotIn(S2KEY.encode(), data, f)
        # the OpenAlex requests keep v1's mailto exactly (none was given: none is sent), the S2 ones send no personal detail
        for url, _ in w.oa_urls:
            self.assertNotIn("mailto", url)

    def test_a_key_in_a_written_file_fails_the_hard_check_without_printing_it(self):
        w = self.w
        os.makedirs(os.path.dirname(w.cache))
        dump(
            {"works": {}, "unknown": [], "note": f"oops {OAKEY}"},
            os.path.join(os.path.dirname(w.cache), "extra_notes.json"),
        )
        with mock.patch("sys.stdout", io.StringIO()):
            rc = w.run()
        self.assertEqual(rc, 1)
        self.assertIn("contains the value of OPENALEX_API_KEY", w.err)
        self.assertNotIn(OAKEY, w.err)

    def test_offline_rerun_from_the_caches_is_identical(self):
        w = self.w
        self.assertEqual(w.run(), 0, w.err)
        first = {
            f: read(os.path.join(w.out, f))
            for f in os.listdir(w.out)
            if f.endswith((".csv", ".sql", ".txt"))
        }
        n_oa, n_s2 = len(w.oa_urls), len(w.s2_requests)
        w.out = os.path.join(w.dir, "out2")
        self.assertEqual(w.run("--offline"), 0, w.err)
        self.assertEqual(
            (len(w.oa_urls), len(w.s2_requests)), (n_oa, n_s2)
        )  # no request in offline mode
        for f, text in first.items():
            self.assertEqual(read(os.path.join(w.out, f)), text, f)
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        self.assertFalse(qa["v2"]["legs"]["s2_papers"]["ran"])
        self.assertFalse(qa["v2"]["legs"]["s2_papers"]["complete"])

    def test_a_rerun_with_caches_asks_for_nothing_new(self):
        w = self.w
        self.assertEqual(w.run(), 0, w.err)
        n_oa, n_s2 = len(w.oa_urls), len(w.s2_requests)
        w.out = os.path.join(w.dir, "out2")
        self.assertEqual(w.run(), 0, w.err)
        # the library DOIs OpenAlex does not know are asked again (v1); everything else is cached
        new_oa = [u for u, _ in w.oa_urls[n_oa:]]
        self.assertEqual(len(new_oa), 1)
        self.assertIn("filter=doi", new_oa[0])
        new_s2 = [(m, u) for m, u, _, _ in w.s2_requests[n_s2:]]
        self.assertEqual([x for x in new_s2 if "match" in x[1]], [])
        # a paper S2 has no vector for is asked again (nothing is cached for it); every other one is cached
        emb = [
            json.loads(b)["ids"]
            for m, u, _, b in w.s2_requests[n_s2:]
            if "embedding" in u
        ]
        self.assertEqual(emb, [sorted(w.no_embedding)])

    def test_budget_leaves_later_legs_incomplete_and_exits_zero(self):
        w = self.w
        self.assertEqual(w.run("--s2-max-calls", "1"), 0, w.err)
        self.assertEqual(len(w.s2_requests), 1)
        legs = json.loads(read(os.path.join(w.out, "qa.json")))["v2"]["legs"]
        self.assertTrue(legs["s2_papers"]["complete"])
        self.assertFalse(legs["s2_match"]["complete"])
        self.assertFalse(legs["s2_embeddings"]["complete"])
        self.assertIn("budget", legs["s2_match"]["note"])

    def test_legs_can_be_chosen(self):
        w = self.w
        self.assertEqual(w.run("--s2-legs", "papers"), 0, w.err)
        self.assertEqual({m for m, _, _, _ in w.s2_requests}, {"POST"})
        legs = json.loads(read(os.path.join(w.out, "qa.json")))["v2"]["legs"]
        self.assertEqual(
            (legs["s2_match"]["ran"], legs["s2_embeddings"]["ran"]), (False, False)
        )
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            bsl.main(["--metadata", w.md, "--out", w.out, "--s2-legs", "papers,bogus"])

    def test_without_a_store_qa_says_so_and_cites_carry_no_s2_data(self):
        w = self.w
        self.assertEqual(w.run(store=False), 0, w.err)
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        self.assertFalse(qa["v2"]["s2_store"]["used"])
        self.assertIn("No --s2-store", qa["v2"]["s2_store"]["note"])
        cites = read_csv(os.path.join(w.out, "cites.csv"))
        self.assertEqual({r["source"] for r in cites}, {"openalex"})
        self.assertEqual(
            read(os.path.join(w.out, "cite_contexts.csv")).strip().split(",")[0],
            "citing_paper_id",
        )
        self.assertIn("No --s2-store", read(os.path.join(w.out, "QA.md")))

    def test_openalex_key_rejected_is_flagged_and_the_run_continues(self):
        w = self.w
        real = w.oa_fake

        def rejecting(url, headers):
            if "api_key" in headers:
                w.oa_urls.append((url, dict(headers)))
                raise urllib.error.HTTPError(url, 401, "no", {}, None)
            return real(url, headers)

        w.oa_fake = rejecting
        bsl._STATE["oa_key_rejected"] = False
        self.assertEqual(w.run(), 0, w.err)
        self.assertIn("continuing WITHOUT the key", w.err)
        self.assertNotIn(OAKEY, w.err)
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        self.assertTrue(qa["v2"]["key_fallback"]["openalex_key_rejected"])

    def test_s2_key_rejected_is_flagged_and_the_run_continues(self):
        w = self.w
        real = w.s2_fake

        def rejecting(method, url, headers, body):
            if "x-api-key" in headers:
                raise urllib.error.HTTPError(url, 403, "no", {}, None)
            return real(method, url, headers, body)

        w.s2_fake = rejecting
        self.assertEqual(w.run(), 0, w.err)
        self.assertIn("WITHOUT the key", w.err)
        self.assertNotIn(S2KEY, w.err)
        qa = json.loads(read(os.path.join(w.out, "qa.json")))
        self.assertTrue(qa["v2"]["key_fallback"]["s2_key_rejected"])
        self.assertTrue(qa["v2"]["legs"]["s2_papers"]["complete"])

    def test_failed_s2_requests_exit_one(self):
        w = self.w

        def dead(method, url, headers, body):
            raise urllib.error.HTTPError(url, 400, "bad", {}, None)

        w.s2_fake = dead
        self.assertEqual(w.run(), 1)
        self.assertFalse(
            json.loads(read(os.path.join(w.out, "qa.json")))["v2"]["legs"]["s2_papers"][
                "complete"
            ]
        )


class FixRound1Test(unittest.TestCase):
    def test_conflict_check_runs_for_every_accepted_match(self):
        # paper A has a mistyped DOI (unknown to both services); S2's best title match is library paper B
        md = {
            "AAAAAAA1": entry(
                doi="10.1/typo",
                title="Gravity field models",
                year="2001",
                item_key="KA",
                creators=[["Lee", "Ann"]],
            ),
            "AAAAAAA2": entry(
                doi="10.1/real",
                title="Gravity field models",
                year="2001",
                item_key="KB",
            ),
        }
        groups = bsl.group_papers(md)
        cands = bsl.title_candidates(
            groups,
            {},
            {"DOI:10.1/typo": None, "DOI:10.1/real": s2rec_(42, "10.1/real")},
        )
        self.assertEqual([c["paper_id"] for c in cands], ["doi:10.1/typo"])
        rec = s2rec_(
            42,
            "10.1/real",
            title="Gravity field models",
            year=2001,
            authors=[{"name": "Ann Lee"}],
        )
        matches = {
            s2_api.match_key(
                cands[0]["paper_id"], cands[0]["title"], cands[0]["year"]
            ): {"record": rec}
        }
        res = bsl.resolve_matches(cands, matches, ["10.1/typo", "10.1/real"], {42})
        self.assertEqual(
            res["records"], {}
        )  # nothing attached to A, so corpus id 42 stays B's
        self.assertEqual(
            [(c["paper_id"], c["reason"]) for c in res["conflicts"]],
            [("doi:10.1/typo", "doi")],
        )
        # B keeps its S2 data: the corpus id is unique
        records = {"doi:10.1/real": s2rec_(42, "10.1/real")}
        unique, ambiguous, _ = bsl.corpus_map(records)
        self.assertEqual((unique, ambiguous), ({42: "doi:10.1/real"}, {}))
        # the same record without a DOI is caught by its corpus id
        rec2 = s2rec_(
            42,
            None,
            title="Gravity field models",
            year=2001,
            authors=[{"name": "Ann Lee"}],
        )
        matches = {
            s2_api.match_key(
                cands[0]["paper_id"], cands[0]["title"], cands[0]["year"]
            ): {"record": rec2}
        }
        res = bsl.resolve_matches(cands, matches, ["10.1/typo", "10.1/real"], {42})
        self.assertEqual(
            (res["records"], [c["reason"] for c in res["conflicts"]]),
            ({}, ["corpus_id"]),
        )
        # a match that finds the paper's OWN DOI is no conflict; with no clash it attaches
        rec3 = s2rec_(
            77,
            "10.1/TYPO",
            title="Gravity field models",
            year=2001,
            authors=[{"name": "Ann Lee"}],
        )
        matches = {
            s2_api.match_key(
                cands[0]["paper_id"], cands[0]["title"], cands[0]["year"]
            ): {"record": rec3}
        }
        res = bsl.resolve_matches(cands, matches, ["10.1/typo", "10.1/real"], {42})
        self.assertEqual(list(res["records"]), ["doi:10.1/typo"])
        self.assertEqual((res["overrides"], res["conflicts"]), ({}, []))

    def test_nul_is_stripped_from_every_csv_text_and_checked(self):
        self.assertEqual(bsl._fmt("a\x00b\ud835c"), "ab�c")
        self.assertEqual(bsl._fmt(["x"]), "{x}")
        with tempfile.TemporaryDirectory() as d:
            bsl.write_csv(os.path.join(d, "t.csv"), ["a"], [["ti\x00tle"], ["ok"]])
            self.assertEqual(
                [r["a"] for r in read_csv(os.path.join(d, "t.csv"))], ["title", "ok"]
            )
            self.assertEqual(bsl.scan_for_nul(d), [])
            with open(os.path.join(d, "bad.csv"), "wb") as f:
                f.write(b"a\nx\x00y\n")
            self.assertEqual(bsl.scan_for_nul(d), ["bad.csv: contains a NUL character"])

    def test_json_writer_leaves_no_tmp_after_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with self.assertRaises(UnicodeEncodeError):
                bsl.write_json(path, {"t": "lone \ud835 surrogate"})
            self.assertEqual(os.listdir(d), [])
            bsl.write_json(path, {"t": s2_api.clean_text("lone \ud835 surrogate")})
            self.assertEqual(
                json.load(open(path, encoding="utf-8")), {"t": "lone � surrogate"}
            )

    def test_openalex_get_json_retries_dropped_connections_and_bad_bodies(self):
        import http.client

        seen = []

        def flaky(url, headers):
            seen.append(1)
            if len(seen) == 1:
                raise http.client.IncompleteRead(b"x")
            if len(seen) == 2:
                raise json.JSONDecodeError("bad", "<html>", 0)
            if len(seen) == 3:
                raise http.client.RemoteDisconnected("closed")
            return {"results": [{"title": "t\ud835"}]}

        with (
            mock.patch.object(bsl, "_urlopen_json", flaky),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
        ):
            self.assertEqual(bsl.get_json(bsl.API, "k"), {"results": [{"title": "t�"}]})
        self.assertEqual(len(seen), 4)

        def dead(url, headers):
            raise http.client.IncompleteRead(b"")

        store = {"works": {}, "unknown": []}
        with (
            mock.patch.object(bsl, "_urlopen_json", dead),
            mock.patch.object(bsl.time, "sleep", lambda s: None),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            stats = bsl.harvest(["10.1/a"], store, "", "")
        self.assertEqual(stats["failed"], 1)  # counted, not a crash

    def test_openalex_key_header_is_unredirected(self):
        import urllib.request as ur

        captured = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

        def fake_urlopen(req, timeout=None):
            captured.append(req)
            return Resp(b'{"results": []}')

        with mock.patch.object(ur, "urlopen", fake_urlopen):
            bsl._urlopen_json(
                bsl.API + "?x=1", {"Accept": "application/json", "api_key": OAKEY}
            )
        req = captured[0]
        self.assertEqual(req.get_header("Api_key"), OAKEY)
        new = ur.HTTPRedirectHandler().redirect_request(
            req, io.BytesIO(b""), 301, "Moved", {}, "https://other.example/x"
        )
        self.assertNotIn(
            "api_key",
            {k.lower() for k in list(new.headers) + list(new.unredirected_hdrs)},
        )

    def test_external_works_with_a_library_doi_are_flagged_not_dropped(self):
        ext = {
            "W101": {
                "id": "x",
                "doi": "https://doi.org/10.1/A",
                "title": "t",
                "authorships": [],
            }
        }
        tables = {}
        qa = bsl.build_external(
            tables,
            {"W101": ["pa", "pb"], "W102": ["pa", "pb"]},
            ext,
            {"10.1/a", "10.1/b"},
        )
        self.assertEqual(
            qa["doi_equals_library_paper"],
            {"count": 1, "examples": [["oa:W101", "10.1/a"]]},
        )
        self.assertEqual(len(tables["external_works"]), 2)


class FixRound1EndToEndTest(unittest.TestCase):
    def test_nul_and_surrogate_in_api_text_do_not_break_a_run(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.make_store()
            w.s2_by_doi["10.1/a"] = s2rec_(10, "10.1/A", tldr=True)
            w.s2_by_doi["10.1/a"]["tldr"]["text"] = "tl\x00dr \ud835 end"
            w.ext["W101"]["title"] = "Ext\x00ernal \ud835"
            self.assertEqual(w.run(), 0, w.err)
            papers = {
                r["paper_id"]: r for r in read_csv(os.path.join(w.out, "papers.csv"))
            }
            self.assertEqual(papers["doi:10.1/a"]["tldr"], "tldr � end")
            ext = {
                r["work_id"]: r
                for r in read_csv(os.path.join(w.out, "external_works.csv"))
            }
            self.assertEqual(ext["oa:W101"]["title"], "External �")
            for fn in os.listdir(os.path.dirname(w.cache)):
                self.assertFalse(fn.endswith(".tmp"), fn)
            hc = json.loads(read(os.path.join(w.out, "qa.json")))["hard_checks"]
            self.assertTrue(hc["passed"])
            self.assertEqual(hc["late_key_scan"], {"passed": True, "violations": []})
            qa = json.loads(read(os.path.join(w.out, "qa.json")))
            self.assertEqual(
                qa["v2"]["external_works"]["doi_equals_library_paper"]["count"], 0
            )

    def test_a_late_key_hit_is_recorded_in_qa(self):
        with tempfile.TemporaryDirectory() as d:
            w = World(d)
            w.make_store()
            w.ext["W101"]["title"] = "x"
            # an outside-work title that carries the key lands in the cache, the CSV and the QA: all are caught
            w.ext["W102"]["title"] = f"leak {OAKEY}"
            with mock.patch("sys.stdout", io.StringIO()):
                rc = w.run()
            self.assertEqual(rc, 1)
            qa = json.loads(read(os.path.join(w.out, "qa.json")))
            self.assertFalse(qa["hard_checks"]["passed"])
            self.assertTrue(
                any("OPENALEX_API_KEY" in v for v in qa["hard_checks"]["violations"])
            )
            self.assertTrue(
                qa["hard_checks"]["late_key_scan"]["violations"]
            )  # the QA echoes the leaked title


if __name__ == "__main__":
    unittest.main()
