"""The library's citation structure (lightrag/zotero_struct.py): related papers,
papers by topic, what connects two papers, and the paper map.

The schema is read through a stub of the PostgreSQL pool that answers each
tagged query from the tables below, and the document-status rows hold papers
of another workspace and unprocessed documents that must not count. Fixtures
are invented.
"""

import asyncio
import importlib
import json
import sys
import threading
import zlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lightrag.zotero_citations as zc
import lightrag.zotero_struct as zs

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

WORKSPACE = "lib"
# ids that sort before every other paper, so no tie-break favours the others
FANS = [f"doi:10.1/0fan{i}" for i in range(60)]


def _paper(pid, title, year, cited_by=10, n_refs=20):
    return {
        "paper_id": pid,
        "doi": pid[4:] if pid.startswith("doi:") else None,
        "title": title,
        "year": year,
        "cited_by_count": cited_by,
        "n_refs_openalex": n_refs,
        "venue": "Journal of Invented Results",
    }


TABLES = {
    "papers": [
        _paper("doi:10.1/a", "Alpha gravity leakage", 2010),
        _paper("doi:10.1/b", "Bravo mascons", 2012),
        _paper("doi:10.1/c", "Charlie filters", 2014),
        _paper("doi:10.1/e", "Echo basins", 2018),
        _paper("doi:10.1/h", "Hub mission overview", 2000, cited_by=5000),
        _paper("doi:10.1/d", "Delta coupling only", 2016),
        _paper("doi:10.1/x", "Xray island", 2020),
        _paper("zot:ITEM0001", "A report without a DOI", None),
    ]
    + [_paper(f, f"Fan paper {i}", 2005 + i % 10) for i, f in enumerate(FANS)],
    "attachments": [
        {"attachment_key": "AAAA0001", "paper_id": "doi:10.1/a"},
        {"attachment_key": "AAAA0002", "paper_id": "doi:10.1/a"},
        {"attachment_key": "BBBB0001", "paper_id": "doi:10.1/b"},
        {"attachment_key": "CCCC0001", "paper_id": "doi:10.1/c"},
        {"attachment_key": "EEEE0001", "paper_id": "doi:10.1/e"},
        {"attachment_key": "HHHH0001", "paper_id": "doi:10.1/h"},
        {"attachment_key": "DDDD0001", "paper_id": "doi:10.1/d"},
        {"attachment_key": "XXXX0001", "paper_id": "doi:10.1/x"},
        {"attachment_key": "KKKK0001", "paper_id": "zot:ITEM0001"},
    ],
    "authors": [
        {"paper_id": "doi:10.1/a", "author_id": "oa:1", "display_name": "Ada Okafor"},
        {"paper_id": "doi:10.1/a", "author_id": "oa:2", "display_name": "Ben Lind"},
        {"paper_id": "doi:10.1/b", "author_id": "oa:2", "display_name": "Ben Lind"},
        {"paper_id": "doi:10.1/x", "author_id": "oa:1", "display_name": "Ada Okafor"},
        {"paper_id": "doi:10.1/c", "author_id": "oa:3", "display_name": "Cy Varga"},
        {"paper_id": "doi:10.1/e", "author_id": "oa:4", "display_name": "Eve Okafor"},
    ],
    "topics": [
        {
            "topic_id": "domain:3",
            "display_name": "Physical Sciences",
            "level": "domain",
            "parent_id": None,
        },
        {
            "topic_id": "field:19",
            "display_name": "Earth and Planetary Sciences",
            "level": "field",
            "parent_id": "domain:3",
        },
        {
            "topic_id": "field:22",
            "display_name": "Engineering",
            "level": "field",
            "parent_id": "domain:3",
        },
        {
            "topic_id": "subfield:1908",
            "display_name": "Geophysics",
            "level": "subfield",
            "parent_id": "field:19",
        },
        {
            "topic_id": "subfield:2202",
            "display_name": "Aerospace Engineering",
            "level": "subfield",
            "parent_id": "field:22",
        },
        {
            "topic_id": "topic:T1",
            "display_name": "Gravity Measurements",
            "level": "topic",
            "parent_id": "subfield:1908",
        },
        {
            "topic_id": "topic:T2",
            "display_name": "Satellite Gravity Missions",
            "level": "topic",
            "parent_id": "subfield:2202",
        },
    ],
    "paper_topics": [
        {"paper_id": "doi:10.1/a", "topic_id": "topic:T1", "is_primary": True},
        {"paper_id": "doi:10.1/a", "topic_id": "topic:T2", "is_primary": False},
        {"paper_id": "doi:10.1/b", "topic_id": "topic:T1", "is_primary": True},
        {"paper_id": "doi:10.1/c", "topic_id": "topic:T1", "is_primary": True},
        {"paper_id": "doi:10.1/e", "topic_id": "topic:T1", "is_primary": True},
        {"paper_id": "doi:10.1/h", "topic_id": "topic:T2", "is_primary": True},
        {"paper_id": "doi:10.1/d", "topic_id": "topic:T2", "is_primary": True},
    ],
    # a -> b -> c -> e is the low-degree route from a to e; a -> h <- e the hub route
    "cites": [
        {"citing_paper_id": "doi:10.1/a", "cited_paper_id": "doi:10.1/b"},
        {"citing_paper_id": "doi:10.1/c", "cited_paper_id": "doi:10.1/b"},
        {"citing_paper_id": "doi:10.1/e", "cited_paper_id": "doi:10.1/c"},
        {"citing_paper_id": "doi:10.1/a", "cited_paper_id": "doi:10.1/h"},
        {"citing_paper_id": "doi:10.1/e", "cited_paper_id": "doi:10.1/h"},
        {"citing_paper_id": "doi:10.1/c", "cited_paper_id": "doi:10.1/a"},
    ]
    + [{"citing_paper_id": f, "cited_paper_id": "doi:10.1/h"} for f in FANS],
    "coupled": [
        {
            "paper_a": "doi:10.1/a",
            "paper_b": "doi:10.1/d",
            "shared_refs": 7,
            "score": 0.4,
        },
        {
            "paper_a": "doi:10.1/a",
            "paper_b": "doi:10.1/c",
            "shared_refs": 3,
            "score": 0.1,
        },
    ],
}

# The document-status table: (workspace, file_path, status)
DOC_STATUS = [
    (WORKSPACE, "AAAA0002__alpha.md", "processed"),
    (WORKSPACE, "BBBB0001__bravo.md", "processed"),
    (WORKSPACE, "CCCC0001__charlie.md", "failed"),
    ("other", "CCCC0001__charlie.md", "processed"),
    ("other", "DDDD0001__delta.md", "processed"),
    (WORKSPACE, "HHHH0001__hub.md", "processed"),
    (WORKSPACE, "notes.txt", "processed"),
]

META = {
    "AAAA0001": {"title": "Alpha", "creators": [["Okafor", "Ada"]], "year": "2010"},
    "AAAA0002": {"title": "Alpha", "creators": [["Okafor", "Ada"]], "year": "2010"},
    "BBBB0001": {"title": "Bravo", "creators": [["Lind", "Ben"]], "year": "2012"},
}


def _owner(row):
    """The paper a row of a chunked table belongs to."""
    return row.get("paper_id") or row.get("citing_paper_id") or row["paper_a"]


class _DB:
    """The PostgreSQL pool as zotero_struct sees it."""

    def __init__(self, tables=None, schema="zs"):
        self.tables = tables or TABLES
        self.schema = schema
        self.calls: list[tuple[str, list | None]] = []
        self.filenode = 1
        # table name -> the v2 columns the schema carries
        self.columns: dict[str, list[str]] = {}
        # table name -> the errors its next reads raise, one per read
        self.fail: dict[str, list[Exception]] = {}

    async def query(self, sql, params=None, multirows=False):
        tag = sql.split("\n", 1)[0]
        assert tag.startswith("-- zotero_struct:"), sql
        name = tag.split(":", 1)[1]
        self.calls.append((name, params))
        if name == "fingerprint":
            if params != [self.schema]:
                return []
            return [
                {"relname": t, "oid": i, "filenode": self.filenode}
                for i, t in enumerate(zs._TABLES)
            ]
        if name == "columns":  # a v1 schema has none of the v2 columns
            assert params == [self.schema]
            return [
                {"table_name": t, "column_name": c}
                for t, cs in self.columns.items()
                for c in cs
            ]
        if name == "documents":
            workspace, status = params
            return [
                {"file_path": f}
                for w, f, s in DOC_STATUS
                if w == workspace and s == status
            ]
        assert f'"{self.schema}".' in sql
        if self.fail.get(name):
            raise self.fail[name].pop(0)
        rows = self.tables[name]
        if name in zs._CHUNKED:  # one slice of rows, papers kept together
            chunks, k = params
            rows = [r for r in rows if zlib.crc32(_owner(r).encode()) % chunks == k]
        return [dict(r) for r in rows]

    def names(self):
        return [n for n, _ in self.calls]


def _rag(db=None, workspace=WORKSPACE):
    return SimpleNamespace(
        doc_status=SimpleNamespace(db=db or _DB(), workspace=workspace),
        max_graph_nodes=1000,
    )


@pytest.fixture(autouse=True)
def structure(tmp_path, monkeypatch):
    meta_path = tmp_path / "zotero_metadata.json"
    meta_path.write_text(json.dumps(META), encoding="utf-8")
    saved = zc.METADATA_PATH
    zc.METADATA_PATH = str(meta_path)
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs")
    zs._reset_for_tests()
    yield
    zs._reset_for_tests()
    zc.METADATA_PATH = saved
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False


def _by_id(items, key="paper_id"):
    return {item[key]: item for item in items}


def _graph_tables(papers, cites=(), coupled=(), year=2010):
    """Tables of just these papers (DOI ``10.1/<name>``) and links."""
    tables = {name: [] for name in TABLES}
    tables["papers"] = [_paper(f"doi:10.1/{n}", n, year) for n in papers]
    tables["cites"] = [
        {"citing_paper_id": f"doi:10.1/{a}", "cited_paper_id": f"doi:10.1/{b}"}
        for a, b in cites
    ]
    tables["coupled"] = [
        {
            "paper_a": f"doi:10.1/{a}",
            "paper_b": f"doi:10.1/{b}",
            "shared_refs": 3,
            "score": score,
        }
        for a, b, score in coupled
    ]
    return tables


# --- availability -----------------------------------------------------------


async def test_unset_schema_is_off_and_reads_nothing(monkeypatch):
    monkeypatch.delenv(zs.SCHEMA_ENV)
    db = _DB()
    with pytest.raises(zs.Unavailable) as e:
        await zs.related_papers(_rag(db), "10.1/a")
    assert e.value.reason == "off"
    assert not await zs.available(_rag(db))
    assert db.calls == []


@pytest.mark.parametrize("name", ["zs; drop schema x", 'zs"', "Zs", "1zs", "z s"])
async def test_a_schema_name_that_is_no_identifier_is_refused(monkeypatch, name):
    monkeypatch.setenv(zs.SCHEMA_ENV, name)
    db = _DB()
    with pytest.raises(zs.Unavailable) as e:
        await zs.paper_map(_rag(db), ["10.1/a"])
    assert e.value.reason == "bad_schema_name"
    assert db.calls == []


async def test_a_storage_without_a_postgres_pool_is_not_postgres():
    rag = SimpleNamespace(doc_status=SimpleNamespace(workspace=WORKSPACE))
    with pytest.raises(zs.Unavailable) as e:
        await zs.list_topics(rag)
    assert e.value.reason == "not_postgres"
    assert not await zs.available(rag)


async def test_a_missing_schema_is_reported_and_not_read(monkeypatch):
    monkeypatch.setenv(zs.SCHEMA_ENV, "elsewhere")
    db = _DB()
    with pytest.raises(zs.Unavailable) as e:
        await zs.papers_by_topic(_rag(db), "gravity")
    assert e.value.reason == "no_schema"
    assert db.names() == ["fingerprint"]
    zs._reset_for_tests()
    assert not await zs.available(_rag(db))


async def test_available_checks_the_tables_only():
    db = _DB()
    assert await zs.available(_rag(db))
    assert db.names() == ["fingerprint"]


async def test_available_asks_the_catalog_once_per_interval(monkeypatch):
    db = _DB()
    rag = _rag(db)
    for _ in range(3):
        assert await zs.available(rag)
    assert db.names() == ["fingerprint"]
    # a load counts as a check too
    await zs.related_papers(rag, "10.1/a")
    assert await zs.available(rag)
    assert db.names().count("fingerprint") == 2
    monkeypatch.setattr(zs, "_CHECK_INTERVAL", 0.0)
    assert await zs.available(rag)
    assert db.names().count("fingerprint") == 3


async def test_a_failing_pool_answers_false_and_is_not_asked_every_poll():
    class Down(_DB):
        async def query(self, sql, params=None, multirows=False):
            self.calls.append(("fingerprint", params))
            raise ConnectionError("pool is down")

    db = Down()
    rag = _rag(db)
    assert [await zs.available(rag) for _ in range(3)] == [False, False, False]
    assert len(db.calls) == 1


async def test_a_hanging_pool_cannot_hold_the_health_answer(monkeypatch):
    monkeypatch.setattr(zs, "_AVAILABLE_TIMEOUT", 0.05)

    class Hung(_DB):
        async def query(self, sql, params=None, multirows=False):
            await asyncio.sleep(30)

    assert await asyncio.wait_for(zs.available(_rag(Hung())), 2) is False


# --- loading and refresh ----------------------------------------------------


async def test_the_structure_is_read_once_and_again_after_a_rebuild(monkeypatch):
    db = _DB()
    rag = _rag(db)
    await zs.related_papers(rag, "10.1/a")
    await zs.related_papers(rag, "10.1/b")
    assert db.names().count("papers") == 1
    assert db.names().count("fingerprint") == 1  # within the check interval
    monkeypatch.setattr(zs, "_CHECK_INTERVAL", 0.0)
    await zs.related_papers(rag, "10.1/a")
    assert db.names().count("fingerprint") == 2
    assert db.names().count("papers") == 1  # same tables: kept
    db.filenode = 2  # dropped and loaded again
    await zs.related_papers(rag, "10.1/a")
    assert db.names().count("papers") == 2


async def test_concurrent_first_calls_cause_one_load_and_convert_off_the_loop(
    monkeypatch,
):
    db = _DB()
    rag = _rag(db)
    threads = []
    real = zs._build

    def spy(rows, fingerprint):
        threads.append(threading.get_ident())
        return real(rows, fingerprint)

    monkeypatch.setattr(zs, "_build", spy)
    await asyncio.gather(*(zs.related_papers(rag, "10.1/a") for _ in range(5)))
    assert db.names().count("papers") == 1
    assert threads and threading.get_ident() not in threads
    # the big tables come in slices that cover every row once
    assert db.names().count("cites") == zs._CHUNKS
    st = await zs._structure(rag)
    assert sum(len(c) for c in st.cites_out) == len(TABLES["cites"])


async def test_a_table_that_vanishes_during_the_load_is_unavailable(monkeypatch):
    db = _DB()
    rag = _rag(db)
    db.fail["cites"] = [RuntimeError("relation does not exist")] * 2
    with pytest.raises(zs.Unavailable) as e:
        await zs.related_papers(rag, "10.1/a")
    assert e.value.reason == "no_schema"
    # one retry under a fresh fingerprint, and nothing half-loaded is kept
    assert db.names().count("fingerprint") == 2
    assert zs._state["struct"] is None
    assert not await zs.available(rag)
    # the failure stands for the check interval, without reading again
    calls = len(db.calls)
    with pytest.raises(zs.Unavailable):
        await zs.paper_map(rag, ["10.1/a"])
    assert len(db.calls) == calls
    # and the structure comes back with the tables
    monkeypatch.setattr(zs, "_CHECK_INTERVAL", 0.0)
    assert (await zs.related_papers(rag, "10.1/a"))["paper"]["paper_id"] == "doi:10.1/a"


async def test_one_failed_read_is_retried():
    db = _DB()
    db.fail["coupled"] = [RuntimeError("connection reset")]
    result = await zs.related_papers(_rag(db), "10.1/a")
    assert result["totals"]["coupled"] == 2
    assert db.names().count("fingerprint") == 2


async def test_a_schema_dropped_during_the_load_is_unavailable():
    class Dropped(_DB):
        async def query(self, sql, params=None, multirows=False):
            if "zotero_struct:cites" in sql:
                self.schema = "gone"  # the fingerprint now finds no tables
                raise RuntimeError("relation does not exist")
            return await super().query(sql, params, multirows)

    with pytest.raises(zs.Unavailable) as e:
        await zs.related_papers(_rag(Dropped()), "10.1/a")
    assert e.value.reason == "no_schema"
    assert zs._state["struct"] is None


async def test_only_processed_documents_of_the_servers_workspace_count():
    db = _DB()
    result = await zs.related_papers(_rag(db), "10.1/a")
    assert ("documents", [WORKSPACE, "processed"]) in db.calls
    related = _by_id(result["related"])
    assert related["doi:10.1/b"]["in_kb"] and related["doi:10.1/b"]["key"] == "BBBB0001"
    # processed only in another workspace / failed in this one
    assert not related["doi:10.1/c"]["in_kb"] and related["doi:10.1/c"]["key"] is None
    assert not related["doi:10.1/d"]["in_kb"]
    # the paper itself is in the knowledge base through its second attachment
    assert result["paper"]["key"] == "AAAA0002"
    assert result["paper"]["keys"] == ["AAAA0001", "AAAA0002"]


# --- related papers ---------------------------------------------------------


async def test_related_papers_cover_both_citation_directions_coupling_and_authors():
    result = await zs.related_papers(_rag(), "AAAA0001")
    related = _by_id(result["related"])
    assert related["doi:10.1/b"]["reasons"] == ["cites", "coauthor"]
    assert related["doi:10.1/h"]["reasons"] == ["cites"]
    assert related["doi:10.1/c"]["reasons"] == ["cited_by", "coupled"]
    assert related["doi:10.1/c"]["shared_refs"] == 3
    assert related["doi:10.1/d"]["reasons"] == ["coupled"]
    assert related["doi:10.1/d"]["coupling_score"] == 0.4
    assert related["doi:10.1/x"]["shared_authors"] == ["Ada Okafor"]
    assert result["totals"] == {
        "cites": 2,
        "cited_by": 1,
        "coupled": 2,
        "coauthor": 2,
        "papers": 5,
        "in_kb": 2,
    }


async def test_related_papers_are_ranked_by_score_and_limited():
    result = await zs.related_papers(_rag(), "10.1/a", limit=3)
    ids = [r["paper_id"] for r in result["related"]]
    # b: cite 1 + author 0.25; c: cite 1 + 2 x 0.1; h: cite 1 (most cited by the
    # library breaks the tie with nothing); d: 2 x 0.4; x: one shared author
    assert ids == ["doi:10.1/b", "doi:10.1/c", "doi:10.1/h"]
    assert [r["score"] for r in result["related"]] == [1.25, 1.2, 1.0]
    assert result["truncated"] and result["totals"]["papers"] == 5


async def test_related_papers_keep_to_the_relations_asked():
    result = await zs.related_papers(_rag(), "10.1/c", ["cited_by"])
    assert [r["paper_id"] for r in result["related"]] == ["doi:10.1/e"]
    assert result["totals"] == {"cited_by": 1, "papers": 1, "in_kb": 0}


@pytest.mark.parametrize(
    "asked, expected",
    [
        ("aaaa0002", "doi:10.1/a"),
        ("https://doi.org/10.1/A", "doi:10.1/a"),
        ("doi:10.1/a", "doi:10.1/a"),
        ("zot:ITEM0001", "zot:ITEM0001"),
        ("KKKK0001", "zot:ITEM0001"),
    ],
)
async def test_a_paper_is_named_by_key_doi_or_structure_id(asked, expected):
    result = await zs.related_papers(_rag(), asked)
    assert result["paper"]["paper_id"] == expected


@pytest.mark.parametrize("asked", ["ZZZZ9999", "10.1/nope", "Alpha gravity", ""])
async def test_an_unknown_paper_is_named_not_guessed(asked):
    result = await zs.related_papers(_rag(), asked)
    assert result["reason"] == "not_in_structure" and result["paper"] is None


# --- topics -----------------------------------------------------------------


async def test_topics_are_found_by_name_at_every_level_with_counts():
    result = await zs.list_topics(_rag(), "gravity")
    rows = _by_id(result["topics"], "topic_id")
    assert set(rows) == {"topic:T1", "topic:T2"}
    assert rows["topic:T1"]["path"] == (
        "Physical Sciences > Earth and Planetary Sciences > Geophysics > "
        "Gravity Measurements"
    )
    assert (rows["topic:T1"]["n_papers"], rows["topic:T1"]["n_primary"]) == (4, 4)
    assert (rows["topic:T2"]["n_papers"], rows["topic:T2"]["n_primary"]) == (3, 2)
    fields = await zs.list_topics(_rag(), "", "field")
    assert [t["topic_id"] for t in fields["topics"]] == ["field:19", "field:22"]


async def test_an_upper_level_topic_holds_the_papers_below_it():
    result = await zs.papers_by_topic(_rag(), "field:22")
    ids = [p["paper_id"] for p in result["papers"]]
    # primary under Engineering first (h is cited most), then a, whose
    # second topic is under it
    assert ids == ["doi:10.1/h", "doi:10.1/d", "doi:10.1/a"]
    assert [p["primary"] for p in result["papers"]] == [True, True, False]
    primary = await zs.papers_by_topic(_rag(), "field:22", primary_only=True)
    assert primary["total"] == 2


async def test_a_topic_named_by_words_takes_the_best_match():
    result = await zs.papers_by_topic(_rag(), "geophysics", limit=2)
    assert result["topic"]["topic_id"] == "subfield:1908"
    assert result["total"] == 4 and result["truncated"]
    # b is cited by two library papers, a and c by one
    assert result["papers"][0]["paper_id"] == "doi:10.1/b"
    nothing = await zs.papers_by_topic(_rag(), "basket weaving")
    assert nothing["reason"] == "no_topic"


def _topic(tid, name, parent="subfield:1908"):
    return {
        "topic_id": tid,
        "display_name": name,
        "level": "topic",
        "parent_id": parent,
    }


def _tables_with_topics(*topics):
    """The shared tables plus topics (which hold no papers)."""
    return dict(TABLES, topics=TABLES["topics"] + list(topics))


# a short word is a whole word of the name, not a piece of one
SHORT_WORD_TOPICS = (
    _topic("topic:M1", "Random Matrices and Applications"),
    _topic("topic:I1", "Arctic and Antarctic ice dynamics"),
    _topic("topic:P1", "Pulsars and Gravitational Waves Research"),
    _topic("topic:S1", "Sea Level Change"),
)


async def test_a_short_topic_word_matches_whole_words_only():
    rag = _rag(_DB(_tables_with_topics(*SHORT_WORD_TOPICS)))
    ice = await zs.papers_by_topic(rag, "ice")
    assert ice["topic"]["topic_id"] == "topic:I1" and ice["alternatives"] == []
    sea = await zs.papers_by_topic(rag, "SEA")
    assert sea["topic"]["topic_id"] == "topic:S1" and sea["alternatives"] == []
    listed = await zs.list_topics(rag, "ice")
    assert [t["topic_id"] for t in listed["topics"]] == ["topic:I1"]
    # a word of four letters or more may be the start of a name word
    assert (await zs.list_topics(rag, "geoph"))["total"] == 1
    assert (await zs.list_topics(rag, "arct antarc"))["total"] == 1
    assert (await zs.list_topics(rag, "geo"))["total"] == 0
    # every word must match
    assert (await zs.list_topics(rag, "ice sea"))["total"] == 0
    assert (await zs.list_topics(rag, "ice dynamics"))["total"] == 1


async def test_topics_rank_exact_name_then_whole_words_then_paper_count():
    rag = _rag(
        _DB(
            _tables_with_topics(
                _topic("topic:W1", "Gravity Measurements Overview"),
                _topic("topic:W2", "Gravity"),
                _topic("topic:W3", "Gravitation theory"),
            )
        )
    )
    found = await zs.list_topics(rag, "gravity")
    # the exact name, then the names with the word spelled out by papers
    # (topic:T1 has 4 primary papers, topic:T2 2, topic:W1 none)
    assert [t["topic_id"] for t in found["topics"]] == [
        "topic:W2",
        "topic:T1",
        "topic:T2",
        "topic:W1",
    ]
    # "gravit" is a start of three name words: no whole-word match, then by papers
    started = await zs.list_topics(rag, "gravit")
    assert [t["topic_id"] for t in started["topics"]][0] == "topic:T1"
    assert {t["topic_id"] for t in started["topics"]} >= {"topic:W3"}


@pytest.mark.parametrize("blank", ["", "   ", "\t", " - "])
async def test_a_blank_topic_is_no_topic(blank):
    result = await zs.papers_by_topic(_rag(), blank)
    assert result["reason"] == "no_topic" and result["papers"] == []


def test_a_blank_topic_is_no_topic_on_the_route():
    client, headers = _client(_rag())
    r = client.get("/graph/papers/by_topic?topic=%20%20", headers=headers)
    assert r.status_code == 200 and r.json()["reason"] == "no_topic"


# --- connect ----------------------------------------------------------------


def _relations(path):
    return [(s["from"], s["relation"], s["to"]) for s in path["steps"]]


async def test_a_direct_citation_is_one_hop_labelled_from_the_first_paper():
    result = await zs.connect_papers(_rag(), "10.1/e", "CCCC0001", max_paths=1)
    assert result["links"]["relation"] == ["cites"]
    assert result["paths"][0]["hops"] == 1
    assert _relations(result["paths"][0]) == [("Okafor 2018", ["cites"], "Varga 2014")]


async def test_paths_avoid_hub_papers_and_label_every_step():
    result = await zs.connect_papers(_rag(), "10.1/a", "10.1/e")
    first, second = result["paths"][:2]
    # both run through one paper; the hub (62 links) costs more than c (3)
    assert [p["paper_id"] for p in first["papers"]] == ["doi:10.1/c"]
    assert first["cost"] == 3.0
    assert _relations(first) == [
        ("Okafor 2010", ["cited_by", "coupled"], "Varga 2014"),
        ("Varga 2014", ["cited_by"], "Okafor 2018"),
    ]
    assert [p["paper_id"] for p in second["papers"]] == ["doi:10.1/h"]
    assert second["cost"] > first["cost"]
    assert len(result["paths"]) == 2  # every other route runs through c or h


async def test_without_the_hub_penalty_the_hub_route_is_as_cheap(monkeypatch):
    monkeypatch.setattr(zs, "_HUB_PENALTY", 0.0)
    result = await zs.connect_papers(_rag(), "10.1/a", "10.1/e")
    assert {p["papers"][0]["paper_id"] for p in result["paths"]} == {
        "doi:10.1/c",
        "doi:10.1/h",
    }
    assert result["paths"][0]["cost"] == result["paths"][1]["cost"] == 2.0


async def test_side_links_name_authors_topics_and_shared_citations():
    result = await zs.connect_papers(_rag(), "10.1/a", "10.1/b")
    links = result["links"]
    assert links["relation"] == ["cites"]
    assert links["shared_authors"] == ["Ben Lind"]
    assert links["same_topic"] == "Gravity Measurements"
    assert links["same_field"] == "Earth and Planetary Sciences"
    assert links["cited_by_both"] == 1  # c cites both
    # the direct link is used once; the alternative goes around it
    assert [p["hops"] for p in result["paths"]] == [1, 2]


async def test_unconnected_unknown_and_identical_papers_say_so():
    island = await zs.connect_papers(_rag(), "10.1/a", "10.1/x")
    assert island["reason"] == "not_connected" and island["paths"] == []
    assert island["links"]["shared_authors"] == ["Ada Okafor"]
    unknown = await zs.connect_papers(_rag(), "10.1/a", "ZZZZ9999")
    assert unknown == {
        "reason": "not_in_structure",
        "missing": ["ZZZZ9999"],
        "paths": [],
    }
    same = await zs.connect_papers(_rag(), "AAAA0001", "10.1/a")
    assert same["reason"] == "same_paper"


async def test_max_hops_bounds_the_search():
    result = await zs.connect_papers(_rag(), "10.1/b", "10.1/e", max_hops=1)
    assert result["reason"] == "not_connected"
    assert result["max_hops"] == 1
    assert "at most 1 steps over citations (either direction)" in result["message"]
    assert "bibliographic coupling" in result["message"]


def _long_cheap_chain_tables():
    """a to b in three steps through the hub q (costly: 255 links), or in four
    through the two quiet papers p1 and p2. Both routes meet at v, which the
    cheaper chain reaches in three steps and the hub in two."""
    papers = ["a", "b", "q", "v", "p1", "p2"] + [f"fan{i}" for i in range(255)]
    cites = [("a", "q"), ("q", "v"), ("v", "b"), ("a", "p1"), ("p1", "p2"), ("p2", "v")]
    cites += [(f"fan{i}", "q") for i in range(255)]
    return _graph_tables(papers, cites)


async def test_a_hop_limit_does_not_hide_a_costlier_shorter_route():
    # Settling each paper once by cost reached v by the cheap chain (three
    # steps), could not go on to b within three, and never tried the hub route.
    rag = _rag(_DB(_long_cheap_chain_tables()))
    short = await zs.connect_papers(rag, "10.1/a", "10.1/b", max_paths=1, max_hops=3)
    assert [p["hops"] for p in short["paths"]] == [3]
    assert [p["paper_id"] for p in short["paths"][0]["papers"]] == [
        "doi:10.1/q",
        "doi:10.1/v",
    ]
    # with room for four steps the cheaper chain wins
    long = await zs.connect_papers(rag, "10.1/a", "10.1/b", max_paths=1, max_hops=4)
    assert [p["paper_id"] for p in long["paths"][0]["papers"]] == [
        "doi:10.1/p1",
        "doi:10.1/p2",
        "doi:10.1/v",
    ]
    assert long["paths"][0]["cost"] < short["paths"][0]["cost"]
    # alternatives are searched the same way: after the chain, the hub route
    both = await zs.connect_papers(rag, "10.1/a", "10.1/b", max_paths=2, max_hops=4)
    assert [p["hops"] for p in both["paths"]] == [4]  # both routes need v
    two = await zs.connect_papers(rag, "10.1/a", "10.1/b", max_paths=2, max_hops=2)
    assert two["reason"] == "not_connected"


async def test_steps_of_equal_label_papers_are_told_apart():
    tables = _graph_tables(
        ["a", "x", "c", "e"],
        cites=[("a", "c"), ("c", "e")],
        coupled=[("a", "x", 0.2)],
    )
    tables["authors"] = [
        {"paper_id": f"doi:10.1/{n}", "author_id": "oa:1", "display_name": "Ada Okafor"}
        for n in ("a", "x")
    ]
    result = await zs.connect_papers(_rag(_DB(tables)), "10.1/x", "10.1/e", 1)
    path = result["paths"][0]
    labels = [path["steps"][0]["from"]] + [s["to"] for s in path["steps"]]
    # x, a, c, e: the two Okafor 2010 papers differ, the others read as before
    assert labels[:2] == ["Okafor 2010b", "Okafor 2010a"]
    assert result["a"]["label"] == "Okafor 2010b"
    assert [p["label"] for p in path["papers"]][0] == "Okafor 2010a"
    assert len(set(labels)) == len(labels)


# --- paper map --------------------------------------------------------------


def _node(result, node_id):
    return next(n for n in result["nodes"] if n["id"] == node_id)


def _edge(result, a, b):
    return next(e for e in result["edges"] if {e["source"], e["target"]} == {a, b})


async def test_a_one_hop_map_shows_the_papers_linked_to_the_seed():
    result = await zs.paper_map(_rag(), ["AAAA0001"])
    ids = [n["id"] for n in result["nodes"]]
    assert ids[0] == "10.1/a"
    assert set(ids) == {"10.1/a", "10.1/b", "10.1/c", "10.1/d", "10.1/h"}
    node = _node(result, "10.1/a")
    assert node["labels"] == ["Okafor 2010"]
    props = node["properties"]
    assert props["entity_type"] == "Earth and Planetary Sciences"
    assert props["title"] == "Alpha gravity leakage"
    assert props["attachment_key"] == "AAAA0002" and props["in_kb"]
    assert props["authors"] == "Ada Okafor; Ben Lind" and props["hop"] == 0
    assert "entity_id" not in props
    # field of the primary topic; the schema's first author when Zotero has none
    assert _node(result, "10.1/h")["properties"]["entity_type"] == "Engineering"
    assert _node(result, "10.1/c")["labels"] == ["Varga 2014"]
    view = result["paper"]
    assert view["mode"] == "map" and view["n_hop1"] == 4 and view["paper"] == "10.1/a"


async def test_map_edges_keep_citation_direction_and_carry_coupling():
    result = await zs.paper_map(_rag(), ["10.1/a"])
    cites = _edge(result, "10.1/a", "10.1/b")
    assert (cites["source"], cites["target"], cites["type"]) == (
        "10.1/a",
        "10.1/b",
        "DIRECTED",
    )
    assert cites["properties"]["keywords"] == "cites"
    both = _edge(result, "10.1/a", "10.1/c")
    assert (both["source"], both["target"]) == ("10.1/c", "10.1/a")
    assert both["properties"]["keywords"] == "cites, coupled"
    assert both["properties"]["weight"] == 1.2
    coupled = _edge(result, "10.1/a", "10.1/d")
    assert coupled["type"] == "UNDIRECTED" and coupled["properties"]["weight"] == 0.8
    assert coupled["properties"]["shared_refs"] == 7
    # c cites b too: edges between neighbours are shown
    assert _edge(result, "10.1/c", "10.1/b")["source"] == "10.1/c"
    assert len(result["edges"]) == 5


async def test_the_cap_keeps_the_seeds_and_the_strongest_neighbours():
    result = await zs.paper_map(_rag(), ["10.1/a"], max_nodes=3)
    # c: citation + coupling; b and h one citation each, the most cited first
    assert [n["id"] for n in result["nodes"]] == ["10.1/a", "10.1/c", "10.1/h"]
    assert result["is_truncated"]
    assert result["paper"]["shown_hop1"] == 2 and result["paper"]["n_hop1"] == 4


async def test_two_hops_rank_after_one_hop_and_damp_hubs():
    result = await zs.paper_map(_rag(), ["10.1/a"], hops=2, max_nodes=6)
    hops = [n["properties"]["hop"] for n in result["nodes"]]
    assert hops == [0, 1, 1, 1, 1, 2]
    # e, reached through c and the hub, beats the hub's 60 citing papers, each
    # reached through the hub alone and damped by its 62 links
    assert result["nodes"][-1]["id"] == "10.1/e"
    assert result["paper"]["n_hop2"] == 61 and result["is_truncated"]


async def test_several_seeds_and_unknown_ones():
    result = await zs.paper_map(_rag(), ["10.1/e", "ZZZZ9999", "KKKK0001"])
    summaries = result["paper"]["papers"]
    assert [s.get("reason") for s in summaries] == [None, "not_in_structure", None]
    assert summaries[2]["paper"] == "KKKK0001"  # no DOI: its key
    assert {"10.1/e", "KKKK0001", "10.1/c", "10.1/h"} <= {
        n["id"] for n in result["nodes"]
    }
    # no author and no year: the title's first words
    assert _node(result, "KKKK0001")["labels"] == ["A report without a DOI"]


async def test_maps_that_cannot_be_drawn_say_why():
    unknown = await zs.paper_map(_rag(), ["ZZZZ9999"])
    assert unknown["nodes"] == [] and unknown["paper"]["reason"] == "not_in_structure"
    island = await zs.paper_map(_rag(), ["10.1/x"])
    assert [n["id"] for n in island["nodes"]] == ["10.1/x"]
    assert island["paper"]["reason"] == "no_links"
    db = _DB()
    many = await zs.paper_map(_rag(db), [f"10.1/fan{i}" for i in range(11)])
    assert many["paper"]["reason"] == "too_many_papers" and db.calls == []


async def test_equal_labels_are_told_apart(monkeypatch):
    tables = dict(TABLES)
    tables["papers"] = [
        dict(p, year=2010) if p["paper_id"] in ("doi:10.1/a", "doi:10.1/x") else p
        for p in TABLES["papers"]
    ]
    tables["coupled"] = TABLES["coupled"] + [
        {
            "paper_a": "doi:10.1/a",
            "paper_b": "doi:10.1/x",
            "shared_refs": 3,
            "score": 0.2,
        }
    ]
    result = await zs.paper_map(_rag(_DB(tables)), ["10.1/a"])
    assert _node(result, "10.1/a")["labels"] == ["Okafor 2010a"]
    assert _node(result, "10.1/x")["labels"] == ["Okafor 2010b"]


# --- map links: thinned, and encoded off the loop -----------------------------

N_AROUND = 20


def _crowd_tables():
    """Seed s cites twenty papers n00..n19, which are coupled in every pair
    with a different score each (about 0.02 to 0.4, so below a citation)."""
    names = [f"n{i:02d}" for i in range(N_AROUND)]
    coupled = [
        (names[i], names[j], (i * N_AROUND + j) / 1000)
        for i in range(N_AROUND)
        for j in range(i + 1, N_AROUND)
    ]
    return _graph_tables(["s"] + names, [("s", n) for n in names], coupled)


def _pairs(result):
    return {frozenset((e["source"], e["target"])) for e in result["edges"]}


async def test_map_links_keep_every_seed_link_and_each_papers_strongest():
    result = await zs.paper_map(_rag(_DB(_crowd_tables())), ["10.1/s"])
    view = result["paper"]
    total = N_AROUND + N_AROUND * (N_AROUND - 1) // 2
    assert (view["n_links"], view["shown_links"]) == (total, len(result["edges"]))
    # links were thinned, but every paper is shown: that is not a truncation
    assert view["shown_links"] < total and view["links_thinned"]
    assert not result["is_truncated"]
    pairs = _pairs(result)
    assert len(pairs) == len(result["edges"])  # an edge kept for both ends: once
    assert all(
        frozenset(("10.1/s", f"10.1/n{i:02d}")) in pairs for i in range(N_AROUND)
    )
    # each paper's five strongest links to the others (its sixth is the seed)
    for i in range(N_AROUND):
        score = lambda j: (min(i, j) * N_AROUND + max(i, j)) / 1000  # noqa: E731
        best = sorted((j for j in range(N_AROUND) if j != i), key=score)[::-1][:5]
        for j in best:
            assert frozenset((f"10.1/n{i:02d}", f"10.1/n{j:02d}")) in pairs
    # nothing in the result is random
    again = await zs.paper_map(_rag(_DB(_crowd_tables())), ["10.1/s"])
    assert [e["id"] for e in again["edges"]] == [e["id"] for e in result["edges"]]


async def test_the_edge_total_is_capped_and_the_weakest_other_links_go_first(
    monkeypatch,
):
    rag = _rag(_DB(_crowd_tables()))
    full = await zs.paper_map(rag, ["10.1/s"], max_nodes=N_AROUND + 1)
    monkeypatch.setattr(zs, "_EDGE_CAP_FACTOR", 1)  # 21 links for 21 papers
    capped = await zs.paper_map(rag, ["10.1/s"], max_nodes=N_AROUND + 1)
    assert len(capped["edges"]) == N_AROUND + 1
    assert capped["paper"]["n_links"] == full["paper"]["n_links"]
    assert capped["paper"]["links_thinned"] and not capped["is_truncated"]
    seed_links = {e["id"] for e in capped["edges"] if "10.1/s" in e["id"].split("-")}
    assert len(seed_links) == N_AROUND  # the seed's links stay
    dropped = [
        e for e in full["edges"] if e["id"] not in {x["id"] for x in capped["edges"]}
    ]
    other = [e for e in capped["edges"] if e["id"] not in seed_links]
    assert len(other) == 1
    assert other[0]["properties"]["weight"] >= max(
        e["properties"]["weight"] for e in dropped
    )


async def test_a_map_with_few_links_is_not_thinned():
    result = await zs.paper_map(_rag(), ["10.1/a"])
    assert result["paper"]["n_links"] == result["paper"]["shown_links"] == 5
    assert not result["is_truncated"] and not result["paper"]["links_thinned"]


def _floating_tables():
    """Seed s cites five papers h0..h4 (one hop), each coupled with all thirty of
    m00..m29 (two hops). m0k's links are all weaker than m(k+1)'s, so the cap
    cuts the weakest papers' links first."""
    hubs = [f"h{i}" for i in range(5)]
    outer = [f"m{i:02d}" for i in range(30)]
    coupled = [(h, m, (j + 1) / 100) for j, m in enumerate(outer) for h in hubs]
    return _graph_tables(["s"] + hubs + outer, [("s", h) for h in hubs], coupled)


async def test_the_edge_cap_leaves_no_shown_paper_without_its_strongest_link(
    monkeypatch,
):
    monkeypatch.setattr(zs, "_EDGE_CAP_FACTOR", 1)
    rag = _rag(_DB(_floating_tables()))
    result = await zs.paper_map(rag, ["10.1/s"], hops=2, max_nodes=36)
    assert len(result["nodes"]) == 36
    linked = {end for e in result["edges"] for end in (e["source"], e["target"])}
    assert {n["id"] for n in result["nodes"]} <= linked
    view = result["paper"]
    assert view["shown_links"] == len(result["edges"]) > 36  # a little above the cap
    assert view["n_links"] == 5 + 5 * 30 and view["links_thinned"]
    again = await zs.paper_map(rag, ["10.1/s"], hops=2, max_nodes=36)
    assert [e["id"] for e in again["edges"]] == [e["id"] for e in result["edges"]]


def _wide_tables(n=200):
    names = [f"n{i:03d}" for i in range(n)]
    return _graph_tables(["s"] + names, [("s", x) for x in names])


async def test_a_map_shows_150_papers_unless_asked_for_more():
    rag = _rag(_DB(_wide_tables()))
    result = await zs.paper_map(rag, ["10.1/s"])
    view = result["paper"]
    assert len(result["nodes"]) == 150 and result["is_truncated"]
    assert (view["n_papers"], view["shown_papers"]) == (201, 150)
    assert (view["max_nodes"], view["max_nodes_cap"]) == (150, 1000)
    assert result["nodes"][0]["id"] == "10.1/s"
    more = await zs.paper_map(rag, ["10.1/s"], max_nodes=300)
    assert len(more["nodes"]) == 201 and not more["is_truncated"]
    assert more["paper"]["max_nodes"] == 300
    # the server's own limit is the ceiling
    rag.max_graph_nodes = 180
    capped = await zs.paper_map(rag, ["10.1/s"], max_nodes=5000)
    assert len(capped["nodes"]) == 180
    assert (capped["paper"]["max_nodes"], capped["paper"]["max_nodes_cap"]) == (
        180,
        180,
    )


async def test_the_map_limit_does_not_follow_the_graph_viewers_node_limit():
    # a server whose graph limit is 1000 still answers a map with 150 papers
    rag = _rag(_DB(_wide_tables()))
    assert rag.max_graph_nodes == 1000
    assert len((await zs.paper_map(rag, ["10.1/s"]))["nodes"]) == 150


async def test_a_server_object_without_a_node_limit_still_draws_the_map():
    rag = _rag()
    del rag.max_graph_nodes
    result = await zs.paper_map(rag, ["10.1/a"])
    assert len(result["nodes"]) == 5 and result["paper"]["max_nodes"] == 150


async def test_a_node_reports_every_attachment_key_of_its_paper():
    result = await zs.paper_map(_rag(), ["10.1/a"])
    props = _node(result, "10.1/a")["properties"]
    assert props["attachment_key"] == "AAAA0002"  # the one in the knowledge base
    assert props["attachment_keys"] == ["AAAA0001", "AAAA0002"]
    # a paper with no key sends none
    island = await zs.paper_map(_rag(), ["zot:ITEM0001"])
    assert _node(island, "KKKK0001")["properties"]["attachment_keys"] == ["KKKK0001"]


async def test_the_map_is_encoded_in_the_worker_thread(monkeypatch):
    threads = []
    real = zs.json.dumps

    def spy(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(zs, "json", SimpleNamespace(dumps=spy))
    rag = _rag()
    body = await zs.paper_map(rag, ["10.1/a"], encode=True)
    assert isinstance(body, bytes) and threads
    assert threading.get_ident() not in threads
    assert json.loads(body) == await zs.paper_map(rag, ["10.1/a"])
    # the too-many answer is encoded too
    many = await zs.paper_map(rag, [f"10.1/fan{i}" for i in range(11)], encode=True)
    assert json.loads(many)["paper"]["reason"] == "too_many_papers"


# --- routes -----------------------------------------------------------------


def _client(rag):
    app = FastAPI()
    app.include_router(_graph_routes.create_graph_routes(rag, api_key="test-key"))
    return TestClient(app), {"X-API-Key": "test-key"}


ROUTES = (
    "/graph/papers/related?paper=10.1/a",
    "/graph/papers/topics?q=gravity",
    "/graph/papers/by_topic?topic=gravity",
    "/graph/papers/connect?a=10.1/a&b=10.1/b",
    "/graphs/paper/map?paper=10.1/a",
)
# more papers than a map takes: the answer must not depend on the setting
MANY_PAPERS = "/graphs/paper/map?" + "&".join(f"paper=10.1/p{i}" for i in range(11))


def test_routes_answer_501_while_the_structure_is_off(monkeypatch):
    monkeypatch.delenv(zs.SCHEMA_ENV)
    db = _DB()
    client, headers = _client(_rag(db))
    for url in (*ROUTES, MANY_PAPERS):
        r = client.get(url, headers=headers)
        assert r.status_code == 501, url
        assert "ZOTERO_STRUCT_SCHEMA is not set" in r.json()["detail"]
    assert db.calls == []


def test_too_many_papers_is_an_answer_only_while_the_structure_is_on():
    client, headers = _client(_rag())
    r = client.get(MANY_PAPERS, headers=headers)
    assert r.status_code == 200
    assert r.json()["paper"]["reason"] == "too_many_papers"


@pytest.mark.parametrize(
    "url",
    [
        "/graph/papers/related?paper=10.1/a&relations=cites,friends",
        "/graph/papers/related?paper=10.1/a&relations=,",
        "/graph/papers/related?paper=10.1/a&limit=0",
        "/graph/papers/related?paper=" + "x" * 301,
        "/graph/papers/topics?level=planet",
        "/graph/papers/connect?a=10.1/a",
        "/graph/papers/connect?a=10.1/a&b=10.1/b&paths=6",
        "/graph/papers/connect?a=10.1/a&b=10.1/b&max_hops=9",
        "/graphs/paper/map?paper=10.1/a&hops=3",
        "/graphs/paper/map?paper=" + "x" * 301,
        "/graphs/paper/map",
    ],
)
def test_routes_refuse_bad_input_before_reading(url):
    db = _DB()
    client, headers = _client(_rag(db))
    assert client.get(url, headers=headers).status_code == 422
    assert db.calls == []


@pytest.mark.parametrize("url", ROUTES)
def test_every_route_needs_the_api_key(url):
    db = _DB()
    client, _ = _client(_rag(db))
    assert client.get(url).status_code in (401, 403)
    assert client.get(url, headers={"X-API-Key": "wrong"}).status_code in (401, 403)
    assert db.calls == []


def test_the_map_answers_json_with_the_shape_of_the_function():
    rag = _rag()
    client, headers = _client(rag)
    r = client.get("/graphs/paper/map?paper=10.1/a", headers=headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    expected = asyncio.run(zs.paper_map(rag, ["10.1/a"]))
    assert r.json() == expected


def test_the_map_route_defaults_to_150_papers_and_takes_a_limit():
    rag = _rag(_DB(_wide_tables()))
    client, headers = _client(rag)
    plain = client.get("/graphs/paper/map?paper=10.1/s", headers=headers).json()
    assert len(plain["nodes"]) == 150 and plain["paper"]["max_nodes"] == 150
    assert plain["is_truncated"] and plain["paper"]["n_papers"] == 201
    more = client.get("/graphs/paper/map?paper=10.1/s&max_nodes=300", headers=headers)
    assert len(more.json()["nodes"]) == 201


def test_routes_pass_their_options():
    rag = _rag()
    rag.max_graph_nodes = 4
    client, headers = _client(rag)
    related = client.get(
        "/graph/papers/related?paper=10.1/a&relations=coupled&limit=1",
        headers=headers,
    ).json()
    assert [r["paper_id"] for r in related["related"]] == ["doi:10.1/d"]
    graph = client.get(
        "/graphs/paper/map?paper=10.1/a&paper=10.1/e&max_nodes=50", headers=headers
    ).json()
    assert len(graph["nodes"]) == 4  # the server's node cap wins
    connect = client.get(
        "/graph/papers/connect?a=10.1/a&b=10.1/e&paths=1", headers=headers
    ).json()
    assert len(connect["paths"]) == 1
    topics = client.get("/graph/papers/topics?level=domain", headers=headers).json()
    assert topics["total"] == 1
    papers = client.get(
        "/graph/papers/by_topic?topic=topic:T2&primary_only=true", headers=headers
    ).json()
    assert papers["total"] == 2
