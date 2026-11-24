"""The citation-graph additions to the paper structure (lightrag/zotero_struct.py):
the v2 columns and tables a schema may carry, the four routes that read them, and
the additive fields of the older routes. A v1 schema must answer as it always did.

The pool is the stub of tests/test_zotero_struct.py, extended with the tables the
schema builder adds; the SQL that stays in the database (citation contexts,
embeddings) is answered by a small imitation of what it computes. Fixtures are
invented.
"""

import math
import zlib

import pytest

import lightrag.zotero_struct as zs
from tests.test_zotero_struct import (  # noqa: F401  (structure: autouse)
    TABLES,
    _client,
    _DB,
    _graph_tables,
    _paper,
    _rag,
    structure,
)

pytestmark = pytest.mark.offline

A, B, C, E, H, D = (f"doi:10.1/{x}" for x in "abcehd")


def _v2_tables():
    papers = []
    for r in TABLES["papers"]:
        r = dict(
            r, tldr=None, s2_citation_count=None, s2_influential_citation_count=None
        )
        if r["paper_id"] == A:
            r.update(tldr="Alpha measures gravity leakage.", s2_citation_count=40)
            r["s2_influential_citation_count"] = 6
        if r["paper_id"] == B:
            r.update(tldr="Bravo maps mascons.", s2_citation_count=12)
            r["s2_influential_citation_count"] = 0
        papers.append(r)
    meta = {
        (A, B): ("both", True, ["background", "methodology"], 2),
        (C, B): ("s2", False, [], 0),
        (E, C): ("openalex", None, [], None),
        (A, H): ("openalex", None, [], None),
        (C, A): ("s2", True, ["result"], 1),
    }
    cites = []
    for r in TABLES["cites"]:
        key = (r["citing_paper_id"], r["cited_paper_id"])
        src, infl, intents, n = meta.get(key, ("openalex", None, [], None))
        cites.append(
            dict(r, source=src, is_influential=infl, intents=intents, n_contexts=n)
        )

    def work(wid, title, cited, n):
        return {
            "work_id": wid,
            "doi": f"10.9/{wid[3:].lower()}",
            "title": title,
            "year": 1990,
            "cited_by_count": cited,
            "first_author": f"Author {wid[3:]}",
            "venue": "Old Journal",
            "n_library_citers": n,
        }

    return dict(
        TABLES,
        papers=papers,
        cites=cites,
        external_works=[
            work("oa:W1", "Classic one", 100, 3),
            work("oa:W2", "Classic two", 500, 3),
            work("oa:W3", "Rare three", 50, 1),
            work("oa:W4", "Rare four", 9, 1),
            work("oa:W5", "Uncited here", 7, 0),
        ],
        cites_external=[
            {"citing_paper_id": p, "work_id": w}
            for p, w in [(A, "oa:W1"), (A, "oa:W2"), (A, "oa:W3"), (C, "oa:W1")]
            + [(C, "oa:W2"), (E, "oa:W1"), (B, "oa:W2"), (D, "oa:W4")]
        ],
    )


CONTEXTS = {
    (A, B): [
        {"ordinal": 2, "context": "Second sentence.", "intents": ["methodology"]},
        {"ordinal": 1, "context": "First sentence.", "intents": ["background"]},
    ],
    (C, A): [
        {"ordinal": n, "context": f"Sentence {n}.", "intents": []} for n in range(1, 26)
    ],
}
# 2-d "embeddings": the cosine to a is 1.0 for itself, then b, then h, e is orthogonal
EMBEDDINGS = {
    A: (1.0, 0.0),
    B: (0.9, 0.1),
    C: (0.2, 0.9),  # never linked to a by citation... c cites a, so linked
    H: (0.7, 0.7),
    E: (0.0, 1.0),
    D: (0.8, 0.5),  # coupled to a, not cited
    "doi:10.1/x": (0.95, 0.2),  # shares an author with a
}
V2_COLUMNS = {
    "papers": list(zs._V2_COLUMNS["papers"]),
    "cites": list(zs._V2_COLUMNS["cites"]),
}


def _cos(u, v):
    return sum(x * y for x, y in zip(u, v)) / (math.hypot(*u) * math.hypot(*v))


class _V2DB(_DB):
    """The stub pool, with the optional tables and the per-request queries."""

    def __init__(
        self,
        tables=None,
        optional=(
            "cite_contexts",
            "cites_external",
            "external_works",
            "paper_embeddings",
        ),
        columns=None,
        **kw,
    ):
        super().__init__(tables or _v2_tables(), **kw)
        self.optional = list(optional)
        self.columns = V2_COLUMNS if columns is None else columns
        self.sql: dict[str, str] = {}

    async def query(self, sql, params=None, multirows=False):
        name = sql.split("\n", 1)[0].split(":", 1)[1]
        self.sql[name] = sql
        if name == "fingerprint":
            rows = await super().query(sql, params, multirows)
            if rows:
                base = len(rows)
                rows += [
                    {"relname": t, "oid": base + i, "filenode": self.filenode}
                    for i, t in enumerate(self.optional)
                ]
            return rows
        if name == "contexts":
            self.calls.append((name, params))
            citing, cited, limit = params
            rows = sorted(CONTEXTS.get((citing, cited), []), key=lambda r: r["ordinal"])
            return rows[:limit]
        if name == "similar":
            self.calls.append((name, params))
            pid, limit, only = params
            if pid not in EMBEDDINGS and not (
                "WITH q AS" in sql and "embedding IS NOT NULL" in sql
            ):
                # the SQL of the first draft: the scalar subquery is NULL, pgvector's
                # <=> is strict, so every neighbour comes back with similarity NULL
                return [
                    {"paper_id": p, "similarity": None}
                    for p in EMBEDDINGS
                    if p != pid and (only is None or p in only)
                ][:limit]
            if pid not in EMBEDDINGS:
                return []  # driven from the paper's own non-null embedding: no row
            rows = [
                {"paper_id": p, "similarity": _cos(EMBEDDINGS[pid], v)}
                for p, v in EMBEDDINGS.items()
                if p != pid and (only is None or p in only)
            ]
            rows.sort(key=lambda r: -r["similarity"])
            return rows[:limit]
        if name == "has_embedding":
            self.calls.append((name, params))
            return {"found": 1} if params[0] in EMBEDDINGS else None
        if name == "external_works":  # chunked by work id
            self.calls.append((name, params))
            chunks, k = params
            return [
                dict(r)
                for r in self.tables[name]
                if zlib.crc32(r["work_id"].encode()) % chunks == k
            ]
        return await super().query(sql, params, multirows)


def _v2(**kw):
    db = _V2DB(**kw)
    return db, _rag(db)


def _ids(rows, key="paper_id"):
    return [r[key] for r in rows]


def _by(rows):
    return {r["paper_id"]: r for r in rows}


# --- loading ----------------------------------------------------------------


async def test_a_v1_schema_loads_and_answers_without_any_v2_field():
    db = _DB()
    rag = _rag(db)
    related = await zs.related_papers(rag, "10.1/a")
    assert "tldr" not in related["paper"]
    assert all("citation" not in r for r in related["related"])
    connect = await zs.connect_papers(rag, "10.1/a", "10.1/b")
    assert "citation" not in connect["links"]
    st = await zs._structure(rag)
    assert st.tables == [] and st.ext == [] and st.cite_meta == {}
    assert zs.optional_tables() == []
    # no v2 table was read, and the columns were asked for once
    assert set(db.names()) <= set(zs._LOADED) | {"fingerprint", "columns", "documents"}
    assert db.names().count("columns") == 1


async def test_the_v2_columns_are_asked_for_only_when_the_schema_has_them():
    db, rag = _v2(columns={"papers": ["tldr"], "cites": ["source", "n_contexts"]})
    await zs.related_papers(rag, "10.1/a")
    assert ", p.tldr" in db.sql["papers"]
    assert "s2_citation_count" not in db.sql["papers"]
    assert db.sql["cites"].startswith(
        "-- zotero_struct:cites\nSELECT citing_paper_id, cited_paper_id, source, n_contexts FROM"
    )
    assert 'FROM "zs".cites' in db.sql["cites"]
    st = await zs._structure(rag)
    assert list(st.paper_extra) == ["tldr"]
    assert st.cite_cols == {"source", "n_contexts"}


async def test_papers_carry_the_tldr_and_the_semantic_scholar_counts():
    _, rag = _v2()
    paper = (await zs.related_papers(rag, "10.1/a"))["paper"]
    assert paper["tldr"] == "Alpha measures gravity leakage."
    assert (paper["s2_citation_count"], paper["s2_influential_citation_count"]) == (
        40,
        6,
    )
    # a paper S2 does not know has the keys, null
    echo = (await zs.related_papers(rag, "10.1/e"))["paper"]
    assert echo["tldr"] is None and echo["s2_citation_count"] is None
    # topic lists use the same summary
    listed = (await zs.papers_by_topic(rag, "topic:T1"))["papers"]
    assert listed and all("tldr" in p for p in listed)


async def test_the_optional_tables_are_found_in_the_fingerprint_and_reread_on_a_swap(
    monkeypatch,
):
    db, rag = _v2(optional=("paper_embeddings",), columns={})
    assert await zs.available(rag)
    assert zs.optional_tables() == ["paper_embeddings"]
    await zs.related_papers(rag, "10.1/a")
    assert "external_works" not in db.names()  # the pair is incomplete: not read
    st = await zs._structure(rag)
    assert st.tables == ["paper_embeddings"]
    names = {t for t, _ in st.fingerprint}
    assert names == set(zs._TABLES) | {"paper_embeddings"}
    # a swap that adds the outside works is seen
    monkeypatch.setattr(zs, "_CHECK_INTERVAL", 0.0)
    db.optional += ["external_works", "cites_external"]
    db.filenode = 2
    await zs.related_papers(rag, "10.1/a")
    assert db.names().count("papers") == 2
    assert (await zs._structure(rag)).tables == [
        "cites_external",
        "external_works",
        "paper_embeddings",
    ]
    assert len((await zs._structure(rag)).ext) == 5


async def test_the_big_v2_tables_are_read_in_slices_that_cover_every_row_once():
    db, rag = _v2()
    st = await zs._structure(rag)
    assert db.names().count("external_works") == zs._CHUNKS
    assert db.names().count("cites_external") == zs._CHUNKS
    assert sorted(w["work_id"] for w in st.ext) == [f"oa:W{n}" for n in range(1, 6)]
    assert sum(len(c) for c in st.paper_ext) == 8
    # the contexts and the embeddings stay in the database
    assert "contexts" not in db.names() and "similar" not in db.names()


async def test_availability_names_the_tables_and_drops_them_when_unreadable(
    monkeypatch,
):
    db, rag = _v2()
    assert await zs.available(rag)
    assert zs.optional_tables() == sorted(db.optional)
    monkeypatch.setenv(zs.SCHEMA_ENV, "elsewhere")
    zs._reset_for_tests()
    assert not await zs.available(rag)
    assert zs.optional_tables() == []


# --- the additive fields of the older routes --------------------------------


async def test_related_papers_carry_the_citation_facts_of_their_edge():
    _, rag = _v2()
    related = {
        r["paper_id"]: r for r in (await zs.related_papers(rag, "10.1/a"))["related"]
    }
    assert related[B]["citation"] == {
        "source": "both",
        "is_influential": True,
        "intents": ["background", "methodology"],
        "n_contexts": 2,
    }
    # c cites a: the row reads cited_by, and the facts are those of c -> a
    assert related[C]["citation"]["intents"] == ["result"]
    assert related[C]["citation"]["is_influential"] is True
    assert related[H]["citation"]["source"] == "openalex"
    assert "citation" not in related[D]  # coupling only


async def test_a_relations_filter_picks_the_direction_of_the_citation():
    g = _graph_tables(["p", "q"], cites=[("p", "q"), ("q", "p")])
    g["cites"][0].update(
        source="s2", is_influential=True, intents=["result"], n_contexts=1
    )
    g["cites"][1].update(
        source="openalex", is_influential=None, intents=[], n_contexts=None
    )
    _, rag = _v2(
        tables=g, optional=(), columns={"cites": list(zs._V2_COLUMNS["cites"])}
    )
    only = await zs.related_papers(rag, "10.1/p", ["cites"])
    assert only["related"][0]["citation"]["source"] == "s2"
    only = await zs.related_papers(rag, "10.1/p", ["cited_by"])
    assert only["related"][0]["citation"]["source"] == "openalex"
    both = await zs.related_papers(rag, "10.1/p", ["cites", "cited_by"])
    # p -> q came from s2, q -> p from openalex: neither is "both" providers
    assert both["related"][0]["citation"] == {
        "source": "s2",
        "source_reverse": "openalex",
        "is_influential": True,
        "intents": ["result"],
        "n_contexts": 1,
    }


async def test_connect_links_gain_the_citation_and_scores_stay_as_before():
    _, rag = _v2()
    links = (await zs.connect_papers(rag, "10.1/a", "10.1/b"))["links"]
    assert links["citation"]["is_influential"] is True
    assert "citation" not in (await zs.connect_papers(rag, "10.1/a", "10.1/d"))["links"]
    v1 = await zs.related_papers(_rag(_DB()), "10.1/a")
    v2 = await zs.related_papers(_rag(_V2DB()), "10.1/a")
    assert [(r["paper_id"], r["score"]) for r in v1["related"]] == [
        (r["paper_id"], r["score"]) for r in v2["related"]
    ]


async def test_the_map_edges_and_nodes_carry_the_citation_and_the_tldr():
    _, rag = _v2()
    graph = await zs.paper_map(rag, ["10.1/a"])
    edge = next(
        e for e in graph["edges"] if {e["source"], e["target"]} == {"10.1/a", "10.1/b"}
    )
    assert edge["properties"]["is_influential"] is True
    assert edge["properties"]["intents"] == "background, methodology"
    assert edge["properties"]["n_contexts"] == 2
    coupling = next(e for e in graph["edges"] if "10.1/d" in {e["source"], e["target"]})
    assert "is_influential" not in coupling["properties"]
    nodes = {n["id"]: n["properties"] for n in graph["nodes"]}
    assert nodes["10.1/a"]["tldr"] == "Alpha measures gravity leakage."
    assert nodes["10.1/a"]["s2_citation_count"] == 40
    assert "tldr" not in nodes["10.1/d"]  # S2 has none for d


# --- citation ----------------------------------------------------------------


async def test_a_citation_returns_the_edge_and_its_contexts_in_order():
    db, rag = _v2()
    out = await zs.citation_context(rag, "10.1/a", "10.1/b")
    assert out["citing"]["paper_id"] == A and out["cited"]["paper_id"] == B
    assert out["edge"] == {
        "source": "both",
        "is_influential": True,
        "intents": ["background", "methodology"],
        "n_contexts": 2,
    }
    assert out["contexts"] == [
        {"ordinal": 1, "text": "First sentence.", "intents": ["background"]},
        {"ordinal": 2, "text": "Second sentence.", "intents": ["methodology"]},
    ]
    assert "reason" not in out
    # the paper ids and the cap travel as parameters
    assert ("contexts", [A, B, zs.MAX_CONTEXTS]) in db.calls
    assert 'FROM "zs".cite_contexts' in db.sql["contexts"]


async def test_contexts_are_capped_at_twenty():
    _, rag = _v2()
    out = await zs.citation_context(rag, "10.1/c", "10.1/a")
    assert len(out["contexts"]) == 20


async def test_an_edge_without_contexts_asks_nothing():
    db, rag = _v2()
    out = await zs.citation_context(rag, "10.1/c", "10.1/b")  # n_contexts 0
    assert out["contexts"] == [] and "reason" not in out
    out = await zs.citation_context(rag, "10.1/e", "10.1/c")  # no S2 row: null
    assert out["contexts"] == [] and out["edge"]["source"] == "openalex"
    assert "contexts" not in db.names()


async def test_no_citation_says_whether_it_runs_the_other_way():
    _, rag = _v2()
    out = await zs.citation_context(rag, "10.1/b", "10.1/a")  # a cites b, not b a
    assert out["reason"] == "no_citation" and out["cited_cites_citing"] is True
    assert out["edge"] is None and out["contexts"] == []
    assert out["citing"]["paper_id"] == B
    out = await zs.citation_context(rag, "10.1/a", "10.1/d")
    assert out["reason"] == "no_citation" and out["cited_cites_citing"] is False


async def test_a_citation_of_an_unknown_paper_names_it():
    _, rag = _v2()
    out = await zs.citation_context(rag, "10.1/a", "10.1/nowhere")
    assert out == {"reason": "not_in_structure", "missing": ["10.1/nowhere"]}


async def test_a_schema_without_the_contexts_table_still_gives_the_edge():
    db, rag = _v2(optional=("paper_embeddings",))
    out = await zs.citation_context(rag, "10.1/a", "10.1/b")
    assert out["reason"] == "no_contexts_table"
    assert out["edge"]["is_influential"] is True and out["contexts"] == []
    assert "contexts" not in db.names()


async def test_on_a_v1_schema_the_edge_has_no_facts_and_no_contexts_table():
    _, rag = _v2(tables=TABLES, optional=(), columns={})
    out = await zs.citation_context(rag, "10.1/a", "10.1/b")
    assert out["reason"] == "no_contexts_table"
    assert out["edge"] == {
        "source": None,
        "is_influential": None,
        "intents": [],
        "n_contexts": None,
    }


# --- similar -----------------------------------------------------------------


async def test_similar_papers_are_ranked_by_cosine_with_their_links():
    db, rag = _v2()
    out = await zs.similar_papers(rag, "10.1/a", limit=4)
    assert _ids(out["similar"]) == [B, "doi:10.1/x", D, H]
    by_id = _by(out["similar"])
    assert by_id[B]["similarity"] == round(_cos(EMBEDDINGS[A], EMBEDDINGS[B]), 4)
    assert by_id[B]["links"] == ["cites", "coauthor"]
    assert by_id[D]["links"] == ["coupled"]
    assert by_id["doi:10.1/x"]["links"] == ["coauthor"]
    assert by_id[H]["links"] == ["cites"]
    assert ("similar", [A, 4, None]) in db.calls
    assert 'FROM "zs".paper_embeddings' in db.sql["similar"]
    assert "<=>" in db.sql["similar"] and "LIMIT $2" in db.sql["similar"]


async def test_similar_work_the_paper_does_not_cite_has_empty_links():
    _, rag = _v2()
    out = await zs.similar_papers(rag, "10.1/a", limit=50)
    assert _by(out["similar"])[E]["links"] == []


async def test_in_kb_keeps_papers_with_text_in_the_knowledge_base():
    db, rag = _v2()
    out = await zs.similar_papers(rag, "10.1/a", limit=50, in_kb=True)
    # a, b and h have processed documents in this workspace
    assert set(_ids(out["similar"])) == {B, H}
    assert out["in_kb"] is True
    (call,) = [c for c in db.calls if c[0] == "similar"]
    assert call[1][2] and set(call[1][2]) == {A, B, H}


async def test_similar_reasons():
    db, rag = _v2()
    assert (await zs.similar_papers(rag, "10.1/nowhere"))[
        "reason"
    ] == "not_in_structure"
    out = await zs.similar_papers(rag, "10.1/h", in_kb=False)
    assert "reason" not in out and out["similar"]
    out = await zs.similar_papers(rag, "KKKK0001")  # no embedding of its own
    assert out["reason"] == "no_embedding" and out["similar"] == []
    db2, rag2 = _v2(optional=("external_works", "cites_external"))
    zs._reset_for_tests()
    out = await zs.similar_papers(rag2, "10.1/a")
    assert out["reason"] == "no_embeddings" and out["similar"] == []
    assert "similar" not in db2.names()


def test_the_similar_sql_is_driven_from_the_papers_own_non_null_embedding():
    sql = zs._SQL["similar"]
    assert sql.split("\n", 1)[1].lstrip().startswith("WITH q AS")
    assert "embedding IS NOT NULL" in sql and 'FROM "{s}".paper_embeddings e, q' in sql


async def test_a_paper_without_an_embedding_is_no_embedding_not_an_error():
    # against the first draft's SQL the stub returns NULL-similarity rows for it, as
    # PostgreSQL does, and float(None) raised: a 500 on the route
    _, rag = _v2()
    out = await zs.similar_papers(rag, "KKKK0001", limit=5)
    assert out["reason"] == "no_embedding" and out["similar"] == []
    client, headers = _client(rag)
    r = client.get("/graph/papers/similar?paper=KKKK0001", headers=headers)
    assert r.status_code == 200 and r.json()["reason"] == "no_embedding"


async def test_null_or_nan_similarities_are_skipped_not_fatal(monkeypatch):
    class Odd(_V2DB):
        async def query(self, sql, params=None, multirows=False):
            if sql.startswith("-- zotero_struct:similar"):
                return [
                    {"paper_id": B, "similarity": None},
                    {"paper_id": H, "similarity": float("nan")},
                    {"paper_id": E, "similarity": 0.5},
                ]
            return await super().query(sql, params, multirows)

    out = await zs.similar_papers(_rag(Odd()), "10.1/a")
    assert _ids(out["similar"]) == [E]


async def test_a_paper_whose_neighbours_are_all_filtered_out_is_not_missing_an_embedding():
    _, rag = _v2()
    out = await zs.similar_papers(rag, "10.1/e", in_kb=True)
    assert "reason" not in out
    assert _ids(out["similar"]) == [H, B, A]


# --- outside works -----------------------------------------------------------


async def test_outside_works_are_ranked_by_library_citers_then_citation_count():
    _, rag = _v2()
    out = await zs.outside_works(rag, "10.1/a")
    assert out["paper"]["paper_id"] == A and out["total"] == 3
    assert _ids(out["works"], "work_id") == ["oa:W2", "oa:W1", "oa:W3"]
    w2 = out["works"][0]
    assert w2["n_library_citers"] == 3 and w2["first_author"] == "Author W2"
    assert w2["title"] == "Classic two" and w2["venue"] == "Old Journal"
    assert w2["doi"] == "10.9/w2" and w2["year"] == 1990
    # the paper itself is not among its own co-citers; the most cited others first
    assert _ids(w2["also_cited_by"]) == [B, C]
    assert out["truncated"] is False


async def test_also_cited_by_names_at_most_five_other_papers():
    g = _graph_tables([f"p{i}" for i in range(8)])
    g["external_works"] = [
        {
            "work_id": "oa:W9",
            "doi": None,
            "title": "Hub",
            "year": 1,
            "cited_by_count": 1,
            "first_author": None,
            "venue": None,
            "n_library_citers": 8,
        }
    ]
    g["cites_external"] = [
        {"citing_paper_id": f"doi:10.1/p{i}", "work_id": "oa:W9"} for i in range(8)
    ]
    _, rag = _v2(tables=g, optional=("cites_external", "external_works"), columns={})
    out = await zs.outside_works(rag, "10.1/p0")
    assert len(out["works"][0]["also_cited_by"]) == zs.ALSO_CITED_BY
    assert "doi:10.1/p0" not in _ids(out["works"][0]["also_cited_by"])


async def test_outside_works_are_limited_and_say_so():
    _, rag = _v2()
    out = await zs.outside_works(rag, "10.1/a", limit=1)
    assert len(out["works"]) == 1 and out["truncated"] is True and out["total"] == 3
    assert (await zs.outside_works(rag, "10.1/x"))["works"] == []


async def test_outside_reasons():
    _, rag = _v2()
    assert (await zs.outside_works(rag, "10.1/nowhere"))["reason"] == "not_in_structure"
    _, v1 = _v2(tables=TABLES, optional=(), columns={})
    zs._reset_for_tests()
    out = await zs.outside_works(v1, "10.1/a")
    assert out["reason"] == "no_external_works" and out["works"] == []


# --- missing -----------------------------------------------------------------


async def test_missing_works_are_those_cited_by_enough_library_papers():
    _, rag = _v2()
    out = await zs.missing_works(rag)
    assert _ids(out["works"], "work_id") == ["oa:W2", "oa:W1"]
    assert out["total"] == 2 and out["min_citers"] == 2 and out["truncated"] is False
    assert "n_citers_in_topic" not in out["works"][0]
    assert _ids(out["works"][0]["also_cited_by"]) == [B, A, C]
    wide = await zs.missing_works(rag, min_citers=1, limit=3)
    assert _ids(wide["works"], "work_id") == ["oa:W2", "oa:W1", "oa:W3"]
    assert wide["total"] == 4 and wide["truncated"] is True


async def test_missing_works_under_a_topic_count_only_its_citers():
    _, rag = _v2()
    out = await zs.missing_works(rag, topic="topic:T1")  # a, b, c, e
    assert [(w["work_id"], w["n_citers_in_topic"]) for w in out["works"]] == [
        ("oa:W2", 3),
        ("oa:W1", 3),
    ]
    assert out["topic"]["topic_id"] == "topic:T1"
    # by words of the name, any level; T2 holds a, h and d
    out = await zs.missing_works(rag, topic="satellite gravity", min_citers=1)
    assert out["topic"]["topic_id"] == "topic:T2"
    assert {w["work_id"]: w["n_citers_in_topic"] for w in out["works"]} == {
        "oa:W1": 1,
        "oa:W2": 1,
        "oa:W3": 1,
        "oa:W4": 1,
    }
    assert all(set(_ids(w["also_cited_by"])) <= {A, D} for w in out["works"])


async def test_missing_works_reasons():
    _, rag = _v2()
    out = await zs.missing_works(rag, topic="no such field of study")
    assert out["reason"] == "no_topic" and out["works"] == []
    _, v1 = _v2(tables=TABLES, optional=(), columns={})
    zs._reset_for_tests()
    assert (await zs.missing_works(v1))["reason"] == "no_external_works"


# --- routes ------------------------------------------------------------------

V2_ROUTES = (
    "/graph/papers/citation?citing=10.1/a&cited=10.1/b",
    "/graph/papers/similar?paper=10.1/a",
    "/graph/papers/outside?paper=10.1/a",
    "/graph/papers/missing",
)


def test_the_new_routes_answer_501_while_the_structure_is_off(monkeypatch):
    monkeypatch.delenv(zs.SCHEMA_ENV)
    db = _V2DB()
    client, headers = _client(_rag(db))
    for url in V2_ROUTES:
        r = client.get(url, headers=headers)
        assert r.status_code == 501, url
        assert "ZOTERO_STRUCT_SCHEMA is not set" in r.json()["detail"]
    assert db.calls == []


@pytest.mark.parametrize("url", V2_ROUTES)
def test_the_new_routes_need_the_api_key(url):
    db = _V2DB()
    client, _ = _client(_rag(db))
    assert client.get(url).status_code in (401, 403)
    assert db.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "/graph/papers/citation?citing=10.1/a",
        "/graph/papers/citation?cited=10.1/a",
        "/graph/papers/citation?citing=10.1/a&cited=" + "x" * 301,
        "/graph/papers/similar?paper=10.1/a&limit=0",
        "/graph/papers/similar?paper=10.1/a&limit=51",
        "/graph/papers/similar?paper=10.1/a&in_kb=maybe",
        "/graph/papers/outside",
        "/graph/papers/outside?paper=10.1/a&limit=201",
        "/graph/papers/missing?limit=0",
        "/graph/papers/missing?limit=201",
        "/graph/papers/missing?min_citers=0",
        "/graph/papers/missing?topic=" + "x" * 201,
    ],
)
def test_the_new_routes_refuse_bad_input_before_reading(url):
    db = _V2DB()
    client, headers = _client(_rag(db))
    assert client.get(url, headers=headers).status_code == 422
    assert db.calls == []


def test_the_new_routes_answer_with_the_shape_of_the_functions():
    client, headers = _client(_rag(_V2DB()))
    r = client.get(V2_ROUTES[0], headers=headers).json()
    assert r["edge"]["n_contexts"] == 2 and len(r["contexts"]) == 2
    r = client.get(
        "/graph/papers/similar?paper=10.1/a&limit=2&in_kb=true", headers=headers
    ).json()
    assert _ids(r["similar"]) == [B, H]
    r = client.get("/graph/papers/outside?paper=10.1/a&limit=2", headers=headers).json()
    assert _ids(r["works"], "work_id") == ["oa:W2", "oa:W1"] and r["truncated"]
    r = client.get(
        "/graph/papers/missing?topic=topic:T1&limit=1&min_citers=3", headers=headers
    ).json()
    assert _ids(r["works"], "work_id") == ["oa:W2"]
    assert r["works"][0]["n_citers_in_topic"] == 3


def test_a_missing_paper_is_an_answer_not_an_error_on_the_routes():
    client, headers = _client(_rag(_V2DB()))
    r = client.get(
        "/graph/papers/citation?citing=10.1/a&cited=10.1/zzz", headers=headers
    )
    assert r.status_code == 200 and r.json()["reason"] == "not_in_structure"
    r = client.get("/graph/papers/outside?paper=10.1/zzz", headers=headers)
    assert r.status_code == 200 and r.json()["reason"] == "not_in_structure"
