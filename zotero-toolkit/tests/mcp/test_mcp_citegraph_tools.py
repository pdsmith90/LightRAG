"""Offline tests of the proxy's citation_context, similar_papers, outside_references and
missing_works, and of the citation facts and TLDR the older tools gained: the requests they
send to the server's paper-structure routes and how they word the answers.

The server is the stdlib fake from conftest.py; every answer is invented.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.offline


def _paper(pid, label, title, key=None, doi=None, **extra):
    return {
        "paper_id": pid,
        "doi": doi,
        "label": label,
        "title": title,
        "venue": "Journal of Invented Results",
        "in_kb": key is not None,
        "key": key,
        "keys": [key] if key else ["ZZZZ0009"],
        **extra,
    }


def _route(fake, path, answer):
    fake.routes[("GET", path)] = lambda params: answer(params)


ME = _paper("doi:10.9/a", "Okafor 2010", "Alpha gravity leakage", "AAAA0001", "10.9/a")
OTHER = _paper("doi:10.9/b", "Lind 2012", "Bravo mascons", "BBBB0002", "10.9/b")
CITATION = {
    "citing": ME,
    "cited": OTHER,
    "edge": {
        "source": "both",
        "is_influential": True,
        "intents": ["background", "methodology"],
        "n_contexts": 3,
    },
    "contexts": [
        {"ordinal": 1, "text": "Following Lind (2012) we map mascons.", "intents": []},
        {"ordinal": 2, "text": "x" * 700, "intents": ["background"]},
    ],
}


def test_citation_context_sends_both_papers_and_words_the_edge(
    load_server, fake_lightrag
):
    _route(fake_lightrag, "/graph/papers/citation", lambda p: (200, CITATION))
    srv = load_server()
    out = srv.citation_context("AAAA0001", "10.9/b")
    (req,) = fake_lightrag.calls("/graph/papers/citation")
    assert req["params"] == {"citing": "AAAA0001", "cited": "10.9/b"}
    lines = out.splitlines()
    assert lines[0] == "### Okafor 2010 cites Lind 2012"
    assert lines[1].startswith("A: Okafor 2010 · Alpha gravity leakage")
    assert lines[2].startswith("B: Lind 2012 · Bravo mascons")
    assert lines[3] == (
        "influential · intents: background, methodology · 3 citing sentence(s) · "
        "source: both"
    )
    assert lines[4] == '1. "Following Lind (2012) we map mascons."'
    assert lines[5].startswith('2. [background] "xxx') and lines[5].endswith('…"')
    assert len(lines[5]) < 640
    assert lines[6] == "Showing 2 of 3 sentences."


def test_a_citation_that_runs_the_other_way_says_so(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (
            200,
            {
                "citing": ME,
                "cited": OTHER,
                "edge": None,
                "contexts": [],
                "reason": "no_citation",
                "cited_cites_citing": True,
            },
        ),
    )
    out = load_server().citation_context("AAAA0001", "BBBB0002")
    assert "no citation from A to B" in out
    assert out.splitlines()[-1].startswith("B cites A: call citation_context")


def test_no_citation_the_other_way_adds_no_hint(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (
            200,
            {
                "citing": ME,
                "cited": OTHER,
                "edge": None,
                "contexts": [],
                "reason": "no_citation",
                "cited_cites_citing": False,
            },
        ),
    )
    assert "swapped" not in load_server().citation_context("AAAA0001", "BBBB0002")


def test_citation_context_reasons(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (200, {"reason": "not_in_structure", "missing": ["10.9/zzz"]}),
    )
    srv = load_server()
    assert srv.citation_context("AAAA0001", "10.9/zzz") == (
        'Not in the library\'s citation structure: "10.9/zzz". Check with find_paper.'
    )
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (
            200,
            {**CITATION, "contexts": [], "reason": "no_contexts_table"},
        ),
    )
    out = srv.citation_context("AAAA0001", "BBBB0002")
    assert "influential" in out
    assert (
        out.splitlines()[-1]
        == "This server's structure has no citation-sentence table."
    )
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (
            200,
            {**CITATION, "contexts": [], "edge": {**CITATION["edge"], "n_contexts": 0}},
        ),
    )
    assert srv.citation_context("AAAA0001", "BBBB0002").splitlines()[-1] == (
        "No citing sentence is stored for this citation."
    )


def test_an_edge_without_facts_still_prints(load_server, fake_lightrag):
    edge = {"source": None, "is_influential": None, "intents": [], "n_contexts": None}
    _route(
        fake_lightrag,
        "/graph/papers/citation",
        lambda p: (
            200,
            {**CITATION, "edge": edge, "contexts": [], "reason": "no_contexts_table"},
        ),
    )
    out = load_server().citation_context("AAAA0001", "BBBB0002")
    assert "No facts about this citation beyond that it exists." in out


SIMILAR = {
    "paper": ME,
    "in_kb": False,
    "similar": [
        dict(OTHER, similarity=0.93, links=["cites", "coauthor"]),
        _paper("doi:10.9/e", "Echo 2018", "Echo basins", similarity=0.8123, links=[]),
    ],
}


def test_similar_papers_sends_the_options_and_marks_unlinked_work(
    load_server, fake_lightrag
):
    _route(fake_lightrag, "/graph/papers/similar", lambda p: (200, SIMILAR))
    srv = load_server()
    out = srv.similar_papers("AAAA0001", limit=999, indexed_only=True)
    (req,) = fake_lightrag.calls("/graph/papers/similar")
    assert req["params"] == {"paper": "AAAA0001", "limit": "50", "in_kb": "true"}
    lines = out.splitlines()
    assert lines[0].startswith("### Similar to Okafor 2010")
    assert lines[1].startswith("1. 0.930 [cites and co-author] Lind 2012")
    assert lines[2].startswith("2. 0.812 [not linked] Echo 2018")
    srv.similar_papers("AAAA0001", limit=0)
    assert fake_lightrag.calls("/graph/papers/similar")[1]["params"] == {
        "paper": "AAAA0001",
        "limit": "1",
        "in_kb": "false",
    }


def test_similar_papers_reasons(load_server, fake_lightrag):
    srv = load_server()
    for reason, text in (
        ("no_embeddings", "has no paper embeddings"),
        ("no_embedding", "No embedding for Okafor 2010"),
    ):
        _route(
            fake_lightrag,
            "/graph/papers/similar",
            lambda p, r=reason: (200, {"paper": ME, "similar": [], "reason": r}),
        )
        assert text in srv.similar_papers("AAAA0001")
    _route(
        fake_lightrag,
        "/graph/papers/similar",
        lambda p: (200, {"paper": None, "reason": "not_in_structure"}),
    )
    assert "Check it with find_paper" in srv.similar_papers("10.9/zzz")
    _route(
        fake_lightrag,
        "/graph/papers/similar",
        lambda p: (200, {"paper": ME, "similar": []}),
    )
    assert srv.similar_papers("AAAA0001", indexed_only=True).splitlines()[-1] == (
        "No similar paper with indexed text."
    )


WORK = {
    "work_id": "oa:W2",
    "doi": "10.9/w2",
    "title": "Classic two",
    "year": 1990,
    "first_author": "Marple",
    "venue": "Old Journal",
    "n_library_citers": 3,
    "also_cited_by": [OTHER, dict(OTHER, label="Varga 2014")],
}


def test_outside_references_lists_works_with_their_citers(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/outside",
        lambda p: (
            200,
            {"paper": ME, "total": 9, "works": [WORK], "truncated": True},
        ),
    )
    srv = load_server()
    out = srv.outside_references("AAAA0001", limit=500)
    (req,) = fake_lightrag.calls("/graph/papers/outside")
    assert req["params"] == {"paper": "AAAA0001", "limit": "50"}
    lines = out.splitlines()
    assert lines[0].startswith("### Outside the library, cited by Okafor 2010")
    assert lines[1] == "9 works; showing 1, most cited by the library first."
    assert lines[2] == (
        "1. Marple 1990 · Classic two · Old Journal · doi:10.9/w2 · "
        "cited by 3 (Lind 2012, Varga 2014)"
    )
    assert lines[3] == "Raise limit (at most 50) for more."


def test_outside_references_reasons(load_server, fake_lightrag):
    srv = load_server()
    _route(
        fake_lightrag,
        "/graph/papers/outside",
        lambda p: (200, {"paper": ME, "works": [], "reason": "no_external_works"}),
    )
    assert "no outside-works tables" in srv.outside_references("AAAA0001")
    _route(
        fake_lightrag,
        "/graph/papers/outside",
        lambda p: (200, {"paper": None, "reason": "not_in_structure"}),
    )
    assert "Check it with find_paper" in srv.outside_references("10.9/zzz")
    _route(
        fake_lightrag,
        "/graph/papers/outside",
        lambda p: (200, {"paper": ME, "total": 0, "works": []}),
    )
    assert srv.outside_references("AAAA0001").splitlines()[-1] == (
        "It cites no outside work that two or more library papers cite "
        "(works only it cites are not stored)."
    )


def test_missing_works_without_and_with_a_topic(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/missing",
        lambda p: (
            200,
            {"min_citers": 2, "total": 1, "works": [WORK], "truncated": False},
        ),
    )
    srv = load_server()
    out = srv.missing_works(topic="  ", limit=0)
    (req,) = fake_lightrag.calls("/graph/papers/missing")
    assert req["params"] == {"limit": "1"}
    lines = out.splitlines()
    assert lines[0] == "### Cited by at least 2 papers in the library, not held"
    assert lines[1] == "1 works; showing 1."
    assert lines[2].startswith("1. Marple 1990 · Classic two")
    topic = {"topic_id": "topic:T1", "name": "Gravity Measurements", "level": "topic"}
    _route(
        fake_lightrag,
        "/graph/papers/missing",
        lambda p: (
            200,
            {
                "min_citers": 2,
                "topic": topic,
                "total": 1,
                "works": [dict(WORK, n_citers_in_topic=2)],
            },
        ),
    )
    out = srv.missing_works(topic="gravity", limit=500)
    assert fake_lightrag.calls("/graph/papers/missing")[1]["params"] == {
        "limit": "50",
        "topic": "gravity",
    }
    lines = out.splitlines()
    assert lines[0] == (
        "### Cited by at least 2 papers under Gravity Measurements (topic), not held"
    )
    assert "cited by 2 in the topic (Lind 2012, Varga 2014)" in lines[2]


def test_missing_works_reasons(load_server, fake_lightrag):
    srv = load_server()
    _route(
        fake_lightrag,
        "/graph/papers/missing",
        lambda p: (200, {"reason": "no_topic", "works": []}),
    )
    assert "topics_only=True" in srv.missing_works("basket weaving")
    _route(
        fake_lightrag,
        "/graph/papers/missing",
        lambda p: (200, {"reason": "no_external_works", "works": []}),
    )
    assert "no outside-works tables" in srv.missing_works()
    _route(
        fake_lightrag,
        "/graph/papers/missing",
        lambda p: (200, {"min_citers": 2, "total": 0, "works": []}),
    )
    assert srv.missing_works().splitlines()[-1] == "No such work."


@pytest.mark.parametrize(
    "call",
    [
        lambda srv: srv.citation_context("AAAA0001", "BBBB0002"),
        lambda srv: srv.similar_papers("AAAA0001"),
        lambda srv: srv.outside_references("AAAA0001"),
        lambda srv: srv.missing_works(),
    ],
)
def test_a_server_without_the_structure_is_named_not_raised(
    load_server, fake_lightrag, call
):
    detail = "Paper structure is not available: ZOTERO_STRUCT_SCHEMA is not set"
    for path in (
        "/graph/papers/citation",
        "/graph/papers/similar",
        "/graph/papers/outside",
        "/graph/papers/missing",
    ):
        _route(fake_lightrag, path, lambda p: (501, {"detail": detail}))
    assert call(load_server()).startswith(detail)


def test_an_older_server_without_the_routes_is_named(load_server, fake_lightrag):
    out = load_server().missing_works()  # the fake answers 404 for unknown routes
    assert out.startswith("This LightRAG server has no paper-structure routes")


def test_related_rows_show_influence_and_intents_once(load_server, fake_lightrag):
    both = {"is_influential": True, "intents": ["methodology"], "source": "s2"}
    plain = {"is_influential": False, "intents": [], "source": "openalex"}
    related = {
        "paper": ME,
        "totals": {"cites": 2, "papers": 2, "in_kb": 0},
        "related": [
            _paper(
                "doi:10.9/b",
                "Lind 2012",
                "Bravo",
                reasons=["cites", "cited_by"],
                citation=both,
            ),
            _paper(
                "doi:10.9/c", "Varga 2014", "Charlie", reasons=["cites"], citation=plain
            ),
            _paper("doi:10.9/d", "Gale 2016", "Delta", reasons=["cited_by"]),
        ],
    }
    _route(fake_lightrag, "/graph/papers/related", lambda p: (200, related))
    lines = load_server().related_papers("AAAA0001").splitlines()
    assert lines[2].startswith(
        "1. [cites (influential: methodology); cited by] Lind 2012"
    )
    assert lines[3].startswith("2. [cites] Varga 2014")
    assert lines[4].startswith("3. [cited by] Gale 2016")


def test_find_paper_rows_show_the_tldr_when_the_server_returns_one(load_server):
    srv = load_server()
    row = {"keys": ["AAAA0001"], "citation": "Okafor (2010). Alpha", "doi": "10.9/a"}
    assert "TLDR" not in srv._indexed_row(row)
    with_tldr = srv._indexed_row({**row, "tldr": "Alpha measures\ngravity."})
    assert with_tldr.endswith("· TLDR: Alpha measures gravity.")
    assert "\n" not in with_tldr
    long = srv._indexed_row({**row, "tldr": "w " * 300})
    assert len(long) < 400 and long.endswith("…")


def test_every_tool_is_listed_with_its_cost_and_limits(load_server):
    import anyio

    srv = load_server()
    tools = {t.name: t for t in anyio.run(srv.mcp.list_tools)}
    for name in (
        "citation_context",
        "similar_papers",
        "outside_references",
        "missing_works",
    ):
        assert name in tools
        assert "under a second" in tools[name].description
        assert (
            "at most" in tools[name].description or "Up to" in tools[name].description
        )
