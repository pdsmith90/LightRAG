"""Offline tests of the proxy's related_papers, papers_by_topic and connect_papers: the
requests they send to the server's paper-structure routes, how they word the answers,
and what they say when the server has no structure (501) or no such routes (404).

The server is the stdlib fake from conftest.py; every answer is invented.
"""

from __future__ import annotations

import pytest
from mcp.server.mcpserver.exceptions import ToolError


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


ME = _paper("doi:10.9/a", "Okafor 2010", "Alpha gravity leakage", "AAAA0001", "10.9/a")
RELATED = {
    "paper": ME,
    "relations": ["cites", "cited_by", "coupled", "coauthor"],
    "totals": {
        "cites": 2,
        "cited_by": 1,
        "coupled": 1,
        "coauthor": 1,
        "papers": 4,
        "in_kb": 1,
    },
    "related": [
        _paper(
            "doi:10.9/b",
            "Lind 2012",
            "Bravo mascons",
            "BBBB0002",
            "10.9/b",
            reasons=["cites", "coauthor"],
            shared_authors=["Ben Lind"],
            score=1.25,
        ),
        _paper(
            "doi:10.9/c",
            "Varga 2014",
            "Charlie filters",
            reasons=["cited_by", "coupled"],
            shared_refs=3,
            coupling_score=0.1,
            score=1.2,
        ),
    ],
    "truncated": True,
}


def _route(fake, path, answer):
    fake.routes[("GET", path)] = lambda params: answer(params)


def test_related_papers_sends_the_paper_and_words_the_rows(load_server, fake_lightrag):
    _route(fake_lightrag, "/graph/papers/related", lambda p: (200, RELATED))
    srv = load_server()
    out = srv.related_papers("AAAA0001", relations="cites, coupled", limit=500)
    (req,) = fake_lightrag.calls("/graph/papers/related")
    assert req["params"] == {
        "paper": "AAAA0001",
        "limit": "50",
        "relations": "cites,coupled",
    }
    assert req["headers"]["x-api-key"] == "test-key-not-a-secret"
    lines = out.splitlines()
    assert lines[0].startswith("### Related to Okafor 2010 · Alpha gravity leakage")
    assert "indexed as AAAA0001" in lines[0]
    assert lines[1] == (
        "cites 2 · cited by 1 · coupled 1 · co-author 1 → 4 papers, 1 indexed."
    )
    assert lines[2].startswith(
        "1. [cites; co-author: Ben Lind] Lind 2012 · Bravo mascons"
    )
    assert "indexed as BBBB0002" in lines[2]
    assert lines[3].startswith(
        "2. [cited by; coupled: 3 shared refs, score 0.1] Varga 2014 · Charlie filters"
    )
    assert "not indexed (Zotero ZZZZ0009)" in lines[3]
    assert lines[4].startswith("Showing 2 of 4")


def test_related_papers_rejects_an_unknown_relation_without_asking(
    load_server, fake_lightrag
):
    srv = load_server()
    out = srv.related_papers("AAAA0001", relations="cites,friends")
    assert out.startswith("Unknown relation(s): friends.")
    assert fake_lightrag.calls("/graph/papers/related") == []


def test_words_are_looked_up_and_the_dois_passed(load_server, fake_lightrag):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda p: (
        200,
        [
            {
                "paper": "10.48550/arxiv.1706.03762",
                "doi": "10.48550/arXiv.1706.03762",
                "keys": ["AAAA0001"],
                "citation": "Vaswani (2017). Attention Is All You Need",
            }
        ],
        {"X-Paper-Search": "2"},
    )
    _route(fake_lightrag, "/graph/papers/related", lambda p: (200, RELATED))
    srv = load_server()
    srv.related_papers("Vaswani 2017 attention")
    (req,) = fake_lightrag.calls("/graph/papers/related")
    assert req["params"]["paper"] == "10.48550/arxiv.1706.03762"


@pytest.mark.parametrize(
    "call",
    [
        lambda srv: srv.related_papers("AAAA0001"),
        lambda srv: srv.papers_by_topic("geophysics"),
        lambda srv: srv.papers_by_topic("geo", topics_only=True),
        lambda srv: srv.connect_papers("AAAA0001", "BBBB0002"),
    ],
)
def test_a_server_without_the_structure_is_named_not_raised(
    load_server, fake_lightrag, call
):
    detail = "Paper structure is not available: ZOTERO_STRUCT_SCHEMA is not set"
    for path in (
        "/graph/papers/related",
        "/graph/papers/topics",
        "/graph/papers/by_topic",
        "/graph/papers/connect",
    ):
        _route(fake_lightrag, path, lambda p: (501, {"detail": detail}))
    srv = load_server()
    out = call(srv)
    assert out.startswith(detail)
    assert "find_paper, paper_passages and search_library still work" in out


def test_an_older_server_without_the_routes_is_named(load_server, fake_lightrag):
    srv = load_server()  # the fake answers 404 Not Found for unknown routes
    out = srv.connect_papers("AAAA0001", "BBBB0002")
    assert out.startswith("This LightRAG server has no paper-structure routes")


def test_a_server_error_is_a_tool_error(load_server, fake_lightrag):
    _route(fake_lightrag, "/graph/papers/related", lambda p: (500, {"detail": "boom"}))
    srv = load_server()
    with pytest.raises(
        ToolError, match=r"(?s)/graph/papers/related failed.*Server response: boom"
    ):
        srv.related_papers("AAAA0001")


def test_a_paper_outside_the_structure_points_to_find_paper(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/related",
        lambda p: (200, {"paper": None, "reason": "not_in_structure"}),
    )
    srv = load_server()
    assert "Check it with find_paper" in srv.related_papers("10.9/zzz")


TOPIC = {
    "topic_id": "subfield:1908",
    "name": "Geophysics",
    "level": "subfield",
    "path": "Physical Sciences > Earth and Planetary Sciences > Geophysics",
    "n_papers": 40,
    "n_primary": 30,
}


def test_papers_by_topic_lists_papers_and_other_matches(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/by_topic",
        lambda p: (
            200,
            {
                "topic": TOPIC,
                "alternatives": [
                    dict(
                        TOPIC,
                        topic_id="topic:T9",
                        name="Geophysics of ice",
                        level="topic",
                    )
                ],
                "primary_only": True,
                "total": 30,
                "in_kb": 25,
                "papers": [
                    dict(ME, primary=True),
                    _paper("doi:10.9/c", "Varga 2014", "Charlie", primary=False),
                ],
                "truncated": True,
            },
        ),
    )
    srv = load_server()
    out = srv.papers_by_topic("geophysics", limit=2, primary_only=True)
    (req,) = fake_lightrag.calls("/graph/papers/by_topic")
    assert req["params"] == {
        "topic": "geophysics",
        "limit": "2",
        "primary_only": "true",
    }
    lines = out.splitlines()
    assert lines[0] == (
        "### Geophysics (subfield; Physical Sciences > Earth and Planetary Sciences > "
        "Geophysics; subfield:1908)"
    )
    assert lines[1] == "30 papers with it as primary topic, 25 indexed."
    assert lines[2].startswith("1. Okafor 2010")
    assert lines[3].startswith("2. [secondary topic] Varga 2014")
    assert (
        lines[-1]
        == "Other topics matching the words: Geophysics of ice (topic, topic:T9)"
    )


def test_topics_only_lists_topics_with_counts(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/topics",
        lambda p: (200, {"total": 1, "topics": [TOPIC]}),
    )
    srv = load_server()
    out = srv.papers_by_topic("geo", topics_only=True)
    (req,) = fake_lightrag.calls("/graph/papers/topics")
    assert req["params"] == {"q": "geo", "limit": "15"}
    assert out.splitlines()[1] == (
        "- subfield:1908 · subfield · Physical Sciences > Earth and Planetary "
        "Sciences > Geophysics · 30 papers with it as primary topic, 40 with it at all"
    )


def test_no_topic_and_an_empty_topic_are_answered(load_server, fake_lightrag):
    _route(
        fake_lightrag, "/graph/papers/by_topic", lambda p: (200, {"reason": "no_topic"})
    )
    srv = load_server()
    assert "topics_only=True" in srv.papers_by_topic("basket weaving")
    assert srv.papers_by_topic("  ").startswith("Name a topic")


CONNECT = {
    "a": ME,
    "b": _paper("doi:10.9/e", "Okafor 2018", "Echo basins", doi="10.9/e"),
    "links": {
        "relation": ["cited_by"],
        "shared_authors": ["Ada Okafor"],
        "same_topic": None,
        "same_subfield": "Geophysics",
        "same_field": "Earth and Planetary Sciences",
        "cited_by_both": 2,
        "cite_both": 5,
        "shared_refs": 4,
        "coupling_score": 0.3,
    },
    "paths": [
        {
            "hops": 1,
            "cost": 1.0,
            "papers": [],
            "steps": [
                {
                    "from": "Okafor 2010",
                    "relation": ["cited_by", "coupled"],
                    "to": "Okafor 2018",
                }
            ],
        },
        {
            "hops": 2,
            "cost": 4.2,
            "papers": [_paper("doi:10.9/c", "Varga 2014", "Charlie", "CCCC0003")],
            "steps": [
                {"from": "Okafor 2010", "relation": ["cites"], "to": "Varga 2014"},
                {"from": "Varga 2014", "relation": ["cited_by"], "to": "Okafor 2018"},
            ],
        },
    ],
}


def test_connect_papers_words_links_and_routes(load_server, fake_lightrag):
    _route(fake_lightrag, "/graph/papers/connect", lambda p: (200, CONNECT))
    srv = load_server()
    out = srv.connect_papers("AAAA0001", "10.9/e", paths=9)
    (req,) = fake_lightrag.calls("/graph/papers/connect")
    assert req["params"] == {"a": "AAAA0001", "b": "10.9/e", "paths": "5"}
    lines = out.splitlines()
    assert lines[0].startswith("### A: Okafor 2010")
    assert lines[1].startswith("### B: Okafor 2018 · Echo basins")
    assert lines[2] == (
        "Direct: B cites A · 4 shared references (coupling score 0.3) · shared authors: "
        "Ada Okafor · same subfield: Geophysics · 2 library papers cite both; both cite "
        "5 of the same library papers"
    )
    assert lines[4] == "1. (1 step) Okafor 2010 —cited by and coupled→ Okafor 2018"
    assert lines[5] == (
        "2. (2 steps) Okafor 2010 —cites→ Varga 2014 —cited by→ Okafor 2018"
    )
    assert lines[6].startswith("   via Varga 2014 · Charlie")


def test_connect_papers_names_what_is_missing(load_server, fake_lightrag):
    _route(
        fake_lightrag,
        "/graph/papers/connect",
        lambda p: (
            200,
            {"reason": "not_in_structure", "missing": ["10.9/zzz"], "paths": []},
        ),
    )
    srv = load_server()
    assert srv.connect_papers("AAAA0001", "10.9/zzz") == (
        'Not in the library\'s citation structure: "10.9/zzz". Check with find_paper.'
    )


def test_connect_papers_points_to_the_common_view_for_shared_entities(load_server):
    import anyio

    srv = load_server()
    tools = {t.name: t for t in anyio.run(srv.mcp.list_tools)}
    assert "mode=common" in tools["connect_papers"].description
