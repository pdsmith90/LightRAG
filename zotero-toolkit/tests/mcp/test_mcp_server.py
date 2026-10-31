"""Offline tests for mcp/lightrag_mcp_server.py against a fake LightRAG server."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

VASWANI = (
    "Ashish Vaswani, Noam Shazeer, Niki Parmar. (2017). Attention Is All You "
    "Need. Advances in Neural Information Processing Systems. "
    "doi:10.48550/arXiv.1706.03762 [zotero://select/library/items/AAAA0001]"
)
HE = (
    "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual "
    "Learning for Image Recognition. doi:10.1109/CVPR.2016.90 "
    "[zotero://select/library/items/BBBB0002]"
)

# Server order is deliberately not sorted by id or name: the proxy must keep it.
REFERENCES = [
    {"reference_id": "1", "file_path": "BBBB0002__Deep_Residual_Learning.md"},
    {"reference_id": "2", "file_path": "AAAA0001__Attention_Is_All_You_Need.md"},
    {"reference_id": "3", "file_path": "CCCC0003__Unindexed_Draft__v2.md"},
    {"reference_id": "4", "file_path": "meeting notes.txt"},
    {"reference_id": "5", "file_path": "/data/inputs/__parsed__/AAAA0001__copy.md"},
]
COMPILED = [
    f"- [1] {HE}",
    f"- [2] {VASWANI}",
    "- [3] Unindexed Draft v2 [zotero://select/library/items/CCCC0003]",
    "- [4] meeting notes.txt",
    f"- [5] {VASWANI}",
]


def _answer_route(response, references):
    return lambda body: (200, {"response": response, "references": references})


# --- citation compilation -------------------------------------------------------


def test_cite_resolves_key_from_metadata(load_server):
    srv = load_server()
    assert srv._cite("AAAA0001__Attention_Is_All_You_Need.md") == VASWANI
    # Empty publication is skipped, not rendered as an empty part.
    assert srv._cite("BBBB0002__whatever.md") == HE


def test_cite_renders_creators_as_family_comma_given(load_server, metadata_file):
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    data["AAAA0001"]["creators"] = [
        ["Vaswani", "Ashish"],
        ["Shazeer", "Noam"],
        ["WHO", ""],
    ]
    metadata_file.write_text(json.dumps(data), encoding="utf-8")
    srv = load_server()
    assert srv._cite("AAAA0001__x.md") == VASWANI.replace(
        "Ashish Vaswani, Noam Shazeer, Niki Parmar.",
        "Vaswani, Ashish; Shazeer, Noam; WHO.",
    )
    assert srv._cite("BBBB0002__x.md") == HE  # no creators: authors verbatim


def test_context_citations_use_creators_when_present(
    fake_lightrag, load_server, metadata_file
):
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    data["BBBB0002"]["creators"] = [["He", "Kaiming"], ["Sun", "Jian"]]
    metadata_file.write_text(json.dumps(data), encoding="utf-8")
    fake_lightrag.routes[("POST", "/query")] = _answer_route("CTX", REFERENCES[:1])
    out = load_server().search_library("q")
    assert "- [1] He, Kaiming; Sun, Jian. (2016). Deep Residual" in out


def test_cite_unknown_key_falls_back_to_deslugged_title(load_server):
    srv = load_server()
    assert (
        srv._cite("ZZZZ9999__Some_Paper___Title.md")
        == "Some Paper Title [zotero://select/library/items/ZZZZ9999]"
    )


def test_cite_unresolvable_file_is_bare_filename(load_server):
    srv = load_server()
    assert srv._cite("notes/meeting notes.txt") == "meeting notes.txt"
    assert srv._cite("abcd1234__lowercase_key.md") == "abcd1234__lowercase_key.md"
    assert srv._cite("") == "(untitled source)"
    assert srv._cite(None) == "(untitled source)"


def test_cite_uses_basename_of_nested_paths(load_server):
    srv = load_server()
    assert srv._cite("/srv/inputs/__parsed__/AAAA0001__x.md") == VASWANI


def test_citation_metadata_refreshes_after_atomic_replacement(
    load_server, metadata_file
):
    srv = load_server()
    path = "AAAA0001__Attention_Is_All_You_Need.md"
    assert srv._cite(path) == VASWANI

    updated = json.loads(metadata_file.read_text(encoding="utf-8"))
    updated["AAAA0001"]["title"] = "Attention Revisited"
    replacement = metadata_file.with_suffix(".tmp")
    replacement.write_text(json.dumps(updated), encoding="utf-8")
    os.replace(replacement, metadata_file)
    assert srv._cite(path) == VASWANI.replace(
        "Attention Is All You Need", "Attention Revisited"
    )

    metadata_file.write_text("{bad json", encoding="utf-8")
    assert srv._cite(path) == VASWANI.replace(
        "Attention Is All You Need", "Attention Revisited"
    )
    updated["AAAA0001"]["year"] = "2018"
    replacement.write_text(json.dumps(updated), encoding="utf-8")
    os.replace(replacement, metadata_file)
    assert "(2018)" in srv._cite(path)


def test_citation_metadata_appearing_after_start_is_used(
    load_server, tmp_path, metadata_file
):
    late = tmp_path / "later.json"
    srv = load_server(ZOTERO_METADATA=str(late))
    assert "Attention Is All You Need" not in srv._cite("AAAA0001__x.md")
    late.write_text(metadata_file.read_text(encoding="utf-8"), encoding="utf-8")
    assert srv._cite("AAAA0001__x.md") == VASWANI


def test_server_citation_wins_over_unsynced_local_metadata(fake_lightrag, load_server):
    ref = {
        "reference_id": "1",
        "file_path": "AAAA0001__Attention_Is_All_You_Need.md",
        "citation": "Vaswani Ashish. (2018). Corrected by Zotero",
    }
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", [ref])
    srv = load_server()
    assert srv.search_library("attention").startswith(
        "### Retrieved documents (compiled citations)\n"
        "- [1] Vaswani Ashish. (2018). Corrected by Zotero "
        "[zotero://select/library/items/AAAA0001]"
    )


def test_missing_metadata_degrades_and_warns_on_stderr(load_server, tmp_path, capsys):
    srv = load_server(ZOTERO_METADATA=str(tmp_path / "absent.json"))
    assert srv._META == {}
    assert (
        srv._cite("AAAA0001__Attention_Is_All_You_Need.md")
        == "Attention Is All You Need [zotero://select/library/items/AAAA0001]"
    )
    captured = capsys.readouterr()
    assert "no citation metadata loaded" in captured.err
    assert captured.out == ""


def test_metadata_path_expands_user(load_server, monkeypatch, tmp_path, metadata_file):
    monkeypatch.setenv("HOME", str(metadata_file.parent))
    srv = load_server(ZOTERO_METADATA="~/zotero_metadata.json")
    assert "AAAA0001" in srv._META


@pytest.mark.parametrize("unset", [None, ""])
def test_default_metadata_path_matches_the_corpus_builder(
    load_server, monkeypatch, tmp_path, metadata_file, unset
):
    # Unset (or empty) ZOTERO_METADATA means ~/Zotero/zotero_metadata.json, the file the
    # corpus builder writes by default, wherever the client launches the server from.
    (tmp_path / "Zotero").mkdir()
    (tmp_path / "Zotero" / "zotero_metadata.json").write_text(metadata_file.read_text())
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("HOME", str(tmp_path))
    srv = load_server(ZOTERO_METADATA=unset)
    assert "AAAA0001" in srv._META


def test_context_lists_every_source_in_server_order_before_context(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", REFERENCES)
    srv = load_server()
    out = srv.search_library("attention")
    assert out == (
        "### Retrieved documents (compiled citations)\n"
        + "\n".join(COMPILED)
        + "\n\nRAW CONTEXT"
    )
    body = fake_lightrag.calls("/query")[0]["json"]
    assert body == {
        "query": "attention",
        "mode": "mix",
        "top_k": 10,
        "only_need_context": True,
    }


def test_context_without_references_is_just_the_context(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", [])
    srv = load_server()
    assert srv.search_library("q") == "RAW CONTEXT"
    assert fake_lightrag.calls("/query/data") == []


GRAPH = (
    "Knowledge Graph Data (Entity):\n\n```json\n"
    '{"entity": "Mar.", "type": "concept", "description": "The month of March."}\n'
    "```\n\nKnowledge Graph Data (Relationship):\n\n```json\n"
    '{"entity1": "ResNet", "entity2": "ImageNet", "description": "evaluated on"}\n'
    "```\n\n"
)


def _context_text(chunks, graph=GRAPH):
    """A context-only response in stock LightRAG's layout."""
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    return (
        f"\n{graph}Document Chunks (Each entry has a reference_id refer to the "
        f"`Reference Document List`):\n\n```json\n{lines}\n```\n\n"
        "Reference Document List (Each entry starts with a [reference_id]):\n\n"
        "```\n[1] Deep Residual Learning\n[2] Attention Is All You Need\n```\n\n"
    )


CHUNKS = [
    {
        "reference_id": "1",
        "content": "Residual learning eases “very deep” training.",
        "content_headings": "Deep Residual Learning → 1 Introduction",
    },
    {"reference_id": "2", "content": "Attention is all you need.\n"},
    {"reference_id": "1", "content": "Identity shortcuts add no parameters."},
]


def test_context_renders_citations_then_labelled_passages_without_graph(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(CHUNKS), REFERENCES[:2]
    )
    out = load_server().search_library("q")
    assert out == (
        "### Retrieved documents (compiled citations)\n"
        + "\n".join(COMPILED[:2])
        + "\n\n### Passages (the server's ranked order; [n] is the document listed above)"
        "\n\n[1] Kaiming He 2016 · Deep Residual Learning → 1 Introduction\n"
        "Residual learning eases “very deep” training."
        "\n\n[2] Ashish Vaswani 2017\nAttention is all you need."
        "\n\n[1] Kaiming He 2016\nIdentity shortcuts add no parameters."
    )
    # The graph, the chunk JSON and the server's own reference list are gone.
    for gone in ("Knowledge Graph", "Mar.", '{"reference_id"', "Reference Document"):
        assert gone not in out


def test_passage_labels_prefer_creators_then_the_server_citation(
    fake_lightrag, load_server, metadata_file
):
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    data["BBBB0002"]["creators"] = [["He", "Kaiming"]]
    del data["AAAA0001"]
    metadata_file.write_text(json.dumps(data), encoding="utf-8")
    refs = [
        {"reference_id": "1", "file_path": "BBBB0002__x.md"},
        {
            "reference_id": "2",
            "file_path": "AAAA0001__x.md",
            "citation": "Vaswani, Ashish; Shazeer, Noam. (2017). Attention.",
        },
        {"reference_id": "3", "file_path": "meeting notes.txt"},
    ]
    chunks = [{"reference_id": n, "content": "t"} for n in ("1", "2", "3")]
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(chunks), refs
    )
    out = load_server().search_library("q")
    assert "[1] He 2016\nt" in out
    assert "[2] Vaswani 2017\nt" in out
    assert "[3] meeting notes.txt\nt" in out


def test_include_graph_appends_entities_and_relations(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(CHUNKS), REFERENCES[:2]
    )
    out = load_server().search_library("q", include_graph=True)
    assert out.endswith(GRAPH.strip())
    assert out.index("### Passages") < out.index("Knowledge Graph Data (Entity)")


def test_context_leaves_out_lowest_ranked_passages_past_the_cap(
    fake_lightrag, load_server
):
    chunks = [
        {"reference_id": "1", "content": f"passage {i} " + "x" * 400} for i in range(10)
    ]
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(chunks), REFERENCES[:1]
    )
    out = load_server(LIGHTRAG_MCP_MAX_CHARS="2500").search_library("q")
    assert len(out) <= 2500
    kept = re.findall(r"passage (\d) ", out)
    assert kept == [str(i) for i in range(len(kept))] and 0 < len(kept) < 10
    assert f"[{10 - len(kept)} lower-ranked passage(s) left out" in out


def test_unknown_context_format_passes_through_whole(fake_lightrag, load_server):
    ctx = "Some server's free-form context\n" + "y" * 60000
    fake_lightrag.routes[("POST", "/query")] = _answer_route(ctx, REFERENCES[:1])
    out = load_server().search_library("q")
    assert (
        out == f"### Retrieved documents (compiled citations)\n{COMPILED[0]}\n\n{ctx}"
    )


# --- paper lookup and one paper's passages ------------------------------------------

VASWANI_PAPER = {
    "paper": "10.48550/arxiv.1706.03762",
    "doi": "10.48550/arxiv.1706.03762",
    "keys": ["AAAA0001"],
    "title": "Attention Is All You Need",
    "citation": "Vaswani, Ashish; Shazeer, Noam et al. (2017). Attention Is All You Need",
    "year": "2017",
}
HE_PAPER = {
    "paper": "10.1109/cvpr.2016.90",
    "doi": "10.1109/cvpr.2016.90",
    "keys": ["BBBB0002"],
    "title": "Deep Residual Learning for Image Recognition",
    "citation": "He, Kaiming; Zhang, Xiangyu et al. (2016). Deep Residual Learning",
    "year": "2016",
}


def _papers_route(index):
    """The fork's /graph/papers/search over the given indexed papers: an exact key
    or DOI, else every word in the title, citation or year."""

    def route(params):
        q = params.get("q", "")
        exact = [p for p in index if q.upper() in p["keys"] or q.lower() == p["doi"]]
        words = q.lower().split()
        loose = [
            p
            for p in index
            if all(
                w in f"{p['title']} {p['citation']} {p['year']}".lower() for w in words
            )
        ]
        return 200, exact or loose

    return route


def test_find_paper_separates_indexed_from_zotero_only(fake_lightrag, load_server):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _papers_route(
        [VASWANI_PAPER]
    )
    out = load_server().find_paper("learning")  # only He et al. has it in the title
    indexed, zotero = out.split("In Zotero but NOT indexed")
    assert "AAAA0001" not in indexed and "BBBB0002" not in indexed
    assert "(none)" in indexed
    assert "- BBBB0002 · Kaiming He et al. (2016). Deep Residual Learning" in zotero
    out = load_server().find_paper("10.48550/arXiv.1706.03762")
    indexed, zotero = out.split("In Zotero but NOT indexed")
    assert "- AAAA0001 · Vaswani, Ashish; Shazeer, Noam et al. (2017)." in indexed
    assert "(none)" in zotero
    lookups = fake_lightrag.calls("/graph/papers/search")
    assert lookups[0]["params"] == {"q": "learning", "limit": "100"}
    # BBBB0002 was asked for by its exact key before being called not indexed.
    assert {"q": "BBBB0002", "limit": "100"} in [r["params"] for r in lookups]


def test_find_paper_moves_items_the_exact_key_lookup_knows(fake_lightrag, load_server):
    def route(params):
        return 200, [HE_PAPER] if params["q"] == "BBBB0002" else []

    fake_lightrag.routes[("GET", "/graph/papers/search")] = route
    out = load_server().find_paper("deep residual")
    indexed, zotero = out.split("In Zotero but NOT indexed")
    assert "- BBBB0002 · He, Kaiming" in indexed and "(none)" in zotero


def test_find_paper_without_a_match_says_absent(fake_lightrag, load_server):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda params: (200, [])
    out = load_server().find_paper("Brashears 2016")
    assert "Neither the index nor the corpus metadata" in out


def test_find_paper_on_a_stock_server_reports_the_index_unknown(
    fake_lightrag, load_server
):
    out = load_server().find_paper("attention")  # no route: the fake answers 404
    assert "Index status unknown" in out
    assert "- AAAA0001 · Ashish Vaswani et al. (2017). Attention Is All You Need" in out


def test_find_paper_with_too_many_indexed_matches_lists_no_zotero_items(
    fake_lightrag, load_server
):
    many = [dict(VASWANI_PAPER, paper=f"p{i}", keys=[f"K{i:07d}"]) for i in range(100)]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda params: (200, many)
    out = load_server().find_paper("attention")
    assert "100 or more indexed papers match" in out
    assert "NOT indexed" not in out and "… and 90 more" in out
    assert len(fake_lightrag.calls("/graph/papers/search")) == 1


def test_paper_lookup_errors_become_tool_errors(fake_lightrag, load_server):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda params: (
        500,
        {"detail": "doc status unavailable"},
    )
    with pytest.raises(ToolError, match="doc status unavailable"):
        load_server().find_paper("attention")


def test_paper_passages_returns_only_that_papers_passages(fake_lightrag, load_server):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _papers_route(
        [VASWANI_PAPER, HE_PAPER]
    )
    refs = [
        {
            "reference_id": "1",
            "file_path": "BBBB0002__Deep.md",
            "citation": HE_PAPER["citation"],
        },
        {
            "reference_id": "2",
            "file_path": "AAAA0001__Attention.md",
            "citation": VASWANI_PAPER["citation"],
        },
    ]
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(CHUNKS), refs
    )
    out = load_server().paper_passages("BBBB0002", "do shortcuts add parameters")
    assert out.startswith(
        f"### {HE_PAPER['citation']} [zotero://select/library/items/BBBB0002]"
    )
    assert (
        "Residual learning eases" in out
        and "Identity shortcuts add no parameters." in out
    )
    assert "Attention is all you need." not in out
    assert out.endswith(
        "### Other papers the question matched (no passages shown)\n"
        f"- [2] {VASWANI_PAPER['citation']} [zotero://select/library/items/AAAA0001]"
    )
    body = fake_lightrag.calls("/query")[0]["json"]
    assert body == {
        "query": "He 2016 Deep Residual Learning for Image Recognition: "
        "do shortcuts add parameters",
        "mode": "naive",
        "top_k": 10,
        "only_need_context": True,
    }


def test_paper_passages_with_several_matches_lists_them_and_does_not_search(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _papers_route(
        [VASWANI_PAPER, HE_PAPER]
    )
    out = load_server().paper_passages("al.", "q")  # both citations contain "al."
    assert "needs exactly one indexed paper" in out
    assert "AAAA0001" in out and "BBBB0002" in out
    assert fake_lightrag.calls("/query") == []


def test_paper_passages_for_an_unindexed_paper_says_so_and_does_not_search(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _papers_route(
        [VASWANI_PAPER]
    )
    out = load_server().paper_passages("residual", "q")
    assert (
        "- BBBB0002 · Kaiming He et al. (2016)"
        in out.split("In Zotero but NOT indexed")[1]
    )
    assert "needs exactly one" not in out
    assert fake_lightrag.calls("/query") == []


def test_paper_passages_without_a_passage_of_the_paper_says_so(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _papers_route(
        [VASWANI_PAPER, HE_PAPER]
    )
    refs = [{"reference_id": "1", "file_path": "BBBB0002__Deep.md"}]
    chunks = [{"reference_id": "1", "content": "Residual."}]
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(chunks), refs
    )
    out = load_server().paper_passages("AAAA0001", "q")
    assert "No passage of this paper ranked for the question" in out
    assert "Residual." not in out and "- [1] Kaiming He" in out


def test_paper_passages_on_a_stock_server_resolves_from_metadata(
    fake_lightrag, load_server
):
    refs = [{"reference_id": "1", "file_path": "AAAA0001__x.md"}]
    chunks = [{"reference_id": "1", "content": "Attention."}]
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        _context_text(chunks), refs
    )
    out = load_server().paper_passages("10.48550/arXiv.1706.03762", "q")
    assert out.startswith(f"### {VASWANI}")
    assert "[1] Ashish Vaswani 2017\nAttention." in out
    query = fake_lightrag.calls("/query")[0]["json"]["query"]
    assert query == "Ashish Vaswani 2017 Attention Is All You Need: q"


def test_answer_replaces_invented_references_with_compiled_sources(
    fake_lightrag, load_server
):
    llm = (
        "Transformers rely on attention [1]. Residual links ease training [2].\n\n"
        "### References\n\n"
        "- [1] Document Title One (2019). Invented Journal.\n"
        "- [2] Document Title Two\n"
        "[3] Another invented entry\n"
    )
    fake_lightrag.routes[("POST", "/query")] = _answer_route(llm, REFERENCES[:2])
    srv = load_server()
    out = srv.search_library("q", answer=True)
    assert out == (
        "Transformers rely on attention [1]. Residual links ease training [2].\n\n"
        "### Sources (compiled from retrieval)\n"
        f"- [1] {HE}\n- [2] {VASWANI}"
    )
    body = fake_lightrag.calls("/query")[0]["json"]
    assert "only_need_context" not in body
    assert "Do not generate a References section" in body["user_prompt"]
    assert fake_lightrag.calls("/query/data") == []


def test_answer_strips_server_sources_block_to_avoid_duplicates(
    fake_lightrag, load_server
):
    llm = "Answer text [1].\n\n### Sources\n- [1] Server-rendered citation\n"
    fake_lightrag.routes[("POST", "/query")] = _answer_route(llm, REFERENCES[:1])
    srv = load_server()
    out = srv.search_library("q", answer=True)
    assert out.count("### Sources") == 1
    assert "Server-rendered citation" not in out
    assert out.endswith(f"### Sources (compiled from retrieval)\n- [1] {HE}")


def test_answer_without_references_falls_back_to_query_data_once(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("Cached answer.", [])
    fake_lightrag.routes[("POST", "/query/data")] = lambda body: (
        200,
        {"status": "success", "data": {"references": REFERENCES[1:2]}},
    )
    srv = load_server()
    out = srv.search_library("q", mode="hybrid", top_k=7, answer=True)
    assert out == (
        f"Cached answer.\n\n### Sources (compiled from retrieval)\n- [2] {VASWANI}"
    )
    data_calls = fake_lightrag.calls("/query/data")
    assert [c["json"] for c in data_calls] == [
        {"query": "q", "mode": "hybrid", "top_k": 7}
    ]
    # An identical query in the same session reuses the memoised references.
    assert srv.search_library("q", mode="hybrid", top_k=7, answer=True) == out
    assert len(fake_lightrag.calls("/query/data")) == 1


def test_answer_fallback_failure_returns_answer_without_sources(
    fake_lightrag, load_server
):
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        "Answer.\n\n### References\n- [1] Invented", []
    )
    fake_lightrag.routes[("POST", "/query/data")] = lambda body: (
        500,
        {"detail": "boom"},
    )
    srv = load_server()
    assert srv.search_library("q", answer=True) == "Answer."


# --- invented-reference stripping ------------------------------------------------


def test_strip_keeps_sources_of_error_prose(load_server):
    srv = load_server()
    text = (
        "Intro.\n\n### Sources of error\n\n"
        "The dominant sources of error are sampling bias and measurement noise.\n\n"
        "References to prior work appear inline."
    )
    assert srv._strip_llm_refs(text) == text


@pytest.mark.parametrize(
    "heading",
    [
        "### References",
        "## Sources",
        "References:",
        "sources",
        "#### REFERENCES :",
    ],
)
def test_strip_removes_heading_variants_and_entries(load_server, heading):
    srv = load_server()
    text = f"Body [1].\n\n{heading}\n\n- [1] Fake A\n* [2] Fake B\n[3] Fake C\n"
    assert srv._strip_llm_refs(text) == "Body [1]."


def test_strip_resumes_prose_after_the_reference_block(load_server):
    srv = load_server()
    text = "Body.\n### References\n- [1] Fake\nClosing remark after the list."
    assert srv._strip_llm_refs(text) == "Body.\nClosing remark after the list."


# --- request shaping: mode, top_k, auth ------------------------------------------


@pytest.mark.parametrize("mode", ["mix", "hybrid", "local", "global", "naive"])
def test_valid_modes_pass_through(fake_lightrag, load_server, mode):
    srv = load_server()
    srv.search_library("q", mode=mode)
    srv.search_library("q", mode=mode, answer=True)
    assert [c["json"]["mode"] for c in fake_lightrag.calls("/query")] == [mode, mode]


@pytest.mark.parametrize("mode", ["bypass", "graph", "", "MIXED"])
def test_invalid_modes_fall_back_to_mix(fake_lightrag, load_server, mode):
    srv = load_server()
    srv.search_library("q", mode=mode)
    srv.search_library("q", mode=mode, answer=True)
    assert {c["json"]["mode"] for c in fake_lightrag.calls("/query")} == {"mix"}


@pytest.mark.parametrize(
    "requested,sent", [(10, 10), (40, 40), (41, 40), (5000, 40), (0, 1), (-3, 1)]
)
def test_top_k_is_clamped_by_default(fake_lightrag, load_server, requested, sent):
    srv = load_server()
    srv.search_library("q", top_k=requested)
    srv.search_library("q", top_k=requested, answer=True)
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query")] == [sent] * 2
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query/data")] == [sent]


def test_top_k_cap_is_configurable(fake_lightrag, load_server):
    srv = load_server(LIGHTRAG_MCP_MAX_TOP_K="15")
    srv.search_library("q", top_k=40)
    srv.search_library("q", top_k=12)
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query")] == [15, 12]


def test_api_key_travels_in_header_never_in_url(fake_lightrag, load_server, api_key):
    srv = load_server()
    srv.search_library("q")
    srv.search_library("q2", answer=True)  # no references -> /query/data too
    srv.find_paper("Vaswani 2017")
    srv.paper_passages("AAAA0001", "q")
    srv.add_note("text")
    assert srv.health().startswith("200:")
    paths = {r["path"] for r in fake_lightrag.requests}
    assert paths == {
        "/query",
        "/query/data",
        "/graph/papers/search",
        "/documents/text",
        "/health",
    }
    for r in fake_lightrag.requests:
        assert r["headers"].get("x-api-key") == api_key
        assert api_key not in r["url"]
        assert api_key not in str(r["json"])
    # Only the paper lookup carries a query string, and it holds the lookup words.
    assert {r["path"] for r in fake_lightrag.requests if "?" in r["url"]} == {
        "/graph/papers/search"
    }


def test_no_api_key_sends_no_header(fake_lightrag, load_server):
    srv = load_server(LIGHTRAG_API_KEY=None)
    srv.search_library("q")
    assert "x-api-key" not in fake_lightrag.requests[0]["headers"]


def test_base_url_trailing_slash_is_ignored(fake_lightrag, load_server):
    srv = load_server(LIGHTRAG_BASE_URL=fake_lightrag.url + "/")
    srv.search_library("q")
    assert fake_lightrag.requests[0]["path"] == "/query"


# --- other tools --------------------------------------------------------------------

NOTE_NAME = r"note-\d{8}T\d{6}Z-[0-9a-f]{8}"


def _documents_text_like_lightrag():
    """/documents/text as stock LightRAG answers it: file_source is the document's
    name, and a name doc_status already holds is refused with 409."""
    seen = set()

    def route(body):
        name = body["file_source"]
        if name in seen:
            return 409, {
                "detail": f"Document storage already contains '{name}' (Status: "
                "processed). Delete the existing record before re-inserting."
            }
        seen.add(name)
        return 200, {"status": "success", "message": "queued", "track_id": "insert_1"}

    return route


def test_add_note_default_names_are_unique(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/documents/text")] = _documents_text_like_lightrag()
    srv = load_server()
    assert "queued" in srv.add_note("First note.")
    assert "queued" in srv.add_note("Second note.")
    first, second = [c["json"] for c in fake_lightrag.calls("/documents/text")]
    assert first["text"] == "First note."
    assert re.fullmatch(NOTE_NAME, first["file_source"])
    assert first["file_source"].endswith(
        "-" + hashlib.sha1(b"First note.").hexdigest()[:8]
    )
    assert re.fullmatch(NOTE_NAME, second["file_source"])
    assert first["file_source"] != second["file_source"]


@pytest.mark.parametrize(
    "description,label",
    [
        ("reading notes", "reading notes"),
        # LightRAG keeps only the last path segment of a source and refuses control
        # characters, so path separators and control characters become spaces.
        (" methods/results\nnotes ", "methods results notes"),
        ("draft\x01one\\two", "draft one two"),
    ],
)
def test_add_note_description_prefixes_the_unique_name(
    fake_lightrag, load_server, description, label
):
    fake_lightrag.routes[("POST", "/documents/text")] = _documents_text_like_lightrag()
    srv = load_server()
    assert "queued" in srv.add_note("A note.", description=description)
    assert "queued" in srv.add_note("Another note.", description=description)
    names = [c["json"]["file_source"] for c in fake_lightrag.calls("/documents/text")]
    for name in names:
        assert re.fullmatch(rf"{re.escape(label)} \({NOTE_NAME}\)", name)
    assert names[0] != names[1]


def test_health_reports_unreachable_server(load_server):
    srv = load_server(
        LIGHTRAG_BASE_URL="http://127.0.0.1:9", LIGHTRAG_MCP_HEALTH_TIMEOUT="2"
    )
    assert srv.health().startswith("unreachable at http://127.0.0.1:9:")


def test_http_errors_become_tool_errors_with_their_text(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = lambda body: (422, {"detail": "bad"})
    srv = load_server()
    with pytest.raises(ToolError, match=r"LightRAG /query failed: .*422"):
        srv.search_library("q")
    with pytest.raises(ToolError, match="422"):
        srv.paper_passages("AAAA0001", "q")


@pytest.mark.parametrize(
    "status,payload,explanation",
    [
        (
            409,
            {
                "detail": "Document storage already contains 'x' (Status: processed). "
                "Delete the existing record before re-inserting."
            },
            "Server response: Document storage already contains 'x'",
        ),
        (
            422,
            {"detail": [{"loc": ["body", "file_source"], "msg": "bad character"}]},
            "bad character",
        ),
        (502, b"upstream unavailable", "Server response: upstream unavailable"),
    ],
)
def test_http_errors_carry_the_server_explanation(
    fake_lightrag, load_server, status, payload, explanation
):
    fake_lightrag.routes[("POST", "/documents/text")] = lambda body: (status, payload)
    srv = load_server()
    with pytest.raises(
        ToolError, match=rf"LightRAG /documents/text failed: .*{status}"
    ) as err:
        srv.add_note("text")
    assert explanation in str(err.value)


def test_query_timeout_is_configurable(fake_lightrag, load_server):
    def slow(body):
        time.sleep(1.5)
        return 200, {"response": "late", "references": []}

    fake_lightrag.routes[("POST", "/query")] = slow
    srv = load_server(LIGHTRAG_MCP_TIMEOUT="0.3")
    started = time.monotonic()
    with pytest.raises(ToolError, match="LightRAG /query failed"):
        srv.search_library("q")
    assert time.monotonic() - started < 1.2


def test_reference_fallback_timeout_caps_the_extra_retrieval(
    fake_lightrag, load_server
):
    def slow(body):
        time.sleep(1.5)
        return 200, {
            "status": "success",
            "data": {"references": [{"reference_id": "1"}]},
        }

    fake_lightrag.routes[("POST", "/query")] = _answer_route("Answer.", [])
    fake_lightrag.routes[("POST", "/query/data")] = slow
    srv = load_server(LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT="0.3")
    started = time.monotonic()
    assert srv.search_library("q", answer=True) == "Answer."
    assert time.monotonic() - started < 1.2


def test_non_json_response_is_returned_as_text(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = lambda body: (200, b"plain context")
    srv = load_server()
    assert srv.search_library("q") == "plain context"
