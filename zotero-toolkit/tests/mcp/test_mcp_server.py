"""Offline tests for mcp/lightrag_mcp_server.py against a fake LightRAG server."""
from __future__ import annotations

import hashlib
import re
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

VASWANI = ("Ashish Vaswani, Noam Shazeer, Niki Parmar. (2017). Attention Is All You "
           "Need. Advances in Neural Information Processing Systems. "
           "doi:10.48550/arXiv.1706.03762 [zotero://select/library/items/AAAA0001]")
HE = ("Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual "
      "Learning for Image Recognition. doi:10.1109/CVPR.2016.90 "
      "[zotero://select/library/items/BBBB0002]")

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


def test_cite_unknown_key_falls_back_to_deslugged_title(load_server):
    srv = load_server()
    assert (srv._cite("ZZZZ9999__Some_Paper___Title.md")
            == "Some Paper Title [zotero://select/library/items/ZZZZ9999]")


def test_cite_unresolvable_file_is_bare_filename(load_server):
    srv = load_server()
    assert srv._cite("notes/meeting notes.txt") == "meeting notes.txt"
    assert srv._cite("abcd1234__lowercase_key.md") == "abcd1234__lowercase_key.md"
    assert srv._cite("") == "(untitled source)"
    assert srv._cite(None) == "(untitled source)"


def test_cite_uses_basename_of_nested_paths(load_server):
    srv = load_server()
    assert srv._cite("/srv/inputs/__parsed__/AAAA0001__x.md") == VASWANI


def test_missing_metadata_degrades_and_warns_on_stderr(load_server, tmp_path, capsys):
    srv = load_server(ZOTERO_METADATA=str(tmp_path / "absent.json"))
    assert srv._META == {}
    assert (srv._cite("AAAA0001__Attention_Is_All_You_Need.md")
            == "Attention Is All You Need [zotero://select/library/items/AAAA0001]")
    captured = capsys.readouterr()
    assert "no citation metadata loaded" in captured.err
    assert captured.out == ""


def test_metadata_path_expands_user(load_server, monkeypatch, tmp_path, metadata_file):
    monkeypatch.setenv("HOME", str(metadata_file.parent))
    srv = load_server(ZOTERO_METADATA="~/zotero_metadata.json")
    assert "AAAA0001" in srv._META


@pytest.mark.parametrize("unset", [None, ""])
def test_default_metadata_path_matches_the_corpus_builder(load_server, monkeypatch, tmp_path,
                                                          metadata_file, unset):
    # Unset (or empty) ZOTERO_METADATA means ~/Zotero/zotero_metadata.json, the file the
    # corpus builder writes by default, wherever the client launches the server from.
    (tmp_path / "Zotero").mkdir()
    (tmp_path / "Zotero" / "zotero_metadata.json").write_text(metadata_file.read_text())
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setenv("HOME", str(tmp_path))
    srv = load_server(ZOTERO_METADATA=unset)
    assert "AAAA0001" in srv._META


def test_context_lists_every_source_in_server_order_before_context(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", REFERENCES)
    srv = load_server()
    out = srv.search_library("attention")
    assert out == ("### Retrieved documents (compiled citations)\n"
                   + "\n".join(COMPILED) + "\n\nRAW CONTEXT")
    body = fake_lightrag.calls("/query")[0]["json"]
    assert body == {"query": "attention", "mode": "mix", "top_k": 10,
                    "only_need_context": True}


def test_context_without_references_is_just_the_context(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", [])
    srv = load_server()
    assert srv.search_library("q") == "RAW CONTEXT"
    assert fake_lightrag.calls("/query/data") == []


def test_get_sources_puts_citations_after_context(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("RAW CONTEXT", REFERENCES)
    srv = load_server()
    out = srv.get_sources("q", mode="local", top_k=5)
    assert out == ("RAW CONTEXT\n\n### Retrieved documents (compiled citations)\n"
                   + "\n".join(COMPILED))
    assert fake_lightrag.calls("/query")[0]["json"]["only_need_context"] is True


def test_answer_replaces_invented_references_with_compiled_sources(fake_lightrag, load_server):
    llm = ("Transformers rely on attention [1]. Residual links ease training [2].\n\n"
           "### References\n\n"
           "- [1] Document Title One (2019). Invented Journal.\n"
           "- [2] Document Title Two\n"
           "[3] Another invented entry\n")
    fake_lightrag.routes[("POST", "/query")] = _answer_route(llm, REFERENCES[:2])
    srv = load_server()
    out = srv.search_library("q", answer=True)
    assert out == ("Transformers rely on attention [1]. Residual links ease training [2].\n\n"
                   "### Sources (compiled from retrieval)\n"
                   f"- [1] {HE}\n- [2] {VASWANI}")
    body = fake_lightrag.calls("/query")[0]["json"]
    assert "only_need_context" not in body
    assert "Do not generate a References section" in body["user_prompt"]
    assert fake_lightrag.calls("/query/data") == []


def test_answer_strips_server_sources_block_to_avoid_duplicates(fake_lightrag, load_server):
    llm = "Answer text [1].\n\n### Sources\n- [1] Server-rendered citation\n"
    fake_lightrag.routes[("POST", "/query")] = _answer_route(llm, REFERENCES[:1])
    srv = load_server()
    out = srv.search_library("q", answer=True)
    assert out.count("### Sources") == 1
    assert "Server-rendered citation" not in out
    assert out.endswith(f"### Sources (compiled from retrieval)\n- [1] {HE}")


def test_answer_without_references_falls_back_to_query_data_once(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route("Cached answer.", [])
    fake_lightrag.routes[("POST", "/query/data")] = lambda body: (
        200, {"status": "success", "data": {"references": REFERENCES[1:2]}})
    srv = load_server()
    out = srv.search_library("q", mode="hybrid", top_k=7, answer=True)
    assert out == ("Cached answer.\n\n### Sources (compiled from retrieval)\n"
                   f"- [2] {VASWANI}")
    data_calls = fake_lightrag.calls("/query/data")
    assert [c["json"] for c in data_calls] == [{"query": "q", "mode": "hybrid", "top_k": 7}]
    # An identical query in the same session reuses the memoised references.
    assert srv.search_library("q", mode="hybrid", top_k=7, answer=True) == out
    assert len(fake_lightrag.calls("/query/data")) == 1


def test_answer_fallback_failure_returns_answer_without_sources(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = _answer_route(
        "Answer.\n\n### References\n- [1] Invented", [])
    fake_lightrag.routes[("POST", "/query/data")] = lambda body: (500, {"detail": "boom"})
    srv = load_server()
    assert srv.search_library("q", answer=True) == "Answer."


# --- invented-reference stripping ------------------------------------------------

def test_strip_keeps_sources_of_error_prose(load_server):
    srv = load_server()
    text = ("Intro.\n\n### Sources of error\n\n"
            "The dominant sources of error are sampling bias and measurement noise.\n\n"
            "References to prior work appear inline.")
    assert srv._strip_llm_refs(text) == text


@pytest.mark.parametrize("heading", [
    "### References", "## Sources", "References:", "sources", "#### REFERENCES :",
])
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
    srv.get_sources("q", mode=mode)
    assert [c["json"]["mode"] for c in fake_lightrag.calls("/query")] == [mode, mode]


@pytest.mark.parametrize("mode", ["bypass", "graph", "", "MIXED"])
def test_invalid_modes_fall_back_to_mix(fake_lightrag, load_server, mode):
    srv = load_server()
    srv.search_library("q", mode=mode)
    srv.search_library("q", mode=mode, answer=True)
    srv.get_sources("q", mode=mode)
    assert {c["json"]["mode"] for c in fake_lightrag.calls("/query")} == {"mix"}


@pytest.mark.parametrize("requested,sent", [(10, 10), (40, 40), (41, 40), (5000, 40),
                                            (0, 1), (-3, 1)])
def test_top_k_is_clamped_by_default(fake_lightrag, load_server, requested, sent):
    srv = load_server()
    srv.search_library("q", top_k=requested)
    srv.search_library("q", top_k=requested, answer=True)
    srv.get_sources("q", top_k=requested)
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query")] == [sent] * 3
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query/data")] == [sent]


def test_top_k_cap_is_configurable(fake_lightrag, load_server):
    srv = load_server(LIGHTRAG_MCP_MAX_TOP_K="15")
    srv.search_library("q", top_k=40)
    srv.search_library("q", top_k=12)
    assert [c["json"]["top_k"] for c in fake_lightrag.calls("/query")] == [15, 12]


def test_api_key_travels_in_header_never_in_url(fake_lightrag, load_server, api_key):
    srv = load_server()
    srv.search_library("q")
    srv.search_library("q2", answer=True)       # no references -> /query/data too
    srv.get_sources("q")
    srv.add_note("text")
    assert srv.health().startswith("200:")
    paths = {r["path"] for r in fake_lightrag.requests}
    assert paths == {"/query", "/query/data", "/documents/text", "/health"}
    for r in fake_lightrag.requests:
        assert r["headers"].get("x-api-key") == api_key
        assert "?" not in r["path"] and api_key not in r["path"]
        assert api_key not in str(r["json"])


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
            return 409, {"detail": f"Document storage already contains '{name}' (Status: "
                                   "processed). Delete the existing record before re-inserting."}
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
    assert first["file_source"].endswith("-" + hashlib.sha1(b"First note.").hexdigest()[:8])
    assert re.fullmatch(NOTE_NAME, second["file_source"])
    assert first["file_source"] != second["file_source"]


@pytest.mark.parametrize("description,label", [
    ("reading notes", "reading notes"),
    # LightRAG keeps only the last path segment of a source and refuses control
    # characters, so path separators and control characters become spaces.
    (" methods/results\nnotes ", "methods results notes"),
    ("draft\x01one\\two", "draft one two"),
])
def test_add_note_description_prefixes_the_unique_name(fake_lightrag, load_server,
                                                       description, label):
    fake_lightrag.routes[("POST", "/documents/text")] = _documents_text_like_lightrag()
    srv = load_server()
    assert "queued" in srv.add_note("A note.", description=description)
    assert "queued" in srv.add_note("Another note.", description=description)
    names = [c["json"]["file_source"] for c in fake_lightrag.calls("/documents/text")]
    for name in names:
        assert re.fullmatch(rf"{re.escape(label)} \({NOTE_NAME}\)", name)
    assert names[0] != names[1]


def test_health_reports_unreachable_server(load_server):
    srv = load_server(LIGHTRAG_BASE_URL="http://127.0.0.1:9", LIGHTRAG_MCP_HEALTH_TIMEOUT="2")
    assert srv.health().startswith("unreachable at http://127.0.0.1:9:")


def test_http_errors_become_tool_errors_with_their_text(fake_lightrag, load_server):
    fake_lightrag.routes[("POST", "/query")] = lambda body: (422, {"detail": "bad"})
    srv = load_server()
    with pytest.raises(ToolError, match=r"LightRAG /query failed: .*422"):
        srv.search_library("q")
    with pytest.raises(ToolError, match="422"):
        srv.get_sources("q")


@pytest.mark.parametrize("status,payload,explanation", [
    (409, {"detail": "Document storage already contains 'x' (Status: processed). "
                     "Delete the existing record before re-inserting."},
     "Server response: Document storage already contains 'x'"),
    (422, {"detail": [{"loc": ["body", "file_source"], "msg": "bad character"}]},
     "bad character"),
    (502, b"upstream unavailable", "Server response: upstream unavailable"),
])
def test_http_errors_carry_the_server_explanation(fake_lightrag, load_server,
                                                  status, payload, explanation):
    fake_lightrag.routes[("POST", "/documents/text")] = lambda body: (status, payload)
    srv = load_server()
    with pytest.raises(ToolError, match=rf"LightRAG /documents/text failed: .*{status}") as err:
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


def test_reference_fallback_timeout_caps_the_extra_retrieval(fake_lightrag, load_server):
    def slow(body):
        time.sleep(1.5)
        return 200, {"status": "success", "data": {"references": [{"reference_id": "1"}]}}
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
