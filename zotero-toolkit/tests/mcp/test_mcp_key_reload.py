"""Offline tests of the proxy's key reload: after a 401/403 it reads LIGHTRAG_API_KEY from
the file named by LIGHTRAG_MCP_ENV_FILE and retries the request once, for every request
the tools send; health reports whether the key is accepted.

The server is the stdlib fake from conftest.py, taught to accept one key; the keys are
synthetic.
"""

from __future__ import annotations

import threading

import pytest
from mcp.server.mcpserver.exceptions import ToolError

OLD = "old-key-0000000000000000000000000"
NEW = "new-key-1111111111111111111111111"


def _accepting(fake, accepted):
    """Every route but /health answers its normal answer for the accepted key only."""
    state = {"key": accepted}
    for route, answer in list(fake.routes.items()):
        if route[1] == "/health":
            continue

        def guarded(body, answer=answer):
            if fake.current.request["headers"].get("x-api-key") != state["key"]:
                return 403, {"detail": "Invalid API Key"}
            return answer(body)

        fake.routes[route] = guarded
    fake.routes[("GET", "/documents/pipeline_status")] = lambda body: (
        (200, {"busy": False})
        if fake.current.request["headers"].get("x-api-key") == state["key"]
        else (403, {"detail": "Invalid API Key"})
    )
    return state


def _answer(fake, method, path, status, payload):
    """A route that needs the accepted key (set up with _accepting afterwards)."""
    fake.routes[(method, path)] = lambda body: (status, payload)


def _env(tmp_path, text):
    path = tmp_path / "key.env"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _keys(fake, path):
    return [r["headers"].get("x-api-key") for r in fake.calls(path)]


def test_a_rotated_key_is_reloaded_and_the_request_sent_once_more(
    fake_lightrag, load_server, tmp_path, capsys
):
    _accepting(fake_lightrag, NEW)
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"OTHER=1\nLIGHTRAG_API_KEY={NEW}\n"),
    )
    assert srv.search_library("q") == "context"
    assert _keys(fake_lightrag, "/query") == [OLD, NEW]
    # the key stays reloaded: the next request carries it at once
    srv.search_library("q2")
    assert _keys(fake_lightrag, "/query")[2:] == [NEW]
    err = capsys.readouterr().err
    assert "reloaded" in err and OLD not in err and NEW not in err


def test_every_request_the_tools_send_goes_through_the_reload(
    fake_lightrag, load_server, tmp_path
):
    """The query, the paper lookup, the passages route and the three structure tools."""
    fake = fake_lightrag
    paper = {
        "paper": "10.48550/arXiv.1706.03762",
        "doi": "10.48550/arXiv.1706.03762",
        "keys": ["AAAA0001"],
        "title": "Attention Is All You Need",
        "citation": "Vaswani, Ashish (2017)",
        "year": "2017",
    }
    fake.routes[("GET", "/graph/papers/search")] = lambda p: (
        200,
        [paper],
        {"X-Paper-Search": "2"},
    )
    fake.routes[("POST", "/graph/papers/passages")] = lambda b: (
        200,
        {"reason": "not_in_kb"},
    )
    fake.routes[("GET", "/graph/papers/related")] = lambda p: (
        200,
        {"reason": "not_in_structure"},
    )
    fake.routes[("GET", "/graph/papers/topics")] = lambda p: (
        200,
        {"total": 0, "topics": []},
    )
    fake.routes[("GET", "/graph/papers/by_topic")] = lambda p: (
        200,
        {"reason": "no_topic"},
    )
    fake.routes[("GET", "/graph/papers/connect")] = lambda p: (
        200,
        {"reason": "not_in_structure", "missing": ["x"], "paths": []},
    )
    _accepting(fake, NEW)
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"LIGHTRAG_API_KEY={NEW}\n"),
    )
    calls = {
        "/query": lambda: srv.search_library("q"),
        "/graph/papers/search": lambda: srv.find_paper("Okafor 2010"),
        "/graph/papers/passages": lambda: srv.paper_passages("AAAA0001", "q"),
        "/graph/papers/related": lambda: srv.related_papers("AAAA0001"),
        "/graph/papers/topics": lambda: srv.papers_by_topic(
            "gravity", topics_only=True
        ),
        "/graph/papers/by_topic": lambda: srv.papers_by_topic("gravity"),
        "/graph/papers/connect": lambda: srv.connect_papers("AAAA0001", "BBBB0002"),
    }
    for path, call in calls.items():
        # as if the key was rotated again since the call before; a refused request
        # would raise, so each call getting through shows its request was retried
        srv.HEADERS["X-API-Key"] = OLD
        srv.KEY = OLD
        call()
        assert _keys(fake, path)[-1] == NEW, path
    # each route was refused at most once, then answered
    assert _keys(fake, "/graph/papers/related") == [OLD, NEW]
    assert _keys(fake, "/graph/papers/connect") == [OLD, NEW]
    assert _keys(fake, "/graph/papers/topics") == [OLD, NEW]
    assert _keys(fake, "/graph/papers/by_topic") == [OLD, NEW]


def test_the_same_key_in_the_file_means_no_retry_and_the_servers_refusal(
    fake_lightrag, load_server, tmp_path
):
    _accepting(fake_lightrag, NEW)
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"LIGHTRAG_API_KEY={OLD}\n"),
    )
    with pytest.raises(ToolError) as exc:
        srv.search_library("q")
    assert "403" in str(exc.value) and "Invalid API Key" in str(exc.value)
    assert _keys(fake_lightrag, "/query") == [OLD]


def test_a_structure_tool_refused_for_good_raises_a_tool_error(
    fake_lightrag, load_server, tmp_path
):
    _accepting(fake_lightrag, NEW)
    fake_lightrag.routes[("GET", "/graph/papers/related")] = lambda p: (
        403,
        {"detail": "Invalid API Key"},
    )
    srv = load_server(LIGHTRAG_API_KEY=OLD)
    with pytest.raises(ToolError, match="Invalid API Key"):
        srv.related_papers("AAAA0001")


def test_without_an_env_file_nothing_is_reloaded_or_retried(fake_lightrag, load_server):
    _accepting(fake_lightrag, NEW)
    srv = load_server(LIGHTRAG_API_KEY=OLD)
    with pytest.raises(ToolError, match="403"):
        srv.search_library("q")
    assert _keys(fake_lightrag, "/query") == [OLD]


def test_a_server_error_does_not_reload(fake_lightrag, load_server, tmp_path):
    fake_lightrag.routes[("POST", "/query")] = lambda b: (500, {"detail": "boom"})
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"LIGHTRAG_API_KEY={NEW}\n"),
    )
    with pytest.raises(ToolError, match="500"):
        srv.search_library("q")
    assert _keys(fake_lightrag, "/query") == [OLD]
    assert srv.HEADERS["X-API-Key"] == OLD


def test_parallel_calls_all_recover_and_the_reload_is_logged_once(
    fake_lightrag, load_server, tmp_path, capsys
):
    _accepting(fake_lightrag, NEW)
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"LIGHTRAG_API_KEY={NEW}\n"),
    )
    barrier = threading.Barrier(8)
    results = []

    def one():
        barrier.wait()
        results.append(srv.search_library("q"))

    threads = [threading.Thread(target=one) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == ["context"] * 8
    assert _keys(fake_lightrag, "/query").count(NEW) == 8
    assert capsys.readouterr().err.count("reloaded") == 1


def test_the_env_file_is_read_for_its_last_key_line_without_quotes(
    fake_lightrag, load_server, tmp_path
):
    path = _env(
        tmp_path,
        f'LIGHTRAG_API_KEY={OLD}\n  # comment\nLIGHTRAG_API_KEY="{NEW}"\nOTHER=x\n',
    )
    srv = load_server(LIGHTRAG_MCP_ENV_FILE=path)
    assert srv._env_file_key() == NEW
    srv2 = load_server(LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, "LIGHTRAG_API_KEY='k'\n"))
    assert srv2._env_file_key() == "k"
    srv3 = load_server(LIGHTRAG_MCP_ENV_FILE=str(tmp_path / "missing.env"))
    assert srv3._env_file_key() == ""
    assert load_server()._env_file_key() == ""


def _api_key_line(text):
    return next(line for line in text.splitlines() if line.startswith("API key:"))


def test_health_says_whether_the_key_is_accepted(fake_lightrag, load_server, tmp_path):
    _accepting(fake_lightrag, OLD)
    srv = load_server(LIGHTRAG_API_KEY=OLD)
    out = srv.health()
    assert out.splitlines()[0].startswith("200 healthy")
    assert out.splitlines()[1] == "API key: accepted"
    assert _api_key_line(srv.health(verbose=True)) == "API key: accepted"


def test_health_says_the_key_was_reloaded(fake_lightrag, load_server, tmp_path):
    _accepting(fake_lightrag, NEW)
    srv = load_server(
        LIGHTRAG_API_KEY=OLD,
        LIGHTRAG_MCP_ENV_FILE=_env(tmp_path, f"LIGHTRAG_API_KEY={NEW}\n"),
    )
    line = _api_key_line(srv.health())
    assert line.startswith("API key: accepted (reloaded from LIGHTRAG_MCP_ENV_FILE")
    assert NEW not in line and OLD not in line


@pytest.mark.parametrize("verbose", [False, True])
def test_health_says_when_the_key_is_rejected_even_though_the_server_is_up(
    fake_lightrag, load_server, tmp_path, verbose
):
    _accepting(fake_lightrag, NEW)
    srv = load_server(LIGHTRAG_API_KEY=OLD)
    out = srv.health(verbose=verbose)
    assert out.startswith("200")  # /health itself takes any key
    line = _api_key_line(out)
    assert line.startswith("API key: REJECTED (HTTP 403: Invalid API Key)")
    assert OLD not in out and NEW not in out


def test_health_does_not_fail_when_the_key_check_cannot_be_made(
    fake_lightrag, load_server
):
    # the fake has no pipeline_status route: a stock server without one answers 404
    out = load_server().health()
    assert _api_key_line(out) == "API key: not checked (HTTP 404)"
