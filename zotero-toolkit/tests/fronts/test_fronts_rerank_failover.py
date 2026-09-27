"""Offline tests for fronts/rerank_failover.py against stdlib fake primary/fallback rerankers."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fronts_test_fakes", Path(__file__).with_name("_fakes.py")
)
fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fakes)

rr = fakes.load_front("rerank_failover.py", "fronts_rerank_failover", ("RERANK_FO_",))

RERANK = "/v1/rerank"
REQUEST = {
    "model": "reranker",
    "query": "what does a knowledge graph store?",
    "documents": ["a", "b", "c"],
    "top_n": 2,
}


def ranked(tag):
    return {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ],
        "by": tag,
    }


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")


@pytest.fixture
def primary():
    f = fakes.Fake()
    yield f
    f.close()


@pytest.fixture
def fallback():
    f = fakes.Fake()
    yield f
    f.close()


@pytest.fixture
def front(monkeypatch, primary, fallback):
    for name, value in {
        "PRIMARY": primary.url,
        "FALLBACK": fallback.url,
        "PRIMARY_TIMEOUT": 5.0,
        "FALLBACK_TIMEOUT": 5.0,
        "_primary_down_until": 0.0,
    }.items():
        monkeypatch.setattr(rr, name, value)
    with fakes.serve(rr) as url:
        yield url


def rerank(front):
    return fakes.call("POST", front + RERANK, REQUEST)


def test_defaults_are_loopback():
    assert rr.LISTEN == ("127.0.0.1", 19510)
    assert (
        rr.PRIMARY == "http://127.0.0.1:8080" and rr.FALLBACK == "http://127.0.0.1:8081"
    )
    assert (rr.PRIMARY_TIMEOUT, rr.FALLBACK_TIMEOUT, rr.BREAKER_SECS) == (
        90.0,
        290.0,
        60.0,
    )
    assert rr.PRIMARY_API_KEY == "" and rr.FALLBACK_API_KEY == "" and rr.LOG == ""


def test_primary_success(front, primary, fallback):
    primary.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("primary")))
    status, headers, body = rerank(front)
    assert status == 200 and json.loads(body) == ranked("primary")
    assert headers["Content-Type"] == "application/json"
    assert json.loads(primary.hits("POST", RERANK)[0]["body"]) == REQUEST
    assert fallback.requests == []


def test_refused_connection_falls_back(front, fallback, monkeypatch):
    monkeypatch.setattr(rr, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    fallback.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("fallback")))
    status, _, body = rerank(front)
    assert status == 200 and json.loads(body) == ranked("fallback")
    assert json.loads(fallback.hits("POST", RERANK)[0]["body"]) == REQUEST
    assert rr._primary_down_until > time.time()


def test_http_500_opens_the_breaker_which_closes_after_its_time(
    front, primary, fallback, monkeypatch, tmp_path
):
    log = tmp_path / "rr.log"
    monkeypatch.setattr(rr, "LOG", str(log))
    monkeypatch.setattr(rr, "BREAKER_SECS", 0.4)
    primary.route("POST", RERANK, lambda h, b: fakes.reply(h, 500, {"error": "boom"}))
    fallback.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("fallback")))

    assert json.loads(rerank(front)[2])["by"] == "fallback"
    assert json.loads(rerank(front)[2])["by"] == "fallback"
    assert len(primary.hits("POST", RERANK)) == 1  # second call skipped the primary
    text = log.read_text()
    assert "status=500" in text and "-> fallback" in text and "(breaker)" in text

    time.sleep(0.5)
    primary.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("primary")))
    assert json.loads(rerank(front)[2])["by"] == "primary"
    assert len(primary.hits("POST", RERANK)) == 2


def test_timeout_falls_back(front, primary, fallback, monkeypatch):
    monkeypatch.setattr(rr, "PRIMARY_TIMEOUT", 0.3)

    def slow(h, _body):
        time.sleep(1.5)
        fakes.reply(h, 200, ranked("primary"))

    primary.route("POST", RERANK, slow)
    fallback.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("fallback")))
    t0 = time.time()
    status, _, body = rerank(front)
    assert status == 200 and json.loads(body)["by"] == "fallback"
    assert time.time() - t0 < 1.4


def test_fallback_status_passes_through_and_both_down_is_502(
    front, fallback, monkeypatch
):
    monkeypatch.setattr(rr, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    fallback.route(
        "POST", RERANK, lambda h, b: fakes.reply(h, 422, {"error": "input too long"})
    )
    status, _, body = rerank(front)
    assert status == 422 and json.loads(body) == {"error": "input too long"}
    monkeypatch.setattr(rr, "FALLBACK", f"http://127.0.0.1:{fakes.free_port()}")
    status, _, body = rerank(front)
    assert status == 502 and "both rerank backends failed" in json.loads(body)["error"]


def test_api_keys_travel_as_bearer_headers(front, primary, fallback, monkeypatch):
    monkeypatch.setattr(rr, "PRIMARY_API_KEY", "pk-test")
    monkeypatch.setattr(rr, "FALLBACK_API_KEY", "fk-test")
    primary.route("POST", RERANK, lambda h, b: fakes.reply(h, 503, {}))
    fallback.route("POST", RERANK, lambda h, b: fakes.reply(h, 200, ranked("fallback")))
    rerank(front)
    assert (
        primary.hits("POST", RERANK)[0]["headers"]["authorization"] == "Bearer pk-test"
    )
    assert (
        fallback.hits("POST", RERANK)[0]["headers"]["authorization"] == "Bearer fk-test"
    )


def test_health(front, primary, fallback, monkeypatch):
    primary.route("GET", "/v1/models", lambda h, b: fakes.reply(h, 200, {"data": []}))
    fallback.route("GET", "/health", lambda h, b: fakes.reply(h, 200, {"status": "ok"}))
    status, _, body = fakes.call("GET", front + "/health")
    assert status == 200 and json.loads(body) == {
        "status": "ok",
        "primary": True,
        "fallback": True,
        "breaker_open": False,
    }

    monkeypatch.setattr(rr, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    status, _, body = fakes.call("GET", front + "/health")
    assert status == 200 and json.loads(body)["primary"] is False

    rr._primary_down_until = time.time() + 30
    monkeypatch.setattr(rr, "FALLBACK", f"http://127.0.0.1:{fakes.free_port()}")
    status, _, body = fakes.call("GET", front + "/health")
    assert status == 503 and json.loads(body) == {
        "status": "down",
        "primary": False,
        "fallback": False,
        "breaker_open": True,
    }


def test_unknown_paths_are_404(front):
    assert fakes.call("GET", front + "/v1/rerank")[0] == 404
    assert fakes.call("POST", front + "/rerank", REQUEST)[0] == 404
