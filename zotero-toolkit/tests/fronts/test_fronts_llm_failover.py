"""Offline tests for fronts/llm_failover.py against stdlib fake primary/fallback servers."""

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

fo = fakes.load_front(
    "llm_failover.py", "fronts_llm_failover", ("LLM_FO_", "OLLAMA_LLM_NUM_CTX")
)

CHAT = "/v1/chat/completions"
PRIMARY_REPLY = {
    "id": "chatcmpl-p",
    "object": "chat.completion",
    "model": "m1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "from primary"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}
EXTRACTION_SYSTEM = (
    "Extract entities and relationships. Output format:\n"
    '{"entities": [{"name": "<n>", "type": "<t>", "description": "<d>"}],'
    ' "relationships": [{"source": "<s>", "target": "<t>", "keywords": "<k>", "description": "<d>"}]}'
)


def ollama_reply(content="from fallback", done_reason="stop", pt=12, ct=5):
    return {
        "model": "x",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": pt,
        "eval_count": ct,
    }


def ok_primary(h, _body):
    fakes.reply(h, 200, PRIMARY_REPLY)


def ok_ollama(h, _body):
    fakes.reply(h, 200, ollama_reply())


def chat(front, model="m1", content="hi", **extra):
    return fakes.call(
        "POST",
        front + CHAT,
        {"model": model, "messages": [{"role": "user", "content": content}], **extra},
    )


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
        "RETRY_502_S": 0.05,
        "_primary_down_until": 0.0,
        "_guard_cache": (0.0, True, []),
        "_reserved_cache": (0.0, 0),
        "_yield": {"next": 0.0, "active_since": 0.0, "drained_at": 0.0, "count": 0},
        "_inflight": {"primary": 0, "fallback": 0},
    }.items():
        monkeypatch.setattr(fo, name, value)
    with fakes.serve(fo) as url:
        yield url


def test_defaults_are_loopback_and_every_policy_is_off():
    assert fo.LISTEN == ("127.0.0.1", 19520)
    assert (
        fo.PRIMARY == "http://127.0.0.1:8080"
        and fo.FALLBACK == "http://127.0.0.1:11434"
    )
    assert (
        fo.FALLBACK_API == "ollama"
        and fo.FALLBACK_MODEL == ""
        and fo.FALLBACK_NUM_CTX == 8192
    )
    assert (fo.PRIMARY_TIMEOUT, fo.FALLBACK_TIMEOUT, fo.BREAKER_SECS) == (
        180.0,
        590.0,
        60.0,
    )
    assert fo.GUARDED_MODELS == {"*"} and fo.ALLOWED_RUNNING == set()
    assert fo._QUIET is None and fo.RESERVED_FILE == "" and fo.YIELD_EVERY_S == 0
    assert (
        fo.SHARED_MODELS == set()
        and fo.SCHEMA_MODELS == set()
        and fo.RETRY_READY_S == 0
    )
    assert (
        fo._big_model_running() == (False, [])
        and not fo._quiet()
        and not fo._reserved()
    )


def test_primary_success_is_forwarded_verbatim(front, primary, fallback):
    primary.route("POST", CHAT, ok_primary)
    raw = b'{"model": "m1",   "messages": [{"role": "user", "content": "hi"}], "temperature": 0}'
    status, headers, body = fakes.call("POST", front + CHAT, raw)
    assert status == 200
    assert headers["X-LLM-Failover-Backend"] == "primary"
    assert json.loads(body) == PRIMARY_REPLY
    assert primary.hits("POST", CHAT)[0]["body"] == raw
    assert fallback.requests == []


def test_primary_stream_is_relayed(front, primary):
    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    primary.route(
        "POST", CHAT, lambda h, b: fakes.stream(h, lines, "text/event-stream")
    )
    status, headers, body = chat(front, stream=True)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "primary"
    assert headers["Content-Type"] == "text/event-stream"
    assert body == b"".join(lines)


def test_refused_connection_falls_back_to_ollama_and_translates(
    front, fallback, monkeypatch
):
    monkeypatch.setattr(fo, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    fallback.route(
        "POST",
        "/api/chat",
        lambda h, b: fakes.reply(h, 200, ollama_reply("hello", "length", 12, 5)),
    )
    status, headers, body = chat(front, temperature=0.2, max_tokens=64, seed=7)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    d = json.loads(body)
    assert d["object"] == "chat.completion" and d["model"] == "m1"
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert d["choices"][0]["finish_reason"] == "length"
    assert d["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }
    sent = json.loads(fallback.hits("POST", "/api/chat")[0]["body"])
    assert sent["model"] == "m1" and sent["stream"] is False
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["options"] == {
        "num_ctx": 8192,
        "temperature": 0.2,
        "num_predict": 64,
        "seed": 7,
    }
    assert (
        fo._primary_down_until > time.time()
    )  # a refused connection opens the breaker


def test_fallback_model_overrides_the_requested_name(front, fallback, monkeypatch):
    monkeypatch.setattr(fo, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    monkeypatch.setattr(fo, "FALLBACK_MODEL", "local-7b")
    fallback.route("POST", "/api/chat", ok_ollama)
    status, _, body = chat(front)
    assert status == 200 and json.loads(body)["model"] == "local-7b"
    assert (
        json.loads(fallback.hits("POST", "/api/chat")[0]["body"])["model"] == "local-7b"
    )


def test_http_500_opens_the_breaker_which_closes_after_its_time(
    front, primary, fallback, monkeypatch, tmp_path
):
    log = tmp_path / "fo.log"
    monkeypatch.setattr(fo, "LOG", str(log))
    monkeypatch.setattr(fo, "BREAKER_SECS", 0.4)
    primary.route("POST", CHAT, lambda h, b: fakes.reply(h, 500, {"error": "boom"}))
    fallback.route("POST", "/api/chat", ok_ollama)

    status, headers, _ = chat(front)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    assert len(primary.hits("POST", CHAT)) == 1
    assert "status=500" in log.read_text() and "(breaker open 0.4 s)" in log.read_text()

    status, headers, _ = chat(front)  # breaker open: primary not even tried
    assert headers["X-LLM-Failover-Backend"] == "fallback"
    assert len(primary.hits("POST", CHAT)) == 1
    assert "(breaker)" in log.read_text()
    primary.route("GET", "/v1/models", lambda h, b: fakes.reply(h, 200, {"data": []}))
    assert json.loads(fakes.call("GET", front + "/health")[2])["breaker_open"] is True

    time.sleep(0.5)
    primary.route("POST", CHAT, ok_primary)
    status, headers, _ = chat(front)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "primary"
    assert len(primary.hits("POST", CHAT)) == 2
    assert json.loads(fakes.call("GET", front + "/health")[2])["breaker_open"] is False


def test_4xx_falls_back_without_opening_the_breaker(front, primary, fallback):
    primary.route(
        "POST", CHAT, lambda h, b: fakes.reply(h, 400, {"error": "context overflow"})
    )
    fallback.route("POST", "/api/chat", ok_ollama)
    assert chat(front)[1]["X-LLM-Failover-Backend"] == "fallback"
    assert fo._primary_down_until == 0.0
    chat(front)
    assert (
        len(primary.hits("POST", CHAT)) == 2
    )  # still tried: the breaker stayed closed


def test_slow_first_byte_times_out_to_fallback(front, primary, fallback, monkeypatch):
    monkeypatch.setattr(fo, "PRIMARY_TIMEOUT", 0.3)

    def slow(h, _body):
        time.sleep(1.5)
        fakes.reply(h, 200, PRIMARY_REPLY)

    primary.route("POST", CHAT, slow)
    fallback.route("POST", "/api/chat", ok_ollama)
    t0 = time.time()
    status, headers, body = chat(front)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    assert json.loads(body)["choices"][0]["message"]["content"] == "from fallback"
    assert time.time() - t0 < 1.4
    assert fo._primary_down_until > time.time()


def test_ollama_stream_is_translated_to_openai_sse(front, fallback, monkeypatch):
    monkeypatch.setattr(fo, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    ndjson = [
        b'{"message":{"role":"assistant","content":"Hel"},"done":false}\n',
        b"\n",
        b"not json\n",
        b'{"message":{"role":"assistant","content":"lo"},"done":false}\n',
        b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"length",'
        b'"prompt_eval_count":3,"eval_count":2}\n',
    ]
    fallback.route(
        "POST",
        "/api/chat",
        lambda h, b: fakes.stream(h, ndjson, "application/x-ndjson"),
    )
    status, headers, body = chat(front, stream=True)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    assert headers["Content-Type"] == "text/event-stream"
    assert json.loads(fallback.hits("POST", "/api/chat")[0]["body"])["stream"] is True
    ev = fakes.sse_events(body)
    assert ev[-1] == "[DONE]"
    chunks = ev[:-1]
    assert all(
        c["object"] == "chat.completion.chunk" and c["model"] == "m1" for c in chunks
    )
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert [c["choices"][0]["delta"].get("content") for c in chunks[1:-1]] == [
        "Hel",
        "lo",
    ]
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert chunks[-1]["choices"][0] == {
        "index": 0,
        "delta": {},
        "finish_reason": "length",
    }
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_openai_fallback_relays_plain_and_stream_with_bearer_keys(
    front, primary, fallback, monkeypatch
):
    monkeypatch.setattr(fo, "FALLBACK_API", "openai")
    monkeypatch.setattr(fo, "FALLBACK_MODEL", "other-model")
    monkeypatch.setattr(fo, "PRIMARY_API_KEY", "pk-test")
    monkeypatch.setattr(fo, "FALLBACK_API_KEY", "fk-test")
    monkeypatch.setattr(
        fo, "BREAKER_SECS", 0.0
    )  # keep trying the primary on every call
    fb_reply = dict(PRIMARY_REPLY, id="chatcmpl-f")
    primary.route("POST", CHAT, lambda h, b: fakes.reply(h, 503, {"error": "loading"}))
    fallback.route("POST", CHAT, lambda h, b: fakes.reply(h, 200, fb_reply))

    status, headers, body = chat(front)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    assert json.loads(body) == fb_reply
    sent = fallback.hits("POST", CHAT)[0]
    assert json.loads(sent["body"])["model"] == "other-model"
    assert sent["headers"]["authorization"] == "Bearer fk-test"
    assert primary.hits("POST", CHAT)[0]["headers"]["authorization"] == "Bearer pk-test"
    assert fallback.hits("POST", "/api/chat") == []

    lines = [b'data: {"choices":[{"delta":{"content":"z"}}]}\n\n', b"data: [DONE]\n\n"]
    fallback.route(
        "POST", CHAT, lambda h, b: fakes.stream(h, lines, "text/event-stream")
    )
    status, headers, body = chat(front, stream=True)
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "fallback"
    assert body == b"".join(lines)


def test_both_backends_failing_returns_502(front, fallback, monkeypatch):
    monkeypatch.setattr(fo, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}")
    fallback.route(
        "POST",
        "/api/chat",
        lambda h, b: fakes.reply(h, 500, {"error": "model not found"}),
    )
    status, _, body = chat(front)
    assert status == 502 and "fallback said 500" in json.loads(body)["error"]
    monkeypatch.setattr(fo, "FALLBACK", f"http://127.0.0.1:{fakes.free_port()}")
    status, _, body = chat(front)
    assert status == 502 and "both backends failed" in json.loads(body)["error"]


def test_health_reports_each_backend(front, primary, fallback, monkeypatch):
    primary.route("GET", "/v1/models", lambda h, b: fakes.reply(h, 200, {"data": []}))
    fallback.route("GET", "/api/tags", lambda h, b: fakes.reply(h, 200, {"models": []}))
    status, _, body = fakes.call("GET", front + "/health")
    d = json.loads(body)
    assert (
        status == 200
        and d["status"] == "ok"
        and d["primary"] is True
        and d["fallback"] is True
    )
    assert d["breaker_open"] is False and d["guard"]["enabled"] is False
    assert (
        primary.hits("GET", "/running") == []
    )  # the guard is off: no llama-swap probe

    primary.route("GET", "/v1/models", lambda h, b: fakes.reply(h, 500, {}))
    status, _, body = fakes.call("GET", front + "/health")
    d = json.loads(body)
    assert (
        status == 200
        and d["status"] == "ok"
        and d["primary"] is False
        and d["fallback"] is True
    )

    monkeypatch.setattr(fo, "FALLBACK", f"http://127.0.0.1:{fakes.free_port()}")
    status, _, body = fakes.call("GET", front + "/health")
    assert status == 503 and json.loads(body)["status"] == "down"

    monkeypatch.setattr(fo, "FALLBACK", fallback.url)
    monkeypatch.setattr(
        fo, "FALLBACK_API", "openai"
    )  # an OpenAI fallback is probed at /v1/models
    fallback.route("GET", "/v1/models", lambda h, b: fakes.reply(h, 200, {"data": []}))
    status, _, body = fakes.call("GET", front + "/health")
    assert status == 200 and json.loads(body)["fallback"] is True


def test_models_lists_the_configured_ids(front, monkeypatch):
    monkeypatch.setattr(fo, "PRIMARY_MODELS", ["small", "extract"])
    monkeypatch.setattr(fo, "FALLBACK_MODEL", "local-7b")
    status, _, body = fakes.call("GET", front + "/v1/models")
    assert status == 200
    assert [(m["id"], m["owned_by"]) for m in json.loads(body)["data"]] == [
        ("small", "primary"),
        ("extract", "primary"),
        ("local-7b", "fallback"),
    ]


def test_bad_requests(front):
    assert fakes.call("GET", front + "/nope")[0] == 404
    assert fakes.call("POST", front + "/v1/completions", {"x": 1})[0] == 404
    assert fakes.call("POST", front + CHAT, b"not json")[0] == 400
    assert fakes.call("POST", front + CHAT, b"[1, 2]")[0] == 400


def test_quiet_window_parsing_and_midnight_crossing(monkeypatch):
    assert fo._parse_window("23:00-02:00") == (1380, 120)
    assert fo._parse_window("4:05-6:40") == (245, 400)
    assert fo._parse_window("nonsense") is None and fo._parse_window("25-3") is None

    def at(hh, mm):
        return time.mktime((2026, 1, 15, hh, mm, 0, 0, 0, -1))

    monkeypatch.setattr(fo, "_QUIET", fo._parse_window("23:00-02:00"))
    assert fo._quiet(at(23, 30)) and fo._quiet(at(1, 59)) and fo._quiet(at(23, 0))
    assert not fo._quiet(at(2, 0)) and not fo._quiet(at(12, 0))
    monkeypatch.setattr(fo, "_QUIET", fo._parse_window("09:00-17:00"))
    assert fo._quiet(at(9, 0)) and not fo._quiet(at(17, 0)) and not fo._quiet(at(8, 59))


def test_quiet_window_keeps_guarded_models_off_the_primary(
    front, primary, fallback, monkeypatch
):
    t = time.localtime()
    cur = t.tm_hour * 60 + t.tm_min
    monkeypatch.setattr(fo, "_QUIET", ((cur - 1) % 1440, (cur + 2) % 1440))
    monkeypatch.setattr(fo, "QUIET_WINDOW", "now")
    monkeypatch.setattr(fo, "GUARDED_MODELS", {"extract"})
    primary.route("POST", CHAT, ok_primary)
    fallback.route("POST", "/api/chat", ok_ollama)
    assert chat(front, model="extract")[1]["X-LLM-Failover-Backend"] == "fallback"
    assert primary.hits("POST", CHAT) == []
    assert (
        chat(front, model="query")[1]["X-LLM-Failover-Backend"] == "primary"
    )  # not guarded
    assert fo._primary_down_until == 0.0  # a policy is not a failure


def test_reservation_file(front, primary, fallback, monkeypatch, tmp_path):
    f = tmp_path / "reserved_until"
    monkeypatch.setattr(fo, "RESERVED_FILE", str(f))

    def reserved_with(text):
        if text is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(text)
        fo._reserved_cache = (0.0, 0)
        return fo._reserved()

    assert reserved_with(None) is False  # missing = no reservation
    assert reserved_with("garbage") is False  # unreadable = no reservation
    assert reserved_with(f"{int(time.time()) - 10}\n") is False
    assert (
        reserved_with(f"{int(time.time()) + 3600} until the benchmark ends\n") is True
    )

    primary.route("POST", CHAT, ok_primary)
    fallback.route("POST", "/api/chat", ok_ollama)
    assert chat(front)[1]["X-LLM-Failover-Backend"] == "fallback"
    assert primary.hits("POST", CHAT) == []
    reserved_with(f"{int(time.time()) - 1}")
    assert chat(front)[1]["X-LLM-Failover-Backend"] == "primary"


def test_running_guard_uses_primary_only_when_every_running_model_is_allowed(
    front, primary, fallback, monkeypatch
):
    monkeypatch.setattr(fo, "ALLOWED_RUNNING", {"small", "embed"})
    running = {
        "running": [
            {"model": "small", "state": "ready"},
            {"model": "big-27b", "state": "ready"},
        ]
    }
    primary.route("GET", "/running", lambda h, b: fakes.reply(h, 200, running))
    primary.route("POST", CHAT, ok_primary)
    fallback.route("POST", "/api/chat", ok_ollama)

    assert chat(front, model="small")[1]["X-LLM-Failover-Backend"] == "fallback"
    assert primary.hits("POST", CHAT) == []

    running["running"] = [
        {"model": "small", "state": "ready"},
        {"model": "embed", "state": "ready"},
    ]
    fo._guard_cache = (0.0, True, [])
    assert chat(front, model="small")[1]["X-LLM-Failover-Backend"] == "primary"

    primary.route(
        "GET", "/running", lambda h, b: fakes.reply(h, 500, {})
    )  # probe failure = fail-safe
    fo._guard_cache = (0.0, True, [])
    assert chat(front, model="small")[1]["X-LLM-Failover-Backend"] == "fallback"
    d = json.loads(fakes.call("GET", front + "/health")[2])
    assert d["guard"]["enabled"] is True and d["guard"]["big_model_running"] is True


def test_shared_dispatch_uses_the_fallback_as_a_second_worker(
    front, primary, fallback, monkeypatch
):
    monkeypatch.setattr(fo, "SHARED_MODELS", {"extract"})
    monkeypatch.setattr(fo, "PRIMARY_SLOTS", 1)
    monkeypatch.setattr(fo, "FALLBACK_SLOTS", 1)
    monkeypatch.setattr(fo, "FALLBACK_MAX_CHARS", 50)
    monkeypatch.setattr(fo, "SLOT_WAIT_S", 0.3)
    primary.route("POST", CHAT, ok_primary)
    fallback.route("POST", "/api/chat", ok_ollama)

    assert (
        chat(front, model="extract")[1]["X-LLM-Failover-Backend"] == "primary"
    )  # free primary slot

    fo._inflight["primary"] = 1  # the primary's only slot is busy
    assert chat(front, model="extract")[1]["X-LLM-Failover-Backend"] == "fallback"
    assert fo._primary_down_until == 0.0  # sharing is not a failure

    t0 = time.time()  # too long for the fallback's context:
    status, headers, _ = chat(
        front, model="extract", content="x" * 60
    )  # waits, then queues on the primary
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "primary"
    assert time.time() - t0 >= 0.3
    assert len(fallback.hits("POST", "/api/chat")) == 1
    fo._inflight["primary"] = 0


def test_extraction_prompts_get_the_schema_and_others_lose_response_format(
    front, primary, fallback, monkeypatch
):
    monkeypatch.setattr(fo, "SCHEMA_MODELS", {"extract"})
    primary.route("POST", CHAT, ok_primary)
    msgs = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": "text"},
    ]
    fakes.call("POST", front + CHAT, {"model": "extract", "messages": msgs})
    sent = json.loads(primary.hits("POST", CHAT)[0]["body"])
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "lightrag_extraction", "schema": fo.EXTRACTION_SCHEMA},
    }

    summary = [
        {"role": "system", "content": "Summarize these descriptions."},
        {"role": "user", "content": "..."},
    ]
    fakes.call(
        "POST",
        front + CHAT,
        {
            "model": "extract",
            "messages": summary,
            "response_format": {"type": "json_object"},
        },
    )
    assert "response_format" not in json.loads(primary.hits("POST", CHAT)[1]["body"])

    fakes.call(
        "POST",
        front + CHAT,
        {
            "model": "query",
            "messages": summary,
            "response_format": {"type": "json_object"},
        },
    )
    assert json.loads(primary.hits("POST", CHAT)[2]["body"])["response_format"] == {
        "type": "json_object"
    }

    monkeypatch.setattr(
        fo, "PRIMARY", f"http://127.0.0.1:{fakes.free_port()}"
    )  # ollama gets the same schema
    fallback.route("POST", "/api/chat", ok_ollama)
    fakes.call("POST", front + CHAT, {"model": "extract", "messages": msgs})
    assert (
        json.loads(fallback.hits("POST", "/api/chat")[0]["body"])["format"]
        == fo.EXTRACTION_SCHEMA
    )


def test_ollama_body_maps_json_object_to_format_json():
    body = json.loads(
        fo._ollama_body(
            {
                "model": "m",
                "messages": [],
                "response_format": {"type": "json_object"},
                "max_tokens": 0,
                "temperature": "hot",
            },
            False,
        )
    )
    assert body["format"] == "json" and body["options"] == {
        "num_ctx": fo.FALLBACK_NUM_CTX
    }


def test_shared_model_retries_once_after_a_502(front, primary, fallback, monkeypatch):
    monkeypatch.setattr(fo, "SHARED_MODELS", {"extract"})
    calls = []

    def flaky(h, _body):
        calls.append(1)
        if len(calls) == 1:
            fakes.reply(h, 502, {"error": "swapping"})
        else:
            fakes.reply(h, 200, PRIMARY_REPLY)

    primary.route("POST", CHAT, flaky)
    status, headers, _ = chat(front, model="extract")
    assert status == 200 and headers["X-LLM-Failover-Backend"] == "primary"
    assert len(calls) == 2 and fallback.requests == [] and fo._primary_down_until == 0.0

    calls.clear()  # a non-shared model falls back at once instead
    fallback.route("POST", "/api/chat", ok_ollama)
    assert chat(front, model="query")[1]["X-LLM-Failover-Backend"] == "fallback"
    assert len(calls) == 1


def test_wait_model_ready_polls_llama_swap_running(front, primary):
    primary.route(
        "GET",
        "/running",
        lambda h, b: fakes.reply(
            h, 200, {"running": [{"model": "m", "state": "ready"}]}
        ),
    )
    assert fo._wait_model_ready("m", 2.0) is True
    t0 = time.time()
    assert (
        fo._wait_model_ready("m", 0) is False and time.time() - t0 < 0.1
    )  # 0 = don't poll


def test_yield_state_machine(monkeypatch):
    monkeypatch.setattr(fo, "YIELD_EVERY_S", 10.0)
    monkeypatch.setattr(fo, "YIELD_HOLD_S", 3.0)
    monkeypatch.setattr(fo, "YIELD_MAX_S", 120.0)
    monkeypatch.setattr(
        fo, "_yield", {"next": 0.0, "active_since": 0.0, "drained_at": 0.0, "count": 0}
    )
    monkeypatch.setattr(fo, "_guard_cache", (999.0, False, ["small"]))
    assert fo._yielding(2, now=1000.0) is False  # first call schedules the first yield
    assert fo._yielding(2, now=1009.9) is False
    assert fo._yielding(2, now=1010.0) is True  # due: stop starting primary calls
    assert fo._yielding(0, now=1011.0) is True  # drained; hold starts
    assert fo._yielding(0, now=1013.9) is True
    assert fo._yielding(0, now=1014.0) is False  # held long enough: resume
    assert fo._yield["count"] == 1 and fo._yield["next"] == 1024.0
    assert fo._guard_cache[0] == 0.0  # forces a fresh /running read
    assert fo._yielding(3, now=1024.0) is True
    assert (
        fo._yielding(3, now=1144.1) is False
    )  # never drained: give up after YIELD_MAX_S
    assert fo._yield["count"] == 2
    monkeypatch.setattr(fo, "YIELD_EVERY_S", 0.0)
    assert fo._yielding(0, now=5000.0) is False  # 0 disables
