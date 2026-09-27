#!/usr/bin/env python3
"""llm_failover.py -- OpenAI-compatible chat-completion front with a fallback backend.

LightRAG has no per-role LLM fallback: when the server named in a role's *_LLM_BINDING_HOST
refuses, stalls or errors, every call of that role fails. Point the role at this front
instead. It forwards POST /v1/chat/completions verbatim to a PRIMARY OpenAI-compatible server
(llama.cpp's llama-server, llama-swap, vLLM, ...) and, on a connection error, a non-200 status
or a timeout before the first byte, replays the request against a FALLBACK:

  * LLM_FO_FALLBACK_API=ollama (default): ollama's NATIVE /api/chat with an explicit num_ctx
    -- the call LightRAG's own ollama binding makes -- translated back to the OpenAI schema,
    plain and SSE-streamed;
  * LLM_FO_FALLBACK_API=openai: another OpenAI-compatible server, relayed as-is.

    GET /health     200 when either backend answers, 503 when neither; the JSON body says which,
                    whether the breaker is open, and the state of every optional policy
    GET /v1/models  LLM_FO_PRIMARY_MODELS plus LLM_FO_FALLBACK_MODEL
    Header          X-LLM-Failover-Backend: primary|fallback on every completion
    Log             one line per request, backend first (LLM_FO_LOG; stderr when unset)

After a primary failure a circuit breaker sends every request straight to the fallback for
LLM_FO_BREAKER_S seconds. A 4xx from the primary is about that request (a prompt over the
context size, say), so it falls back without opening the breaker. Once a streamed primary
response has started there is no fallback: the client has already received bytes.

Optional policies, all off by default (see README.md): a daily quiet window and a "reserved
until" file that keep requests off the primary; a llama-swap guard that uses the primary only
while every running model is on an allow list; a periodic yield that lets pending model loads
in; shared dispatch that treats the fallback as a second worker; a JSON schema attached to
LightRAG's entity-extraction prompts.

Stdlib only, Python >= 3.10. All settings are read from the environment at start-up.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request


def _csv(name: str, default: str = "") -> set[str]:
    return {m.strip() for m in os.environ.get(name, default).split(",") if m.strip()}


LISTEN = (
    os.environ.get("LLM_FO_HOST", "127.0.0.1"),
    int(os.environ.get("LLM_FO_PORT", "19520")),
)
# Base URLs WITHOUT /v1: the front appends /v1/chat/completions, /v1/models, /api/chat, ...
PRIMARY = os.environ.get("LLM_FO_PRIMARY", "http://127.0.0.1:8080").rstrip("/")
PRIMARY_API_KEY = os.environ.get("LLM_FO_PRIMARY_API_KEY", "")
PRIMARY_MODELS = [
    m.strip()
    for m in os.environ.get("LLM_FO_PRIMARY_MODELS", "").split(",")
    if m.strip()
]
FALLBACK = os.environ.get("LLM_FO_FALLBACK", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_API = (
    os.environ.get("LLM_FO_FALLBACK_API", "ollama").strip().lower()
)  # ollama | openai
FALLBACK_API_KEY = os.environ.get("LLM_FO_FALLBACK_API_KEY", "")
FALLBACK_MODEL = os.environ.get(
    "LLM_FO_FALLBACK_MODEL", ""
)  # "" = the model named in the request
FALLBACK_NUM_CTX = int(
    os.environ.get("LLM_FO_NUM_CTX", os.environ.get("OLLAMA_LLM_NUM_CTX", "8192"))
)
# Past PRIMARY_TIMEOUT without a first byte the primary is wedged or starved: use the fallback.
# Keep FALLBACK_TIMEOUT just under the client's own timeout (LightRAG's LLM_TIMEOUT or the
# role's *_LLM_TIMEOUT), so the front reports the failure rather than the client timing out.
PRIMARY_TIMEOUT = float(os.environ.get("LLM_FO_PRIMARY_TIMEOUT", "180"))
FALLBACK_TIMEOUT = float(os.environ.get("LLM_FO_FALLBACK_TIMEOUT", "590"))
BREAKER_SECS = float(
    os.environ.get("LLM_FO_BREAKER_S", "60")
)  # after a primary failure, go straight to the fallback
LOG = os.environ.get("LLM_FO_LOG", "")

# ---- policy scope ---------------------------------------------------------------------------
# The quiet window, the reservation file, the running-model guard and the yield apply only to
# requests whose "model" is in GUARDED_MODELS ("*" = every model). Each policy stays off until
# its own setting enables it, so the scope alone changes nothing.
GUARDED_MODELS = _csv("LLM_FO_GUARDED_MODELS", "*")


def _guarded(model) -> bool:
    return "*" in GUARDED_MODELS or model in GUARDED_MODELS


# ---- running-model guard (llama-swap primaries) ----------------------------------------------
# When the primary is a llama-swap instance whose GPU also hosts larger models, a request for a
# small model in a non-exclusive group never evicts anything: it loads BESIDE a resident large
# model and pushes that model's buffers out of VRAM (its prompt processing slows several-fold).
# With ALLOWED_RUNNING set, guarded requests first ask llama-swap's GET /running what is loaded
# and use the fallback unless every running model is on the allow list. A failed probe counts
# as "not allowed" (fail-safe = fallback). The verdict is cached for 1 s: under ingest load
# requests arrive every few seconds, and a stale verdict is the one way a request could still
# reach the primary while another model starts loading. Empty ALLOWED_RUNNING disables the guard.
ALLOWED_RUNNING = _csv("LLM_FO_ALLOWED_RUNNING")
GUARD_CACHE_SECS = 1.0
_guard_cache = (0.0, True, [])  # (checked_at, big_model_running, running models)


def _big_model_running():
    """(True/False, [running models]) -- True on any probe failure, (False, []) with the guard off."""
    global _guard_cache
    if not ALLOWED_RUNNING:
        return False, []
    now = time.time()
    if now - _guard_cache[0] < GUARD_CACHE_SECS:
        return _guard_cache[1], _guard_cache[2]
    try:
        req = urllib.request.Request(
            PRIMARY + "/running", headers=_headers(PRIMARY_API_KEY)
        )
        with urllib.request.urlopen(req, timeout=3.0) as r:
            running = sorted(
                {m.get("model", "") for m in json.load(r).get("running", [])}
            )
        big = bool(set(running) - ALLOWED_RUNNING)
    except Exception:
        running, big = [], True
    _guard_cache = (now, big, running)
    return big, running


# ---- quiet window ---------------------------------------------------------------------------
# llama-swap does not swap a model out while it has requests in flight, so a steady stream of
# ingest calls starves any OTHER model requested on the same server: that request waits for a
# lull that never comes, which under a steady ingest load can mean many minutes. The
# running-model guard cannot see that swap -- the requested model is not in /running until the
# old one has drained. If the primary's GPU has scheduled
# work of its own, name its window here and guarded requests use the fallback throughout it,
# whatever /running says. Format "HH:MM-HH:MM" in this host's local time; it may cross midnight;
# "" (default) disables. Start it a few minutes early so in-flight calls can drain.
QUIET_WINDOW = os.environ.get("LLM_FO_QUIET_WINDOW", "")


def _parse_window(spec):
    try:
        a, b = spec.split("-")
        h1, m1 = (int(x) for x in a.split(":"))
        h2, m2 = (int(x) for x in b.split(":"))
        return h1 * 60 + m1, h2 * 60 + m2
    except Exception:
        return None


_QUIET = _parse_window(QUIET_WINDOW) if QUIET_WINDOW else None


def _quiet(now=None):
    """True while the quiet window is active (see QUIET_WINDOW)."""
    if _QUIET is None:
        return False
    t = time.localtime(time.time() if now is None else now)
    cur = t.tm_hour * 60 + t.tm_min
    start, end = _QUIET
    return (start <= cur < end) if start <= end else (cur >= start or cur < end)


# ---- reservation file -----------------------------------------------------------------------
# The quiet window covers scheduled work; anything else that wants the primary's GPU (a
# benchmark, a manual job) writes an epoch second into RESERVED_FILE ("reserved until") and
# guarded requests use the fallback at once, so the primary drains before the holder swaps its
# own model in (the same swap-starvation case as the window). A file that is missing, cannot be
# read, or names a moment already past is no reservation. "" (default) disables.
# Manual use:  date -d '+2 hours' +%s > "$LLM_FO_RESERVED_FILE"   /   rm -f "$LLM_FO_RESERVED_FILE"
RESERVED_FILE = os.environ.get("LLM_FO_RESERVED_FILE", "")
_reserved_cache = (0.0, 0)  # (checked_at, until_epoch)


def _reserved_until():
    """Epoch second the primary is reserved until (0 = none), re-read at most once a second."""
    global _reserved_cache
    if not RESERVED_FILE:
        return 0
    now = time.time()
    if now - _reserved_cache[0] < 1.0:
        return _reserved_cache[1]
    try:
        with open(RESERVED_FILE) as f:
            until = int(float(f.read().strip().split()[0]))
    except (OSError, ValueError, IndexError):
        until = 0
    _reserved_cache = (now, until)
    return until


def _reserved(now=None):
    """True while RESERVED_FILE names a moment still in the future."""
    return (time.time() if now is None else now) < _reserved_until()


# ---- periodic yield to pending model loads ---------------------------------------------------
# The window and the reservation cover SCHEDULED loads. An unscheduled one -- another client
# picking a different model on the primary -- still queues behind shared dispatch, which keeps
# PRIMARY_SLOTS calls in flight, and /running does not show the waiting model. The front cannot
# SEE a pending load, so it makes room for one on a clock: every YIELD_EVERY_S it stops starting
# primary calls until the primary's in-flight count drains to 0, holds the primary idle for
# YIELD_HOLD_S (a queued load swaps in during that gap), then re-reads /running uncached. A
# waiting load then waits at most ~YIELD_EVERY_S + one call; the fallback keeps working
# throughout. Applies to guarded SHARED_MODELS. YIELD_EVERY_S=0 (default) disables.
YIELD_EVERY_S = float(os.environ.get("LLM_FO_YIELD_EVERY_S", "0"))
YIELD_HOLD_S = float(os.environ.get("LLM_FO_YIELD_HOLD_S", "3"))
YIELD_MAX_S = float(
    os.environ.get("LLM_FO_YIELD_MAX_S", "120")
)  # give up waiting for a drain after this
_yield = {"next": 0.0, "active_since": 0.0, "drained_at": 0.0, "count": 0}


def _yielding(inflight_primary, now=None):
    """True while the primary must not start a new call. Caller holds _state_lock (reads _inflight)."""
    global _guard_cache
    if YIELD_EVERY_S <= 0:
        return False
    now = time.time() if now is None else now
    y = _yield
    if not y["active_since"]:
        if y["next"] == 0.0:
            y["next"] = now + YIELD_EVERY_S
        if now < y["next"]:
            return False
        y["active_since"], y["drained_at"] = now, 0.0
    if inflight_primary == 0 and not y["drained_at"]:
        y["drained_at"] = now
    done = (y["drained_at"] and now - y["drained_at"] >= YIELD_HOLD_S) or now - y[
        "active_since"
    ] > YIELD_MAX_S
    if done:
        y["active_since"], y["drained_at"], y["next"] = 0.0, 0.0, now + YIELD_EVERY_S
        y["count"] += 1
        _guard_cache = (
            0.0,
            True,
            [],
        )  # force a fresh /running read: a swap may have started
        return False
    return True


# ---- shared dispatch --------------------------------------------------------------------------
# For SHARED_MODELS (typically LightRAG's extraction role) the fallback is a second WORKER, not
# only a fallback: each request goes to the backend with a free slot -- the primary first
# (PRIMARY_SLOTS = its --parallel), then the fallback (FALLBACK_SLOTS = e.g. OLLAMA_NUM_PARALLEL)
# -- so LightRAG can keep PRIMARY_SLOTS + FALLBACK_SLOTS calls in flight and both GPUs stay busy.
# Serve the same weights and quantization on both ends so the graph does not depend on which
# side answered. Prompts above FALLBACK_MAX_CHARS (20000 chars is ~6k tokens, which fits a
# num_ctx of 8192) stay on the primary, because the fallback's smaller context would silently
# truncate them. The breaker and the guarded policies still force the fallback; when both sides
# are full the request waits up to SLOT_WAIT_S, then queues on the primary.
# Empty SHARED_MODELS (default) disables.
SHARED_MODELS = _csv("LLM_FO_SHARED_MODELS")
PRIMARY_SLOTS = int(os.environ.get("LLM_FO_PRIMARY_SLOTS", "4"))
FALLBACK_SLOTS = int(os.environ.get("LLM_FO_FALLBACK_SLOTS", "2"))
FALLBACK_MAX_CHARS = int(os.environ.get("LLM_FO_FALLBACK_MAX_CHARS", "20000"))
_inflight = {"primary": 0, "fallback": 0}
SLOT_WAIT_S = float(
    os.environ.get("LLM_FO_SLOT_WAIT_S", "90")
)  # shared model: wait this long for a free slot
# Shared models get one retry after a primary 502/503: a llama-swap swap or unload answers 502
# for the few seconds a model takes to relaunch. A global unload can take longer than the plain
# delay, so with llama-swap also set RETRY_READY_S (40 works) to first wait until its /running
# reports the model ready. RETRY_READY_S=0 (default) does not poll.
RETRY_502_S = float(os.environ.get("LLM_FO_RETRY_502_S", "4"))
RETRY_READY_S = float(os.environ.get("LLM_FO_RETRY_READY_S", "0"))


def _wait_model_ready(model: str, timeout: float) -> bool:
    """Poll llama-swap /running until `model` is ready (or timeout). Requests during an unload 502."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                PRIMARY + "/running", headers=_headers(PRIMARY_API_KEY)
            )
            with urllib.request.urlopen(req, timeout=3.0) as r:
                for m in json.load(r).get("running", []):
                    if m.get("model") == model and m.get("state") == "ready":
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


class _slot:
    """Count an in-flight request against a backend for the duration of the call."""

    def __init__(self, backend):
        self.b = backend

    def __enter__(self):
        with _state_lock:
            _inflight[self.b] += 1

    def __exit__(self, *exc):
        with _state_lock:
            _inflight[self.b] -= 1


# ---- JSON constraint by prompt shape -----------------------------------------------------------
# A LightRAG role's EXTRA_BODY rides on EVERY call that role makes -- entity/relation merge
# summaries included, which must be plain paragraphs -- so a role-wide response_format is wrong.
# And llama.cpp has shipped builds that do not enforce `response_format: json_object` (the prose
# comes back unchanged) while they do enforce `json_schema`. So for SCHEMA_MODELS the front
# decides per request from the prompt: LightRAG's JSON extraction and gleaning system prompt
# (ENTITY_EXTRACTION_USE_JSON=true) carries an output template with both "entities" and
# "relationships" -> attach the extraction schema (llama-server json_schema / ollama
# format=<schema>); anything else -> no constraint, and any incoming response_format is dropped.
# The schema mirrors that prompt: entities[{name,type,description}], relationships[{source,
# target,keywords,description}]. SCHEMA_MODELS defaults to SHARED_MODELS; "" disables.
SCHEMA_MODELS = _csv("LLM_FO_SCHEMA_MODELS", os.environ.get("LLM_FO_SHARED_MODELS", ""))
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "keywords": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "keywords", "description"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


def _is_extraction_prompt(payload: dict) -> bool:
    for m in payload.get("messages") or []:
        if (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content"), str)
        ):
            c = m["content"]
            return '"entities"' in c and '"relationships"' in c
    return False


def _shape_request(payload: dict) -> dict:
    """SCHEMA_MODELS only: extraction/gleaning prompts get the schema, everything else gets no constraint."""
    p = dict(payload)
    p.pop("response_format", None)
    if _is_extraction_prompt(p):
        p["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "lightrag_extraction", "schema": EXTRACTION_SCHEMA},
        }
    return p


_state_lock = threading.Lock()
_primary_down_until = 0.0


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    if LOG:
        try:
            with open(LOG, "a") as f:
                f.write(line)
            return
        except OSError:
            pass
    sys.stderr.write(line)


def _headers(key: str, json_body: bool = False) -> dict:
    """Request headers; an API key travels only as a bearer header, never in the URL."""
    h = {"Content-Type": "application/json"} if json_body else {}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _open(url: str, data: bytes, key: str, timeout: float):
    req = urllib.request.Request(
        url, data=data, method="POST", headers=_headers(key, json_body=True)
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _probe(url: str, key: str = "", timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=_headers(key)), timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _openai_completion(
    content: str, finish: str, prompt_tokens: int, completion_tokens: int, model
) -> dict:
    return {
        "id": f"chatcmpl-fo-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _openai_chunk(delta: dict, finish, model, usage=None) -> bytes:
    d = {
        "id": f"chatcmpl-fo-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        d["usage"] = usage
    return b"data: " + json.dumps(d).encode() + b"\n\n"


def _ollama_body(payload: dict, stream: bool) -> bytes:
    options = {"num_ctx": FALLBACK_NUM_CTX}
    if isinstance(payload.get("temperature"), (int, float)):
        options["temperature"] = payload["temperature"]
    if isinstance(payload.get("max_tokens"), int) and payload["max_tokens"] > 0:
        options["num_predict"] = payload["max_tokens"]
    if isinstance(payload.get("seed"), int):
        options["seed"] = payload["seed"]
    body = {
        "model": FALLBACK_MODEL or payload.get("model"),
        "messages": payload.get("messages") or [],
        "stream": stream,
        "options": options,
    }
    rf = payload.get("response_format")
    if (
        isinstance(rf, dict)
        and rf.get("type") == "json_schema"
        and isinstance((rf.get("json_schema") or {}).get("schema"), dict)
    ):
        body["format"] = rf["json_schema"][
            "schema"
        ]  # ollama structured outputs: the same schema llama-server enforces
    elif isinstance(rf, dict) and rf.get("type") == "json_object":
        body["format"] = "json"  # ollama's grammar-constrained JSON
    return json.dumps(body).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # quiet; we log ourselves
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str = "application/json",
        backend: str = "",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if backend:
            self.send_header("X-LLM-Failover-Backend", backend)
        self.end_headers()
        self.wfile.write(body)

    def _start_stream(self, backend: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-LLM-Failover-Backend", backend)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _relay(
        self, resp, stream: bool, backend: str, tag: str, t0: float, note: str = ""
    ) -> None:
        """Relay a 200 from an OpenAI-compatible backend: whole body, or SSE lines as they arrive."""
        try:
            if not stream:
                data = resp.read()
                try:
                    u = json.loads(data).get("usage") or {}
                    toks = f" tokens={u.get('prompt_tokens', '?')}+{u.get('completion_tokens', '?')}"
                except Exception:
                    toks = ""
                _log(
                    f"{backend:<8} {tag} status=200 dt={time.time() - t0:.1f}s{toks}{note}"
                )
                self._send(
                    200,
                    data,
                    resp.headers.get("Content-Type", "application/json"),
                    backend,
                )
                return
            # streaming: relay SSE lines as they arrive; after the first byte no fallback
            self._start_stream(backend)
            nbytes = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                self.wfile.write(line)
                self.wfile.flush()
                nbytes += len(line)
            _log(
                f"{backend:<8} {tag} status=200 dt={time.time() - t0:.1f}s sse_bytes={nbytes}{note}"
            )
        except Exception as e:
            _log(
                f"{backend:<8} {tag} FAILED mid-response after {time.time() - t0:.1f}s ({type(e).__name__}: {e})"
            )
            self.close_connection = True
        finally:
            resp.close()

    def do_GET(self):
        if self.path == "/v1/models":
            data = [
                {"id": m, "object": "model", "owned_by": "primary"}
                for m in PRIMARY_MODELS
            ]
            if FALLBACK_MODEL and FALLBACK_MODEL not in PRIMARY_MODELS:
                data.append(
                    {"id": FALLBACK_MODEL, "object": "model", "owned_by": "fallback"}
                )
            self._send(200, json.dumps({"object": "list", "data": data}).encode())
            return
        if self.path != "/health":
            self._send(404, b'{"error":"not found"}')
            return
        primary = _probe(PRIMARY + "/v1/models", PRIMARY_API_KEY)
        fallback = _probe(
            FALLBACK + ("/api/tags" if FALLBACK_API == "ollama" else "/v1/models"),
            FALLBACK_API_KEY,
        )
        with _state_lock:
            breaker = time.time() < _primary_down_until
        big, running = _big_model_running()
        until = _reserved_until()
        body = json.dumps(
            {
                "status": "ok" if (primary or fallback) else "down",
                "primary": primary,
                "fallback": fallback,
                "breaker_open": breaker,
                "primary_models": PRIMARY_MODELS,
                "fallback_api": FALLBACK_API,
                "fallback_model": FALLBACK_MODEL or None,
                "guard": {
                    "enabled": bool(ALLOWED_RUNNING),
                    "guarded_models": sorted(GUARDED_MODELS),
                    "big_model_running": big,
                    "running_on_primary": running,
                },
                "quiet": {"window": QUIET_WINDOW, "active": _quiet()},
                "yield": {
                    "every_s": YIELD_EVERY_S,
                    "hold_s": YIELD_HOLD_S,
                    "active": bool(_yield["active_since"]),
                    "count": _yield["count"],
                },
                "reserved": {
                    "file": RESERVED_FILE or None,
                    "active": _reserved(),
                    "until": (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until))
                        if until
                        else None
                    ),
                },
                "shared": {
                    "models": sorted(SHARED_MODELS),
                    "slots": {"primary": PRIMARY_SLOTS, "fallback": FALLBACK_SLOTS},
                    "inflight": dict(_inflight),
                    "fallback_max_chars": FALLBACK_MAX_CHARS,
                },
                "schema_models": sorted(SCHEMA_MODELS),
            }
        ).encode()
        self._send(200 if (primary or fallback) else 503, body)

    def do_POST(self):
        global _primary_down_until
        if self.path != "/v1/chat/completions":
            self._send(404, b'{"error":"not found"}')
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            payload = json.loads(body)
            assert isinstance(payload, dict)
        except Exception:
            self._send(400, b'{"error":"body must be a JSON object"}')
            return
        model = payload.get("model")
        if model in SCHEMA_MODELS:
            payload = _shape_request(payload)
            body = (
                json.dumps(payload).encode()
            )  # what the primary receives; the fallback body is built from payload
        guarded = _guarded(model)
        stream = bool(payload.get("stream"))
        msgs = payload.get("messages") or []
        chars = sum(
            len(m.get("content") or "")
            for m in msgs
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
        tag = f"model={model} stream={int(stream)} msgs={len(msgs)} chars={chars}"
        t0 = time.time()
        with _state_lock:
            skip_primary = time.time() < _primary_down_until
        why_fallback = "breaker" if skip_primary else ""
        if not skip_primary and guarded and _quiet():
            skip_primary = True
            why_fallback = "quiet"
            tag += f" quiet({QUIET_WINDOW})"
        if not skip_primary and guarded and _reserved():
            skip_primary = True
            why_fallback = "reserved"
            tag += f" reserved(until {time.strftime('%H:%M', time.localtime(_reserved_until()))})"
        if not skip_primary and guarded:
            big, running = _big_model_running()
            if big:
                skip_primary = True
                why_fallback = "guarded"
                tag += f" guarded(running={','.join(running) or 'unreachable'})"
        if model in SHARED_MODELS:
            # Slot-aware scheduling: take the primary while it has a free slot, else the fallback
            # while IT has one, else WAIT (bounded) rather than overload either side -- a 502 burst
            # from a global unload would otherwise have the breaker dump every call on the fallback's few slots.
            # While waiting the breaker and the policies are re-read, so a recovered primary is used
            # as soon as it comes back. Long prompts (> FALLBACK_MAX_CHARS) only wait for the primary.
            deadline = time.time() + SLOT_WAIT_S
            waited = False
            while True:
                with _state_lock:
                    breaker_now = time.time() < _primary_down_until
                primary_ok = not breaker_now
                if primary_ok and guarded:
                    primary_ok = (
                        not _quiet() and not _reserved() and not _big_model_running()[0]
                    )
                with _state_lock:
                    if primary_ok and guarded and _yielding(_inflight["primary"]):
                        primary_ok, why_fallback = False, "yield"
                    if primary_ok and _inflight["primary"] < PRIMARY_SLOTS:
                        skip_primary, why_fallback = False, ""
                        break
                    if (
                        chars <= FALLBACK_MAX_CHARS
                        and _inflight["fallback"] < FALLBACK_SLOTS
                    ):
                        skip_primary, why_fallback = (
                            True,
                            ("shared" if primary_ok else (why_fallback or "breaker")),
                        )
                        break
                if time.time() > deadline:
                    skip_primary, why_fallback = (
                        (not primary_ok),
                        ("" if primary_ok else (why_fallback or "breaker")),
                    )
                    break
                waited = True
                time.sleep(0.5)
            if waited:
                tag += " waited"
            if why_fallback == "shared":
                tag += " shared"

        # ---- primary: OpenAI-compatible, request forwarded verbatim ----
        client_error = False
        if not skip_primary:
            with _slot("primary"):
                url = PRIMARY + "/v1/chat/completions"
                try:
                    resp = _open(url, body, PRIMARY_API_KEY, PRIMARY_TIMEOUT)
                except urllib.error.HTTPError as e:
                    reason = f"status={e.code} body={e.read()[:120]!r}"
                    client_error = (
                        400 <= e.code < 500
                    )  # about THIS request (e.g. context overflow), not the primary's health
                    resp = None
                    if e.code in (502, 503) and model in SHARED_MODELS:
                        # a swap or unload on the primary: the model relaunches in a few seconds
                        time.sleep(RETRY_502_S)
                        # if it is not back, llama-swap may need a request to relaunch it -- the retry is that request
                        _wait_model_ready(model, RETRY_READY_S)
                        try:
                            resp = _open(url, body, PRIMARY_API_KEY, PRIMARY_TIMEOUT)
                            tag += " retried"
                        except urllib.error.HTTPError as e2:
                            reason = (
                                f"status={e2.code} after retry body={e2.read()[:120]!r}"
                            )
                            resp = None
                        except Exception as e2:
                            reason = f"{type(e2).__name__} after retry: {e2}"
                            resp = None
                except Exception as e:  # timeout, refused, reset, ...
                    reason = f"{type(e).__name__}: {e}"
                    resp = None
                if resp is not None:
                    self._relay(resp, stream, "primary", tag, t0)
                    return
                # A 4xx (a prompt over the slot context, say) must not open the breaker: that would
                # send EVERY call to the fallback for BREAKER_SECS over one oversized request.
                # Connection errors, timeouts and 5xx still do.
                if not client_error:
                    with _state_lock:
                        _primary_down_until = time.time() + BREAKER_SECS
                _log(
                    f"primary  {tag} FAILED after {time.time() - t0:.1f}s ({reason}) -> fallback"
                    f"{'' if client_error else f' (breaker open {BREAKER_SECS:g} s)'}"
                )

        # ---- fallback: ollama's native /api/chat translated to the OpenAI schema, or OpenAI as-is ----
        with _slot("fallback"):
            t1 = time.time()
            note = f" ({why_fallback})" if why_fallback else ""
            fb_model = FALLBACK_MODEL or model
            if FALLBACK_API == "openai":
                url = FALLBACK + "/v1/chat/completions"
                data = (
                    json.dumps({**payload, "model": FALLBACK_MODEL}).encode()
                    if FALLBACK_MODEL
                    else body
                )
            else:
                url, data = FALLBACK + "/api/chat", _ollama_body(payload, stream)
            try:
                resp = _open(url, data, FALLBACK_API_KEY, FALLBACK_TIMEOUT)
            except urllib.error.HTTPError as e:
                err = e.read()[:300]
                _log(
                    f"fallback {tag} FAILED status={e.code} after {time.time() - t1:.1f}s {err!r}"
                )
                self._send(
                    502,
                    json.dumps(
                        {
                            "error": f"both backends failed; fallback said {e.code}: {err.decode(errors='replace')}"
                        }
                    ).encode(),
                )
                return
            except Exception as e:
                _log(
                    f"fallback {tag} FAILED after {time.time() - t1:.1f}s ({type(e).__name__}: {e})"
                )
                self._send(
                    502, json.dumps({"error": f"both backends failed: {e}"}).encode()
                )
                return
            if FALLBACK_API == "openai":
                self._relay(resp, stream, "fallback", tag, t1, note)
                return
            try:
                if not stream:
                    d = json.loads(resp.read())
                    content = (d.get("message") or {}).get("content") or ""
                    finish = "length" if d.get("done_reason") == "length" else "stop"
                    pt, ct = (
                        int(d.get("prompt_eval_count") or 0),
                        int(d.get("eval_count") or 0),
                    )
                    _log(
                        f"fallback {tag} status=200 dt={time.time() - t1:.1f}s tokens={pt}+{ct}{note}"
                    )
                    self._send(
                        200,
                        json.dumps(
                            _openai_completion(content, finish, pt, ct, fb_model)
                        ).encode(),
                        "application/json",
                        "fallback",
                    )
                    return
                self._start_stream("fallback")
                self.wfile.write(
                    _openai_chunk({"role": "assistant", "content": ""}, None, fb_model)
                )
                pt = ct = 0
                finish = "stop"
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    piece = (d.get("message") or {}).get("content") or ""
                    if piece:
                        self.wfile.write(
                            _openai_chunk({"content": piece}, None, fb_model)
                        )
                        self.wfile.flush()
                    if d.get("done"):
                        pt, ct = (
                            int(d.get("prompt_eval_count") or 0),
                            int(d.get("eval_count") or 0),
                        )
                        finish = (
                            "length" if d.get("done_reason") == "length" else "stop"
                        )
                        break
                self.wfile.write(
                    _openai_chunk(
                        {},
                        finish,
                        fb_model,
                        {
                            "prompt_tokens": pt,
                            "completion_tokens": ct,
                            "total_tokens": pt + ct,
                        },
                    )
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                _log(
                    f"fallback {tag} status=200 dt={time.time() - t1:.1f}s tokens={pt}+{ct}{note}"
                )
            except Exception as e:
                _log(
                    f"fallback {tag} FAILED mid-response after {time.time() - t1:.1f}s ({type(e).__name__}: {e})"
                )
                self.close_connection = True
            finally:
                resp.close()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    if FALLBACK_API not in ("ollama", "openai"):
        sys.exit(
            f"LLM_FO_FALLBACK_API must be 'ollama' or 'openai', not {FALLBACK_API!r}"
        )
    if QUIET_WINDOW and _QUIET is None:
        sys.exit(
            f"LLM_FO_QUIET_WINDOW must look like HH:MM-HH:MM, not {QUIET_WINDOW!r}"
        )
    _log(
        f"start listen={LISTEN[0]}:{LISTEN[1]} primary={PRIMARY} fallback={FALLBACK} ({FALLBACK_API}, "
        f"model={FALLBACK_MODEL or 'as requested'}, num_ctx={FALLBACK_NUM_CTX}) "
        f"primary_timeout={PRIMARY_TIMEOUT:g}s breaker={BREAKER_SECS:g}s"
    )
    Server(LISTEN, Handler).serve_forever()
