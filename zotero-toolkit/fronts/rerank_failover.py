#!/usr/bin/env python3
"""rerank_failover.py -- rerank front for LightRAG: a primary reranker first, a fallback on failure.

LightRAG has no rerank fallback of its own: when the configured rerank host fails, it logs an
error and silently answers from UNRANKED chunks, which reads as a normal answer rather than a
broken one. This front forwards POST /v1/rerank (the Cohere/Jina-style body llama.cpp's
llama-server, llama-swap and vLLM accept) to a PRIMARY reranker and, on any error, non-200
status or timeout, replays the same request against a FALLBACK reranker. After a primary
failure a circuit breaker sends every request straight to the fallback for RERANK_FO_BREAKER_S
seconds.

    LightRAG .env   RERANK_BINDING=cohere
                    RERANK_BINDING_HOST=http://127.0.0.1:19510/v1/rerank
    GET /health     200 when either backend answers, 503 when neither; JSON body says which.
    Log             one line per request (RERANK_FO_LOG; stderr when unset).

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

LISTEN = (
    os.environ.get("RERANK_FO_HOST", "127.0.0.1"),
    int(os.environ.get("RERANK_FO_PORT", "19510")),
)
# Base URLs WITHOUT /v1: the front appends /v1/rerank (and /v1/models, /health for probes).
PRIMARY = os.environ.get("RERANK_FO_PRIMARY", "http://127.0.0.1:8080").rstrip("/")
FALLBACK = os.environ.get("RERANK_FO_FALLBACK", "http://127.0.0.1:8081").rstrip("/")
PRIMARY_API_KEY = os.environ.get("RERANK_FO_PRIMARY_API_KEY", "")
FALLBACK_API_KEY = os.environ.get("RERANK_FO_FALLBACK_API_KEY", "")
# Past PRIMARY_TIMEOUT the primary is starved or reloading: use the fallback. Keep
# FALLBACK_TIMEOUT just under LightRAG's RERANK_TIMEOUT, and raise that above PRIMARY_TIMEOUT
# plus a fallback rerank: its default (30 s as of LightRAG 1.5.7) suits hosted APIs, not a
# local reranker scoring ~50 chunks.
PRIMARY_TIMEOUT = float(os.environ.get("RERANK_FO_PRIMARY_TIMEOUT", "90"))
FALLBACK_TIMEOUT = float(os.environ.get("RERANK_FO_FALLBACK_TIMEOUT", "290"))
BREAKER_SECS = float(
    os.environ.get("RERANK_FO_BREAKER_S", "60")
)  # after a primary failure, go straight to the fallback
LOG = os.environ.get("RERANK_FO_LOG", "")

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


def _post(
    base: str, path: str, body: bytes, key: str, timeout: float
) -> tuple[int, bytes, str]:
    req = urllib.request.Request(
        base + path, data=body, method="POST", headers=_headers(key, json_body=True)
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "application/json")


def _probe(url: str, key: str = "", timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=_headers(key)), timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # keep stdout quiet; we log ourselves
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._send(404, b'{"error":"not found"}')
            return
        primary = _probe(PRIMARY + "/v1/models", PRIMARY_API_KEY)
        fallback = _probe(FALLBACK + "/health", FALLBACK_API_KEY)
        with _state_lock:
            breaker = time.time() < _primary_down_until
        body = json.dumps(
            {
                "status": "ok" if (primary or fallback) else "down",
                "primary": primary,
                "fallback": fallback,
                "breaker_open": breaker,
            }
        ).encode()
        self._send(200 if (primary or fallback) else 503, body)

    def do_POST(self):
        global _primary_down_until
        if self.path != "/v1/rerank":
            self._send(404, b'{"error":"not found"}')
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            ndocs = len(json.loads(body).get("documents") or [])
        except Exception:
            ndocs = -1
        t0 = time.time()
        with _state_lock:
            skip_primary = time.time() < _primary_down_until
        if not skip_primary:
            try:
                status, data, ctype = _post(
                    PRIMARY, "/v1/rerank", body, PRIMARY_API_KEY, PRIMARY_TIMEOUT
                )
                if status == 200:
                    _log(f"primary  n={ndocs} status=200 dt={time.time() - t0:.1f}s")
                    self._send(200, data, ctype)
                    return
                reason = f"status={status} body={data[:120]!r}"
            except Exception as e:  # timeout, refused, reset, ...
                reason = f"{type(e).__name__}: {e}"
            with _state_lock:
                _primary_down_until = time.time() + BREAKER_SECS
            _log(
                f"primary  n={ndocs} FAILED after {time.time() - t0:.1f}s ({reason}) -> fallback"
            )
        t1 = time.time()
        try:
            status, data, ctype = _post(
                FALLBACK, "/v1/rerank", body, FALLBACK_API_KEY, FALLBACK_TIMEOUT
            )
            _log(
                f"fallback n={ndocs} status={status} dt={time.time() - t1:.1f}s"
                f"{' (breaker)' if skip_primary else ''}"
            )
            self._send(status, data, ctype)
        except Exception as e:
            _log(
                f"fallback n={ndocs} FAILED after {time.time() - t1:.1f}s ({type(e).__name__}: {e})"
            )
            self._send(
                502, json.dumps({"error": f"both rerank backends failed: {e}"}).encode()
            )


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    _log(
        f"start listen={LISTEN[0]}:{LISTEN[1]} primary={PRIMARY} fallback={FALLBACK} "
        f"primary_timeout={PRIMARY_TIMEOUT:g}s breaker={BREAKER_SECS:g}s"
    )
    Server(LISTEN, Handler).serve_forever()
