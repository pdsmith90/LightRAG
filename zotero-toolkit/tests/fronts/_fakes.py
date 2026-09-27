"""Stdlib fake backends and helpers for the fronts tests (loaded by path, not imported as a package)."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_front(filename: str, module_name: str, env_prefixes: tuple[str, ...]):
    """Import fronts/<filename> with every setting it reads removed from the environment."""
    saved = {k: v for k, v in os.environ.items() if k.startswith(env_prefixes)}
    for k in saved:
        del os.environ[k]
    try:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "fronts" / filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.update(saved)
    return mod


def free_port() -> int:
    """A loopback port nothing listens on (connecting to it is refused)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def reply(h, status: int, obj, ctype: str = "application/json") -> None:
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    h.send_response(status)
    h.send_header("Content-Type", ctype)
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def stream(h, chunks, ctype: str) -> None:
    """Send a 200 whose body is `chunks`, written one by one, ended by closing the connection."""
    h.send_response(200)
    h.send_header("Content-Type", ctype)
    h.send_header("Connection", "close")
    h.end_headers()
    for c in chunks:
        h.wfile.write(c)
        h.wfile.flush()
    h.close_connection = True


class _QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False      # a deliberately slow handler must not hold up teardown

    def handle_error(self, request, client_address):
        pass    # a front that timed out closes its socket; the fake's late write is expected to fail


class Fake:
    """A loopback HTTP server whose routes the test sets: route(method, path, fn(handler, body))."""

    def __init__(self):
        self.requests: list[dict] = []
        self.routes: dict = {}
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _handle(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                fake.requests.append({"method": method, "path": self.path,
                                      "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                fn = fake.routes.get((method, self.path))
                if fn is None:
                    reply(self, 404, {"error": "no route"})
                    return
                fn(self, body)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self.server = _QuietServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def route(self, method: str, path: str, fn) -> None:
        self.routes[(method, path)] = fn

    def hits(self, method: str, path: str) -> list[dict]:
        return [r for r in self.requests if r["method"] == method and r["path"] == path]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@contextlib.contextmanager
def serve(mod):
    """Run a front's Server on an ephemeral loopback port; yields its base URL."""
    srv = mod.Server(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def call(method: str, url: str, body=None, timeout: float = 10.0):
    """(status, headers, body bytes) -- never raises on an HTTP error status."""
    data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def sse_events(body: bytes) -> list:
    """Parse an SSE body into decoded JSON events; the literal [DONE] marker stays a string."""
    out = []
    for line in body.decode().splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def wait_until(pred, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()
