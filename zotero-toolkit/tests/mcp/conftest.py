"""Shared fixtures for the MCP proxy tests.

The LightRAG server is faked with a stdlib HTTP server on a loopback port, so
the proxy's real httpx calls (URL, headers, JSON body) are exercised end to end
without any network or LightRAG install.

No __init__.py lives in this directory on purpose: a package named "mcp" here
would shadow the real mcp SDK.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SERVER_PY = Path(__file__).resolve().parents[2] / "mcp" / "lightrag_mcp_server.py"
TEST_API_KEY = "test-key-not-a-secret"

# Synthetic Zotero keys; the bibliographic data is public.
METADATA = {
    "AAAA0001": {
        "filename": "Vaswani - 2017 - Attention is all you need.pdf",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": "2017",
        "doi": "10.48550/arXiv.1706.03762",
        "publication": "Advances in Neural Information Processing Systems",
        "tags": [],
        "abstract": "",
        "source": "parent",
    },
    "BBBB0002": {
        "filename": "He - 2016 - Deep residual learning.pdf",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": "2016",
        "doi": "10.1109/CVPR.2016.90",
        "publication": "",
        "tags": [],
        "abstract": "",
        "source": "parent",
    },
}


class FakeLightRAG:
    """Records every request; answers from per-route handlers tests can replace."""

    def __init__(self):
        self.requests: list[dict] = []
        self.routes = {
            ("GET", "/health"): lambda body: (200, {"status": "healthy"}),
            ("POST", "/query"): lambda body: (200, {"response": "context", "references": []}),
            ("POST", "/query/data"): lambda body: (200, {"status": "success",
                                                         "data": {"references": []}}),
            ("POST", "/documents/text"): lambda body: (200, {"status": "success",
                                                             "message": "queued"}),
        }
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw) if raw else None
                fake.requests.append({"method": method, "path": self.path,
                                      "headers": {k.lower(): v for k, v in self.headers.items()},
                                      "json": body})
                route = fake.routes.get((method, self.path))
                status, payload = route(body) if route else (404, {"detail": "Not Found"})
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._serve("GET")

            def do_POST(self):
                self._serve("POST")

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def calls(self, path):
        return [r for r in self.requests if r["path"] == path]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api_key():
    return TEST_API_KEY


@pytest.fixture
def fake_lightrag():
    fake = FakeLightRAG()
    yield fake
    fake.close()


@pytest.fixture
def metadata_file(tmp_path):
    path = tmp_path / "zotero_metadata.json"
    path.write_text(json.dumps(METADATA), encoding="utf-8")
    return path


_counter = itertools.count()


@pytest.fixture
def load_server(monkeypatch, fake_lightrag, metadata_file):
    """Import a fresh copy of the proxy module under the given environment.

    The module reads its configuration at import time, exactly as it does when
    an MCP client launches it, so every test gets its own import.
    """
    def load(**env):
        base = {
            "LIGHTRAG_BASE_URL": fake_lightrag.url,
            "LIGHTRAG_API_KEY": TEST_API_KEY,
            "ZOTERO_METADATA": str(metadata_file),
        }
        base.update(env)
        for name in ("LIGHTRAG_BASE_URL", "LIGHTRAG_API_KEY", "ZOTERO_METADATA",
                     "LIGHTRAG_MCP_TIMEOUT", "LIGHTRAG_MCP_REF_FALLBACK_TIMEOUT",
                     "LIGHTRAG_MCP_HEALTH_TIMEOUT", "LIGHTRAG_MCP_MAX_TOP_K"):
            monkeypatch.delenv(name, raising=False)
        for name, value in base.items():
            if value is not None:
                monkeypatch.setenv(name, value)
        spec = importlib.util.spec_from_file_location(
            f"lightrag_mcp_server_under_test_{next(_counter)}", SERVER_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load
