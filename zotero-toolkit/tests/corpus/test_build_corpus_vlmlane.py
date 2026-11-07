#!/usr/bin/env python3
"""The local OCR lane is never wrongly declared down (2026-11-05).
Run:  .venv/bin/python test_build_corpus_vlmlane.py     (plain asserts; pytest also collects it)
On the reference deployment the embedding front took :19530 on 2026-11-03; ocr_server.sh's health test (any
{"status":"ok"}) accepted it as the OCR server and _vlm_alive then found no pepperocr-vl in its /v1/models, so every
document needing OCR was skipped for hours. The default port is now :19531, ocr_server.sh gets the port as OCR_PORT, a
port held by another service is never started onto, and a server that is only busy (silent past the probe timeout) is
waited for. Real HTTP servers on loopback stand in for the lanes; ocr_server.sh is a fake; no GPU, no llama-swap."""

import contextlib
import http.server
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

try:
    import corpus_testlib  # noqa: F401  -- fork layout: tests/corpus/, puts corpus/ on sys.path
except ImportError:
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__))
    )  # bundle layout: beside build_corpus.py
import build_corpus as bc


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


@contextlib.contextmanager
def server(models, slow=0, port=0):
    """A loopback /v1/models answering `models`; the first `slow` requests stall 1.2 s (the probe gives up at 0.5 s)."""
    hits = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            if hits["n"] <= slow:
                time.sleep(1.2)
            body = json.dumps(
                {"data": [{"id": m} for m in models]}
                if self.path == "/v1/models"
                else {"status": "ok"}
            ).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", hits
    finally:
        srv.shutdown()
        srv.server_close()


def free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def fast_probes():
    """Probe timeouts capped at 0.5 s so a 'busy' server is detected quickly."""
    orig = bc._vlm_probe
    with patched(
        bc, "_vlm_probe", lambda base, timeout=3.0: orig(base, min(timeout, 0.5))
    ):
        yield


@contextlib.contextmanager
def lane(url, runner=None):
    """The local lane only: the remote lane closed, `ocr_server.sh` replaced by a fake, a fresh endpoint cache."""
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "ocr_server.sh")
        open(script, "w").write("#!/bin/sh\n")
        os.chmod(script, 0o755)
        runs = []

        def fake_run(args, **kw):
            runs.append((list(args), kw.get("env", {}).get("OCR_PORT")))
            return (
                runner(args, **kw)
                if runner
                else subprocess.CompletedProcess(
                    args, 0, "start: already serving\n", ""
                )
            )

        with (
            patched(bc, "OCR_VLM_LOCAL_URL", url),
            patched(bc, "OCR_VLM_LOCAL_START", script),
            patched(bc, "_vlm_quiet", lambda now=None: True),
            patched(bc.subprocess, "run", fake_run),
            patched(bc, "_vlm_endpoint_cache", None),
            fast_probes(),
        ):
            yield script, runs


# ---------------------------------------------------------------- (c) the lane
def test_default_local_port_is_not_the_embedding_front():
    env = {k: v for k, v in os.environ.items() if k != "OCR_VLM_LOCAL_URL"}
    here = os.path.dirname(os.path.abspath(bc.__file__))
    out = subprocess.run(
        [sys.executable, "-c", "import build_corpus as b; print(b.OCR_VLM_LOCAL_URL)"],
        cwd=here,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert out == "http://127.0.0.1:19531", out


def test_another_service_on_the_port_is_never_started_onto():
    # what :19530 answered on 11-04/05: the embedding front, 200 OK on /health and a model list without pepperocr-vl
    with (
        server(["bge-m3:567m", "qwen2.5:7b-instruct"]) as (url, hits),
        lane(url) as (script, runs),
    ):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            found = bc._vlm_endpoint()
    assert found == ("", ""), found
    assert "another service holds the port" in out.getvalue(), out.getvalue()
    assert runs == [], (
        runs
    )  # `start` would have said "already serving" and been believed


def test_a_busy_local_server_is_waited_for_not_deferred():
    with server(["pepperocr-vl"], slow=2) as (url, hits), lane(url) as (script, runs):
        assert (
            bc._vlm_probe(url) == "busy"
        )  # silent past the probe timeout: a listener, not down
        hits["n"] = 0  # the next two requests stall again: alive, probe, then the wait
        assert bc._vlm_endpoint() == ("local", url), (
            "a server busy with a page must be waited for"
        )
    assert runs == [], runs  # it was up the whole time: nothing to start


def test_a_server_that_stays_busy_is_not_started_onto_after_the_wait_budget():
    with (
        server(["pepperocr-vl"], slow=10**6) as (url, hits),
        lane(url) as (script, runs),
        patched(bc, "OCR_VLM_BUSY_WAIT_S", 1),
    ):
        found = bc._vlm_endpoint()
    assert found == ("", "") and runs == [], (found, runs)


def test_probe_states():
    with server(["pepperocr-vl"]) as (url, _), fast_probes():
        assert bc._vlm_probe(url) == "model"
    with server(["bge-m3:567m"]) as (url, _), fast_probes():
        assert bc._vlm_probe(url) == "other" and not bc._vlm_alive(url)
    with fast_probes():
        assert bc._vlm_probe(f"http://127.0.0.1:{free_port()}") == "down"


def test_start_gets_the_port_of_the_url_and_a_server_that_comes_up_is_used():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    started = []

    def start(args, **kw):
        cm = server(
            ["pepperocr-vl"], port=int(kw["env"]["OCR_PORT"])
        )  # ocr_server.sh serving on $OCR_PORT
        started.append((cm, cm.__enter__()))
        return subprocess.CompletedProcess(args, 0, "start: serving pepperocr-vl\n", "")

    try:
        with lane(url, runner=start) as (script, runs):
            assert bc._vlm_endpoint() == ("local", url)
        assert runs == [([script, "start", "--for-build"], str(port))], runs
    finally:
        for cm, _ in started:
            cm.__exit__(None, None, None)


def test_start_that_succeeds_without_a_server_coming_up_finds_no_lane():
    url = f"http://127.0.0.1:{free_port()}"  # nothing listens; the fake start claims otherwise
    with lane(url) as (script, runs):
        assert bc._vlm_endpoint() == ("", "")
    assert len(runs) == 1


def test_refused_start_finds_no_lane_and_does_not_raise():
    url = f"http://127.0.0.1:{free_port()}"

    def refuse(args, **kw):
        raise subprocess.CalledProcessError(3, args)

    with (
        lane(url, runner=refuse) as (script, runs),
        contextlib.redirect_stdout(io.StringIO()) as out,
    ):
        assert bc._vlm_endpoint() == ("", "")
    assert "could not start the local OCR server" in out.getvalue()


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
