"""Offline tests for the `start --for-build` gate of fronts/ocr_server.sh.

The script runs from a temp directory with a stub llama-server and an unreachable LightRAG (.env PORT=1), so a
start that passes the gate stops at rag_busy with exit 3 "LightRAG unreachable" and never launches anything.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "fronts" / "ocr_server.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("flock") is None,
    reason="needs /proc and util-linux flock",
)


@pytest.fixture
def here(tmp_path):
    shutil.copy(SCRIPT, tmp_path / "ocr_server.sh")
    (tmp_path / "bin").mkdir()
    stub = tmp_path / "bin" / "llama-server"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    (tmp_path / "g.gguf").touch()
    (tmp_path / "p.gguf").touch()
    (tmp_path / ".env").write_text("PORT=1\n")
    return tmp_path


def run(here, shell=None):
    """Run `start --for-build`, optionally from a bash -c wrapper (which then is the script's parent)."""
    env = dict(
        os.environ,
        OCR_LLAMA_BIN=str(here / "bin"),
        OCR_GGUF=str(here / "g.gguf"),
        OCR_MMPROJ=str(here / "p.gguf"),
        OCR_PORT="1",
        OCR_OLLAMA="http://127.0.0.1:1",
    )
    cmd = ["bash", str(here / "ocr_server.sh"), "start", "--for-build"]
    if shell:
        cmd = [
            "bash",
            "-c",
            shell + ' && bash "$0" start --for-build',
            str(here / "ocr_server.sh"),
        ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def test_a_foreign_pause_refuses_and_is_left_alone(here):
    other = subprocess.Popen(["sleep", "60"])
    try:
        pause = here / ".feeder_pause"
        pause.write_text(f"reingest_cycle 2026-01-01 01:00:00 pid {other.pid}\n")
        code, out = run(here)
        assert code == 3 and "held by another job" in out
        assert pause.read_text().startswith("reingest_cycle ")
    finally:
        other.kill()
        other.wait()


def test_b_callers_own_pause_and_lock_pass_the_gate(here):
    # the reingest cycle's shape: it writes its pid into .feeder_pause and holds the trickle lock while its
    # conversion (a child process) asks for OCR
    pause, lock = here / ".feeder_pause", here / ".reingest_trickle.lock"
    shell = f'echo "reingest_cycle 2026-01-01 01:00:00 pid $$" > {pause} && exec 7>>{lock} && flock 7'
    code, out = run(here, shell)
    assert code == 3 and "LightRAG unreachable" in out, out
    assert "the feeder pause is our caller's" in out
    assert pause.read_text().startswith("reingest_cycle "), (
        "the caller's pause was overwritten or removed"
    )


def test_c_no_pause_takes_and_releases_its_own(here):
    code, out = run(here)
    assert code == 3 and "LightRAG unreachable" in out, out
    assert not (here / ".feeder_pause").exists()


def test_d_foreign_trickle_lock_refuses(here):
    with open(here / ".reingest_trickle.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        code, out = run(here)
    assert code == 3 and "holds the trickle lock" in out, out


def test_e_init_is_never_a_caller(here):
    (here / ".feeder_pause").write_text("another_job 2026-01-01 pid 1\n")
    code, out = run(here)
    assert code == 3 and "held by another job" in out, out
