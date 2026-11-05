"""Offline tests of mcp/paper_pdf_tools.py, the PDF helper behind paper_equation and
paper_table: number and tag parsing, the OCR answer's LaTeX picked per equation number,
citing sentences, table cells and the order split, then both commands on invented PDFs
(the layout model replaced by invented boxes) with a stand-in llama-server: OCR used,
switched off, a server that dies or hangs, and the server's death when its caller is
killed.

All numbers, names and layouts here are invented; no model, no GPU, nothing is left
running. The PDF-level tests are skipped when PyMuPDF is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HELPER_PY = Path(__file__).resolve().parents[2] / "mcp" / "paper_pdf_tools.py"


@pytest.fixture
def t(monkeypatch, tmp_path):
    """A fresh copy of the helper module with its cache in tmp_path and no OCR model."""
    for name in ("LIGHTRAG_MCP_OCR", "LIGHTRAG_MCP_CACHE", "LIGHTRAG_MCP_OCR_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location(
        "paper_pdf_tools_under_test", HELPER_PY
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CACHE_DIR = str(tmp_path / "cache")
    return mod


# ---------------------------------------------------------------- pure functions


def test_numbers(t):
    assert t.norm_number("(6)") == "6" and t.norm_number("Eq. 6") == "6"
    assert t.norm_number(" A.1 ") == "a.1" and t.norm_number("2\u20133") == "2-3"
    assert t.norm_number("II", roman=True) == "2"
    assert t.norm_number("XIV", roman=True) == "14"
    assert t.norm_number("II") == "ii"
    assert t.tag_of("(6)") == "6" and t.tag_of("(A.1)") == "A.1"
    assert t.tag_of("(2.3)") == "2.3" and t.tag_of("(6a)") == "6a"
    for word in ("(x)", "(6),", "6", "(1+x)"):
        assert t.tag_of(word) is None


def test_split_math_drops_prose_math_and_reads_environments(t):
    two = "(c) Mantle for $n=0$ \\quad $(b/a < s < 1)$.\n\n$$\n\\dot z = 1\n$$\n\n$$\n- x\n$$"
    ents, delimited = t.split_math(two)
    assert delimited and [e["latex"] for e in ents] == ["\\dot z = 1", "- x"]
    ents, _ = t.split_math("$a = b,$\n\n$c = d$\n\n$e = f \\times 2$")
    assert [e["latex"] for e in ents] == ["a = b,", "c = d", "e = f \\times 2"]
    ents, _ = t.split_math("$a = b$ (1)\n\nwhere $x$ is a thing")
    assert [e["latex"] for e in ents] == ["a = b"]
    ents, _ = t.split_math(
        "\\begin{equation}\nR = 1\n\\tag{19}\n\\end{equation}\n\n"
        "\\begin{equation}\nS = 2\n\\tag{20}\n\\end{equation}"
    )
    assert ents == [{"latex": "R = 1", "tag": "19"}, {"latex": "S = 2", "tag": "20"}]
    ents, delimited = t.split_math("plain words\n\nmore words")
    assert not delimited and len(ents) == 2


def test_pick_equation(t):
    abc = [{"latex": "a", "tag": None}, {"latex": "b,", "tag": None}]
    abc.append({"latex": "c", "tag": None})
    assert t.pick_equation(abc, ["5", "6", "7"], "6") == (["$b, \\tag{6}$"], "order")
    assert t.pick_equation(abc, ["5", "6", "7"], "(7)")[0] == ["$c \\tag{7}$"]
    assert t.pick_equation(abc, ["5", "6"], "6")[0] == []
    assert t.pick_equation(abc[:2], ["24"], "24") == (
        ["$a$", "$b, \\tag{24}$"],
        "single-number",
    )
    tagged = [{"latex": "R", "tag": "19"}, {"latex": "S", "tag": "20"}]
    assert t.pick_equation(tagged, ["19"], "20") == (["$S \\tag{20}$"], "model-tag")
    assert t.pick_equation(abc, ["5", "6", "7"], "9")[0] == []


def test_number_check(t):
    layer = ["x", "=", "2", "\u00d7", "10", "\u22128", "3.5"]
    assert t.number_check(layer, ["x = 2 \\times 10^{-8} 3.5"]) == []
    layer = ["x", "=", "2", "10", "8", "3.5"]
    assert t.number_check(layer, ["x = 2 \\times 10^{-8} 3.6"]) == ["3.5"]


def test_citing_sentences(t):
    pages = [
        "We start from the widget model. The drift follows Eq. (2) and\n"
        "further resid-\nual terms are small.",
        "Noise was added according to eqs ( 6 ) and ( 7 ), respectively. "
        "Next we turn to the sprocket data.",
        "Results for Equations 4\u20136 are in Table 3. "
        "Eqs. (10), (11) and (12) were fitted, but not (13).",
    ]
    found = t.citing_sentences(pages, "2")
    assert found == [
        {
            "page": 1,
            "text": "The drift follows Eq. (2) and further residual terms are small.",
        }
    ]
    found = t.citing_sentences(pages, "7")
    assert found[0]["page"] == 2 and "Next we turn" not in found[0]["text"]
    assert found[0]["text"].startswith("Noise was added according to eqs ( 6 ) and")
    assert t.citing_sentences(pages, "5")[0]["page"] == 3  # a range
    assert t.citing_sentences(pages, "11")[0]["page"] == 3  # a list
    assert t.citing_sentences(pages, "13") == []
    assert t.citing_sentences(pages, "99") == []
    # a debris line that merely holds the number is not a citation
    assert t.citing_sentences(["x 2 10 \u22128 f (2)"], "2") == []


def test_table_cells(t):
    clean = t.clean_cell_line
    assert clean("Range (m/s<sup>2</sup>)") == "Range (m/s\u00b2)"
    assert clean("6.6_\u00d7_10<sup>_\u2212_6</sup>") == "6.6 \u00d7 10\u207b\u2076"
    assert clean("(m/s<sup>2</sup>/<sup>_\u25e6_</sup>C)") == "(m/s\u00b2/\u00b0C)"
    assert (
        clean("n (m/s<sup>2</sup>/<sup>_\u221a_</sup>Hz)") == "n (m/s\u00b2/\u221aHz)"
    )
    assert clean("~~Volume (L)~~ **bold** _it_") == "Volume (L) bold it"
    assert clean("x<sup>a+b</sup> y<sub>3</sub> z<sub>ab</sub>") == (
        "x^(a+b) y\u2083 z_(ab)"
    )
    assert clean("Variation (km<sup>_\u2212_</sup><sup>1</sup>)") == (
        "Variation (km\u207b\u00b9)"
    )
    assert clean("Noise foor level", {"floor", "noise", "level"}) == "Noise floor level"
    assert clean("Offer of coffee", {"offer", "of", "coffee"}) == "Offer of coffee"
    assert t.join_label_lines(
        ["Mass", "(kg)", "Noise level", "in band", "a) first"]
    ) == ["Mass (kg)", "Noise level in band", "a) first"]


BLOCK = (
    "|**Item**|**Mass (kg)**|**Cost (EUR)**|\n|---|---|---|\n"
    "|Gizmo<br>Sprocket<br>(spare)<br>Flange|4.2<br>1.1<br>0.7|10<br>20<br>30|"
)


def test_table_split_by_order(t):
    rows = t.parse_pipe_table(BLOCK)
    assert rows[0] == ["**Item**", "**Mass (kg)**", "**Cost (EUR)**"]
    assert len(rows) == 2
    tidy = t.tidy_table(rows)
    assert tidy["rows"] == [
        ["Item", "Mass (kg)", "Cost (EUR)"],
        ["Gizmo", "4.2", "10"],
        ["Sprocket (spare)", "1.1", "20"],
        ["Flange", "0.7", "30"],
    ]
    assert (tidy["split_rows"], tidy["unsplit_rows"]) == (1, 0)
    # unequal line counts: the row stays whole, its lines joined by <br>
    bad = t.tidy_table(
        t.parse_pipe_table(BLOCK.replace("4.2<br>1.1<br>0.7", "4.2<br>1.1"))
    )
    assert (bad["split_rows"], bad["unsplit_rows"]) == (0, 1)
    assert bad["rows"][1] == [
        "Gizmo<br>Sprocket<br>(spare)<br>Flange",
        "4.2<br>1.1",
        "10<br>20<br>30",
    ]
    plain = "|a|b|\n|---|---|\n|x \\| y|2|\n|z|3|"
    tidy = t.tidy_table(t.parse_pipe_table(plain))
    assert tidy["rows"] == [["a", "b"], ["x | y", "2"], ["z", "3"]]
    assert tidy["split_rows"] == 0


def test_table_blocks_and_nearest_box(t):
    plain = "|a|b|\n|---|---|\n|z|3|"
    tabs = t.md_tables("Text\n\n" + BLOCK + "\n\nMore text\n\n" + plain + "\n")
    assert len(tabs) == 2 and tabs[0].startswith("|**Item**")
    assert tabs[1].endswith("|z|3|")
    assert (
        t.match_md_table(tabs, "Item Mass Cost Gizmo Sprocket spare Flange 4.2")
        == (tabs[0])
    )
    assert t.match_md_table(tabs, "Entirely other words here") is None
    cap = (90, 90, 540, 105)
    boxes = [(90, 115, 540, 200), (90, 600, 540, 700), (400, 300, 540, 320)]
    assert t.nearest_box(cap, boxes, 800) == (90, 115, 540, 200)
    below = (90, 210, 540, 225)  # a caption under its table
    assert t.nearest_box(below, [(90, 115, 540, 200)], 800) == (90, 115, 540, 200)
    assert t.nearest_box(cap, [(90, 600, 540, 700)], 800) is None  # too far
    assert t.nearest_box((0, 90, 100, 105), [(300, 115, 540, 200)], 800) is None


# ---------------------------------------------------------------- PDFs and a stand-in server

STUB = """#!{python}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv[1:]
port = int(a[a.index("--port") + 1])
assert "--device" in a and a[a.index("--device") + 1] == "none" and a[a.index("-ngl") + 1] == "0", a
assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1" and "LD_LIBRARY_PATH" not in os.environ
open(os.environ["STUB_PIDFILE"], "a").write(str(os.getpid()) + "\\n")
mode = os.environ.get("STUB_MODE", "ok")
if mode == "die":
    print("boom: cannot load model"); sys.exit(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{{}}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = json.loads(self.rfile.read(n))
        assert body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        if mode == "slow": time.sleep(60)
        text = os.environ.get("STUB_TEXT", "$x = y + z$")
        fin = os.environ.get("STUB_FINISH", "stop")
        d = json.dumps({{"choices": [{{"message": {{"content": text}}, "finish_reason": fin}}]}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


class Stub:
    """A stand-in llama-server, and the pids it has been started with."""

    def __init__(self, t, monkeypatch, tmp_path):
        self.pids_file = tmp_path / "pids"
        self.script = tmp_path / "fake-llama-server"
        self.script.write_text(STUB.format(python=sys.executable))
        self.script.chmod(0o755)
        (tmp_path / "model.gguf").write_bytes(b"x")
        (tmp_path / "mmproj.gguf").write_bytes(b"x")
        monkeypatch.setattr(t, "LLAMA_SERVER", str(self.script))
        monkeypatch.setattr(t, "OCR_MODEL", str(tmp_path / "model.gguf"))
        monkeypatch.setattr(t, "OCR_MMPROJ", str(tmp_path / "mmproj.gguf"))
        monkeypatch.setenv("STUB_PIDFILE", str(self.pids_file))
        monkeypatch.setenv("STUB_MODE", "ok")

    def started(self):
        return len(self.pids_file.read_text().split()) if self.pids_file.exists() else 0

    def alive(self):
        out = []
        for pid in (
            self.pids_file.read_text().split() if self.pids_file.exists() else []
        ):
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split()[2]
            except OSError:
                continue
            if state != "Z":
                out.append(int(pid))
        return out


@pytest.fixture
def stub(t, monkeypatch, tmp_path):
    s = Stub(t, monkeypatch, tmp_path)
    yield s
    for pid in s.alive():  # never leave one behind, whatever failed
        os.kill(pid, signal.SIGKILL)


MD = (
    "|**Item**|**Mass**|**Cost**|\n|---|---|---|\n"
    "|Gizmo<br>Sprocket<br>(spare)<br>Flange|4.2<br>1.1<br>0.7|10<br>20<br>30|"
)


def fake_layout(doc, pno, cache):
    return {
        0: {"boxes": [["text", 70, 85, 540, 105]], "tables": []},
        1: {"boxes": [["formula", 100, 280, 540, 340]], "tables": []},
        2: {
            "boxes": [["caption", 88, 90, 300, 105], ["table", 85, 115, 540, 160]],
            "tables": [MD],
        },
    }[pno]


@pytest.fixture
def pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (72, 100),
        "We start from the widget model. The drift follows Eq. (2) and further "
        "terms are small.",
    )
    p2 = doc.new_page()
    p2.insert_text((72, 100), "Section 2. Widgets obey the following law.")
    p2.insert_text((120, 300), "x = y + z")
    p2.insert_text((520, 300), "(2)")
    p2.insert_text((120, 330), "u = v")
    p2.insert_text((520, 330), "(3)")
    p3 = doc.new_page()
    p3.insert_text((90, 100), "Table 1. Invented widget masses.")
    words = "Item Mass Cost Gizmo Sprocket spare Flange 4.2 1.1 0.7 10 20 30".split()
    for i, w in enumerate(words):
        p3.insert_text((90 + 30 * (i % 12), 130 + 12 * (i // 12)), w)
    p3.insert_text((90, 400), "Table 2 shows nothing else of interest here.")
    for k in range(3):  # prose pages: nothing the tools look for
        doc.new_page().insert_text(
            (72, 100), f"Discussion part {k}: the widgets stay put in all cases."
        )
    path = tmp_path / "invented.pdf"
    doc.save(path)
    return str(path)


def test_equation_with_ocr_picks_the_numbered_formula(t, stub, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("STUB_TEXT", "$x = y + z,$\n\n$u = v$")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "ocr" and d["equation"] == ["$u = v \\tag{3}$"]
    assert d["pick"] == "order" and d["page"] == 2 and d["box_tags"] == ["2", "3"]
    assert not d["scanned"] and d["missing_numbers"] == []
    assert "x = y + z" in d["text_layer"] and d["citing"] == []
    assert not stub.alive()
    # the same box again comes from the cache: no second server
    d = t.cmd_equation(pdf, "(2)")
    assert d["ocr_cached"] and d["equation"] == ["$x = y + z, \\tag{2}$"]
    assert d["citing"][0]["page"] == 1 and "Eq. (2)" in d["citing"][0]["text"]
    assert stub.started() == 1
    # a replaced model file does not reuse the old transcriptions
    Path(t.OCR_MODEL).write_bytes(b"a different, longer model file")
    d = t.cmd_equation(pdf, "(2)")
    assert not d["ocr_cached"] and stub.started() == 2


def test_equation_with_unmatched_formulas_shows_the_block(t, stub, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("STUB_TEXT", "$only one$")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "ocr-unmatched" and d["ocr_block"] == ["$only one$"]


def test_equation_in_a_box_the_model_called_a_table(t, stub, pdf, monkeypatch):
    monkeypatch.setenv("STUB_TEXT", "$x = y + z,$\n\n$u = v$")
    monkeypatch.setattr(
        t,
        "layout_page",
        lambda doc, pno, cache: {
            "boxes": [["table", 100, 285, 540, 340]] if pno == 1 else [],
            "tables": [],
        },
    )
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "ocr" and d["box_class"] == "table"
    assert d["equation"] == ["$u = v \\tag{3}$"]
    # a tall text box that merely ends a line with the number is not an equation
    monkeypatch.setattr(
        t,
        "layout_page",
        lambda doc, pno, cache: {
            "boxes": [["text", 100, 150, 540, 340]] if pno == 1 else [],
            "tables": [],
        },
    )
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "no formula box" in d["reason"]


def test_equation_falls_back_to_the_labelled_text_layer(t, stub, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("LIGHTRAG_MCP_OCR", "off")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "switched off" in d["reason"]
    assert "u = v" in d["text_layer"]
    monkeypatch.delenv("LIGHTRAG_MCP_OCR")
    monkeypatch.setenv("STUB_MODE", "die")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "exited with 3" in d["reason"]
    assert "boom" in d["reason"]
    monkeypatch.setenv("STUB_MODE", "ok")
    monkeypatch.setenv("STUB_FINISH", "length")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "did not finish" in d["reason"]
    monkeypatch.setattr(t, "LLAMA_SERVER", "")
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "not found on this host" in d["reason"]
    assert not stub.alive()


def test_a_hung_server_is_stopped_at_the_budget(t, stub, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("STUB_MODE", "slow")
    monkeypatch.setattr(t, "OCR_BUDGET_S", 6)
    started = time.monotonic()
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "timed out" in d["reason"]
    assert time.monotonic() - started < 25
    assert not stub.alive(), "the hung server was left running"


def test_equation_not_found_and_page_filter(t, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    d = t.cmd_equation(pdf, "9")
    assert d["status"] == "not-found" and d["numbers_seen"] == ["2", "3"]
    assert t.cmd_equation(pdf, "3", only_page=1)["status"] == "not-found"


def test_equation_without_the_layout_model_gives_the_text_near_the_number(
    t, pdf, monkeypatch
):
    monkeypatch.setattr(t, "layout_page", lambda doc, pno, cache: None)
    d = t.cmd_equation(pdf, "3")
    assert d["status"] == "text-layer" and "no formula box" in d["reason"]
    assert "u = v" in d["text_layer"]


def test_table_rows_are_split_by_order(t, pdf, monkeypatch):
    monkeypatch.setattr(t, "layout_page", fake_layout)
    d = t.cmd_table(pdf, "1")
    assert d["status"] == "table" and d["page"] == 3
    assert d["caption"] == "Table 1. Invented widget masses."
    assert d["rows"] == [
        ["Item", "Mass", "Cost"],
        ["Gizmo", "4.2", "10"],
        ["Sprocket (spare)", "1.1", "20"],
        ["Flange", "0.7", "30"],
    ]
    assert t.cmd_table(pdf, "I")["status"] == "table"  # roman = arabic
    # "Table 2 shows ..." in prose is not a caption
    d = t.cmd_table(pdf, "2")
    assert d["status"] == "not-found" and d["captions_seen"] == ["1"]
    monkeypatch.setattr(
        t, "layout_page", lambda doc, pno, cache: {"boxes": [], "tables": []}
    )
    assert t.cmd_table(pdf, "1")["status"] == "no-table"


def test_page_scans_are_cached_and_survive_a_cut_off_run(t, pdf, monkeypatch):
    """Page words and texts are read once per file: a retry (or the next question about the
    same paper) does not re-read every page, and a run that is cut off keeps its progress."""
    pymupdf = pytest.importorskip("pymupdf")
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("LIGHTRAG_MCP_OCR", "off")
    real = pymupdf.Page.get_text
    calls = []

    def spy(self, option="text", *a, **k):
        calls.append((self.number, option))
        return real(self, option, *a, **k)

    def cut_off(self, option="text", *a, **k):
        if option == "words" and self.number >= 3:
            raise RuntimeError("cut off")
        return spy(self, option, *a, **k)

    monkeypatch.setattr(pymupdf.Page, "get_text", cut_off)
    with pytest.raises(RuntimeError):
        t.cmd_equation(pdf, "3")
    assert sorted(t.Cache(pdf).data["scan"]) == ["0", "1", "2"]  # kept on the way out
    calls.clear()
    monkeypatch.setattr(pymupdf.Page, "get_text", spy)
    assert (
        t.cmd_equation(pdf, "3")["status"] == "text-layer"
    )  # the retry reads what is missing
    assert sorted(p for p, o in calls if o == "words") == [1, 3, 4, 5]
    # each page's text once (page 1 is also read through a clip for the box)
    assert sorted(p for p, o in calls if o == "text" and p != 1) == [0, 2, 3, 4, 5]
    calls.clear()
    d = t.cmd_equation(pdf, "2")  # everything is cached now
    assert sorted(p for p, o in calls if o == "words") == [1]
    assert [c for c in calls if c[1] == "text" and c[0] != 1] == []
    assert d["citing"][0]["page"] == 1
    calls.clear()
    t.cmd_table(pdf, "1")
    assert sum(1 for c in calls if c[1] == "blocks") == 6
    calls.clear()
    t.cmd_table(pdf, "1")
    assert [c for c in calls if c[1] == "blocks"] == []


def test_a_stale_scan_does_not_crash_the_answer(t, pdf, monkeypatch):
    """The cached scan names a number the live words lack: a labelled text-layer answer,
    not a StopIteration turned into a generic error."""
    monkeypatch.setattr(t, "layout_page", fake_layout)
    monkeypatch.setenv("LIGHTRAG_MCP_OCR", "off")
    t.cmd_equation(pdf, "3")
    cache = t.Cache(pdf)
    cache.data["scan"]["1"]["tags"] = [["9", 520.0, 310.0, 540.0, 320.0]]
    cache.save()
    d = t.cmd_equation(pdf, "9")
    assert d["status"] == "text-layer" and "no formula box" in d["reason"]
    assert d["number_label"] == "9" and d["page"] == 2


def test_lock_fd_is_closed_when_the_wait_runs_out(t, stub):
    import fcntl

    os.makedirs(t.CACHE_DIR, exist_ok=True)
    with open(Path(t.CACHE_DIR) / "ocr.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        srv = t.OcrServer(time.monotonic() + 0.3)
        with pytest.raises(t.OcrUnavailable, match="held the model"):
            srv.__enter__()
        assert srv._lock.closed, "the lock file was left open"


def _alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_helper_dies_with_its_parent(t, stub, tmp_path):
    """SIGKILL of the proxy side: the helper gets SIGTERM from the kernel, unwinds, and its
    OCR server goes with it (the helper holds the OCR lock, so it must not run on)."""
    mid_pid = tmp_path / "mid.pid"
    mid = f"""
import importlib.util, os, signal, sys, time
spec = importlib.util.spec_from_file_location("h", {str(HELPER_PY)!r})
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
t._die_with_parent()
t.CACHE_DIR = {t.CACHE_DIR!r}
t.LLAMA_SERVER, t.OCR_MODEL, t.OCR_MMPROJ = {t.LLAMA_SERVER!r}, {t.OCR_MODEL!r}, {t.OCR_MMPROJ!r}
with t.OcrServer(time.monotonic() + 30):
    open({str(mid_pid)!r}, "w").write(str(os.getpid()))
    print("up", flush=True)
    time.sleep(60)
"""
    top = f"""
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", {mid!r}], stdout=subprocess.PIPE, text=True)
print(p.stdout.readline().strip(), flush=True)
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", top], stdout=subprocess.PIPE, text=True
    )
    assert parent.stdout.readline().strip() == "up"
    helper = int(mid_pid.read_text())
    assert _alive(helper) and len(stub.alive()) == 1
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    for _ in range(50):
        if not _alive(helper) and not stub.alive():
            break
        time.sleep(0.1)
    assert not _alive(helper), "the helper outlived its parent"
    assert not stub.alive(), "the OCR server outlived the helper"


def test_a_parent_that_is_already_gone_is_caught(t):
    """PR_SET_PDEATHSIG only fires on a death AFTER it is armed: a process reparented before
    that must notice by comparing pids (the helper SIGTERMs itself; the OCR server SIGKILLs
    itself)."""
    child = f"""
import importlib.util, os, signal, sys, time
spec = importlib.util.spec_from_file_location("h", {str(HELPER_PY)!r})
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
signal.signal(signal.SIGTERM, lambda *_: (print("term", flush=True), sys.exit(0)))
t._die_with_parent(os.getppid() + int(sys.argv[1]))
time.sleep(1.5)
print("survived", flush=True)
"""
    run = [sys.executable, "-c", child]
    # a recorded parent that is not the live one: reparented, so the helper stops itself
    gone = subprocess.run([*run, "1"], capture_output=True, text=True, timeout=30)
    assert gone.stdout.strip() == "term"
    kept = subprocess.run([*run, "0"], capture_output=True, text=True, timeout=30)
    assert kept.stdout.strip() == "survived"
    # the OCR server's preexec: a parent other than the real one, so the child kills itself
    p = subprocess.Popen(
        ["sleep", "30"], preexec_fn=lambda: t._pdeathsig(os.getpid() + 1)
    )
    assert p.wait(timeout=10) == -signal.SIGKILL
    p = subprocess.Popen(
        ["sleep", "0.3"], preexec_fn=lambda parent=os.getpid(): t._pdeathsig(parent)
    )
    assert p.wait(timeout=10) == 0


def test_ocr_server_dies_with_its_caller(t, stub, tmp_path):
    """SIGKILL of the helper (the proxy's timeout) must not leave the OCR server behind."""
    code = f"""
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("h", {str(HELPER_PY)!r})
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
t.CACHE_DIR = {t.CACHE_DIR!r}
t.LLAMA_SERVER, t.OCR_MODEL, t.OCR_MMPROJ = {t.LLAMA_SERVER!r}, {t.OCR_MODEL!r}, {t.OCR_MMPROJ!r}
with t.OcrServer(time.monotonic() + 30):
    print("up", flush=True)
    time.sleep(60)
"""
    p = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
    )
    assert p.stdout.readline().strip() == "up"
    assert len(stub.alive()) == 1
    p.send_signal(signal.SIGKILL)
    p.wait()
    for _ in range(50):
        if not stub.alive():
            break
        time.sleep(0.1)
    assert not stub.alive(), "the OCR server survived its caller"


def test_helper_runs_at_lower_priority():
    code = f"""
import importlib.util, os
spec = importlib.util.spec_from_file_location("h", {str(HELPER_PY)!r})
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
before = os.nice(0)
t._be_nice()
print(before, os.nice(0))
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    before, after = map(int, r.stdout.split())
    assert after == min(19, before + 10)


def test_cli_prints_one_json_line_even_on_failure(tmp_path):
    pytest.importorskip("pymupdf")
    run = [sys.executable, "-I", str(HELPER_PY)]
    r = subprocess.run(
        [*run, "table", str(tmp_path / "nope.pdf"), "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = json.loads(r.stdout)
    assert r.returncode == 0 and out["status"] == "error"
    assert "no such file" in out["message"]
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    r = subprocess.run(
        [*run, "equation", str(bad), "1"], capture_output=True, text=True, timeout=60
    )
    assert json.loads(r.stdout)["status"] == "error"
    assert len(r.stdout.strip().splitlines()) == 1
