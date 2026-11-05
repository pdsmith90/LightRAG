"""Offline tests of the proxy's paper_equation and paper_table: how they pick the one
paper and its local PDF, how they call the PDF helper and what they say when it fails,
how each helper answer is worded, and both tools over stdio.

The helper is a stand-in script that prints a canned answer, so these tests need no
PyMuPDF and no OCR model. All keys, names and answers are invented.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PY = Path(__file__).resolve().parents[2] / "mcp" / "lightrag_mcp_server.py"


def _item(title, creators, year, doi, filename):
    return {
        "title": title,
        "authors": [f"{f} {g}" for f, g in creators],
        "creators": [list(c) for c in creators],
        "year": year,
        "doi": doi,
        "publication": "Journal of Invented Results",
        "tags": [],
        "abstract": "",
        "filename": filename,
    }


ITEMS = {
    "AAAA0001": _item(
        "Weighing Gizmo Batches",
        [("Quillfeather", "Ada")],
        "2021",
        "10.9999/g.1",
        "A.pdf",
    ),
    "BBBB0002": _item(
        "Tending the Orchard Widget Budget",
        [("Brightwater", "Ezra")],
        "1993",
        "10.9999/w.2",
        "B.pdf",
    ),
    "DDDD0004": _item(
        "Gizmo Notes in a Book", [("Marrowby", "Orin")], "2016", "10.9999/d.4", "D.epub"
    ),
    "EEEE0005": _item(
        "Sprocket Primer Only in Zotero",
        [("Larkspur", "Nell")],
        "2008",
        "10.9999/e.5",
        "E.pdf",
    ),
}
INDEXED = ("AAAA0001", "BBBB0002")  # the rest are only in Zotero

HELPER = """import json, sys
mode, pdf, number = sys.argv[1:4]
if mode == "equation":
    out = {"status": "ocr", "number": number, "number_label": number, "page": 9,
           "pages": 17, "equation": ["$a = b, \\\\tag{%s}$" % number], "pick": "order",
           "box_tags": [number], "ocr_seconds": 1.5, "ocr_model": "fake-ocr.gguf",
           "text_layer": "a = b , (6)", "scanned": False, "missing_numbers": [],
           "citing": [{"page": 9, "text": "The widgets obey eq. (6)."}],
           "other_pages": [], "argv": sys.argv[1:]}
else:
    out = {"status": "table", "number": number, "page": 3, "pages": 17,
           "caption": "Table 3. Widget masses.", "rows": [["Item", "Mass"], ["Gizmo", "4.2"]],
           "split_rows": 0, "unsplit_rows": 0, "argv": sys.argv[1:]}
print(json.dumps(out))
"""


def _paper(key):
    m = ITEMS[key]
    return {
        "paper": m["doi"],
        "doi": m["doi"],
        "keys": [key],
        "title": m["title"],
        "citation": f"{m['creators'][0][0]}, {m['creators'][0][1]} ({m['year']})",
        "year": m["year"],
    }


def _search(params):
    headers = {"X-Paper-Search": "2"}
    if "keys" in params:
        want = set(params["keys"].split(","))
        return 200, [_paper(k) for k in INDEXED if k in want], headers
    q = params.get("q", "")
    return 200, [_paper(k) for k in INDEXED if k == q.upper()], headers


@pytest.fixture
def setup(fake_lightrag, load_server, metadata_file, tmp_path):
    metadata_file.write_text(json.dumps(ITEMS), encoding="utf-8")
    storage = metadata_file.parent / "storage"
    for key in ("AAAA0001", "DDDD0004", "EEEE0005"):
        (storage / key).mkdir(parents=True)
        (storage / key / ITEMS[key]["filename"]).write_bytes(b"%PDF-invented")
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search
    helper = tmp_path / "helper.py"
    helper.write_text(HELPER)

    def load(**env):
        return load_server(
            LIGHTRAG_MCP_PDF_PYTHON=sys.executable,
            LIGHTRAG_MCP_PDF_HELPER=str(helper),
            **env,
        )

    return load, storage, helper


def test_equation_names_the_paper_its_file_and_the_latex(setup):
    load, storage, _ = setup
    out = load().paper_equation("AAAA0001", "6")
    lines = out.splitlines()
    assert lines[0].startswith("### Quillfeather, Ada. (2021). Weighing Gizmo Batches")
    assert lines[1] == f"Local file: {storage / 'AAAA0001' / 'A.pdf'}"
    assert (
        "Equation (6), page 9 of 17: LaTeX read from the PDF image by fake-ocr.gguf"
        in out
    )
    assert "(1.5 s on the CPU)" in out
    assert "\n\n$a = b, \\tag{6}$\n\n" in out
    assert 'Cited on page 9: "The widgets obey eq. (6)."' in out
    assert "Text layer of the same box" in out and "OCR NOT USED" not in out


def test_table_is_a_markdown_table(setup):
    load, _, _ = setup
    out = load().paper_table("AAAA0001", "3")
    assert "Table 3. Widget masses.  (page 3 of 17)" in out
    assert "| Item | Mass |\n|---|---|\n| Gizmo | 4.2 |" in out
    assert "no OCR model" in out


def test_a_paper_only_in_zotero_is_read_from_its_file(setup):
    load, storage, _ = setup
    srv = load()
    seen = []
    real = srv._pdf_tool
    srv._pdf_tool = lambda *a: seen.append(a) or real(*a)
    out = srv.paper_equation("EEEE0005", "2", 4)
    assert seen == [("equation", str(storage / "EEEE0005" / "E.pdf"), "2", 4)]
    assert "Sprocket Primer Only in Zotero" in out and "Equation (2)" in out


def test_no_local_pdf_or_not_a_pdf_is_said_plainly(setup):
    load, _, _ = setup
    srv = load()
    out = srv.paper_equation("BBBB0002", "1")
    assert "no local PDF of this paper" in out and "paper_equation" in out
    out = srv.paper_table("DDDD0004", "1")
    assert "reads PDFs, and this paper's file is not one" in out and "D.epub" in out


def test_no_paper_or_several_ask_for_one_key(setup, monkeypatch):
    load, _, _ = setup
    srv = load()
    out = srv.paper_equation("no such paper anywhere", "1")
    assert "No indexed paper or Zotero item contains every word" in out
    assert "Equation" not in out
    meta = srv._metadata()
    monkeypatch.setattr(
        srv,
        "_lookup",
        lambda paper: (
            None,
            [("DDDD0004", meta["DDDD0004"]), ("EEEE0005", meta["EEEE0005"])],
            False,
            [],
            [],
        ),
    )
    out = srv.paper_table("anything", "1")
    assert "paper_table needs exactly one paper" in out and "EEEE0005" in out


def test_helper_failures_become_sentences(setup, tmp_path):
    load, _, helper = setup
    srv = load()
    pdf = str(tmp_path / "x.pdf")

    def run(body, **kw):
        helper.write_text(body)
        return srv._pdf_tool("table", pdf, "2", **kw)

    assert run(
        "import json,sys; print('noise'); print(json.dumps({'argv': sys.argv[1:]}))"
    )["argv"] == ["table", pdf, "2"]
    assert run("import json,sys; print(json.dumps({'argv': sys.argv[1:]}))", page=9)[
        "argv"
    ][-2:] == ["--page", "9"]
    err = "import json; print(json.dumps({'status':'error','message':'no such file'}))"
    assert "could not read this file: no such file" in run(err)
    out = run("import sys; sys.stderr.write('Traceback: boom'); sys.exit(3)")
    assert "failed (exit 3)" in out and "boom" in out
    assert "failed (exit 0)" in run("print('not json')")
    srv.PDF_TIMEOUT = 1.0
    assert "did not finish in 1 s" in run("import time; time.sleep(30)")
    srv.PDF_PYTHON = str(tmp_path / "no-such-python")
    assert "could not be started" in run("print(1)")


def test_each_helper_answer_is_worded(setup):
    load, _, _ = setup
    srv = load()
    head = ["### Cite\nLocal file: /x.pdf"]
    eq = {
        "status": "ocr",
        "number_label": "6",
        "page": 9,
        "pages": 17,
        "equation": ["$a = b, \\tag{6}$"],
        "pick": "order",
        "ocr_seconds": 25.2,
        "ocr_model": "fake-ocr.gguf",
        "text_layer": "a = b , (6)",
        "scanned": False,
        "missing_numbers": [],
        "citing": [],
        "other_pages": [12, 14],
    }
    out = srv._equation_text(head, "6", eq)
    assert "page(s) 12, 14" in out and 'paper_equation(paper, "6", page=N)' in out
    odd = srv._equation_text(head, "6", {**eq, "box_class": "table"})
    assert 'called this box a "table", not a formula' in odd
    cached = srv._equation_text(
        head, "6", {**eq, "ocr_cached": True, "pick": "single-number"}
    )
    assert "cached" in cached and "one number for the box" in cached
    checked = srv._equation_text(
        head, "6", {**eq, "missing_numbers": ["10", "3.5"], "scanned": True}
    )
    assert "scan" in checked and "not in the LaTeX: 10, 3.5" in checked
    unmatched = srv._equation_text(
        head,
        "6",
        {
            **eq,
            "status": "ocr-unmatched",
            "box_tags": ["5", "6", "7"],
            "ocr_block": ["$a$", "$b$"],
            "pick": "2 formulas for 3 equation number(s)",
        },
    )
    assert "(5), (6), (7)" in unmatched and "$a$\n$b$" in unmatched
    layer = srv._equation_text(
        head,
        "6",
        {
            "status": "text-layer",
            "page": 4,
            "pages": 12,
            "scanned": True,
            "number_label": "6",
            "reason": "llama-server not found on this host (unset)",
            "text_layer": "z . 2 (6)",
        },
    )
    assert "OCR NOT USED (llama-server not found" in layer and "NOT LaTeX" in layer
    assert "Text layer: z . 2 (6)" in layer and "OCR layer" in layer
    assert "LaTeX read from" not in layer
    seen = srv._equation_text(
        head, "9", {"status": "not-found", "numbers_seen": ["1", "2", "A.1"]}
    )
    assert "ends with (9)" in seen and "1, 2, A.1" in seen
    assert "no text layer" in srv._equation_text(head, "9", {"status": "no-text-layer"})
    table = srv._table_text(
        head,
        "6",
        {
            "status": "table",
            "page": 17,
            "pages": 32,
            "caption": "Table 6. Specs.",
            "rows": [["Parameter", "A|B"], ["Mass (kg)", "9.4"], ["short"]],
            "split_rows": 1,
            "unsplit_rows": 2,
        },
    )
    assert (
        "| Parameter | A\\|B |\n|---|---|\n| Mass (kg) | 9.4 |\n| short |  |" in table
    )
    assert "split by order" in table and "2 row(s) hold several lines" in table
    missing = srv._table_text(
        head, "7", {"status": "not-found", "captions_seen": ["1", "2"]}
    )
    assert 'No caption "Table 7" found. Captions the PDF has: 1, 2.' in missing
    nothing = srv._table_text(
        head,
        "7",
        {"status": "no-table", "page": 5, "caption": "Table 7. X.", "reason": "no box"},
    )
    assert "Caption found on page 5" in nothing and "may be a picture" in nothing
    lines = srv._table_text(
        head,
        "7",
        {
            "status": "text-layer",
            "page": 5,
            "pages": 9,
            "caption": "Table 7. X.",
            "reason": "no pipe table",
            "lines": ["a", "1"],
        },
    )
    assert "no pipe table. The lines, in reading order:\na\n1" in lines


def test_stdio_lists_and_answers_both_tools(
    setup, fake_lightrag, metadata_file, api_key, tmp_path
):
    _, storage, helper = setup
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PY)],
        env={
            "LIGHTRAG_BASE_URL": fake_lightrag.url,
            "LIGHTRAG_API_KEY": api_key,
            "ZOTERO_METADATA": str(metadata_file),
            "ZOTERO_STORAGE": str(storage),
            "LIGHTRAG_MCP_PDF_PYTHON": sys.executable,
            "LIGHTRAG_MCP_PDF_HELPER": str(helper),
        },
    )

    async def main():
        with anyio.fail_after(60):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    eq = await session.call_tool(
                        "paper_equation", {"paper": "AAAA0001", "number": "6"}
                    )
                    tb = await session.call_tool(
                        "paper_table", {"paper": "AAAA0001", "number": "3"}
                    )
                    return tools, eq, tb

    tools, eq, tb = anyio.run(main)
    assert {t.name for t in tools.tools} >= {"paper_equation", "paper_table"}
    for res in (eq, tb):
        assert not res.is_error and res.structured_content is None
        assert len(res.content) == 1
    assert "$a = b, \\tag{6}$" in eq.content[0].text
    assert "| Gizmo | 4.2 |" in tb.content[0].text
