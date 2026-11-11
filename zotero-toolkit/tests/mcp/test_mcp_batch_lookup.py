"""Offline tests for find_paper with several lookups in one call, the age note on a
lookup that finds nothing, the locks that make the shared metadata state safe for the
batch's threads, and the short health answer.

All keys, names and titles here are invented. The fake LightRAG server is conftest's;
the paper lookup route is replaced by a configurable one that can delay or fail chosen
queries and records which lookups ran at once.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

NOTE_START = "The Zotero item list was built "
DAILY = " and is rebuilt once a day"
OVERDUE = "; it should be rebuilt daily and is overdue"


def _item(title, creators, year, doi=""):
    return {
        "title": title,
        "authors": [f"{f} {g}".strip() for f, g in creators],
        "creators": [list(c) for c in creators],
        "year": year,
        "doi": doi,
        "publication": "Journal of Invented Results",
        "tags": [],
        "abstract": "",
    }


ITEMS = {
    # the first three are indexed on the fake server, the rest are only in Zotero
    "AAAA0001": _item(
        "Weighing Gizmo Batches for the Widget Bakery",
        [("Quillfeather", "Ada"), ("Dirimoglu", "Nermin")],
        "2021",
        doi="10.9999/gizmo.2021.1",
    ),
    "BBBB0002": _item(
        "Tending the Orchards Widget Budget",
        [("Brightwater", "Ezra"), ("Thornfield", "Ivo")],
        "1993",
        doi="10.9999/widget.1993.2",
    ),
    "CCCC0003": _item(
        "Calibration Errors in Slanted Sprocket Counts",
        [("Larkspur", "Nell")],
        "2008",
        doi="10.9999/sprocket.2008.3",
    ),
    "DDDD0004": _item(
        "Levelling the Gadget Integral",
        [("Marrowby", "Orin"), ("Oakhurst", "Tess")],
        "2016",
    ),
    "EEEE0005": _item(
        "Widget and Gadget Primer Only in Zotero", [("Ashgrove", "Lia")], "2012"
    ),
    "FFFF0006": _item(
        "Gadget Second Primer, Also Unindexed", [("Ashgrove", "Lia")], "2013"
    ),
}
INDEXED = ("AAAA0001", "BBBB0002", "CCCC0003")


def _paper(key):
    item = ITEMS[key]
    family, given = item["creators"][0]
    return {
        "paper": item["doi"] or key,
        "doi": item["doi"],
        "keys": [key],
        "title": item["title"],
        "citation": f"{family}, {given} ({item['year']}). {item['title']}",
        "year": item["year"],
    }


class Search:
    """GET /graph/papers/search over the indexed papers: an exact key or DOI, else every
    word in the title or citation. `delay` and `fail` act on a q= query; `finished` is
    the order in which q= queries ended and `most` the most that were in flight at once.
    `nosearch` answers 404, as a stock LightRAG Server does."""

    def __init__(self):
        self.index = [_paper(k) for k in INDEXED]
        self.delay: dict[str, float] = {}
        self.fail: set[str] = set()
        self.finished: list[str] = []
        self.inflight = self.most = 0
        self.nosearch = False
        self.lock = threading.Lock()

    def __call__(self, params):
        if self.nosearch:
            return 404, {"detail": "Not Found"}, {}
        headers = {"X-Paper-Search": "2"}
        if "keys" in params:
            want = set(params["keys"].split(","))
            return 200, [p for p in self.index if want & set(p["keys"])], headers
        q = params["q"]
        with self.lock:
            self.inflight += 1
            self.most = max(self.most, self.inflight)
        try:
            if q in self.fail:
                return 500, {"detail": "boom"}, {}
            time.sleep(self.delay.get(q, 0))
            if q == "everything":  # a list the server cuts at its limit
                papers = [
                    {"keys": [f"ZZZZ{i:04d}"], "title": f"Widget {i}"}
                    for i in range(100)
                ]
                return 200, papers, headers
            exact = [p for p in self.index if q.upper() in p["keys"]]
            words = q.lower().split()
            loose = [
                p
                for p in self.index
                if words
                and all(w in f"{p['title']} {p['citation']}".lower() for w in words)
            ]
            return 200, exact or loose, headers
        finally:
            with self.lock:
                self.inflight -= 1
                self.finished.append(q)


@pytest.fixture
def search(fake_lightrag, metadata_file):
    metadata_file.write_text(json.dumps(ITEMS), encoding="utf-8")
    route = Search()
    fake_lightrag.routes[("GET", "/graph/papers/search")] = route
    return route


@pytest.fixture
def srv(load_server, search):
    return load_server()


def _plain(srv, paper, **kw):
    return srv._lookup_text(paper, *srv._lookup(paper), **kw)


def _lines(out):
    """(header, numbered lines, footer) of a batch answer."""
    parts = out.split("\n")
    return parts[0], parts[1:-1], parts[-1]


# --- one line is the old lookup ---------------------------------------------------


@pytest.mark.parametrize(
    "paper", ["AAAA0001", "DDDD0004", "AAAA0001\n", "  \nDDDD0004\n  "]
)
def test_one_line_is_the_plain_lookup_text(srv, paper):
    out = srv.find_paper(paper)
    assert out == _plain(srv, paper)
    assert NOTE_START not in out and not out.startswith("Batch lookup")


@pytest.mark.parametrize("paper", ["Nobody 1850 nothing", "Larkspur 2009 calibration"])
def test_a_miss_differs_from_the_plain_text_only_by_the_age_note(srv, paper):
    note = srv._meta_age_note()
    assert note.startswith(NOTE_START)
    base = _plain(srv, paper)
    out = srv.find_paper(paper)
    assert out != base and out.replace(" " + note, "", 1) == base
    if "Near matches" in out:  # the note ends the paragraph, before the near matches
        assert out.index(note) < out.index("Near matches")


def test_no_other_tool_adds_the_age_note(srv):
    assert NOTE_START not in srv.paper_table("Nobody 1850 nothing", "1")
    assert NOTE_START not in srv.paper_passages("Nobody 1850 nothing", "q")


@pytest.mark.parametrize("sep", [" ", "\x0b", "\x0c", "\x85", " ", "\x1e"])
def test_only_newline_forms_split_lines(srv, sep):
    # text copied from a PDF carries these inside one title: still one lookup
    paper = f"Brightwater{sep}1993"
    out = srv.find_paper(paper)
    assert not out.startswith("Batch lookup")
    assert out == _plain(srv, paper, age_note=srv._meta_age_note())


@pytest.mark.parametrize("sep", ["\n", "\r\n", "\r"])
def test_newline_forms_that_split_lines(srv, sep):
    out = srv.find_paper(f"AAAA0001{sep}DDDD0004")
    assert out.startswith("Batch lookup of 2 ")


# --- the batch ----------------------------------------------------------------------


def _record_lookups(srv, monkeypatch):
    calls = []
    real = srv._lookup
    monkeypatch.setattr(srv, "_lookup", lambda q: calls.append(q) or real(q))
    return calls


def test_a_mixed_batch_answers_one_line_each_in_input_order(srv, monkeypatch):
    calls = _record_lookups(srv, monkeypatch)
    out = srv.find_paper(
        "\nAAAA0001\n\n  DDDD0004  \nNobody 1850 nothing\nAAAA0001\n   \n"
    )
    head, lines, foot = _lines(out)
    assert sorted(calls) == sorted({"AAAA0001", "DDDD0004", "Nobody 1850 nothing"})
    assert head.startswith("Batch lookup of 3 ")
    assert "1 repeated line(s) were looked up once" in head
    assert lines[0].startswith(
        '1. "AAAA0001" → INDEXED (1): - AAAA0001 · Quillfeather, Ada (2021)'
    )
    assert lines[1].startswith(
        '2. "DDDD0004" → IN ZOTERO, NOT INDEXED (1): - DDDD0004 · '
    )
    assert lines[2] == '3. "Nobody 1850 nothing" → NO ITEM HAS EVERY WORD'
    assert len(lines) == 3
    assert "does not show the paper is missing" in foot
    assert "BibTeX keys of web pages" in foot
    assert srv._meta_age_note() in foot
    assert foot.endswith("Look one item up alone for its full listing.")


def test_every_status_a_batch_line_can_carry(srv, search, monkeypatch):
    out = srv.find_paper(
        "widget\neverything\nLarkspur 2009 calibration\nNobody 1850 nothing"
    )
    got = {
        ln.split(" → ")[0].split('"')[1]: ln.split(" → ")[1] for ln in _lines(out)[1]
    }
    # two indexed papers and one Zotero-only item match
    assert got["widget"].startswith("INDEXED (2): - ")
    assert (
        " [+1 more indexed: look it up alone]; also 1 in Zotero not indexed"
        in got["widget"]
    )
    assert got["everything"] == "100 OR MORE INDEXED PAPERS MATCH: add title words"
    assert got["Larkspur 2009 calibration"].startswith(
        "NO ITEM HAS EVERY WORD; near match, not exact: - CCCC0003 · Larkspur"
    )
    assert got["Larkspur 2009 calibration"].endswith(" · indexed")
    # a server with no paper lookup cannot say which are indexed
    search.nosearch = True
    srv._PAPER_CACHE.clear()
    out = srv.find_paper("DDDD0004\nNobody 1850 nothing")
    lines = _lines(out)[1]
    assert lines[0].startswith(
        '1. "DDDD0004" → INDEX STATUS UNKNOWN (this server has no paper lookup): - DDDD0004 · '
    )
    assert lines[1] == (
        '2. "Nobody 1850 nothing" → INDEX STATUS UNKNOWN (this server has no paper lookup)'
    )


def test_items_the_server_was_not_asked_about_are_never_called_absent(srv):
    srv.CONFIRM_MAX = 1  # one key asked, the other two matches not
    out = srv.find_paper("gadget\nNobody 1850 nothing")
    first = _lines(out)[1][0]
    assert first.startswith('1. "gadget" → IN ZOTERO, NOT INDEXED (1): - ')
    assert "; also 2 in Zotero, index status not checked" in first
    srv.CONFIRM_MAX = 0  # no key asked at all
    out = srv.find_paper("gadget\nNobody 1850 nothing")
    first = _lines(out)[1][0]
    assert first.startswith('1. "gadget" → IN ZOTERO, INDEX STATUS NOT CHECKED (3): - ')
    assert "NO ITEM" not in first


def test_forty_lookups_run_and_the_header_names_the_rest(srv, monkeypatch):
    calls = _record_lookups(srv, monkeypatch)
    out = srv.find_paper("\n".join(f"Nobody{i} 1850" for i in range(45)))
    head = out.split("\n")[0]
    assert sorted(calls) == sorted(f"Nobody{i} 1850" for i in range(40))
    assert "5 more NOT looked up" in head
    assert "at most 40" in head and "second call" in head
    assert '40. "Nobody39 1850"' in out and "41. " not in out


def test_repeated_lines_count_once_toward_the_limit(srv):
    first = [f"Nobody{i} 1850" for i in range(35)]
    head = srv.find_paper("\n".join(first + first[:10])).split("\n")[0]
    assert "NOT looked up" not in head
    assert "10 repeated line(s)" in head and "Batch lookup of 35 " in head
    more = [f"Nobody{i} 1850" for i in range(45)]
    head = srv.find_paper("\n".join(more + more[:5])).split("\n")[0]
    assert "5 more NOT looked up" in head and "5 repeated line(s)" in head
    assert "Batch lookup of 40 " in head


def test_one_failing_lookup_does_not_fail_the_batch(srv, search):
    search.fail = {"boom word"}
    out = srv.find_paper("AAAA0001\nboom word\nDDDD0004")
    head, lines, foot = _lines(out)
    assert lines[1].startswith('2. "boom word" → lookup failed: ')
    assert "500" in lines[1]
    assert lines[0].startswith('1. "AAAA0001" → INDEXED (1)')
    assert lines[2].startswith('3. "DDDD0004" → IN ZOTERO')
    assert len(lines) == 3  # the error text is one line


def test_results_keep_input_order_when_lookups_finish_out_of_order(srv, search):
    search.delay = {"AAAA0001": 0.6}
    out = srv.find_paper("AAAA0001\nBBBB0002\nDDDD0004\nNobody 1850 nothing")
    assert search.finished.index("AAAA0001") > search.finished.index("BBBB0002")
    lines = _lines(out)[1]
    assert [ln.split('"')[1] for ln in lines] == [
        "AAAA0001",
        "BBBB0002",
        "DDDD0004",
        "Nobody 1850 nothing",
    ]
    assert [ln[:2] for ln in lines] == ["1.", "2.", "3.", "4."]


def test_lookups_run_four_at_a_time(srv, search):
    search.delay = {f"Nobody{i} 1850": 0.25 for i in range(12)}
    srv.find_paper("\n".join(f"Nobody{i} 1850" for i in range(12)))
    assert search.most == srv.BATCH_THREADS == 4


def test_a_stalled_server_cannot_hold_the_batch_past_its_budget(srv, search):
    search.delay = {"Nobody1 1850": 1.5, "Nobody2 1850": 1.5}
    srv.BATCH_BUDGET = 0.3
    start = time.monotonic()
    out = srv.find_paper("AAAA0001\nNobody1 1850\nDDDD0004\nNobody2 1850\nBBBB0002")
    assert time.monotonic() - start < 1.0
    lines = _lines(out)[1]
    assert lines[0].startswith('1. "AAAA0001" → INDEXED')
    assert lines[2].startswith('3. "DDDD0004" → IN ZOTERO')
    assert lines[4].startswith('5. "BBBB0002" → INDEXED')
    for n in (2, 4):
        assert lines[n - 1] == (
            f'{n}. "Nobody{n // 2} 1850" → not answered within 0.3 s '
            "(server slow; this is not absence): look it up alone"
        )


def test_lookups_never_started_when_the_budget_ends_are_reported_too(srv, search):
    search.delay = {f"Nobody{i} 1850": 1.5 for i in range(10)}
    srv.BATCH_BUDGET = 0.3
    start = time.monotonic()
    out = srv.find_paper("\n".join(f"Nobody{i} 1850" for i in range(10)))
    assert time.monotonic() - start < 1.0
    assert out.count("not answered within 0.3 s") == 10


def test_an_over_long_answer_is_cut_and_says_how_many_lines(srv):
    srv.MAX_CHARS = 1500
    out = srv.find_paper(
        "\n".join(f"Larkspur {2000 + i} calibration" for i in range(20))
    )
    lines = out.split("\n")
    numbered = [ln for ln in lines if ln[:1].isdigit()]
    cut = [ln for ln in lines if ln.startswith("… ")]
    assert len(out) <= 1500 and 0 < len(numbered) < 20 and len(cut) == 1
    assert cut[0].startswith(f"… {20 - len(numbered)} more lines cut")


# --- the age of the item list -------------------------------------------------------


@pytest.mark.parametrize(
    "hours,shown,tail",
    [
        (9.4, "9 h ago", DAILY),
        (47.9, "47 h ago", DAILY),
        (48.1, "2 d ago", OVERDUE),
        (100, "4 d ago", OVERDUE),
        (-5, "time in the future: clock skew", DAILY),
    ],
)
def test_age_note_in_hours_then_days(srv, metadata_file, hours, shown, tail):
    built = time.time() - hours * 3600
    os.utime(metadata_file, (built, built))
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(built))
    assert srv._meta_age_note() == (
        f"The Zotero item list was built {stamp} ({shown}){tail}: "
        "an item added to Zotero after that is not in it yet."
    )


def test_age_note_is_the_age_of_the_copy_in_use(srv, metadata_file, tmp_path):
    built = time.time() - 30 * 3600
    os.utime(metadata_file, (built, built))
    note = srv._meta_age_note()
    assert "(30 h ago)" in note
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    os.replace(bad, metadata_file)  # a fresh mtime and inode, but it does not parse
    assert srv._meta_age_note() == note and len(srv._metadata()) > 0


def test_age_note_is_empty_when_the_file_is_missing(srv, tmp_path):
    srv.META_PATH = str(tmp_path / "absent.json")
    assert srv._meta_age_note() == ""
    out = srv.find_paper("Nobody 1850 nothing")  # nothing to add: the plain text
    assert NOTE_START not in out and out == _plain(srv, "Nobody 1850 nothing")


# --- shared state ---------------------------------------------------------------------


def test_threads_meeting_a_stale_item_list_read_it_and_build_the_rows_once(
    load_server, metadata_file, tmp_path, monkeypatch
):
    big = {
        f"K{i:07d}": {
            "title": f"Widget budget number {i}",
            "year": "2001",
            "authors": [f"Family{i} Given"],
            "creators": [[f"Family{i}", "Given"]],
            "doi": f"10.1/{i}",
        }
        for i in range(20000)
    }
    metadata_file.write_text(json.dumps(big), encoding="utf-8")
    srv = load_server()
    srv._rows()
    big["K9999999"] = {"title": "A newer item"}
    new = tmp_path / "new.json"
    new.write_text(json.dumps(big), encoding="utf-8")
    os.replace(new, metadata_file)  # the sync's rename: a new inode
    reads, builds = [], []
    real_load = srv.json.load
    monkeypatch.setattr(
        srv.json, "load", lambda f, *a, **k: reads.append(1) or real_load(f, *a, **k)
    )

    class CountingRows(dict):
        def update(self, *a, **k):
            builds.append(1)
            super().update(*a, **k)

    monkeypatch.setattr(srv, "_ROWS", CountingRows(srv._ROWS))
    barrier, seen = threading.Barrier(8), []

    def worker():
        barrier.wait()
        seen.append(srv._rows())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reads) == 1 and len(builds) == 1
    assert len(seen) == 8
    assert all(
        "K9999999" in meta and meta is seen[0][0] and rows is seen[0][1]
        for meta, rows in seen
    )


def test_a_rebuilt_list_with_the_old_mtime_and_size_is_still_reloaded(
    srv, metadata_file, tmp_path
):
    """A copy that keeps the build time (scp -p) can arrive with the mtime and size of
    the file it replaces: the inode of the file the rename puts in place says reload."""
    old = metadata_file.stat()
    other = {
        **ITEMS,
        "EEEE0005": {
            **ITEMS["EEEE0005"],
            "title": "Widget and Gadget Primer Only in Zoterp",
        },
    }
    new = tmp_path / "zotero_metadata.json.tmp.1"
    new.write_text(json.dumps(other), encoding="utf-8")
    os.utime(new, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert new.stat().st_size == old.st_size
    assert new.stat().st_mtime_ns == old.st_mtime_ns
    assert srv._metadata()["EEEE0005"]["title"].endswith("Zotero")
    os.replace(new, metadata_file)
    assert srv._metadata()["EEEE0005"]["title"].endswith("Zoterp")


# --- health ---------------------------------------------------------------------------

HEALTH = {
    "status": "healthy",
    "core_version": "1.4.9",
    "api_version": "0300",
    "pipeline_busy": True,
    "pipeline_active": True,
    "pipeline_scanning": True,
    "pipeline_destructive_busy": False,
    "pipeline_pending_enqueues": 3,
    "configuration": {
        **{f"opt_{i}": "x" * 120 for i in range(54)},
        "llm_model": "chat-model",
        "embedding_model": "embed-model",
        "rerank_model": "rerank-model",
        "role_llm_config": {
            "extract": {"binding": "openai", "model": "extract-model", "host": "h"}
        },
    },
    "llm_queue_status": {
        n: {
            "queue_name": n,
            "max_async": 4,
            "queued": q,
            "running": r,
            "failed_total": f,
        }
        for n, q, r, f in (
            ("extract", 2, 1, 0),
            ("keyword", 0, 0, 0),
            ("query", 0, 0, 0),
            ("vlm", 0, 0, 0),
        )
    },
    "embedding_queue_status": {
        "queued": 5,
        "running": 2,
        "in_flight": 2,
        "failed_total": 3,
        "available": True,
    },
    "rerank_queue_status": {
        "queued": 0,
        "running": 0,
        "in_flight": 0,
        "failed_total": 0,
        "available": False,
    },
}
TAIL = "verbose=True returns the full server report (several KB of configuration)."


def _health_route(fake_lightrag, code, payload):
    fake_lightrag.routes[("GET", "/health")] = lambda body: (code, payload)


def test_health_default_is_a_short_summary(fake_lightrag, load_server):
    _health_route(fake_lightrag, 200, HEALTH)
    srv = load_server()
    out = srv.health()
    assert out == (
        "200 healthy · LightRAG core 1.4.9 (api 0300) · pipeline busy, scanning · 3 pending enqueues\n"
        "queues (queued/running): extract 2/1; keyword 0/0; query 0/0; vlm 0/0; "
        "embedding 5/2, failed 3; rerank 0/0 UNAVAILABLE\n"
        "models: llm chat-model, embedding embed-model, rerank rerank-model, "
        "extract extract-model\n" + TAIL
    )
    assert len(out) < 600 < len(json.dumps(HEALTH))


def test_health_verbose_is_the_whole_report(fake_lightrag, load_server):
    _health_route(fake_lightrag, 200, HEALTH)
    assert load_server().health(verbose=True) == f"200: {json.dumps(HEALTH)}"


def test_health_pipeline_states(fake_lightrag, load_server):
    _health_route(
        fake_lightrag,
        200,
        {
            **HEALTH,
            "pipeline_busy": False,
            "pipeline_scanning": False,
            "pipeline_destructive_busy": True,
        },
    )
    assert "pipeline idle, destructive job running · 3" in load_server().health()


@pytest.mark.parametrize(
    "body",
    [
        {"status": "healthy", "core_version": "1.4.9"},
        {
            k: v
            for k, v in HEALTH.items()
            if k not in ("llm_queue_status", "configuration")
        },
        {
            **HEALTH,
            "llm_queue_status": "x",
            "configuration": None,
            "embedding_queue_status": [],
        },
    ],
)
def test_health_leaves_out_fields_the_server_dropped(fake_lightrag, load_server, body):
    _health_route(fake_lightrag, 200, body)
    out = load_server().health()
    assert out.startswith("200 healthy") and out.endswith(TAIL)
    assert "models:" not in out and "extract 2/1" not in out


def test_health_with_only_a_status_and_a_version(fake_lightrag, load_server):
    _health_route(fake_lightrag, 200, {"status": "healthy", "core_version": "1.4.9"})
    assert load_server().health() == "200 healthy · LightRAG core 1.4.9\n" + TAIL


def test_health_cuts_long_fields_and_ignores_a_boolean_failure_count(
    fake_lightrag, load_server
):
    _health_route(
        fake_lightrag,
        200,
        {
            **HEALTH,
            "configuration": {**HEALTH["configuration"], "llm_model": "m" * 2000},
            "embedding_queue_status": {"queued": 1, "running": 0, "failed_total": True},
        },
    )
    out = load_server().health()
    assert len(out) < 600
    assert "llm " + "m" * 80 + "," in out and "m" * 81 not in out
    assert "embedding 1/0;" in out and "failed True" not in out


def test_health_not_200_or_not_an_object_is_shown_as_it_came(
    fake_lightrag, load_server
):
    srv = load_server()
    _health_route(fake_lightrag, 503, {"detail": "x" * 500})
    assert srv.health() == "503: " + json.dumps({"detail": "x" * 500})[:300]
    _health_route(fake_lightrag, 200, "plain text, not an object")
    assert srv.health() == '200: "plain text, not an object"'
    _health_route(fake_lightrag, 200, b"<html>not json</html>")
    assert srv.health() == "200: <html>not json</html>"
    deep = "[" * 100000 + "]" * 100000
    assert srv._health_summary(200, deep) == "200: " + deep[:300]


def test_health_takes_verbose_as_a_tool_argument(fake_lightrag, load_server):
    import anyio

    _health_route(fake_lightrag, 200, HEALTH)
    srv = load_server()

    async def main():
        tools = {t.name: t for t in await srv.mcp.list_tools()}
        short = await srv.mcp.call_tool("health", {})
        full = await srv.mcp.call_tool("health", {"verbose": True})
        return tools, short, full

    tools, short, full = anyio.run(main)
    assert tools["health"].input_schema["properties"]["verbose"]["type"] == "boolean"
    assert short.content[0].text.startswith("200 healthy")
    assert full.content[0].text == f"200: {json.dumps(HEALTH)}"
