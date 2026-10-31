"""Paper search that forgives typography (lightrag/zotero_papers.py).

Zotero titles carry typographic hyphens (U+2010), curly apostrophes and
diacritics that a typed query does not, and agents look papers up by BibTeX
keys such as ``doePruningOrchardsAppleTrees2001``. Both sides of the match are folded
the same way, a BibTeX key is split into its words, and ``keys=`` answers for a
batch of exact storage keys in one request. Every answer of the route carries
``X-Paper-Search: 2`` so a client can tell this server from an older one.
Fixtures are invented. The query and haystack cases that the MCP proxy's lookup
must satisfy too are in zotero-toolkit/tests/mcp/paper_search_cases.json, which
both suites read.
"""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lightrag.zotero_citations as zc
import lightrag.zotero_papers as zp
from lightrag.base import DocStatus

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

SHARED = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "zotero-toolkit/tests/mcp/paper_search_cases.json"
    ).read_text(encoding="utf-8")
)

META = {
    "AAAA0001": {
        "title": "Glazing the pottery with KILN‐3 and TREFOIL‐FO",
        "authors": ["Vargıkömeroğlu Ayşe", "Doe Jane"],
        "year": "2011",
        "doi": "10.9/a",
    },
    "AAAA0002": {  # a second attachment of the same paper
        "title": "Glazing the pottery with KILN‐3 and TREFOIL‐FO",
        "authors": ["Vargıkömeroğlu Ayşe", "Doe Jane"],
        "year": "2011",
        "doi": "10.9/A",
    },
    "BBBB0001": {
        "title": "Pruning the orchard’s apple trees: a primer",
        "authors": ["Doe Jane"],
        "year": "2001",
        "doi": "10.9/b",
    },
    "CCCC0001": {
        "title": "Mulch layers of the PLUM‐7/PEAR‐2 trial",
        "authors": ["Østergaard Lars", "Maßgeber Uwe"],
        "year": "2019",
    },
    "DDDD0001": {
        "title": "Not ingested",
        "authors": ["Doe Jane"],
        "year": "2001",
        "doi": "10.9/d",
    },
}

# doc id -> (corpus file, chunk ids); EEEE0001 is indexed but has no metadata
DOCS = {
    "doc-a1": ("AAAA0001__Glazing.md", ["a1-c0"]),
    "doc-a2": ("AAAA0002__Glazing.md", ["a2-c0"]),
    "doc-b": ("BBBB0001__Pruning.md", ["b-c0"]),
    "doc-c": ("CCCC0001__Mulch.md", ["c-c0"]),
    "doc-e": ("EEEE0001__Untitled.md", ["e-c0"]),
}


class _Status:
    def __init__(self, file_path, chunks):
        self.file_path, self.chunks_list = file_path, chunks


class _DocStatus:
    def __init__(self, docs=None, gate=None):
        self.calls = 0
        self.docs = DOCS if docs is None else docs
        self.gate = gate

    async def get_docs_by_statuses(self, statuses):
        assert statuses == [DocStatus.PROCESSED]
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return {d: _Status(f, c) for d, (f, c) in self.docs.items()}


def _rag(**kw):
    return SimpleNamespace(doc_status=_DocStatus(**kw))


def _reset():
    zp._docs_cache.update(at=0.0, owner=None, by_key={}, pending=None)


@pytest.fixture
def library(tmp_path):
    meta_path = tmp_path / "zotero_metadata.json"
    meta_path.write_text(json.dumps(META), encoding="utf-8")
    saved_meta, saved_refs = zc.METADATA_PATH, zp.REFERENCES_PATH
    zc.METADATA_PATH = str(meta_path)
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    zp.REFERENCES_PATH = str(tmp_path / "absent.json")
    zp._refs, zp._refs_mtime, zp._refs_next_stat, zp._refs_warned = {}, None, 0.0, True
    _reset()
    yield
    zc.METADATA_PATH = saved_meta
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    zp.REFERENCES_PATH = saved_refs
    zp._refs, zp._refs_mtime, zp._refs_next_stat, zp._refs_warned = {}, None, 0.0, False
    _reset()


# --- FOLD --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, folded",
    [
        ("Café Ångström", "cafe angstrom"),
        ("Kılıçöğür", "kilicogur"),
        ("Łódź Øre Đorđe Ðð", "lodz ore dorde dd"),
        ("Straße Æsir Œuvre Þorn þ", "strasse aesir oeuvre thorn th"),
        ("A‐B‑C‒D–E—F―G−H­I", "a-b-c-d-e-f-g-h-i"),
        ("Pear's Pear’s Pear‘s Pearʼs x′ `q`", "pears pears pears pears x q"),
        ("TREFOIL‐FO", "trefoil-fo"),
    ],
)
def test_fold(text, folded):
    assert zp.fold(text) == folded


@pytest.mark.parametrize(
    "word",
    [
        "trefoil-fo",
        "TREFOIL-FO",
        "trefoilfo",
        "trefoil‐fo",
        "kiln-3",
        "kiln3",
        "(kiln-3),",
        '"pottery',
        "vargikomeroglu",
        "Vargıkömeroğlu",
    ],
)
async def test_query_words_match_a_typographic_title(library, word):
    hits = await zp.search_papers(_rag(), word)
    assert [h["paper"] for h in hits] == ["10.9/a"]


async def test_hyphen_slash_parts_and_apostrophes_match(library):
    rag = _rag()
    assert [h["paper"] for h in await zp.search_papers(rag, "plum-7/pear-2")] == [
        "CCCC0001"
    ]
    assert [h["paper"] for h in await zp.search_papers(rag, "orchards apple")] == [
        "10.9/b"
    ]
    assert [h["paper"] for h in await zp.search_papers(rag, "Orchard's")] == ["10.9/b"]
    assert [
        h["paper"] for h in await zp.search_papers(rag, "ostergaard massgeber")
    ] == ["CCCC0001"]
    # every word must still occur
    assert await zp.search_papers(rag, "orchards kiln-3") == []


async def test_a_bibtex_key_names_its_paper(library):
    rag = _rag()
    hits = await zp.search_papers(rag, "doePruningOrchardsAppleTrees2001")
    assert [h["paper"] for h in hits] == ["10.9/b"]
    # a suffix letter, acronym runs and digits split as words
    assert [
        h["paper"]
        for h in await zp.search_papers(rag, "doeGlazingPotteryKilnTrefoil2011a")
    ] == ["10.9/a"]
    assert [
        h["paper"]
        for h in await zp.search_papers(rag, "ostergaardMulchLayersPLUM7PEAR2019")
    ] == ["CCCC0001"]
    # the year and the family name still have to match
    assert await zp.search_papers(rag, "doePruningOrchardsAppleTrees2002") == []
    assert await zp.search_papers(rag, "roePruningOrchardsAppleTrees2001") == []


def test_query_words_split_only_a_bibtex_shaped_token():
    assert zp.query_words("doePruningOrchardsAppleTrees2001") == [
        ("doe", False),
        ("pruning", True),
        ("orchards", True),
        ("apple", True),
        ("trees", True),
        ("2001", False),
    ]
    assert zp.query_words("TREFOILFollowOn") == [("trefoilfollowon", False)]
    assert zp.query_words("kim2001 Glaze") == [("kim2001", False), ("glaze", False)]
    assert zp.query_words("xTREFOILFollowOn2019") == [
        ("x", False),
        ("trefoil", True),
        ("follow", True),
        ("on", True),
        ("2019", False),
    ]


async def test_a_token_without_letters_or_digits_is_not_a_word(library):
    # a file-name style query: "Author - Year - Title"
    assert zp.query_words("Doe - 2001 - Pruning") == [
        ("doe", False),
        ("2001", False),
        ("pruning", False),
    ]
    assert zp.query_words("- \u2010 ()") == []
    rag = _rag()
    assert [
        h["paper"] for h in await zp.search_papers(rag, "Doe - 2001 - Pruning")
    ] == ["10.9/b"]
    assert await zp.search_papers(rag, " - ") == []


# The cases the proxy's lookup must satisfy too (paper_search_cases.json).


@pytest.mark.parametrize("raw, folded", SHARED["fold"])
def test_shared_fold_cases(raw, folded):
    assert zp.fold(raw) == folded


@pytest.mark.parametrize("query, words", SHARED["words"])
def test_shared_query_words(query, words):
    assert [list(w) for w in zp.query_words(query)] == words


@pytest.fixture
def shared_library(tmp_path):
    """Every shared item indexed, one document each."""
    meta_path = tmp_path / "zotero_metadata.json"
    meta_path.write_text(json.dumps(SHARED["items"]), encoding="utf-8")
    saved = zc.METADATA_PATH
    zc.METADATA_PATH = str(meta_path)
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    _reset()
    yield _rag(
        docs={f"doc-{k}": (f"{k}__Paper.md", [f"{k}-c0"]) for k in SHARED["items"]}
    )
    zc.METADATA_PATH = saved
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    _reset()


@pytest.mark.parametrize("query, keys", SHARED["lookups"])
async def test_shared_lookups(shared_library, query, keys):
    hits = await zp.search_papers(shared_library, query)
    assert sorted(k for h in hits for k in h["keys"]) == keys


async def test_existing_matches_still_hold(library):
    rag = _rag()
    assert [h["paper"] for h in await zp.search_papers(rag, "10.9/A")] == ["10.9/a"]
    assert [h["paper"] for h in await zp.search_papers(rag, "bbbb0001")] == ["10.9/b"]
    assert await zp.search_papers(rag, "not ingested") == []
    hits = await zp.search_papers(rag, "doe")
    # title hits first, then the newest
    assert [h["paper"] for h in hits] == ["10.9/a", "10.9/b"]
    assert hits[0]["keys"] == ["AAAA0001", "AAAA0002"]


# --- keys= -------------------------------------------------------------------


async def test_keys_list_the_indexed_papers_carrying_them(library):
    rag = _rag()
    hits = await zp.search_papers(
        rag, "", keys=["aaaa0002", "BBBB0001", "DDDD0001", "ZZZZ9999"]
    )
    assert sorted(h["paper"] for h in hits) == ["10.9/a", "10.9/b"]
    a = next(h for h in hits if h["paper"] == "10.9/a")
    assert a["keys"] == ["AAAA0001", "AAAA0002"]
    # an indexed key without metadata is still there
    assert [h["paper"] for h in await zp.search_papers(rag, "", keys=["EEEE0001"])] == [
        "EEEE0001"
    ]
    # with words too, both have to hold
    assert [
        h["paper"]
        for h in await zp.search_papers(rag, "primer", keys=["AAAA0001", "BBBB0001"])
    ] == ["10.9/b"]


# --- the route ---------------------------------------------------------------


def _client(monkeypatch=None, fake=None):
    if fake is not None:
        monkeypatch.setattr(zp, "search_papers", fake)
    app = FastAPI()
    app.include_router(_graph_routes.create_graph_routes(_rag(), api_key="k"))
    return TestClient(app), {"X-API-Key": "k"}


def test_route_passes_keys_and_marks_every_answer(monkeypatch):
    seen = {}

    async def fake(rag, q, limit=20, keys=None):
        seen.update(q=q, limit=limit, keys=keys)
        return []

    client, h = _client(monkeypatch, fake)
    r = client.get(
        "/graph/papers/search?keys=AAAA0001,%20bbbb0002,,CCCC0003", headers=h
    )
    assert r.status_code == 200 and r.headers["X-Paper-Search"] == "2"
    assert seen == {"q": "", "limit": 3, "keys": ["AAAA0001", "BBBB0002", "CCCC0003"]}
    r = client.get("/graph/papers/search?q=doe", headers=h)
    assert r.status_code == 200 and r.headers["X-Paper-Search"] == "2"
    assert seen == {"q": "doe", "limit": 20, "keys": None}
    client.get("/graph/papers/search?q=doe&limit=7&keys=AAAA0001", headers=h)
    assert seen == {"q": "doe", "limit": 7, "keys": ["AAAA0001"]}


def test_route_refuses_no_query_and_too_many_keys(monkeypatch):
    async def fake(rag, q, limit=20, keys=None):
        raise AssertionError("not reached")

    client, h = _client(monkeypatch, fake)
    r = client.get("/graph/papers/search", headers=h)
    assert r.status_code == 422 and r.headers["X-Paper-Search"] == "2"
    many = ",".join(f"K{i:07d}" for i in range(51))
    r = client.get(f"/graph/papers/search?keys={many}", headers=h)
    assert r.status_code == 422 and r.headers["X-Paper-Search"] == "2"


def test_route_marks_an_error_answer(monkeypatch):
    async def fake(rag, q, limit=20, keys=None):
        raise RuntimeError("boom")

    client, h = _client(monkeypatch, fake)
    r = client.get("/graph/papers/search?q=x", headers=h)
    assert r.status_code == 500 and r.headers["X-Paper-Search"] == "2"


# --- the processed-document list ---------------------------------------------


async def test_concurrent_cold_callers_share_one_load(library):
    gate = asyncio.Event()
    rag = _rag(gate=gate)
    calls = [asyncio.create_task(zp._kb_documents(rag)) for _ in range(5)]
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(*calls)
    assert rag.doc_status.calls == 1
    assert all(r is results[0] for r in results)
    assert "AAAA0001" in results[0]


async def test_a_stale_list_is_served_while_one_refresh_runs(library):
    rag = _rag()
    first = await zp._kb_documents(rag)
    zp._docs_cache["at"] -= zp._DOCS_TTL + 1
    rag.doc_status.docs = dict(DOCS, **{"doc-f": ("FFFF0001__New.md", ["f-c0"])})
    gate = rag.doc_status.gate = asyncio.Event()
    # stale callers get the old list at once; one refresh runs behind them
    assert await asyncio.wait_for(zp._kb_documents(rag), 2) is first
    assert await asyncio.wait_for(zp._kb_documents(rag), 2) is first
    gate.set()
    await zp._docs_cache["pending"][1]
    assert rag.doc_status.calls == 2
    assert "FFFF0001" in await zp._kb_documents(rag)
    assert rag.doc_status.calls == 2


async def test_another_storage_never_gets_a_stale_list(library):
    await zp._kb_documents(_rag())
    other = _rag(docs={"doc-x": ("XXXX0001__X.md", ["x-c0"])})
    assert list(await zp._kb_documents(other)) == ["XXXX0001"]
