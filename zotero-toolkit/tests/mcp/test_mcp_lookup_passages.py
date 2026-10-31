"""Offline tests for the proxy's folded paper lookup, the one-paper passages route,
the corpus header strip and the search_library result size.

All keys, names and titles here are invented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus"


def _item(title, creators, year, doi="", tags=(), abstract=""):
    return {
        "title": title,
        "authors": [f"{f} {g}".strip() for f, g in creators],
        "creators": [list(c) for c in creators],
        "year": year,
        "doi": doi,
        "publication": "Journal of Invented Results",
        "tags": list(tags),
        "abstract": abstract,
    }


ITEMS = {
    # U+2010 hyphens, a right single quote and a dotless i, as Zotero stores them.
    "AAAA0001": _item(
        "Weighing Gizmo\u20107/Sprocket\u20102 Batches for the Widget\u2010FO Bakery",
        [("Quillfeather", "Ada"), ("D\u0131r\u0131mo\u011flu", "Nermin")],
        "2021",
        doi="10.9999/gizmo.2021.1",
    ),
    "BBBB0002": _item(
        "Tending the Orchard\u2019s Widget Budget",
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
}


def _write(metadata_file, items):
    metadata_file.write_text(json.dumps(items), encoding="utf-8")


def _paper(key, item, keys=None):
    family = item["creators"][0][0]
    return {
        "paper": item["doi"] or key,
        "doi": item["doi"],
        "keys": keys or [key],
        "title": item["title"],
        "citation": f"{family}, {item['creators'][0][1]} ({item['year']}). {item['title']}",
        "year": item["year"],
    }


def _search_route(index, version="2", delay=0.0):
    """/graph/papers/search over the given papers. The q leg is a plain lowercase
    substring match (an old server's), so it misses folded forms; keys= answers the
    papers carrying any of the keys. version=None sends no X-Paper-Search header."""

    def route(params):
        if delay:
            time.sleep(delay)
        headers = {"X-Paper-Search": version} if version else {}
        if "keys" in params:
            want = set(params["keys"].split(","))
            return 200, [p for p in index if want & set(p["keys"])], headers
        q = params.get("q", "")
        exact = [p for p in index if q.upper() in p["keys"] or q.lower() == p["doi"]]
        words = q.lower().split()
        loose = [
            p
            for p in index
            if words
            and all(w in f"{p['title']} {p['citation']}".lower() for w in words)
        ]
        return 200, exact or loose, headers

    return route


@pytest.fixture
def items(metadata_file):
    _write(metadata_file, ITEMS)
    return ITEMS


# --- P1: text only, no structured duplicate -------------------------------------


def test_tools_return_text_without_structured_output(load_server, items):
    import anyio

    srv = load_server()

    async def main():
        tools = await srv.mcp.list_tools()
        result = await srv.mcp.call_tool("health", {})
        return tools, result

    tools, result = anyio.run(main)
    assert len(tools) == 5
    assert all(t.output_schema is None for t in tools)
    assert result.structured_content is None
    assert result.content[0].text.startswith("200:")


# --- FOLD and word matching ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,folded",
    [
        ("Widget\u2010FO", "widget-fo"),
        ("Orchard\u2019s", "orchards"),
        ("Orchard's `x` \u2018y\u2019 \u02bcz \u2032", "orchards x y z "),
        ("D\u0131r\u0131mo\u011flu", "dirimoglu"),
        (
            "\u0141uk \u00d8ster \u0110or \u00d0a Stra\u00dfe \u00c6r \u0152uvre \u00deorn",
            "luk oster dor da strasse aer oeuvre thorn",
        ),
        ("a\u2011b\u2012c\u2013d\u2014e\u2015f\u2212g\u00adh", "a-b-c-d-e-f-g-h"),
        ("Caf\u00e9 Na\u00efve \ufb01eld", "cafe naive field"),
    ],
)
def test_fold(load_server, raw, folded):
    assert load_server()._fold(raw) == folded


@pytest.mark.parametrize(
    "word", ["widget-fo", "widgetfo", "sprocket-2", "gizmo-7/sprocket-2", "gizmo7"]
)
def test_query_word_matches_a_unicode_hyphen_title(load_server, word):
    srv = load_server()
    hay = srv._fold(ITEMS["AAAA0001"]["title"])
    assert srv._word_in(word, hay)


def test_query_word_that_is_absent_does_not_match(load_server):
    srv = load_server()
    hay = srv._fold(ITEMS["AAAA0001"]["title"])
    assert not srv._word_in("gizmo-8", hay)
    assert not srv._word_in("sprockets", hay)


def test_bibtex_key_splits_into_family_title_words_and_year(load_server):
    words, title_words, family = load_server()._terms(
        "brightwaterTendingOrchardsWidget1993"
    )
    assert family == "brightwater"
    assert words == ["brightwater", "1993"]
    assert title_words == ["tending", "orchards", "widget"]
    _, acronym, _ = load_server()._terms("quillfeatherGIZMO7WeighingBatches2021b")
    assert acronym == ["gizmo", "7", "weighing", "batches"]


# The cases the server's paper search must satisfy too (paper_search_cases.json,
# which tests/test_paper_search_fold.py in the LightRAG tree reads as well).
SHARED = json.loads(
    (Path(__file__).with_name("paper_search_cases.json")).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("raw,folded", SHARED["fold"])
def test_shared_fold_cases(load_server, raw, folded):
    assert load_server()._fold(raw) == folded


@pytest.mark.parametrize("query,words", SHARED["words"])
def test_shared_query_words(load_server, query, words):
    assert [list(w) for w in load_server()._query_words(query)] == words


@pytest.mark.parametrize("query,keys", SHARED["lookups"])
def test_shared_lookups(load_server, metadata_file, query, keys):
    _write(metadata_file, SHARED["items"])
    found = load_server()._metadata_matches(query)
    assert sorted(k for k, _ in found) == keys


def test_terms_strip_punctuation_and_fold(load_server):
    words, _, _ = load_server()._terms('("Widget\u2010FO", Orchard\u2019s!) -')
    assert words == ["widget-fo", "orchards"]
    # curly quotes, guillemets and emphasis stars come off too, "et al" after a
    # name and a bare "&" are no words
    words, _, _ = load_server()._terms(
        "\u201cWidget\u201d \u00abbudget\u00bb *x* & Doe et al. 2011"
    )
    assert words == ["widget", "budget", "x", "doe", "2011"]


# --- P2: local lookup, confirmation, cache -------------------------------------


@pytest.mark.parametrize(
    "query,key",
    [
        ("Quillfeather Widget-FO", "AAAA0001"),
        ("gizmo-7/sprocket-2 2021", "AAAA0001"),
        ("Dirimoglu 2021", "AAAA0001"),
        ("Brightwater 1993 Orchards widget", "BBBB0002"),
        ("brightwaterTendingOrchardsWidget1993", "BBBB0002"),
    ],
)
def test_folded_lookup_finds_and_confirms_an_indexed_paper(
    fake_lightrag, load_server, items, query, key
):
    index = [_paper(k, m) for k, m in items.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    out = load_server().find_paper(query)
    indexed, zotero = out.split("In Zotero but NOT indexed")
    assert f"- {key} · " in indexed
    assert "(none)" in zotero
    calls = [r["params"] for r in fake_lightrag.calls("/graph/papers/search")]
    # The words search, then ONE request confirming every local candidate by key.
    assert calls[0] == {"q": query, "limit": "100"}
    assert len(calls) == 2 and set(calls[1]["keys"].split(",")) == {key}
    assert "q" not in calls[1]


def test_old_server_confirms_candidates_one_key_at_a_time_concurrently(
    fake_lightrag, load_server, metadata_file
):
    many = {
        f"EEEE{i:04d}": _item(f"Widget Survey Part {i}", [("Ashgrove", "Lia")], "2010")
        for i in range(12)
    }
    _write(metadata_file, many)
    index = [_paper(k, m) for k, m in many.items() if k.endswith(("1", "3"))]
    # The q search knows nothing (old matching); the key searches know the index.
    route = _search_route(index, version=None, delay=0.3)
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda p: (
        (200, [], {}) if p["q"] == "Ashgrove survey" else route(p)
    )
    srv = load_server()
    started = time.monotonic()
    out = srv.find_paper("Ashgrove survey")
    elapsed = time.monotonic() - started
    keyed = [
        r["params"]["q"]
        for r in fake_lightrag.calls("/graph/papers/search")
        if r["params"]["q"] != "Ashgrove survey"
    ]
    # 4 at a time until 10 items are confirmed not indexed: 2 + 4 + 4 of 12 per batch
    assert len(keyed) == 12
    assert elapsed < 0.3 + 12 * 0.3 / 2  # 4 workers, not one after another
    indexed = out.split("In Zotero but NOT indexed")[0]
    assert "EEEE0001" in indexed and "EEEE0003" in indexed
    assert "NOT checked" not in out


def _many_widgets(metadata_file, n):
    many = {
        f"FFFF{i:04d}": _item(
            f"Widget\u2010FO Study Part {i}", [("Ashgrove", "Lia")], "2010"
        )
        for i in range(n)
    }
    _write(metadata_file, many)
    return many


def test_old_server_confirms_more_than_ten_candidates_it_indexes(
    fake_lightrag, load_server, metadata_file
):
    many = _many_widgets(metadata_file, 14)
    index = [_paper(k, m) for k, m in many.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(
        index, version=None
    )
    out = load_server().find_paper("widget-fo study")
    indexed, zotero = out.split("In Zotero but NOT indexed")
    assert "Indexed, so its text is searchable (14)" in indexed
    assert "(0)" in zotero and "NOT checked" not in out


def test_candidates_past_the_confirmation_cap_are_marked_not_checked(
    fake_lightrag, load_server, metadata_file
):
    many = _many_widgets(metadata_file, 10)
    index = [_paper(k, m) for k, m in many.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(
        index, version=None
    )
    srv = load_server()
    srv.CONFIRM_MAX = 6
    out = srv.find_paper("widget-fo study")
    asked = {r["params"]["q"] for r in fake_lightrag.calls("/graph/papers/search")}
    indexed, rest = out.split("In Zotero but NOT indexed")
    zotero, unchecked = rest.split("index status NOT checked")
    assert "(6)" in indexed and "(0)" in zotero
    assert "(4)" in unchecked
    # the four marked unchecked are exactly the ones the server was not asked about
    listed = {f"FFFF{i:04d}" for i in range(10) if f"FFFF{i:04d}" in unchecked}
    assert len(listed) == 4 and not listed & asked


def test_new_server_confirms_in_batches_of_fifty(
    fake_lightrag, load_server, metadata_file
):
    many = _many_widgets(metadata_file, 60)
    index = [_paper(k, m) for k, m in many.items()]
    route = _search_route(index)
    fake_lightrag.routes[("GET", "/graph/papers/search")] = lambda p: (
        (200, [], {"X-Paper-Search": "2"}) if "q" in p else route(p)
    )
    out = load_server().find_paper("widget-fo study")
    keyed = [
        r["params"]["keys"].split(",")
        for r in fake_lightrag.calls("/graph/papers/search")
        if "keys" in r["params"]
    ]
    assert [len(k) for k in keyed] == [50, 10]
    assert "Indexed, so its text is searchable (60)" in out
    assert (
        "NOT checked" not in out
        and "NOT indexed, so no tool here can read its text (0)" in out
    )


def test_paper_search_results_are_cached_per_query(fake_lightrag, load_server, items):
    index = [_paper(k, m) for k, m in items.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    srv = load_server()
    first = srv.find_paper("Larkspur")
    n = len(fake_lightrag.calls("/graph/papers/search"))
    assert srv.find_paper("Larkspur") == first
    assert len(fake_lightrag.calls("/graph/papers/search")) == n
    srv.PAPER_CACHE_TTL = 0
    srv.find_paper("Larkspur")
    assert len(fake_lightrag.calls("/graph/papers/search")) > n


def test_paper_search_cache_is_bounded(fake_lightrag, load_server, items):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([])
    srv = load_server()
    for i in range(srv.PAPER_CACHE_SIZE + 20):
        srv._paper_search(q=f"word{i}")
    assert len(srv._PAPER_CACHE) <= srv.PAPER_CACHE_SIZE


def test_indexed_list_puts_the_title_phrase_first(
    fake_lightrag, load_server, metadata_file
):
    titles = [f"Widget Gardens and Topic {i}" for i in range(8)]
    titles += [f"Gardens of Widget Hedges {i}" for i in range(15)]
    titles.insert(20, "Widget Gardens")
    meta = {
        f"FFFF{i:04d}": _item(t, [("Pellmore", "Ros")], "2001")
        for i, t in enumerate(titles)
    }
    _write(metadata_file, meta)
    index = [_paper(k, m) for k, m in meta.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    out = load_server().find_paper("Widget Gardens")
    rows = [line for line in out.splitlines() if line.startswith("- FFFF")]
    assert rows[0].startswith("- FFFF0020 · ")
    # Then the titles holding the phrase, before those holding both words apart.
    assert all("Widget Gardens and Topic" in r for r in rows[1:9])
    assert "… and 14 more NOT shown. The paper you want may be among them" in out
    assert "A cut list does not show absence." in out


def _short_surname_meta():
    # "Ho" occurs inside "Hoeing"; the paper by Ho is older and has no title hit.
    return {
        "FFFF0001": _item(
            "Hoeing the Rows of a Shoreline Garden", [("Quillfeather", "Ada")], "2020"
        ),
        "FFFF0002": _item("Thinning Seedlings", [("Ho", "Wen")], "2015"),
    }


def test_indexed_list_puts_whole_word_matches_first(
    fake_lightrag, load_server, metadata_file
):
    meta = _short_surname_meta()
    _write(metadata_file, meta)
    index = [_paper(k, m) for k, m in meta.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    rows = [
        line
        for line in load_server().find_paper("Ho").splitlines()
        if line.startswith("- FFFF")
    ]
    assert [r[2:10] for r in rows] == ["FFFF0002", "FFFF0001"]


def test_zotero_only_list_puts_whole_word_matches_first(load_server, metadata_file):
    _write(metadata_file, _short_surname_meta())
    found = [k for k, _ in load_server()._metadata_matches("Ho")]
    assert found == ["FFFF0002", "FFFF0001"]


def test_empty_doi_lookup_does_not_suggest_the_doi(fake_lightrag, load_server, items):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([])
    out = load_server().find_paper("10.9999/absent.2020.9")
    assert "Try 'Author Year' alone." in out
    assert "Try the DOI" not in out


def test_empty_lookup_wording_no_longer_claims_absence(
    fake_lightrag, load_server, items
):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([])
    out = load_server().find_paper("Nobody 1950 nothing")
    assert (
        "No indexed paper or Zotero item contains every word (hyphens, apostrophes, "
        "accents and quotes are ignored)." in out
    )
    assert "Try the DOI, the Zotero key, or 'Author Year' alone" in out
    assert "does not show the paper is missing" in out
    assert "is not in the library" not in out and "citation key" not in out
    assert "Near matches" not in out
    assert "show absence" not in out.replace("does not show absence", "")


def test_year_off_by_one_is_a_labelled_near_match(fake_lightrag, load_server, items):
    index = [_paper("CCCC0003", items["CCCC0003"])]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    out = load_server().find_paper("Larkspur 2009 calibration")
    assert "No indexed paper or Zotero item contains every word" in out
    near = out.split(
        "No item has every word. Near matches (NOT exact — check before citing):"
    )[1]
    line = next(x for x in near.splitlines() if "CCCC0003" in x)
    assert line.startswith("- CCCC0003 · Larkspur (2008). Calibration")
    assert line.endswith(" · indexed")


def test_an_empty_exact_match_never_claims_absence_beside_a_near_match(
    fake_lightrag, load_server, items
):
    index = [_paper("CCCC0003", items["CCCC0003"])]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    for tool in (
        load_server().find_paper,
        lambda q: load_server().paper_passages(q, "q"),
    ):
        out = tool("Larkspur 2009 calibration")
        assert "- CCCC0003 · Larkspur (2008). Calibration" in out
        assert "does not show the paper is missing" in out
        assert "The near matches below may be the paper itself" in out
        assert "not in the library" not in out and "citation key" not in out
        assert "Try the DOI" not in out


def test_surname_near_match_is_marked_zotero_only(fake_lightrag, load_server, items):
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([])
    out = load_server().find_paper("Marrowby levelling sprocket")
    near = out.split("Near matches (NOT exact — check before citing):")[1]
    line = next(x for x in near.splitlines() if "DDDD0004" in x)
    assert line.endswith(" · Zotero only")
    # A surname with the wrong year is not a near match.
    out = load_server().find_paper("Marrowby 2010 sprocket")
    assert "Near matches" not in out
    # The BibTeX family is the surname.
    out = load_server().find_paper("marrowbyLevellingSprockets2017")
    assert "DDDD0004" in out.split("Near matches")[1]


def test_paper_passages_never_searches_a_near_match(fake_lightrag, load_server, items):
    index = [_paper(k, m) for k, m in items.items()]
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(index)
    out = load_server().paper_passages("Larkspur 2009 calibration", "q")
    assert "CCCC0003" in out and "Near matches" in out
    assert "call it again with one Zotero key or DOI" in out
    assert fake_lightrag.calls("/query") == []
    assert fake_lightrag.calls("/graph/papers/passages") == []


# --- P3: one paper's passages ---------------------------------------------------


def _header(item, key):
    """The corpus header build_corpus.header() writes (a literal copy; see the
    drift test below)."""
    lines = [f"# {item['title']}", ""]
    if item.get("authors"):
        lines.append(f"**Authors:** {'; '.join(item['authors'])}")
    bits = []
    if item.get("year"):
        bits.append(f"**Year:** {item['year']}")
    if item.get("publication"):
        bits.append(f"**Published in:** {item['publication']}")
    if item.get("doi"):
        bits.append(f"**DOI:** {item['doi']}")
    if bits:
        lines.append("  ".join(bits))
    if item.get("tags"):
        lines.append(f"**Tags:** {', '.join(item['tags'][:20])}")
    lines.append(f"**Zotero:** zotero://select/library/items/{key}")
    if item.get("abstract"):
        lines += ["", "> **Abstract.** " + " ".join(item["abstract"].split())]
    lines += ["", "---", ""]
    return "\n".join(lines)


def test_header_copy_matches_build_corpus():
    sys.path.insert(0, str(CORPUS_DIR))
    try:
        spec = importlib.util.find_spec("build_corpus")
        if spec is None:
            pytest.skip("build_corpus not importable")
        try:
            import build_corpus
        except Exception as exc:  # its PDF dependencies are not installed here
            pytest.skip(f"build_corpus not importable: {exc}")
    finally:
        sys.path.remove(str(CORPUS_DIR))
    item = dict(
        ITEMS["BBBB0002"], tags=["t1", "t2"], abstract="An  invented\nabstract."
    )
    for variant in (item, dict(item, abstract="", tags=[], doi="", authors=[])):
        assert build_corpus.header({"K": variant}, "K") == _header(variant, "K")


ABSTRACT = (
    "We count widgets on the orchard and find the budget closes within ten percent."
)


@pytest.mark.parametrize("collapsed", [False, True])
def test_header_strip_drops_the_title_block_and_a_repeated_abstract(
    load_server, collapsed
):
    srv = load_server()
    item = dict(ITEMS["BBBB0002"], tags=["t1"], abstract=ABSTRACT)
    head = _header(item, "BBBB0002")
    if collapsed:  # the chunker drops blank lines
        head = "\n".join(line for line in head.splitlines() if line) + "\n"
    body = "Abstract\nWe count widgets on the orchard and find the budget\ncloses within ten percent.\n\n1 Introduction\nText."
    assert srv._strip_header(head + body) == body
    other = "1 Introduction\nText that does not repeat it."
    assert srv._strip_header(head + other) == f"> **Abstract.** {ABSTRACT}\n\n{other}"


def test_header_strip_leaves_other_chunks_alone(load_server):
    srv = load_server()
    no_zotero = "# A Heading\n\n**Authors:** Someone\n\n---\n\nBody."
    assert srv._strip_header(no_zotero) == no_zotero
    assert srv._strip_header("Plain body text.") == "Plain body text."
    head = _header(dict(ITEMS["DDDD0004"], abstract=""), "DDDD0004")
    assert srv._strip_header(head + "Body.") == "Body."


def _passages_route(response, status=200):
    return lambda body: (status, response)


def test_paper_passages_uses_the_passages_route(fake_lightrag, load_server, items):
    paper = _paper("BBBB0002", items["BBBB0002"])
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([paper])
    head = _header(dict(items["BBBB0002"], abstract=ABSTRACT), "BBBB0002")
    fake_lightrag.routes[("POST", "/graph/papers/passages")] = _passages_route(
        {
            "keys": ["BBBB0002"],
            "n_chunks": 20,
            "passages": [
                {
                    "key": "BBBB0002",
                    "file_path": "BBBB0002__Tending.md",
                    "chunk_id": "chunk-b",
                    "chunk_order_index": 3,
                    "content": "The budget closes within ten percent.",
                    "content_headings": "Tending → 4 Results",
                    "score": 0.9,
                },
                {
                    "key": "BBBB0002",
                    "file_path": "BBBB0002__Tending.md",
                    "chunk_id": "chunk-a",
                    "chunk_order_index": 0,
                    "content": head + "1 Introduction\nWidgets.",
                    "content_headings": None,
                    "score": 0.5,
                },
            ],
        }
    )
    out = load_server().paper_passages("BBBB0002", "does the widget budget close")
    assert out.startswith("### Brightwater, Ezra; Thornfield, Ivo. (1993). Tending")
    assert (
        "[1] Brightwater 1993 · Tending → 4 Results · chunk 4 of 20\n"
        "The budget closes within ten percent." in out
    )
    assert (
        f"[2] Brightwater 1993 · chunk 1 of 20\n> **Abstract.** {ABSTRACT}\n\n"
        "1 Introduction\nWidgets." in out
    )
    assert "**Zotero:**" not in out and "Other papers" not in out
    body = fake_lightrag.calls("/graph/papers/passages")[0]["json"]
    assert body == {
        "keys": ["BBBB0002"],
        "question": "does the widget budget close",
        "top_n": 3,
    }
    assert fake_lightrag.calls("/query") == []


def test_paper_passages_route_without_a_passage_says_what_to_try(
    fake_lightrag, load_server, items
):
    paper = _paper("BBBB0002", items["BBBB0002"])
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([paper])
    fake_lightrag.routes[("POST", "/graph/papers/passages")] = _passages_route(
        {"keys": ["BBBB0002"], "n_chunks": 20, "passages": []}
    )
    out = load_server().paper_passages("BBBB0002", "q")
    assert "No passage" in out
    assert 'search_library(mode="naive")' in out


def test_multi_copy_work_leads_with_the_copy_that_has_a_doi(
    fake_lightrag, load_server, metadata_file
):
    meta = {
        "AAAA0009": _item("Widget budget", [("Brightwater", "Ezra")], "1993"),
        "BBBB0010": _item(
            "Tending the Orchard's Widget Budget",
            [("Brightwater", "Ezra")],
            "1993",
            doi="10.9999/widget.1993.2",
        ),
    }
    _write(metadata_file, meta)
    paper = _paper("BBBB0010", meta["BBBB0010"], keys=["AAAA0009", "BBBB0010"])
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([paper])
    fake_lightrag.routes[("POST", "/graph/papers/passages")] = _passages_route(
        {"keys": ["AAAA0009", "BBBB0010"], "n_chunks": 5, "passages": []}
    )
    out = load_server().paper_passages("10.9999/widget.1993.2", "q")
    assert out.startswith("### Brightwater, Ezra. (1993). Tending the Orchard's")
    assert "items/BBBB0010]" in out.splitlines()[0]
    body = fake_lightrag.calls("/graph/papers/passages")[0]["json"]
    assert set(body["keys"]) == {"AAAA0009", "BBBB0010"}


def _context_text(chunks):
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    return f"Document Chunks (x):\n\n```json\n{lines}\n```\n"


def test_old_server_falls_back_to_query_with_a_family_year_lead(
    fake_lightrag, load_server, metadata_file
):
    long_title = "A Very Long Invented Title About Widgets That Goes On And On Further"
    items = dict(ITEMS, HHHH0001=_item(long_title, [("Ferncastle", "Uma")], "2019"))
    _write(metadata_file, items)
    paper = _paper("BBBB0002", items["BBBB0002"])
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route([paper])
    refs = (
        [{"reference_id": "1", "file_path": "BBBB0002__x.md"}]
        + [
            {"reference_id": str(n), "file_path": f"{k}__x.md"}
            for n, k in enumerate(["AAAA0001", "HHHH0001", "DDDD0004"], start=2)
        ]
        + [
            {"reference_id": str(n), "file_path": f"ZZZZ{n:04d}__Some_Other_Paper.md"}
            for n in range(5, 9)
        ]
    )
    chunks = [{"reference_id": "1", "content": "Budget text."}]
    fake_lightrag.routes[("POST", "/query")] = lambda body: (
        200,
        {"response": _context_text(chunks), "references": refs},
    )
    out = load_server().paper_passages("BBBB0002", "does it close")
    assert fake_lightrag.calls("/query")[0]["json"]["query"] == (
        "Brightwater 1993: does it close"
    )
    assert "[1] Brightwater 1993\nBudget text." in out
    trailer = out.split("### Other papers the question matched (no passages shown)\n")[
        1
    ]
    lines = trailer.splitlines()
    assert len(lines) == 5
    assert lines[0] == f"- AAAA0001 · Quillfeather 2021 · {ITEMS['AAAA0001']['title']}"
    assert lines[1] == f"- HHHH0001 · Ferncastle 2019 · {long_title[:60]}…"
    assert lines[3] == "- ZZZZ0005 · Some Other Paper"


def test_old_server_lead_adds_title_words_when_family_and_year_tie(
    fake_lightrag, load_server, metadata_file
):
    meta = {
        "GGGG0001": _item(
            "Sprocket Drift in Cold Weather", [("Vantreese", "Ola")], "2012"
        ),
        "GGGG0002": _item("Sprocket Drift Models", [("Vantreese", "Ola")], "2012"),
    }
    _write(metadata_file, meta)
    fake_lightrag.routes[("GET", "/graph/papers/search")] = _search_route(
        [_paper("GGGG0001", meta["GGGG0001"])]
    )
    out = load_server().paper_passages("GGGG0001", "q")
    assert "No passage of this paper" in out
    assert fake_lightrag.calls("/query")[0]["json"]["query"] == (
        "Vantreese 2012 Cold Weather: q"
    )


# --- P4 + P5: search_library --------------------------------------------------------


def test_search_library_sends_passages_as_chunk_top_k(fake_lightrag, load_server):
    srv = load_server()
    assert srv.DEFAULT_PASSAGES == 8
    srv.search_library("q")
    srv.search_library("q", passages=3)
    srv.search_library("q", passages=99)
    srv.search_library("q", passages=0)
    sent = [c["json"]["chunk_top_k"] for c in fake_lightrag.calls("/query")]
    assert sent == [8, 3, 12, 1]


def test_search_library_strips_headers_and_merges_duplicate_copies(
    fake_lightrag, load_server, items
):
    refs = [
        {"reference_id": "1", "file_path": "BBBB0002__x.md"},
        {"reference_id": "2", "file_path": "CCCC0003__x.md"},
        {"reference_id": "4", "file_path": "AAAA0001__copy.md"},
    ]
    head = _header(items["BBBB0002"], "BBBB0002")
    chunks = [
        {
            "reference_id": "1",
            "content": "Same  text\nof two copies.",
            "content_headings": "Intro",
        },
        {"reference_id": "2", "content": head + "Body after the header."},
        {"reference_id": "4", "content": "Same text of two\ncopies.  "},
    ]
    fake_lightrag.routes[("POST", "/query")] = lambda body: (
        200,
        {"response": _context_text(chunks), "references": refs},
    )
    out = load_server().search_library("q")
    assert "[1][4] Brightwater 1993 · Intro\nSame  text\nof two copies." in out
    assert out.count("of two") == 1
    assert "[2] Larkspur 2008\nBody after the header." in out
    assert "**Zotero:**" not in out
