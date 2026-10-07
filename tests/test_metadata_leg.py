"""METADATA_CHUNK_TOP_K / CITATION_HOP_TOP_K: works named by author and year, and
works cited by the candidate chunks, join the retrieval candidates.

A paper named by surnames and a year has its author line in one chunk and its
method words in others, so no single chunk of it matches the whole query while
the papers citing it match well. The author-year leg resolves the names against
the Zotero metadata instead; the citation leg follows author-year citations in
the best candidate chunks to library works. Fixtures are invented.
"""

import json

import pytest

import lightrag.zotero_citations as zc
from lightrag.operate import _get_metadata_context, _merge_all_chunks, _prepend_chunks
from lightrag.zotero_citations import (
    _surname_forms,
    cited_works_in,
    find_works,
    named_work_notices,
    specific_named_work,
)

pytestmark = pytest.mark.offline

META = {
    "AAAA0001": {
        "title": "Decorrelation of simulated range-rate residuals",
        "authors": ["Okafor Chidi", "Lindqvist Maja"],
        "year": "2019",
    },
    "AAAA0002": {"title": "Second", "authors": ["Okafor Chidi"], "year": "2021"},
    "AAAA0003": {
        "title": "Third",
        "authors": ["Lindqvist Maja", "Okafor Chidi", "Varga Tamás"],
        "year": "2019",
    },
    "AAAA0004": {"title": "Hyphen", "authors": ["Ghobadi-Far Khosro"], "year": "2020"},
}
META.update(
    {
        f"AAAA001{i}": {"title": f"N{i}", "authors": ["Nakamura Ren"], "year": "2019"}
        for i in range(4)
    }
)
META.update(
    {
        f"AAAA002{i}": {"title": f"L{i}", "authors": ["Lee Min"], "year": "2015"}
        for i in range(10)
    }
)

BIBLIOGRAPHY = """OKAFOR C. & LINDQVIST M. 2019. Decorrelation of simulated range-rate residuals. J. Invented Geod. 3, 1–12.
VARGA T. 2018. A synthetic covariance model. J. Invented Geod. 2, 44–51.
NAKAMURA R. 2019. Four papers in one year. Proc. Imaginary Conf. 7, 9–17.
LEE M. 2015. Ten identical titles. Rev. Fictional Sci. 1, 100–108.
GHOBADI-FAR K. 2020. On hyphenated surnames. J. Invented Geod. 4, 5–6."""


def _reset(path):
    zc.METADATA_PATH = path
    zc._meta, zc._mtime, zc._next_stat, zc._warned, zc._ay_index = (
        {},
        None,
        0.0,
        False,
        None,
    )


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    path = tmp_path / "zotero_metadata.json"
    path.write_text(json.dumps(META), encoding="utf-8")
    saved = zc.METADATA_PATH
    _reset(str(path))
    yield path
    _reset(saved)


# ---------------------------------------------------------------------------
# find_works
# ---------------------------------------------------------------------------


def test_surnames_and_year_name_the_works(metadata):
    assert find_works("Okafor Lindqvist 2019 range-rate residuals") == [
        ("AAAA0001", 3),
        ("AAAA0003", 3),
    ]
    assert find_works("okafor 2021 residuals") == [("AAAA0002", 2)]


def test_ties_at_the_best_score_all_return_and_titles_break_them(metadata):
    # both Okafor-Lindqvist 2019 works tie: the cap never cuts a tie
    assert find_works("Okafor Lindqvist 2019", max_works=1) == [
        ("AAAA0001", 3),
        ("AAAA0003", 3),
    ]
    # the title sharing the query's other words goes first
    assert find_works("Lindqvist Okafor 2019 third")[0] == ("AAAA0003", 3)


def test_adjacent_years_stand_in_when_the_year_matches_nothing(metadata):
    assert find_works("Okafor 2020 residuals") == [
        ("AAAA0001", 1),
        ("AAAA0002", 1),
        ("AAAA0003", 1),
    ]
    assert find_works("Okafor 2005 residuals") == []  # two years off never


def test_an_adjacent_year_work_joins_when_it_matches_more_surnames(tmp_path):
    # the pair's paper is filed a year off; the exact-year match names one author
    meta = {
        "BBBB0001": {
            "title": "Pair",
            "authors": ["Okafor Chidi", "Lindqvist Maja"],
            "year": "2022",
        },
        "BBBB0002": {"title": "Solo", "authors": ["Okafor Chidi"], "year": "2021"},
        "BBBB0003": {
            "title": "Pair too",
            "authors": ["Okafor Chidi", "Lindqvist Maja"],
            "year": "2019",
        },
    }
    path = tmp_path / "zotero_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    saved = zc.METADATA_PATH
    _reset(str(path))
    try:
        assert find_works("Okafor Lindqvist 2021 residuals") == [
            ("BBBB0001", 2),
            ("BBBB0002", 2),
        ]
    finally:
        _reset(saved)


def test_surname_forms_stop_at_the_given_names():
    assert _surname_forms("Varga Grace E.") == {"varga"}
    assert _surname_forms("van der Lind Tamás") == {"van der lind", "lind"}
    assert _surname_forms("Ferreira Da Costa Inês") == {
        "ferreira da costa",
        "ferreira",
        "costa",
    }
    assert _surname_forms("Nakamura\u2010Lindqvist Ren") == {
        "nakamura\u2010lindqvist",
        "nakamura",
        "lindqvist",
    }
    assert _surname_forms("Okafor, Chidi") == {"okafor"}


def test_institutions_and_mailboxes_contribute_no_surname():
    assert _surname_forms("Data Support, mailbox@example.invalid") == set()
    assert _surname_forms("University Consortium For Imaginary Research") == set()
    assert _surname_forms("Satellites Programme Of An Example Observatory") == set()
    assert _surname_forms("Programme Office") == set()
    assert _surname_forms("Varga Eric F.") == {"varga"}
    assert _surname_forms("Acme") == set()  # a lone token is no person
    assert _surname_forms("For Mei") == set()  # a function word is no surname


def test_hyphenated_surnames_answer_to_their_parts(metadata):
    assert find_works("ghobadi 2020") == [("AAAA0004", 2)]
    assert find_works("Ghobadi-Far 2020") == [("AAAA0004", 2)]


def test_a_year_in_the_query_must_match(metadata):
    assert find_works("Okafor 2005 residuals") == []


def test_without_a_year_one_surname_counts_only_when_rare(metadata):
    assert find_works("nakamura residuals") == []  # four works
    assert find_works("ghobadi residuals") == [("AAAA0004", 1)]
    assert find_works("Lindqvist Okafor residuals") == [
        ("AAAA0001", 2),
        ("AAAA0003", 2),
    ]


def test_a_common_surname_and_year_is_too_ambiguous(metadata):
    assert find_works("lee 2015 something") == []


def test_no_surname_or_no_metadata_finds_nothing(metadata, tmp_path):
    assert find_works("gravity field smoothing 2019") == []
    _reset(str(tmp_path / "missing.json"))
    assert find_works("Okafor 2019") == []


# ---------------------------------------------------------------------------
# named_work_notices
# ---------------------------------------------------------------------------


def test_specific_named_work_needs_a_clear_winner(metadata):
    # two of the query's words are in the first title only: it means that paper
    named = find_works("Okafor Lindqvist 2019 simulated residuals")
    assert (
        specific_named_work("Okafor Lindqvist 2019 simulated residuals", named)
        == "AAAA0001"
    )
    # one shared word decides nothing; nor does nothing at all
    for query in ("Okafor Lindqvist 2019 residuals", "Okafor Lindqvist 2019 method"):
        assert specific_named_work(query, find_works(query)) is None
    assert specific_named_work("okafor 2021", find_works("okafor 2021")) == "AAAA0002"
    assert specific_named_work("x", []) is None


def test_paper_identity_joins_copies_and_keeps_unknown_keys_apart(tmp_path):
    meta = {
        "DDDD0001": {
            "title": "A long enough title to name one paper",
            "authors": ["Okafor Chidi"],
            "year": "2019",
        },
        "DDDD0002": {
            "title": "A long enough title to name one paper",
            "authors": ["Okafor Chidi"],
            "year": "2019",
        },
        "DDDD0003": {"title": "Short", "authors": ["Okafor Chidi"], "year": "2019"},
    }
    path = tmp_path / "zotero_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    saved = zc.METADATA_PATH
    _reset(str(path))
    try:
        assert zc.paper_identity("DDDD0001") == zc.paper_identity("DDDD0002__copy.md")
        assert (
            zc.paper_identity("DDDD0003")
            == zc.paper_identity("DDDD0003__x.md")
            == "key:DDDD0003"
        )
        assert zc.paper_identity("DDDD0003") != zc.paper_identity("DDDD0001")
        assert zc.paper_identity("EEEE9999") == "key:EEEE9999"
        assert zc.paper_identity("notes/free text.md") == "path:notes/free text.md"
    finally:
        _reset(saved)


def test_citations_drop_the_title_markup(tmp_path):
    meta = {
        "CCCC0001": {
            "title": 'Spread <i>F</i> and <span style="font-variant:small-caps;">GRACE</span>.',
            "authors": ["Okafor Chidi"],
            "year": "2019",
        }
    }
    path = tmp_path / "zotero_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    saved = zc.METADATA_PATH
    _reset(str(path))
    try:
        assert (
            zc.citation_for_key("CCCC0001")
            == "Okafor Chidi. (2019). Spread F and GRACE"
        )
        assert (
            zc.citation_for("CCCC0001__paper.md")
            == "Okafor Chidi. (2019). Spread F and GRACE"
        )
        assert zc.citation_for_key("CCCC9999") == ""
    finally:
        _reset(saved)


def test_notices_name_the_missing_works_and_say_why(metadata):
    # ambiguous naming: the pair's two 2019 papers tie
    named = find_works("Okafor Lindqvist 2019 method")
    query = "Okafor Lindqvist 2019 method"
    # one of them is among the references: nothing to say
    assert named_work_notices(query, ["AAAA0003__x.md"], named, {"AAAA0001"}) == []
    out = named_work_notices(query, ["AAAA0009__citer.md"], named, {"AAAA0001"})
    assert len(out) == 2
    assert "Okafor" in out[0] and "(2019)" in out[0]
    assert out[0].endswith("not among the sources retrieved for this question.")
    assert "Third" in out[1] and "not in this knowledge base" in out[1]
    # specific naming: only that work counts, even when a sibling is a source
    query = "Okafor Lindqvist 2019 simulated residuals"
    named = find_works(query)
    out = named_work_notices(query, ["AAAA0003__x.md"], named, {"AAAA0003"})
    assert (
        len(out) == 1
        and "residuals" in out[0]
        and "not in this knowledge base" in out[0]
    )
    assert named_work_notices(query, ["AAAA0001__x.md"], named, {"AAAA0001"}) == []
    # no year: too weak a naming to warn about; nothing named: nothing
    assert (
        named_work_notices("Okafor Lindqvist simulated residuals", [], named, set())
        == []
    )
    assert named_work_notices(query, [], [], set()) == []


# ---------------------------------------------------------------------------
# cited_works_in
# ---------------------------------------------------------------------------

TEXT = (
    "As Okafor and Lindqvist (2019) showed, the residuals decorrelate; later work "
    "(Okafor, 2021) confirmed it, see also Okafor et al. 2019 and Lee et al. (2015)."
)


def test_citations_resolve_to_library_works_most_cited_first(metadata):
    assert cited_works_in(TEXT) == ["AAAA0001", "AAAA0003", "AAAA0002"]
    assert cited_works_in(TEXT, exclude={"AAAA0001"}) == ["AAAA0003", "AAAA0002"]
    assert cited_works_in(TEXT, max_works=1) == ["AAAA0001"]
    assert cited_works_in("") == []


def test_ambiguous_or_unknown_citations_are_skipped(metadata):
    assert cited_works_in("Lee et al. (2015) and Unknown (2019)") == []


# ---------------------------------------------------------------------------
# _get_metadata_context
# ---------------------------------------------------------------------------


class _Chunks:
    def __init__(self, global_config):
        self.global_config = global_config
        self.calls: list[tuple[list, str, int]] = []

    async def get_chunks_for_works(self, keys, query, per_work):
        self.calls.append((list(keys), query, per_work))
        rows = []
        for key in keys:
            for i in range(per_work):
                rows.append(
                    {
                        "id": f"{key}-{i}",
                        "full_doc_id": f"doc-{key}",
                        "file_path": f"{key}__paper.md",
                        "content": BIBLIOGRAPHY
                        if (key, i) == ("AAAA0003", 1)
                        else f"text of {key} chunk {i}",
                        "chunk_order_index": i,
                        "work": key,
                    }
                )
        return rows


class _NoLeg:
    global_config = {"metadata_chunk_top_k": 4}


QUERY = "Okafor Lindqvist 2019 range-rate residuals"


@pytest.mark.asyncio
async def test_named_and_cited_works_contribute_chunks(metadata):
    db = _Chunks(
        {
            "metadata_chunk_top_k": 4,
            "citation_hop_top_k": 2,
            "drop_bibliography_chunks": True,
        }
    )
    hop = [{"content": "see (Okafor, 2021) for the covariance", "chunk_id": "v1"}]
    chunks = await _get_metadata_context(QUERY, db, hop_sources=hop)
    assert db.calls == [(["AAAA0001", "AAAA0003"], QUERY, 2), (["AAAA0002"], QUERY, 2)]
    assert [c["chunk_id"] for c in chunks] == [
        "AAAA0001-0",
        "AAAA0001-1",
        "AAAA0003-0",  # AAAA0003-1 is a reference list: dropped
        "AAAA0002-0",
        "AAAA0002-1",
    ]
    assert {c["source_type"] for c in chunks[:3]} == {"metadata"}
    assert {c["source_type"] for c in chunks[3:]} == {"citation"}
    assert chunks[0]["file_path"] == "AAAA0001__paper.md"


@pytest.mark.asyncio
async def test_pin_marks_the_specific_work_only_when_one_is_named(metadata):
    pinned = _Chunks(
        {"metadata_chunk_top_k": 4, "citation_hop_top_k": 0, "named_work_pin": True}
    )
    # ambiguous naming: both tied works are pinned, the reranker decides later
    chunks = await _get_metadata_context("Okafor Lindqvist 2019 method", pinned)
    assert chunks and all(c.get("pinned") for c in chunks)
    # specific naming: only the named work's chunks keep the pin
    chunks = await _get_metadata_context(
        "Okafor Lindqvist 2019 simulated residuals", pinned
    )
    assert {c["file_path"].split("__")[0] for c in chunks if c.get("pinned")} == {
        "AAAA0001"
    }
    assert any(not c.get("pinned") for c in chunks)
    # no year, or the option off: nothing pinned
    chunks = await _get_metadata_context("Lindqvist Okafor residuals", pinned)
    assert chunks and not any(c.get("pinned") for c in chunks)
    plain = _Chunks({"metadata_chunk_top_k": 4, "citation_hop_top_k": 0})
    chunks = await _get_metadata_context(
        "Okafor Lindqvist 2019 simulated residuals", plain
    )
    assert chunks and not any(c.get("pinned") for c in chunks)


async def test_legs_are_off_at_zero_and_independent(metadata):
    off = _Chunks({"metadata_chunk_top_k": 0, "citation_hop_top_k": 0})
    assert (
        await _get_metadata_context(
            QUERY, off, hop_sources=[{"content": "(Okafor, 2021)"}]
        )
        == []
    )
    assert off.calls == []
    hop_only = _Chunks({"metadata_chunk_top_k": 0, "citation_hop_top_k": 1})
    chunks = await _get_metadata_context(
        "anything", hop_only, hop_sources=[{"content": TEXT}]
    )
    assert hop_only.calls == [(["AAAA0001"], "anything", 2)]
    assert [c["chunk_id"] for c in chunks] == ["AAAA0001-0", "AAAA0001-1"]
    named_only = _Chunks({"metadata_chunk_top_k": 3, "citation_hop_top_k": 0})
    chunks = await _get_metadata_context(
        "okafor 2021", named_only, hop_sources=[{"content": TEXT}]
    )
    assert named_only.calls == [(["AAAA0002"], "okafor 2021", 3)]
    assert len(chunks) == 3
    assert await _get_metadata_context(QUERY, None) == []


@pytest.mark.asyncio
async def test_storage_without_the_method_warns_once(metadata, lightrag_log_records):
    assert await _get_metadata_context(QUERY, _NoLeg()) == []
    assert await _get_metadata_context(QUERY, _NoLeg()) == []
    assert (
        sum(
            "has no get_chunks_for_works" in r.getMessage()
            for r in lightrag_log_records
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Merge order: metadata chunks lead, lexical interleave, vector follow
# ---------------------------------------------------------------------------


def _chunk(chunk_id):
    return {"chunk_id": chunk_id, "content": chunk_id, "file_path": f"{chunk_id}.md"}


def test_prepend_keeps_order_and_drops_repeats():
    assert [
        c["chunk_id"]
        for c in _prepend_chunks([_chunk("m1")], [_chunk("m1"), _chunk("v1")])
    ] == ["m1", "v1"]


@pytest.mark.asyncio
async def test_merge_puts_metadata_chunks_first():
    merged = await _merge_all_chunks(
        filtered_entities=[],
        filtered_relations=[],
        vector_chunks=[_chunk("v1"), _chunk("v2")],
        query="q",
        lexical_chunks=[_chunk("l1")],
        metadata_chunks=[_chunk("m1"), _chunk("v2")],
    )
    assert [c["chunk_id"] for c in merged] == ["m1", "l1", "v2", "v1"]
