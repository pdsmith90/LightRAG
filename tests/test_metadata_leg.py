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
from lightrag.zotero_citations import cited_works_in, find_works

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
    assert find_works("Okafor Lindqvist 2019", max_works=1) == [("AAAA0001", 3)]


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
