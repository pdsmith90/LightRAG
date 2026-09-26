"""Bibliography chunk filter: ``is_bibliography_chunk`` and its use in
``process_chunks_unified``.

A reference list is dense in exactly the words a query uses -- paper titles --
so its chunks score well in vector search and with rerankers while carrying no
content. With ``drop_bibliography_chunks`` on, ``process_chunks_unified`` removes
them before anything else runs: the reranker never receives them and they
cannot take a ``chunk_top_k`` or token-budget slot. With it off the function
behaves exactly as before.

The detector must keep prose that cites a few DOIs, data-availability
paragraphs, and chunks that straddle the end of the prose and at most four
entries; those cases are pinned here alongside the formats it must catch. A
chunk whose reference tail reaches five entries or a packed line is dropped
whole, prose included: a documented residue, pinned here as well. The year
requirement and each packed-line threshold are pinned by a pair of inputs
that differ only in the count concerned.
"""

from __future__ import annotations

import re
import time

import pytest

from lightrag.base import QueryParam
from lightrag.utils import (
    filter_bibliography_chunks,
    is_bibliography_chunk,
    process_chunks_unified,
)

pytestmark = pytest.mark.offline


# Author-year entries with DOIs, one entry per paragraph.
AUTHOR_YEAR = """Karpukhin, V., B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih (2020), Dense passage retrieval for open-domain question answering, in Proc. EMNLP, pp. 6769–6781, doi:10.18653/v1/2020.emnlp-main.550.

Devlin, J., M.-W. Chang, K. Lee, and K. Toutanova (2019), BERT: Pre-training of deep bidirectional transformers for language understanding, in Proc. NAACL-HLT, pp. 4171–4186, doi:10.18653/v1/N19-1423.

Johnson, J., M. Douze, and H. Jégou (2021), Billion-scale similarity search with GPUs, IEEE Trans. Big Data, 7(3), 535–547, doi:10.1109/TBDATA.2019.2921572.

Malkov, Y. A., and D. A. Yashunin (2020), Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs, IEEE Trans. Pattern Anal. Mach. Intell., 42(4), 824–836, doi:10.1109/TPAMI.2018.2889473.

Robertson, S., and H. Zaragoza (2009), The probabilistic relevance framework: BM25 and beyond, Found. Trends Inf. Retr., 3(4), 333–389, doi:10.1561/1500000019.
"""

# Older style: upper-case surnames, initials after the name, no DOIs.
OLD_STYLE = """LUHN H. P. 1957. A statistical approach to mechanized encoding and searching of literary information. IBM J. Res. Dev. 1, 309–317.
MARON M. E. & KUHNS J. L. 1960. On relevance, probabilistic indexing and information retrieval. J. ACM 7, 216–244.
SALTON G., WONG A. & YANG C. S. 1975. A vector space model for automatic indexing. Commun. ACM 18, 613–620.
ROBERTSON S. E. 1977. The probability ranking principle in IR. J. Doc. 33, 294–304.
SALTON G. & McGILL M. J. 1983. Introduction to Modern Information Retrieval. McGraw-Hill, New York.
"""

# A whole list extracted as one paragraph, as some PDF extractors emit it.
PACKED = " ".join(AUTHOR_YEAR.split("\n\n"))

# Prose that cites three DOIs inline.
PROSE = (
    "Dense retrievers outperform sparse ones on open-domain question answering "
    "(Karpukhin et al., 2020), as Lewis et al. (2020) anticipated; see "
    "https://doi.org/10.18653/v1/2020.emnlp-main.550, doi:10.1561/1500000019 and "
    "doi:10.18653/v1/N19-1423 for the evaluations.\n\n"
    "We compare both against a BM25 baseline on twelve public benchmarks and "
    "report recall after reranking."
)

# A long related-work paragraph: many years, several DOIs, over 600 characters,
# but it does not open like a reference entry, so it is prose.
RELATED_WORK = (
    "Recent work on retrieval-augmented generation spans sparse ranking functions "
    "(Robertson and Zaragoza, 2009), dense passage retrieval (Karpukhin et al., "
    "2020), pretrained encoders (Devlin et al., 2019), approximate nearest-neighbour "
    "indexes (Malkov and Yashunin, 2020; Johnson et al., 2021) and graph-based "
    "indexes built over extracted entities (2023, 2024). The evaluations we build on "
    "are archived under doi:10.18653/v1/2020.emnlp-main.550, doi:10.1561/1500000019 "
    "and doi:10.18653/v1/N19-1423, and each of them reports recall at several cut-offs "
    "rather than a single score, which is what makes their numbers comparable with "
    "the ones reported in the remainder of this section."
)

DATA_AVAILABILITY = (
    "Data availability. The evaluation queries are archived at "
    "https://doi.org/10.5281/zenodo.0000001; the passage corpus at "
    "https://doi.org/10.5281/zenodo.0000002; relevance judgments at "
    "https://doi.org/10.5281/zenodo.0000003; model checkpoints at "
    "https://doi.org/10.5281/zenodo.0000004; and run files at "
    "https://doi.org/10.5281/zenodo.0000005."
)


# ---------------------------------------------------------------------------
# is_bibliography_chunk
# ---------------------------------------------------------------------------


def test_author_year_list_with_dois_is_bibliography():
    assert is_bibliography_chunk(AUTHOR_YEAR)


def test_old_style_list_without_dois_is_bibliography():
    assert is_bibliography_chunk(OLD_STYLE)


def test_numbered_list_is_bibliography():
    numbered = "\n".join(
        f"[{i}] {line}" for i, line in enumerate(OLD_STYLE.splitlines(), start=1)
    )
    assert is_bibliography_chunk(numbered)


@pytest.mark.parametrize("dated", [False, True], ids=["undated-kept", "dated-dropped"])
def test_entry_lines_need_a_year(dated):
    """OLD_STYLE with every year printed as "n.d.": five lines that still open
    like entries and carry a page range or an initials pattern, none of which
    counts as an entry without its year."""
    text, undated = re.subn(r"\b\d{4}\.", "n.d.", OLD_STYLE)
    assert undated == 5
    if dated:
        text = OLD_STYLE
    assert is_bibliography_chunk(text) is dated


def test_packed_single_paragraph_list_is_bibliography():
    assert "\n" not in PACKED.strip()
    assert is_bibliography_chunk(PACKED)


@pytest.mark.parametrize(
    ("length", "flagged"),
    [(600, False), (601, True)],
    ids=["600-chars-kept", "601-chars-dropped"],
)
def test_packed_line_must_be_longer_than_600_characters(length, flagged):
    """A chunk boundary can cut a packed list anywhere. Cut after 600
    characters, PACKED still holds three DOI/URL markers and five years, so
    only its length decides."""
    line = PACKED[:length]
    assert len(line) == length
    assert is_bibliography_chunk(line) is flagged


@pytest.mark.parametrize(
    ("markers", "flagged"),
    [(2, False), (3, True)],
    ids=["2-markers-kept", "3-markers-dropped"],
)
def test_packed_line_needs_three_doi_or_url_markers(markers, flagged):
    """PACKED with all but ``markers`` of its DOIs printed bare: a bare
    ``10.1109/...`` is no DOI/URL marker, and the line keeps its years and
    stays over 900 characters long."""
    line = PACKED.replace("doi:", "", PACKED.count("doi:") - markers)
    assert len(line) > 900
    assert is_bibliography_chunk(line) is flagged


@pytest.mark.parametrize(
    ("years", "flagged"),
    [(4, False), (5, True)],
    ids=["4-years-kept", "5-years-dropped"],
)
def test_packed_line_needs_five_years(years, flagged):
    """OLD_STYLE packed into one paragraph, its first three entries followed
    by their DOIs: 612 characters, three DOI/URL markers and five years, none
    of them inside a DOI. Printing one year as "n.d." leaves four."""
    dois = [
        "doi:10.1147/rd.14.0309",
        "doi:10.1145/321033.321035",
        "doi:10.1145/361219.361220",
    ]
    lines = OLD_STYLE.splitlines()
    line = " ".join([f"{e} {d}" for e, d in zip(lines, dois)] + lines[len(dois) :])
    if years == 4:
        line = line.replace("1957.", "n.d.")
    assert len(line) > 600
    assert is_bibliography_chunk(line) is flagged


def test_prose_citing_a_few_dois_is_not_bibliography():
    assert not is_bibliography_chunk(PROSE)


def test_long_related_work_paragraph_is_not_bibliography():
    assert len(RELATED_WORK) > 600
    assert not is_bibliography_chunk(RELATED_WORK)


def test_data_availability_paragraph_is_not_bibliography():
    assert not is_bibliography_chunk(PROSE + "\n\n" + DATA_AVAILABILITY)


def test_chunk_straddling_prose_and_four_entries_is_kept():
    """Fewer than five entries after the last prose paragraph: kept, because a
    chunk at the prose/reference boundary still carries the prose."""
    entries = AUTHOR_YEAR.split("\n\n")[:4]
    straddle = PROSE + "\n\nReferences\n\n" + "\n\n".join(entries)
    assert not is_bibliography_chunk(straddle)


@pytest.mark.parametrize(
    "tail",
    [AUTHOR_YEAR, PACKED],
    ids=["five-entries", "packed-line"],
)
def test_chunk_straddling_prose_and_a_full_reference_tail_is_dropped_whole(tail):
    """Accepted residue, "Dropped together with the prose before the list" in
    docs/design/BibliographyChunkFilter.md: the verdict covers the whole chunk,
    so once the tail reaches five entries or one packed line the prose before
    it is dropped too. Cutting the chunk at its reference heading instead would
    change this test and that section together."""
    straddle = PROSE + "\n\nReferences\n\n" + tail
    assert is_bibliography_chunk(straddle)
    assert filter_bibliography_chunks([{"chunk_id": "s", "content": straddle}]) == []


def test_empty_text_is_not_bibliography():
    assert not is_bibliography_chunk("")


def test_hyphen_joined_capitals_are_scanned_in_linear_time():
    """Chunk text is uploaded content. A line that opens like an entry, carries
    a year and no citation cue reaches the author-initials pattern; if every
    capital inside a hyphen- or apostrophe-joined run could start a match, each
    would rescan the rest of the run. 64k characters took seconds that way."""
    for joiner in ("-", "'", "’"):
        line = "Smith, 2020 " + f"A{joiner}" * 32_000
        started = time.perf_counter()
        assert not is_bibliography_chunk(line)
        assert time.perf_counter() - started < 0.5


def test_volume_issue_cue_is_scanned_in_linear_time():
    """A line that opens like an entry and carries a year reaches the citation-cue
    pattern. If the whitespace on either side of the optional separator after a
    ``48(12)`` token could each take part of a run, every split of a long run
    would be tried before the match fails. 32k spaces took seconds that way."""
    for whitespace in (" ", "\t"):
        line = "Smith, 2020 1(1)" + whitespace * 32_000 + "x"
        started = time.perf_counter()
        assert not is_bibliography_chunk(line)
        assert time.perf_counter() - started < 0.5


# ---------------------------------------------------------------------------
# filter_bibliography_chunks
# ---------------------------------------------------------------------------


def test_filter_keeps_order_and_only_removes_bibliographies():
    chunks = [
        {"chunk_id": "a", "content": PROSE},
        {"chunk_id": "b", "content": AUTHOR_YEAR},
        {"chunk_id": "c", "content": DATA_AVAILABILITY},
        {"chunk_id": "d", "content": OLD_STYLE},
        {"chunk_id": "e", "content": None},
    ]
    kept = filter_bibliography_chunks(chunks)
    assert [c["chunk_id"] for c in kept] == ["a", "c", "e"]
    assert len(chunks) == 5  # input list is not mutated


# ---------------------------------------------------------------------------
# process_chunks_unified
# ---------------------------------------------------------------------------


class _RecordingReranker:
    """Records every document list it is handed; keeps the incoming order."""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def __call__(self, query, documents, top_n=None):
        self.calls.append(list(documents))
        return [
            {"index": i, "relevance_score": 1.0 - i / 100}
            for i in range(len(documents))
        ]


def _candidates() -> list[dict]:
    return [
        {"chunk_id": "prose", "content": PROSE, "file_path": "a.pdf"},
        {"chunk_id": "refs", "content": AUTHOR_YEAR, "file_path": "a.pdf"},
        {"chunk_id": "avail", "content": DATA_AVAILABILITY, "file_path": "b.pdf"},
        {"chunk_id": "packed", "content": PACKED, "file_path": "c.pdf"},
        {"chunk_id": "old", "content": OLD_STYLE, "file_path": "d.pdf"},
        {"chunk_id": "related", "content": RELATED_WORK, "file_path": "d.pdf"},
    ]


async def _process(global_config: dict, *, enable_rerank: bool = True) -> list[dict]:
    return await process_chunks_unified(
        query="which retriever works best?",
        unique_chunks=_candidates(),
        query_param=QueryParam(enable_rerank=enable_rerank, chunk_top_k=None),
        global_config=global_config,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False])
async def test_flag_off_is_a_no_op(flag):
    reranker = _RecordingReranker()
    config = {"rerank_model_func": reranker, "min_rerank_score": 0.0}
    if flag is not None:
        config["drop_bibliography_chunks"] = flag

    result = await _process(config)

    everything = [c["content"] for c in _candidates()]
    assert reranker.calls == [everything]
    assert [c["chunk_id"] for c in result] == [c["chunk_id"] for c in _candidates()]


@pytest.mark.asyncio
async def test_flag_on_drops_bibliographies_before_rerank():
    reranker = _RecordingReranker()
    config = {
        "rerank_model_func": reranker,
        "min_rerank_score": 0.0,
        "drop_bibliography_chunks": True,
    }

    result = await _process(config)

    # The reranker is called once and never sees a reference list ...
    assert len(reranker.calls) == 1
    assert reranker.calls[0] == [PROSE, DATA_AVAILABILITY, RELATED_WORK]
    # ... and prose with DOIs and the data-availability paragraph survive.
    assert [c["chunk_id"] for c in result] == ["prose", "avail", "related"]
    assert [c["id"] for c in result] == ["DC1", "DC2", "DC3"]


@pytest.mark.asyncio
async def test_flag_on_without_rerank_still_drops_bibliographies():
    config = {"drop_bibliography_chunks": True}

    result = await _process(config, enable_rerank=False)

    assert [c["chunk_id"] for c in result] == ["prose", "avail", "related"]


@pytest.mark.asyncio
async def test_flag_on_frees_chunk_top_k_slots_for_lower_ranked_candidates():
    """The filter runs before ``chunk_top_k``, so when the pool is larger than
    ``chunk_top_k`` (KG modes) the next candidates take the freed slots."""
    result = await process_chunks_unified(
        query="which retriever works best?",
        unique_chunks=_candidates(),
        query_param=QueryParam(enable_rerank=False, chunk_top_k=3),
        global_config={"drop_bibliography_chunks": True},
    )

    assert [c["chunk_id"] for c in result] == ["prose", "avail", "related"]


@pytest.mark.asyncio
async def test_flag_on_with_only_bibliographies_returns_nothing_and_skips_rerank():
    reranker = _RecordingReranker()
    config = {"rerank_model_func": reranker, "drop_bibliography_chunks": True}

    result = await process_chunks_unified(
        query="which retriever works best?",
        unique_chunks=[
            {"chunk_id": "refs", "content": AUTHOR_YEAR},
            {"chunk_id": "old", "content": OLD_STYLE},
        ],
        query_param=QueryParam(enable_rerank=True, chunk_top_k=None),
        global_config=config,
    )

    assert result == []
    assert reranker.calls == []
