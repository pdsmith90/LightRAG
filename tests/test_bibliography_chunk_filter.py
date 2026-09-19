"""Bibliography chunk filter (pdrag-deploy, 2026-09-19). pytest-compatible; also runs
as a script:  python tests/test_bibliography_chunk_filter.py"""
import asyncio

from lightrag.utils import (
    drop_bibliography_chunks,
    is_bibliography_chunk,
    process_chunks_unified,
)

PROSE = (
    "Mascon solutions reduce leakage relative to spherical-harmonic products (Save et al., 2016), "
    "as Landerer and Swenson (2012) anticipated; see https://doi.org/10.1002/2016WR019494 and "
    "doi:10.1029/2011WR011453 for the two evaluations.\n\nWe evaluate them against hydrologic "
    "models over 176 basins (2002–2016) and quantify the residual signal after the GIA correction."
)
AGU = """Landerer, F. W., and S. C. Swenson (2012), Accuracy of scaled GRACE terrestrial water storage estimates, Water Resour. Res., 48, W04531, doi:10.1029/2011WR011453.

Long, D., B. R. Scanlon, L. Longuevergne, A. Y. Sun, D. N. Fernando, and H. Save (2013), GRACE satellite monitoring of large depletion in water storage, Geophys. Res. Lett., 40, 3395–3401, doi:10.1002/grl.50655.

Long, D., L. Longuevergne, and B. R. Scanlon (2015), Global analysis of approaches for deriving total water storage changes from GRACE satellites, Water Resour. Res., 51, 2574–2594, doi:10.1002/2014WR016853.

Longuevergne, L., B. R. Scanlon, and C. R. Wilson (2010), GRACE hydrological estimates for small basins, Water Resour. Res., 46, W11517, doi:10.1029/2009WR008564.

Luthcke, S. B., T. J. Sabaka, B. D. Loomis, A. A. Arendt, J. J. McCarthy, and J. Camp (2013), Antarctica, Greenland and Gulf of Alaska land-ice evolution, J. Glaciol., 59(216), 613–631.
"""
OLD_STYLE = """HEISKANEN W. A. & MORITZ H. 1967. Physical Geodesy. Freeman, San Francisco.
KAULA W. M. 1966. Theory of Satellite Geodesy. Blaisdell, Waltham, pp. 1–124.
TALWANI M. 1998. Errors in the total Bouguer reduction. Geophysics 63, 1125–1130.
WAHR J., MOLENAAR M. & BRYAN F. 1998. Time variability of the Earth's gravity field. J. Geophys. Res. 103, 30205–30229.
TAPLEY B. D., BETTADPUR S., RIES J. C., THOMPSON P. F. & WATKINS M. M. 2004. GRACE measurements of mass variability in the Earth system. Science 305, 503–505.
"""
PACKED = " ".join(AGU.split("\n\n"))  # one paragraph, five DOIs, no line structure


def test_reference_list_with_dois_is_bibliography():
    assert is_bibliography_chunk(AGU)


def test_old_style_list_without_dois_is_bibliography():
    assert is_bibliography_chunk(OLD_STYLE)


def test_packed_single_paragraph_list_is_bibliography():
    assert is_bibliography_chunk(PACKED)


def test_prose_with_inline_citations_and_two_dois_is_not():
    assert not is_bibliography_chunk(PROSE)


def test_chunk_straddling_prose_and_three_entries_is_kept():
    straddle = PROSE + "\n\nReferences\n\n" + "\n\n".join(AGU.split("\n\n")[:3])
    assert not is_bibliography_chunk(straddle)


def test_data_availability_paragraph_with_dataset_dois_is_kept():
    avail = ("Data availability. CSR RL06 mascons: https://doi.org/10.15781/cgq9-nh24; JPL mascons: "
             "https://doi.org/10.5067/TEMSC-3JC62; GSFC mascons: https://doi.org/10.5067/GRACE-GSFC-M2; "
             "GLDAS: https://doi.org/10.5067/E7TYRXPJKWOQ; ERA5: https://doi.org/10.24381/cds.adbb2d47.")
    assert not is_bibliography_chunk(PROSE + "\n\n" + avail)


def test_drop_keeps_order_and_only_removes_bibliographies():
    chunks = [{"id": "a", "content": PROSE}, {"id": "b", "content": AGU},
              {"id": "c", "content": PROSE.upper()}, {"id": "d", "content": OLD_STYLE}]
    assert [c["id"] for c in drop_bibliography_chunks(chunks)] == ["a", "c"]


def test_process_chunks_unified_honours_the_opt_in():
    from lightrag.base import QueryParam
    chunks = [{"id": "a", "content": PROSE}, {"id": "b", "content": AGU}]
    qp = QueryParam(enable_rerank=False, chunk_top_k=None)
    on = asyncio.run(process_chunks_unified("q", list(chunks), qp,
                                            {"drop_bibliography_chunks": True, "tokenizer": None}))
    off = asyncio.run(process_chunks_unified("q", list(chunks), qp,
                                             {"drop_bibliography_chunks": False, "tokenizer": None}))
    # process_chunks_unified renumbers ids (DC1, DC2, ...) for citation markers, so compare content
    assert [c["content"][:30] for c in on] == [PROSE[:30]]
    assert [c["content"][:30] for c in off] == [PROSE[:30], AGU[:30]]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
