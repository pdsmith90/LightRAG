#!/usr/bin/env python3
"""Behavioural tests for build_corpus._strip_references.
Run:  python -m pytest tests/corpus      (plain asserts; also runnable as a script)

The prose block is deliberately paper-sized (~8 KB against ~1.1 KB of references):
a document that is mostly bibliography is kept whole by the _REF_MAX_SHRINK valve,
which the last test covers explicitly. Every prose line carries a year, so the
block doubles as a negative test for the entry detector."""

import os, sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "corpus")
)
from build_corpus import _strip_references, clean_md

PROSE = (
    "\n".join(
        "The 2014 trial compared three scheduling heuristics on a benchmark suite (Okafor et al., 2011). "
        "Throughput rose in most runs, and we list the gain and the slowdown seen for each workload "
        f"in region {i} next to the seed-to-seed spread measured in the rerun of 2013."
        for i in range(30)
    )
    + "\n"
)
AGU_REFS = """Benjamini, Y., and Y. Hochberg (1995), Controlling the false discovery rate: A practical and powerful approach to multiple testing, J. R. Stat. Soc. B, 57, 289–300, doi:10.1111/j.2517-6161.1995.tb02031.x.
Dee, D. P., S. M. Uppala, A. J. Simmons, P. Berrisford, P. Poli, S. Kobayashi, et al. (2011), The ERA-Interim reanalysis: Configuration and performance of the data assimilation system, Q. J. R. Meteorol. Soc., 137, 553–597, doi:10.1002/qj.828.
Hansen, J., R. Ruedy, M. Sato, and K. Lo (2010), Global surface temperature change, Rev. Geophys., 48, RG4004, doi:10.1029/2010RG000345.
Hurrell, J. W. (1995), Decadal trends in the North Atlantic Oscillation: Regional temperatures and precipitation, Science, 269, 676–679, doi:10.1126/science.269.5224.676.
Madden, R. A., and P. R. Julian (1972), Description of global-scale circulation cells in the tropics with a 40–50 day period, J. Atmos. Sci., 29(6), 1109–1123.
Mantua, N. J., S. R. Hare, Y. Zhang, J. M. Wallace, and R. C. Francis (1997), A Pacific interdecadal climate oscillation with impacts on salmon production, Bull. Am. Meteorol. Soc., 78(6), 1069–1079.
"""
APA_BULLETS = """- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81, 637–654. https://doi.org/10.1086/260062
- Brooks, F. P. Jr. (1975). The mythical man‐month: Essays on software engineering. Reading, MA: Addison‐Wesley.
- Riess, A. G., Filippenko, A. V., Challis, P., Clocchiatti, A., Diercks, A., Garnavich, P. M., et al. (1998). Observational evidence from supernovae for an accelerating universe and a cosmological constant. Astron. J., 116, 1009–1038. https://doi.org/10.1086/300499
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back‐propagating errors. Nature, 323, 533–536. https://doi.org/10.1038/323533a0
- Mandelbrot, B. (1967). How long is the coast of Britain? Statistical self‐similarity and fractional dimension. Science, 156(3775), 636–638. https://doi.org/10.1126/science.156.3775.636
"""


def test_agu_list_with_bold_heading_removed_prose_kept():
    doc = (
        "# Pooled estimates of regional survey bias\n\n"
        + PROSE
        + "\n## **References**\n\n"
        + AGU_REFS
    )
    out = _strip_references(doc)
    assert out.count("scheduling heuristics") == 30 and "region 29" in out
    assert "Benjamini" not in out and "Mantua" not in out and "References" not in out


def test_apa_bulleted_list_removed_following_section_kept():
    doc = (
        PROSE
        + "\nREFERENCES\n"
        + APA_BULLETS
        + "\n## Appendix A\n\nThe survey weights are tabulated below.\n"
    )
    out = _strip_references(doc)
    assert (
        "Scholes" not in out and "Mandelbrot" not in out and "\nREFERENCES\n" not in out
    )
    assert "Appendix A" in out and "survey weights are tabulated" in out


def test_textbook_per_chapter_lists_removed_chapters_kept():
    ch = lambda k: (
        f"## Chapter {k}\n\nThe cell is the basic structural unit of every organism.\n"
        + PROSE
        + "\n### References\n\n"
        + AGU_REFS
        + "\n"
    )
    out = _strip_references(ch(1) + ch(2))
    assert (
        out.count("basic structural unit") == 2
        and out.count("scheduling heuristics") == 60
    )
    assert "Benjamini" not in out and "### References" not in out


def test_wrapped_entries_and_trailing_continuation():
    doc = (
        PROSE
        + """
References

Charney, J. G., R. Fjörtoft, and J. von Neumann (1950), Numerical integration of the barotropic vorticity
equation, Tellus, 2(4), 237–254.
Hanks, T. C., and H. Kanamori (1979), A moment magnitude scale, J. Geophys. Res.,
84(B5), 2348–2350, doi:10.1029/JB084iB05p02348.
Perlmutter, S., G. Aldering, G. Goldhaber, R. A. Knop, P. Nugent, et al. (1999), Measurements of Omega and
Lambda from 42 high-redshift supernovae, Astrophys. J., 517, 565–586.
Watts, D. J., and S. H. Strogatz (1998), Collective dynamics of 'small-world'
networks, Nature, 393, 440–442, doi:10.1038/30918.
Tversky, A., and D. Kahneman (1974), Judgment under uncertainty: Heuristics and biases,
Science, 185, 1124–1131, doi:10.1126/science.185.4157.1124.
"""
    )
    out = _strip_references(doc)
    assert out.count("scheduling heuristics") == 30
    assert "Charney" not in out and "science.185.4157.1124" not in out, out[-300:]


def test_prose_with_inline_citations_untouched():
    doc = (
        PROSE
        + "\n".join(
            [
                "Black and Scholes (1973) derived an option price from a riskless hedging argument.",
                "Later work (Merton, 1973; Cox et al., 1979) showed the price is model dependent.",
                "In Figure 2 the 2008 financial crisis appears as a 40 percent drop in market value.",
                "The 1987 market crash is visible in the implied volatility series after 1988.",
                "Since 2018, a second exchange has continued the record with a faster matching engine.",
                "Trends over 2002–2016 are dominated by growth in Asia and North America.",
            ]
        )
        + "\n"
    )
    assert _strip_references(doc) == doc


def test_table_of_products_with_years_and_dois_untouched():
    doc = (
        PROSE
        + """
| Product | Release | Year | DOI |
|---|---|---|---|
| Household panel | wave 9 | 2011 | doi:10.5555/hp.wave9.0411 |
| Labour survey | Q3 | 2014 | doi:10.5555/ls.2014.q3 |
| Price index | base 2010 | 2015 | doi:10.5555/pi.b2010.77 |
| Trade flows 2012 | 2012 | 2013 | doi:10.5555/tf.2012.1188 |
| Census extract | E2 | 2016 | doi:10.5555/ce.e2.5021 |
"""
    )
    assert _strip_references(doc) == doc


def test_short_list_kept():
    doc = PROSE + "\n## References\n\n" + "\n".join(AGU_REFS.splitlines()[:3]) + "\n"
    assert _strip_references(doc) == doc


def test_safety_valve_keeps_a_document_that_is_mostly_references():
    doc = "Short note.\n\n## References\n\n" + AGU_REFS + AGU_REFS
    assert _strip_references(doc) == doc


def test_numbered_heading_and_clean_md_integration():
    doc = "# Title\n\n" + PROSE + "\n## 7. References\n\n" + AGU_REFS
    out = clean_md(doc)
    assert "region 29" in out and "Benjamini" not in out and "7. References" not in out


# false positives a sweep over a real corpus found
def test_ocr_page_lines_with_years_and_volume_are_not_entries():
    page = (
        "LETTERS TO NATURE a meridional reflection at 3.4 A and a layer line every 34 A appear in the "
        "photographs of the wet fibres, and both persist when the NATURE VOL. 171 25 APRIL 1953 specimens are "
        "dried slowly over two days, Department of Crystallography, 7 Rue J. Perrin, 69007 "
        "Lyon, pp. 737 of the letter propose that two coiled chains share one common axis. "
    ) * 3
    doc = (
        "# A synthetic letter on fibre diffraction\n\n**Year:** 1953  **DOI:** 10.5555/171737a0\n\n---\n"
        + "\n\n".join([page] * 8)
        + "\n"
    )
    assert _strip_references(doc) == doc


def test_two_page_comment_keeps_header_and_title():
    doc = (
        "# Corrections to: Rational approximations for the incomplete beta function by Roe (2016) in J. Ex. 45(7):902–917\n\n"
        "**Authors:** Doe Jo-Ann\n**Year:** 2021  **Published in:** Journal of Examples  **DOI:** 10.5555/jex.2021.0042\n"
        "**Zotero:** zotero://select/library/items/TEST0003\n\n> **Abstract.** A fast method based on continued fractions (Roe 2016) has been proposed, doi:10.5555/jex.2016.0117.\n\n---\n"
        "Journal of Examples (2021) 50:3 https://doi.org/10.5555/jex.2021.0042\n\n**COMMENT**\n\n"
        "# Corrections to: Rational approximations for the incomplete beta function by Roe (2016) in J. Ex. 45(7):902–917\n\n"
        "The published Eq. (9) omits a factor of 2 in the 2016 derivation; Table 2 (2021) lists the corrected values, pp. 5–6.\n"
    )
    assert _strip_references(doc) == doc


def test_bibliography_packed_into_one_paragraph_removed():
    packed = " ".join(AGU_REFS.splitlines())  # six entries, one line, ~1.1 KB
    doc = PROSE + "\n## References\n\n" + packed + "\n"
    out = _strip_references(doc)
    assert (
        out.count("scheduling heuristics") == 30
        and "Benjamini" not in out
        and "## References" not in out
    )


def test_long_paragraph_that_starts_with_a_citation_is_kept():
    para = (
        "Benjamini, Y., and Y. Hochberg (1995) showed that error rates depend on the number of tests; "
        "Storey (2002) and Efron et al. (2001) extended this to empirical Bayes, and since 2018 open-source "
        "packages have continued the work. "
    ) * 6
    doc = PROSE + para + "\n"
    assert _strip_references(doc) == doc


# ---- v3: styles that real books, theses and journal documents use ----
SPRINGER_CHAPTER = """- 1.1 C.E. Shannon: Communication theory of secrecy systems, Bell Syst. Tech. J. **28** (4), 656–715 (1949)

- 1.2 R.C. Merkle: Secure communications over insecure channels, Commun. ACM **21** (4), 294–299 (1978)

- 1.3 A.M. Turing: On computable numbers, with an application to the Entscheidungsproblem, Proc. Lond. Math. Soc. **42** (1), 230–265 (1937)

- 1.4 R.L. Rivest, A. Shamir: How to expose an eavesdropper, Commun. ACM **27** , 393–395 (1984)

- 1.5 D.E. Knuth: The Art of Computer Programming: Seminumerical Algorithms (Addison-Wesley, Reading 1969)

- 1.6 M.R. Garey, D.S. Johnson: Computers and Intractability: A Guide to the Theory of NP-Completeness (W.H. Freeman, San Francisco 1979)
"""
SPRINGER_BASIC_BULLETS = """- Doe EG (1971) Rarefaction curves for small samples of ground beetles in Fisher’s log series. Ecoser B-9, Rep, School of Biology, University of Example, Exampleton
- Roe R (1993) Growth curves, mortality rates and recruitment estimates for a managed coastal fish stock. Rep 212, Department of Zoology, Example College, Riverton
- Roe R (1996) Stock assessment from catch series by spectral methods. Rev Écol Ex 41:77–95
- Roe R, Poe MG (1999) On sampling effort in bottom trawl surveys. In: Sørli E, Moe K, Coe CC (eds) Festschrift to John Roe. Fiskeriinstitutt, Bergen, pp 51–66
- Krämer W (2001) Zur Schätzung von Fischbeständen mit Markierungs- und Wiederfangverfahren. PhD Thesis, Universität Beispielstadt
- Böhm B, Moe A, Lenz H-P (2008) SONAR: the new acoustic fish counting system. J Fish Ex 33:410–419
"""
ANNUAL_REVIEWS = """- Hodgkin AL, Huxley AF, Katz B. 1952. Measurement of current-voltage relations in the membrane of the giant axon of Loligo. _J. Physiol._ 116:424–48
- Hodgkin AL, Huxley AF. 1952. A quantitative description of membrane current and its application to conduction and excitation in nerve. _J. Physiol._ 117:500–44
- MacArthur RH, Wilson EO. 1963. An equilibrium theory of insular zoogeography. _Evolution_ 17:373–87
- Watson JD, Crick FHC. 1953. Molecular structure of nucleic acids: a structure for deoxyribose nucleic acid. _Nature_ 171:737–38
- Sokal RR, Rohlf FJ. 1995. _Biometry_ . New York: Freeman. 3rd ed.
- Erdős P, Rényi A. 1960. On the evolution of random graphs. _Publ. Math. Inst. Hung. Acad. Sci._ 5:17–61
"""
CHICAGO_BOOKS = """- Kernighan, Brian W. and Rob Pike. 1999. _The Practice of Programming._ Reading, MA: Addison-Wesley.
- Kuhn, Thomas. 1962. _The Structure of Scientific Revolutions._ Chicago, IL: University of Chicago Press.
- Bureau of Labor Statistics. 1997. _BLS Handbook of Methods._ Washington, DC: U.S. Government Printing Office.
- Doe, JR. 2004. _Why Do Prices Stick? Evidence from Retail Scanner Data._ Springfield, IL: Example Press.
- National Research Council. 1996. _National Science Education Standards._ Washington, DC: National Academy Press.
- Samuelson, P.A. 1947. _Foundations of Economic Analysis._ Cambridge, MA: Harvard University Press.
- Tufte, Edward R. 1983. _The Visual Display of Quantitative Information._ Cheshire, CT: Graphics Press.
"""
PANDOC_ENDNOTE = """<span id="_ENREF_1" class="anchor"></span>Hersbach, H., Bell, B.,
Berrisford, P., Hirahara, S., Horányi, A., Muñoz‐Sabater, J., et al. (2020),
The ERA5 global reanalysis, *Quarterly Journal of the Royal
Meteorological Society*, *146*(730), 1999-2049.

<span id="_ENREF_2" class="anchor"></span>Dziewonski, A. M., & Anderson, D. L. (1981), Preliminary reference
Earth model, *Physics of the Earth and Planetary
Interiors*, *25*(4), 297-356.

<span id="_ENREF_3" class="anchor"></span>Stein, S., & Wysession, M. (2003), An
introduction to seismology, earthquakes, and Earth structure, *Blackwell
Publishing (Malden, MA)*.

<span id="_ENREF_4" class="anchor"></span>Gutenberg, B., & Richter, C. F. (1944), Frequency of earthquakes
in California, *Bulletin of the Seismological Society of America*,
*34*(4), 185-188.

<span id="_ENREF_5" class="anchor"></span>Lorenz, E. N. (1963), Deterministic nonperiodic flow,
*Journal of the Atmospheric Sciences*, *20*(2),
130-141.

<span id="_ENREF_6" class="anchor"></span>Kalnay, E., Kanamitsu, M., Kistler, R., Collins, W., Deaven, D.,
Gandin, L., et al. (1996), The NCEP/NCAR 40-year reanalysis project,
*Bulletin of the American Meteorological Society*, *77*(3), 437-472.
"""
OCR_WRAPPED = """Doe, H., Field trials of the winter wheat varieties sown on October 3,
1955, Yield response at the boundary of two soil types, Agron. Ex., 7, 44-58, 1958.
-- , Roe, T., and A. Poe, Grain yield on the Hill Farm plots before and after the severe frost of February 9, 1956, Bull. Agric. Ex. Stn., 14, 201-212, 1957.
Poe, A., and T. Roe, Grain yield in the vicinity of
the Hill Farm plots after the severe frost in 1956, Bull. Agric. Ex. Stn., 15, 88-97, 1959.
Poe, A., Grain yield in the northern part of the valley,
upper Example Valley, Bull. Agric. Ex. Stn., 21, 330-341, 1962.
Moe, G., Yield losses associated with the 1958 summer
drought, Agron. Ex., 11, 612-620, 1961.
Zoe, L. R., Root growth of cereals and nature of tillering on the
chalk downlands, J. Agric. Ex., 48, 1107-1119, 1963.
"""
LINE_NUMBERED = """- 513 Doe, R.M., Roe, G., Ibáñez, A., Van Buren, M., 2009. A colony’s foraging range inferred from its nest 514 size, location, and worker count. Journal of Examples 112, 55–68.
516 Cooley, J.W., Tukey, J.W., 1965. An algorithm for the machine calculation of complex Fourier series. Mathematics 517 of Computation 19, 297–301.
518 Poe, M., 2011. Nest architecture of social insects: ants, wasps, and kin. Journal of Examples 131, 402–431.
521 Moe, M., Coe, A., Zoe, A., 2012. Africa’s and Asia’s savanna termites limited by seasonal 522 food supply. Journal of Examples 19, 7021–7033.
523 Axelrod, R., Hamilton, W.D., 1981. The evolution of cooperation. Science 211, 1390–1396.
- 535 Roe, G.D., Fay, R.D., 2006. Daily and weekly activity rhythms from long-term hive recordings. 536 Journal of Examples 24, 3308.
"""
EMPHASIS_OCR = """- Box, **_G._** E. **P. (1976).** Science and statistics. **_J. Am. Stat. Assoc., 71,_ 791-799.**
- Box, **G.** E. P. **(1979).** Robustness in the strategy of scientific model building. **_In_** R. L. Launer and G. N. Wilkinson (Eds.), **_Robustness in Statistics_** 1979, pp. **201-236.** Academic Press.
- Cox, **D. R. (1972).** Regression models and life-tables. **_J. R. Stat. Soc. B, 34,_ 187-220.**
- Efron, **B. (1979).** Bootstrap methods: another look at the jackknife. **_Ann. Stat., 7,_ 1-26.**
- Hastings, **W. K. (1970).** Monte Carlo sampling methods using Markov chains and their applications. **_Biometrika, 57,_ 97-109.**
- Wald, **A. (1947).** _Sequential Analysis._ New York: Wiley.
"""


def test_springer_chapter_lists_removed_two_chapters_kept():
    ch = lambda k: (
        f"## {k} Introduction to Cryptography\n\nMachine ciphers began with Enigma in 1918.\n"
        + PROSE
        + "\n###### **References**\n\n"
        + SPRINGER_CHAPTER
        + "\n"
    )
    out = _strip_references(ch(1) + ch(2))
    assert (
        out.count("began with Enigma") == 2 and out.count("scheduling heuristics") == 60
    )
    assert "Shannon" not in out and "Knuth" not in out and "**References**" not in out


def test_springer_basic_bullets_removed():
    doc = (
        PROSE
        + "\n## **References**\n\n"
        + SPRINGER_BASIC_BULLETS
        + "\n## **Appendix**\n\nThe variance derivations follow.\n"
    )
    out = _strip_references(doc)
    assert "Roe R" not in out and "Krämer" not in out and "## **References**" not in out
    assert (
        "variance derivations follow" in out
        and out.count("scheduling heuristics") == 30
    )


def test_annual_reviews_style_removed():
    doc = PROSE + "\nLITERATURE CITED\n\n" + ANNUAL_REVIEWS
    out = _strip_references(doc)
    assert (
        "Hodgkin AL" not in out and "Sokal" not in out and "LITERATURE CITED" not in out
    )
    assert out.count("scheduling heuristics") == 30


def test_chicago_book_list_with_organisation_entries_removed():
    doc = (
        PROSE
        + "\n###### References\n\n"
        + CHICAGO_BOOKS
        + "\n##### Chapter 3\n\n##### Project Evaluation\n\nEvaluation begins with the project charter.\n"
    )
    out = _strip_references(doc)
    assert (
        "Kuhn" not in out
        and "National Research Council" not in out
        and "###### References" not in out
    )
    assert "Chapter 3" in out and "project charter" in out


def test_pandoc_endnote_wrapped_entries_removed_with_heading_and_first_fragment():
    doc = (
        PROSE
        + "\nReferences\n\n"
        + PANDOC_ENDNOTE
        + "\nSupporting Information\n\nFigure S1 shows the residuals.\n"
    )
    out = _strip_references(doc)
    assert "Hersbach" not in out and "_ENREF_1" not in out and "437-472" not in out
    assert (
        "\nReferences\n" not in out
        and "Figure S1 shows" in out
        and out.count("scheduling heuristics") == 30
    )


def test_ocr_hard_wrapped_entries_with_same_author_dashes_removed():
    doc = PROSE + "\nREFERENCES\n\n" + OCR_WRAPPED
    out = _strip_references(doc)
    assert "Doe" not in out and "Roe" not in out and "Zoe" not in out
    assert out.count("scheduling heuristics") == 30


def test_line_numbered_manuscript_references_removed():
    doc = PROSE + "\n512 References\n\n" + LINE_NUMBERED
    out = _strip_references(doc)
    assert (
        "Ibáñez" not in out
        and "activity rhythms" not in out
        and out.count("scheduling heuristics") == 30
    )


def test_emphasis_interleaved_ocr_entries_removed():
    doc = PROSE + "\n###### **_412 References_**\n\n" + EMPHASIS_OCR
    out = _strip_references(doc)
    assert "Box" not in out and "Wald" not in out and "412 References" not in out
    assert out.count("scheduling heuristics") == 30


def test_table_of_contents_with_indices_and_years_kept():
    toc = (
        "\n".join(
            [
                "- 1.1 Introduction ..... 2",
                "- 1.2 Early Calculating Machines (1936–1951) ..... 4",
                "- 1.3 The Minicomputer Years, 1959–1978 ..... 11",
                "- 1.4 Packet Switching After 1969 ..... 17",
                "- 1.5 COBOL, Fortran and Pascal Since 1970 ..... 26",
                "- 1.6 Outlook to 2035 ..... 38",
                "- 2.1 Character Encodings and 1963 Standards ..... 45",
            ]
        )
        + "\n"
    )
    doc = "# Handbook of Computing\n\n## Contents\n\n" + toc + "\n" + PROSE
    assert _strip_references(doc) == doc


def test_further_reading_cross_references_kept():
    doc = (
        PROSE
        + "\nFurther reading:\n\n"
        + "\n".join(
            [
                "Chapter 2: “Installation and First Steps”, page 19",
                "Section 11.4: “Configuration Keys”, page 233",
                "Section 12.2: “Logging Levels”, page 258",
                "Section 14.5: “Checking Results Against the Sample Data”, page 301",
                "Section 15.3: “Migrating Archives from the Year 2008”, page 327",
            ]
        )
        + "\n\n##### 14.5.1. Checking\n\nWhen the import finishes, compare its totals with the sample data.\n"
    )
    assert _strip_references(doc) == doc


def test_hard_wrapped_ocr_prose_with_years_kept():
    prose = (
        "\n".join(
            [
                "However, the 2008 financial crisis produced a sharp contraction that",
                "Finally, NBER dating resolved the 2010 trough, as the 2011 slowdown later showed,",
                "In Table 3, values for 2005–2010 are compared with the model of 2013,",
                "Moreover, the 1929 market crash predates the modern data era, so the",
                "Nevertheless, results since 2002 agree with the 2016 survey estimates,",
                "Consequently, the 1933 trough remains the deepest recorded, and",
                "Therefore, trends over 2002–2016 are dominated by services growth, which",
                "Additionally, the 2018 launch of the new index continued the record with",
            ]
        )
        + "\n"
    )
    doc = "# Title\n\n**Year:** 2020\n\n---\n" + prose + "\n" + PROSE
    assert _strip_references(doc) == doc


def test_v3_never_edits_text_it_keeps():
    body = (
        PROSE
        + '\nThe **bold** and _italic_ words, <span id="x">tags</span>, and Ren´e\'s accent stay.\n'
    )
    doc = body + "\n## References\n\n" + AGU_REFS
    out = _strip_references(doc)
    assert out.startswith(body) and "Benjamini" not in out


def test_two_column_interleaved_prose_paragraph_and_its_heading_kept():
    bold = lambda l: "- **" + l.rstrip() + "**"
    refs = [bold(l) for l in AGU_REFS.splitlines()]
    prose = (
        "**Across the four field seasons of 2004-2007 the trap counts rose steadily at the upland sites "
        "while the lowland counts stayed flat, which points to local habitat change rather than regional "
        "weather as the main driver; separating the two needs a longer record and more sites.**"
    )
    doc = (
        PROSE
        + "\n## References\n\n"
        + "\n\n".join(refs[:4])
        + "\n\n##### **Conclusions**\n\n"
        + prose
        + "\n\n"
        + "\n\n".join(refs[4:] + refs[:3])
        + "\n"
    )
    out = _strip_references(doc)
    assert "Benjamini" not in out and "Mantua" not in out and "## References" not in out
    assert (
        "##### **Conclusions**" in out
        and "trap counts rose" in out
        and out.count("scheduling heuristics") == 30
    )


def test_ocr_blank_separated_wrapped_entries_with_junk_prefixes_removed():
    lines = OCR_WRAPPED.splitlines()
    lines[3] = "/'" + lines[3]  # "/'Poe, A., and T. Roe, ..." OCR junk
    lines[7] = "> " + lines[7]
    doc = PROSE + "\nREFERENCES\n\n" + "\n\n".join(lines) + "\n"
    out = _strip_references(doc)
    assert (
        "Doe" not in out
        and "Poe" not in out
        and "Zoe" not in out
        and "REFERENCES" not in out
    )
    assert out.count("scheduling heuristics") == 30


def test_degruyter_colon_year_style_three_word_surnames_umlauts_and_org_entries_removed():
    refs = """De Vries-van Berg, A., Kaminski, R., Lloyd-Evans, M. et al. (2011): Combining trap and transect counts to estimate butterfly abundance on farmland. J. Ex. Ecol. 23: 144–157.

Hölder, O. (1889): Ueber einen Mittelwerthsatz. Nachr. Ges. Wiss. Göttingen 2: 38–47.

Gödel, K. (1931): Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I. Monatsh. Math. Phys. 38: 173–198.

ILO (1999): _International Labour Organization_ . Annual Report 1998, ILO Geneva.

Schäfer, K. (1987): _Untersuchungen zur Brutbiologie des Weißstorchs nach Heinroth’scher Methode_ . Diss. B 57, Freiburg.

Kova ˇc, M., Hughes, D.E. (1983): Sampling errors of household consumption surveys. Scand. J. Ex. 12: 88–97.

Lindqvist, Maria (2012): Population ecology: Models, data and estimation. 2<sup>nd</sup> ed., Springer, New York.
"""
    doc = (
        PROSE
        + "\n274 References\n\n"
        + refs
        + "\n##### **Index**\n\nAbundance index 12\n"
    )
    out = _strip_references(doc)
    assert (
        "Hölder" not in out
        and "Schäfer" not in out
        and "Vries-van" not in out
        and "274 References" not in out
    )
    assert (
        "##### **Index**" in out
        and "Abundance index" in out
        and out.count("scheduling heuristics") == 30
    )


def test_line_numbered_draft_with_bracket_indices_and_wrapped_year_lines_removed():
    refs = """- 1874 [1] D. Hartmann, J. Moe, and U. Coe.  Counting songbirds by ear.

- 1875 _Journal of Examples in Ecology_ , 19(3):031207, 2008.

- 1876 [2] SONGLOG-I (Song Logging Unit-I)/SONGLOG-II. URL https://archive.example.org/web/devices/field-recorders/s/songlog.

- 1878 [3] N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller. Equation 1879 of state calculations by fast computing machines. _J. Chem. Phys._ , 1880 21(6):1087–1092, 1953.

- 1881 [4] C. Fontaine, H. Ju¨rgens, and P. Novak. Progress of the ringing scheme. _Adv. Ex. Ecol._ , 1882 12:301–309, March 2006.

- 1883 [5] A. Lund, C. Moreau, and R. K. Sato. Long-term monitoring of migrant

- 1884 populations: Lessons and open problems. _J. Ex. Ornithol._ , 58:3301– 1885 3319, 2004. doi: 10.5555/jeo.2004.0311.

- 1886 [6] ECO. _Survey Design and Sampling Requirements for National Bird_ 1887 _Atlases, 1991-2000: Report of an Expert Workshop_ . ECO, Geneva, 1990.

- 1888 [7] J. F. Clauser, M. A. Horne, A. Shimony, and R. A. Holt. Proposed experiment to test local hidden-variable theories. _Phys. Rev. Lett._ , 23(15):880, 1969.
"""
    doc = PROSE + "\n### 1873 **References**\n\n" + refs + "1894\n\n1895\n"
    out = _strip_references(doc)
    assert (
        "Hartmann" not in out
        and "Clauser" not in out
        and "Fontaine" not in out
        and "**References**" not in out
    )
    assert out.count("scheduling heuristics") == 30


# the valve's second tier and its build-log lines. A review article whose references
# came to just over the 0.60 valve was ingested with its whole reference list, which
# then made up most of its chunks.
BIG_BODY = PROSE + PROSE  # ~16 KB of prose, over _REF_BIG_BODY


def test_relaxed_valve_strips_a_review_whose_list_is_most_of_the_text():
    doc = BIG_BODY + "\n## References\n\n" + AGU_REFS * 26  # references 65% of the text
    out = _strip_references(doc)
    assert (
        out.count("scheduling heuristics") == 60
        and "Benjamini" not in out
        and "Mantua" not in out
    )


def test_valve_still_keeps_whole_above_the_relaxed_ceiling():
    doc = (
        BIG_BODY + "\n## References\n\n" + AGU_REFS * 60
    )  # 81%: the detector may be wrong
    assert _strip_references(doc) == doc


def test_valve_logs_the_document_only_during_a_build():
    import contextlib, io, build_corpus as bc

    doc = "Short note.\n\n## References\n\n" + AGU_REFS + AGU_REFS
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bc._strip_references(doc)  # no document named: silent
    assert buf.getvalue() == ""
    bc._ref_doc = "ABCD1234__note.md"
    try:
        with contextlib.redirect_stdout(buf):
            bc._strip_references(doc)
            bc._strip_references(doc)  # clean_md runs twice on PDFs
    finally:
        bc._ref_doc = ""
    assert buf.getvalue().count("VALVE kept whole ABCD1234__note.md") == 1
    buf = io.StringIO()
    bc._ref_doc = "WXYZ5678__review.md"
    try:
        with contextlib.redirect_stdout(buf):
            bc._strip_references(BIG_BODY + "\n## References\n\n" + AGU_REFS * 26)
    finally:
        bc._ref_doc = ""
    assert "VALVE relaxed WXYZ5678__review.md" in buf.getvalue()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
