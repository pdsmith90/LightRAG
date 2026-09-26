"""Unit tests for build_corpus.py: clean_md passes, the metadata header, naming, file choice."""
import os
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_testlib  # noqa: F401  (puts corpus/ on sys.path)
from build_corpus import _drop_repeats, clean_md, header, pick_primary, slug

B64_RUN = (string.ascii_letters + string.digits + "+/") * 4          # 256 chars, unbroken
TOPICS = ["hydration", "kneading", "proofing", "shaping", "scoring", "baking", "cooling", "storage"]
PARA = ("This paragraph on {w} walks through a recipe for sourdough bread, from feeding the "
        "starter to shaping the loaf, and explains why {w} matters most on cold days "
        "when the dough rises slowly overnight and the crust needs a hotter oven.")


def test_base64_runs_and_data_uris_removed_short_tokens_kept():
    md = f"Before.\n{B64_RUN}\n![fig](data:image/png;base64,AAAA)\nA hash abc123DEF456 stays.\n"
    out = clean_md(md)
    assert B64_RUN not in out and "data:image" not in out
    assert "Before." in out and "abc123DEF456 stays" in out
    assert clean_md("x " + B64_RUN[:199] + " y") == "x " + B64_RUN[:199] + " y"   # under the 200 bar


def test_html_residue_stripped_text_and_math_kept():
    md = ('<div class="c"><span id="a1">Survey</span> <a href="https://example.org">weights</a></div>\n'
          "<script>var tracking = 1;</script><style>.x{color:red}</style>\n"
          "line one<br><br><br>line two<br/>line three\n"
          "For all a<b, and c>d, the bound holds.\n")
    out = clean_md(md)
    assert "Survey weights" in out and "<span" not in out and "<a " not in out and "</div>" not in out
    assert "tracking" not in out and "color:red" not in out
    assert "line one\nline two\nline three" in out
    assert "a<b, and c>d" in out


def test_running_headers_dropped_after_first_structure_never():
    running = "Journal of Examples (2021) 50:3 https://doi.org/10.5555/jex.2021.0042"
    table = "| Product | Release | Year | Source catalogue entry for the tabulated values |"
    lines = []
    for i in range(8):
        lines += [running.replace("50:3", f"50:{i}"), PARA.format(w=TOPICS[i]), table]
    out = _drop_repeats("\n".join(lines))
    assert out.count("Journal of Examples") == 1            # first kept, digit-normalised repeats dropped
    assert out.count(table) == 8                           # structural markdown is never a candidate
    assert all(PARA.format(w=w) in out for w in TOPICS)


def test_repeat_valve_leaves_a_snapshot_whole():
    # a document that would lose more than 25% to the repeat rule is not a paper with
    # running headers; the rule must not mangle it
    line = "<span>word run that repeats in a saved viewer page again</span>"
    md = "\n".join([line] * 20 + ["A single line of real prose that is kept."])
    assert _drop_repeats(md) == md


def test_header_fields_truncation_and_zotero_link():
    meta = {"TEST0001": {"title": " A Title ", "authors": [f"Author{i} A." for i in range(15)],
                         "year": "2016", "publication": "J. Appl. Ecol.", "doi": "10.1000/x",
                         "tags": [f"t{i}" for i in range(25)], "abstract": "Line one\n  line   two"}}
    h = header(meta, "TEST0001")
    assert h.startswith("# A Title\n\n")
    assert "Author11 A.; et al." in h and "Author12" not in h
    assert "**Year:** 2016  **Published in:** J. Appl. Ecol.  **DOI:** 10.1000/x" in h
    assert "t19" in h and "t20" not in h
    assert "**Zotero:** zotero://select/library/items/TEST0001" in h
    assert "> **Abstract.** Line one line two" in h and h.endswith("\n---\n")
    bare = header({}, "TEST0002")
    assert bare.startswith("# (untitled)\n") and "items/TEST0002" in bare and "**Year:**" not in bare


def test_slug():
    assert slug("Moreau et al. 2016 – surveys (v05)") == "Moreau_et_al._2016_surveys_v05"
    assert slug("") == "untitled" and slug("///") == "untitled"
    assert len(slug("x" * 200)) == 80


def test_pick_primary_prefers_named_file_then_kind_then_size(tmp_path):
    def f(name, size):
        (tmp_path / name).write_bytes(b"x" * size)
    f("snapshot.html", 900)
    f("small.pdf", 10)
    f("big.pdf", 500)
    f("book.epub", 5000)
    f(".zotero-ft-cache", 50)
    f("image.png", 50)
    assert pick_primary(str(tmp_path), None) == (str(tmp_path / "big.pdf"), "pdf")
    assert pick_primary(str(tmp_path), "snapshot.html") == (str(tmp_path / "snapshot.html"), "html")
    assert pick_primary(str(tmp_path), "gone.pdf") == (str(tmp_path / "big.pdf"), "pdf")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".zotero-ft-cache").write_text("cached text")
    assert pick_primary(str(empty), None) == (None, None)
