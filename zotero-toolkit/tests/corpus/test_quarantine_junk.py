"""Tests for corpus/quarantine_junk.py: the ratio rule, the denylist, and never deleting."""

import json
import os
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_testlib import run_script
from quarantine_junk import junk_ratio, load_denylist, pua_ratio

PROSE = (
    "The 2014 trial compared three scheduling heuristics on one benchmark suite, and we "
    "report the throughput gain for every workload.\n"
) * 20
B64 = (
    string.ascii_letters + string.digits + "+/"
) * 50  # one unbroken 3.2 KB base64-alphabet run
JUNK = (
    '<div class="nav"><a href="/">Home</a></div>\n![logo](data:image/png;base64,'
    + B64
    + ")\n"
    + B64
    + "\nNews\n"
)


def test_junk_ratio():
    assert junk_ratio("") == 0.0
    assert junk_ratio(PROSE) == 0.0
    assert junk_ratio(JUNK) >= 0.90
    assert junk_ratio("<span>" * 10 + "word") > 0.9


def test_load_denylist_merges_files_skips_missing_and_comments(tmp_path):
    a, b = tmp_path / "junk.txt", tmp_path / "dupes.txt"
    a.write_text(
        "# webpage snapshots\nTEST0001\nTEST0002   # trailing comment\n\n",
        encoding="utf-8",
    )
    b.write_text("TEST0003__old_copy.md superseded by TEST0004\n", encoding="utf-8")
    got = load_denylist(f"{a}, {tmp_path / 'missing.txt'} ,{b}")
    assert got == {"TEST0001", "TEST0002", "TEST0003__old_copy.md"}
    assert load_denylist("") == set() and load_denylist(None) == set()


def _corpus(tmp_path):
    corpus = tmp_path / "rag_corpus"
    (corpus / "__parsed__").mkdir(parents=True)
    files = {
        "TEST0001__real_paper.md": PROSE,
        "TEST0002__saved_page.md": JUNK,  # ratio rule
        "TEST0003__abstract.md": PROSE,  # denylisted by filename
        "TEST0003__recovered.md": PROSE,  # its sibling must stay
        "TEST0004__anything.md": PROSE,  # denylisted by key
        "TEST0005__tiny.md": B64[:300],  # junk, but under --min-chars
    }
    for name, text in files.items():
        (corpus / name).write_text(text, encoding="utf-8")
    (corpus / "__parsed__" / "TEST0006__ingested.md").write_text(JUNK, encoding="utf-8")
    deny = tmp_path / "deny.txt"
    deny.write_text("TEST0003__abstract.md\nTEST0004\n", encoding="utf-8")
    return corpus, deny, files


def test_moves_junk_and_denylisted_files_never_deletes(tmp_path):
    corpus, deny, files = _corpus(tmp_path)
    r = run_script("quarantine_junk.py", "--corpus", corpus, "--denylist", deny)
    assert r.returncode == 0, r.stderr
    q = tmp_path / "rag_corpus_junk_quarantine"
    moved = {
        "TEST0002__saved_page.md",
        "TEST0003__abstract.md",
        "TEST0004__anything.md",
    }
    assert set(os.listdir(q)) == moved | {"quarantined.jsonl"}
    for name in moved:  # moved intact, not rewritten
        assert (q / name).read_text(encoding="utf-8") == files[name]
    assert sorted(os.listdir(corpus)) == sorted(
        [
            "__parsed__",
            "TEST0001__real_paper.md",
            "TEST0003__recovered.md",
            "TEST0005__tiny.md",
        ]
    )
    assert os.listdir(corpus / "__parsed__") == [
        "TEST0006__ingested.md"
    ]  # ingested: never touched
    log = [
        json.loads(line) for line in (q / "quarantined.jsonl").read_text().splitlines()
    ]
    assert {(x["file"], x["reason"]) for x in log} == {
        ("TEST0002__saved_page.md", "ratio"),
        ("TEST0003__abstract.md", "denylist"),
        ("TEST0004__anything.md", "denylist"),
    }


def test_dry_run_moves_nothing(tmp_path):
    corpus, deny, files = _corpus(tmp_path)
    r = run_script(
        "quarantine_junk.py", "--corpus", corpus, "--denylist", deny, "--dry-run"
    )
    assert r.returncode == 0 and "would quarantine 3 file(s)" in r.stdout
    assert not (tmp_path / "rag_corpus_junk_quarantine").exists()
    assert set(files) <= set(os.listdir(corpus))


def test_default_denylist_is_empty_and_missing_corpus_is_not_an_error(tmp_path):
    corpus, _, _ = _corpus(tmp_path)
    r = run_script("quarantine_junk.py", "--corpus", corpus)
    assert r.returncode == 0
    assert sorted(
        x
        for x in os.listdir(tmp_path / "rag_corpus_junk_quarantine")
        if x.endswith(".md")
    ) == ["TEST0002__saved_page.md"]  # ratio rule only
    r = run_script("quarantine_junk.py", "--corpus", tmp_path / "nope")
    assert r.returncode == 0 and "nothing to do" in r.stdout


# A scanned PDF whose fonts map every glyph into the Private Use Area: eight-character
# "words" of U+F020.. codes, the shape pymupdf4llm produces from such a text layer.
GLYPHS = "".join(chr(0xF020 + (i % 90)) for i in range(5000))
GARBAGE = (
    "# 978-0-000-00000-0\n\n"
    + " ".join(GLYPHS[i : i + 8] for i in range(0, 5000, 8))
    + "\n###### \n"
)
# Genuine text whose symbol font puts a few Greek letters into the PUA (about 4 %).
GREEK_HEAVY = PROSE + " ".join(chr(0xF061 + (i % 20)) for i in range(100))


def test_pua_ratio():
    assert pua_ratio("") == 0.0
    assert pua_ratio(PROSE) == 0.0
    assert pua_ratio(GARBAGE) > 0.9
    assert pua_ratio(GREEK_HEAVY) < 0.05
    assert pua_ratio(GLYPHS[:300]) == 0.0  # under the 500 non-space floor
    assert pua_ratio("".join(chr(0xF0000 + i) for i in range(600))) == 1.0  # plane 15


def test_glyph_code_files_are_quarantined_and_greek_is_kept(tmp_path):
    corpus = tmp_path / "rag_corpus"
    corpus.mkdir()
    (corpus / "TEST0010__garbled_scan.md").write_text(GARBAGE, encoding="utf-8")
    (corpus / "TEST0011__greek_paper.md").write_text(GREEK_HEAVY, encoding="utf-8")
    (corpus / "TEST0012__saved_page.md").write_text(JUNK, encoding="utf-8")
    r = run_script("quarantine_junk.py", "--corpus", corpus)
    assert r.returncode == 0, r.stderr
    assert "[glyphs  ]" in r.stdout and "PUA glyph codes" in r.stdout
    assert sorted(os.listdir(corpus)) == ["TEST0011__greek_paper.md"]
    q = tmp_path / "rag_corpus_junk_quarantine"
    log = {
        json.loads(line)["file"]: json.loads(line)
        for line in (q / "quarantined.jsonl").read_text().splitlines()
    }
    assert log["TEST0010__garbled_scan.md"]["reason"] == "glyphs"
    assert log["TEST0010__garbled_scan.md"]["pua_ratio"] > 0.9
    assert log["TEST0012__saved_page.md"]["reason"] == "ratio"
    assert log["TEST0012__saved_page.md"]["pua_ratio"] == 0.0


def test_min_pua_raises_the_bar(tmp_path):
    corpus = tmp_path / "rag_corpus"
    corpus.mkdir()
    (corpus / "TEST0010__garbled_scan.md").write_text(GARBAGE, encoding="utf-8")
    r = run_script("quarantine_junk.py", "--corpus", corpus, "--min-pua", "0.999")
    assert r.returncode == 0 and "0 junk files" in r.stdout
    assert os.listdir(corpus) == ["TEST0010__garbled_scan.md"]
