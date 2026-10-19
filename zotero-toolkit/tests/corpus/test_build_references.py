"""Tests for corpus/build_references.py: OpenAlex reference lists kept to library DOIs.

OpenAlex is never called: fetch_works / fetch_one are replaced by fakes that answer
from an invented catalogue. DOIs and work ids are invented.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_testlib  # noqa: F401  (puts corpus/ on sys.path)
import build_references as br

# DOI -> (work id, referenced work ids); W9 is not a library paper
CATALOGUE = {
    "10.1/a": ("W1", ["W2", "W3", "W9", "W2"]),
    "10.1/b": ("W2", ["W3"]),
    "10.1/c": ("W3", []),
    "10.1/odd,doi": ("W4", ["W1"]),
}

METADATA = {
    "KEY00001": {"doi": "10.1/A"},
    "KEY00011": {"doi": "https://doi.org/10.1/a"},  # a second attachment of one paper
    "KEY00002": {"doi": "doi:10.1/b"},
    "KEY00003": {"doi": "10.1/c"},
    "KEY00004": {"doi": "10.1/odd,doi"},
    "KEY00005": {"doi": "10.1/unknown"},
    "KEY00006": {"doi": ""},
    "KEY00007": {"doi": "not a doi"},
}


@pytest.fixture
def fake_openalex(monkeypatch):
    calls = []

    def fetch_works(dois, mailto, api_key):
        calls.append(("batch", list(dois)))
        assert all("|" not in d and "," not in d for d in dois)
        return {
            d: {"id": CATALOGUE[d][0], "refs": CATALOGUE[d][1]}
            for d in dois
            if d in CATALOGUE
        }

    def fetch_one(doi, mailto, api_key):
        calls.append(("one", doi))
        work = CATALOGUE.get(doi)
        return {"id": work[0], "refs": work[1]} if work else None

    monkeypatch.setattr(br, "fetch_works", fetch_works)
    monkeypatch.setattr(br, "fetch_one", fetch_one)
    monkeypatch.setattr(br, "PAUSE", 0)
    return calls


def test_norm_doi():
    assert br.norm_doi(" https://dx.doi.org/10.1/AbC ") == "10.1/abc"
    assert br.norm_doi("doi: 10.1/x") == "10.1/x"
    assert br.norm_doi("10.1") == ""
    assert br.norm_doi("") == ""


def test_library_dois_are_distinct_and_normalised():
    assert br.library_dois(METADATA) == [
        "10.1/a",
        "10.1/b",
        "10.1/c",
        "10.1/odd,doi",
        "10.1/unknown",
    ]


def test_references_are_kept_to_library_papers(fake_openalex):
    dois = br.library_dois(METADATA)
    cache = {}
    assert br.harvest(dois, cache, "", "", refresh=False) == 0
    refs = br.build(dois, cache)
    # W9 is not in the library, the repeated W2 counts once in `cited`
    assert refs["10.1/a"] == {"n_refs": 4, "cited": ["10.1/b", "10.1/c"]}
    assert refs["10.1/b"] == {"n_refs": 1, "cited": ["10.1/c"]}
    assert refs["10.1/c"] == {"n_refs": 0, "cited": []}
    assert refs["10.1/odd,doi"] == {"n_refs": 1, "cited": ["10.1/a"]}
    assert "10.1/unknown" not in refs
    # the DOI with a comma was asked about alone
    assert ("one", "10.1/odd,doi") in fake_openalex


def test_a_rerun_asks_only_about_unseen_dois(fake_openalex):
    dois = br.library_dois(METADATA)
    cache = {}
    br.harvest(dois, cache, "", "", refresh=False)
    fake_openalex.clear()
    br.harvest(dois, cache, "", "", refresh=False)
    # only the DOI OpenAlex did not know is asked about again
    assert fake_openalex == [("batch", ["10.1/unknown"])]
    fake_openalex.clear()
    br.harvest(dois, cache, "", "", refresh=True)
    assert len([c for c in fake_openalex if c[0] == "batch"]) == 1
    assert ("one", "10.1/odd,doi") in fake_openalex


def test_a_failed_batch_is_counted_and_the_rest_kept(monkeypatch, fake_openalex):
    def failing(dois, mailto, api_key):
        raise OSError("connection reset")

    monkeypatch.setattr(br, "fetch_works", failing)
    cache = {}
    assert br.harvest(br.library_dois(METADATA), cache, "", "", refresh=False) == 1
    assert list(cache) == ["10.1/odd,doi"]


def test_main_writes_output_and_cache(tmp_path, monkeypatch, fake_openalex, capsys):
    meta = tmp_path / "zotero_metadata.json"
    meta.write_text(json.dumps(METADATA), encoding="utf-8")
    out = tmp_path / "zotero_references.json"
    monkeypatch.setattr(
        sys, "argv", ["build_references.py", "--metadata", str(meta), "--out", str(out)]
    )
    with pytest.raises(SystemExit) as done:
        br.main()
    assert done.value.code == 0
    refs = json.loads(out.read_text(encoding="utf-8"))
    assert refs["10.1/a"]["cited"] == ["10.1/b", "10.1/c"]
    cache = json.loads(
        (tmp_path / "openalex_references_cache.json").read_text(encoding="utf-8")
    )
    assert cache["10.1/b"] == {"id": "W2", "refs": ["W3"]}
    assert (
        "3 cite library papers (4 references), 0 failed requests"
        in capsys.readouterr().out
    )
