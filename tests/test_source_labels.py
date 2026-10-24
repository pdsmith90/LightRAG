"""The WebUI's source display: a corpus file path becomes the Zotero item's
"Authors · Title · Year · DOI" label (lightrag/zotero_citations.py and
POST /graph/sources/labels). Fixtures are invented.
"""

import importlib
import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import lightrag.zotero_citations as zc

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

META = {
    "AAAA0001": {
        "title": "Mascon <i>leakage</i> over Antarctica.",
        "authors": ["Okafor Chidi", "Lindqvist Maja"],
        "year": "2019",
        "doi": "10.1/P",
    },
    "BBBB0002": {
        "title": "Many hands",
        "authors": ["A One", "B Two", "C Three", "D Four"],
        "year": "2004",
    },
    "CCCC0003": {"title": "Untitled-year report", "authors": []},
    "DDDD0004": {
        "title": "Structured names",
        "authors": ["Efron Bradley", "Tibshirani Robert"],
        "creators": [["Efron", "Bradley"], ["Tibshirani", "Robert"]],
        "year": "1993",
    },
    "EEEE0005": {
        "title": "Four creators",
        "authors": ["A One", "B Two", "C Three", "D Four"],
        "creators": [["One", "A."], ["Two", "B."], ["Three", "C."], ["Four", "D."]],
    },
    "FFFF0006": {
        "title": "Family only",
        "authors": ["WHO", "van der Berg A. B."],
        "creators": [["WHO", ""], ["van der Berg", "A. B."], ["", "Prince"]],
    },
}

_HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
def library(tmp_path):
    meta_path = tmp_path / "zotero_metadata.json"
    meta_path.write_text(json.dumps(META), encoding="utf-8")
    saved = zc.METADATA_PATH
    zc.METADATA_PATH = str(meta_path)
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False
    yield
    zc.METADATA_PATH = saved
    zc._meta, zc._mtime, zc._next_stat, zc._warned = {}, None, 0.0, False


@pytest.fixture
def client(library):
    app = FastAPI()
    app.include_router(_graph_routes.create_graph_routes(object(), api_key="test-key"))
    return TestClient(app)


def test_label_joins_authors_title_year_doi(library):
    got = zc.source_label("rag_corpus/AAAA0001__Mascon_leakage.md")
    assert got["label"] == (
        "Okafor Chidi, Lindqvist Maja · Mascon leakage over Antarctica · 2019 · 10.1/P"
    )
    assert got["doi"] == "10.1/P" and got["year"] == "2019"


def test_label_omits_missing_parts_and_shortens_long_author_lists(library):
    assert (
        zc.source_label("BBBB0002__Many_hands.md")["label"]
        == "A One, B Two, C Three et al. · Many hands · 2004"
    )
    assert zc.source_label("CCCC0003__x.md")["label"] == "Untitled-year report"


@pytest.mark.parametrize(
    "path", ["unknown_source", "manual_creation", "notes.md", "ZZZZ9999__gone.md", ""]
)
def test_no_entry_means_no_label(library, path):
    got = zc.source_label(path)
    assert got["label"] is None and got["file_path"] == path


def test_endpoint_returns_a_label_per_path_in_order(client):
    paths = ["rag_corpus/AAAA0001__m.md", "unknown_source"]
    r = client.post(
        "/graph/sources/labels", json={"file_paths": paths}, headers=_HEADERS
    )
    assert r.status_code == 200
    labels = r.json()["labels"]
    assert [x["file_path"] for x in labels] == paths
    assert labels[0]["label"].startswith("Okafor Chidi")
    assert labels[1]["label"] is None


def test_endpoint_requires_auth_and_caps_list_size(client):
    assert client.post(
        "/graph/sources/labels", json={"file_paths": ["a"]}
    ).status_code in (401, 403)
    too_many = {"file_paths": ["x"] * 501}
    r = client.post("/graph/sources/labels", json=too_many, headers=_HEADERS)
    assert r.status_code == 422


def test_creators_render_family_comma_given_joined_by_semicolons(library):
    got = zc.source_label("DDDD0004__x.md")["label"]
    assert got == "Efron, Bradley; Tibshirani, Robert · Structured names · 1993"


def test_four_creators_keep_three_and_et_al(library):
    got = zc.source_label("EEEE0005__x.md")["label"]
    assert got == "One, A.; Two, B.; Three, C. et al. · Four creators"


def test_family_only_and_given_only_creators_and_initial_ending(library):
    assert zc._authors(META["FFFF0006"]) == "WHO; van der Berg, A. B.; Prince"
    assert zc.source_label("FFFF0006__x.md")["label"] == (
        "WHO; van der Berg, A. B.; Prince · Family only"
    )
    assert zc._authors({"authors": ["Doe J."], "creators": [["", ""]]}) == "Doe J."


def test_entry_without_creators_keeps_comma_joined_authors(library):
    assert zc._authors(META["AAAA0001"]) == "Okafor Chidi, Lindqvist Maja"
