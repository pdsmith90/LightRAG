"""Offline tests for lightrag/zotero_add.py: Zotero, CrossRef and WebDAV are faked.

The fake world below answers the three hosts through ``httpx.MockTransport`` and
keeps the state a real library would, so each test asserts what was created,
what the WebDAV store holds and what was rolled back.
"""

import io
import json
import os
import re
import sqlite3
import sys
import zipfile

import httpx
import pytest

from lightrag import zotero_add as za

pytestmark = pytest.mark.offline

TITLE = "Orbital debris mitigation through adaptive collision avoidance planning"
DOI = "10.1234/orbit.2024.001"


def make_pdf(lines: list[str], title: str | None = None) -> bytes:
    """A one-page PDF whose text layer is ``lines`` (pypdf can read it)."""
    return make_pdf_pages([lines], title)


def make_pdf_pages(pages: list[list[str]], title: str | None = None) -> bytes:
    """A PDF whose page text layers are ``pages`` (pypdf can read them)."""
    font = 3 + 2 * len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids.encode(), len(pages)),
    ]
    for i, lines in enumerate(pages):
        text = "BT /F1 11 Tf 40 760 Td 14 TL " + " ".join(
            f"({ln.replace('(', '[').replace(')', ']')}) Tj T*" for ln in lines
        )
        stream = (text + " ET").encode("latin-1")
        objs.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (4 + 2 * i, font)
        )
        objs.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if title:
        objs.append(b"<< /Title (%s) >>" % title.encode("latin-1"))
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    info = b" /Info %d 0 R" % len(objs) if title else b""
    out.write(
        b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objs) + 1, info, xref)
    )
    return out.getvalue()


def crossref_record(**over):
    rec = {
        "DOI": DOI,
        "title": [TITLE],
        "type": "journal-article",
        "author": [
            {"family": "Lindqvist", "given": "Anna"},
            {"family": "Okafor", "given": "Ben"},
        ],
        "issued": {"date-parts": [[2024, 5]]},
        "container-title": ["Journal of Space Safety"],
        "volume": "12",
        "issue": "3",
        "page": "101-120",
        "publisher": "Acme Press",
        "ISSN": ["1234-5678"],
        "URL": "https://doi.org/" + DOI,
        "abstract": "<jats:p>An abstract.</jats:p>",
    }
    rec.update(over)
    return rec


# the ``select`` values api.crossref.org accepts on /works (its 400 lists them)
CROSSREF_SELECTS = set(
    "abstract URL resource member posted score created degree update-policy "
    "short-title license ISSN container-title issued update-to issue prefix "
    "approved indexed article-number clinical-trial-number accepted author "
    "group-title DOI is-referenced-by-count updated-by event chair "
    "standards-body original-title funder translator published archive "
    "published-print alternative-id subject subtitle published-online "
    "publisher-location content-domain reference title link type publisher "
    "volume references-count ISBN issn-type assertion deposited page "
    "contributor content-created short-container-title relation editor".split()
)

TEMPLATES = {
    "journalArticle": {
        "itemType": "journalArticle",
        "title": "",
        "creators": [{"creatorType": "author", "firstName": "", "lastName": ""}],
        "abstractNote": "",
        "publicationTitle": "",
        "volume": "",
        "issue": "",
        "pages": "",
        "date": "",
        "language": "",
        "DOI": "",
        "ISSN": "",
        "url": "",
        "extra": "",
        "tags": [],
        "collections": [],
        "relations": {},
    },
    "book": {
        "itemType": "book",
        "title": "",
        "creators": [],
        "publisher": "",
        "date": "",
        "ISBN": "",
        "extra": "",
        "tags": [],
        "collections": [],
        "relations": {},
    },
    "document": {
        "itemType": "document",
        "title": "",
        "creators": [],
        "extra": "",
        "tags": [],
        "collections": [],
        "relations": {},
    },
}


class World:
    """Fake Zotero API + CrossRef + WebDAV."""

    def __init__(self):
        self.items: dict[str, dict] = {}  # key -> {"version": int, "data": dict}
        self.seq = 0
        self.crossref_works: dict[str, dict] = {}
        self.crossref_search: list[dict] = []
        self.crossref_status = 200
        self.dav: dict[str, bytes] = {}
        self.dav_log: list[str] = []
        self.fail_put: str | None = None  # suffix of the PUT that answers 500
        self.fail_patch = False
        self.fail_delete = False
        self.calls: list[tuple[str, str]] = []

    # ---- helpers
    def add_item(self, data: dict) -> str:
        self.seq += 1
        key = f"K{self.seq:07d}"
        data = dict(data, key=key)
        self.items[key] = {"key": key, "version": 10 + self.seq, "data": data}
        return key

    def children(self, parent: str) -> list[dict]:
        return [i for i in self.items.values() if i["data"].get("parentItem") == parent]

    def top(self) -> list[dict]:
        return [i for i in self.items.values() if not i["data"].get("parentItem")]

    # ---- transport
    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        self.calls.append((request.method, f"{host}{path}"))
        if host == "api.zotero.org":
            return self.zotero(request, path)
        if host == "api.crossref.org":
            return self.crossref(request, path)
        return self.webdav(request, path)

    def zotero(self, request, path):
        m = request.method
        if path == "/items/new":
            t = TEMPLATES.get(request.url.params["itemType"])
            if t is None:
                return httpx.Response(400, text="Invalid item type")
            return httpx.Response(200, json=json.loads(json.dumps(t)))
        base = "/users/42/items"
        if m == "POST" and path == base:
            assert request.headers["Zotero-API-Key"] == "k"
            assert request.headers["Zotero-Write-Token"]
            data = json.loads(request.content)[0]
            if data.get("title") == "REFUSE":
                return httpx.Response(
                    200, json={"successful": {}, "failed": {"0": {"code": 400}}}
                )
            key = self.add_item(data)
            return httpx.Response(
                200,
                json={
                    "successful": {
                        "0": {"key": key, "version": self.items[key]["version"]}
                    },
                    "failed": {},
                },
            )
        if m == "GET" and path == base + "/top":
            q = request.url.params["q"].lower()
            hits = [i for i in self.top() if q in i["data"].get("title", "").lower()]
            return httpx.Response(200, json=hits)
        mm = re.fullmatch(base + r"/(\w+)(/children)?", path)
        if mm:
            key, kids = mm.group(1), mm.group(2)
            if kids:
                return httpx.Response(200, json=self.children(key))
            item = self.items.get(key)
            if m == "GET":
                return httpx.Response(200, json=item) if item else httpx.Response(404)
            assert request.headers["If-Unmodified-Since-Version"] == str(
                item["version"]
            )
            if m == "PATCH":
                if self.fail_patch:
                    return httpx.Response(500, text="boom")
                item["data"].update(json.loads(request.content))
                item["version"] += 1
                return httpx.Response(204)
            if m == "DELETE":
                if self.fail_delete:
                    return httpx.Response(500, text="boom")
                for child in self.children(key):
                    del self.items[child["key"]]
                del self.items[key]
                return httpx.Response(204)
        return httpx.Response(404, text=f"unhandled {m} {path}")

    def crossref(self, request, path):
        if self.crossref_status != 200:
            return httpx.Response(self.crossref_status)
        if path.startswith("/works/"):
            rec = self.crossref_works.get(path[len("/works/") :].lower())
            if rec is None:
                return httpx.Response(404)
            return httpx.Response(200, json={"message": rec})
        bad = set(request.url.params["select"].split(",")) - CROSSREF_SELECTS
        if bad:  # the live API answers 400 for these
            return httpx.Response(400, text=f"select-not-available {sorted(bad)}")
        return httpx.Response(200, json={"message": {"items": self.crossref_search}})

    def webdav(self, request, path):
        assert request.url.path.startswith("/dav/zotero/")
        assert request.headers["Authorization"].startswith("Basic ")
        name = path.rsplit("/", 1)[-1]
        self.dav_log.append(f"{request.method} {name}")
        if request.method == "MKCOL":
            return httpx.Response(201)
        if request.method == "DELETE":
            self.dav.pop(name, None)
            return httpx.Response(204)
        if request.method == "PUT":
            assert request.headers["Content-Length"] == str(len(request.content))
            if self.fail_put and name.endswith(self.fail_put):
                return httpx.Response(500)
            self.dav[name] = request.content
            return httpx.Response(201)
        return httpx.Response(405)


@pytest.fixture
def world():
    return World()


@pytest.fixture
def cfg(tmp_path):
    return za.ZoteroUploadConfig(
        user_id="42",
        library_type="user",
        api_key="k",
        webdav_url="http://dav.test/dav",
        webdav_user="u",
        webdav_pass="p",
        ledger_path=str(tmp_path / "ledger.json"),
        collection="COLL0001",
    )


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(za, "_sleep", lambda s: None)
    monkeypatch.setattr(za, "_sync_proc", None)
    from lightrag import zotero_citations

    monkeypatch.setattr(zotero_citations, "_load", lambda: {})


def write_pdf(tmp_path, lines, name="paper.pdf", title=None):
    p = tmp_path / name
    p.write_bytes(make_pdf(lines, title))
    return str(p)


def write_pdf_pages(tmp_path, pages, name="paper.pdf", title=None):
    p = tmp_path / name
    p.write_bytes(make_pdf_pages(pages, title))
    return str(p)


def run(cfg, world, path, name="paper.pdf"):
    return za.add_pdf_to_zotero(
        cfg, path, name, transport=httpx.MockTransport(world.handler)
    )


def paper_lines(doi=True):
    lines = [
        TITLE,
        "Anna Lindqvist and Ben Okafor",
        "Journal of Space Safety 12 (2024)",
    ]
    if doi:
        lines.append(f"https://doi.org/{DOI}")
    return lines


# ------------------------------------------------------------------ happy paths


def test_crossref_hit_creates_item_attachment_and_webdav_files(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record()
    path = write_pdf(tmp_path, paper_lines())

    result = run(cfg, world, path)

    assert (result.outcome, result.metadata, result.doi) == ("created", "crossref", DOI)
    parent = world.items[result.item_key]["data"]
    assert parent["itemType"] == "journalArticle"
    assert parent["title"] == TITLE and parent["DOI"] == DOI
    assert parent["publicationTitle"] == "Journal of Space Safety"
    assert parent["date"] == "2024-05" and parent["abstractNote"] == "An abstract."
    assert parent["creators"][0] == {
        "creatorType": "author",
        "lastName": "Lindqvist",
        "firstName": "Anna",
    }
    assert parent["tags"] == [{"tag": "lightrag-upload"}]
    assert parent["collections"] == ["COLL0001"]

    att = world.items[result.attachment_key]["data"]
    assert att["parentItem"] == result.item_key
    assert (
        att["linkMode"] == "imported_file" and att["contentType"] == "application/pdf"
    )
    assert att["md5"] == result.md5 and re.fullmatch(r"\d{13}", str(att["mtime"]))

    key = result.attachment_key
    assert world.dav_log == [
        "MKCOL ",
        f"DELETE {key}.prop",
        f"PUT {key}.zip",
        f"PUT {key}.prop",
    ]
    with zipfile.ZipFile(io.BytesIO(world.dav[key + ".zip"])) as zf:
        assert zf.namelist() == ["paper.pdf"]
        assert zf.read("paper.pdf") == open(path, "rb").read()
    prop = world.dav[key + ".prop"].decode()
    assert prop == (
        f'<properties version="1"><mtime>{att["mtime"]}</mtime><hash>{result.md5}</hash></properties>'
    )
    ledger = json.load(open(cfg.ledger_path))
    assert ledger["by_md5"][result.md5]["item_key"] == result.item_key
    assert ledger["by_doi"][DOI]["attachment_key"] == key


def test_title_search_match_without_a_doi(world, cfg, tmp_path):
    world.crossref_search = [crossref_record()]
    path = write_pdf(tmp_path, paper_lines(doi=False))
    result = run(cfg, world, path)
    assert result.metadata == "crossref" and result.doi == DOI


def test_title_search_hit_not_found_in_the_pdf_is_ignored(world, cfg, tmp_path):
    world.crossref_search = [
        crossref_record(
            title=["A completely different study of glacier hydrology models"]
        )
    ]
    path = write_pdf(tmp_path, paper_lines(doi=False))
    result = run(cfg, world, path)
    assert result.metadata == "stub"


def test_doi_whose_title_is_not_in_the_pdf_is_not_trusted(world, cfg, tmp_path):
    # the PDF cites another paper's DOI; CrossRef knows it, but it is not this paper
    world.crossref_works["10.9999/cited.2001.7"] = crossref_record(
        DOI="10.9999/cited.2001.7", title=["Glacier hydrology models of the Alps"]
    )
    lines = [TITLE, "Anna Lindqvist", "see doi:10.9999/cited.2001.7"]
    result = run(cfg, world, write_pdf(tmp_path, lines))
    assert result.metadata == "stub"
    extra = world.items[result.item_key]["data"]["extra"]
    assert "Candidate DOI (unverified): 10.9999/cited.2001.7" in extra


def test_doi_record_without_a_title_is_not_trusted(world, cfg, tmp_path):
    # a CrossRef stub record with no title cannot be checked against the PDF
    world.crossref_works[DOI] = crossref_record(title=[""])
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert result.metadata == "stub"
    extra = world.items[result.item_key]["data"]["extra"]
    assert f"Candidate DOI (unverified): {DOI}" in extra


def test_doi_whose_title_is_only_in_the_reference_list_is_not_trusted(
    world, cfg, tmp_path
):
    # the title sits on page 2, in the references: it belongs to a cited work
    world.crossref_works[DOI] = crossref_record()
    pages = [
        ["Abstract", "We compare three mitigation strategies."],
        ["2 Method", "The simulation runs for one orbital period."],
        ["References", f"Lindqvist A. {TITLE}. J Space Safety 2024. doi:{DOI}"],
    ]
    result = run(cfg, world, write_pdf_pages(tmp_path, pages))
    assert result.metadata == "stub"
    extra = world.items[result.item_key]["data"]["extra"]
    assert f"Candidate DOI (unverified): {DOI}" in extra


def test_doi_whose_title_is_on_page_0_is_still_trusted(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record()
    pages = [paper_lines(), ["Abstract", "Body text follows."], ["References", ""]]
    result = run(cfg, world, write_pdf_pages(tmp_path, pages))
    assert (result.outcome, result.metadata, result.doi) == ("created", "crossref", DOI)


def test_no_match_gives_a_titled_stub_with_the_attachment(world, cfg, tmp_path):
    path = write_pdf(tmp_path, ["scanned notes", "nothing to look up"])
    result = run(cfg, world, path, name="My notes.pdf")
    stub = world.items[result.item_key]["data"]
    assert stub["itemType"] == "document" and stub["title"] == "Uploaded: My notes.pdf"
    assert {"tag": "unmatched-upload"} in stub["tags"]
    assert result.metadata == "stub" and result.doi == ""
    assert world.items[result.attachment_key]["data"]["filename"] == "My notes.pdf"
    assert f"PUT {result.attachment_key}.zip" in world.dav_log


def test_crossref_outage_still_adds_a_stub_and_says_so(world, cfg, tmp_path):
    world.crossref_status = 503
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert result.metadata == "stub"
    assert any("CrossRef was unreachable" in w for w in result.warnings)
    assert "CrossRef was unreachable" in result.summary


def test_book_type_without_a_doi_field_keeps_the_doi_in_extra(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record(
        type="monograph", ISBN=["9780000000002"]
    )
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    book = world.items[result.item_key]["data"]
    assert book["itemType"] == "book" and "DOI" not in book
    assert f"DOI: {DOI}" in book["extra"] and book["ISBN"] == "9780000000002"


def test_unknown_type_template_falls_back_to_document(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record(type="dataset")
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert world.items[result.item_key]["data"]["itemType"] == "document"


# -------------------------------------------------------------------- duplicates


def test_same_file_twice_is_refused_the_second_time(world, cfg, tmp_path):
    path = write_pdf(tmp_path, ["scanned notes"])
    first = run(cfg, world, path)
    n_items = len(world.items)
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, path, name="renamed.pdf")
    assert exc.value.status_code == 409 and first.item_key in str(exc.value)
    assert len(world.items) == n_items


def test_doi_already_holding_a_pdf_is_refused_and_creates_nothing(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record()
    parent = world.add_item({"itemType": "journalArticle", "title": TITLE, "DOI": DOI})
    world.add_item(
        {
            "itemType": "attachment",
            "parentItem": parent,
            "contentType": "application/pdf",
            "md5": "a" * 32,
        }
    )
    before = len(world.items)
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert exc.value.status_code == 409 and parent in str(exc.value)
    assert len(world.items) == before and world.dav == {}


def test_doi_in_the_local_metadata_is_refused(world, cfg, tmp_path, monkeypatch):
    from lightrag import zotero_citations

    monkeypatch.setattr(
        zotero_citations,
        "_load",
        lambda: {"ATT00001": {"doi": DOI.upper(), "title": "T"}},
    )
    world.crossref_works[DOI] = crossref_record()
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert exc.value.status_code == 409 and "ATT00001" in str(exc.value)
    assert world.items == {}


def test_doi_of_an_item_without_a_pdf_gets_the_pdf_attached(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record()
    parent = world.add_item({"itemType": "journalArticle", "title": TITLE, "DOI": DOI})
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert (result.outcome, result.item_key, result.metadata) == (
        "attached",
        parent,
        "existing",
    )
    assert len(world.top()) == 1  # no second parent
    assert world.items[result.attachment_key]["data"]["parentItem"] == parent


def test_empty_attachment_row_of_that_item_is_filled_not_duplicated(
    world, cfg, tmp_path
):
    world.crossref_works[DOI] = crossref_record()
    parent = world.add_item({"itemType": "journalArticle", "title": TITLE, "DOI": DOI})
    empty = world.add_item(
        {
            "itemType": "attachment",
            "parentItem": parent,
            "contentType": "application/pdf",
        }
    )
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert result.attachment_key == empty and len(world.children(parent)) == 1
    assert world.items[empty]["data"]["md5"] == result.md5


def test_unusable_duplicate_search_does_not_block_and_is_reported(world, cfg, tmp_path):
    world.crossref_works[DOI] = crossref_record()
    original = world.zotero

    def flaky(request, path):
        if path.endswith("/top"):
            return httpx.Response(500)
        return original(request, path)

    world.zotero = flaky
    result = run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert result.outcome == "created"
    assert any("Duplicate check" in w for w in result.warnings)


def test_md5_found_in_zotero_sqlite_is_refused(world, cfg, tmp_path):
    path = write_pdf(tmp_path, ["scanned notes"])
    md5 = za._md5(path)
    db = tmp_path / "zotero.sqlite"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE items(itemID INTEGER PRIMARY KEY, key TEXT);"
        "CREATE TABLE itemAttachments(itemID INT, storageHash TEXT);"
        "CREATE TABLE deletedItems(itemID INT);"
        f"INSERT INTO items VALUES(1,'LIVE0001'),(2,'GONE0002');"
        f"INSERT INTO itemAttachments VALUES(1,'{md5}'),(2,'{md5}');"
    )
    con.commit()
    con.close()
    from dataclasses import replace

    cfg2 = replace(cfg, sqlite_path=str(db))
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg2, world, path)
    assert exc.value.status_code == 409 and "LIVE0001" in str(exc.value)

    con = sqlite3.connect(db)
    con.executescript(
        "INSERT INTO deletedItems VALUES(1);DELETE FROM items WHERE itemID=2;"
    )
    con.commit()
    con.close()
    assert run(cfg2, world, path).outcome == "created"  # the only hit is trashed


# --------------------------------------------------------------------- failures


def test_webdav_failure_rolls_everything_back_and_says_so(world, cfg, tmp_path):
    world.fail_put = ".zip"
    world.crossref_works[DOI] = crossref_record()
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert exc.value.status_code == 502
    assert "WebDAV PUT" in str(exc.value)
    assert "Everything this upload created was removed again" in str(exc.value)
    assert world.items == {} and world.dav == {}
    assert not os.path.exists(cfg.ledger_path)


def test_prop_write_failure_leaves_no_orphan_zip(world, cfg, tmp_path):
    world.fail_put = ".prop"
    with pytest.raises(za.ZoteroUploadError):
        run(cfg, world, write_pdf(tmp_path, ["scanned notes"]))
    assert world.items == {} and world.dav == {}


def test_md5_patch_failure_rolls_back_the_uploaded_files_too(world, cfg, tmp_path):
    world.fail_patch = True
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, ["scanned notes"]))
    assert "setting md5/mtime" in str(exc.value)
    assert world.items == {} and world.dav == {}


def test_failed_rollback_names_what_is_left(world, cfg, tmp_path):
    world.fail_put = ".zip"
    world.fail_delete = True
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, ["scanned notes"]))
    msg = str(exc.value)
    assert "Cleanup was incomplete; remove by hand" in msg
    assert "Zotero item K0000001" in msg and "Zotero item K0000002" in msg


def test_failure_on_an_existing_item_keeps_the_item(world, cfg, tmp_path):
    world.fail_put = ".zip"
    world.crossref_works[DOI] = crossref_record()
    parent = world.add_item({"itemType": "journalArticle", "title": TITLE, "DOI": DOI})
    with pytest.raises(za.ZoteroUploadError):
        run(cfg, world, write_pdf(tmp_path, paper_lines()))
    assert list(world.items) == [parent]


def test_refused_item_creation_is_a_502_with_nothing_left(world, cfg, tmp_path):
    result_world = world
    result_world.crossref_works[DOI] = crossref_record(title=["REFUSE"])
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, write_pdf(tmp_path, ["REFUSE", f"doi:{DOI}"]))
    assert exc.value.status_code == 502 and world.items == {}


def test_not_a_pdf_and_wrong_extension_are_400(world, cfg, tmp_path):
    fake = tmp_path / "x.pdf"
    fake.write_bytes(b"<html>nope</html>")
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, str(fake))
    assert exc.value.status_code == 400
    with pytest.raises(za.ZoteroUploadError) as exc:
        run(cfg, world, str(fake), name="x.docx")
    assert exc.value.status_code == 400
    assert world.calls == []


def test_zotero_unreachable_is_a_502(cfg, tmp_path):
    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(za.ZoteroUploadError) as exc:
        za.add_pdf_to_zotero(
            cfg,
            write_pdf(tmp_path, ["scanned notes"]),
            "a.pdf",
            transport=httpx.MockTransport(boom),
        )
    assert exc.value.status_code == 502


# --------------------------------------------------------------------- retries


def test_retry_after_http_date_falls_back_to_the_default(cfg, monkeypatch):
    slept = []
    monkeypatch.setattr(za, "_sleep", slept.append)
    replies = [
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        httpx.Response(200, json={}),
    ]
    with httpx.Client(transport=httpx.MockTransport(lambda r: replies.pop(0))) as http:
        r = za._Zotero(cfg, http).request("GET", cfg.library + "/items/K0000001")
    assert r.status_code == 200
    assert slept == [2]


def test_retry_after_seconds_are_honoured_and_capped(cfg, monkeypatch):
    slept = []
    monkeypatch.setattr(za, "_sleep", slept.append)
    replies = [
        httpx.Response(503, headers={"Retry-After": "7"}),
        httpx.Response(503, headers={"Retry-After": "600"}),
        httpx.Response(200, json={}),
    ]
    with httpx.Client(transport=httpx.MockTransport(lambda r: replies.pop(0))) as http:
        r = za._Zotero(cfg, http).request("GET", cfg.library + "/items/K0000001")
    assert r.status_code == 200
    assert slept == [7, 20]


# ----------------------------------------------------------------------- config


def test_upload_flag_parsing():
    assert not za.upload_enabled({})
    assert not za.upload_enabled({"ZOTERO_UPLOAD": "false"})
    assert za.upload_enabled({"ZOTERO_UPLOAD": " TRUE "})


def test_load_config_names_every_missing_setting():
    with pytest.raises(za.ZoteroUploadError) as exc:
        za.load_config({"ZOTERO_UPLOAD": "true"})
    assert exc.value.status_code == 503
    for name in ("ZOTERO_USER_ID", "ZOTERO_API_KEY", "ZOTERO_WEBDAV_URL"):
        assert name in str(exc.value)


def test_load_config_from_files_and_env(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    dav = tmp_path / "dav"
    dav.write_text(
        "# c\nWEBDAV_URL=http://h/\nWEBDAV_USER=u\nWEBDAV_PASS=p\nWEBDAV_ENABLED=1\n"
    )
    cfg = za.load_config(
        {
            "ZOTERO_USER_ID": "7",
            "ZOTERO_API_KEY_FILE": str(key),
            "ZOTERO_WEBDAV_FILE": str(dav),
            "ZOTERO_WEBDAV_USER": "override",
            "ZOTERO_SYNC_COMMAND": "true",
        }
    )
    assert cfg.api_key == "secret-key" and cfg.webdav_url == "http://h/"
    assert (cfg.webdav_user, cfg.webdav_pass, cfg.library_type) == (
        "override",
        "p",
        "user",
    )
    assert cfg.library == "https://api.zotero.org/users/7"
    assert "secret-key" not in repr(cfg) and "'p'" not in repr(cfg)


def test_webdav_disabled_in_the_credentials_file_is_refused(tmp_path):
    dav = tmp_path / "dav"
    dav.write_text(
        "WEBDAV_URL=http://h/\nWEBDAV_USER=u\nWEBDAV_PASS=p\nWEBDAV_ENABLED=0\n"
    )
    with pytest.raises(za.ZoteroUploadError) as exc:
        za.load_config(
            {
                "ZOTERO_USER_ID": "7",
                "ZOTERO_API_KEY": "k",
                "ZOTERO_WEBDAV_FILE": str(dav),
            }
        )
    assert "WEBDAV_ENABLED" in str(exc.value)


def test_unreadable_key_file_is_a_config_error(tmp_path):
    with pytest.raises(za.ZoteroUploadError) as exc:
        za.load_config(
            {
                "ZOTERO_USER_ID": "7",
                "ZOTERO_API_KEY_FILE": str(tmp_path / "missing"),
                "ZOTERO_WEBDAV_URL": "h",
                "ZOTERO_WEBDAV_USER": "u",
                "ZOTERO_WEBDAV_PASS": "p",
            }
        )
    assert "ZOTERO_API_KEY_FILE unreadable" in str(exc.value)


# ------------------------------------------------------------------------- DOIs


def test_normalize_doi():
    assert za.normalize_doi("https://doi.org/10.1000/ABC.1).") == "10.1000/abc.1"
    assert (
        za.normalize_doi("doi: 10.1016/S0021-9991(99)00001-2")
        == "10.1016/s0021-9991(99)00001-2"
    )
    assert za.normalize_doi("no doi here") == ""
    assert za.normalize_doi(None) == ""


def test_doi_candidates_rank_metadata_then_labelled_then_frequent():
    info = za.PdfInfo(
        pages=["10.1111/cited.1 10.1111/cited.1 10.1111/cited.1 DOI: 10.2222/own.2"],
        meta_dois=["10.3333/meta.3"],
    )
    assert za.doi_candidates(info) == [
        "10.3333/meta.3",
        "10.2222/own.2",
        "10.1111/cited.1",
    ]


# ------------------------------------------------------------------------- sync


def _cfg_with(cfg, **kw):
    from dataclasses import replace

    return replace(cfg, **kw)


def test_sync_not_configured_says_so(cfg):
    out = za.trigger_sync(cfg)
    assert (out.triggered, out.state) == (False, "not_configured")
    assert "next scheduled" in out.message


def test_sync_started_then_single_flight(cfg, tmp_path):
    marker = tmp_path / "ran"
    script = tmp_path / "sync.py"
    script.write_text(
        f"import sys, time\nopen({str(marker)!r}, 'a').write('x')\ntime.sleep(2)\n"
    )
    c = _cfg_with(cfg, sync_command=f"{sys.executable} {script}")
    first = za.trigger_sync(c)
    second = za.trigger_sync(c)
    assert (first.triggered, first.state) == (True, "started")
    assert (second.triggered, second.state) == (False, "already_running")
    za._sync_proc.wait(timeout=15)
    assert marker.read_text() == "x"  # launched exactly once
    third = za.trigger_sync(c)  # the first run is over, so a new one may start
    assert third.state == "started"
    za._sync_proc.wait(timeout=15)
    assert marker.read_text() == "xx"


def test_sync_is_not_started_while_the_build_lock_is_held(cfg, tmp_path):
    import fcntl

    lock = tmp_path / "build.lock"
    marker = tmp_path / "ran"
    c = _cfg_with(
        cfg,
        sync_command=f"{sys.executable} -c \"open({str(marker)!r}, 'w')\"",
        sync_lock_file=str(lock),
    )
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        held = za.trigger_sync(c)
    finally:
        os.close(fd)
    assert (held.triggered, held.state) == (False, "already_running")
    assert not marker.exists()
    free = za.trigger_sync(c)
    assert free.state == "started"
    za._sync_proc.wait(timeout=15)
    assert marker.exists()


def test_sync_command_that_cannot_start_is_reported(cfg, tmp_path):
    out = za.trigger_sync(_cfg_with(cfg, sync_command=str(tmp_path / "no-such-binary")))
    assert (out.triggered, out.state) == (False, "failed")
    assert "could not be started" in out.message


def test_sync_output_goes_to_the_configured_log(cfg, tmp_path):
    log = tmp_path / "sync.log"
    c = _cfg_with(
        cfg,
        sync_command=f"{sys.executable} -c \"print('hello')\"",
        sync_log=str(log),
    )
    assert za.trigger_sync(c).state == "started"
    za._sync_proc.wait(timeout=15)
    assert log.read_text().strip() == "hello"
