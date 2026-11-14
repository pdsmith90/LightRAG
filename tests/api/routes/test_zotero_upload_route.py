"""``POST /documents/upload`` under ``ZOTERO_UPLOAD``, and the flag-off contract.

The Zotero work itself is covered by ``tests/test_zotero_add.py``; here the
module's two entry points are replaced so the tests pin the HTTP layer: the
response shape, the status mapping, the size limit, that nothing reaches the
input directory, and that with the flag off the upstream upload runs unchanged.
"""

import importlib
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import zotero_add  # noqa: E402

create_document_routes = _document_routes.create_document_routes
DocumentManager = _document_routes.DocumentManager
HTTPException = _document_routes.HTTPException

pytestmark = pytest.mark.offline

_PDF = b"%PDF-1.4\n% fake body\n"


def _endpoint(tmp_path, rag=None):
    router = create_document_routes(
        rag or SimpleNamespace(), DocumentManager(str(tmp_path))
    )
    return [
        r.endpoint
        for r in router.routes
        if getattr(r, "name", "") == "upload_to_input_dir"
    ][-1]


def _upload(name="paper.pdf", body=_PDF):
    return _document_routes.UploadFile(filename=name, file=BytesIO(body))


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=None)
    )
    for var in ("ZOTERO_UPLOAD", "ZOTERO_SYNC_COMMAND"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def zotero_on(monkeypatch):
    monkeypatch.setenv("ZOTERO_UPLOAD", "true")
    monkeypatch.setenv("ZOTERO_USER_ID", "42")
    monkeypatch.setenv("ZOTERO_API_KEY", "k")
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "http://dav.test/")
    monkeypatch.setenv("ZOTERO_WEBDAV_USER", "u")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASS", "p")
    calls = SimpleNamespace(added=[], synced=[], seen_bytes=[])

    def fake_add(cfg, path, filename, **kw):
        calls.added.append(filename)
        with open(path, "rb") as fh:
            calls.seen_bytes.append(fh.read())
        return zotero_add.UploadResult(
            outcome="created",
            item_key="ITEM0001",
            attachment_key="ATT00001",
            title="A title",
            doi="10.1/x",
            metadata="crossref",
            md5="0" * 32,
            filename=filename,
        )

    def fake_sync(cfg):
        calls.synced.append(cfg.user_id)
        return zotero_add.SyncOutcome(True, "started", "A sync was started.")

    monkeypatch.setattr(zotero_add, "add_pdf_to_zotero", fake_add)
    monkeypatch.setattr(zotero_add, "trigger_sync", fake_sync)
    return calls


async def test_upload_adds_to_zotero_and_reports_the_sync(tmp_path, zotero_on):
    resp = await _endpoint(tmp_path)(set(), _upload("../sub/paper.pdf"))

    assert resp.status_code == 200
    import json

    body = json.loads(resp.body)
    assert body["status"] == "success"
    assert body["item_key"] == "ITEM0001" and body["attachment_key"] == "ATT00001"
    assert body["metadata"] == "crossref" and body["doi"] == "10.1/x"
    assert body["sync_triggered"] is True and body["sync_state"] == "started"
    assert "added to the Zotero library" in body["message"]
    assert body["message"].endswith("A sync was started.")
    assert body["sync_message"] == "A sync was started."
    assert "track_id" not in body
    assert zotero_on.added == ["paper.pdf"]  # directory part dropped
    assert zotero_on.seen_bytes == [_PDF]
    assert list(tmp_path.iterdir()) == []  # nothing indexed or stored locally


async def test_response_is_honest_when_no_sync_was_started(
    tmp_path, zotero_on, monkeypatch
):
    monkeypatch.setattr(
        zotero_add,
        "trigger_sync",
        lambda cfg: zotero_add.SyncOutcome(
            False, "not_configured", "No sync was triggered."
        ),
    )
    import json

    body = json.loads((await _endpoint(tmp_path)(set(), _upload())).body)
    assert body["sync_triggered"] is False and body["sync_state"] == "not_configured"
    assert body["message"].endswith("No sync was triggered.")


async def test_failed_upload_does_not_trigger_a_sync(tmp_path, zotero_on, monkeypatch):
    def refuse(cfg, path, filename, **kw):
        raise zotero_add.ZoteroUploadError(
            "Already in the Zotero library: item X.", 409
        )

    monkeypatch.setattr(zotero_add, "add_pdf_to_zotero", refuse)
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), _upload())
    assert (
        exc.value.status_code == 409
        and "Already in the Zotero library" in exc.value.detail
    )
    assert zotero_on.synced == []


async def test_zotero_failure_is_a_502_with_its_account(
    tmp_path, zotero_on, monkeypatch
):
    def broken(cfg, path, filename, **kw):
        raise zotero_add.ZoteroUploadError(
            "Adding to Zotero failed: WebDAV PUT -> 500."
        )

    monkeypatch.setattr(zotero_add, "add_pdf_to_zotero", broken)
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), _upload())
    assert exc.value.status_code == 502 and "WebDAV PUT" in exc.value.detail


async def test_only_pdfs_are_accepted(tmp_path, zotero_on):
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), _upload("notes.docx"))
    assert exc.value.status_code == 400 and "PDF" in exc.value.detail
    assert zotero_on.added == []


async def test_max_upload_size_is_enforced_while_streaming(
    tmp_path, zotero_on, monkeypatch
):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=10)
    )
    upload = _upload(body=_PDF)
    upload.size = None  # force the streaming check instead of the declared size
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), upload)
    assert exc.value.status_code == 413
    assert zotero_on.added == []


async def test_max_upload_size_uses_the_declared_size_first(
    tmp_path, zotero_on, monkeypatch
):
    monkeypatch.setattr(
        _document_routes, "global_args", SimpleNamespace(max_upload_size=10)
    )
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), _upload())
    assert exc.value.status_code == 413


async def test_enabled_but_unconfigured_is_a_503_not_a_silent_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZOTERO_UPLOAD", "true")
    for var in ("ZOTERO_USER_ID", "ZOTERO_API_KEY", "ZOTERO_API_KEY_FILE"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path)(set(), _upload())
    assert exc.value.status_code == 503 and "ZOTERO_USER_ID" in exc.value.detail
    assert list(tmp_path.iterdir()) == []


async def test_flag_off_runs_the_upstream_upload_unchanged(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the Zotero path must not run with the flag off")

    monkeypatch.setattr(zotero_add, "add_pdf_to_zotero", forbidden)
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    rag = SimpleNamespace(workspace="flag-off-test")

    # Upstream's allowlist message, not the Zotero path's "Only PDF files".
    with pytest.raises(HTTPException) as exc:
        await _endpoint(tmp_path, rag)(set(), _upload("notes.zzz"))
    assert exc.value.status_code == 400
    assert "Unsupported file type. Supported types" in exc.value.detail


def _client(tmp_path):
    app = FastAPI()
    app.include_router(
        create_document_routes(
            SimpleNamespace(), DocumentManager(str(tmp_path)), api_key="k"
        )
    )
    return TestClient(app)


def test_supported_file_types_is_pdf_only_with_the_flag_on(tmp_path, zotero_on):
    body = (
        _client(tmp_path)
        .get("/documents/supported_file_types", headers={"X-API-Key": "k"})
        .json()
    )
    assert body == {"supported_extensions": [".pdf"], "engines": {}}


def test_supported_file_types_is_unchanged_with_the_flag_off(tmp_path):
    body = (
        _client(tmp_path)
        .get("/documents/supported_file_types", headers={"X-API-Key": "k"})
        .json()
    )
    assert (
        ".md" in body["supported_extensions"] and ".txt" in body["supported_extensions"]
    )
    assert "native" in body["engines"]
