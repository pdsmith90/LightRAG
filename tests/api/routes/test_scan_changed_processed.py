"""SCAN_DIVERT_CHANGED_PROCESSED: opt-in diversion of a changed processed source.

A document ingested as a stub before its full text arrived used to be archived
unread by every scan. With the flag on, a plain-text file whose length differs
materially from the stored document goes to ``__changed__/`` with a warning;
the doc_status row is never touched. Driven through the real
``iter_new_files`` and the real scan loop.
"""

import asyncio
import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag.base import DocProcessingStatus, DocStatus, SourceUnique  # noqa: E402
from lightrag.constants import PARSED_DIR_NAME  # noqa: E402

DocumentManager = _document_routes.DocumentManager
run_scanning_process = _document_routes.run_scanning_process
_changed = _document_routes._content_materially_changed
CHANGED_DIR_NAME = _document_routes.CHANGED_DIR_NAME

pytestmark = pytest.mark.offline

DOC_ID = "doc-invented-1"


@pytest.fixture(autouse=True)
def _ensure_shared_storage_initialized():
    importlib.import_module("lightrag.kg.shared_storage").initialize_share_data()
    yield


class _ProcessedDocStatus:
    def __init__(self, content_length):
        self.content_length = content_length
        self.deleted: list = []
        self.row = DocProcessingStatus(
            content_summary="",
            content_length=content_length if content_length is not None else 0,
            file_path="note.md",
            status=DocStatus.PROCESSED,
            created_at="",
            updated_at="",
        )
        if content_length is None:
            self.row = {"status": "processed"}  # no content_length key at all

    async def resolve_doc_source_strict(self, _key):
        record = SimpleNamespace(status=DocStatus.PROCESSED)
        return SourceUnique(doc_id=DOC_ID, doc=record)

    async def get_full_docs_by_ids(self, doc_ids, *, strict=False):
        return {DOC_ID: self.row} if DOC_ID in doc_ids else {}

    async def delete(self, ids):
        self.deleted.extend(ids)


class _Rag:
    def __init__(self, content_length):
        self.workspace = f"changed-{uuid4().hex[:8]}"
        self.doc_status = _ProcessedDocStatus(content_length)
        self.full_docs = SimpleNamespace(supports_strict_point_reads=True)

    async def arollback_failed_custom_chunk_patches(self, **_kwargs):
        return {"rolled_back": [], "failed": []}

    async def apipeline_process_enqueue_documents(self):
        return None


async def _history(workspace):
    shared = importlib.import_module("lightrag.kg.shared_storage")
    await shared.initialize_pipeline_status(workspace=workspace)
    status = await shared.get_namespace_data("pipeline_status", workspace=workspace)
    return status


async def _scan(tmp_path, monkeypatch, *, flag, stored, files, rag=None):
    rag = rag or _Rag(stored)
    await _history(rag.workspace)
    monkeypatch.setattr(
        _document_routes.global_args,
        "scan_divert_changed_processed",
        flag,
        raising=False,
    )
    counts: dict[str, int] = {}
    original = _document_routes._ScanJobReporter.count

    def _spy(self, key, delta=1):
        counts[key] = counts.get(key, 0) + delta
        return original(self, key, delta)

    monkeypatch.setattr(_document_routes._ScanJobReporter, "count", _spy)

    async def _noop(_rag, candidates, _track):
        return len(candidates)

    monkeypatch.setattr(_document_routes, "pipeline_enqueue_scan_batch", _noop)
    doc_manager = DocumentManager(str(tmp_path))
    for name, text in files.items():
        (doc_manager.input_dir / name).write_text(text, encoding="utf-8")
    await run_scanning_process(rag, doc_manager, "track-changed")
    return rag, doc_manager, counts


def _names(directory):
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_flag_off_archives_a_much_larger_file(tmp_path, monkeypatch):
    async def _run():
        _rag, dm, counts = await _scan(
            tmp_path, monkeypatch, flag=False, stored=170, files={"note.md": "x" * 9000}
        )
        assert _names(dm.input_dir / PARSED_DIR_NAME) == ["note.md"]
        assert not (dm.input_dir / CHANGED_DIR_NAME).exists()
        assert "processed_changed" not in counts

    asyncio.run(_run())


def test_flag_on_diverts_a_much_larger_md(tmp_path, monkeypatch):
    async def _run():
        rag, dm, counts = await _scan(
            tmp_path, monkeypatch, flag=True, stored=170, files={"note.md": "x" * 9000}
        )
        assert _names(dm.input_dir / CHANGED_DIR_NAME) == ["note.md"]
        assert not (dm.input_dir / PARSED_DIR_NAME).exists()
        assert not (dm.input_dir / "note.md").exists()
        assert counts.get("processed_changed") == 1
        assert rag.doc_status.deleted == []
        assert rag.doc_status.row.content_length == 170
        status = await _history(rag.workspace)
        msg = [m for m in status["history_messages"] if "changed" in m][-1]
        assert DOC_ID in msg and "170" in msg and "9000" in msg
        assert "delete document " + DOC_ID in msg

    asyncio.run(_run())


def test_flag_on_similar_length_is_archived(tmp_path, monkeypatch):
    async def _run():
        _rag, dm, counts = await _scan(
            tmp_path, monkeypatch, flag=True, stored=5000, files={"note.md": "x" * 6000}
        )
        assert _names(dm.input_dir / PARSED_DIR_NAME) == ["note.md"]
        assert "processed_changed" not in counts

    asyncio.run(_run())


def test_flag_on_much_smaller_file_is_diverted(tmp_path, monkeypatch):
    async def _run():
        _rag, dm, counts = await _scan(
            tmp_path, monkeypatch, flag=True, stored=9000, files={"note.txt": "x" * 100}
        )
        assert _names(dm.input_dir / CHANGED_DIR_NAME) == ["note.txt"]
        assert counts.get("processed_changed") == 1

    asyncio.run(_run())


def test_flag_on_non_text_suffix_is_archived(tmp_path, monkeypatch):
    async def _run():
        _rag, dm, counts = await _scan(
            tmp_path,
            monkeypatch,
            flag=True,
            stored=170,
            files={"note.docx": "x" * 9000},
        )
        assert not (dm.input_dir / CHANGED_DIR_NAME).exists()
        assert "processed_changed" not in counts

    asyncio.run(_run())


@pytest.mark.parametrize("stored", [None, 0])
def test_flag_on_missing_content_length_is_archived(tmp_path, monkeypatch, stored):
    async def _run():
        _rag, dm, counts = await _scan(
            tmp_path,
            monkeypatch,
            flag=True,
            stored=stored,
            files={"note.md": "x" * 9000},
        )
        assert _names(dm.input_dir / PARSED_DIR_NAME) == ["note.md"]
        assert "processed_changed" not in counts

    asyncio.run(_run())


def test_name_collision_in_changed_dir_is_suffixed(tmp_path, monkeypatch):
    async def _run():
        changed = tmp_path / CHANGED_DIR_NAME
        changed.mkdir()
        (changed / "note.md").write_text("earlier", encoding="utf-8")
        _rag, dm, _counts = await _scan(
            tmp_path, monkeypatch, flag=True, stored=170, files={"note.md": "y" * 9000}
        )
        assert _names(dm.input_dir / CHANGED_DIR_NAME) == ["note.md", "note_001.md"]
        assert (changed / "note.md").read_text(encoding="utf-8") == "earlier"

    asyncio.run(_run())


def test_diverted_file_is_not_rediscovered(tmp_path, monkeypatch):
    async def _run():
        rag, dm, counts = await _scan(
            tmp_path, monkeypatch, flag=True, stored=170, files={"note.md": "x" * 9000}
        )
        assert counts.get("processed_changed") == 1
        rag2, _dm, counts2 = await _scan(
            tmp_path, monkeypatch, flag=True, stored=170, files={}, rag=rag
        )
        assert counts2.get("processed_changed") is None
        assert _names(dm.input_dir / CHANGED_DIR_NAME) == ["note.md"]
        assert not (dm.input_dir / PARSED_DIR_NAME).exists()

    asyncio.run(_run())


def test_move_failure_leaves_file_in_place(tmp_path, monkeypatch):
    async def _run():
        original = _document_routes.Path.rename

        def _boom(self, target):
            raise OSError("disk full")

        monkeypatch.setattr(_document_routes.Path, "rename", _boom)
        try:
            _rag, dm, counts = await _scan(
                tmp_path,
                monkeypatch,
                flag=True,
                stored=170,
                files={"note.md": "x" * 9000},
            )
        finally:
            monkeypatch.setattr(_document_routes.Path, "rename", original)
        assert (dm.input_dir / "note.md").exists()
        assert counts.get("errors") == 1
        assert "processed_changed" not in counts

    asyncio.run(_run())


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (1000, 3000, True),  # exactly 2x, delta 2000
        (1000, 2999, False),  # delta 1999
        (1000, 2000, False),  # 2x but delta 1000
        (5000, 6000, False),
        (3000, 1000, True),  # exactly half, delta 2000
        (3000, 1001, False),  # delta 1999
        (3000, 1500, False),  # half but delta 1500
        (170, 1_160_000, True),
        (0, 2000, True),
        (4000, 4000, False),
    ],
)
def test_content_materially_changed_boundaries(old, new, expected):
    assert _changed(old, new) is expected
