"""/health reports ``zotero_struct`` (whether the paper structure can be read) and
``zotero_struct_tables`` (the optional v2 tables it found) only while
ZOTERO_STRUCT_SCHEMA is set; unset, the report is what it always was."""

import pytest

import lightrag.zotero_struct as zs
from tests.api.test_health_auth import (  # noqa: F401  (_isolate_env: autouse)
    _build_client,
    _isolate_env,
    _set_auth_mode,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _fresh_state():
    zs._reset_for_tests()
    yield
    zs._reset_for_tests()


def _configuration(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)
    return client.get("/health").json()["configuration"]


def test_health_has_no_zotero_struct_key_while_the_setting_is_unset(monkeypatch):
    monkeypatch.delenv(zs.SCHEMA_ENV, raising=False)

    async def must_not_be_asked(_rag):
        raise AssertionError("the catalog must not be read while the setting is unset")

    monkeypatch.setattr(zs, "available", must_not_be_asked)
    configuration = _configuration(monkeypatch)
    assert "zotero_struct" not in configuration
    assert "zotero_struct_tables" not in configuration


def test_health_reports_a_readable_structure(monkeypatch):
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs")

    async def readable(_rag):
        return True

    monkeypatch.setattr(zs, "available", readable)
    assert _configuration(monkeypatch)["zotero_struct"] is True


def test_health_reports_false_when_the_structure_cannot_be_read(monkeypatch):
    # the fake server has no PostgreSQL document-status storage
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs")
    assert _configuration(monkeypatch)["zotero_struct"] is False


def test_health_lists_the_optional_tables_of_a_readable_structure(monkeypatch):
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs")

    async def readable(_rag):
        zs._state.update(
            known=("zs", True, None), tables=["external_works", "paper_embeddings"]
        )
        return True

    monkeypatch.setattr(zs, "available", readable)
    configuration = _configuration(monkeypatch)
    assert configuration["zotero_struct"] is True
    assert configuration["zotero_struct_tables"] == [
        "external_works",
        "paper_embeddings",
    ]


def test_health_lists_no_tables_when_the_structure_cannot_be_read(monkeypatch):
    monkeypatch.setenv(zs.SCHEMA_ENV, "zs")
    configuration = _configuration(monkeypatch)
    assert configuration["zotero_struct"] is False
    assert configuration["zotero_struct_tables"] == []
