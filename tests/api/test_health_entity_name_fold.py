"""/health shows ENTITY_NAME_FOLD's lists and alias files (absolute) and the
state of its map next to ``entity_name_fold``. Paths are invented."""

import pytest

import lightrag.entity_name_fold as fold_mod
from tests.api.test_health_auth import (  # noqa: F401  (_isolate_env: autouse)
    _build_client,
    _isolate_env,
    _set_auth_mode,
)

pytestmark = pytest.mark.offline


def test_health_shows_the_fold_files_and_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTITY_NAME_FOLD_LISTS", "fold_lists.json")
    monkeypatch.setenv("ENTITY_NAME_ALIASES", str(tmp_path / "aliases.csv"))
    monkeypatch.setattr(
        fold_mod, "_STATE", {"state": "armed", "lists": "x", "aliases": 2, "keys": 9}
    )
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)

    config = client.get("/health").json()["configuration"]

    assert config["entity_name_fold_lists"] == str(tmp_path / "fold_lists.json")
    assert config["entity_name_aliases"] == str(tmp_path / "aliases.csv")
    assert config["entity_name_fold_state"] == {
        "state": "armed",
        "lists": "x",
        "aliases": 2,
        "keys": 9,
    }


def test_health_shows_empty_fold_paths_when_unset(monkeypatch):
    monkeypatch.delenv("ENTITY_NAME_FOLD_LISTS", raising=False)
    monkeypatch.delenv("ENTITY_NAME_ALIASES", raising=False)
    monkeypatch.setattr(fold_mod, "_STATE", {"state": "off"})
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)

    config = client.get("/health").json()["configuration"]

    assert config["entity_name_fold_lists"] == ""
    assert config["entity_name_aliases"] == ""
    assert config["entity_name_fold_state"] == {"state": "off"}
