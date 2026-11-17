"""/health shows, next to ``kg_contract_validator``, how many relation types and
inverse names the active entity-type profile declares: 0 and 0 when the
extraction contract is off."""

import pytest

from tests.api.test_health_auth import (  # noqa: F401  (_isolate_env: autouse)
    _FakeLightRAG,
    _build_client,
    _isolate_env,
    _set_auth_mode,
)

pytestmark = pytest.mark.offline


def _health_config(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)
    return client.get("/health").json()["configuration"]


def test_health_shows_the_contract_profile(monkeypatch):
    monkeypatch.setattr(
        _FakeLightRAG,
        "_entity_extraction_prompt_profile",
        {
            "entity_types_guidance": "- Thing: x",
            "relationship_types": {
                "part_of": "SOURCE is in TARGET",
                "uses": "SOURCE takes TARGET",
                "same_as": "SOURCE and TARGET name one thing",
            },
            "relationship_inverses": {"part_of": "has_part", "uses": "used_by"},
        },
        raising=False,
    )
    config = _health_config(monkeypatch)
    assert config["kg_contract_profile"] == {
        "relationship_types": 3,
        "relationship_inverses": 2,
    }
    assert config["kg_contract_validator"] is False


def test_health_shows_the_contract_off_without_the_keys(monkeypatch):
    monkeypatch.setattr(
        _FakeLightRAG,
        "_entity_extraction_prompt_profile",
        {"entity_types_guidance": "- Thing: x"},
        raising=False,
    )
    config = _health_config(monkeypatch)
    assert config["kg_contract_profile"] == {
        "relationship_types": 0,
        "relationship_inverses": 0,
    }
