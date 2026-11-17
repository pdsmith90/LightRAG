"""KG_CONTRACT_VALIDATOR: the API server passes the extraction contract's
validator as LightRAG's kg_extraction_validator, and only when it is set."""

import logging
import sys

import pytest

from lightrag import kg_contract
from lightrag.api import config
from lightrag.chunker import plugins, registry

pytestmark = pytest.mark.offline


class _Constructed(Exception):
    pass


def _constructor_kwargs(monkeypatch, tmp_path, env_value):
    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_DUPLICATES", set())
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_failures", [])
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [])
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)
    for name, value in (
        ("CUSTOM_CHUNKER", ""),
        ("LLM_BINDING", "openai"),
        ("EMBEDDING_BINDING", "openai"),
        ("RERANK_BINDING", "null"),
        ("UI_TEMPLATES_DIR", ""),
        ("AUTH_ACCOUNTS", ""),
    ):
        monkeypatch.setenv(name, value)
    if env_value is None:
        monkeypatch.delenv("KG_CONTRACT_VALIDATOR", raising=False)
    else:
        monkeypatch.setenv("KG_CONTRACT_VALIDATOR", env_value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--working-dir",
            str(tmp_path / "rag"),
            "--input-dir",
            str(tmp_path / "inputs"),
        ],
    )
    args = config.initialize_config(force=True)
    from lightrag.api import lightrag_server as server

    captured = []

    def construct(**kwargs):
        captured.append(kwargs)
        raise _Constructed

    monkeypatch.setattr(server, "LightRAG", construct)
    monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
    monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
    with pytest.raises(_Constructed):
        server.create_app(args)
    return args, captured[0]


@pytest.mark.parametrize("env_value", [None, "false"])
def test_off_leaves_the_constructor_default(monkeypatch, tmp_path, env_value):
    args, kwargs = _constructor_kwargs(monkeypatch, tmp_path, env_value)
    assert args.kg_contract_validator is False
    assert "kg_extraction_validator" not in kwargs


def test_on_passes_the_contract_validator(monkeypatch, tmp_path):
    args, kwargs = _constructor_kwargs(monkeypatch, tmp_path, "true")
    assert args.kg_contract_validator is True
    assert kwargs["kg_extraction_validator"] is kg_contract.kg_extraction_validator
