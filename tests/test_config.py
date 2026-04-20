from __future__ import annotations

import pytest

from rag_openai_api_workflow.config import get_settings


def test_get_settings_loads_required_and_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_VECTOR_STORE_ID", raising=False)
    monkeypatch.delenv("OPENAI_VECTOR_STORE_NAME", raising=False)

    settings = get_settings(load_env=False)

    assert settings.openai_api_key == "test-key"
    assert settings.model == "gpt-5.4-mini"
    assert settings.vector_store_name == "rag-openai-docs"
    assert settings.vector_store_id is None


def test_get_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
        get_settings(load_env=False)