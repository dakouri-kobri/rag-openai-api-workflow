from __future__ import annotations

from rag_openai_api_workflow.storage import load_vector_store_id, save_vector_store_id


def test_load_vector_store_id_returns_none_when_file_missing(tmp_path) -> None:
    path = tmp_path / "vector_store.json"

    assert load_vector_store_id(path) is None


def test_save_and_load_vector_store_id(tmp_path) -> None:
    path = tmp_path / "vector_store.json"

    save_vector_store_id("vs_test123", path)

    assert load_vector_store_id(path) == "vs_test123"