from __future__ import annotations

from types import SimpleNamespace

from rag_openai_api_workflow.storage import load_vector_store_id
from rag_openai_api_workflow.vector_store import create_vector_store


class FakeVectorStores:
    def __init__(self) -> None:
        self.create_kwargs = {}

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="vs_test123")


class FakeClient:
    def __init__(self) -> None:
        self.vector_stores = FakeVectorStores()


def test_create_vector_store_saves_id(tmp_path) -> None:
    save_path = tmp_path / "vector_store.json"
    client = FakeClient()

    vector_store_id = create_vector_store(
        client,
        "test-store",
        expires_after_days=7,
        save_path=save_path,
    )

    assert vector_store_id == "vs_test123"
    assert load_vector_store_id(save_path) == "vs_test123"
    assert client.vector_stores.create_kwargs["name"] == "test-store"
    assert client.vector_stores.create_kwargs["expires_after"]["days"] == 7