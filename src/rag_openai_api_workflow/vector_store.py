from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_openai_api_workflow.storage import VECTOR_STORE_FILE, save_vector_store_id


def create_vector_store(
    client: Any,
    name: str,
    *,
    expires_after_days: int = 30,
    save_path: Path = VECTOR_STORE_FILE,
) -> str:
    """Create an OpenAI vector store and save its ID for later CLI commands."""
    vector_store = client.vector_stores.create(
        name=name,
        expires_after={
            "anchor": "last_active_at",
            "days": expires_after_days,
        },
    )

    save_vector_store_id(vector_store.id, save_path)
    return vector_store.id