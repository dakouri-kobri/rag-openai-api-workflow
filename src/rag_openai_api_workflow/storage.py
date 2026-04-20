from __future__ import annotations

import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
VECTOR_STORE_FILE = ARTIFACTS_DIR / "vector_store.json"


def save_vector_store_id(
    vector_store_id: str,
    path: Path = VECTOR_STORE_FILE,
) -> None:
    """Save the active vector store ID to a local ignored artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"vector_store_id": vector_store_id}, indent=2),
        encoding="utf-8",
    )


def load_vector_store_id(path: Path = VECTOR_STORE_FILE) -> str | None:
    """Load the active vector store ID if it has already been saved locally."""
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("vector_store_id")