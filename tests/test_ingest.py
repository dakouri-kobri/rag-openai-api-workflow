from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_openai_api_workflow.documents import collect_supported_files

UploadedFile = dict[str, Path | str]


def upload_file(client: Any, file_path: Path) -> UploadedFile:
    """Upload a local file to OpenAI file storage and return its file ID."""
    with file_path.open("rb") as file_handle:
        uploaded_file = client.files.create(
            file=file_handle,
            purpose="assistants",
        )

    return {
        "path": file_path,
        "file_id": uploaded_file.id,
    }


def build_vector_store_file_entry(uploaded_file: UploadedFile) -> dict[str, Any]:
    """Build the per-file vector-store payload with metadata and chunking."""
    file_path = Path(uploaded_file["path"])

    return {
        "file_id": uploaded_file["file_id"],
        "attributes": {
            "source": "sample_knowledge_base",
            "version": "1.0",
            "file_stem": file_path.stem,
            "extension": file_path.suffix.lower().replace(".", ""),
        },
        "chunking_strategy": {
            "type": "static",
            "static": {
                "max_chunk_size_tokens": 800,
                "chunk_overlap_tokens": 400,
            },
        },
    }


def attach_files_to_vector_store(
    client: Any,
    vector_store_id: str,
    uploaded_files: list[UploadedFile],
) -> Any:
    """Attach uploaded files to a vector store and wait for processing."""
    files = [build_vector_store_file_entry(uploaded) for uploaded in uploaded_files]

    return client.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id,
        files=files,
    )


def ingest_path(client: Any, path: Path, vector_store_id: str) -> Any:
    """Upload supported files from `path` and attach them to the vector store."""
    file_paths = collect_supported_files(path)

    if not file_paths:
        raise FileNotFoundError(f"No supported files found in {path}")

    # OpenAI vector stores use uploaded file IDs, so upload first, attach second.
    uploaded_files = [upload_file(client, file_path) for file_path in file_paths]

    return attach_files_to_vector_store(
        client=client,
        vector_store_id=vector_store_id,
        uploaded_files=uploaded_files,
    )