from __future__ import annotations

from typing import Any


def build_version_filter(version: str) -> dict[str, str]:
    """Create a simple metadata filter for document version."""
    return {
        "type": "eq",
        "key": "version",
        "value": version,
    }


def search_vector_store(
    client: Any,
    vector_store_id: str,
    query: str,
    *,
    max_results: int = 5,
    score_threshold: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> Any:
    """Run semantic search against an OpenAI vector store."""
    return client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        filters=filters,
        max_num_results=max_results,
        rewrite_query=True,
        ranking_options={
            "ranker": "auto",
            "score_threshold": score_threshold,
        },
    )


def extract_search_result_text(result: Any) -> str:
    """Extract readable text from one vector-store search result."""
    content_parts = getattr(result, "content", [])
    texts = []

    for part in content_parts:
        if isinstance(part, dict):
            text = part.get("text", "")
        else:
            text = getattr(part, "text", "")

        if text:
            texts.append(text)

    return "\n".join(texts)


def format_search_results(results: Any) -> str:
    """Format vector-store search results as a compact context block."""
    lines = []

    for index, result in enumerate(results.data, start=1):
        filename = getattr(result, "filename", "unknown")
        score = getattr(result, "score", 0.0)
        text = extract_search_result_text(result)

        lines.append(f"[{index}] {filename} | score={score:.3f}\n{text}")

    return "\n\n".join(lines)