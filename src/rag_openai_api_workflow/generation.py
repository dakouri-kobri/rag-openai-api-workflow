from __future__ import annotations

from typing import Any

from rag_openai_api_workflow.retrieval import format_search_results

SYSTEM_PROMPT = (
    "You are a careful RAG assistant. Answer only from the provided context. "
    "If the context is insufficient, say so. Cite sources using [1], [2], etc."
)


def build_grounded_user_prompt(question: str, context: str) -> str:
    """Build the user prompt that combines the question and retrieved context."""
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Write a clear, concise answer."
    )


def answer_question(
    client: Any,
    *,
    model: str,
    question: str,
    search_results: Any,
) -> str:
    """Generate an answer grounded in vector-store search results."""
    context = format_search_results(search_results)

    if not context.strip():
        return "I could not find enough relevant context to answer the question."

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_grounded_user_prompt(question, context),
            },
        ],
    )

    return response.output_text