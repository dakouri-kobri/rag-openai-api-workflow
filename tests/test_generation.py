from __future__ import annotations

from types import SimpleNamespace

from rag_openai_api_workflow.generation import answer_question, build_grounded_user_prompt


class FakeResponses:
    def __init__(self) -> None:
        self.create_kwargs = {}

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(output_text="Grounded answer [1].")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_build_grounded_user_prompt_includes_question_and_context() -> None:
    prompt = build_grounded_user_prompt(
        question="What is RAG?",
        context="[1] notes.md\nRAG uses retrieved context.",
    )

    assert "What is RAG?" in prompt
    assert "RAG uses retrieved context." in prompt


def test_answer_question_returns_fallback_when_no_context() -> None:
    client = FakeClient()
    search_results = SimpleNamespace(data=[])

    answer = answer_question(
        client,
        model="test-model",
        question="What is RAG?",
        search_results=search_results,
    )

    assert "could not find enough relevant context" in answer


def test_answer_question_calls_responses_api_with_context() -> None:
    client = FakeClient()
    search_results = SimpleNamespace(
        data=[
            SimpleNamespace(
                filename="rag_notes.md",
                score=0.95,
                content=[SimpleNamespace(text="RAG uses retrieved context.")],
            )
        ]
    )

    answer = answer_question(
        client,
        model="test-model",
        question="What is RAG?",
        search_results=search_results,
    )

    assert answer == "Grounded answer [1]."
    assert client.responses.create_kwargs["model"] == "test-model"
    assert "RAG uses retrieved context." in client.responses.create_kwargs["input"][1]["content"]