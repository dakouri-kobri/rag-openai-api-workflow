from __future__ import annotations

from rag_openai_api_workflow.cli import build_parser


def test_parser_handles_ingest_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["ingest", "--path", "data/sample"])

    assert args.command == "ingest"
    assert args.path == "data/sample"


def test_parser_handles_search_command() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "search",
            "What is RAG?",
            "--max-results",
            "3",
            "--score-threshold",
            "0.7",
            "--version",
            "1.0",
        ]
    )

    assert args.command == "search"
    assert args.query == "What is RAG?"
    assert args.max_results == 3
    assert args.score_threshold == 0.7
    assert args.version == "1.0"


def test_parser_handles_ask_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["ask", "Explain project-driven learning."])

    assert args.command == "ask"
    assert args.question == "Explain project-driven learning."