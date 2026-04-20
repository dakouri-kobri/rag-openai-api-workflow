from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from rag_openai_api_workflow.client import make_client
from rag_openai_api_workflow.config import get_settings
from rag_openai_api_workflow.generation import answer_question
from rag_openai_api_workflow.ingest import ingest_path
from rag_openai_api_workflow.retrieval import (
    build_version_filter,
    format_search_results,
    search_vector_store,
)
from rag_openai_api_workflow.storage import load_vector_store_id
from rag_openai_api_workflow.vector_store import create_vector_store

console = Console()


def resolve_vector_store_id(args_vector_store_id: str | None) -> str:
    """Resolve vector store ID from CLI, environment, or local artifact."""
    settings = get_settings()

    vector_store_id = (
        args_vector_store_id
        or settings.vector_store_id
        or load_vector_store_id()
    )

    if not vector_store_id:
        raise RuntimeError(
            "No vector store ID found. Run `uv run rag-openai ingest --path data/sample` first."
        )

    return vector_store_id


def handle_ingest(args: argparse.Namespace) -> None:
    """Create or reuse a vector store, then ingest local documents."""
    settings = get_settings()
    client = make_client(settings)

    vector_store_id = args.vector_store_id or settings.vector_store_id

    if not vector_store_id:
        vector_store_id = create_vector_store(client, settings.vector_store_name)
        console.print(f"[green]Created vector store:[/green] {vector_store_id}")

    batch = ingest_path(
        client=client,
        path=Path(args.path),
        vector_store_id=vector_store_id,
    )

    console.print("[green]Ingestion complete.[/green]")
    console.print(f"Vector store ID: {vector_store_id}")
    console.print(f"Batch status: {batch.status}")


def handle_search(args: argparse.Namespace) -> None:
    """Search the vector store and print formatted chunks."""
    client = make_client()
    vector_store_id = resolve_vector_store_id(args.vector_store_id)

    filters = build_version_filter(args.version) if args.version else None

    results = search_vector_store(
        client=client,
        vector_store_id=vector_store_id,
        query=args.query,
        max_results=args.max_results,
        score_threshold=args.score_threshold,
        filters=filters,
    )

    console.print(format_search_results(results))


def handle_ask(args: argparse.Namespace) -> None:
    """Search first, then generate a grounded answer."""
    settings = get_settings()
    client = make_client(settings)
    vector_store_id = resolve_vector_store_id(args.vector_store_id)

    results = search_vector_store(
        client=client,
        vector_store_id=vector_store_id,
        query=args.question,
        max_results=args.max_results,
        score_threshold=args.score_threshold,
    )

    answer = answer_question(
        client=client,
        model=settings.model,
        question=args.question,
        search_results=results,
    )

    console.print(answer)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for all project commands."""
    parser = argparse.ArgumentParser(
        description="Project: RAG workflow with OpenAI vector-store"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Upload files to a vector store")
    ingest_parser.add_argument("--path", required=True, help="File or directory to ingest")
    ingest_parser.add_argument("--vector-store-id", default=None)
    ingest_parser.set_defaults(func=handle_ingest)

    search_parser = subparsers.add_parser("search", help="Search the vector store")
    search_parser.add_argument("query")
    search_parser.add_argument("--vector-store-id", default=None)
    search_parser.add_argument("--max-results", type=int, default=5)
    search_parser.add_argument("--score-threshold", type=float, default=0.0)
    search_parser.add_argument("--version", default=None)
    search_parser.set_defaults(func=handle_search)

    ask_parser = subparsers.add_parser("ask", help="Ask a grounded RAG question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--vector-store-id", default=None)
    ask_parser.add_argument("--max-results", type=int, default=5)
    ask_parser.add_argument("--score-threshold", type=float, default=0.0)
    ask_parser.set_defaults(func=handle_ask)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc