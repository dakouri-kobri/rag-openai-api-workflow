from __future__ import annotations

import pytest

from rag_openai_api_workflow.documents import collect_supported_files


def test_collect_supported_files_from_directory(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    markdown_file = docs_dir / "b.md"
    text_file = docs_dir / "a.txt"
    ignored_file = docs_dir / "ignore.csv"

    markdown_file.write_text("markdown", encoding="utf-8")
    text_file.write_text("text", encoding="utf-8")
    ignored_file.write_text("ignored", encoding="utf-8")

    files = collect_supported_files(docs_dir)

    assert files == [text_file, markdown_file]


def test_collect_supported_files_from_single_file(tmp_path) -> None:
    file_path = tmp_path / "note.md"
    file_path.write_text("hello", encoding="utf-8")

    assert collect_supported_files(file_path) == [file_path]


def test_collect_supported_files_raises_for_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_supported_files(tmp_path / "missing")