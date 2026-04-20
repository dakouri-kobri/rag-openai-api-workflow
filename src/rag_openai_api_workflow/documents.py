from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".html"}


def collect_supported_files(path: Path) -> list[Path]:
    """Return supported document files from a file or directory in stable order."""
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )