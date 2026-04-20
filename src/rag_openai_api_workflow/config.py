from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str
    model: str
    vector_store_name: str
    vector_store_id: str | None


def get_settings(*, load_env: bool = True) -> Settings:
    """
    Load and validate project settings from `.env` or shell variables.
    Tests can pass `load_env=False` to avoid reading the real local `.env` file.
    """

    if load_env:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your API key."
        )

    return Settings(
        openai_api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
        vector_store_name=os.getenv("OPENAI_VECTOR_STORE_NAME", "rag-openai-docs"),
        vector_store_id=os.getenv("OPENAI_VECTOR_STORE_ID") or None,
    )
