from __future__ import annotations

from openai import OpenAI

from rag_openai_api_workflow.config import Settings, get_settings


def make_client(settings: Settings | None = None) -> OpenAI:
    """Create an OpenAI client using the configured API key."""

    settings = settings or get_settings()

    return OpenAI(api_key=settings.openai_api_key)