from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from langsmith import Client
from langsmith.run_helpers import tracing_context

from app.core.config import settings


def is_langsmith_enabled() -> bool:
    return bool(settings.langsmith_enabled and settings.langsmith_api_key)


@lru_cache
def get_langsmith_client() -> Client | None:
    if not is_langsmith_enabled():
        return None

    os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
    return Client(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_api_url or None,
        workspace_id=settings.langsmith_workspace_id or None,
    )


@contextmanager
def legal_tracing_context(
    *,
    operation: str,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[None]:
    client = get_langsmith_client()
    if client is None:
        yield
        return

    trace_metadata = {"operation": operation, **(metadata or {})}
    trace_tags = ["legal-agent", operation, *(tags or [])]

    with tracing_context(
        project_name=settings.langsmith_project,
        metadata=trace_metadata,
        tags=trace_tags,
        enabled=True,
        client=client,
    ):
        yield
