from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import langsmith as ls
from langsmith.run_trees import RunTree

from app.config import Settings, get_settings


_TRUTHY = {"1", "true", "t", "yes", "y", "on"}


def langsmith_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    tracing = str(getattr(settings, "langsmith_tracing", "false"))
    api_key = str(getattr(settings, "langsmith_api_key", ""))
    return _is_truthy(tracing) and bool(api_key.strip())


def configure_langsmith(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    api_key = str(getattr(settings, "langsmith_api_key", ""))
    project = str(getattr(settings, "langsmith_project", ""))
    if api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)


@contextmanager
def langsmith_trace(
    name: str,
    *,
    settings: Settings | None = None,
    inputs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: list[str] | None = None,
    run_type: str = "chain",
) -> Iterator[RunTree | None]:
    settings = settings or get_settings()
    if not langsmith_enabled(settings):
        yield None
        return

    configure_langsmith(settings)
    with ls.tracing_context(
        enabled=True,
        project_name=getattr(settings, "langsmith_project", "default"),
        tags=tags,
        metadata=dict(metadata or {}),
    ):
        with ls.trace(
            name,
            run_type,
            inputs=dict(inputs or {}),
            project_name=getattr(settings, "langsmith_project", "default"),
            metadata=dict(metadata or {}),
            tags=tags,
        ) as run:
            yield run


def end_langsmith_trace(run: RunTree | None, outputs: Mapping[str, Any]) -> None:
    if run is not None:
        run.end(outputs=dict(outputs))


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY
