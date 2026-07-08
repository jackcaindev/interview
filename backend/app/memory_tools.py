from __future__ import annotations

from uuid import uuid4

from langchain.tools import ToolRuntime
from langgraph.store.base import BaseStore, SearchItem

from app.memory_store import MEMORY_NAMESPACE


def remember_memory(memory: str, runtime: ToolRuntime, topic: str | None = None) -> str:
    """Remember stable workspace context or preferences for future conversations."""
    store = _require_store(runtime)
    cleaned_memory = memory.strip()
    cleaned_topic = topic.strip() if topic else "general"

    if not cleaned_memory:
        return "No memory was saved because the memory text was blank."

    memory_id = str(uuid4())
    store.put(
        MEMORY_NAMESPACE,
        memory_id,
        {
            "memory": cleaned_memory,
            "topic": cleaned_topic or "general",
        },
        index=["memory", "topic"],
    )
    return f"Remembered workspace memory {memory_id}."


def recall_memory(query: str, runtime: ToolRuntime, limit: int = 3) -> str:
    """Recall stable workspace context or preferences from prior conversations."""
    store = _require_store(runtime)
    cleaned_query = query.strip()

    if not cleaned_query:
        return "No memories were recalled because the search query was blank."

    bounded_limit = min(max(limit, 1), 10)
    memories = store.search(MEMORY_NAMESPACE, query=cleaned_query, limit=bounded_limit)

    if not memories:
        return "No relevant workspace memories were found."

    return "\n".join(_format_memory(memory, index) for index, memory in enumerate(memories, start=1))


def _require_store(runtime: ToolRuntime) -> BaseStore:
    if runtime.store is None:
        raise RuntimeError("Long-term memory store is not configured.")
    return runtime.store


def _format_memory(memory: SearchItem, index: int) -> str:
    text = memory.value.get("memory", "")
    topic = memory.value.get("topic", "general")
    return f"{index}. [{topic}] {text}"
