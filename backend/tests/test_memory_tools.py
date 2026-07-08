from types import SimpleNamespace

from langgraph.store.memory import InMemoryStore

from app.memory_store import MEMORY_NAMESPACE
from app.memory_tools import recall_memory, remember_memory


def test_remember_memory_stores_workspace_memory() -> None:
    store = InMemoryStore()
    runtime = SimpleNamespace(store=store)

    result = remember_memory(
        "Line 2 supervisors prefer metric units in summaries.",
        runtime,
        topic="line 2",
    )

    memories = store.search(MEMORY_NAMESPACE)
    assert result.startswith("Remembered workspace memory ")
    assert len(memories) == 1
    assert memories[0].value == {
        "memory": "Line 2 supervisors prefer metric units in summaries.",
        "topic": "line 2",
    }


def test_recall_memory_returns_matching_memories() -> None:
    store = InMemoryStore()
    runtime = SimpleNamespace(store=store)
    store.put(
        MEMORY_NAMESPACE,
        "memory-1",
        {"memory": "Line 2 supervisors prefer metric units in summaries.", "topic": "line 2"},
    )

    result = recall_memory("metric summaries", runtime)

    assert "1. [line 2] Line 2 supervisors prefer metric units in summaries." in result


def test_recall_memory_handles_empty_results() -> None:
    store = InMemoryStore()
    runtime = SimpleNamespace(store=store)

    assert recall_memory("anything", runtime) == "No relevant workspace memories were found."
