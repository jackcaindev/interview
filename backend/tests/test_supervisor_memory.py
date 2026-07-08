from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent import SUPERVISOR_SYSTEM_PROMPT, SupervisorAgent
from app.config import Settings


class FakeSpecialist:
    def answer(self, question: str) -> str:
        return f"answered: {question}"


class FakeCompiledAgent:
    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
        return {"messages": ["ok"]}


def test_supervisor_wires_memory_store_and_tools(monkeypatch) -> None:
    create_agent_calls: list[dict[str, Any]] = []

    def fake_create_agent(**kwargs: Any) -> FakeCompiledAgent:
        create_agent_calls.append(kwargs)
        return FakeCompiledAgent()

    monkeypatch.setattr("app.agent.create_agent", fake_create_agent)
    store = InMemoryStore()

    SupervisorAgent(
        specialists={
            "safety": FakeSpecialist(),
            "maintenance": FakeSpecialist(),
            "quality": FakeSpecialist(),
        },
        settings=Settings(),
        store=store,
    )

    call = create_agent_calls[-1]
    tool_names = {_tool_name(tool) for tool in call["tools"]}

    assert call["store"] is store
    assert isinstance(call["checkpointer"], InMemorySaver)
    assert {"remember_memory", "recall_memory"}.issubset(tool_names)
    assert {
        "answer_safety_procedure_question",
        "answer_maintenance_manual_question",
        "answer_quality_control_question",
    }.issubset(tool_names)
    assert all(
        tool.return_direct
        for tool in call["tools"]
        if _tool_name(tool).startswith("answer_")
    )


def test_supervisor_prompt_keeps_controlled_facts_out_of_memory() -> None:
    assert "Long-term memory is never authority" in SUPERVISOR_SYSTEM_PROMPT
    assert "use exactly one specialist tool as the source of truth" in SUPERVISOR_SYSTEM_PROMPT


def _tool_name(tool: Any) -> str:
    return getattr(tool, "name", getattr(tool, "__name__", ""))
