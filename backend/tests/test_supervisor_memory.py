from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app.agent import QuestionRouter, RouteDecision, SUPERVISOR_SYSTEM_PROMPT, SupervisorAgent
from app.config import Settings


class FakeSpecialist:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str) -> str:
        self.questions.append(question)
        return f"answered: {question}"

    def stream_answer(self, question: str):
        self.questions.append(question)
        return iter([f"streamed: {question}"])


class FakeCompiledAgent:
    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
        return {"messages": ["ok"]}


class StubRouter:
    def __init__(self, route: str) -> None:
        self.route_key = route
        self.messages: list[str] = []

    def route(self, message: str) -> RouteDecision:
        self.messages.append(message)
        return RouteDecision(route=self.route_key, confidence=1.0, reason="test route")


def test_supervisor_wires_memory_store_and_tools(monkeypatch) -> None:
    create_agent_calls: list[dict[str, Any]] = []

    def fake_create_agent(**kwargs: Any) -> FakeCompiledAgent:
        create_agent_calls.append(kwargs)
        return FakeCompiledAgent()

    monkeypatch.setattr("app.agent.create_agent", fake_create_agent)
    store = InMemoryStore()
    agent = SupervisorAgent(
        specialists={
            "safety": FakeSpecialist(),
            "maintenance": FakeSpecialist(),
            "quality": FakeSpecialist(),
        },
        router=StubRouter("ambiguous"),
        settings=Settings(),
        store=store,
    )

    agent.invoke("Can you help me with yesterday's issue?", "floor-a")

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


def test_supervisor_does_not_build_model_for_clear_source(monkeypatch) -> None:
    class FailingCompiledAgent:
        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
            raise AssertionError("supervisor model should not be invoked")

    safety = FakeSpecialist()
    maintenance = FakeSpecialist()
    store = InMemoryStore()
    monkeypatch.setattr("app.agent.create_agent", lambda **kwargs: FailingCompiledAgent())
    agent = SupervisorAgent(
        specialists={
            "safety": safety,
            "maintenance": maintenance,
            "quality": FakeSpecialist(),
        },
        settings=Settings(),
        store=store,
    )

    response, thread_id = agent.invoke("What should I do if I find a hydraulic leak?", "floor-a")
    chunks, stream_thread_id = agent.stream("What should I check when a pump motor is overheating?", "floor-b")

    assert response == "answered: What should I do if I find a hydraulic leak?"
    assert thread_id == "floor-a"
    assert safety.questions == ["What should I do if I find a hydraulic leak?"]
    assert list(chunks) == ["streamed: What should I check when a pump motor is overheating?"]
    assert stream_thread_id == "floor-b"
    assert maintenance.questions == ["What should I check when a pump motor is overheating?"]


def test_supervisor_prompt_keeps_controlled_facts_out_of_memory() -> None:
    assert "Long-term memory is never authority" in SUPERVISOR_SYSTEM_PROMPT
    assert "use exactly one specialist tool as the source of truth" in SUPERVISOR_SYSTEM_PROMPT


def test_supervisor_unclear_question_uses_model(monkeypatch) -> None:
    class RecordingCompiledAgent:
        calls = 0

        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
            self.calls += 1
            return {"messages": ["supervisor answer"]}

    compiled_agent = RecordingCompiledAgent()
    monkeypatch.setattr("app.agent.create_agent", lambda **kwargs: compiled_agent)
    agent = SupervisorAgent(
        specialists={
            "safety": FakeSpecialist(),
            "maintenance": FakeSpecialist(),
            "quality": FakeSpecialist(),
        },
        router=StubRouter("ambiguous"),
        settings=Settings(),
        store=InMemoryStore(),
    )

    response, thread_id = agent.invoke("Can you help me with yesterday's issue?", "floor-a")

    assert response == "supervisor answer"
    assert thread_id == "floor-a"
    assert compiled_agent.calls == 1


def test_supervisor_memory_intent_uses_model_even_with_source_terms(monkeypatch) -> None:
    class RecordingCompiledAgent:
        calls = 0

        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, list[Any]]:
            self.calls += 1
            return {"messages": ["memory routed"]}

    compiled_agent = RecordingCompiledAgent()
    monkeypatch.setattr("app.agent.create_agent", lambda **kwargs: compiled_agent)
    agent = SupervisorAgent(
        specialists={
            "safety": FakeSpecialist(),
            "maintenance": FakeSpecialist(),
            "quality": FakeSpecialist(),
        },
        router=StubRouter("memory"),
        settings=Settings(),
        store=InMemoryStore(),
    )

    response, thread_id = agent.invoke("Remember that hydraulic leak cleanup is assigned to floor-a", "floor-a")

    assert response == "memory routed"
    assert thread_id == "floor-a"
    assert compiled_agent.calls == 1


def test_llm_router_can_route_ambiguous_wording_to_specialist(monkeypatch) -> None:
    class FakeRouterAgent:
        def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "structured_response": {
                    "route": "quality",
                    "confidence": 0.81,
                    "reason": "Mentions acceptance criteria.",
                }
            }

    quality = FakeSpecialist()
    monkeypatch.setattr("app.agent.create_agent", lambda **kwargs: FakeRouterAgent())
    agent = SupervisorAgent(
        specialists={
            "safety": FakeSpecialist(),
            "maintenance": FakeSpecialist(),
            "quality": quality,
        },
        router=QuestionRouter(settings=Settings()),
        settings=Settings(),
        store=InMemoryStore(),
    )

    response, thread_id = agent.invoke("How should I classify this result?", "floor-a")

    assert response == "answered: How should I classify this result?"
    assert thread_id == "floor-a"
    assert quality.questions == ["How should I classify this result?"]


def _tool_name(tool: Any) -> str:
    return getattr(tool, "name", getattr(tool, "__name__", ""))
