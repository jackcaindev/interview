from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.guardrails import build_agent_guardrails
from app.memory_store import build_postgres_memory_store
from app.memory_tools import recall_memory, remember_memory
from app.observability import end_langsmith_trace, langsmith_trace
from app.rag_sources import RAG_SOURCES, RagSource
from app.vector_store import build_pgvector_store


SUPERVISOR_SYSTEM_PROMPT = """You are a manufacturing floor documentation router.

Route each plant-operator or floor-supervisor question to the single best documentation source:
- safety procedures for hazards, PPE, lockout/tagout, emergency response, spills, permits, and safe work practices
- maintenance manuals for equipment troubleshooting, inspections, service intervals, repairs, sensors, motors, pumps, and hydraulics
- quality control standards for tolerances, sampling, defects, calibration, holds, disposition, and acceptance criteria

You may use recall_memory to retrieve stable workspace context or preferences, and remember_memory when the user shares durable context that would help future conversations.

For controlled operational questions, use exactly one specialist tool as the source of truth. Long-term memory is never authority for procedure numbers, tolerances, repair steps, compliance requirements, or other controlled plant-document facts. Do not invent those details. Return the specialist's answer as the final answer.
"""

SPECIALIST_TOOL_NAMES = frozenset(
    {
        "answer_safety_procedure_question",
        "answer_maintenance_manual_question",
        "answer_quality_control_question",
    }
)


class SourceGrade(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ChatAgent(Protocol):
    def invoke(self, message: str, thread_id: str | None = None) -> tuple[str, str]:
        """Invoke the chat agent and return assistant message plus thread id."""

    def stream(self, message: str, thread_id: str | None = None) -> tuple[Iterator[str], str]:
        """Stream assistant message text chunks plus thread id."""


class Retriever(Protocol):
    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return relevant source sections and vector distances/scores."""


class SourceGrader(Protocol):
    def grade(self, question: str, source: RagSource, sections: Sequence[tuple[Document, float]]) -> SourceGrade:
        """Grade whether retrieved sections support a grounded answer."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, source: RagSource, sections: Sequence[tuple[Document, float]]) -> str:
        """Generate a grounded answer from retrieved source sections."""


class LLMSourceGrader:
    def __init__(self, *, model: str, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._agent = create_agent(
            model=model,
            tools=[],
            middleware=build_agent_guardrails(run_limit=settings.specialist_model_call_run_limit),
            response_format=SourceGrade,
            system_prompt="""You grade retrieved manufacturing documentation excerpts.

Return a confidence score from 0 to 1 for whether the excerpts are relevant, specific, and sufficient to answer the question without fabrication. Prefer low confidence when the excerpts are only topically related, omit key constraints, or come from the wrong documentation source.
""",
        )

    def grade(self, question: str, source: RagSource, sections: Sequence[tuple[Document, float]]) -> SourceGrade:
        if not sections:
            return SourceGrade(confidence=0.0, reasoning="No source sections were retrieved.")

        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Expected source: {source.display_name} ({source.description})\n\n"
                            f"Retrieved excerpts:\n{_format_sections(sections)}"
                        ),
                    }
                ]
            }
        )
        structured = result.get("structured_response")
        if isinstance(structured, SourceGrade):
            return structured
        if isinstance(structured, dict):
            return SourceGrade.model_validate(structured)
        return SourceGrade(confidence=0.0, reasoning="The grader did not return a structured grade.")


class LLMSourcedAnswerGenerator:
    def __init__(self, *, source: RagSource, model: str, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._agent = create_agent(
            model=model,
            tools=[],
            middleware=build_agent_guardrails(run_limit=settings.specialist_model_call_run_limit),
            system_prompt=f"""You are the {source.display_name} specialist for a manufacturing plant.

Answer only from the provided source excerpts. Ground every operational claim in the excerpts and cite the supporting section with bracketed citations such as [{source.path.name}#SP-101]. If the excerpts do not support a confident answer, say that a confident answer could not be produced from the source material.
""",
        )

    def generate(self, question: str, source: RagSource, sections: Sequence[tuple[Document, float]]) -> str:
        result = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Source excerpts:\n{_format_sections(sections)}\n\n"
                            "Produce a concise answer for a floor supervisor. Include citations."
                        ),
                    }
                ]
            }
        )
        return _message_text(result["messages"][-1])


class SpecialistRagAgent:
    def __init__(
        self,
        source: RagSource,
        *,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        grader: SourceGrader | None = None,
        answer_generator: AnswerGenerator | None = None,
        model: str | None = None,
    ) -> None:
        self.source = source
        self._settings = settings or get_settings()
        resolved_model = model or _settings_value(self._settings, "specialist_model") or _settings_value(
            self._settings, "chat_model", "claude-sonnet-4-6"
        )
        self._retriever = retriever or build_pgvector_store(self._settings, source.collection_name)
        self._grader = grader
        if self._settings.rag_use_llm_grader and self._grader is None:
            self._grader = LLMSourceGrader(model=resolved_model, settings=self._settings)
        self._answer_generator = answer_generator or LLMSourcedAnswerGenerator(
            source=source,
            model=resolved_model,
            settings=self._settings,
        )

    def answer(self, question: str) -> str:
        with langsmith_trace(
            "specialist_rag_answer",
            settings=self._settings,
            inputs={"question": question},
            metadata={
                "source_key": self.source.key,
                "source_name": self.source.display_name,
                "collection_name": self.source.collection_name,
            },
            tags=["manufacturing-agent", "rag", self.source.key],
        ) as run:
            answer = self._answer(question)
            end_langsmith_trace(run, {"answer": answer})
            return answer

    def _answer(self, question: str) -> str:
        if not self._settings.rag_use_llm_grader:
            sections = self._retriever.similarity_search_with_score(question, k=self._settings.rag_top_k)
            if not sections:
                return (
                    f"A confident answer could not be produced from {self.source.display_name} because "
                    "no relevant source sections were retrieved. Escalate to the approved plant document owner "
                    "or floor supervisor instead of relying on an unsupported answer."
                )
            return self._answer_generator.generate(question, self.source, sections)

        if self._grader is None:
            raise RuntimeError("RAG LLM grading is enabled, but no source grader is configured.")

        best_grade = SourceGrade(confidence=0.0, reasoning="No retrieval attempt was made.")
        max_attempts = max(1, min(self._settings.rag_max_retries, 3))

        for query in self._retry_queries(question)[:max_attempts]:
            sections = self._retriever.similarity_search_with_score(query, k=self._settings.rag_top_k)
            grade = self._grader.grade(question, self.source, sections)
            if grade.confidence > best_grade.confidence:
                best_grade = grade

            if grade.confidence >= self._settings.rag_confidence_threshold:
                return self._answer_generator.generate(question, self.source, sections)

        return (
            f"A confident answer could not be produced from {self.source.display_name} after "
            f"{max_attempts} retrieval attempts. The best source confidence was "
            f"{best_grade.confidence:.2f}. Reason: {best_grade.reasoning} "
            "Escalate to the approved plant document owner or floor supervisor instead of relying on an unsupported answer."
        )

    def _retry_queries(self, question: str) -> list[str]:
        return [
            question,
            f"{question}\nDocumentation source: {self.source.display_name}. Focus: {self.source.description}.",
            f"{question}\nRelated controlled vocabulary: {', '.join(self.source.retry_terms)}.",
        ]


class SupervisorAgent:
    def __init__(
        self,
        *,
        specialists: Mapping[str, SpecialistRagAgent] | None = None,
        model: str | None = None,
        settings: Settings | None = None,
        store: BaseStore | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._settings = settings
        supervisor_model = model or _settings_value(settings, "supervisor_model") or _settings_value(
            settings, "chat_model", "claude-sonnet-4-6"
        )
        specialist_model = _settings_value(settings, "specialist_model") or _settings_value(
            settings, "chat_model", "claude-sonnet-4-6"
        )
        self._specialists = specialists or build_default_specialists(model=specialist_model)
        self._store = store or build_postgres_memory_store(settings)

        def answer_safety_procedure_question(question: str) -> str:
            """Use for hazards, PPE, lockout/tagout, emergency response, spills, permits, and safe work practices."""
            return self._specialists["safety"].answer(question)

        def answer_maintenance_manual_question(question: str) -> str:
            """Use for equipment troubleshooting, inspections, service intervals, repairs, sensors, motors, pumps, and hydraulics."""
            return self._specialists["maintenance"].answer(question)

        def answer_quality_control_question(question: str) -> str:
            """Use for tolerances, sampling, defects, calibration, holds, disposition, and acceptance criteria."""
            return self._specialists["quality"].answer(question)

        self._agent = create_agent(
            model=supervisor_model,
            tools=[
                _return_direct_tool(answer_safety_procedure_question),
                _return_direct_tool(answer_maintenance_manual_question),
                _return_direct_tool(answer_quality_control_question),
                remember_memory,
                recall_memory,
            ],
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            middleware=build_agent_guardrails(run_limit=settings.supervisor_model_call_run_limit),
            checkpointer=InMemorySaver(),
            store=self._store,
        )

    def invoke(self, message: str, thread_id: str | None = None) -> tuple[str, str]:
        resolved_thread_id = thread_id or str(uuid4())
        with langsmith_trace(
            "supervisor_chat",
            settings=self._settings,
            inputs={"message": message, "thread_id": resolved_thread_id},
            metadata={"thread_id": resolved_thread_id},
            tags=["manufacturing-agent", "supervisor"],
        ) as run:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                {
                    "configurable": {"thread_id": resolved_thread_id},
                    "run_name": "supervisor_graph",
                    "tags": ["manufacturing-agent", "supervisor"],
                    "metadata": {"thread_id": resolved_thread_id},
                },
            )
            response = _message_text(result["messages"][-1])
            end_langsmith_trace(run, {"message": response, "thread_id": resolved_thread_id})
            return response, resolved_thread_id

    def stream(self, message: str, thread_id: str | None = None) -> tuple[Iterator[str], str]:
        resolved_thread_id = thread_id or str(uuid4())
        return self._stream_with_observability(message, resolved_thread_id), resolved_thread_id

    def _stream_with_observability(self, message: str, thread_id: str) -> Iterator[str]:
        with langsmith_trace(
            "supervisor_chat_stream",
            settings=self._settings,
            inputs={"message": message, "thread_id": thread_id},
            metadata={"thread_id": thread_id, "streaming": True},
            tags=["manufacturing-agent", "supervisor", "streaming"],
        ) as run:
            chunks: list[str] = []
            for chunk in self._stream_message(message, thread_id):
                chunks.append(chunk)
                yield chunk
            end_langsmith_trace(run, {"message": "".join(chunks), "thread_id": thread_id})

    def _stream_message(self, message: str, thread_id: str) -> Iterator[str]:
        final_message = ""
        emitted_token = False
        emitted_tool_call_ids: set[str] = set()
        tool_result_seen = False

        for mode, data in self._agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            {
                "configurable": {"thread_id": thread_id},
                "run_name": "supervisor_graph_stream",
                "tags": ["manufacturing-agent", "supervisor", "streaming"],
                "metadata": {"thread_id": thread_id},
            },
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                token_text = _stream_token_text(data, tool_result_seen=tool_result_seen)
                if token_text:
                    emitted_token = True
                    yield token_text
            elif mode == "values":
                tool_result_seen = tool_result_seen or _stream_has_tool_result(data)
                tool_result_text, tool_call_id = _stream_direct_tool_result_text(data)
                if tool_result_text and tool_call_id not in emitted_tool_call_ids:
                    emitted_tool_call_ids.add(tool_call_id)
                    emitted_token = True
                    yield tool_result_text
                final_message = _stream_final_message_text(data) or final_message

        if not emitted_token and final_message:
            yield final_message


def build_default_specialists(*, model: str | None = None) -> dict[str, SpecialistRagAgent]:
    settings = get_settings()
    resolved_model = model or _settings_value(settings, "specialist_model") or _settings_value(
        settings, "chat_model", "claude-sonnet-4-6"
    )
    return {
        source_key: SpecialistRagAgent(source, settings=settings, model=resolved_model)
        for source_key, source in RAG_SOURCES.items()
    }


def _return_direct_tool(func: Callable[..., str]) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=func.__name__,
        description=func.__doc__,
        return_direct=True,
    )


def _settings_value(settings: Any, name: str, default: str = "") -> str:
    value = getattr(settings, name, default)
    return value if isinstance(value, str) else default


def _format_sections(sections: Sequence[tuple[Document, float]]) -> str:
    formatted: list[str] = []
    for index, (document, score) in enumerate(sections, start=1):
        citation = document.metadata.get("citation", "unknown-source")
        section_title = document.metadata.get("section_title", "Untitled section")
        formatted.append(
            f"[{index}] citation={citation} vector_score={score:.4f}\n"
            f"section={section_title}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(formatted)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content)


def _stream_token_text(data: Any, *, tool_result_seen: bool) -> str:
    message, metadata = _stream_message_parts(data)

    if not tool_result_seen:
        return ""
    if not isinstance(message, AIMessageChunk):
        return ""
    if metadata.get("langgraph_node") != "model":
        return ""

    return _message_text(message)


def _stream_final_message_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""

    messages = data.get("messages")
    if isinstance(messages, Sequence) and messages:
        latest_message = messages[-1]
        if isinstance(latest_message, AIMessage):
            return _message_text(latest_message)
    return ""


def _stream_direct_tool_result_text(data: Any) -> tuple[str, str]:
    if not isinstance(data, dict):
        return "", ""

    messages = data.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        return "", ""

    latest_message = messages[-1]
    if not isinstance(latest_message, ToolMessage):
        return "", ""
    if latest_message.name not in SPECIALIST_TOOL_NAMES:
        return "", ""

    tool_call_id = latest_message.tool_call_id or latest_message.id or _message_text(latest_message)
    return _message_text(latest_message), tool_call_id


def _stream_has_tool_result(data: Any) -> bool:
    if not isinstance(data, dict):
        return False

    messages = data.get("messages")
    return isinstance(messages, Sequence) and bool(messages) and isinstance(messages[-1], ToolMessage)


def _stream_message_parts(data: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(data, (tuple, list)) and data:
        message = data[0]
        metadata = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}
        return message, metadata

    return data, {}
