from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app.cache import JsonCache, build_json_cache, rag_answer_cache_key
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

SOURCE_CONFIDENCE_LABEL = "Source confidence"

DIRECT_ROUTE_KEYWORDS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "safety": (
        ("hydraulic leak", 5),
        ("lockout", 4),
        ("tagout", 4),
        ("loto", 4),
        ("hot work", 4),
        ("fire watch", 4),
        ("spill", 3),
        ("ppe", 3),
        ("permit", 2),
        ("hazard", 2),
        ("emergency", 2),
        ("absorbent", 2),
        ("cleanup", 2),
        ("injury", 2),
        ("leak", 1),
    ),
    "maintenance": (
        ("pump motor", 5),
        ("hydraulic press", 5),
        ("photoeye", 4),
        ("sensor", 3),
        ("overheat", 3),
        ("thermal overload", 3),
        ("preventive maintenance", 3),
        ("belt tracking", 3),
        ("conveyor belt", 3),
        ("troubleshoot", 3),
        ("motor", 2),
        ("pump", 2),
        ("bearing", 2),
        ("repair", 2),
        ("inspection", 1),
    ),
    "quality": (
        ("quality hold", 5),
        ("nonconforming", 5),
        ("out of tolerance", 4),
        ("sampling", 4),
        ("tool change", 4),
        ("changeover", 3),
        ("torque", 3),
        ("calibration", 3),
        ("defect", 3),
        ("acceptance", 2),
        ("gauge", 2),
        ("dimension", 2),
        ("surface", 2),
        ("burr", 2),
        ("hold", 2),
    ),
}

MEMORY_ROUTE_TERMS = (
    "remember",
    "recall",
    "memory",
    "preference",
    "what did i say",
    "what did we discuss",
)


RouteKey = Literal["safety", "maintenance", "quality", "memory", "ambiguous"]
DOMAIN_ROUTE_KEYS = frozenset({"safety", "maintenance", "quality"})


class SourceGrade(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class RouteDecision(BaseModel):
    route: RouteKey
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


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


class QuestionRouter:
    def __init__(self, *, settings: Settings | None = None, model: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = model or _settings_value(self._settings, "router_model") or _settings_value(
            self._settings,
            "supervisor_model",
        ) or _settings_value(self._settings, "chat_model", "claude-sonnet-4-6")
        self._agent: Any | None = None

    def route(self, message: str) -> RouteDecision:
        deterministic = _deterministic_route_decision(message)
        if deterministic.route != "ambiguous":
            return deterministic
        return self._llm_route(message)

    def _llm_route(self, message: str) -> RouteDecision:
        try:
            result = self._compiled_agent().invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Route this manufacturing help-desk question to exactly one route.\n"
                                f"Question: {message}"
                            ),
                        }
                    ]
                }
            )
        except Exception:
            return RouteDecision(route="ambiguous", confidence=0.0, reason="Router model failed.")

        structured = result.get("structured_response") if isinstance(result, dict) else None
        if isinstance(structured, RouteDecision):
            return structured
        if isinstance(structured, dict):
            try:
                return RouteDecision.model_validate(structured)
            except Exception:
                return RouteDecision(route="ambiguous", confidence=0.0, reason="Router model returned invalid output.")
        return RouteDecision(route="ambiguous", confidence=0.0, reason="Router model returned no route decision.")

    def _compiled_agent(self) -> Any:
        if self._agent is None:
            self._agent = create_agent(
                model=self._model,
                tools=[],
                middleware=build_agent_guardrails(run_limit=1),
                response_format=RouteDecision,
                system_prompt="""You are a manufacturing help-desk router.

Choose exactly one route:
- safety: hazards, PPE, lockout/tagout, emergency response, spills, permits, safe work practices
- maintenance: troubleshooting, inspections, service intervals, repairs, sensors, motors, pumps, hydraulics
- quality: tolerances, sampling, defects, calibration, quality holds, disposition, acceptance criteria
- memory: the user asks to remember, recall, or use prior conversation/workspace context
- ambiguous: the question is unclear, out of scope, or cannot be safely routed to one source

Return a route decision only. Do not answer the question.
""",
            )
        return self._agent


class LLMSourceGrader:
    def __init__(self, *, model: str, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._agent = create_agent(
            model=model,
            tools=[],
            middleware=build_agent_guardrails(run_limit=settings.specialist_model_call_run_limit),
            response_format=SourceGrade,
            system_prompt="""You grade retrieved manufacturing documentation excerpts.

Return a confidence score from 0 to 1 for whether the excerpts are relevant, specific, and sufficient to answer the question without fabrication.

Use this rubric:
- 0.90-1.00: The excerpts directly answer the question and include the required operational details.
- 0.85-0.89: The excerpts clearly support the answer with only minor non-critical omissions.
- 0.72-0.84: The excerpts support the core answer but are incomplete or ambiguous.
- 0.40-0.71: The excerpts are topical but not sufficient for a confident answer.
- 0.00-0.39: The excerpts come from the wrong source, no relevant excerpts were retrieved, or an answer would require fabrication.

Prefer scores of at least 0.85 for exact-match source-backed questions. Prefer low confidence when the excerpts are only topically related, omit key constraints, or come from the wrong documentation source.
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

    def stream_generate(self, question: str, source: RagSource, sections: Sequence[tuple[Document, float]]) -> Iterator[str]:
        final_message = ""
        emitted_token = False

        for mode, data in self._agent.stream(
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
            },
            stream_mode=["messages", "values"],
        ):
            if mode == "messages":
                token_text = _stream_model_token_text(data)
                if token_text:
                    emitted_token = True
                    yield token_text
            elif mode == "values":
                final_message = _stream_final_message_text(data) or final_message

        if not emitted_token and final_message:
            yield final_message


class SpecialistRagAgent:
    def __init__(
        self,
        source: RagSource,
        *,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        grader: SourceGrader | None = None,
        answer_generator: AnswerGenerator | None = None,
        cache: JsonCache | None = None,
        model: str | None = None,
    ) -> None:
        self.source = source
        self._settings = settings or get_settings()
        resolved_model = model or _settings_value(self._settings, "specialist_model") or _settings_value(
            self._settings, "chat_model", "claude-sonnet-4-6"
        )
        self._model = resolved_model
        self._retriever = retriever or build_pgvector_store(self._settings, source.collection_name)
        self._grader = grader
        if self._settings.rag_use_llm_grader and self._grader is None:
            self._grader = LLMSourceGrader(model=resolved_model, settings=self._settings)
        self._answer_generator = answer_generator or LLMSourcedAnswerGenerator(
            source=source,
            model=resolved_model,
            settings=self._settings,
        )
        self._cache = cache or build_json_cache(self._settings)

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

    def stream_answer(self, question: str) -> Iterator[str]:
        with langsmith_trace(
            "specialist_rag_answer_stream",
            settings=self._settings,
            inputs={"question": question},
            metadata={
                "source_key": self.source.key,
                "source_name": self.source.display_name,
                "collection_name": self.source.collection_name,
                "streaming": True,
            },
            tags=["manufacturing-agent", "rag", self.source.key, "streaming"],
        ) as run:
            chunks: list[str] = []
            for chunk in self._stream_answer(question):
                chunks.append(chunk)
                yield chunk
            end_langsmith_trace(run, {"answer": "".join(chunks)})

    def _answer(self, question: str) -> str:
        cache_key = rag_answer_cache_key(
            settings=self._settings,
            source=self.source,
            question=question,
            model=self._model,
        )
        if cache_key:
            cached = self._cache.get_json(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("answer"), str):
                return cached["answer"]

        answer = self._answer_uncached(question)
        if cache_key:
            self._cache.set_json(
                cache_key,
                {"answer": answer},
                ttl_seconds=self._settings.rag_cache_ttl_seconds,
            )
        return answer

    def _stream_answer(self, question: str) -> Iterator[str]:
        cache_key = rag_answer_cache_key(
            settings=self._settings,
            source=self.source,
            question=question,
            model=self._model,
        )
        if cache_key:
            cached = self._cache.get_json(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("answer"), str):
                yield cached["answer"]
                return

        if self._settings.rag_use_llm_grader:
            answer = self._answer(question)
            yield answer
            return

        chunks: list[str] = []
        for chunk in self._stream_answer_uncached(question):
            chunks.append(chunk)
            yield chunk

        if cache_key:
            self._cache.set_json(
                cache_key,
                {"answer": "".join(chunks)},
                ttl_seconds=self._settings.rag_cache_ttl_seconds,
            )

    def _stream_answer_uncached(self, question: str) -> Iterator[str]:
        sections = self._retriever.similarity_search_with_score(question, k=self._settings.rag_top_k)
        if not sections:
            answer = _append_confidence_footer(
                f"A confident answer could not be produced from {self.source.display_name} because "
                "no relevant source sections were retrieved. Escalate to the approved plant document owner "
                "or floor supervisor instead of relying on an unsupported answer.",
                0.0,
            )
            yield answer
            return answer

        confidence = _retrieval_confidence(sections)
        chunks: list[str] = []
        for chunk in _stream_answer_generator(self._answer_generator, question, self.source, sections):
            chunks.append(chunk)
            yield chunk

        footer = f"\n\n**{SOURCE_CONFIDENCE_LABEL}:** {_confidence_percent(confidence)}%"
        chunks.append(footer)
        yield footer
        return "".join(chunks)

    def _answer_uncached(self, question: str) -> str:
        if not self._settings.rag_use_llm_grader:
            sections = self._retriever.similarity_search_with_score(question, k=self._settings.rag_top_k)
            if not sections:
                return _append_confidence_footer(
                    f"A confident answer could not be produced from {self.source.display_name} because "
                    "no relevant source sections were retrieved. Escalate to the approved plant document owner "
                    "or floor supervisor instead of relying on an unsupported answer.",
                    0.0,
                )
            return _append_confidence_footer(
                self._answer_generator.generate(question, self.source, sections),
                _retrieval_confidence(sections),
            )

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
                return _append_confidence_footer(
                    self._answer_generator.generate(question, self.source, sections),
                    grade.confidence,
                )

        return _append_confidence_footer(
            f"A confident answer could not be produced from {self.source.display_name} after "
            f"{max_attempts} retrieval attempts. Reason: {best_grade.reasoning} "
            "Escalate to the approved plant document owner or floor supervisor instead of relying on an unsupported answer.",
            best_grade.confidence,
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
        router: QuestionRouter | None = None,
        settings: Settings | None = None,
        store: BaseStore | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._settings = settings
        self._supervisor_model = model or _settings_value(settings, "supervisor_model") or _settings_value(
            settings, "chat_model", "claude-sonnet-4-6"
        )
        self._specialist_model = _settings_value(settings, "specialist_model") or _settings_value(
            settings, "chat_model", "claude-sonnet-4-6"
        )
        self._router = router or QuestionRouter(settings=settings)
        self._specialists = dict(specialists or {})
        self._store = store
        self._agent: Any | None = None

    def _build_agent(self) -> Any:
        def answer_safety_procedure_question(question: str) -> str:
            """Use for hazards, PPE, lockout/tagout, emergency response, spills, permits, and safe work practices."""
            return self._specialist("safety").answer(question)

        def answer_maintenance_manual_question(question: str) -> str:
            """Use for equipment troubleshooting, inspections, service intervals, repairs, sensors, motors, pumps, and hydraulics."""
            return self._specialist("maintenance").answer(question)

        def answer_quality_control_question(question: str) -> str:
            """Use for tolerances, sampling, defects, calibration, holds, disposition, and acceptance criteria."""
            return self._specialist("quality").answer(question)

        self._store = self._store or build_postgres_memory_store(self._settings)
        return create_agent(
            model=self._supervisor_model,
            tools=[
                _return_direct_tool(answer_safety_procedure_question),
                _return_direct_tool(answer_maintenance_manual_question),
                _return_direct_tool(answer_quality_control_question),
                remember_memory,
                recall_memory,
            ],
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            middleware=build_agent_guardrails(run_limit=self._settings.supervisor_model_call_run_limit),
            checkpointer=InMemorySaver(),
            store=self._store,
        )

    def _compiled_agent(self) -> Any:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def _specialist(self, source_key: str) -> SpecialistRagAgent:
        specialist = self._specialists.get(source_key)
        if specialist is None:
            specialist = SpecialistRagAgent(
                RAG_SOURCES[source_key],
                settings=self._settings,
                model=self._specialist_model,
            )
            self._specialists[source_key] = specialist
        return specialist

    def invoke(self, message: str, thread_id: str | None = None) -> tuple[str, str]:
        resolved_thread_id = thread_id or str(uuid4())
        route_decision = self._router.route(message)
        if route_decision.route in DOMAIN_ROUTE_KEYS:
            return self._specialist(route_decision.route).answer(message), resolved_thread_id

        with langsmith_trace(
            "supervisor_chat",
            settings=self._settings,
            inputs={"message": message, "thread_id": resolved_thread_id},
            metadata={"thread_id": resolved_thread_id},
            tags=["manufacturing-agent", "supervisor"],
        ) as run:
            result = self._compiled_agent().invoke(
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
        route_decision = self._router.route(message)
        if route_decision.route in DOMAIN_ROUTE_KEYS:
            return self._stream_direct_specialist(message, route_decision.route), resolved_thread_id

        return self._stream_with_observability(message, resolved_thread_id), resolved_thread_id

    def _stream_direct_specialist(self, message: str, source_key: str) -> Iterator[str]:
        yield from self._specialist(source_key).stream_answer(message)

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

        for mode, data in self._compiled_agent().stream(
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


def _deterministic_route_decision(message: str) -> RouteDecision:
    normalized = _normalize_route_text(message)
    if any(term in normalized for term in MEMORY_ROUTE_TERMS):
        return RouteDecision(route="memory", confidence=1.0, reason="Message includes memory intent.")

    scores = {
        source_key: sum(weight for term, weight in keywords if term in normalized)
        for source_key, keywords in DIRECT_ROUTE_KEYWORDS.items()
    }
    best_source, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < 2:
        return RouteDecision(route="ambiguous", confidence=0.0, reason="No deterministic route matched.")

    tied_sources = [source_key for source_key, score in scores.items() if score == best_score]
    if len(tied_sources) != 1:
        return RouteDecision(route="ambiguous", confidence=0.0, reason="Multiple deterministic routes tied.")
    return RouteDecision(
        route=best_source,
        confidence=min(1.0, best_score / 5),
        reason=f"Matched deterministic {best_source} terms.",
    )


def _direct_source_key(message: str) -> str | None:
    decision = _deterministic_route_decision(message)
    return decision.route if decision.route in DOMAIN_ROUTE_KEYS else None


def _normalize_route_text(message: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", message.casefold())).strip()


def _stream_answer_generator(
    generator: AnswerGenerator,
    question: str,
    source: RagSource,
    sections: Sequence[tuple[Document, float]],
) -> Iterator[str]:
    stream_generate = getattr(generator, "stream_generate", None)
    if callable(stream_generate):
        yield from stream_generate(question, source, sections)
        return
    yield generator.generate(question, source, sections)


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


def _append_confidence_footer(answer: str, confidence: float) -> str:
    return f"{answer.rstrip()}\n\n**{SOURCE_CONFIDENCE_LABEL}:** {_confidence_percent(confidence)}%"


def _confidence_percent(confidence: float) -> int:
    return round(_clamp_confidence(confidence) * 100)


def _retrieval_confidence(sections: Sequence[tuple[Document, float]]) -> float:
    if not sections:
        return 0.0

    return max(_score_to_confidence(score) for _, score in sections)


def _score_to_confidence(score: float) -> float:
    if score < 0:
        return 0.0
    if score <= 1:
        return _clamp_confidence(1 - score)
    return _clamp_confidence(1 / (1 + score))


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, confidence))


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


def _stream_model_token_text(data: Any) -> str:
    message, metadata = _stream_message_parts(data)

    if not isinstance(message, AIMessageChunk):
        return ""
    if metadata and metadata.get("langgraph_node") != "model":
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
