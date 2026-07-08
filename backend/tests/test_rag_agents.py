from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agent import (
    SourceGrade,
    SpecialistRagAgent,
    _stream_direct_tool_result_text,
    _stream_final_message_text,
    _stream_has_tool_result,
    _stream_token_text,
)
from app.cache import rag_answer_cache_key
from app.rag_sources import RAG_SOURCES, load_source_documents


class FakeSettings:
    openai_embedding_model = "fake-embedding-model"
    chat_model = "fake-chat-model"
    specialist_model = ""
    rag_top_k = 2
    rag_confidence_threshold = 0.72
    rag_max_retries = 9
    rag_use_llm_grader = False
    rag_cache_ttl_seconds = 900
    rag_cache_namespace = "test"
    rag_cache_max_question_chars = 2000
    redis_url = ""
    redis_timeout_seconds = 0.5


class StrictFakeSettings(FakeSettings):
    rag_use_llm_grader = True


class ShortQuestionCacheSettings(FakeSettings):
    rag_cache_max_question_chars = 10


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        self.queries.append(query)
        return [
            (
                Document(
                    page_content="SP-102 Hydraulic Leak Response\n\nStop traffic and isolate pooled fluid.",
                    metadata={
                        "citation": "safety_procedures.md#SP-102",
                        "section_title": "SP-102 Hydraulic Leak Response",
                    },
                ),
                0.12,
            )
        ]


class SequenceGrader:
    def __init__(self, confidences: list[float]) -> None:
        self.confidences = confidences
        self.calls = 0

    def grade(self, question, source, sections):
        confidence = self.confidences[min(self.calls, len(self.confidences) - 1)]
        self.calls += 1
        return SourceGrade(confidence=confidence, reasoning=f"grade {confidence}")


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, source, sections):
        self.calls += 1
        return "Isolate the hydraulic leak area. [safety_procedures.md#SP-102]"


class StreamingFakeGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = 0

    def stream_generate(self, question, source, sections):
        self.stream_calls += 1
        yield "Isolate "
        yield "the hydraulic leak area. [safety_procedures.md#SP-102]"


class DictCache:
    def __init__(self) -> None:
        self.values: dict[str, dict] = {}
        self.gets = 0
        self.sets = 0

    def get_json(self, key: str) -> dict | None:
        self.gets += 1
        return self.values.get(key)

    def set_json(self, key: str, value: dict, *, ttl_seconds: int) -> None:
        self.sets += 1
        self.values[key] = value


def test_load_source_documents_chunks_by_markdown_section() -> None:
    documents = load_source_documents(RAG_SOURCES["safety"])

    assert len(documents) == 4
    assert documents[0].metadata["section_id"] == "SP-101"
    assert documents[0].metadata["citation"] == "safety_procedures.md#SP-101"
    assert "Lockout/Tagout" in documents[0].page_content


def test_specialist_default_fast_path_skips_grader_and_uses_retrieval_confidence() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.1, 0.9])
    generator = FakeGenerator()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=FakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
    )

    answer = agent.answer("What should I do about a hydraulic leak?")

    assert len(retriever.queries) == 1
    assert grader.calls == 0
    assert generator.calls == 1
    assert "[safety_procedures.md#SP-102]" in answer
    assert answer.endswith("**Source confidence:** 88%")


def test_specialist_strict_path_uses_grader_confidence() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.86])
    generator = FakeGenerator()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=StrictFakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
    )

    answer = agent.answer("What should I do about a hydraulic leak?")

    assert len(retriever.queries) == 1
    assert grader.calls == 1
    assert generator.calls == 1
    assert "[safety_procedures.md#SP-102]" in answer
    assert answer.endswith("**Source confidence:** 86%")


def test_specialist_reuses_cached_answer_for_identical_question() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.9])
    generator = FakeGenerator()
    cache = DictCache()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=FakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
        cache=cache,
    )

    first_answer = agent.answer("What should I do about a hydraulic leak?")
    second_answer = agent.answer("  what SHOULD I do about a hydraulic leak?  ")

    assert first_answer == second_answer
    assert len(retriever.queries) == 1
    assert grader.calls == 0
    assert generator.calls == 1
    assert cache.gets == 2
    assert cache.sets == 1


def test_specialist_streams_cached_answer_immediately() -> None:
    retriever = FakeRetriever()
    generator = StreamingFakeGenerator()
    cache = DictCache()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=FakeSettings(),
        retriever=retriever,
        answer_generator=generator,
        cache=cache,
    )
    cache_key = rag_answer_cache_key(
        settings=FakeSettings(),
        source=RAG_SOURCES["safety"],
        question="What should I do about a hydraulic leak?",
        model="fake-chat-model",
    )
    assert cache_key is not None
    cache.values[cache_key] = {"answer": "cached answer"}

    chunks = list(agent.stream_answer("What should I do about a hydraulic leak?"))

    assert chunks == ["cached answer"]
    assert len(retriever.queries) == 0
    assert generator.stream_calls == 0


def test_specialist_streams_uncached_answer_and_stores_completed_text() -> None:
    retriever = FakeRetriever()
    generator = StreamingFakeGenerator()
    cache = DictCache()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=FakeSettings(),
        retriever=retriever,
        answer_generator=generator,
        cache=cache,
    )

    chunks = list(agent.stream_answer("What should I do about a hydraulic leak?"))
    answer = "".join(chunks)

    assert chunks[0] == "Isolate "
    assert "[safety_procedures.md#SP-102]" in answer
    assert answer.endswith("**Source confidence:** 88%")
    assert len(retriever.queries) == 1
    assert generator.stream_calls == 1
    assert cache.sets == 1
    assert next(iter(cache.values.values()))["answer"] == answer


def test_specialist_skips_cache_for_oversized_question() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.9])
    generator = FakeGenerator()
    cache = DictCache()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=ShortQuestionCacheSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
        cache=cache,
    )

    agent.answer("What should I do about a hydraulic leak?")

    assert len(retriever.queries) == 1
    assert grader.calls == 0
    assert generator.calls == 1
    assert cache.gets == 0
    assert cache.sets == 0


def test_specialist_caps_retries_at_three_and_refuses_low_confidence() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.1, 0.2, 0.3])
    generator = FakeGenerator()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=StrictFakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
    )

    answer = agent.answer("What should I do about a hydraulic leak?")

    assert len(retriever.queries) == 3
    assert grader.calls == 3
    assert generator.calls == 0
    assert "A confident answer could not be produced" in answer
    assert answer.endswith("**Source confidence:** 30%")


def test_specialist_generates_grounded_answer_after_ambiguous_supported_grade() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.74])
    generator = FakeGenerator()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=StrictFakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
    )

    answer = agent.answer("Can maintenance inspect the hydraulic leak now?")

    assert len(retriever.queries) == 1
    assert grader.calls == 1
    assert generator.calls == 1
    assert "[safety_procedures.md#SP-102]" in answer
    assert answer.endswith("**Source confidence:** 74%")


def test_specialist_generates_grounded_answer_after_confident_grade() -> None:
    retriever = FakeRetriever()
    grader = SequenceGrader([0.2, 0.9])
    generator = FakeGenerator()
    agent = SpecialistRagAgent(
        RAG_SOURCES["safety"],
        settings=StrictFakeSettings(),
        retriever=retriever,
        grader=grader,
        answer_generator=generator,
    )

    answer = agent.answer("What should I do about a hydraulic leak?")

    assert len(retriever.queries) == 2
    assert generator.calls == 1
    assert "[safety_procedures.md#SP-102]" in answer
    assert answer.endswith("**Source confidence:** 90%")


def test_stream_token_filter_emits_only_final_model_chunks_after_tool_result() -> None:
    assert _stream_token_text(
        (AIMessageChunk(content="routing text"), {"langgraph_node": "model"}),
        tool_result_seen=False,
    ) == ""
    assert _stream_token_text(
        (ToolMessage(content="tool answer", tool_call_id="tool-1"), {"langgraph_node": "tools"}),
        tool_result_seen=True,
    ) == ""
    assert _stream_token_text(
        (AIMessageChunk(content="final text"), {"langgraph_node": "model"}),
        tool_result_seen=True,
    ) == "final text"


def test_stream_direct_tool_result_text_emits_specialist_tool_output() -> None:
    tool_message = ToolMessage(
        content="Isolate the hydraulic leak area. [safety_procedures.md#SP-102]",
        name="answer_safety_procedure_question",
        tool_call_id="tool-1",
    )

    assert _stream_direct_tool_result_text({"messages": [tool_message]}) == (
        "Isolate the hydraulic leak area. [safety_procedures.md#SP-102]",
        "tool-1",
    )


def test_stream_direct_tool_result_text_ignores_memory_tool_output() -> None:
    tool_message = ToolMessage(
        content="No relevant workspace memories were found.",
        name="recall_memory",
        tool_call_id="tool-1",
    )

    assert _stream_direct_tool_result_text({"messages": [tool_message]}) == ("", "")


def test_stream_final_message_fallback_uses_only_assistant_messages() -> None:
    tool_message = ToolMessage(content="tool answer", tool_call_id="tool-1")

    assert _stream_has_tool_result({"messages": [tool_message]})
    assert _stream_final_message_text({"messages": [tool_message]}) == ""
    assert _stream_final_message_text({"messages": [AIMessage(content="final answer")]}) == "final answer"
