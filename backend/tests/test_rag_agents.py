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
from app.rag_sources import RAG_SOURCES, load_source_documents


class FakeSettings:
    chat_model = "fake-chat-model"
    specialist_model = ""
    rag_top_k = 2
    rag_confidence_threshold = 0.72
    rag_max_retries = 9
    rag_use_llm_grader = False


class StrictFakeSettings(FakeSettings):
    rag_use_llm_grader = True


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


def test_load_source_documents_chunks_by_markdown_section() -> None:
    documents = load_source_documents(RAG_SOURCES["safety"])

    assert len(documents) == 4
    assert documents[0].metadata["section_id"] == "SP-101"
    assert documents[0].metadata["citation"] == "safety_procedures.md#SP-101"
    assert "Lockout/Tagout" in documents[0].page_content


def test_specialist_default_fast_path_skips_grader_and_retries() -> None:
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
    assert answer.endswith("[safety_procedures.md#SP-102]")


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
    assert "best source confidence was 0.30" in answer


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
    assert answer.endswith("[safety_procedures.md#SP-102]")


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
