from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_FILES = {
    "safety": "safety_procedures.md",
    "maintenance": "maintenance_manuals.md",
    "quality": "quality_control_standards.md",
}

REFUSAL_TERMS = (
    "can't answer",
    "cannot answer",
    "could not answer",
    "not covered",
    "outside",
    "not in the manufacturing",
    "approved plant document",
    "controlled documentation",
    "floor supervisor",
)

CITATION_PATTERN = re.compile(
    r"(safety_procedures|maintenance_manuals|quality_control_standards)\.md#[A-Z]+-\d+",
    re.IGNORECASE,
)


def source_routing_evaluator(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    expected_source = reference_outputs.get("expected_source")
    answer = _answer_text(outputs)

    if expected_source is None:
        has_controlled_source = _infer_source(answer) is not None
        return _score(
            "source_routing",
            0.0 if has_controlled_source else 1.0,
            "Irrelevant request should not cite or route to controlled plant documents.",
        )

    inferred_source = _infer_source(answer)
    score = 1.0 if inferred_source == expected_source else 0.0
    return _score(
        "source_routing",
        score,
        f"Expected {expected_source}; inferred {inferred_source or 'none'} for question: {inputs.get('question')}",
    )


def citation_evaluator(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    del inputs
    answer = _answer_text(outputs)
    expected_citations = reference_outputs.get("expected_citations") or []

    if not expected_citations:
        score = 1.0 if not CITATION_PATTERN.search(answer) else 0.0
        return _score("citation_grounding", score, "No controlled citation expected for this case.")

    matched = [citation for citation in expected_citations if citation.lower() in answer.lower()]
    score = len(matched) / len(expected_citations)
    return _score(
        "citation_grounding",
        score,
        f"Matched {len(matched)} of {len(expected_citations)} expected citations.",
    )


def required_content_evaluator(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    del inputs
    answer = _normalize(_answer_text(outputs))
    required_terms = reference_outputs.get("required_terms") or []
    if not required_terms:
        return _score("required_content", 1.0, "No required content terms for this case.")

    matched = 0
    missing: list[Sequence[str]] = []
    for alternatives in required_terms:
        if any(_normalize(term) in answer for term in alternatives):
            matched += 1
        else:
            missing.append(alternatives)

    score = matched / len(required_terms)
    return _score(
        "required_content",
        score,
        f"Matched {matched} of {len(required_terms)} required content groups. Missing: {missing}",
    )


def irrelevant_refusal_evaluator(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    del inputs
    if reference_outputs.get("expected_behavior") != "refuse_no_source":
        return _score("irrelevant_refusal", 1.0, "Relevant case; refusal not required.")

    answer = _normalize(_answer_text(outputs))
    has_refusal = any(term in answer for term in REFUSAL_TERMS)
    has_citation = CITATION_PATTERN.search(answer) is not None
    score = 1.0 if has_refusal and not has_citation else 0.0
    return _score(
        "irrelevant_refusal",
        score,
        "Irrelevant requests should be refused without controlled-document citations.",
    )


EVALUATORS = [
    source_routing_evaluator,
    citation_evaluator,
    required_content_evaluator,
    irrelevant_refusal_evaluator,
]


def _answer_text(outputs: Mapping[str, Any]) -> str:
    value = outputs.get("answer") or outputs.get("message") or outputs.get("output") or ""
    return str(value)


def _infer_source(answer: str) -> str | None:
    normalized = answer.lower()
    for source, filename in SOURCE_FILES.items():
        if filename in normalized:
            return source
    return None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _score(key: str, score: float, comment: str) -> dict[str, Any]:
    return {"key": key, "score": score, "comment": comment}
