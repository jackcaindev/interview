from evals.evaluators import (
    citation_evaluator,
    irrelevant_refusal_evaluator,
    required_content_evaluator,
    source_routing_evaluator,
)
from evals.golden_dataset import GOLDEN_CASES


def test_golden_dataset_has_required_scenario_coverage() -> None:
    case_types = {case["outputs"]["case_type"] for case in GOLDEN_CASES}

    assert 3 <= len(GOLDEN_CASES) <= 6
    assert "perfect_match" in case_types
    assert "ambiguous" in case_types
    assert "irrelevant" in case_types


def test_source_and_citation_evaluators_pass_grounded_safety_answer() -> None:
    reference = GOLDEN_CASES[0]["outputs"]
    outputs = {
        "answer": (
            "Stop nearby traffic, establish a visible boundary, and have operators use "
            "safety glasses, cut-resistant gloves, and slip-resistant footwear. "
            "[safety_procedures.md#SP-102]"
        )
    }

    assert source_routing_evaluator({}, outputs, reference)["score"] == 1.0
    assert citation_evaluator({}, outputs, reference)["score"] == 1.0
    assert required_content_evaluator({}, outputs, reference)["score"] == 1.0


def test_required_content_evaluator_scores_partial_answers() -> None:
    reference = GOLDEN_CASES[2]["outputs"]
    outputs = {"answer": "Check the cooling fan and current draw. [maintenance_manuals.md#MM-203]"}

    score = required_content_evaluator({}, outputs, reference)["score"]

    assert 0.0 < score < 1.0


def test_irrelevant_refusal_evaluator_requires_refusal_without_citation() -> None:
    reference = GOLDEN_CASES[-1]["outputs"]

    passing = {"answer": "I cannot answer that from the controlled manufacturing documentation."}
    failing = {"answer": "Use a high-energy playlist. [quality_control_standards.md#QC-301]"}

    assert irrelevant_refusal_evaluator({}, passing, reference)["score"] == 1.0
    assert irrelevant_refusal_evaluator({}, failing, reference)["score"] == 0.0
