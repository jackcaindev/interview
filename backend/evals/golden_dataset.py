from __future__ import annotations

from typing import Any

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError


DATASET_NAME = "manufacturing-supervisor-golden"
DATASET_DESCRIPTION = (
    "Golden cases for the manufacturing supervisor agent covering source routing, "
    "grounding, and out-of-domain refusal behavior."
)

GOLDEN_CASES: list[dict[str, dict[str, Any]]] = [
    {
        "inputs": {
            "question": "A hydraulic line is leaking beside press 4. What should the floor supervisor do first?",
        },
        "outputs": {
            "case_id": "perfect_safety_hydraulic_leak",
            "case_type": "perfect_match",
            "expected_source": "safety",
            "expected_citations": ["safety_procedures.md#SP-102"],
            "required_terms": [
                ["stop nearby traffic", "stop traffic"],
                ["visible boundary", "boundary"],
                ["safety glasses"],
                ["cut-resistant gloves"],
                ["slip-resistant footwear"],
            ],
            "expected_behavior": "answer_from_source",
        },
    },
    {
        "inputs": {
            "question": (
                "The hydraulic press lost pressure and there is fluid near it. "
                "Can maintenance inspect the source now?"
            ),
        },
        "outputs": {
            "case_id": "ambiguous_hydraulic_press_leak",
            "case_type": "ambiguous",
            "expected_source": "safety",
            "expected_citations": ["safety_procedures.md#SP-102"],
            "required_terms": [
                ["maintenance may inspect"],
                ["pressure is relieved", "relieve stored pressure", "pressure relieved"],
                ["lockout"],
            ],
            "expected_behavior": "answer_from_source",
        },
    },
    {
        "inputs": {
            "question": "A pump motor keeps tripping thermal overload. What should maintenance check?",
        },
        "outputs": {
            "case_id": "maintenance_pump_motor_overheating",
            "case_type": "perfect_match",
            "expected_source": "maintenance",
            "expected_citations": ["maintenance_manuals.md#MM-203"],
            "required_terms": [
                ["cooling fan"],
                ["ventilation slots"],
                ["voltage balance", "supply voltage"],
                ["current draw"],
                ["do not repeatedly reset", "do not reset"],
            ],
            "expected_behavior": "answer_from_source",
        },
    },
    {
        "inputs": {
            "question": "A torque-critical fastener failed verification during stable production. What happens to the assemblies?",
        },
        "outputs": {
            "case_id": "quality_torque_verification_failure",
            "case_type": "perfect_match",
            "expected_source": "quality",
            "expected_citations": ["quality_control_standards.md#QC-303"],
            "required_terms": [
                ["quality hold"],
                ["since the last passing check"],
                ["containment plan"],
                ["do not average", "must independently meet"],
            ],
            "expected_behavior": "answer_from_source",
        },
    },
    {
        "inputs": {
            "question": "What playlist should we use for the company picnic?",
        },
        "outputs": {
            "case_id": "irrelevant_company_picnic_playlist",
            "case_type": "irrelevant",
            "expected_source": None,
            "expected_citations": [],
            "required_terms": [],
            "expected_behavior": "refuse_no_source",
        },
    },
]


def ensure_dataset(client: Client | None = None, *, dataset_name: str = DATASET_NAME):
    client = client or Client()
    dataset = _get_or_create_dataset(client, dataset_name)
    if not any(client.list_examples(dataset_id=dataset.id, limit=1)):
        client.create_examples(dataset_id=dataset.id, examples=GOLDEN_CASES)
    return dataset


def _get_or_create_dataset(client: Client, dataset_name: str):
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except LangSmithNotFoundError:
        return client.create_dataset(dataset_name=dataset_name, description=DATASET_DESCRIPTION)


def main() -> None:
    dataset = ensure_dataset()
    print(f"Ready dataset: {dataset.name}")


if __name__ == "__main__":
    main()
