from __future__ import annotations

from typing import Any

from langsmith import Client

from app.agent import SupervisorAgent
from evals.evaluators import EVALUATORS
from evals.golden_dataset import DATASET_NAME, ensure_dataset


_agent: SupervisorAgent | None = None


def target(inputs: dict[str, Any]) -> dict[str, str]:
    question = str(inputs["question"])
    thread_id = inputs.get("thread_id")
    answer, resolved_thread_id = _get_agent().invoke(
        question,
        str(thread_id) if thread_id is not None else None,
    )
    return {"answer": answer, "thread_id": resolved_thread_id}


def _get_agent() -> SupervisorAgent:
    global _agent
    if _agent is None:
        _agent = SupervisorAgent()
    return _agent


def main() -> None:
    client = Client()
    dataset = ensure_dataset(client, dataset_name=DATASET_NAME)
    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=EVALUATORS,
        experiment_prefix="manufacturing-supervisor",
        description="Golden evaluation for source routing, grounding, and irrelevant-input handling.",
        metadata={"dataset": DATASET_NAME},
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
