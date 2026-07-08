import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.main import app, get_chat_agent


class FakeAgent:
    def invoke(self, message: str, thread_id: str | None = None) -> tuple[str, str]:
        return f"supervisor received: {message}", thread_id or "generated-thread"

    def stream(self, message: str, thread_id: str | None = None) -> tuple[Iterator[str], str]:
        return iter(["supervisor ", f"received: {message}"]), thread_id or "generated-thread"


def client() -> TestClient:
    app.dependency_overrides[get_chat_agent] = lambda: FakeAgent()
    return TestClient(app)


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_chat_generates_thread_id_when_missing() -> None:
    response = client().post("/chat", json={"message": "Which procedure covers a hydraulic leak?"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "supervisor received: Which procedure covers a hydraulic leak?",
        "thread_id": "generated-thread",
    }


def test_chat_reuses_provided_thread_id() -> None:
    response = client().post(
        "/chat",
        json={"message": "What about cleanup?", "thread_id": "floor-a-shift-1"},
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == "floor-a-shift-1"


def test_chat_rejects_blank_message() -> None:
    response = client().post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_rejects_missing_message() -> None:
    response = client().post("/chat", json={})

    assert response.status_code == 422


def test_chat_stream_generates_thread_id_when_missing() -> None:
    response = client().post("/chat/stream", json={"message": "Which procedure covers a hydraulic leak?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert parse_sse(response.text) == [
        ("thread", {"thread_id": "generated-thread"}),
        ("token", {"text": "supervisor "}),
        ("token", {"text": "received: Which procedure covers a hydraulic leak?"}),
        ("done", {"thread_id": "generated-thread"}),
    ]


def test_chat_stream_reuses_provided_thread_id() -> None:
    response = client().post(
        "/chat/stream",
        json={"message": "What about cleanup?", "thread_id": "floor-a-shift-1"},
    )

    assert response.status_code == 200
    assert parse_sse(response.text)[0] == ("thread", {"thread_id": "floor-a-shift-1"})
    assert parse_sse(response.text)[-1] == ("done", {"thread_id": "floor-a-shift-1"})


def test_chat_stream_rejects_blank_message() -> None:
    response = client().post("/chat/stream", json={"message": "   "})

    assert response.status_code == 422


def parse_sse(payload: str) -> list[tuple[str, dict[str, str]]]:
    events: list[tuple[str, dict[str, str]]] = []

    for block in payload.strip().split("\n\n"):
        event_name = "message"
        data_lines: list[str] = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        events.append((event_name, json.loads("\n".join(data_lines))))

    return events
