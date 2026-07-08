import json
from collections.abc import Iterator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent import ChatAgent, SupervisorAgent
from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(title="Manufacturing Supervisor Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: SupervisorAgent | None = None


def get_chat_agent() -> ChatAgent:
    global _agent
    if _agent is None:
        _agent = SupervisorAgent()
    return _agent


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {
        "status": "ok",
        "database_url_configured": str(bool(settings.database_url)).lower(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: ChatAgent = Depends(get_chat_agent)) -> ChatResponse:
    message, thread_id = agent.invoke(request.message, request.thread_id)
    return ChatResponse(message=message, thread_id=thread_id)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, agent: ChatAgent = Depends(get_chat_agent)) -> StreamingResponse:
    return StreamingResponse(
        _chat_stream_events(request, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _chat_stream_events(request: ChatRequest, agent: ChatAgent) -> Iterator[str]:
    try:
        chunks, thread_id = agent.stream(request.message, request.thread_id)
        yield _sse_event("thread", {"thread_id": thread_id})

        for chunk in chunks:
            if chunk:
                yield _sse_event("token", {"text": chunk})

        yield _sse_event("done", {"thread_id": thread_id})
    except Exception as exc:
        yield _sse_event("error", {"message": str(exc) or "Unable to stream chat response."})


def _sse_event(event: str, data: dict[str, str]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
