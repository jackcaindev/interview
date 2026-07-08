import json
from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agent import ChatAgent, SupervisorAgent
from app.config import Settings, get_settings
from app.schemas import ChatRequest, ChatResponse, CommonQuestion, CommonQuestionsResponse


def _parse_cors_origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


def _psycopg_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


def _check_redis(redis_url: str, timeout_seconds: float) -> None:
    if not redis_url:
        return

    import redis

    client = redis.Redis.from_url(
        redis_url,
        socket_connect_timeout=timeout_seconds,
        socket_timeout=timeout_seconds,
    )
    client.ping()


app = FastAPI(title="Manufacturing Supervisor Agent API")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: SupervisorAgent | None = None

COMMON_QUESTIONS = (
    CommonQuestion(
        id="hydraulic-leak-response",
        category="Safety",
        label="Hydraulic leak response",
        question="What should I do if I find a hydraulic leak on the floor?",
    ),
    CommonQuestion(
        id="pump-motor-overheating",
        category="Maintenance",
        label="Pump motor overheating",
        question="What should I check when a pump motor is overheating?",
    ),
    CommonQuestion(
        id="tool-change-sampling",
        category="Quality",
        label="Tool-change sampling checks",
        question="What sampling checks are required after a tool change?",
    ),
    CommonQuestion(
        id="hot-work-fire-watch",
        category="Safety",
        label="Hot work and fire watch",
        question="When do I need a hot work permit and fire watch?",
    ),
    CommonQuestion(
        id="nonconforming-material-hold",
        category="Quality",
        label="Nonconforming material hold",
        question="What information is required when placing material on quality hold?",
    ),
)


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


@app.get("/ready")
def ready(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    try:
        import psycopg

        with psycopg.connect(_psycopg_database_url(settings.database_url), connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not ready.") from exc

    try:
        _check_redis(settings.redis_url, settings.redis_timeout_seconds)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Redis cache is not ready.") from exc

    return {"status": "ready"}


@app.get("/common-questions", response_model=CommonQuestionsResponse)
def common_questions() -> CommonQuestionsResponse:
    return CommonQuestionsResponse(questions=list(COMMON_QUESTIONS))


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
