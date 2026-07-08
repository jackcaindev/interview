from typing import Any

from pydantic import BaseModel, Field, field_validator


CHAT_MESSAGE_MAX_LENGTH = 2000
THREAD_ID_MAX_LENGTH = 128
THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=CHAT_MESSAGE_MAX_LENGTH)
    thread_id: str | None = Field(default=None, max_length=THREAD_ID_MAX_LENGTH, pattern=THREAD_ID_PATTERN)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("thread_id", mode="before")
    @classmethod
    def normalize_thread_id(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        value = value.strip()
        return value or None


class ChatResponse(BaseModel):
    message: str
    thread_id: str


class CommonQuestion(BaseModel):
    id: str
    category: str
    label: str
    question: str


class CommonQuestionsResponse(BaseModel):
    questions: list[CommonQuestion]
