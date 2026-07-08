from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from app.config import Settings
from app.rag_sources import RagSource


class JsonCache(Protocol):
    def get_json(self, key: str) -> dict[str, Any] | None:
        """Return a cached JSON object, or None on miss/unavailable cache."""

    def set_json(self, key: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        """Store a JSON object for ttl_seconds, ignoring transient cache failures."""


@dataclass(frozen=True)
class NoopJsonCache:
    def get_json(self, key: str) -> dict[str, Any] | None:
        return None

    def set_json(self, key: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        return None


class InMemoryJsonCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = Lock()

    def get_json(self, key: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            return json.loads(json.dumps(value))

    def set_json(self, key: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._items[key] = (expires_at, json.loads(json.dumps(dict(value))))


class RedisJsonCache:
    def __init__(self, *, redis_url: str, timeout_seconds: float) -> None:
        import redis

        self._redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._redis.get(key)
        except Exception:
            return None

        if not raw:
            return None

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            self._delete(key)
            return None

        return value if isinstance(value, dict) else None

    def set_json(self, key: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        try:
            self._redis.setex(key, ttl_seconds, json.dumps(dict(value), sort_keys=True))
        except Exception:
            return

    def ping(self) -> None:
        self._redis.ping()

    def _delete(self, key: str) -> None:
        try:
            self._redis.delete(key)
        except Exception:
            return


class FallbackJsonCache:
    def __init__(self, primary: JsonCache, fallback: JsonCache) -> None:
        self._primary = primary
        self._fallback = fallback

    def get_json(self, key: str) -> dict[str, Any] | None:
        return self._primary.get_json(key) or self._fallback.get_json(key)

    def set_json(self, key: str, value: Mapping[str, Any], *, ttl_seconds: int) -> None:
        self._primary.set_json(key, value, ttl_seconds=ttl_seconds)
        self._fallback.set_json(key, value, ttl_seconds=ttl_seconds)


def build_json_cache(settings: Settings) -> JsonCache:
    if settings.rag_cache_ttl_seconds <= 0:
        return NoopJsonCache()
    if settings.redis_url:
        return FallbackJsonCache(
            RedisJsonCache(
                redis_url=settings.redis_url,
                timeout_seconds=settings.redis_timeout_seconds,
            ),
            InMemoryJsonCache(),
        )
    return InMemoryJsonCache()


def rag_answer_cache_key(
    *,
    settings: Settings,
    source: RagSource,
    question: str,
    model: str,
) -> str | None:
    normalized_question = _normalize_question(question)
    if len(normalized_question) > settings.rag_cache_max_question_chars:
        return None

    payload = {
        "embedding_model": settings.openai_embedding_model,
        "grader_enabled": settings.rag_use_llm_grader,
        "max_retries": settings.rag_max_retries,
        "model": model,
        "question": normalized_question,
        "source_collection": source.collection_name,
        "source_fingerprint": _file_fingerprint(source.path),
        "source_key": source.key,
        "threshold": settings.rag_confidence_threshold,
        "top_k": settings.rag_top_k,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    namespace = settings.rag_cache_namespace.strip() or "manufacturing-agent"
    return f"{namespace}:rag-answer:{digest}"


def _normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _file_fingerprint(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"
