from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from app.config import Settings, get_settings
from app.vector_store import build_embeddings


MEMORY_NAMESPACE = ("manufacturing-help-desk", "memories")
MEMORY_TABLE_NAME = "langgraph_memory_store"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Return one embedding vector for text."""


def build_postgres_memory_store(settings: Settings | None = None) -> PostgresMemoryStore:
    settings = settings or get_settings()
    store = PostgresMemoryStore(
        database_url=settings.database_url,
        embeddings=build_embeddings(settings),
        embedding_dimensions=settings.memory_embedding_dimensions,
    )
    store.setup()
    return store


def postgres_memory_connection_string(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


def memory_table_schema_sql(table_name: str = MEMORY_TABLE_NAME, *, embedding_dimensions: int = 1536) -> str:
    table = _validate_identifier(table_name)
    dimensions = _validate_embedding_dimensions(embedding_dimensions)
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    namespace TEXT[] NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL,
    embedding vector({dimensions}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, key)
)
""".strip()


class PostgresMemoryStore(BaseStore):
    """LangGraph BaseStore backed by Postgres JSONB plus pgvector embeddings."""

    def __init__(
        self,
        *,
        database_url: str,
        embeddings: QueryEmbedder,
        embedding_dimensions: int = 1536,
        table_name: str = MEMORY_TABLE_NAME,
    ) -> None:
        self._database_url = postgres_memory_connection_string(database_url)
        self._embeddings = embeddings
        self._embedding_dimensions = _validate_embedding_dimensions(embedding_dimensions)
        self._table_name = _validate_identifier(table_name)

    def setup(self) -> None:
        with self._connect(autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(memory_table_schema_sql(self._table_name, embedding_dimensions=self._embedding_dimensions))
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_namespace_gin_idx "
                f"ON {self._table_name} USING gin(namespace)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_value_gin_idx "
                f"ON {self._table_name} USING gin(value)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table_name}_embedding_hnsw_idx "
                f"ON {self._table_name} USING hnsw (embedding vector_cosine_ops) "
                "WHERE embedding IS NOT NULL"
            )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        with self._connect() as conn:
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._get(conn, op))
                elif isinstance(op, PutOp):
                    self._put(conn, op)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    results.append(self._search(conn, op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._list_namespaces(conn, op))
                else:
                    raise TypeError(f"Unsupported store operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    def _connect(self, *, autocommit: bool = False):
        return psycopg.connect(self._database_url, autocommit=autocommit, row_factory=dict_row)

    def _get(self, conn: psycopg.Connection, op: GetOp) -> Item | None:
        row = conn.execute(
            f"""
            SELECT namespace, key, value, created_at, updated_at
            FROM {self._table_name}
            WHERE namespace = %s::text[] AND key = %s
            """,
            (list(op.namespace), op.key),
        ).fetchone()
        return _row_to_item(row) if row else None

    def _put(self, conn: psycopg.Connection, op: PutOp) -> None:
        if op.value is None:
            conn.execute(
                f"DELETE FROM {self._table_name} WHERE namespace = %s::text[] AND key = %s",
                (list(op.namespace), op.key),
            )
            return

        text_to_embed = _text_for_index(op.value, op.index)
        embedding = self._embed(text_to_embed) if text_to_embed else None
        conn.execute(
            f"""
            INSERT INTO {self._table_name} (namespace, key, value, embedding)
            VALUES (%s::text[], %s, %s, %s::vector)
            ON CONFLICT (namespace, key)
            DO UPDATE SET
                value = EXCLUDED.value,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """,
            (list(op.namespace), op.key, Jsonb(op.value), _vector_literal(embedding) if embedding else None),
        )

    def _search(self, conn: psycopg.Connection, op: SearchOp) -> list[SearchItem]:
        where_sql, params = _search_where_clause(op.namespace_prefix, op.filter)

        if op.query:
            embedding = self._embed(op.query)
            vector = _vector_literal(embedding)
            rows = conn.execute(
                f"""
                SELECT namespace, key, value, created_at, updated_at,
                    1 - (embedding <=> %s::vector) AS score
                FROM {self._table_name}
                WHERE {where_sql} AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector, updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (vector, *params, vector, op.limit, op.offset),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT namespace, key, value, created_at, updated_at, NULL::float AS score
                FROM {self._table_name}
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (*params, op.limit, op.offset),
            ).fetchall()

        return [_row_to_search_item(row) for row in rows]

    def _list_namespaces(self, conn: psycopg.Connection, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        rows = conn.execute(f"SELECT DISTINCT namespace FROM {self._table_name}").fetchall()
        namespaces = sorted(
            {
                _truncate_namespace(tuple(row["namespace"]), op.max_depth)
                for row in rows
                if _matches_namespace(tuple(row["namespace"]), op.match_conditions)
            }
        )
        return namespaces[op.offset : op.offset + op.limit]

    def _embed(self, text: str) -> list[float]:
        embedding = self._embeddings.embed_query(text)
        if len(embedding) != self._embedding_dimensions:
            raise ValueError(
                f"Expected embedding dimension {self._embedding_dimensions}, got {len(embedding)}."
            )
        return embedding


def _search_where_clause(namespace_prefix: tuple[str, ...], filter: dict[str, Any] | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if namespace_prefix:
        clauses.append("namespace[1:%s] = %s::text[]")
        params.extend([len(namespace_prefix), list(namespace_prefix)])

    exact_filter = _exact_jsonb_filter(filter)
    if exact_filter:
        clauses.append("value @> %s")
        params.append(Jsonb(exact_filter))

    return " AND ".join(clauses) if clauses else "TRUE", params


def _exact_jsonb_filter(filter: dict[str, Any] | None) -> dict[str, Any]:
    if not filter:
        return {}

    exact: dict[str, Any] = {}
    for key, value in filter.items():
        if isinstance(value, dict):
            if set(value) == {"$eq"}:
                exact[key] = value["$eq"]
                continue
            raise NotImplementedError("PostgresMemoryStore supports exact JSONB filters and $eq filters.")
        exact[key] = value
    return exact


def _row_to_item(row: Any) -> Item:
    return Item(
        namespace=tuple(row["namespace"]),
        key=row["key"],
        value=row["value"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_search_item(row: Any) -> SearchItem:
    return SearchItem(
        namespace=tuple(row["namespace"]),
        key=row["key"],
        value=row["value"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=row["score"],
    )


def _text_for_index(value: dict[str, Any], index: bool | list[str] | None) -> str:
    if index is False:
        return ""

    if index is None:
        return json.dumps(value, sort_keys=True)

    parts = [_string_value_for_path(value, path) for path in index]
    return "\n".join(part for part in parts if part)


def _string_value_for_path(value: dict[str, Any], path: str) -> str:
    if path == "$":
        return json.dumps(value, sort_keys=True)

    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return ""
        current = current[part]

    if isinstance(current, str):
        return current
    if current is None:
        return ""
    return json.dumps(current, sort_keys=True)


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _matches_namespace(
    namespace: tuple[str, ...],
    match_conditions: tuple[MatchCondition, ...] | None,
) -> bool:
    if not match_conditions:
        return True

    return all(
        _namespace_starts_with(namespace, condition.path)
        if condition.match_type == "prefix"
        else _namespace_ends_with(namespace, condition.path)
        for condition in match_conditions
    )


def _namespace_starts_with(namespace: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return namespace[: len(prefix)] == prefix


def _namespace_ends_with(namespace: tuple[str, ...], suffix: tuple[str, ...]) -> bool:
    return not suffix or namespace[-len(suffix) :] == suffix


def _truncate_namespace(namespace: tuple[str, ...], max_depth: int | None) -> tuple[str, ...]:
    if max_depth is None:
        return namespace
    return namespace[:max_depth]


def _validate_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return identifier


def _validate_embedding_dimensions(dimensions: int) -> int:
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be positive.")
    return dimensions
