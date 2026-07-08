import pytest
from langgraph.store.base import MatchCondition

from app.memory_store import (
    _matches_namespace,
    _text_for_index,
    _truncate_namespace,
    _vector_literal,
    memory_table_schema_sql,
    postgres_memory_connection_string,
)


def test_postgres_memory_connection_string_uses_psycopg_driver_url() -> None:
    assert (
        postgres_memory_connection_string("postgresql+psycopg://postgres:postgres@localhost/db")
        == "postgresql://postgres:postgres@localhost/db"
    )
    assert (
        postgres_memory_connection_string("postgresql://postgres:postgres@localhost/db")
        == "postgresql://postgres:postgres@localhost/db"
    )


def test_memory_table_schema_sql_uses_vector_dimension_and_rejects_bad_identifier() -> None:
    sql = memory_table_schema_sql("memory_items", embedding_dimensions=1536)

    assert "CREATE TABLE IF NOT EXISTS memory_items" in sql
    assert "embedding vector(1536)" in sql
    assert "PRIMARY KEY (namespace, key)" in sql

    with pytest.raises(ValueError):
        memory_table_schema_sql("memory-items", embedding_dimensions=1536)


def test_memory_formatting_helpers_select_index_text_and_vector_literal() -> None:
    value = {"memory": "Use metric units.", "topic": "line 2", "ignored": "not indexed"}

    assert _text_for_index(value, ["memory", "topic"]) == "Use metric units.\nline 2"
    assert _text_for_index(value, False) == ""
    assert _vector_literal([0, 1.25, -2]) == "[0.0,1.25,-2.0]"


def test_namespace_matching_and_truncation() -> None:
    namespace = ("manufacturing-help-desk", "memories", "line-2")

    assert _matches_namespace(
        namespace,
        (MatchCondition(match_type="prefix", path=("manufacturing-help-desk",)),),
    )
    assert _matches_namespace(
        namespace,
        (MatchCondition(match_type="suffix", path=("line-2",)),),
    )
    assert not _matches_namespace(
        namespace,
        (MatchCondition(match_type="prefix", path=("other",)),),
    )
    assert _truncate_namespace(namespace, 2) == ("manufacturing-help-desk", "memories")
