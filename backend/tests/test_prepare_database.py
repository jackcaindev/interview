from __future__ import annotations

from app.config import Settings
from app.prepare_database import prepare_database, psycopg_connection_string


def test_psycopg_connection_string_strips_sqlalchemy_driver() -> None:
    assert (
        psycopg_connection_string("postgresql+psycopg://postgres:postgres@localhost/db")
        == "postgresql://postgres:postgres@localhost/db"
    )
    assert psycopg_connection_string("postgresql://postgres:postgres@localhost/db") == (
        "postgresql://postgres:postgres@localhost/db"
    )


def test_prepare_database_creates_vector_extension(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str) -> None:
            calls.append(("execute", sql))

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            calls.append(("cursor", None))
            return FakeCursor()

    def fake_connect(database_url: str, *, autocommit: bool) -> FakeConnection:
        calls.append(("connect", (database_url, autocommit)))
        return FakeConnection()

    monkeypatch.setattr("app.prepare_database.psycopg.connect", fake_connect)

    prepare_database(Settings(DATABASE_URL="postgresql+psycopg://user:password@db/app"))

    assert calls == [
        ("connect", ("postgresql://user:password@db/app", True)),
        ("cursor", None),
        ("execute", "CREATE EXTENSION IF NOT EXISTS vector"),
    ]
