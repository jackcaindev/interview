from __future__ import annotations

import psycopg

from app.config import Settings, get_settings


def psycopg_connection_string(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


def prepare_database(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    database_url = psycopg_connection_string(settings.database_url)

    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")


def main() -> None:
    prepare_database()


if __name__ == "__main__":
    main()
