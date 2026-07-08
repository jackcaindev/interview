from __future__ import annotations

from app.config import Settings


def pgvector_connection_string(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def build_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": settings.openai_embedding_model}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return OpenAIEmbeddings(**kwargs)


def build_pgvector_store(settings: Settings, collection_name: str):
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=build_embeddings(settings),
        collection_name=collection_name,
        connection=pgvector_connection_string(settings.database_url),
        use_jsonb=True,
    )
