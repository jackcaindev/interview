from app.config import Settings
from app.vector_store import build_embeddings


def test_build_embeddings_uses_openai_text_embedding_3_small() -> None:
    settings = Settings(OPENAI_API_KEY="test-key")

    embeddings = build_embeddings(settings)

    assert type(embeddings).__name__ == "OpenAIEmbeddings"
    assert embeddings.model == "text-embedding-3-small"
