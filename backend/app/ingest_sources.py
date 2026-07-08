from __future__ import annotations

import argparse

from app.config import get_settings
from app.rag_sources import RAG_SOURCES, RagSource, load_source_documents
from app.vector_store import build_pgvector_store


def ingest_source(source: RagSource, *, reset_collection: bool = False) -> int:
    settings = get_settings()
    documents = load_source_documents(source)
    vector_store = build_pgvector_store(settings, source.collection_name)

    if reset_collection and hasattr(vector_store, "delete_collection"):
        vector_store.delete_collection()
        vector_store = build_pgvector_store(settings, source.collection_name)

    ids = [f"{source.key}:{doc.metadata['section_id']}" for doc in documents]
    vector_store.add_documents(documents, ids=ids)
    return len(documents)


def ingest_all(*, reset_collection: bool = False) -> dict[str, int]:
    return {
        source.key: ingest_source(source, reset_collection=reset_collection)
        for source in RAG_SOURCES.values()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest manufacturing RAG source sections into PGVector.")
    parser.add_argument(
        "--source",
        choices=sorted(RAG_SOURCES),
        help="Ingest only one source. Defaults to all sources.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete each target PGVector collection before inserting source sections.",
    )
    args = parser.parse_args()

    if args.source:
        counts = {args.source: ingest_source(RAG_SOURCES[args.source], reset_collection=args.reset)}
    else:
        counts = ingest_all(reset_collection=args.reset)

    for source_key, count in counts.items():
        print(f"{source_key}: ingested {count} sections")


if __name__ == "__main__":
    main()
