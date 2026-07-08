# Manufacturing Agentic Workflow

Full-stack vertical slice for a manufacturing supervisor agent. The React UI posts chat messages to FastAPI, and FastAPI invokes a LangChain supervisor agent backed by LangGraph in-memory thread persistence for short-term conversation state and a Postgres LangGraph store for cross-thread long-term memory. The supervisor routes questions to specialist RAG agents for safety procedures, maintenance manuals, and quality control standards.

## Structure

- `frontend/`: React, TypeScript, Vite, Tailwind, shadcn/ui, pnpm
- `backend/`: FastAPI, uv, LangChain, LangGraph
- `docker/postgres/`: pgvector initialization

## Environment

Copy `env.example` to `.env` if needed. This workspace already expects `.env` at the repository root.

## Local Development

```bash
cd backend
uv sync
uv run python -m app.ingest_sources --reset
uv run uvicorn app.main:app --reload
```

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:5173`. The frontend posts to `/api/chat`, and Vite proxies that request to `http://localhost:8000/chat` by default.

## Docker

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/health`
- Postgres: `localhost:${POSTGRES_PORT:-5432}`

The Postgres service uses pgvector and initializes `CREATE EXTENSION IF NOT EXISTS vector;` for RAG retrieval and long-term memory search.

## Memory

Short-term memory remains thread-scoped through the existing LangGraph `InMemorySaver` checkpointer. Long-term memory uses a Postgres-backed LangGraph store under the shared namespace `("manufacturing-help-desk", "memories")`.

The supervisor can call `remember_memory` for durable workspace context and `recall_memory` for relevant memories across conversations. These memories are not authoritative for controlled plant-document facts; safety, maintenance, and quality answers still come from the specialist RAG tools.

Set `MEMORY_EMBEDDING_DIMENSIONS=1536` to match the default `text-embedding-3-small` embedding size.

## RAG Sources

Seed source documents live in `backend/knowledge/`. Each `##` section is ingested as one chunk with citation metadata. Run ingestion after Postgres is available:

```bash
cd backend
uv run python -m app.ingest_sources --reset
```

The ingestion uses OpenAI embeddings with `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` and writes one PGVector collection per documentation source.
