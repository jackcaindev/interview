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
- Backend readiness: `http://localhost:8000/ready`
- Postgres: `localhost:${POSTGRES_PORT:-5432}`
- Redis: `localhost:${REDIS_PORT:-6379}`

The Postgres service uses pgvector and initializes `CREATE EXTENSION IF NOT EXISTS vector;` for RAG retrieval and long-term memory search.

The frontend container builds static assets and serves them with Nginx. In Docker Compose, Nginx proxies `/api/*` requests to the backend service.

## Memory

Short-term memory remains thread-scoped through the existing LangGraph `InMemorySaver` checkpointer. Long-term memory uses a Postgres-backed LangGraph store under the shared namespace `("manufacturing-help-desk", "memories")`.

The supervisor can call `remember_memory` for durable workspace context and `recall_memory` for relevant memories across conversations. These memories are not authoritative for controlled plant-document facts; safety, maintenance, and quality answers still come from the specialist RAG tools.

Set `MEMORY_EMBEDDING_DIMENSIONS=1536` to match the default `text-embedding-3-small` embedding size.

## Caching

Specialist RAG answers are cached by source, normalized question, source-document fingerprint, model, embedding model, and RAG settings. Supervisor chat turns are not cached because they depend on thread state and long-term memory.

Set `REDIS_URL=redis://localhost:6379/0` to use Redis. If Redis is configured but unavailable, the backend also writes through to a per-process in-memory fallback cache. If `REDIS_URL` is empty, only the per-process cache is used. Set `RAG_CACHE_TTL_SECONDS=0` to disable caching. `/ready` checks Redis when `REDIS_URL` is configured.

`RAG_USE_LLM_GRADER=false` is the default fast path. Turning it on adds a separate LLM source-confidence check before answer generation, which can improve strictness but increases response latency.

The router uses deterministic rules for clear safety, maintenance, and quality questions, then falls back to a structured LLM router for ambiguous wording. Set `ROUTER_MODEL` to use a cheaper or faster routing model; otherwise it falls back to `SUPERVISOR_MODEL` and then `CHAT_MODEL`.

## RAG Sources

Seed source documents live in `backend/knowledge/`. Each `##` section is ingested as one chunk with citation metadata. Run ingestion after Postgres is available:

```bash
cd backend
uv run python -m app.ingest_sources --reset
```

The ingestion uses OpenAI embeddings with `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` and writes one PGVector collection per documentation source.

## Production Deployment

Render deployment is defined in `render.yaml` with these managed resources:

- `manufacturing-agent-api`: Docker web service for `backend/`
- `manufacturing-agent-web`: static site for `frontend/`
- `manufacturing-agent-db`: Render Postgres 16 with private-only network access
- `manufacturing-agent-cache`: Render Key Value cache with private-only network access

The backend pre-deploy command runs:

```bash
uv run python -m app.prepare_database && uv run python -m app.ingest_sources --reset
```

This enables `pgvector` with `CREATE EXTENSION IF NOT EXISTS vector` and ingests the seed plant documents before the service starts serving traffic.

Backend environment variables:

```bash
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
HELP_DESK_ACCESS_TOKEN=<shared demo access code>
DATABASE_URL=<from manufacturing-agent-db>
REDIS_URL=<from manufacturing-agent-cache>
CHAT_MODEL=claude-sonnet-4-6
ROUTER_MODEL=
CORS_ALLOWED_ORIGINS=https://manufacturing-agent-web.onrender.com
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

Frontend environment variables:

```bash
VITE_API_BASE_URL=https://manufacturing-agent-api.onrender.com
```

`VITE_API_BASE_URL` and `CORS_ALLOWED_ORIGINS` must be updated together if Render assigns different public hostnames because of a naming collision. When `VITE_API_BASE_URL` is set, the browser calls the hosted backend directly. When it is blank, local development uses the Vite `/api` proxy.

Required secrets are `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `HELP_DESK_ACCESS_TOKEN`. The access token protects the public chat endpoints from casual abuse; the frontend prompts for this code and sends it to the backend as a bearer token. LangSmith is optional; set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` only when tracing should be enabled. The local `.env` file is ignored by git, but rotate any local keys if they have been copied, logged, or shared outside this machine.

For an interview demo, generate a short-lived random code, for example:

```bash
openssl rand -base64 24
```

Give the interviewer `https://manufacturing-agent-web.onrender.com` and the access code. Rotate `HELP_DESK_ACCESS_TOKEN` or suspend the Render services after the interview window.

Pre-submit checks:

```bash
npm run build:frontend
UV_CACHE_DIR=/tmp/uv-cache npm run test:backend
docker compose config --quiet
curl https://manufacturing-agent-api.onrender.com/health
curl https://manufacturing-agent-api.onrender.com/ready
```

Then open the frontend URL and ask a safety, maintenance, or quality question. A production-ready answer should include citations and a source-confidence score.
