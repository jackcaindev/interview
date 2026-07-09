# Manufacturing Agent Backend

FastAPI backend exposing `POST /chat` for a LangChain supervisor agent that routes to safety, maintenance, and quality-control RAG specialists. Short-term conversation state uses the existing LangGraph checkpointer; cross-thread long-term memory uses the Postgres-backed LangGraph store.

```bash
uv sync
uv run python -m app.prepare_database
uv run python -m app.ingest_sources --reset
uv run uvicorn app.main:app --reload
uv run pytest
```

`app.prepare_database` is idempotent and enables the Postgres `vector` extension required by PGVector retrieval and long-term memory. Render runs it in the backend pre-deploy command before source ingestion.

Set `HELP_DESK_ACCESS_TOKEN` in production to require a shared access code for `/common-questions`, `/chat`, and `/chat/stream`. Leave it blank for local development without an access prompt.

## Cache

Specialist RAG answers are cached by source, normalized question, source-document fingerprint, model, embedding model, and RAG settings. Set `REDIS_URL` to use Redis, leave it empty for a per-process cache, or set `RAG_CACHE_TTL_SECONDS=0` to disable caching. If Redis is configured but unavailable, answers also write through to a per-process fallback cache.

`RAG_USE_LLM_GRADER=false` is the default fast path. Set it to `true` only when you want an extra LLM confidence check before answer generation and can accept the added latency.

The chat router handles clear safety, maintenance, and quality questions with deterministic routing, then uses a structured LLM router for ambiguous wording. Set `ROUTER_MODEL` to tune that router independently from the supervisor and specialist answer models.

## LangSmith observability and evals

Tracing is disabled by default. To send supervisor and specialist traces to LangSmith, set:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-langsmith-api-key>
export LANGSMITH_PROJECT=manufacturing-supervisor
```

Run the golden evaluation after the database is running, source documents are ingested, and model provider keys are configured:

```bash
uv run python -m evals.golden_dataset
uv run python -m evals.run_langsmith_eval
```

The golden dataset contains five cases: safety, maintenance, quality, one ambiguous hydraulic-press question, and one irrelevant picnic-playlist question. Custom evaluators score source routing, citation grounding, required content, and refusal behavior for irrelevant inputs.
