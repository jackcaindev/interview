from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv("../.env")
load_dotenv()


class Settings(BaseSettings):
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    langsmith_tracing: str = Field(default="false", alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="manufacturing-supervisor", alias="LANGSMITH_PROJECT")
    help_desk_access_token: str = Field(default="", alias="HELP_DESK_ACCESS_TOKEN")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="manufacturing_agents", alias="POSTGRES_DB")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/manufacturing_agents",
        alias="DATABASE_URL",
    )
    chat_model: str = Field(default="claude-sonnet-4-6", alias="CHAT_MODEL")
    supervisor_model: str = Field(default="", alias="SUPERVISOR_MODEL")
    router_model: str = Field(default="", alias="ROUTER_MODEL")
    specialist_model: str = Field(default="", alias="SPECIALIST_MODEL")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_confidence_threshold: float = Field(default=0.72, alias="RAG_CONFIDENCE_THRESHOLD")
    rag_max_retries: int = Field(default=3, alias="RAG_MAX_RETRIES")
    rag_use_llm_grader: bool = Field(default=False, alias="RAG_USE_LLM_GRADER")
    memory_embedding_dimensions: int = Field(default=1536, alias="MEMORY_EMBEDDING_DIMENSIONS")
    supervisor_model_call_run_limit: int = Field(default=4, alias="SUPERVISOR_MODEL_CALL_RUN_LIMIT")
    specialist_model_call_run_limit: int = Field(default=1, alias="SPECIALIST_MODEL_CALL_RUN_LIMIT")
    redis_url: str = Field(default="", alias="REDIS_URL")
    redis_timeout_seconds: float = Field(default=0.5, alias="REDIS_TIMEOUT_SECONDS")
    rag_cache_ttl_seconds: int = Field(default=900, alias="RAG_CACHE_TTL_SECONDS")
    rag_cache_namespace: str = Field(default="manufacturing-agent", alias="RAG_CACHE_NAMESPACE")
    rag_cache_max_question_chars: int = Field(default=2000, alias="RAG_CACHE_MAX_QUESTION_CHARS")
    vite_api_proxy_target: str = Field(default="http://localhost:8000", alias="VITE_API_PROXY_TARGET")
    cors_allowed_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        alias="CORS_ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
