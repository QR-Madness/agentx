"""Configuration settings for the agent memory system."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuration settings for the agent memory system."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    # PostgreSQL
    postgres_uri: str = "postgresql://agent:changeme@localhost:5432/agent_memory"

    # Redis
    redis_uri: str = "redis://localhost:6379"

    # Connection timeouts (seconds)
    connection_timeout: int = 5

    # Embeddings
    embedding_provider: str = "openai"  # or "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: str = ""

    # Local embedding model (if using local)
    local_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Memory settings
    episodic_retention_days: int = 90
    fact_confidence_threshold: float = 0.7
    salience_decay_rate: float = 0.95  # Daily decay multiplier
    max_working_memory_items: int = 50

    # Retrieval settings
    default_top_k: int = 10
    reranking_enabled: bool = True

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
