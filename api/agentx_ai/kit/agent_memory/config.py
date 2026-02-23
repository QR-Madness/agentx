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
    embedding_provider: str = "local"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"  # for OpenAI
    embedding_dimensions: int = 768  # 768 for local (nomic), 1536 for OpenAI
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
    max_query_length: int = 10000  # Maximum query length in chars
    max_results_per_conversation: int = 3  # Diversity: max results from same conversation

    # Default retrieval weights (used when no per-request overrides provided)
    retrieval_weight_episodic: float = 0.3
    retrieval_weight_semantic_facts: float = 0.25
    retrieval_weight_semantic_entities: float = 0.2
    retrieval_weight_procedural: float = 0.15
    retrieval_weight_recency: float = 0.1

    # Multi-channel retrieval
    channel_active_boost: float = 1.2  # Boost factor for active channel results vs _global

    # Cross-encoder reranking (optional, higher accuracy but slower)
    cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval caching
    retrieval_cache_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 60
    retrieval_cache_key_prefix: str = "retrieval_cache"

    # ===========================================
    # Consolidation LLM Settings
    # ===========================================
    # All consolidation stages can be configured independently.
    # Use "lmstudio" for local/free, "anthropic" for quality.
    # Model IDs should match your provider's model names.

    # --- Extraction (main fact/entity extraction) ---
    extraction_enabled: bool = True
    extraction_provider: str = "lmstudio"  # "lmstudio", "anthropic", "openai"
    extraction_model: str = "lmstudio-community/llama-3.2-1b-instruct"
    extraction_temperature: float = 0.3
    extraction_max_tokens: int = 2000
    extraction_timeout: float = 30.0
    extraction_condense_facts: bool = True  # Condense verbose statements to atomic facts

    # --- Relevance Filter (skip "thanks", "ok" turns) ---
    relevance_filter_enabled: bool = True
    relevance_filter_provider: str = "lmstudio"
    relevance_filter_model: str = "lmstudio-community/llama-3.2-1b-instruct"
    relevance_filter_temperature: float = 0.1  # Low temp for consistent YES/NO
    relevance_filter_max_tokens: int = 10  # Only need YES or NO

    # --- Contradiction Detection (check new facts vs existing) ---
    contradiction_detection_enabled: bool = False  # Off by default until stable
    contradiction_provider: str = "lmstudio"
    contradiction_model: str = "lmstudio-community/llama-3.2-1b-instruct"
    contradiction_temperature: float = 0.2
    contradiction_max_tokens: int = 500

    # --- User Correction Handling (detect "actually...", "no I meant...") ---
    correction_detection_enabled: bool = False  # Off by default until stable
    correction_provider: str = "lmstudio"
    correction_model: str = "lmstudio-community/llama-3.2-1b-instruct"
    correction_temperature: float = 0.2
    correction_max_tokens: int = 500

    # --- Entity Linking (match facts to existing entities) ---
    entity_linking_enabled: bool = True
    entity_linking_similarity_threshold: float = 0.75  # Min embedding similarity
    entity_linking_use_llm_disambiguation: bool = False  # Use LLM when ambiguous
    entity_linking_provider: str = "lmstudio"
    entity_linking_model: str = "lmstudio-community/llama-3.2-1b-instruct"

    # Entity types to recognize
    entity_types: list = [
        "Person", "Organization", "Location", "Concept",
        "Technology", "Product", "Event"
    ]

    # Relationship types to recognize
    relationship_types: list = [
        "works_at", "knows", "uses", "prefers", "created",
        "located_in", "part_of", "related_to", "mentioned_with"
    ]

    # Audit logging settings
    audit_log_level: str = "writes"  # off | writes | reads | verbose
    audit_retention_days: int = 30
    audit_partition_ahead_days: int = 7
    audit_sample_rate: float = 1.0  # 0.0-1.0, sampling rate for high-volume reads

    # Cross-channel promotion thresholds
    promotion_min_confidence: float = 0.85
    promotion_min_access_count: int = 5
    promotion_min_conversations: int = 2

    # Consolidation job intervals (minutes)
    job_consolidate_interval: int = 15
    job_patterns_interval: int = 60
    job_decay_interval: int = 1440  # 24 hours
    job_cleanup_interval: int = 1440  # 24 hours
    job_audit_partitions_interval: int = 1440  # 24 hours
    job_promote_interval: int = 60  # Same as patterns
    job_entity_linking_interval: int = 30  # Run after consolidation

    # Worker health settings
    worker_heartbeat_interval: int = 30  # seconds
    worker_heartbeat_ttl: int = 90  # seconds (3x interval)

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
