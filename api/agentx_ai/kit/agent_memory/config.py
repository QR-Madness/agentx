"""Configuration settings for the agent memory system."""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Path to the user-configurable settings file, anchored to the repo root (this
# file is api/agentx_ai/kit/agent_memory/config.py) so reads and writes hit the
# same file regardless of the process CWD. A unit test asserts it agrees with
# ConfigManager.CONFIG_PATH so the two settings stores can't drift apart.
MEMORY_SETTINGS_PATH = Path(__file__).resolve().parents[4] / "data" / "memory_settings.json"

# Pre-anchoring installs resolved the path against the CWD; if a process ran
# outside the repo root its overrides live in <cwd>/data/memory_settings.json.
_LEGACY_SETTINGS_PATH = Path("data/memory_settings.json")

# Settings cache TTL in seconds
SETTINGS_CACHE_TTL = 60.0


class Settings(BaseSettings):
    """Configuration settings for the agent memory system."""

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"  # noqa: S105 — dev default; real value from NEO4J_PASSWORD env

    # PostgreSQL
    postgres_uri: str = "postgresql://agent:changeme@localhost:5432/agent_memory"

    # Redis
    redis_uri: str = "redis://localhost:6379"

    # Connection timeouts (seconds)
    connection_timeout: int = 5

    # Connection pooling
    neo4j_max_connection_lifetime: int = 300  # seconds before a pooled conn is recycled
    postgres_pool_size: int = 10
    postgres_pool_max_overflow: int = 20

    # Embeddings
    embedding_provider: str = "local"  # "openai" or "local"
    embedding_model: str = "text-embedding-3-small"  # for OpenAI
    embedding_dimensions: int = 1024  # 1024 for local (bge-m3), 1536 for OpenAI
    openai_api_key: str = ""

    # Local embedding model (if using local)
    local_embedding_model: str = "BAAI/bge-m3"

    # Embedding request queue (serializes all embed calls through one worker so the
    # thread-unsafe local model and rate-limited remote provider can't collide/drop).
    embedding_queue_enabled: bool = True
    embedding_batch_max_size: int = 32      # max texts coalesced into one compute call
    embedding_batch_window_ms: int = 5      # window to coalesce concurrent requests
    embedding_request_timeout: float = 30.0  # max wait for a single embed() result
    embedding_queue_max_size: int = 1024    # bounded queue depth (backpressure)
    embedding_max_retries: int = 3          # retries on transient (remote) failures

    # Query-embedding cache (identical recall queries skip the queue entirely)
    embedding_cache_enabled: bool = True
    embedding_cache_max_size: int = 2048
    embedding_cache_ttl_seconds: float = 900.0

    # Workspaces / Document RAG (todo/backlog/workspaces.md). Bytes live in a
    # content-addressed blob store on disk; manifest in PG; vectors in pgvector.
    workspace_quota_bytes: int = 512 * 1024 * 1024      # per-workspace byte budget (512 MiB)
    workspace_max_file_bytes: int = 50 * 1024 * 1024    # per-file cap (50 MiB)
    workspace_chunk_size: int = 1000                    # chars per chunk for ingestion
    workspace_chunk_overlap: int = 150                  # char overlap between chunks
    workspace_allowed_extensions: list[str] = [         # parseable file types (v1)
        "pdf", "txt", "md", "markdown", "rst", "csv", "json", "yaml", "yml",
        "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "c", "h", "cpp",
        "sh", "sql", "html", "css", "toml", "ini", "log", "text",
    ]
    workspace_summary_model: str = ""                   # blank → feature_default_model / fallback
    workspace_agent_write_tools: bool = True            # advertise create_document/update_document
    workspace_agent_writable_extensions: list[str] = [  # types the *agent* may write (narrower than uploads)
        "md", "markdown", "txt",
    ]

    # Memory settings
    episodic_retention_days: int = 90
    fact_confidence_threshold: float = 0.7
    salience_decay_rate: float = 0.95  # Daily decay multiplier
    max_working_memory_items: int = 50
    user_recap_enabled: bool = True  # Refresh the cached cross-conversation recap during consolidation

    # Retrieval settings
    default_top_k: int = 10
    reranking_enabled: bool = True
    max_query_length: int = 10000  # Maximum query length in chars
    max_results_per_conversation: int = 3  # Diversity: max results from same conversation
    always_include_recent_turns: int = 3  # Always include last N turns from current conversation

    # Default retrieval weights (used when no per-request overrides provided)
    retrieval_weight_episodic: float = 0.3
    retrieval_weight_semantic_facts: float = 0.25
    retrieval_weight_semantic_entities: float = 0.2
    retrieval_weight_procedural: float = 0.15
    retrieval_weight_recency: float = 0.1

    # Multi-channel retrieval
    channel_active_boost: float = 1.2  # Boost factor for active channel results vs _global

    # Cross-encoder reranking — default ON since the §2.11 post-fusion stage
    # passed its eval gate (+20pp MRR, p95 ~450ms; see Memory-Roadmap §2.7).
    # Lazy-loaded on first recall; a load failure never breaks recall.
    # Upgrade candidate: BAAI/bge-reranker-v2-m3 (heavier, config-only).
    cross_encoder_enabled: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval caching
    retrieval_cache_enabled: bool = True
    retrieval_cache_ttl_seconds: int = 60
    retrieval_cache_key_prefix: str = "retrieval_cache"

    # ===========================================
    # RecallLayer Settings (Enhanced Retrieval)
    # ===========================================
    # Multiple retrieval techniques to bridge the semantic gap between
    # questions and stored facts. Enable/disable each technique independently.

    # --- Feature Toggles ---
    recall_enable_hybrid: bool = True  # BM25 + vector fusion (recommended)
    recall_enable_entity_centric: bool = True  # Graph traversal (recommended)
    recall_enable_query_expansion: bool = True  # Question→statement (recommended)
    recall_enable_hyde: bool = False  # Hypothetical doc embedding (expensive LLM call)
    recall_enable_self_query: bool = False  # LLM filter extraction (expensive LLM call)

    # --- General Recall Settings ---
    # Minimum fact confidence applied to RecallLayer vector searches. Kept
    # separate from fact_confidence_threshold (0.7) so recall can surface
    # lower-confidence facts that ranking will still order appropriately.
    recall_min_confidence: float = 0.5

    # --- Two-Stage Recall (§2.11) ---
    # Candidate pool width for stage 1: hybrid/self-query over-fetch size, and
    # the RRF output width when a downstream rerank stage will cut back to
    # top_k. Reranking is documented ineffective at ~20 candidates; ~50 is the
    # sweet spot for MiniLM-class cross-encoders on CPU.
    recall_candidate_pool: int = 50
    # Bounded demotion in the rerank stage: the cross-encoder may promote a
    # candidate freely but demote it at most this many positions below its
    # stage-1 (fused) rank — a hedge against encoder blind spots (negation-
    # heavy claims with no lexical overlap get buried by MiniLM despite a
    # strong fused rank). <= 0 disables the cap (pure cross-encoder order).
    # Eval-tuned: cap 2 restores every category to >= its fused baseline at
    # the same MRR as pure CE order (cap sweep in Memory-Roadmap §2.7).
    recall_ce_max_demotion: int = 2

    # --- First-Person Attribution Guard ---
    # For first-person queries ("what do I…"), penalize facts whose claim is
    # not about the user (claims start with their subject by extraction
    # convention). Penalize-only: a third-party fact can still be the right
    # answer, so nothing is ever dropped.
    # Default OFF — failed its §2.11 eval gate: under the cross-encoder stage
    # a rank penalty buys 0.0 abstention (the distractor IS the most relevant
    # candidate, CE rank 1) at a −3pp MRR cost on multi-hop. Numbers in
    # Memory-Roadmap §2.7; a CE-score *threshold* is the promising successor.
    recall_first_person_guard: bool = False
    recall_first_person_penalty: float = 0.5

    # --- Hybrid Search Settings ---
    recall_hybrid_bm25_weight: float = 0.3  # BM25 contribution to RRF
    recall_hybrid_vector_weight: float = 0.7  # Vector contribution to RRF
    recall_hybrid_rrf_k: int = 60  # RRF constant (standard value)

    # --- Entity-Centric Settings ---
    recall_entity_similarity_threshold: float = 0.65  # Min similarity to consider entity
    recall_entity_max_entities: int = 5  # Max entities to traverse
    recall_entity_graph_depth: int = 1  # Graph traversal depth

    # --- Query Expansion Settings ---
    recall_expansion_max_variants: int = 3  # Max query variants to generate

    # --- HyDE Settings (only used if recall_enable_hyde=True) ---
    # Models use provider:model format (e.g., "lmstudio:google/gemma-3-4b")
    recall_hyde_model: str = "lmstudio:google/gemma-3-4b"
    recall_hyde_temperature: float = 0.7
    recall_hyde_max_tokens: int = 150

    # --- Self-Query Settings (only used if recall_enable_self_query=True) ---
    recall_self_query_model: str = "lmstudio:google/gemma-3-4b"
    recall_self_query_temperature: float = 0.2
    recall_self_query_max_tokens: int = 200

    # ===========================================
    # Consolidation LLM Settings
    # ===========================================
    # All consolidation stages can be configured independently.
    # Models use provider:model format (e.g., "lmstudio:model", "anthropic:claude-3-5-haiku-latest")

    # Bulk default for every memory stage model below: a stage left empty (or set
    # to "inherit") resolves to this; if this is also empty, to the global default
    # chat model (see ExtractionService._resolve_stage_model). Lets a user point
    # all of memory at one model in a single setting, overriding individual stages
    # only where they want to. Empty = inherit the default chat model.
    feature_default_model: str = ""

    # --- Extraction (main fact/entity extraction) ---
    extraction_enabled: bool = True
    # "inherit" → follows the model-family tier (fast_utility role) → feature_default_model
    # → default chat model. Fresh installs let the role drive consolidation instead of
    # assuming a specific local model. Existing memory_settings keep their saved value;
    # use "Adopt roles for all stages" to clear a concrete override and inherit.
    extraction_model: str = "inherit"
    extraction_temperature: float = 0.2  # Lower for more consistent extraction
    extraction_max_tokens: int = 2000
    extraction_timeout: float = 30.0
    extraction_condense_facts: bool = True  # Condense verbose statements to atomic facts
    # Custom system prompt (empty = use default hardcoded prompt)
    extraction_system_prompt: str = ""

    # --- Relevance Filter (skip "thanks", "ok" turns) ---
    relevance_filter_enabled: bool = True
    relevance_filter_model: str = "inherit"  # → fast_utility role (see extraction_model)
    relevance_filter_temperature: float = 0.1  # Low temp for consistent YES/NO
    relevance_filter_max_tokens: int = 500  # Reasoning models need more tokens
    # Custom relevance prompt (empty = use default hardcoded prompt)
    relevance_filter_prompt: str = ""

    # --- Contradiction Detection (check new facts vs existing) ---
    contradiction_detection_enabled: bool = True  # Three-layer pipeline: fast gate → semantic search → LLM
    contradiction_model: str = "inherit"  # → fast_utility role (see extraction_model)
    contradiction_temperature: float = 0.2
    contradiction_max_tokens: int = 500
    # Layer 1: Semantic duplicate threshold (cosine similarity)
    semantic_duplicate_threshold: float = 0.92
    # Layer 2: Contradiction candidate search
    contradiction_similarity_threshold: float = 0.5
    contradiction_max_candidates: int = 10

    # --- Fact→Entity Linking ---
    # When a fact's entity_name resolves to no existing entity (not in the batch,
    # not found by name/alias/slug), auto-create a placeholder Entity so the fact is
    # never orphaned. Disable to fall back to recover-or-drop (still counted in metrics).
    link_autocreate_stub_entities: bool = True

    # --- User Correction Handling (detect "actually...", "no I meant...") ---
    correction_detection_enabled: bool = True  # Implicit corrections caught at extraction time
    correction_model: str = "inherit"  # → fast_utility role (see extraction_model)
    correction_temperature: float = 0.2
    correction_max_tokens: int = 500

    # --- Combined Relevance + Extraction (reduces LLM calls by ~75%) ---
    # "inherit" → deep_reasoning role (a reasoning model gives better quality here)
    # → feature_default_model → default chat model. See extraction_model.
    combined_extraction_model: str = "inherit"
    combined_extraction_temperature: float = 0.3  # Slightly higher for reasoning
    combined_extraction_max_tokens: int = 2000

    # --- Procedural distillation (Slice 1: candidates -> Procedures) ---
    # "inherit" follows feature_default_model ("default model for all memory
    # stages") -> global default -> bulk combined_extraction_model, so this stage
    # tracks the user's chosen memory model even though the client's "apply to all
    # stages" doesn't yet list it. resolve_with_fallback adds the provider safety net.
    procedural_distill_model: str = "inherit"
    procedural_distill_temperature: float = 0.2
    procedural_distill_max_tokens: int = 1000
    procedural_distill_timeout_s: int = 90  # bound the LLM call so a dead provider can't stall the job
    procedural_dedupe_threshold: float = 0.85  # cosine >= -> reinforce instead of create
    procedural_distill_batch_limit: int = 100  # max pending candidates per run
    # Reflex core (always-on procedure injection into the prompt)
    reflex_core_enabled: bool = True
    reflex_core_limit: int = 5
    # Stable salient core (Foundation #3): always-on high-salience fact/entity
    # injection — maintained, not searched. Limit kept <=10 to match the
    # to_context_string render cap. Min salience filters out low-value/forgotten.
    salient_core_enabled: bool = True
    salient_core_limit: int = 8
    salient_core_min_salience: float = 0.6

    # --- Confidence Calibration (map LLM certainty to calibrated scores) ---
    confidence_explicit: float = 0.95  # User directly stated
    confidence_implied: float = 0.85   # Strongly implied
    confidence_inferred: float = 0.70  # Reasonably inferred
    confidence_uncertain: float = 0.50 # Ambiguous or hedged

    # --- Entity Linking (backfill: link orphaned facts to existing entities) ---
    entity_linking_enabled: bool = True
    # Lower bound of the write-time semantic band (§2.10 Slice 1): matches scoring
    # >= this but < entity_linking_auto_threshold are LOGGED ONLY (gray zone —
    # corpus for a later adjudicator), never auto-merged.
    entity_linking_similarity_threshold: float = 0.75
    entity_linking_use_llm_disambiguation: bool = False  # Use LLM when ambiguous
    entity_linking_model: str = "inherit"  # → fast_utility role (see extraction_model)
    # Max orphan facts scanned per backfill run (full-history; bounds memory/time).
    entity_linking_max_facts: int = 5000
    # Longest entity name (in words) the claim n-gram matcher will consider.
    entity_linking_max_ngram: int = 4

    # --- Windowed Extraction (§2.10 Slice 1) ---
    # Extract over conversation WINDOWS (multiple turns per LLM call) instead of
    # per-turn: recurring mentions resolve to one entity, relationships may span
    # turns, and per-fact turn provenance is preserved. Off → legacy per-turn path.
    extraction_windowing_enabled: bool = True
    # Greedy window budget over turn contents (input side; sized so worst-case
    # JSON output stays under the window output cap below).
    extraction_window_max_tokens: int = 1200
    extraction_window_max_turns: int = 6
    # Output cap for window calls (floor — the stage max_tokens wins if larger).
    # Reasoning models think before the JSON; a dense 4-turn window's payload
    # plus thinking overflows the per-turn 2000 default and truncation eats the
    # trailing relationships array first.
    extraction_window_max_output_tokens: int = 3000
    # Rolling registry (prior windows' extractions re-shown to later windows).
    extraction_registry_max_entities: int = 15
    extraction_registry_max_facts: int = 20

    # --- Semantic entity resolution at write time (§2.10 Slice 1) ---
    # After exact name/alias/slug lookup misses, consult vector similarity over
    # existing entity embeddings. Same-type only; Agent entities never merge.
    semantic_entity_linking_enabled: bool = True
    # Auto-link (reuse + alias fold) when the top match scores >= this.
    entity_linking_auto_threshold: float = 0.90
    # Entities with NULL embeddings backfilled per entity_linking job run
    # (consolidation historically stored embedding=None; self-healing).
    entity_embedding_backfill_batch: int = 200

    # Entity types to recognize
    entity_types: list = [
        "Person", "Organization", "Location", "Concept",
        "Technology", "Product", "Event",
        # First-class agents: facts about a specific agent link to its Agent
        # entity (canonical key in properties.agent_id). See consolidation
        # _ensure_agent_entities (Phase 16 multi-agent attribution).
        "Agent",
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
    job_distill_procedures_interval: int = 30  # Distill procedure candidates

    # Worker health settings
    worker_heartbeat_interval: int = 30  # seconds
    worker_heartbeat_ttl: int = 90  # seconds (3x interval)

    class Config:
        env_file = ".env"
        extra = "ignore"


# Mutable settings that can be updated at runtime
_runtime_settings: Settings | None = None
_settings_cache_time: float = 0.0
# Process-local pin (see pin_memory_settings) — checked BEFORE the TTL cache.
_pinned_settings: Settings | None = None
# Last overrides-file load failure ("TypeName: message"), None when healthy.
_settings_load_error: str | None = None


def get_settings() -> Settings:
    """
    Get settings instance with runtime overrides applied.

    A process-local pin (pin_memory_settings) wins over everything — the TTL
    cache cannot revert it mid-run. Otherwise settings are cached with a TTL;
    once the cache is older than SETTINGS_CACHE_TTL seconds it refreshes, so UI
    changes take effect without restarting the API server.
    """
    global _runtime_settings, _settings_cache_time

    if _pinned_settings is not None:
        return _pinned_settings

    current_time = time.time()
    cache_expired = (current_time - _settings_cache_time) > SETTINGS_CACHE_TTL

    if _runtime_settings is None or cache_expired:
        _runtime_settings = _load_settings_with_overrides()
        _settings_cache_time = current_time

    return _runtime_settings


@contextmanager
def pin_memory_settings(override: Settings):
    """
    Pin get_settings() to a fixed Settings object for the duration of the block.

    The ONE sanctioned way to apply a temporary override (tests, eval harnesses,
    experiments): every get_settings() read in the memory kit sees `override`,
    the TTL cache cannot revert it mid-run, and data/memory_settings.json is
    never touched. Nesting-safe — an inner pin restores the outer pin on exit;
    the outermost exit busts the TTL cache so the next read reloads fresh.
    Process-global, so single-process use only (matches the eval/test harnesses).
    """
    global _pinned_settings, _runtime_settings, _settings_cache_time
    prior = _pinned_settings
    _pinned_settings = override
    try:
        yield override
    finally:
        _pinned_settings = prior
        if prior is None:
            _runtime_settings = None
            _settings_cache_time = 0.0


def _load_settings_with_overrides() -> Settings:
    """Load base settings and apply any overrides from memory_settings.json."""
    global _settings_load_error
    base = Settings()

    path = MEMORY_SETTINGS_PATH
    if not path.exists():
        # Legacy-CWD fallback: overrides written by a pre-anchoring process
        # running outside the repo root. Read them, but nag until moved.
        try:
            legacy_differs = (
                _LEGACY_SETTINGS_PATH.exists()
                and _LEGACY_SETTINGS_PATH.resolve() != MEMORY_SETTINGS_PATH
            )
        except OSError:
            legacy_differs = False
        if legacy_differs:
            logger.warning(
                "Reading memory settings from CWD-relative %s (legacy location); "
                "move it to %s",
                _LEGACY_SETTINGS_PATH.resolve(), MEMORY_SETTINGS_PATH,
            )
            path = _LEGACY_SETTINGS_PATH

    if path.exists():
        try:
            with open(path) as f:
                overrides = json.load(f)
            # Apply overrides to a copy of base settings
            base_dict = base.model_dump()
            base_dict.update(overrides)
            loaded = Settings(**base_dict)
            _settings_load_error = None
            return loaded
        except Exception as e:
            # INV-1: loading never raises — fall back to base settings, but say
            # so (the old silent fallback hid corrupt files indefinitely). The
            # error is surfaced via get_settings_file_status() in the UI.
            _settings_load_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "Failed to load memory settings overrides from %s — using "
                "defaults: %s", path, _settings_load_error,
            )
            return base

    _settings_load_error = None
    return base


def get_settings_file_status() -> dict[str, Any]:
    """Health of the overrides file, for the settings UI (never raises)."""
    get_settings()  # refresh the load-error state if the TTL cache expired
    return {
        "path": str(MEMORY_SETTINGS_PATH),
        "exists": MEMORY_SETTINGS_PATH.exists(),
        "error": _settings_load_error,
    }


def validate_memory_settings(candidate: dict[str, Any]) -> dict[str, str]:
    """
    Validate candidate overrides against the Settings schema before saving.

    Merges `candidate` over the current effective settings and re-validates the
    whole model. Returns {key: message} per failing key (empty dict = valid) so
    the API can reject the whole write with per-key errors.
    """
    from pydantic import ValidationError

    merged = get_settings().model_dump()
    merged.update(candidate)
    try:
        Settings(**merged)
    except ValidationError as e:
        errors: dict[str, str] = {}
        for err in e.errors():
            loc = err.get("loc") or ("settings",)
            errors[str(loc[0])] = err.get("msg", "invalid value")
        return errors
    return {}


def save_memory_settings(settings_dict: dict[str, Any]) -> None:
    """
    Save consolidation settings to file.

    Only saves the keys that are different from defaults or are
    user-configurable consolidation settings.

    Args:
        settings_dict: Dictionary of settings to save
    """
    global _runtime_settings, _settings_cache_time

    # Route trajectory_compression_* keys to ConfigManager (they don't live in Settings)
    settings_dict = _apply_trajectory_compression_settings(settings_dict)

    if settings_dict:
        # Ensure data directory exists
        MEMORY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load existing settings if any
        existing = {}
        if MEMORY_SETTINGS_PATH.exists():
            try:
                with open(MEMORY_SETTINGS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        # Merge new settings
        existing.update(settings_dict)

        # Write back
        with open(MEMORY_SETTINGS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

    # Clear cached settings so next get_settings() reloads immediately
    _runtime_settings = None
    _settings_cache_time = 0.0


def load_memory_settings() -> dict[str, Any]:
    """
    Load current memory settings as a dictionary.

    Returns settings from file merged with defaults.
    """
    settings = get_settings()
    return settings.model_dump()


def get_consolidation_settings() -> dict[str, Any]:
    """
    Get only the consolidation-related settings for the UI.

    Returns a subset of settings relevant to the consolidation settings panel.
    """
    settings = get_settings()

    return {
        # Bulk default for all stage models (empty = inherit the default chat model).
        "feature_default_model": settings.feature_default_model,

        # Extraction (model uses provider:model format, e.g., "lmstudio:google/gemma-3-4b")
        "extraction_enabled": settings.extraction_enabled,
        "extraction_model": settings.extraction_model,
        "extraction_temperature": settings.extraction_temperature,
        "extraction_max_tokens": settings.extraction_max_tokens,
        "extraction_condense_facts": settings.extraction_condense_facts,
        "extraction_system_prompt": settings.extraction_system_prompt,

        # Relevance filter
        "relevance_filter_enabled": settings.relevance_filter_enabled,
        "relevance_filter_model": settings.relevance_filter_model,
        "relevance_filter_temperature": settings.relevance_filter_temperature,
        "relevance_filter_max_tokens": settings.relevance_filter_max_tokens,
        "relevance_filter_prompt": settings.relevance_filter_prompt,

        # Combined relevance + extraction (handles ~75% of consolidation traffic)
        "combined_extraction_model": settings.combined_extraction_model,
        "combined_extraction_temperature": settings.combined_extraction_temperature,
        "combined_extraction_max_tokens": settings.combined_extraction_max_tokens,

        # Procedural distillation (Slice 1) + reflex core
        "procedural_distill_model": settings.procedural_distill_model,
        "procedural_distill_temperature": settings.procedural_distill_temperature,
        "procedural_distill_max_tokens": settings.procedural_distill_max_tokens,
        "procedural_distill_timeout_s": settings.procedural_distill_timeout_s,
        "procedural_dedupe_threshold": settings.procedural_dedupe_threshold,
        "procedural_distill_batch_limit": settings.procedural_distill_batch_limit,
        "reflex_core_enabled": settings.reflex_core_enabled,
        "reflex_core_limit": settings.reflex_core_limit,
        "salient_core_enabled": settings.salient_core_enabled,
        "salient_core_limit": settings.salient_core_limit,
        "salient_core_min_salience": settings.salient_core_min_salience,

        # Entity linking
        "entity_linking_enabled": settings.entity_linking_enabled,
        "entity_linking_similarity_threshold": settings.entity_linking_similarity_threshold,
        "entity_linking_max_facts": settings.entity_linking_max_facts,
        "entity_linking_max_ngram": settings.entity_linking_max_ngram,
        "entity_linking_use_llm_disambiguation": settings.entity_linking_use_llm_disambiguation,
        "entity_linking_model": settings.entity_linking_model,

        # Quality thresholds
        "fact_confidence_threshold": settings.fact_confidence_threshold,
        "promotion_min_confidence": settings.promotion_min_confidence,

        # Job intervals
        "job_consolidate_interval": settings.job_consolidate_interval,
        "job_promote_interval": settings.job_promote_interval,
        "job_entity_linking_interval": settings.job_entity_linking_interval,
        "job_distill_procedures_interval": settings.job_distill_procedures_interval,

        # Fact verification pipeline
        "link_autocreate_stub_entities": settings.link_autocreate_stub_entities,
        "contradiction_detection_enabled": settings.contradiction_detection_enabled,
        "contradiction_model": settings.contradiction_model,
        "contradiction_temperature": settings.contradiction_temperature,
        "contradiction_max_tokens": settings.contradiction_max_tokens,
        "correction_detection_enabled": settings.correction_detection_enabled,
        "correction_model": settings.correction_model,
        "correction_temperature": settings.correction_temperature,
        "correction_max_tokens": settings.correction_max_tokens,
        "semantic_duplicate_threshold": settings.semantic_duplicate_threshold,
        "contradiction_similarity_threshold": settings.contradiction_similarity_threshold,
        "contradiction_max_candidates": settings.contradiction_max_candidates,

        # Entity/relationship types (read-only display)
        "entity_types": settings.entity_types,
        "relationship_types": settings.relationship_types,

        # Trajectory compression (stored in ConfigManager, surfaced here for the UI)
        **_get_trajectory_compression_settings(),
    }


_TRAJECTORY_COMPRESSION_KEYS: dict[str, Any] = {
    # ui_key: (config_path, default)
    "trajectory_compression_enabled": ("trajectory_compression.enabled", True),
    "trajectory_compression_model": ("trajectory_compression.model", "anthropic:claude-haiku-4-5-20251001"),
    "trajectory_compression_temperature": ("trajectory_compression.temperature", 0.2),
    "trajectory_compression_max_tokens": ("trajectory_compression.max_tokens", 1500),
    "trajectory_compression_threshold_ratio": ("trajectory_compression.threshold_ratio", 0.75),
    "trajectory_compression_preserve_recent_rounds": ("trajectory_compression.preserve_recent_rounds", 2),
}


def _get_trajectory_compression_settings() -> dict[str, Any]:
    """Read trajectory-compression keys from ConfigManager for the UI."""
    from agentx_ai.config import get_config_manager
    cfg = get_config_manager()
    return {
        ui_key: cfg.get(path, default)
        for ui_key, (path, default) in _TRAJECTORY_COMPRESSION_KEYS.items()
    }


def _apply_trajectory_compression_settings(settings_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Pull any trajectory_compression_* keys out of settings_dict and write them
    to ConfigManager. Returns the remaining dict (other keys untouched).
    """
    if not any(k in settings_dict for k in _TRAJECTORY_COMPRESSION_KEYS):
        return settings_dict
    from agentx_ai.config import get_config_manager
    cfg = get_config_manager()
    remaining = dict(settings_dict)
    for ui_key, (path, _default) in _TRAJECTORY_COMPRESSION_KEYS.items():
        if ui_key in remaining:
            cfg.set(path, remaining.pop(ui_key))
    return remaining


def get_recall_settings() -> dict[str, Any]:
    """
    Get RecallLayer settings for the UI.

    Returns settings for the enhanced retrieval (RecallLayer) panel.
    """
    settings = get_settings()

    return {
        # Feature toggles
        "recall_enable_hybrid": settings.recall_enable_hybrid,
        "recall_enable_entity_centric": settings.recall_enable_entity_centric,
        "recall_enable_query_expansion": settings.recall_enable_query_expansion,
        "recall_enable_hyde": settings.recall_enable_hyde,
        "recall_enable_self_query": settings.recall_enable_self_query,

        # General recall settings
        "recall_min_confidence": settings.recall_min_confidence,

        # Two-stage recall (§2.11)
        "recall_candidate_pool": settings.recall_candidate_pool,
        "recall_ce_max_demotion": settings.recall_ce_max_demotion,
        "cross_encoder_enabled": settings.cross_encoder_enabled,
        "cross_encoder_model": settings.cross_encoder_model,

        # First-person attribution guard
        "recall_first_person_guard": settings.recall_first_person_guard,
        "recall_first_person_penalty": settings.recall_first_person_penalty,

        # Hybrid search settings
        "recall_hybrid_bm25_weight": settings.recall_hybrid_bm25_weight,
        "recall_hybrid_vector_weight": settings.recall_hybrid_vector_weight,
        "recall_hybrid_rrf_k": settings.recall_hybrid_rrf_k,

        # Entity-centric settings
        "recall_entity_similarity_threshold": settings.recall_entity_similarity_threshold,
        "recall_entity_max_entities": settings.recall_entity_max_entities,
        "recall_entity_graph_depth": settings.recall_entity_graph_depth,

        # Query expansion settings
        "recall_expansion_max_variants": settings.recall_expansion_max_variants,

        # HyDE settings (model uses provider:model format)
        "recall_hyde_model": settings.recall_hyde_model,
        "recall_hyde_temperature": settings.recall_hyde_temperature,
        "recall_hyde_max_tokens": settings.recall_hyde_max_tokens,

        # Self-query settings (model uses provider:model format)
        "recall_self_query_model": settings.recall_self_query_model,
        "recall_self_query_temperature": settings.recall_self_query_temperature,
        "recall_self_query_max_tokens": settings.recall_self_query_max_tokens,
    }
