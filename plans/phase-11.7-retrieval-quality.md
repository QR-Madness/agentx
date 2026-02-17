# Phase 11.7 Implementation Plan: Retrieval Quality

**Goal**: Improve retrieval beyond the current basic implementation with proper multi-channel support, weighted scoring, caching, and comprehensive metrics.

**Current State Assessment**:
- Multi-channel filtering is partially implemented in `episodic.py` and `semantic.py` (queries `channel OR _global`)
- Results are not tagged with source channel
- Reranking in `retrieval.py` is basic (conversation diversity filter only)
- No caching exists
- Basic metrics via audit logger, not comprehensive

---

## Task 1: Enhanced Multi-Channel Retrieval

**Files**: `retrieval.py`, `interface.py`, `models.py`

### 1.1 Update MemoryBundle for Channel Tagging

Add `source_channel` field to result items in `MemoryBundle`:

```python
# models.py - Update docstrings to clarify channel tagging
# Results in relevant_turns, entities, facts should include 'channel' field
```

### 1.2 Add `channels` Parameter to Retrieval

Modify `MemoryRetriever.retrieve()` and `AgentMemory.remember()`:

- Add `channels: Optional[List[str]] = None` parameter
- Default behavior: `[active_channel, "_global"]` (deduplicated)
- Allow explicit override: `remember(query, channels=["project-a", "project-b", "_global"])`

### 1.3 Implement Channel Boost Factor

Add configuration and implementation:

```python
# config.py
channel_active_boost: float = 1.2  # Boost factor for active channel results

# retrieval.py
# Apply boost: if result.channel == active_channel, score *= channel_active_boost
```

### 1.4 Merge Results from Multiple Channels

Current queries already return results from multiple channels. Enhance to:
- Deduplicate by ID (same entity/fact appearing in multiple channels)
- Prefer higher-scored duplicate
- Preserve channel attribution in results

**Acceptance Criteria**:
- [x] Results include `channel` field (already present in queries)
- [ ] `remember()` accepts `channels` parameter
- [ ] Active channel results get configurable boost
- [ ] Results properly deduplicated across channels

---

## Task 2: Proper Reranking with Weighted Scoring

**Files**: `retrieval.py`

### 2.1 Implement Weighted Score Combination

Replace simple diversity filter with proper scoring:

```python
def _rerank(self, bundle: MemoryBundle, query_embedding: List[float], weights: RetrievalWeights) -> MemoryBundle:
    """
    Rerank results using weighted combination of:
    - Vector similarity score (from search)
    - Recency score (time-based decay)
    - Salience score (entity/fact importance)
    - Channel boost (active channel preference)
    """
```

Scoring formula per result:
```
final_score = (
    weights.episodic * similarity_score +
    weights.recency * recency_score +
    channel_boost_factor
)
```

### 2.2 Add Relevance Score Normalization

Normalize scores across memory types to comparable scales:

```python
def _normalize_scores(self, results: List[Dict], score_key: str = "score") -> List[Dict]:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not results:
        return results
    scores = [r.get(score_key, 0) for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return results
    for r in results:
        r["normalized_score"] = (r.get(score_key, 0) - min_s) / (max_s - min_s)
    return results
```

### 2.3 Add Cross-Encoder Reranking Option

Optional cross-encoder reranking (off by default due to latency):

```python
# config.py
cross_encoder_enabled: bool = False
cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# retrieval.py
def _cross_encoder_rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
    """Rerank using cross-encoder model for higher accuracy."""
    if not self.cross_encoder:
        return results
    # Score each (query, result_text) pair
    # Return sorted by cross-encoder score
```

**Acceptance Criteria**:
- [ ] Weighted scoring replaces diversity-only filter
- [ ] Scores normalized across memory types
- [ ] Cross-encoder reranking available (config flag)
- [ ] Existing behavior preserved when weights are default

---

## Task 3: Retrieval Caching

**Files**: `retrieval.py`, `config.py`, `connections.py`

### 3.1 Add Cache Configuration

```python
# config.py
retrieval_cache_enabled: bool = True
retrieval_cache_ttl_seconds: int = 60
retrieval_cache_key_prefix: str = "retrieval_cache"
```

### 3.2 Implement Redis-Based Cache

```python
# retrieval.py
def _get_cache_key(self, query: str, user_id: str, channel: str, params_hash: str) -> str:
    """Generate cache key for retrieval."""
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"{self.cache_prefix}:{user_id}:{channel}:{query_hash}:{params_hash}"

def _get_cached(self, cache_key: str) -> Optional[MemoryBundle]:
    """Get cached retrieval result."""

def _set_cached(self, cache_key: str, bundle: MemoryBundle) -> None:
    """Cache retrieval result with TTL."""
```

### 3.3 Implement Cache Invalidation

Invalidate cache on writes to same scope:

```python
# interface.py - In store_turn(), learn_fact(), upsert_entity()
def _invalidate_retrieval_cache(self, user_id: str, channel: str) -> None:
    """Invalidate retrieval cache for user/channel scope."""
    pattern = f"{self.cache_prefix}:{user_id}:{channel}:*"
    # Use Redis SCAN + DELETE for pattern-based invalidation
```

**Acceptance Criteria**:
- [ ] Cache implemented with configurable TTL
- [ ] Cache key includes query hash, user, channel, params
- [ ] Cache invalidated on relevant writes
- [ ] Cache can be disabled via config

---

## Task 4: Retrieval Metrics

**Files**: `retrieval.py`, `config.py`, `audit.py`

### 4.1 Define Metrics to Track

```python
@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""
    query_hash: str
    user_id: str
    channel: str

    # Hit rates
    episodic_count: int
    semantic_facts_count: int
    semantic_entities_count: int
    procedural_count: int
    total_count: int

    # Latency breakdown (ms)
    embedding_latency: int
    episodic_latency: int
    semantic_latency: int
    procedural_latency: int
    reranking_latency: int
    total_latency: int

    # Cache
    cache_hit: bool

    # Channels searched
    channels_searched: List[str]
    results_per_channel: Dict[str, int]
```

### 4.2 Instrument Retrieval Pipeline

Add timing and counting at each stage:

```python
def retrieve(self, ...):
    metrics = RetrievalMetrics(...)

    t0 = time.perf_counter()
    query_embedding = self.embedder.embed_single(query)
    metrics.embedding_latency = int((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    # ... episodic retrieval
    metrics.episodic_latency = int((time.perf_counter() - t1) * 1000)
    metrics.episodic_count = len(bundle.relevant_turns)

    # ... continue for other strategies

    self._log_metrics(metrics)
```

### 4.3 Log Metrics to Audit Table

When `audit_log_level >= reads`:

```python
def _log_metrics(self, metrics: RetrievalMetrics) -> None:
    """Log retrieval metrics to audit table."""
    if self._audit_logger and self._audit_logger.should_log_reads():
        self._audit_logger.log_retrieval_metrics(
            user_id=metrics.user_id,
            channel=metrics.channel,
            metrics=asdict(metrics)
        )
```

**Acceptance Criteria**:
- [ ] Latency tracked per retrieval strategy
- [ ] Hit rates tracked per memory type
- [ ] Channel-specific result counts
- [ ] Metrics logged to audit table when appropriate
- [ ] Cache hit/miss tracked

---

## Implementation Order

1. **Task 2.1-2.2**: Weighted scoring and normalization (foundational)
2. **Task 1.2-1.4**: Multi-channel parameters and merging
3. **Task 4.1-4.3**: Metrics instrumentation (needed for validation)
4. **Task 3.1-3.3**: Caching (optimization)
5. **Task 2.3**: Cross-encoder reranking (optional enhancement)

---

## Testing Strategy

### Unit Tests
- Test `RetrievalWeights.from_dict()` and `merge()` (existing)
- Test score normalization with edge cases
- Test cache key generation
- Test channel deduplication logic

### Integration Tests (require Docker)
- Test multi-channel retrieval returns results from both channels
- Test cache hit/miss behavior
- Test cache invalidation on writes
- Test metrics are logged to audit table

---

## Configuration Summary

New config options in `config.py`:

```python
# Multi-channel
channel_active_boost: float = 1.2

# Reranking
cross_encoder_enabled: bool = False
cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Caching
retrieval_cache_enabled: bool = True
retrieval_cache_ttl_seconds: int = 60
retrieval_cache_key_prefix: str = "retrieval_cache"
```

---

## Dependencies

- No new Python packages required for Tasks 1, 2.1-2.2, 3, 4
- Cross-encoder reranking (Task 2.3) requires: `sentence-transformers` (already in dependencies)

---

## Estimated Scope

| Task | Files Modified | Complexity |
|------|---------------|------------|
| 1. Multi-channel | 3 | Medium |
| 2. Reranking | 1 | Medium |
| 3. Caching | 3 | Medium |
| 4. Metrics | 3 | Low |

Total: ~4-6 files modified, moderate complexity overall.
