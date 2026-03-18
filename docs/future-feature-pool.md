# Future Feature Pool

Ideas and planned improvements for future phases.

---

## Retrieval Quality Enhancements

### Working Memory Scratchpad
Always prepend a structured scratchpad to context:
- Current topic/task
- Active entities being discussed
- Recent corrections
- Open questions

Low effort, improves coherence by giving the model orientation.

### Conversation Summarization
Maintain rolling summaries instead of/alongside raw turns:
- Per-session summary (updated every N turns)
- Per-topic clusters

Retrieval becomes: `recent_turns + relevant_summaries + relevant_facts`
Compresses historical context significantly.

### Query Intent Classification
Before retrieval, classify the query type:
- **Follow-up**: Heavily weight recency, use last conversation
- **Callback**: "Remember when we..." — search older history
- **New topic**: Broad semantic search
- **Factual recall**: Focus on entities/facts, not turns

Could be rule-based (detect "you said", "earlier", "last time") or lightweight LLM.

### Negative/Correction Tracking
When `correction_detection_enabled` is on:
- Mark superseded facts as `temporal_context: "past"`
- Link correction to original fact
- Prioritize corrections in retrieval

### Fact Staleness Detection
Facts decay at different rates:
- "User is working on project X" — might change weekly
- "User prefers dark mode" — stable preference
- "User's dog is named Max" — very stable

Add `expected_stability: transient|stable|permanent` and surface staleness warnings.

### Multi-hop Entity Traversal
Current graph traversal is `depth=2` but doesn't reason over paths.

For: "What tools do I use for work projects?"
Need: User → works_at → Company → has_project → Project → uses_tool → Tool

Add lightweight path-finding retrieval mode.

---

## MCP Tools

### Conversation MCP Tool
Expose memory as MCP tools for external agents:

```
memory_recall(query, filters?) → relevant context
memory_store(fact) → store user-provided fact
conversation_summary(conversation_id?) → summarize recent/specific conversation
```

Use cases:
- Multi-agent workflows sharing context
- User explicitly storing/querying memories
- External tools needing conversation awareness

---

## Extraction Improvements

### Claude Sonnet for Extraction
Switch extraction from local models to Claude Sonnet:
- Better structured output adherence
- Nuance detection (hedging, uncertainty, temporal qualifiers)
- Superior entity resolution

Tradeoff is cost/latency, but consolidation jobs are async/batched.

### Improved Extraction Prompts
Current prompts are generic. Need:
- Few-shot examples
- Better schema definitions
- Domain-specific tuning

---

## Completed

- [x] Always-include recent turns (regardless of relevancy) — `always_include_recent_turns` config
- [x] Pleasantry-prefix extraction fix — Updated relevance prompts to handle "Thanks! I work at Google" patterns
