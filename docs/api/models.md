# API Models

Data models and schemas used across the API. All models use Pydantic unless noted as `@dataclass`.

---

## Provider Models

Defined in `providers/base.py`.

### Message

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | MessageRole | yes | `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `content` | string | yes | Message text |
| `name` | string | no | Tool name (for tool messages) |
| `tool_call_id` | string | no | ID linking to a tool call |
| `tool_calls` | list[dict] | no | Tool calls requested by assistant |

### ToolCall

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique tool call ID |
| `name` | string | Tool function name |
| `arguments` | dict | Parsed arguments |

### StreamChunk

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Token text (may be empty) |
| `finish_reason` | string | `null`, `"stop"`, `"tool_calls"` |
| `tool_calls` | list[ToolCall] | Tool calls collected during stream |

### CompletionResult

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Generated text |
| `finish_reason` | string | `"stop"`, `"length"`, `"tool_calls"` |
| `tool_calls` | list[ToolCall] | Tool calls (if function calling) |
| `usage` | dict | `{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}` |
| `model` | string | Model ID used |
| `raw_response` | dict | Full provider response (debug) |

### ModelCapabilities (`@dataclass`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `supports_tools` | bool | `false` | Function calling support |
| `supports_vision` | bool | `false` | Image input support |
| `supports_streaming` | bool | `true` | Streaming support |
| `supports_json_mode` | bool | `false` | JSON mode support |
| `context_window` | int | `4096` | Max input tokens |
| `max_output_tokens` | int | `null` | Max output tokens |
| `cost_per_1k_input` | float | `null` | Input cost in USD |
| `cost_per_1k_output` | float | `null` | Output cost in USD |

---

## Agent Models

Defined in `agent/core.py`.

### AgentResult

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Short UUID for the task |
| `status` | AgentStatus | `"idle"`, `"planning"`, `"reasoning"`, `"executing"`, `"complete"`, `"failed"`, `"cancelled"` |
| `answer` | string | Final response text |
| `thinking` | string | Extracted `<think>` tag content |
| `has_thinking` | bool | Whether thinking was extracted |
| `plan_steps` | int | Number of plan steps executed |
| `reasoning_steps` | int | Number of reasoning steps |
| `tools_used` | list[string] | Names of tools invoked |
| `models_used` | list[string] | Model IDs used |
| `total_tokens` | int | Total token count |
| `total_time_ms` | float | Elapsed time in milliseconds |
| `trace` | list[dict] | Debug trace (optional) |

### AgentConfig (`@dataclass`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"agentx"` | Agent name |
| `user_id` | string | `null` | User identifier |
| `default_model` | string | `"llama3.2"` | Default model |
| `reasoning_model` | string | `null` | Override for reasoning |
| `drafting_model` | string | `null` | Override for drafting |
| `max_iterations` | int | `20` | Max task iterations |
| `timeout_seconds` | float | `300.0` | Task timeout |
| `enable_planning` | bool | `true` | Enable task planning |
| `enable_reasoning` | bool | `true` | Enable reasoning strategies |
| `enable_drafting` | bool | `false` | Enable drafting |
| `enable_memory` | bool | `true` | Enable memory system |
| `enable_tools` | bool | `true` | Enable MCP tools |
| `default_reasoning_strategy` | string | `"auto"` | Strategy: `"auto"`, `"cot"`, `"tot"`, `"react"`, `"reflection"` |
| `allowed_tools` | list[string] | `null` | Whitelist tools (null = all) |
| `blocked_tools` | list[string] | `null` | Blacklist tools |
| `max_tool_rounds` | int | `10` | Max tool round-trips |

---

## Memory Models

Defined in `kit/agent_memory/models.py`.

### Turn

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto UUID | Turn identifier |
| `conversation_id` | string | required | Parent conversation |
| `index` | int | required | Position in conversation |
| `timestamp` | datetime | now (UTC) | When the turn occurred |
| `role` | string | required | `"user"`, `"assistant"`, `"system"`, `"tool"` |
| `content` | string | required | Message content |
| `embedding` | list[float] | `null` | Vector embedding |
| `token_count` | int | `null` | Token count |
| `metadata` | dict | `{}` | Extra data (model, latency, etc.) |
| `channel` | string | `"_global"` | Memory channel |

### Entity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto UUID | Entity identifier |
| `name` | string | required | Entity name |
| `type` | string | required | `"Person"`, `"Organization"`, `"Concept"`, etc. |
| `aliases` | list[string] | `[]` | Alternative names |
| `description` | string | `null` | Entity description |
| `embedding` | list[float] | `null` | Vector embedding |
| `salience` | float | `0.5` | Importance score (0–1) |
| `properties` | dict | `{}` | Arbitrary key-value properties |
| `first_seen` | datetime | now | When first encountered |
| `last_accessed` | datetime | now | Last retrieval time |
| `access_count` | int | `0` | Retrieval count |
| `channel` | string | `"_global"` | Memory channel |

### Fact

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto UUID | Fact identifier |
| `claim` | string | required | The factual claim |
| `claim_hash` | string | auto | SHA256 hash for duplicate detection |
| `confidence` | float | `0.8` | Confidence score (0–1) |
| `source` | string | required | `"extraction"`, `"user_stated"`, `"inferred"` |
| `source_turn_id` | string | `null` | Turn this fact was extracted from |
| `entity_ids` | list[string] | `[]` | Linked entities |
| `embedding` | list[float] | `null` | Vector embedding |
| `created_at` | datetime | now | Creation time |
| `channel` | string | `"_global"` | Memory channel |
| `last_accessed` | datetime | now | Last retrieval time |
| `access_count` | int | `0` | Retrieval count |
| `salience` | float | `0.5` | Importance score |
| `temporal_context` | string | `null` | `"current"`, `"past"`, `"future"` |
| `superseded_at` | datetime | `null` | When superseded by correction |
| `superseded_by_id` | string | `null` | ID of superseding fact |
| `supersedes_id` | string | `null` | ID of fact this supersedes |
| `flagged_for_review` | bool | `false` | Flagged for contradiction review |

### Goal

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto UUID | Goal identifier |
| `description` | string | required | Goal description |
| `status` | string | `"active"` | `"active"`, `"completed"`, `"abandoned"`, `"blocked"` |
| `priority` | int | `3` | Priority 1–5 |
| `parent_goal_id` | string | `null` | Parent goal for hierarchy |
| `embedding` | list[float] | `null` | Vector embedding |
| `created_at` | datetime | now | Creation time |
| `deadline` | datetime | `null` | Optional deadline |
| `channel` | string | `"_global"` | Memory channel |

### Strategy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | auto UUID | Strategy identifier |
| `description` | string | required | Strategy description |
| `context_pattern` | string | required | Matching pattern (regex/keywords) |
| `tool_sequence` | list[string] | `[]` | Ordered tool names |
| `embedding` | list[float] | `null` | Vector embedding |
| `success_count` | int | `0` | Successful uses |
| `failure_count` | int | `0` | Failed uses |
| `last_used` | datetime | `null` | Last invocation |
| `channel` | string | `"_global"` | Memory channel |

### MemoryBundle

Aggregated retrieval result returned by `AgentMemory.remember()`.

| Field | Type | Description |
|-------|------|-------------|
| `relevant_turns` | list[dict] | Recent relevant conversation turns |
| `entities` | list[dict] | Matching entities |
| `facts` | list[dict] | Relevant facts |
| `strategies` | list[dict] | Applicable strategies |
| `active_goals` | list[dict] | Current active goals |
| `user_context` | dict | User-level context |

Has a `to_context_string()` method that formats the bundle for LLM prompt injection.

---

## Prompt Models

Defined in `prompts/models.py`.

### PromptSection

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique section identifier |
| `name` | string | Display name |
| `type` | SectionType | `"persona"`, `"task"`, `"format"`, `"constraints"`, `"examples"`, `"context"`, `"custom"` |
| `content` | string | Prompt text |
| `enabled` | bool | Whether active in composition |
| `order` | int | Sort order within profile |

### PromptProfile

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique profile identifier |
| `name` | string | Display name |
| `description` | string | Usage description |
| `sections` | list[PromptSection] | Ordered sections |
| `is_default` | bool | Whether this is the default profile |

### GlobalPrompt

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Global prompt text |
| `enabled` | bool | Whether active |

### PromptConfig

Combines all prompt components for a request.

| Field | Type | Description |
|-------|------|-------------|
| `global_prompt` | GlobalPrompt | Core persona prompt |
| `profile` | PromptProfile | Selected profile |
| `mcp_tools_prompt` | string | Auto-generated tools prompt |
| `structured_output` | StructuredOutputConfig | Output constraints |
| `additional_context` | string | Request-specific context |
| `system_override` | string | Replaces entire system prompt |

Composition order: Global prompt → MCP tools → Profile sections → Additional context.

---

## MCP Models

### ServerConfig (`@dataclass`, defined in `mcp/server_registry.py`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Server identifier |
| `transport` | TransportType | `"stdio"`, `"sse"`, `"streamable_http"` |
| `command` | string | Command for stdio transport |
| `args` | list[string] | Command arguments |
| `env` | dict | Environment variables (resolved from `$VAR` syntax) |
| `url` | string | URL for SSE/HTTP transport |
| `headers` | dict | HTTP headers (env vars resolved) |

### ToolInfo (`@dataclass`, defined in `mcp/tool_executor.py`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tool name |
| `description` | string | Tool description |
| `input_schema` | dict | JSON Schema for parameters |
| `server_name` | string | Source server |

### ResourceInfo (`@dataclass`, defined in `mcp/client.py`)

| Field | Type | Description |
|-------|------|-------------|
| `uri` | string | Resource URI |
| `name` | string | Resource name |
| `description` | string | Resource description |
| `mime_type` | string | Content type |
| `server_name` | string | Source server |

---

## SSE Event Schemas

Events emitted by `POST /api/agent/chat/stream`:

| Event | Schema |
|-------|--------|
| `start` | `{"task_id": string, "model": string}` |
| `chunk` | `{"content": string}` |
| `tool_call` | `{"tool": string, "arguments": dict}` |
| `tool_result` | `{"tool": string, "content": string}` |
| `done` | `{"task_id": string, "thinking": string?, "has_thinking": bool, "total_time_ms": float, "session_id": string}` |
| `error` | `{"error": string}` |
