# Agent Teams Internals

How Agent Teams works inside the server. For using it, see [Agent Teams](../features/multi-agent.md).

Internally the feature keeps its original name, **Agent Alloy**: the module (`alloy/`), the API
routes (`/api/alloy/…`) and the config keys (`alloy.*`) all use it. In the UI a workflow is a
**Team**, the supervisor is the **Lead**, and specialists are **Members**.

## Delegation modes

- **Structured teams** — a saved workflow (`workflow_id` on the chat request) puts the team's lead
  in charge of the conversation with an "orchestrate, don't execute" supervisor prompt.
- **Ad-hoc delegation** — in ordinary chats, any agent can hand a subtask to any profile on the
  team roster (`available_for_delegation`, opt-in, off by default). The agent gets the
  `delegate_to` tool *and* a roster block in its system prompt listing each teammate's specialty
  (`delegation_hint`, falling back to the profile description), with deliberately soft guidance:
  handle it yourself by default, delegate when a teammate is clearly better suited. Ad-hoc
  delegation is enabled by default (`alloy.allow_adhoc_delegation`), and nothing delegates until
  profiles join the roster. The Relay's **Solo/Team** toggle sends `disable_delegation` to
  suppress it for one chat; it's ignored under a workflow, since a team run *is* delegation.

## Data Model

A workflow binds exactly one supervisor to zero or more specialists. Members are referenced
by their immutable Docker-style `agent_id` (e.g. `bold-cosmic-falcon`), **not** by display
name — so renaming a profile never breaks a workflow.

**`Workflow`** (`api/agentx_ai/alloy/models.py`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Kebab-case identifier (`^[a-z0-9][a-z0-9-]*$`) |
| `name` | string | Display name |
| `description` | string? | Free-text description |
| `supervisor_agent_id` | string | `agent_id` of the supervisor profile |
| `members` | `WorkflowMember[]` | Supervisor + specialists |
| `routes` | `WorkflowRoute[]` | Declarative routing — **schema only in v1, not executed** |
| `shared_channel` | string | Workflow-scoped memory channel; auto-derived as `_alloy_{id}` when blank |
| `canvas` | object | Opaque editor state for the (future) Factory UI; no backend semantics |

**`WorkflowMember`**:

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | string | The member's immutable agent identifier |
| `role` | `"supervisor"` \| `"specialist"` | Exactly one member must be `supervisor` |
| `delegation_hint` | string? | Expertise hint shown to the supervisor in the `delegate_to` tool description |

Validation (on create/update) requires: a valid `id` pattern, exactly one supervisor, and
every `agent_id` resolving to an existing profile in `data/agent_profiles.yaml`.

## Delegation at Runtime

When a workflow is active, the supervisor gains a single extra tool, **`delegate_to`**:

```json
{
  "agent_id": "specialist-agent-id",
  "task": "A self-contained task description with all the context the specialist needs"
}
```

The `AlloyExecutor` (`alloy/executor.py`) intercepts the call, resolves the specialist's
profile, and spawns a fresh `Agent` for it. Specialists run **in isolation** — they receive
only the delegated task plus relevant memories from the shared channel, never the full
conversation history. Specialists do **not** receive the `delegate_to` tool, so they cannot
re-delegate. The specialist's full output is returned to the supervisor as a tool result,
which the supervisor then synthesizes into its reply.

All members read and write the shared channel (`_alloy_{workflow_id}`), so delegated results
and extracted facts are visible across the workflow. Each delegation also creates a child
`Goal` (Phase 15 goal tracking) linked to that channel.

### Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `alloy.max_delegation_depth` | `3` | Maximum delegation nesting depth |
| `alloy.allow_adhoc_delegation` | `true` | Allow `delegate_to` + the roster prompt in ordinary (workflow-less) chats. Safe default: the roster itself is opt-in per profile (`available_for_delegation` defaults `false`) |
| `alloy.delegation_timeout_seconds` | — | Per-delegation timeout |

## Execution & Streaming

Workflows execute through the existing streaming chat endpoint — pass a `workflow_id`:

```
POST /api/agent/chat/stream
```
```json
{
  "message": "Research and summarize the latest on X",
  "agent_profile_id": "some-profile",
  "workflow_id": "my-workflow",
  "session_id": "optional"
}
```

When `workflow_id` is set, the supervisor profile becomes the active agent, its
`memory_channel` is switched to the workflow's `shared_channel`, and the `delegate_to` tool is
added with an enum of the workflow's specialists. Without `workflow_id`, behavior is unchanged
(single-agent mode).

In addition to the standard chat SSE events, delegations emit:

| Event | Key fields |
|-------|------------|
| `delegation_start` | `delegation_id`, `target_agent_id`, `task`, `depth`, `supervisor_agent_id`, `shared_channel` |
| `delegation_chunk` | `delegation_id`, `target_agent_id`, `content` (streamed specialist tokens) |
| `delegation_tool_call` | `delegation_id`, `target_agent_id`, `tool`, `tool_call_id`, `arguments` |
| `delegation_tool_result` | `delegation_id`, `target_agent_id`, `tool`, `content`, `success`, `duration_ms` |
| `delegation_complete` | `delegation_id`, `target_agent_id`, `status`, `error`, `result_preview`, `exhibits?` (wires for exhibits produced inside the delegation, cap 5) |

Multimodal specialist output passes through: a specialist's `exhibit` and
`workspace_attached` events are forwarded **top-level, unchanged** (the image/exhibit
renders under the delegation card, and workspace auto-attach works exactly like the main
loop). The delegating agent's tool result carries an "already displayed" note so it
doesn't re-invent image URLs, and the wires are persisted as `present_exhibit` turns so
reloaded conversations rebuild the cards. Specialist internal tools run under the
specialist's own identity (usage attribution) with the conversation's user/workspace
inherited, and the specialist profile's `allowed_tools`/`blocked_tools` gating applies
inside delegations.

## Storage & API

Workflows are persisted to `data/workflows.yaml` and managed by the `WorkflowManager`
singleton (mirroring `ProfileManager`). CRUD is exposed under `/api/alloy/workflows` — see
[Multi-Agent endpoints](../api/endpoints.md#multi-agent-agent-alloy).

On the client, `AlloyWorkflowContext` (`contexts/AlloyWorkflowContext.tsx`) manages the
workflow list and selection; the active workflow's id is passed to the chat stream.

## Ambassador

An ambassador is an agent profile with `kind: 'ambassador'`. Ambassadors are excluded from chat:
the default agent, routing and `delegate_to` all filter on `kind == 'agent'`.

Its **no-pollution invariant** is load-bearing: the ambassador writes only to a Redis **sidecar**
(the `amb_thread:` family), never to `conversation_logs` or `conv_summary:`. Its tool belt never
executes a write:
- **Reads run on their own:** `summarize_conversation`, `explore_conversation`,
  `read_conversation`, `list_conversations`, `survey_conversations`, `search_conversations`,
  `read_conversation_state`, `read_conversation_results`, `list_active_runs`, `list_agents`,
  `usage_report`, `recall_memory`.
- **Writes are proposals:** `rename_conversation`, `archive_conversation`,
  `delete_conversation` and `dispatch_task` return a proposal that the client shows as a confirm
  strip. The change lands only when you confirm, and a relayed or dispatched message lands as
  your own user turn.

Voice goes through the neutral speech seam (`kit/speech.py`), using a provider's
`/audio/speech` (TTS) and `/audio/transcriptions` (STT). A transcript routes through the same
answer core as typed text, so voice gets the same tools and continuity. Per-profile settings live
in the profile's `ambassador` block: personas, speech and voice models, and verbosity.

## Status

Shipped (v1 plus the routing/delegation waves through v0.21.5, then later waves):

- Data model + `WorkflowManager`, the `delegate_to` tool, `AlloyExecutor`, and specialist isolation
- Live `delegation_*` SSE streaming, goal-tracking integration, the workflow CRUD API, and the client context
- **Parallel / fan-out delegation** with a client trace/replay modal (per-delegation tokens, cost, and wall-clock)
- **Per-turn attribution** — `agent_id` on turns and `conversation_logs`, restored to display names on reload
- **Explicit routing** — `target_agent_id` on the chat-stream request (priority `workflow_id > target_agent_id > agent_profile_id > default`)
- **Per-agent tool isolation** — `allowed_tools` / `blocked_tools` enforced per profile
- **Ad-hoc agent-to-agent delegation** — `delegate_to` in workflow-less chats, gated by `alloy.allow_adhoc_delegation` (depth-limited, no self-delegation)
- **First-class conversational delegation** — a `delegation_roster` system-prompt block (teammates + specialties from the opt-in roster, same source as the tool enum), per-profile `delegation_hint`, opt-in `available_for_delegation` (default off), the global gate default-ON, and a per-conversation Solo/Team toggle (`disable_delegation`)
- **@-mention routing** — an inline `@agent-id` / `@name` routes a turn; `AgentParticipant` graph nodes + a client `@`-autocomplete composer
- **Organizations** — managers, roles and the chain of command (`v0.21.240`), and the **Organization** surface: the whole org as one chart, with team editing beside it (`v0.21.280`), and arranging it by dragging (`v0.21.283`) and quick edits of an agent's role, roster and specialty (`v0.21.284`)
- **Ambassador** — the read-only tool belt with proposal-only actions, Inquiry threads, the Command Deck and two-way voice (`v0.21.233`–`v0.21.236`)

Deferred (see [Roadmap → Phase 16](../roadmap.md)): editing the organization chart directly
(drawing reporting lines), execution of declarative `routes` (accepted and stored but ignored), async (background)
delegation, and per-workflow tool subsetting.
