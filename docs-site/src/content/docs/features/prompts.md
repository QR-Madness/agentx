# Prompts

Every turn, an agent runs on a **system prompt** — the instructions that define who it is and how
it behaves. AgentX composes that prompt fresh each turn from a few durable layers, so you can
shape an agent's voice and rules without touching code, and change them without breaking anything
downstream.

## How a prompt is composed

The final system prompt is assembled, in order, from:

1. **The global prompt** — your core persona and house rules, applied to *every* agent and
   conversation.
2. **The auto-generated tools prompt** — a description of the [connected tools](mcp.md), built for
   you whenever MCP servers are attached.
3. **The agent profile's sections** — the selected [profile's](agent-profiles.md) enabled
   sections, in their configured order.
4. **Injected context** — recalled [memory](memory.md) and any request-specific overrides.

A request can also carry a full **system override** that replaces the whole composed prompt for
that one turn. See the
[composition flow](../architecture/system-design.md#system-prompt-composition) on the System
Design page.

## The global prompt

The global prompt is the one voice that persists across every profile switch — the core persona
and the rules you always want honored. It carries no agent name of its own; each agent's name is
injected from its profile, so a single global prompt serves every agent. Edit it in the **Prompt
Library**.

## Profile sections

Where the global prompt is universal, **profile sections** tailor an individual agent. A profile
is an ordered stack of typed, individually toggleable sections — switch one off to drop it,
reorder them to shift emphasis:

| Type | Purpose |
|------|---------|
| `persona` | Identity and personality |
| `task` | Task-specific instructions |
| `format` | Output-format requirements |
| `constraints` | Behavioral rules and guardrails |
| `examples` | Few-shot examples |
| `context` | Background information |
| `custom` | Anything else |

Because sections are durable and layered, you can keep a set of reusable building blocks and
compose different agents from them. The **Prompt Library** is where you build and preview all of
it — and **Enhance prompt** (in the [Relay](chat.md)) can draft or sharpen a section for you.

## Under the hood

The `PromptManager` seeds sensible defaults, loads your customizations from
`data/system_prompts.yaml`, and saves changes back as you edit — the whole layered stack is
durable. The tools prompt is regenerated from the live tool list each turn. The programmatic
surface — profiles, sections, the global prompt, and a compose-preview — is in the
[API Reference](../api/endpoints.md#prompts).

## Related

- [Agent Profiles](agent-profiles.md) — sections live on a profile
- [Chat](chat.md) — how the composed prompt enters a turn
- [Connectors & Tools](mcp.md) — the auto-generated tools prompt
