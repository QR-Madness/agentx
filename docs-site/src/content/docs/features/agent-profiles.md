# Agent Profiles

An **agent profile** is the configuration that defines one agent — its name and face, the
model it runs on, *how it thinks*, what it remembers, and what it's allowed to touch. One
profile produces one agent. AgentX ships with a ready **AgentX** agent, and you build the
rest: a careful planner, a web researcher, an image maker, a blunt critic — each a profile.

Everything else in the platform hangs off profiles: [Agent Teams](/docs/features/multi-agent)
delegate between them, [Memory](/docs/features/memory) is scoped by them, and the
[thinking pattern](/docs/features/reasoning) an agent uses is set on them. So this is the
first surface worth learning.

!!! tip "Where profiles live"
    Profiles are stored on **your** API server, not the client. Whether you open AgentX as
    the desktop app or the hosted web client at **agx.thejpnet.net** connected to your own
    server, you're editing the same profiles.

## Open the editor

Open the profile editor from the command palette (`⌘K` / `Ctrl K` → **Agent profiles**) or by
clicking the agent's icon in the top bar.

You'll see your profiles alongside the one you're editing. Each carries the agent's **name**,
**avatar**, trait **tags**, a one-line **description**, and its immutable **agent id** — a
Docker-style handle like `giddy-witty-falcon` that never changes even when you rename the agent, so
teammates and memory keep pointing at the right one.

Edits **autosave** as you type; a brand-new profile gets an explicit **Create** instead.

## Core — identity, model, thinking, prompt

The **Core** tab is where day-to-day tuning happens.

### Model

Leave it on **System default** to inherit the global default model, or pick a specific
`provider:model` (e.g. an OpenRouter reasoning model, a local LM Studio model). This is how
you give one agent a big reasoning model and another a fast, cheap one. See
[Providers](/docs/features/providers) for the model catalog and how resolution + fallback work.

### Generation

The **Temperature** slider runs from **Focused** (deterministic, good for analysis and
planning) to **Creative** (looser, good for brainstorming and drafting). `0.7` is a balanced
default.

### Thinking pattern

This is the headline control — *how the agent reasons before it answers*:

| Pattern | When to reach for it |
|---------|----------------------|
| **Auto** | Let AgentX pick per message (the sensible default) |
| **Native** | Trust the model's own thinking; no added scaffold |
| **Step-by-step** | Force explicit chain-of-thought |
| **Step-back** | Distill the governing principles first, then answer |
| **Reflection** | Draft, critique, revise |
| **Deep reflection** | Watch the draft + self-critique stream live before the improved answer |
| **Consensus** | Sample several solutions, keep the agreement (good for math/logic) |

Set it per profile here, or override it for a single conversation in the
[Relay](/docs/features/chat) <span class="ax-icon ax-icon--brain" aria-hidden="true"></span> — see
[Reasoning](/docs/features/reasoning) for the full behavior.

### System prompt

This is the agent's persona and standing instructions.

- **Base template** *(optional)* — start from a saved prompt template.
- **Agent instructions** — the agent's own voice and rules, woven into the composed prompt
  *after* the global layers. Keep it about *this* agent ("You reason carefully and show your
  working; you never assert a conclusion you can't justify"), not platform-wide rules.
- **Enhance** rewrites a rough draft into a sharper prompt; **Insert from library** pulls in a
  reusable snippet; **Effective prompt preview** shows exactly what the model will receive
  (your instructions + the global layers + tools), with a live token count.

The composition rules — the global layer stack, ordering, and templates — are covered in
[Prompts](/docs/features/prompts).

### Team membership

Flip **Join the team roster** on to let *other* agents hand this one subtasks, and write a
one-line **Specialty** so teammates know what it's good at ("Turns messy findings into a tight,
cited brief"). Off by default — nothing delegates to an agent until it opts in. This is the
front door to [Agent Teams](/docs/features/multi-agent).

## Tools — what the agent can reach

The **Tools** tab controls the agent's hands. Turn tools on or off wholesale, then optionally
**gate** them: an allow-list restricts the agent to specific tools, and a block-list removes
them (block wins over allow). This is how you keep a "planner" tool-free so it thinks instead
of acting, while a "researcher" gets web search and nothing else. Tools come from your
connected servers and the built-ins — see [Connectors & Tools](/docs/features/mcp).

## Advanced — memory and direct mode

- **Memory channel** — which memory scope this agent reads and writes (`_global` by default).
  Every agent also has a private `_self_` channel for self-knowledge. Point two agents at the
  same channel to have them share a memory pool; give one its own channel to keep it separate.
  See [Memory](/docs/features/memory).
- **Direct mode** — bypass the whole harness: the model sees *only* your message — no system
  prompt, no memory, no tools. Best for a pure transform (a fast classifier or rewriter) or an
  image model. It's forced on automatically for image-only models.

## Deleting a profile

For any profile other than the default, a **Delete** button floats in the corner of the
editor. The default agent can't be deleted (something has to answer). Deletions of seeded
agents *stick* — they won't come back on the next launch.

## What ships by default

Fresh installs seed three agents, then get out of your way:

- **AgentX** — a balanced general assistant (the default; can't be deleted).
- **Researcher** — web search + cited answers, already on the team roster.
- **Deluxe Image Creator** — hand it a visual brief and the finished image lands in the chat;
  also on the roster.

Seeds are one-time — edit or delete them freely and they won't re-seed. Set any agent as the
default from its profile.

!!! note "Ambassador profiles"
    A profile's **kind** is normally `agent`. The other kind, **ambassador**, is a companion
    that briefs *you* about a conversation on the side (by text or voice) without ever posting
    into the transcript. It's edited here too, but its "system prompt" becomes *communication*
    personas. See [Agent Teams → Ambassador](/docs/features/multi-agent#ambassador--the-parallel-relay).

## Day-to-day

- **Switch agents mid-conversation** from the composer's **agent chip** — the active agent's
  avatar shows there. Different questions, different agents.
- **Build a small bench, not one do-everything agent.** A focused planner (step-back thinking,
  no tools), a researcher (web tools, on the roster), a writer (creative temperature) beats a
  single overloaded profile — and it sets you up for [Agent Teams](/docs/features/multi-agent).
- **`@-mention`** an agent by name or id to route one turn to it.
