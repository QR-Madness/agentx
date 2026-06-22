"""
Ambassador tool belt — a small, curated, **read-only** set of tools the ambassador
may call to observe the conversation world (Phase 16.7, Slice 2).

This is deliberately *separate* from the main agent's MCP / internal tools: nothing
here can write a transcript, send a message, or mutate state. Every tool is
SELECT-only over the durable conversation store, so the load-bearing no-pollution
invariant is enforced at this boundary — the ambassador can *look*, never *touch*.

The tools map to the user's intents:
  * ``summarize_conversation`` — "can you summarize this?"
  * ``explore_conversation``  — "can you explore more on this?" (a deeper dig, optionally on a topic)
  * ``read_conversation``     — read a *specific* conversation (used after a survey)
  * ``list_conversations``    — enumerate recent sessions ("what have my agents discovered?")
  * ``list_agents``           — the agent roster: who the agents are, their roles, and
                                **what each one's model can do** (modalities — image/audio/
                                vision/tools) — the input for multi-modal routing.

``execute_tool`` dispatches by name, is **read-only**, and **never raises** — a
failure returns a short human string the model can read, so a tool hiccup never
breaks the ambassador's turn.
"""

from __future__ import annotations

import logging
from typing import Any

from .conversation_history import list_recent_conversations, load_recent_labeled_turns

logger = logging.getLogger(__name__)

# Tool outputs re-enter the model context, so cap how much transcript a single
# read returns (rough token budget; the reader trims oldest-first).
_READ_TOKEN_BUDGET = 3000
_SURVEY_SNIPPET = 160
# Each agent's role blurb in the roster is capped so a long system prompt can't
# blow the tool output.
_ROSTER_BLURB = 200


# --- OpenAI-style function schemas (advertised to tool-capable models) --------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "summarize_conversation",
            "description": (
                "Read the conversation you're watching and summarize what's happened. "
                "Use this when the person asks you to summarize, recap, or catch them up. "
                "Omit conversation_id to summarize the active conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "A specific conversation to summarize; omit for the active one.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_conversation",
            "description": (
                "Read the conversation and dig deeper into it — what the agent found, "
                "the reasoning, what's still open. Use this when the person asks you to "
                "explore, dig in, or go deeper. Pass a topic to focus the dig."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional focus for the deeper look (a subject, a turn, a thread).",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "A specific conversation; omit for the active one.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_conversation",
            "description": (
                "Read the recent transcript of a specific conversation by id. Use this "
                "after list_conversations to look into one of the sessions it returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "The conversation to read.",
                    }
                },
                "required": ["conversation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_conversations",
            "description": (
                "List the person's recent conversations (their agents' sessions), newest "
                "first, with a short preview of each. Use this for cross-conversation "
                "questions like 'what have my agents discovered?' — then read the ones "
                "that look relevant with read_conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many recent conversations to list (default 20, max 50).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": (
                "List the person's agents (the roster), with each agent's role and — "
                "importantly — what its model can do: input/output modalities and whether "
                "it supports tools, vision, speech (audio out), or transcription (audio in). "
                "Use this to answer 'who are my agents?', 'what can <name> do?', or to pick "
                "the right agent for a task — especially a multi-modal one ('which agent can "
                "handle an image / audio?')."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


# --- Read helpers ------------------------------------------------------------


def _render_transcript(conversation_id: str, fallback_label: str) -> str:
    """Render a conversation read-only, labelling each assistant turn with **its own**
    producing agent (``metadata.agent_name``) so the ambassador never mislabels one
    conversation's agent with another's. ``fallback_label`` is used only for unstamped
    assistant turns — pass the active agent's name for the watched conversation, and
    an empty string (→ generic "Agent") for any *other* conversation."""
    rows = load_recent_labeled_turns(conversation_id, token_budget=_READ_TOKEN_BUDGET)
    if not rows:
        return "(That conversation is empty — nothing has happened in it yet.)"
    lines = []
    for role, content, agent_name in rows:
        if role == "user":
            who = "You"
        else:
            who = (agent_name or "").strip() or (fallback_label or "Agent")
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def _render_survey(limit: int) -> str:
    convs = list_recent_conversations(limit)
    if not convs:
        return "(No conversations found.)"
    lines = []
    for c in convs:
        cid = c.get("conversation_id", "")
        title = (c.get("first_user") or "Untitled").strip().replace("\n", " ")
        if len(title) > _SURVEY_SNIPPET:
            title = title[:_SURVEY_SNIPPET].rstrip() + "…"
        last = (c.get("last_message") or "").strip().replace("\n", " ")
        if len(last) > _SURVEY_SNIPPET:
            last = last[:_SURVEY_SNIPPET].rstrip() + "…"
        count = c.get("message_count", 0)
        when = c.get("last_at", "")
        agents = (c.get("agents") or "").strip()
        piece = f"- id={cid} · {count} messages · last active {when}"
        if agents:
            piece += f"\n    agent(s): {agents}"
        piece += f"\n    topic: {title}"
        if last:
            piece += f"\n    latest: {last}"
        lines.append(piece)
    return "Recent conversations (newest first):\n" + "\n".join(lines)


def _blurb_for(profile: Any) -> str:
    """A short capability blurb: the profile's own ``description`` when set, else the
    first paragraph of its ``system_prompt``, length-capped. Never the full prompt."""
    desc = (getattr(profile, "description", None) or "").strip()
    if not desc:
        prompt = (getattr(profile, "system_prompt", None) or "").strip()
        desc = prompt.split("\n\n", 1)[0].strip() if prompt else ""
    desc = desc.replace("\n", " ").strip()
    if len(desc) > _ROSTER_BLURB:
        desc = desc[:_ROSTER_BLURB].rstrip() + "…"
    return desc


def _flag(ok: bool) -> str:
    return "✓" if ok else "✗"


def _caps_line(model: str, caps: Any) -> str:
    """A compact, routing-relevant capability line for one model. Pure formatting —
    reports only what the provider reported (never asserts an unstated capability)."""
    in_mods = ", ".join(getattr(caps, "input_modalities", None) or ["text"])
    out_mods = ", ".join(getattr(caps, "output_modalities", None) or ["text"])
    return (
        f"model: {model} · in: {in_mods} · out: {out_mods} · "
        f"tools {_flag(getattr(caps, 'supports_tools', False))} · "
        f"vision {_flag(getattr(caps, 'supports_vision', False))} · "
        f"speech {_flag(getattr(caps, 'supports_speech', False))} · "
        f"transcribe {_flag(getattr(caps, 'supports_transcription', False))}"
    )


def _render_roster() -> str:
    """List the agent profiles (``kind == 'agent'`` only — ambassadors are excluded
    everywhere) with role, delegation availability, role blurb, and each agent's live
    model capabilities resolved from the provider. Capability resolution degrades
    **per agent**, so one unresolvable model never drops the agent from the roster."""
    from ..providers.registry import get_registry
    from .profiles import get_profile_manager

    agents = get_profile_manager().list_profiles_by_kind("agent")
    if not agents:
        return "(No agent profiles found.)"

    registry = get_registry()
    blocks: list[str] = []
    for p in agents:
        head = f"- {p.name} (id={p.agent_id})"
        if getattr(p, "is_default", False):
            head += " · primary"
        rows = [head]

        tags = [t.strip() for t in (getattr(p, "tags", None) or []) if t and t.strip()]
        if tags:
            rows.append(f"    role: {', '.join(tags)}")
        rows.append(
            f"    delegation: {'available' if getattr(p, 'available_for_delegation', False) else 'not available'}"
        )

        model = (getattr(p, "default_model", None) or "").strip()
        if not model:
            # The live floor is the agent profile's model, threaded per-turn; there's
            # no clean global-default accessor, so don't guess — just say so.
            rows.append("    model: (inherits the default model)")
        else:
            try:
                provider, model_id, _ = registry.resolve_with_fallback(model)
                rows.append("    " + _caps_line(model, provider.get_capabilities(model_id)))
            except Exception as e:  # noqa: BLE001 — one bad model never drops its agent
                logger.warning(f"list_agents: capabilities for '{model}' unavailable: {e}")
                rows.append(f"    model: {model} · (capabilities unavailable)")

        blurb = _blurb_for(p)
        if blurb:
            rows.append(f"    about: {blurb}")
        blocks.append("\n".join(rows))

    return "Your agents:\n" + "\n".join(blocks)


# --- Dispatch ----------------------------------------------------------------


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    focused_conversation_id: str,
    agent_name: str = "",
) -> str:
    """Execute one ambassador tool, read-only. Returns a string for the model.

    ``focused_conversation_id`` is the conversation the ambassador is *discussing*
    (the subject the no-id tools default to) — not "the one active agent". Agent
    names come per-turn from the conversation itself (``metadata.agent_name``);
    ``agent_name`` is only a fallback label for unstamped turns of the focused
    conversation. Never raises — a bad/unknown call returns a readable note so the
    agentic loop stays alive."""
    args = arguments if isinstance(arguments, dict) else {}
    focused_label = (agent_name or "").strip()

    def _fallback_for(cid: str) -> str:
        # Only the focused conversation may borrow the passed label for an unstamped
        # turn; any *other* conversation falls back to a generic label so its turns
        # are never mislabelled as the focused conversation's agent.
        return focused_label if cid == focused_conversation_id else ""

    try:
        if name in ("summarize_conversation", "explore_conversation"):
            cid = (args.get("conversation_id") or "").strip() or focused_conversation_id
            if not cid:
                return "(There's no conversation open to read yet.)"
            body = _render_transcript(cid, _fallback_for(cid))
            topic = (args.get("topic") or "").strip()
            if name == "explore_conversation" and topic:
                return f"(Focus the deeper look on: {topic})\n\n{body}"
            return body
        if name == "read_conversation":
            cid = (args.get("conversation_id") or "").strip()
            if not cid:
                return "(read_conversation needs a conversation_id — use list_conversations first.)"
            return _render_transcript(cid, _fallback_for(cid))
        if name == "list_conversations":
            try:
                limit = int(args.get("limit") or 20)
            except (TypeError, ValueError):
                limit = 20
            return _render_survey(limit)
        if name == "list_agents":
            return _render_roster()
    except Exception as e:  # noqa: BLE001 — a tool never breaks the loop
        logger.warning(f"ambassador tool '{name}' failed: {e}")
        return f"(The {name} tool couldn't complete: {str(e)[:160]})"
    return f"(Unknown tool: {name})"
