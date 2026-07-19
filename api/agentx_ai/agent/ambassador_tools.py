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
  * ``list_conversations``    — enumerate recent sessions (a quick index)
  * ``survey_conversations``  — a digest-rich cross-conversation view (each session's own
                                running summary when it has one) for an application-wide
                                "what have my agents been working on?" summary
  * ``list_agents``           — the agent roster: who the agents are, their roles, and
                                **what each one's model can do** (modalities — image/audio/
                                vision/tools) — the input for multi-modal routing.
  * ``rename_inquiry``        — set the CURRENT Inquiry's own title (the sole **write**: it
                                touches only the ambassador's thread meta, never a conversation)

``execute_tool`` dispatches by name and **never raises** — a failure returns a short
human string the model can read, so a tool hiccup never breaks the ambassador's turn.
Every tool is read-only over the conversation world; the lone exception is
``rename_inquiry``, which writes only the ambassador's *own* Inquiry title (its Redis
sidecar meta), so the no-pollution guarantee is intact.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import get_config_manager
from .conversation_history import list_recent_conversations, load_recent_labeled_turns
from .conversation_summary_storage import get_summary

logger = logging.getLogger(__name__)

# Tool outputs re-enter the model context, so cap how much transcript a single
# read returns (rough token budget; the reader trims oldest-first).
_READ_TOKEN_BUDGET = 3000
_SURVEY_SNIPPET = 160
# A conversation's rolling summary is already condensed, but cap it so a survey
# across many conversations can't blow the tool output.
_SURVEY_SUMMARY = 600
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
                "first, with a short preview of each — a quick index. Use this to find a "
                "specific session, then read it with read_conversation. For a digest-rich "
                "'what have my agents been working on?' picture, use survey_conversations."
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
            "name": "survey_conversations",
            "description": (
                "Survey the person's recent conversations and compose an application-wide "
                "summary — 'what have my agents been working on / discovered lately?'. Each "
                "conversation comes with its own running summary when it has one (longer "
                "sessions), so you get the gist across many conversations at once without "
                "reading each transcript. Prefer this over list_conversations for "
                "cross-conversation questions; use read_conversation to dig into one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "How many recent conversations to survey (default 12, max 30).",
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
    {
        "type": "function",
        "function": {
            "name": "rename_inquiry",
            "description": (
                "Rename the current Inquiry (this thread) to a short, descriptive title that "
                "reflects what it's about — do this once you know the focus, especially in the "
                "command deck where Inquiries are named workspaces. Retitles only your own "
                "Inquiry; it never renames or touches the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The new Inquiry title (a few words).",
                    }
                },
                "required": ["title"],
            },
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


def _custom_title(c: dict) -> str:
    """The user-set conversation title (``conversation_meta``), clipped — ``""``
    when the conversation only has a derived topic."""
    title = (c.get("title") or "").strip().replace("\n", " ")
    if len(title) > _SURVEY_SNIPPET:
        title = title[:_SURVEY_SNIPPET].rstrip() + "…"
    return title


def _render_survey(limit: int) -> str:
    convs = list_recent_conversations(limit)
    if not convs:
        return "(No conversations found.)"
    lines = []
    for c in convs:
        cid = c.get("conversation_id", "")
        title = _custom_title(c) or (c.get("first_user") or "Untitled").strip().replace("\n", " ")
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


_GOAL_MARK = {"completed": "✓", "active": "◷", "abandoned": "✗", "blocked": "⊘"}
_MAX_GOALS = 4
_GOAL_DESC = 80


def _conversation_goals_line(conversation_id: str) -> str:
    """A best-effort ``goals: …`` line for a conversation — what its agent set out to do
    (completed/active/…). Reads the memory graph, which (unlike the Postgres/Redis reads)
    fails fast when Neo4j is down, so this is wrapped to degrade to ``""`` — the survey
    must never depend on it. Only goals created after conversation-stamping shipped carry
    a ``conversation_id``; older ones simply won't appear."""
    try:
        from ..kit.memory_utils import get_agent_memory

        memory = get_agent_memory(user_id="default")  # the user the agent writes goals under
        if memory is None:
            return ""
        goals = memory.get_goals_for_conversation(conversation_id)
        if not goals:
            return ""
        parts = []
        for g in goals[:_MAX_GOALS]:
            desc = " ".join((g.description or "").split())
            if len(desc) > _GOAL_DESC:
                desc = desc[:_GOAL_DESC].rstrip() + "…"
            parts.append(f"{_GOAL_MARK.get(g.status, '•')} {desc}")
        return "    goals: " + " · ".join(parts)
    except Exception as e:  # noqa: BLE001 — goals are optional; never sink the survey
        logger.debug(f"survey goals unavailable for {conversation_id}: {e}")
        return ""


def _conv_fingerprint(c: dict) -> str:
    """A change key for a conversation's aide-digest cache: bumps when it grows."""
    return f"{c.get('message_count', 0)}:{c.get('last_at', '')}"


def _render_deep_survey(limit: int) -> str:
    """A digest-rich cross-conversation view: each recent conversation with its own
    rolling summary (``get_summary`` — already condensed) when it has one, else an
    **aide digest** (a cheap model condenses that one conversation read-only), else the
    first/last snippet, plus each conversation's goals (best-effort). Lets the ambassador
    compose an application-wide summary without reading each transcript into its own
    context (map-reduce: aides map, the ambassador reduces). Read-only; never raises."""
    convs = list_recent_conversations(limit)
    if not convs:
        return "(No conversations found.)"

    # Pass 1: resolve each conversation's rolling summary once; the ones without a
    # summary (and with content) are aide candidates.
    summaries = {c.get("conversation_id", ""): (get_summary(c.get("conversation_id", "")) or "").strip()
                 for c in convs}
    aide_digests: dict[str, str] = {}
    candidates = [
        (c.get("conversation_id", ""), _conv_fingerprint(c))
        for c in convs
        if not summaries.get(c.get("conversation_id", "")) and (c.get("message_count", 0) or 0) > 0
    ]
    if candidates:
        try:
            from .aide_swarm import get_aide_service

            aide = get_aide_service()
            if aide.enabled:
                cap = get_config_manager().get("ambassador.aide.max_per_survey", 8)
                if len(candidates) > cap:
                    logger.info(
                        f"survey aide fan-out capped at {cap}/{len(candidates)} "
                        "un-summarized conversations (rest use snippets)"
                    )
                aide_digests = aide.digest_many_sync(candidates)
        except Exception as e:  # noqa: BLE001 — the swarm never sinks the survey
            logger.debug(f"aide swarm unavailable for survey: {e}")

    # Pass 2: render. Priority per conversation: rolling summary → aide digest → snippet.
    lines = []
    for c in convs:
        cid = c.get("conversation_id", "")
        count = c.get("message_count", 0)
        when = c.get("last_at", "")
        agents = (c.get("agents") or "").strip()
        named = _custom_title(c)
        piece = f"- id={cid} · {count} messages · last active {when}"
        if named:
            piece += f'\n    title: "{named}"'
        if agents:
            piece += f"\n    agent(s): {agents}"

        summary = summaries.get(cid, "").replace("\n", " ")
        digest = aide_digests.get(cid, "").replace("\n", " ") if not summary else ""
        if summary:
            if len(summary) > _SURVEY_SUMMARY:
                summary = summary[:_SURVEY_SUMMARY].rstrip() + "…"
            piece += f"\n    summary: {summary}"
        elif digest:
            if len(digest) > _SURVEY_SUMMARY:
                digest = digest[:_SURVEY_SUMMARY].rstrip() + "…"
            piece += f"\n    digest: {digest}"
        else:
            # No rolling summary or aide digest — fall back to the opening + latest
            # snippet, like the quick index.
            title = (c.get("first_user") or "Untitled").strip().replace("\n", " ")
            if len(title) > _SURVEY_SNIPPET:
                title = title[:_SURVEY_SNIPPET].rstrip() + "…"
            last = (c.get("last_message") or "").strip().replace("\n", " ")
            if len(last) > _SURVEY_SNIPPET:
                last = last[:_SURVEY_SNIPPET].rstrip() + "…"
            piece += f"\n    topic: {title}"
            if last:
                piece += f"\n    latest: {last}"

        goals_line = _conversation_goals_line(cid)
        if goals_line:
            piece += f"\n{goals_line}"
        lines.append(piece)
    return (
        "Survey of recent conversations (newest first; compose an application-wide "
        "summary from these):\n" + "\n".join(lines)
    )


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
    user_id: str = "default",
) -> str:
    """Execute one ambassador tool, read-only. Returns a string for the model.

    ``focused_conversation_id`` is the conversation the ambassador is *discussing*
    (the subject the no-id tools default to) — not "the one active agent". Agent
    names come per-turn from the conversation itself (``metadata.agent_name``);
    ``agent_name`` is only a fallback label for unstamped turns of the focused
    conversation. ``user_id`` scopes the user-keyed reads (active runs, memory
    recall). Never raises — a bad/unknown call returns a readable note so the
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
            topic = (args.get("topic") or "").strip()
            focus = topic if name == "explore_conversation" else ""
            # Prefer a cheap aide digest so the raw transcript never enters the
            # ambassador's context — fall back to the raw read when the swarm is
            # off/unavailable or returns nothing.
            try:
                from .aide_swarm import get_aide_service

                aide = get_aide_service()
                if aide.enabled:
                    digest = aide.digest_conversation_sync(cid, focus=focus, label=_fallback_for(cid))
                    if digest:
                        return digest
            except Exception as e:  # noqa: BLE001 — fall back to the raw read
                logger.debug(f"aide digest fell back to raw read for {cid}: {e}")
            body = _render_transcript(cid, _fallback_for(cid))
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
        if name == "survey_conversations":
            try:
                limit = int(args.get("limit") or 12)
            except (TypeError, ValueError):
                limit = 12
            limit = max(1, min(limit, 30))  # heavier rows than the quick index → tighter cap
            return _render_deep_survey(limit)
        if name == "list_agents":
            return _render_roster()
        if name == "rename_inquiry":
            # The belt's only write — and self-scoped: it titles the ambassador's OWN
            # current Inquiry (its sidecar meta), never the conversation.
            title = (args.get("title") or "").strip()
            if not title:
                return "(rename_inquiry needs a title.)"
            if not focused_conversation_id:
                return "(There's no Inquiry open to rename.)"
            from .ambassador_storage import set_thread_title

            set_thread_title(focused_conversation_id, title, auto=True)
            return f"(Renamed this Inquiry to “{title[:80]}”.)"
    except Exception as e:  # noqa: BLE001 — a tool never breaks the loop
        logger.warning(f"ambassador tool '{name}' failed: {e}")
        return f"(The {name} tool couldn't complete: {str(e)[:160]})"
    return f"(Unknown tool: {name})"
