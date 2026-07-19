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
  * ``read_conversation_state`` — the structured working state (goals/decisions/open threads/
                                artifacts + digest): the pre-condensed "where does this stand?"
                                dashboard; survey → state → transcript is the drill-down ladder
  * ``search_conversations``  — lexical full-text search across ALL conversations
                                ("which conversation discussed X?" beyond the recent window)
  * ``list_active_runs``      — the live half: streaming runs + queued/running background
                                jobs ("what are my agents doing right now?")
  * ``usage_report``          — spend/token analysis over the usage ledger (totals +
                                per-model/agent/source)
  * ``recall_memory``         — what the agents have *learned* (facts/entities/goals) —
                                knowledge, not logs
  * ``list_agents``           — the agent roster: who the agents are, their roles, and
                                **what each one's model can do** (modalities — image/audio/
                                vision/tools) — the input for multi-modal routing.
  * ``rename_inquiry``        — set the CURRENT Inquiry's own title (the sole self-scoped
                                **write**: it touches only the ambassador's thread meta,
                                never a conversation)

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
    {
        "type": "function",
        "function": {
            "name": "read_conversation_state",
            "description": (
                "Read a conversation's structured working state — its goals, decisions, "
                "open threads, artifacts and narrative, plus the rolling digest of aged-out "
                "turns. The conversation's own pre-condensed dashboard: cheaper and sharper "
                "than reading the transcript for 'where does this stand?'. Drill-down ladder: "
                "survey_conversations → read_conversation_state → read_conversation. Omit "
                "conversation_id for the active one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "A specific conversation; omit for the active one.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_conversations",
            "description": (
                "Search ALL of the person's conversations by keywords — 'which conversation "
                "discussed X?' beyond the recent window. Bare words must all match, "
                "\"quoted phrases\" match adjacently, -word excludes. Returns matching "
                "conversations with highlighted snippets; drill in with read_conversation. "
                "This searches transcripts (what was said); for what the agents have "
                "*learned*, use recall_memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords / \"quoted phrase\" / -excluded terms.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max conversations to return (default 8, max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_active_runs",
            "description": (
                "See what the person's agents are doing RIGHT NOW: live streaming runs and "
                "queued/running background jobs (including dispatched tasks), with status, "
                "the message being worked, and age. Use for 'what are my agents doing?', "
                "'did my dispatch start / finish?'. Recent finished/failed runs appear too."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max runs/jobs per source (default 10, max 25).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "usage_report",
            "description": (
                "Report model spend and token usage from the usage ledger: totals plus "
                "per-model / per-agent / per-source breakdowns over a window. Answers "
                "'what did this conversation cost?', 'which agent burns the most?', "
                "'what have I spent this week?'. Pass conversation_id (or 'this') to "
                "scope to one conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Window in days (default 7, max 90).",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Scope to one conversation ('this' = the active one); omit for everything.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Ask the agents' long-term memory what they actually KNOW about something — "
                "learned facts, entities, active goals — distinct from transcript search "
                "(what was merely said). Use for 'what do my agents know about X?', "
                "'what has been learned about Y?'. Knowledge, not logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to recall (a subject, entity, or question).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many memories to recall (default 6, max 12).",
                    },
                    "channel": {
                        "type": "string",
                        "description": "A specific memory channel; omit for the shared (_global) one.",
                    },
                },
                "required": ["query"],
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
        state_line = _conversation_state_line(cid)
        if state_line:
            piece += f"\n{state_line}"
        lines.append(piece)
    return (
        "Survey of recent conversations (newest first; compose an application-wide "
        "summary from these):\n" + "\n".join(lines)
    )


def _conversation_state_line(conversation_id: str) -> str:
    """A best-effort ``state: …`` survey hint from the structured working state
    (open-threads count + freshest decision). Same doctrine as the goals line:
    optional, never sinks the survey."""
    try:
        from .conversation_state_storage import get_state

        state = get_state(conversation_id)
        if state.is_empty():
            return ""
        bits = []
        open_threads = state.entries("open_threads")
        if open_threads:
            plural = "s" if len(open_threads) != 1 else ""
            bits.append(f"{len(open_threads)} open thread{plural}")
        decisions = state.entries("decisions")
        if decisions:
            latest = " ".join((decisions[-1].text or "").split())
            if len(latest) > 80:
                latest = latest[:80].rstrip() + "…"
            bits.append(f"latest decision: {latest}")
        if not bits:
            return ""
        return "    state: " + " · ".join(bits)
    except Exception as e:  # noqa: BLE001 — the hint is optional; never sink the survey
        logger.debug(f"survey state hint unavailable for {conversation_id}: {e}")
        return ""


def _render_state(conversation_id: str) -> str:
    """The full structured-state dashboard for one conversation."""
    from .conversation_state_storage import render_state_block

    block = render_state_block(conversation_id)
    if not block:
        return (
            "(No structured state recorded for this conversation yet — "
            "it builds up as the conversation grows.)"
        )
    return block


def _render_search(query: str, limit: int) -> str:
    """Grouped full-text hits across all conversations, snippets bolded."""
    from .conversation_history import search_conversation_logs

    results = search_conversation_logs(query, limit=limit)
    if not results:
        return f'(No conversations matched "{query}" — try fewer or different words.)'
    lines = [f'Conversations matching "{query}" (best match first):']
    for r in results:
        piece = f"- id={r['conversation_id']}"
        if r.get("title"):
            piece += f" · \"{r['title']}\""
        if r.get("last_at"):
            piece += f" · last active {r['last_at']}"
        if r.get("agents"):
            piece += f" · agent(s): {r['agents']}"
        for s in r.get("snippets", []):
            who = "You" if s.get("role") == "user" else (s.get("role") or "agent").capitalize()
            piece += f"\n    {who}: …{s.get('snippet', '')}…"
        lines.append(piece)
    lines.append("(Drill into one with read_conversation.)")
    return "\n".join(lines)


_RUN_MARK = {"running": "▶", "queued": "⧗", "done": "✓", "failed": "✗", "cancelled": "⊘"}


def _age_of(iso: str) -> str:
    """Human age from an ISO timestamp; degrades to the raw string."""
    try:
        from datetime import UTC, datetime

        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        secs = max(0, int((datetime.now(UTC) - dt).total_seconds()))
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:  # noqa: BLE001 — cosmetic only
        return iso or ""


def _render_active_runs(user_id: str, limit: int) -> str:
    """Live streaming runs + queued/running background jobs, merged. Each source
    degrades independently (a down Redis half must not hide the other). The
    ambassador's own sidecar runs are unindexed (``indexed=False``) so they never
    self-report here."""
    from .conversation_history import latest_agent_name

    runs: list[dict] = []
    jobs: list[dict] = []
    try:
        from ..streaming.chat_run import store as run_store

        runs = run_store.list_runs(user_id, limit=limit)
    except Exception as e:  # noqa: BLE001 — degrade to the other source
        logger.debug(f"active-runs: run registry unavailable: {e}")
    try:
        from ..background.chat_jobs import list_background_chats

        jobs = list_background_chats(user_id, limit=limit)
    except Exception as e:  # noqa: BLE001 — degrade to the other source
        logger.debug(f"active-runs: background jobs unavailable: {e}")

    def _agent_label(session_id: str) -> str:
        try:
            return latest_agent_name(session_id)
        except Exception:  # noqa: BLE001 — label is cosmetic
            return ""

    def _row(kind: str, status: str, message: str, session_id: str, created_at: str) -> tuple:
        live = status in ("running", "queued")
        mark = _RUN_MARK.get(status, "•")
        msg = " ".join((message or "").split())
        if len(msg) > 80:
            msg = msg[:80].rstrip() + "…"
        piece = f"- {mark} {status} {kind}"
        agent = _agent_label(session_id) if session_id else ""
        if agent:
            piece += f" · agent: {agent}"
        if session_id:
            piece += f" · conversation {session_id}"
        if created_at:
            piece += f" · started {_age_of(created_at)}"
        if msg:
            piece += f"\n    task: {msg}"
        return (0 if live else 1, piece)

    rows = [
        _row("run", r.get("status", ""), r.get("message", ""), r.get("session_id", ""),
             r.get("created_at", ""))
        for r in runs
    ] + [
        _row("background job", j.get("status", ""), j.get("message", ""),
             j.get("session_id", ""), j.get("created_at", ""))
        for j in jobs
    ]
    if not rows:
        return "(No active runs — everything's quiet right now.)"
    rows.sort(key=lambda t: t[0])
    live_count = sum(1 for t in rows if t[0] == 0)
    header = (
        f"Active right now: {live_count}" if live_count
        else "Nothing running right now; recent finished runs:"
    )
    return header + "\n" + "\n".join(p for _, p in rows)


def _render_usage_report(days: int, conversation_id: str) -> str:
    """Compact spend/token report from the usage ledger."""
    from .usage_ledger import aggregate_usage

    report = aggregate_usage(days, conversation_id=conversation_id or None)
    t = report["totals"]
    scope = f"conversation {conversation_id}" if conversation_id else "all activity"
    if not t["turns"]:
        return f"(No metered usage for {scope} in the last {report['days']} day(s).)"
    lines = [
        f"Usage over the last {report['days']} day(s) — {scope}:",
        (
            f"- total: {t['turns']} metered calls · {t['tokens_total']:,} tokens "
            f"({t['tokens_input']:,} in / {t['tokens_output']:,} out) · "
            f"{t['cost_total']:.4f} {t['cost_currency']}"
        ),
    ]
    for label, key, rows in (
        ("model", "model", report["by_model"]),
        ("agent", "agent_id", report["by_agent"]),
        ("source", "source", report["by_source"]),
    ):
        top = rows[:3]
        if top:
            lines.append(
                f"- top by {label}: " + " · ".join(
                    f"{r[key]}: {r['cost_total']:.4f} ({r['tokens_total']:,} tok)" for r in top
                )
            )
    return "\n".join(lines)


def _render_recall(query: str, top_k: int, channel: str, user_id: str) -> str:
    """What the agents' long-term memory knows — knowledge, not logs. Unlike the
    Postgres/Redis reads, ``remember`` can raise (and costs an embedder
    round-trip), so this wraps its own degrade for a friendlier note (the
    ``_conversation_goals_line`` doctrine)."""
    try:
        from ..kit.memory_utils import get_agent_memory

        memory = get_agent_memory(user_id=user_id, channel="_global")
        if memory is None:
            return "(The memory system is unavailable right now.)"
        bundle = memory.remember(
            query,
            top_k=top_k,
            channels=[channel] if channel else ["_global"],
        )
        text = bundle.to_context_string(turn_char_limit=400, max_turns=4).strip()
        if not text:
            return f'(The agents\' memory holds nothing on "{query}" yet.)'
        # Character-level clip aligned with the transcript read budget.
        if len(text) > _READ_TOKEN_BUDGET * 4:
            text = text[: _READ_TOKEN_BUDGET * 4].rstrip() + "…"
        return f'What the agents\' memory holds on "{query}":\n{text}'
    except Exception as e:  # noqa: BLE001 — memory must degrade, never sink the turn
        logger.debug(f"recall_memory degraded: {e}")
        return f"(Memory recall is unavailable right now: {str(e)[:120]})"


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
        if name == "read_conversation_state":
            cid = (args.get("conversation_id") or "").strip() or focused_conversation_id
            if not cid:
                return "(There's no conversation open to read state from.)"
            return _render_state(cid)
        if name == "search_conversations":
            query = (args.get("query") or "").strip()
            if not query:
                return "(search_conversations needs a query.)"
            try:
                limit = int(args.get("limit") or 8)
            except (TypeError, ValueError):
                limit = 8
            return _render_search(query, max(1, min(limit, 20)))
        if name == "list_active_runs":
            try:
                limit = int(args.get("limit") or 10)
            except (TypeError, ValueError):
                limit = 10
            return _render_active_runs(user_id, max(1, min(limit, 25)))
        if name == "usage_report":
            try:
                days = int(args.get("days") or 7)
            except (TypeError, ValueError):
                days = 7
            cid = (args.get("conversation_id") or "").strip()
            if cid.lower() in ("this", "current", "active"):
                cid = focused_conversation_id
            return _render_usage_report(max(1, min(days, 90)), cid)
        if name == "recall_memory":
            query = (args.get("query") or "").strip()
            if not query:
                return "(recall_memory needs a query.)"
            try:
                top_k = int(args.get("top_k") or 6)
            except (TypeError, ValueError):
                top_k = 6
            channel = (args.get("channel") or "").strip()
            return _render_recall(query, max(1, min(top_k, 12)), channel, user_id)
    except Exception as e:  # noqa: BLE001 — a tool never breaks the loop
        logger.warning(f"ambassador tool '{name}' failed: {e}")
        return f"(The {name} tool couldn't complete: {str(e)[:160]})"
    return f"(Unknown tool: {name})"
