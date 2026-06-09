"""
Ambassador service — a dedicated agent that runs *parallel* to a conversation
and briefs the user on a single turn, acting as the middleman of information
between the conversation and the user.

It is built on the existing (reliable) agent-profile system: an ambassador is
any :class:`AgentProfile` carrying an ``ambassador`` section, which tunes *how*
it briefs (the profile still supplies identity/model/temperature/persona). The
global default ambassador profile is ``config.ambassador.profile_id`` (falling
back to the default profile).

No-pollution invariant (load-bearing — keep these true):
  * Writes ONLY to the ``ambassador:`` Redis sidecar (``ambassador_storage``) —
    never ``conversation_logs`` (no ``store_turn`` / ``_persist_turns``) and
    never ``conv_summary:``.
  * Its SSE events are namespaced ``ambassador_*`` and are never consumed by the
    main tool loop / session.
  * Its conversation read is ``SELECT``-only (``load_recent_turns``); nothing it
    produces re-enters the main agent's turn context or rehydration.

Bulletproofing: a missing/unreachable provider degrades gracefully (one
``ambassador_done`` with ``empty_provider`` status — never raises), so the main
conversation is never affected. Briefings token-stream via ``provider.stream``;
spoken briefings (OpenRouter TTS) ride :meth:`AmbassadorService.synthesize` off
the profile's ``ambassador.speech_model``/``voice`` block (the speak endpoint
degrades to a clean 422 when no speech provider is configured). The user-speaks
half of voice mode (STT) is a separate, later seam.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

from ..config import get_config_manager
from ..providers.base import Message, MessageRole, SpeechResult
from ..providers.registry import ProviderRegistry, get_registry
from . import ambassador_storage as store
from .conversation_history import load_recent_turns

logger = logging.getLogger(__name__)


class SpeechUnavailable(Exception):
    """Raised when spoken briefings can't be produced (no provider/model
    configured, or the resolved model doesn't support speech). Carries a stable
    ``code`` so the view can return a structured 422 the client can act on."""

    def __init__(self, message: str, *, code: str = "voice_unconfigured") -> None:
        super().__init__(message)
        self.code = code

# Rough token budget per grounding turn (mirrors conversation_history estimates).
_TOKENS_PER_TURN = 400

# Built-in model floor — used only when neither the settings override nor the
# chosen profile specify a model. resolve_with_fallback degrades from here.
_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Voice (TTS) defaults. The shipped speech model is OpenRouter-only, so spoken
# briefings need an OpenRouter key — the speak endpoint degrades gracefully when
# it's absent. MAI-Voice-2 voices use the Azure locale format.
_DEFAULT_SPEECH_MODEL = "openrouter:microsoft/mai-voice-2"
_DEFAULT_SPEECH_VOICE = "en-US-Harper:MAI-Voice-2"
# Speech-to-text (the user-speaks half). OpenRouter-only, like TTS — the
# transcribe endpoint degrades gracefully when no key is configured.
_DEFAULT_TRANSCRIPTION_MODEL = "openrouter:openai/whisper-1"
# Guard against pathological uploads (short push-to-talk clips are tiny).
_MAX_AUDIO_BYTES = 25 * 1024 * 1024
# Hard ceiling on synthesized text (TTS bills per character; briefings/Q&A are
# short + markdown-free by persona rules, so this only guards pathological input).
_MAX_SPEECH_CHARS = 4000

def _default_persona(agent_name: str = "") -> str:
    """The Ambassador's core voice. ``agent_name`` (the *briefed* agent's display
    name, when known) lets it speak about the agent by name instead of the flat,
    transcript-narrator 'the assistant'."""
    agent_ref = agent_name.strip() or "your agent"
    name_rule = (
        f"- The agent in the conversation is named {agent_name.strip()}. Call it by that "
        f"name, or 'your agent'. "
        if agent_name.strip()
        else "- Call the agent 'your agent'. "
    )
    return (
        "You are an Ambassador. The person you're talking to is sitting in a conversation "
        "with an AI agent, watching it work — and you lean over to tell THEM, directly, "
        "what just happened in the agent's latest turn. You speak TO that person. You are "
        "not a participant: you don't answer or continue the conversation, you only tell "
        "them what their agent just did.\n\n"

        "WHO IS WHO — this is the most important thing to get right:\n"
        "- The person you're talking to is 'you'. Address them in the second person, always. "
        "NEVER call them 'the user' — narrating 'the user asked…' to their own face is a "
        "report, not a person talking. They are 'you'.\n"
        f"{name_rule}NEVER call it 'the assistant', 'the AI', 'the model', or a bare 'it' "
        "the way a transcript would.\n\n"

        "HOW TO SPEAK:\n"
        "Talk the way a person actually talks out loud to a friend beside them. Tell them what "
        f"{agent_ref} found, did, decided, or asked — the real substance — in plain, flowing "
        "sentences. They already know what they themselves said, so don't recite their own "
        "message back to them; lead with what the agent did with it. If their request matters "
        "for context, touch it lightly in passing, never restate it in full.\n\n"

        "Do NOT narrate the turn as an event you watched from the outside. No 'the agent came "
        "back with', no 'it then asked', no 'it broke those down into'. No play-by-play. And no "
        "filler color — drop empty adjectives like 'a couple of solid directories', 'handy "
        "categories', 'a nice breakdown'. Just say what's actually there, cleanly and warmly. "
        "Never invent anything that wasn't in the turn; the surrounding context is only to help "
        "you understand it, not to recap it.\n\n"

        "If the turn shows you what the agent actually DID — a web search it ran, sources it "
        "pulled, a table or diagram it built — work that into your words in plain terms ('it "
        "searched for X and turned up the county index and the downtown association', 'it laid "
        "the results out in a table'). That substance is often the real point of the turn. But "
        "stay spoken: never dump raw results, never paste URLs, never read out a list.\n\n"

        "FORM (hard rules): NO markdown, NO headings, NO labels like 'Briefing:', NO bullet "
        "points, NO numbered lists, NO bold or asterisks. One flowing spoken voice — lean and "
        "natural, never stiff, never templated.\n\n"

        "LENGTH (hard rule): a briefing is a glance, not a recap. Keep it SHORT and obey the "
        "length limit you're given below. Think as much as you need to get it right, but what "
        "you actually say to the person must stay brief — when you've said the heart of it, stop."
    )


def _qa_persona(agent_name: str = "") -> str:
    """The Ambassador's voice when answering a free-form question about the
    conversation (vs. briefing a single turn). Same identity + form rules; the
    task is to answer, grounded only in what actually happened."""
    agent_ref = agent_name.strip() or "their agent"
    name_rule = (
        f"- The agent in the conversation is named {agent_name.strip()}. Call it by that "
        f"name, or 'your agent'. "
        if agent_name.strip()
        else "- Call the agent 'your agent'. "
    )
    return (
        "You are an Ambassador. The person you're talking to is watching a conversation "
        "between themselves and an AI agent, and they have asked YOU a question about it. "
        "Answer them directly. You are the interpreter standing beside them — not a "
        "participant in the conversation, and you never continue or answer it yourself.\n\n"

        "WHO IS WHO — get this right:\n"
        "- The person you're talking to is 'you'. Address them in the second person. NEVER "
        "call them 'the user'.\n"
        f"{name_rule}NEVER call it 'the assistant', 'the AI', 'the model', or a bare 'it' "
        "like a transcript.\n\n"

        f"GROUND YOUR ANSWER: answer only from what {agent_ref} and the person actually said "
        "and did in the conversation you're given. Be concrete — name the sources, tools, or "
        "moments that matter. If the answer isn't in the conversation, say so plainly ('I "
        "don't see that in what they did') — never invent, guess, or answer the underlying "
        "question yourself as if you were the agent.\n\n"

        "HOW TO SPEAK: like a person talking out loud to a friend beside them — plain, natural, "
        "flowing sentences. Keep it as short as the question allows; answer what was asked, then "
        "stop.\n\n"

        "FORM (hard rules): NO markdown, NO headings, NO labels, NO bullet points, NO numbered "
        "lists, NO bold or asterisks. One flowing spoken voice."
    )


def _draft_persona(agent_name: str = "") -> str:
    """The Ambassador's voice when *drafting a message the user will send to the
    agent*. Unlike briefings/Q&A (the ambassador's own voice), the draft is written
    in the user's first person — it becomes the user's own turn — so the ambassador
    is a ghostwriter here, not a speaker."""
    agent_ref = agent_name.strip() or "the agent"
    return (
        "You are the Ambassador, helping the person beside you send a message to "
        f"{agent_ref} in the conversation they're watching. Turn their rough intent into "
        "a clear, well-formed message — written in the FIRST PERSON as the person "
        "themselves (it will be sent as their OWN message, not yours), direct and "
        "specific, using the conversation for context so the agent knows what they mean.\n\n"

        "Output ONLY the message text — no preamble, no quotation marks, no 'Here's a "
        "draft', no sign-off, no markdown. Just the message, ready to send. Keep it natural "
        "and concise; never invent requirements the person didn't imply."
    )


def _voice_command_persona(agent_name: str = "") -> str:
    """The Ambassador's voice when interpreting a spoken **voice command** in voice
    mode. It decides whether the person is asking the ambassador a question (→ it
    answers) or instructing the agent (→ it drafts a relay for the user to send),
    and replies as strict JSON so the client can route it."""
    agent_ref = agent_name.strip() or "the agent"
    return (
        "You are the Ambassador, on a live voice call beside the person while they watch a "
        f"conversation with their AI agent ({agent_ref}). They just spoke to you. Decide what "
        "they want and respond with ONLY a JSON object — nothing else.\n\n"

        "Two possible actions:\n"
        '- "answer" — they are asking YOU a question, or want information/explanation ABOUT the '
        "conversation (what happened, what the agent found or did, a clarification). You answer it "
        "yourself, grounded only in the conversation.\n"
        f'- "relay" — they are giving an instruction meant for {agent_ref}: something they want it '
        'to DO, or a message to pass along ("tell it to…", "ask the agent to…", "have it use X"). '
        "You turn it into a clear first-person message FROM the person TO the agent, ready to send.\n\n"

        'Respond with exactly: {"action": "answer" | "relay", "text": "..."}\n'
        '- For "answer": "text" is your spoken reply — plain, conversational, no markdown, no lists.\n'
        '- For "relay": "text" is the first-person message to the agent — direct, ready to send, no '
        "preamble or quotation marks.\n"
        'When genuinely unsure, prefer "answer". Output nothing but the JSON object.'
    )


def _sub_placeholders(text: str, *, agent_name: str = "") -> str:
    """Apply the whitelisted prompt placeholders ({agent_name}/{date}/{time})."""
    from ..prompts.placeholders import substitute_placeholders
    return substitute_placeholders(text, agent_name=agent_name)


def _persona_override(amb, field: str) -> Optional[str]:
    """A non-blank functional-persona override on the ambassador config, else None."""
    if not amb:
        return None
    val = getattr(amb, field, None)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


_VERBOSITY_HINT = {
    "brief": "LENGTH LIMIT: one or two sentences — just the heart of it. Then stop.",
    "normal": "LENGTH LIMIT: a short spoken paragraph, three or four sentences at most. Do not exceed it.",
    "deep": "LENGTH LIMIT: at most two short paragraphs — walk through the reasoning and what's left open, then stop.",
}

# Ambassadors think freely, so the token budget must leave generous headroom for
# the model's *reasoning* on top of the (deliberately short) visible briefing —
# otherwise a thinking model (e.g. Gemini) spends the budget reasoning and the
# answer truncates mid-sentence. Length is controlled by the prompt's LENGTH
# LIMIT, NOT by a tight token cap. `_THINKING_HEADROOM` is the reasoning room; the
# per-verbosity value is the room for the visible answer on top of it.
_THINKING_HEADROOM = 2048
_VERBOSITY_TOKENS = {"brief": 256, "normal": 512, "deep": 1024}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class AmbassadorService:
    """Briefs conversation turns on demand, in parallel and without pollution."""

    def __init__(self) -> None:
        self._registry: Optional[ProviderRegistry] = None

    @property
    def registry(self) -> ProviderRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def _config(self) -> dict[str, Any]:
        config = get_config_manager()
        return {
            "enabled": config.get("ambassador.enabled", True),
            "profile_id": config.get("ambassador.profile_id", None),
            "model": config.get("ambassador.model") or None,
            "max_context_turns": config.get("ambassador.max_context_turns", 8),
            # Slice 2 agentic tool loop — ON by default (this is an experimental app).
            # The loop degrades internally to the grounded one-shot on any failure
            # (e.g. a model that can't take tools), so it can't hard-fail a turn;
            # `ambassador.tools_enabled=false` is the kill-switch.
            "tools_enabled": config.get("ambassador.tools_enabled", True),
            # Bound on agentic tool rounds before forcing a final answer.
            "max_tool_rounds": config.get("ambassador.max_tool_rounds", 4),
            # Optional hard ceiling. Left unset by default so a thinking model has
            # room to reason without the visible briefing getting clipped — length
            # is governed by the prompt's LENGTH LIMIT, not by this cap.
            "max_tokens": config.get("ambassador.max_tokens", None),
        }

    def _max_tokens(self, profile, cfg: dict[str, Any]) -> int:
        """Token budget = thinking headroom + room for the (short) visible answer.

        Ambassadors think freely, so the cap must accommodate reasoning *plus* the
        briefing; otherwise a thinking model spends the budget reasoning and the
        answer truncates mid-sentence. The visible length stays short via the
        prompt's LENGTH LIMIT, not via this cap. An explicit ``ambassador.max_tokens``
        setting, if present, is honored as a hard ceiling."""
        amb = getattr(profile, "ambassador", None) if profile else None
        verbosity = getattr(amb, "verbosity", "normal") if amb else "normal"
        answer_room = _VERBOSITY_TOKENS.get(verbosity, _VERBOSITY_TOKENS["normal"])
        budget = _THINKING_HEADROOM + answer_room
        ceiling = cfg.get("max_tokens")
        return min(budget, ceiling) if ceiling else budget

    def _resolve_profile(self, profile_id: Optional[str]):
        """Pick the ambassador profile: explicit id → **default ambassador** → None.

        (An ambassador is now its own profile kind; briefings use the default
        ambassador. `None` is fine — the persona builders fall back to the shipped
        code defaults, so a briefing never hard-fails.)"""
        from .profiles import get_profile_manager

        pm = get_profile_manager()
        profile = None
        if profile_id:
            profile = pm.get_profile(profile_id)
        if profile is None:
            profile = pm.get_default_ambassador()
        return profile

    def _current_run_id(self) -> Optional[str]:
        """Best-effort: the detached runner sets this ambiently per run."""
        try:
            from ..streaming.status import current_run_id

            return current_run_id.get()
        except Exception:
            return None

    def _build_qa_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        base = _persona_override(amb, "qa_persona") or _qa_persona(agent_name)
        parts = [base]
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Personality:\n{profile.system_prompt.strip()}")
        if amb and getattr(amb, "briefing_prompt", "").strip():
            parts.append(f"Additional instructions:\n{amb.briefing_prompt.strip()}")
        return _sub_placeholders("\n\n".join(parts), agent_name=agent_name)

    def _build_tools_persona(self, profile, agent_name: str = "") -> str:
        """The Q&A persona plus a note that the ambassador has read-only tools to
        look at conversations — so it can fetch what it needs (summarize/explore the
        active conversation, or survey across the person's sessions) before it
        answers, instead of relying on a single pre-stuffed transcript."""
        base = self._build_qa_persona(profile, agent_name)
        tools_note = (
            "YOU HAVE TOOLS. You can look at the conversation world read-only before you "
            "answer:\n"
            "- summarize_conversation — read the conversation you're watching to summarize it.\n"
            "- explore_conversation — read it and dig deeper (optionally on a topic).\n"
            "- list_conversations — list the person's recent sessions (for 'what have my agents "
            "found across everything?').\n"
            "- read_conversation — read one specific conversation by id (after listing).\n\n"
            "Call a tool when you need to see something; you may call several. When you have what "
            "you need, answer the person directly in your own spoken voice — never read tool "
            "output back verbatim, never mention the tools. These tools only READ; you never "
            "change anything."
        )
        return f"{base}\n\n{tools_note}"

    def _build_draft_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        base = _persona_override(amb, "draft_persona") or _draft_persona(agent_name)
        parts = [base]
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Personality:\n{profile.system_prompt.strip()}")
        return _sub_placeholders("\n\n".join(parts), agent_name=agent_name)

    def _build_qa_prompt(
        self,
        *,
        question: str,
        context: str,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> str:
        agent_label = agent_name.strip() or "the agent"
        sections = []
        if context.strip():
            sections.append(
                "The conversation so far (this is ALL you may answer from):\n"
                f"{context.strip()}"
            )
        else:
            sections.append(
                "The conversation is empty so far — nothing has happened in it yet, so "
                "there's nothing to answer from."
            )
        artifacts_block = self._render_artifacts(artifacts, agent_label)
        if artifacts_block:
            sections.append(artifacts_block)
        sections.append(f"The person asks you:\n{question.strip()}")
        sections.append(
            "Answer their question directly, in your own spoken voice, grounded only in "
            "the conversation above. If there's nothing in the conversation yet (or the "
            "answer isn't there), say so plainly — never invent an answer."
        )
        return "\n\n".join(sections)

    def _build_draft_prompt(
        self,
        *,
        intent: str,
        context: str,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> str:
        agent_label = agent_name.strip() or "the agent"
        sections = []
        if context.strip():
            sections.append("The conversation so far (for context):\n" f"{context.strip()}")
        artifacts_block = self._render_artifacts(artifacts, agent_label)
        if artifacts_block:
            sections.append(artifacts_block)
        sections.append(
            "What the person wants to say (their rough intent):\n" f"{intent.strip()}"
        )
        sections.append(
            f"Write their message to {agent_label} now — first person, ready to send."
        )
        return "\n\n".join(sections)

    def _build_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        base = _persona_override(amb, "briefing_persona") or _default_persona(agent_name)
        parts = [base]
        # The profile's own system prompt is the ambassador's "Communications"
        # (personality) voice — it colors how it speaks.
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Personality:\n{profile.system_prompt.strip()}")
        verbosity = getattr(amb, "verbosity", "normal") if amb else "normal"
        parts.append(_VERBOSITY_HINT.get(verbosity, _VERBOSITY_HINT["normal"]))
        if amb and getattr(amb, "briefing_prompt", "").strip():
            parts.append(f"Additional briefing instructions:\n{amb.briefing_prompt.strip()}")
        return _sub_placeholders("\n\n".join(parts), agent_name=agent_name)

    def _render_artifacts(self, artifacts: Optional[dict], agent_label: str) -> str:
        """Compact, prose-able summary of what the agent *did* this turn (tools it
        ran, sources it pulled, exhibits it built) — grounding beyond the reply."""
        if not artifacts:
            return ""
        lines: list[str] = []

        for tool in (artifacts.get("tools") or [])[:12]:
            name = (tool.get("name") or "a tool").strip()
            piece = f"ran {name}"
            detail = (tool.get("detail") or "").strip()
            if detail:
                piece += f": {detail}"
            if tool.get("ok") is False:
                piece += " (which failed)"
            result = (tool.get("result") or "").strip()
            if result:
                piece += f" → {result}"
            lines.append(piece)

        sources = artifacts.get("sources") or []
        rendered_sources = []
        for src in sources[:12]:
            label = (src.get("label") or "").strip()
            url = (src.get("url") or "").strip()
            if label and url:
                rendered_sources.append(f"{label} ({url})")
            elif label or url:
                rendered_sources.append(label or url)
        if rendered_sources:
            lines.append("pulled sources: " + "; ".join(rendered_sources))

        for ex in (artifacts.get("exhibits") or [])[:8]:
            kind = (ex.get("kind") or "an artifact").strip()
            piece = f"presented a {kind}"
            title = (ex.get("title") or "").strip()
            if title:
                piece += f" ({title})"
            detail = (ex.get("detail") or "").strip()
            if detail:
                piece += f" — {detail}"
            lines.append(piece)

        if not lines:
            return ""
        body = "\n".join(f"- {ln}" for ln in lines)
        return (
            f"What {agent_label} actually did this turn — facts to ground your briefing "
            "(weave them into your own spoken words; do NOT list them back or read out URLs):\n"
            f"{body}"
        )

    def _build_turn_prompt(
        self,
        *,
        user_text: str,
        assistant_text: str,
        context: str,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> str:
        agent_label = agent_name.strip() or "Your agent"
        sections = []
        if context.strip():
            sections.append(
                "Recent conversation context (for grounding only — do not brief this):\n"
                f"{context.strip()}"
            )
        turn = "--- The turn to brief ---\n"
        if user_text.strip():
            turn += f"You said:\n{user_text.strip()}\n\n"
        turn += f"{agent_label} replied:\n{assistant_text.strip()}"
        sections.append(turn)
        artifacts_block = self._render_artifacts(artifacts, agent_label)
        if artifacts_block:
            sections.append(artifacts_block)
        sections.append(
            "Now tell the person, directly and in your own spoken voice, what just "
            "happened in that turn."
        )
        return "\n\n".join(sections)

    def _thread_history(
        self, conversation_id: str, *, exclude_id: str = "", limit: int = 8
    ) -> list[Message]:
        """Prior ambassador Q&A turns as conversation history, so the ambassador
        remembers the conversation it is having *with the user* (continuity) — its
        own thread, parallel to and never mixed into the main transcript.

        Read-only over the ``qa:`` sidecar (which also captures spoken voice
        answers), settled turns only, oldest-first, capped. The in-flight turn is
        excluded by id. Never raises — continuity is best-effort, so a Redis hiccup
        just yields a one-shot answer instead of failing the turn."""
        try:
            items = store.list_qa(conversation_id)
        except Exception as e:  # pragma: no cover - Redis offline
            logger.debug(f"ambassador thread history load failed: {e}")
            return []
        done = [
            q
            for q in items
            if q.get("qa_id") != exclude_id
            and q.get("status") == "done"
            and (q.get("answer") or "").strip()
        ]
        done.sort(key=lambda q: q.get("created_at") or "")
        msgs: list[Message] = []
        for q in done[-limit:]:
            question = (q.get("question") or "").strip()
            answer = (q.get("answer") or "").strip()
            if question:
                msgs.append(Message(role=MessageRole.USER, content=question))
            if answer:
                msgs.append(Message(role=MessageRole.ASSISTANT, content=answer))
        return msgs

    def _grounding_context(
        self, conversation_id: str, max_turns: int, agent_name: str = ""
    ) -> str:
        """Read-only recent transcript for grounding. Empty on any failure."""
        try:
            msgs = load_recent_turns(
                conversation_id, token_budget=max_turns * _TOKENS_PER_TURN
            )
        except Exception as e:  # pragma: no cover - DB offline
            logger.debug(f"ambassador grounding load failed: {e}")
            return ""
        agent_label = agent_name.strip() or "Agent"
        lines = []
        for m in msgs[-(max_turns * 2):]:
            role = "You" if m.role == MessageRole.USER else agent_label
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    async def brief_turn(
        self,
        conversation_id: str,
        message_id: str,
        *,
        assistant_text: str,
        user_text: str = "",
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Brief one conversation turn. Yields ``ambassador_*`` SSE events.

        Never raises: a failure settles the sidecar status and emits an error
        event so the parallel run (and the main conversation) stay healthy.
        """
        cfg = self._config()
        run_id = self._current_run_id()
        store.set_status(conversation_id, message_id, "streaming", run_id=run_id)
        yield _sse("ambassador_start", {"message_id": message_id, "run_id": run_id})

        if not cfg["enabled"]:
            msg = "The ambassador is disabled in settings."
            store.set_summary(conversation_id, message_id, msg, status="error")
            yield _sse("ambassador_error", {"message_id": message_id, "error": "disabled"})
            return

        profile = self._resolve_profile(cfg["profile_id"])
        # Precedence: the explicit settings model (Settings → Ambassador) wins;
        # else the chosen profile's model; else the built-in floor.
        profile_model = getattr(profile, "default_model", None) if profile else None
        model = cfg["model"] or profile_model or _DEFAULT_MODEL
        temperature = getattr(profile, "temperature", 0.2) if profile else 0.2

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(
                model, preferred_fallback=_DEFAULT_MODEL
            )
        except Exception as e:  # noqa: BLE001 — any resolution failure degrades gracefully
            logger.warning(f"Ambassador provider unavailable: {e}")
            note = "No model provider is configured for the ambassador."
            store.set_summary(conversation_id, message_id, note, status="empty_provider")
            yield _sse(
                "ambassador_done",
                {"message_id": message_id, "status": "empty_provider", "summary": note},
            )
            return

        context = self._grounding_context(
            conversation_id, cfg["max_context_turns"], agent_name
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._build_persona(profile, agent_name),
            ),
            Message(
                role=MessageRole.USER,
                content=self._build_turn_prompt(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    context=context,
                    agent_name=agent_name,
                    artifacts=artifacts,
                ),
            ),
        ]

        # Token-stream the briefing (progressive reveal + replay on re-attach),
        # persisting deltas to the sidecar and settling on cancel/error.
        async for ev in self._stream_and_settle(
            item_id=message_id,
            provider=provider,
            model_id=model_id,
            temperature=temperature,
            max_tokens=self._max_tokens(profile, cfg),
            messages=messages,
            on_chunk=lambda t: store.append_chunk(conversation_id, message_id, t),
            on_done=lambda s: store.set_summary(conversation_id, message_id, s, status="done"),
            on_cancel=lambda: store.set_status(
                conversation_id, message_id, "cancelled", run_id=run_id
            ),
            on_error=lambda e: store.set_status(
                conversation_id, message_id, "error", run_id=run_id, error=e
            ),
            empty_text="(The ambassador returned an empty briefing.)",
            log_label="briefing",
        ):
            yield ev

    async def answer_question(
        self,
        conversation_id: str,
        qa_id: str,
        question: str,
        *,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Answer a free-form question about the conversation. Yields ``ambassador_*``
        SSE (keyed by ``qa_id`` in the ``message_id`` field, so the client SSE pump is
        shared with briefings). Never raises; persists only to the Q&A sidecar."""
        cfg = self._config()
        run_id = self._current_run_id()
        store.create_qa(conversation_id, qa_id, question, run_id=run_id)
        yield _sse("ambassador_start", {"message_id": qa_id, "run_id": run_id})

        if not cfg["enabled"]:
            store.set_qa_answer(
                conversation_id, qa_id, "The ambassador is disabled in settings.", status="error"
            )
            yield _sse("ambassador_error", {"message_id": qa_id, "error": "disabled"})
            return

        profile = self._resolve_profile(cfg["profile_id"])
        profile_model = getattr(profile, "default_model", None) if profile else None
        model = cfg["model"] or profile_model or _DEFAULT_MODEL
        temperature = getattr(profile, "temperature", 0.2) if profile else 0.2

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(
                model, preferred_fallback=_DEFAULT_MODEL
            )
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning(f"Ambassador provider unavailable: {e}")
            note = "No model provider is configured for the ambassador."
            store.set_qa_answer(conversation_id, qa_id, note, status="empty_provider")
            yield _sse(
                "ambassador_done",
                {"message_id": qa_id, "status": "empty_provider", "summary": note},
            )
            return

        # Slice 2: the agentic, read-only tool path (gated by `ambassador.tools_enabled`).
        # It settles the sidecar itself and degrades internally to the grounded
        # one-shot below, so flipping the flag on can never hard-fail a turn.
        if cfg["tools_enabled"]:
            async for ev in self._answer_with_tools(
                conversation_id=conversation_id,
                qa_id=qa_id,
                question=question,
                provider=provider,
                model_id=model_id,
                temperature=temperature,
                max_tokens=self._max_tokens(profile, cfg),
                max_rounds=cfg["max_tool_rounds"],
                profile=profile,
                agent_name=agent_name,
                artifacts=artifacts,
                run_id=run_id,
            ):
                yield ev
            return

        # Q&A grounds on a wider window than a single-turn brief — the question can
        # reference anything in the conversation.
        context = self._grounding_context(
            conversation_id, cfg["max_context_turns"] * 2, agent_name
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._build_qa_persona(profile, agent_name),
            ),
            # Prior Q&A as real dialogue turns — the ambassador's own conversation
            # with the user, so a follow-up ("what about the second one?") has
            # context. The current (streaming) turn is excluded by id.
            *self._thread_history(conversation_id, exclude_id=qa_id),
            Message(
                role=MessageRole.USER,
                content=self._build_qa_prompt(
                    question=question,
                    context=context,
                    agent_name=agent_name,
                    artifacts=artifacts,
                ),
            ),
        ]

        async for ev in self._stream_and_settle(
            item_id=qa_id,
            provider=provider,
            model_id=model_id,
            temperature=temperature,
            max_tokens=self._max_tokens(profile, cfg),
            messages=messages,
            on_chunk=lambda t: store.append_qa_chunk(conversation_id, qa_id, t),
            on_done=lambda s: store.set_qa_answer(conversation_id, qa_id, s, status="done"),
            on_cancel=lambda: store.set_qa_status(
                conversation_id, qa_id, "cancelled", run_id=run_id
            ),
            on_error=lambda e: store.set_qa_status(
                conversation_id, qa_id, "error", run_id=run_id, error=e
            ),
            empty_text="(The ambassador had no answer.)",
            log_label="Q&A",
        ):
            yield ev

    async def _answer_with_tools(
        self,
        *,
        conversation_id: str,
        qa_id: str,
        question: str,
        provider,
        model_id: str,
        temperature: float,
        max_tokens: int,
        max_rounds: int,
        profile,
        agent_name: str,
        artifacts: Optional[dict],
        run_id: Optional[str],
    ) -> AsyncGenerator[str, None]:
        """Agentic Q&A: let the ambassador call its **read-only** tool belt to look
        at the conversation world, then answer in its own voice. Settles the qa
        sidecar itself and emits ``ambassador_tool_call``/``ambassador_tool_result``
        SSE as it works. Never raises — any failure degrades to a grounded one-shot
        answer over the same context (so enabling the loop can't break a turn)."""
        from .ambassador_tools import TOOL_SCHEMAS, execute_tool

        context = self._grounding_context(
            conversation_id, self._config()["max_context_turns"] * 2, agent_name
        )

        def _seed_messages(system: str) -> list[Message]:
            return [
                Message(role=MessageRole.SYSTEM, content=system),
                *self._thread_history(conversation_id, exclude_id=qa_id),
                Message(
                    role=MessageRole.USER,
                    content=self._build_qa_prompt(
                        question=question,
                        context=context,
                        agent_name=agent_name,
                        artifacts=artifacts,
                    ),
                ),
            ]

        messages = _seed_messages(self._build_tools_persona(profile, agent_name))
        answer = ""
        settled = False
        try:
            for _round in range(max(1, max_rounds)):
                result = await provider.complete(
                    messages,
                    model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
                calls = result.tool_calls or []
                if not calls:
                    answer = (result.content or "").strip()
                    break
                # Record the assistant's tool-call turn so the TOOL replies correlate.
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=result.content or "",
                        tool_calls=[
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in calls
                        ],
                    )
                )
                for tc in calls:
                    yield _sse(
                        "ambassador_tool_call",
                        {"message_id": qa_id, "tool": tc.name, "args": tc.arguments},
                    )
                    output = execute_tool(
                        tc.name,
                        tc.arguments,
                        active_conversation_id=conversation_id,
                        agent_name=agent_name,
                    )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=output,
                        )
                    )
                    yield _sse(
                        "ambassador_tool_result",
                        {"message_id": qa_id, "tool": tc.name, "ok": True},
                    )
            else:
                # Round cap reached with tools still pending — force a final answer.
                final = await provider.complete(
                    messages, model_id, temperature=temperature, max_tokens=max_tokens, tool_choice="none"
                )
                answer = (final.content or "").strip()

            answer = answer or "(The ambassador had no answer.)"
            store.set_qa_answer(conversation_id, qa_id, answer, status="done")
            settled = True
            yield _sse("ambassador_chunk", {"message_id": qa_id, "text": answer})
            yield _sse("ambassador_done", {"message_id": qa_id, "status": "done", "summary": answer})
        except GeneratorExit:
            # Cancelled mid-loop — settle without yielding (can't yield while closing).
            store.set_qa_status(conversation_id, qa_id, "cancelled", run_id=run_id)
            settled = True
            raise
        except Exception as e:  # noqa: BLE001 — degrade to a grounded one-shot answer
            logger.warning(f"Ambassador tool loop failed for {qa_id}: {e}")
            try:
                fb = await provider.complete(
                    _seed_messages(self._build_qa_persona(profile, agent_name)),
                    model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = (fb.content or "").strip() or "(The ambassador had no answer.)"
                store.set_qa_answer(conversation_id, qa_id, answer, status="done")
                settled = True
                yield _sse("ambassador_chunk", {"message_id": qa_id, "text": answer})
                yield _sse("ambassador_done", {"message_id": qa_id, "status": "done", "summary": answer})
            except Exception as e2:  # noqa: BLE001 — give up cleanly, never raise
                logger.warning(f"Ambassador tool-loop fallback failed for {qa_id}: {e2}")
                store.set_qa_status(conversation_id, qa_id, "error", run_id=run_id, error=str(e2)[:500])
                settled = True
                yield _sse("ambassador_error", {"message_id": qa_id, "error": str(e2)[:500]})
        finally:
            if not settled:
                store.set_qa_status(conversation_id, qa_id, "cancelled", run_id=run_id)

    async def draft_relay_message(
        self,
        conversation_id: str,
        intent: str,
        *,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> str:
        """Draft a message FROM the user TO the agent (the outbound relay): turn a
        rough intent into a clear, first-person message the user reviews/edits before
        sending. Returns the draft text. Never raises — falls back to the raw intent
        so the relay still works when no provider is configured."""
        intent = (intent or "").strip()
        if not intent:
            return ""
        cfg = self._config()
        if not cfg["enabled"]:
            return intent

        profile = self._resolve_profile(cfg["profile_id"])
        profile_model = getattr(profile, "default_model", None) if profile else None
        model = cfg["model"] or profile_model or _DEFAULT_MODEL
        temperature = getattr(profile, "temperature", 0.3) if profile else 0.3

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(
                model, preferred_fallback=_DEFAULT_MODEL
            )
        except Exception as e:  # noqa: BLE001 — degrade to the raw intent
            logger.warning(f"Ambassador draft provider unavailable: {e}")
            return intent

        context = self._grounding_context(
            conversation_id, cfg["max_context_turns"], agent_name
        )
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_draft_persona(profile, agent_name)),
            Message(
                role=MessageRole.USER,
                content=self._build_draft_prompt(
                    intent=intent, context=context, agent_name=agent_name, artifacts=artifacts
                ),
            ),
        ]
        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=self._max_tokens(profile, cfg),
            )
            return (result.content or "").strip() or intent
        except Exception as e:  # noqa: BLE001 — never block the relay
            logger.warning(f"Ambassador draft failed: {e}")
            return intent

    def _build_voice_command_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        parts = [_voice_command_persona(agent_name)]
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Personality:\n{profile.system_prompt.strip()}")
        if amb and getattr(amb, "briefing_prompt", "").strip():
            parts.append(f"Additional instructions:\n{amb.briefing_prompt.strip()}")
        return _sub_placeholders("\n\n".join(parts), agent_name=agent_name)

    def _build_voice_command_prompt(
        self,
        *,
        transcript: str,
        context: str,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> str:
        agent_label = agent_name.strip() or "the agent"
        sections = []
        if context.strip():
            sections.append(
                "The conversation so far (for grounding):\n" f"{context.strip()}"
            )
        else:
            sections.append(
                "The conversation is empty so far — nothing has happened in it yet. If they "
                "ask about it, answer that plainly; only relay if they clearly want to "
                "instruct the agent."
            )
        artifacts_block = self._render_artifacts(artifacts, agent_label)
        if artifacts_block:
            sections.append(artifacts_block)
        sections.append(f"What the person just said to you:\n{transcript.strip()}")
        sections.append("Decide the action and respond with only the JSON object.")
        return "\n\n".join(sections)

    @staticmethod
    def _parse_voice_command(content: str) -> tuple[str, str]:
        """Parse the routing model's reply into ``(action, text)``. Defensive:
        strips code fences, extracts the first JSON object, and falls back to
        treating the whole reply as an ``answer`` when anything is off."""
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw[:4].lower() == "json":
                raw = raw[4:]
            raw = raw.strip()
        data: dict[str, Any] = {}
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    data = parsed
            except (json.JSONDecodeError, ValueError):
                data = {}
        action = data.get("action")
        text = (data.get("text") or "").strip()
        if action not in ("answer", "relay") or not text:
            # Couldn't route — speak whatever the model produced.
            return "answer", text or raw
        return action, text

    async def route_voice_command(
        self,
        conversation_id: str,
        transcript: str,
        *,
        agent_name: str = "",
        artifacts: Optional[dict] = None,
    ) -> dict:
        """Interpret a spoken voice command: the ambassador decides whether to
        **answer** it (spoken Q&A, persisted to the `qa:` sidecar so the Text tab
        shows it) or draft a **relay** for the user to send to the agent. Returns
        ``{action, text, qa_id?}``. Never raises — degrades to a spoken notice so
        the call never breaks. (A future ``target`` selects a specific agent for
        cross-agent delegation; v1 always targets the active conversation.)"""
        transcript = (transcript or "").strip()
        if not transcript:
            return {"action": "answer", "text": "", "qa_id": None}

        cfg = self._config()
        if not cfg["enabled"]:
            return {"action": "answer", "text": "The ambassador is disabled in settings.", "qa_id": None}

        profile = self._resolve_profile(cfg["profile_id"])
        profile_model = getattr(profile, "default_model", None) if profile else None
        model = cfg["model"] or profile_model or _DEFAULT_MODEL
        temperature = getattr(profile, "temperature", 0.2) if profile else 0.2

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(
                model, preferred_fallback=_DEFAULT_MODEL
            )
        except Exception as e:  # noqa: BLE001 — degrade gracefully
            logger.warning(f"Ambassador voice-command provider unavailable: {e}")
            return {
                "action": "answer",
                "text": "No model provider is configured for the ambassador.",
                "qa_id": None,
            }

        context = self._grounding_context(
            conversation_id, cfg["max_context_turns"] * 2, agent_name
        )
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content=self._build_voice_command_persona(profile, agent_name),
            ),
            Message(
                role=MessageRole.USER,
                content=self._build_voice_command_prompt(
                    transcript=transcript,
                    context=context,
                    agent_name=agent_name,
                    artifacts=artifacts,
                ),
            ),
        ]

        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=self._max_tokens(profile, cfg),
            )
            action, text = self._parse_voice_command(result.content)
        except Exception as e:  # noqa: BLE001 — the call must never break
            logger.warning(f"Ambassador voice-command failed: {e}")
            return {"action": "answer", "text": "Sorry, I couldn't process that.", "qa_id": None}

        qa_id: Optional[str] = None
        if action == "answer" and text:
            # Persist spoken answers as Q&A so the Text tab replays them.
            qa_id = f"vc_{uuid.uuid4().hex[:16]}"
            try:
                store.create_qa(conversation_id, qa_id, transcript)
                store.set_qa_answer(conversation_id, qa_id, text, status="done")
            except Exception as e:  # noqa: BLE001 — persistence is best-effort
                logger.debug(f"voice-command qa persist failed: {e}")
                qa_id = None

        return {"action": action, "text": text, "qa_id": qa_id}

    async def synthesize(
        self,
        text: str,
        *,
        profile_id: Optional[str] = None,
        voice: Optional[str] = None,
        model: Optional[str] = None,
    ) -> SpeechResult:
        """Synthesize spoken audio for ``text`` (a briefing / Q&A answer).

        Resolution precedence — explicit arg → the resolved ambassador profile's
        ``ambassador.voice`` block → global ``ambassador.*`` config → shipped
        default (``microsoft/mai-voice-2``). The speech model is resolved
        **strictly** (no chat fallback — TTS must never degrade to a text model);
        an unconfigured/unsupported model raises :class:`SpeechUnavailable` so the
        caller can return a clean "add an OpenRouter key for voice" message.
        """
        text = (text or "").strip()
        if not text:
            raise SpeechUnavailable("Nothing to speak.", code="empty_text")
        if len(text) > _MAX_SPEECH_CHARS:
            text = text[:_MAX_SPEECH_CHARS].rstrip()

        config = get_config_manager()
        profile = self._resolve_profile(profile_id)
        amb = getattr(profile, "ambassador", None) if profile else None

        speech_model = (
            model
            or getattr(amb, "speech_model", None)
            or config.get("ambassador.speech_model")
            or _DEFAULT_SPEECH_MODEL
        )
        speech_voice = (
            voice
            or getattr(amb, "voice", None)
            or config.get("ambassador.voice")
            or _DEFAULT_SPEECH_VOICE
        )
        speed = getattr(amb, "speech_speed", None) if amb else None

        try:
            provider, model_id = self.registry.get_provider_for_model(speech_model)
        except Exception as e:  # noqa: BLE001 — unconfigured provider → clean 422
            raise SpeechUnavailable(
                f"No speech provider is configured for '{speech_model}'. "
                "Add an OpenRouter API key to enable voice.",
                code="voice_unconfigured",
            ) from e

        try:
            return await provider.synthesize_speech(
                text,
                model=model_id,
                voice=speech_voice,
                response_format="mp3",
                speed=speed,
            )
        except NotImplementedError as e:
            raise SpeechUnavailable(
                f"'{speech_model}' does not support speech synthesis. "
                "Choose a text-to-speech model in the ambassador's voice settings.",
                code="model_unsupported",
            ) from e
        except Exception as e:  # noqa: BLE001 — surface a clean failure
            logger.warning(f"Ambassador speech synthesis failed: {e}")
            raise SpeechUnavailable(str(e)[:300], code="synth_failed") from e

    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: str = "webm",
        profile_id: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe spoken audio to text (the user-speaks half of voice mode).

        Mirrors :meth:`synthesize`: the STT model resolves **strictly** with
        precedence explicit arg → the ambassador profile's ``transcription_model``
        → global ``ambassador.transcription_model`` → shipped default
        (``openai/whisper-1``). An unconfigured/unsupported model raises
        :class:`SpeechUnavailable` so the caller returns a clean ``422``. Returns the
        transcript text (the client routes it into the reviewable input — never
        auto-sent)."""
        if not audio:
            raise SpeechUnavailable("No audio to transcribe.", code="empty_audio")
        if len(audio) > _MAX_AUDIO_BYTES:
            raise SpeechUnavailable("Audio recording is too large.", code="audio_too_large")

        config = get_config_manager()
        profile = self._resolve_profile(profile_id)
        amb = getattr(profile, "ambassador", None) if profile else None

        stt_model = (
            model
            or getattr(amb, "transcription_model", None)
            or config.get("ambassador.transcription_model")
            or _DEFAULT_TRANSCRIPTION_MODEL
        )

        try:
            provider, model_id = self.registry.get_provider_for_model(stt_model)
        except Exception as e:  # noqa: BLE001 — unconfigured provider → clean 422
            raise SpeechUnavailable(
                f"No transcription provider is configured for '{stt_model}'. "
                "Add an OpenRouter API key to enable voice input.",
                code="transcription_unconfigured",
            ) from e

        try:
            result = await provider.transcribe_speech(
                audio, model=model_id, audio_format=audio_format, language=language
            )
        except NotImplementedError as e:
            raise SpeechUnavailable(
                f"'{stt_model}' does not support transcription. "
                "Choose a speech-to-text model in the ambassador's voice settings.",
                code="transcription_model_unsupported",
            ) from e
        except Exception as e:  # noqa: BLE001 — surface a clean failure
            logger.warning(f"Ambassador transcription failed: {e}")
            raise SpeechUnavailable(str(e)[:300], code="transcription_failed") from e

        return result.text

    async def _stream_and_settle(
        self,
        *,
        item_id: str,
        provider,
        model_id: str,
        temperature: float,
        max_tokens: int,
        messages: list[Message],
        on_chunk,
        on_done,
        on_cancel,
        on_error,
        empty_text: str,
        log_label: str,
    ) -> AsyncGenerator[str, None]:
        """Shared streaming core for briefings + Q&A: token-stream the completion,
        persist via the injected callbacks, and settle the sidecar on done / cancel
        (``GeneratorExit`` from ``gen.aclose()``) / error — never leaving a record
        stuck on ``streaming``. SSE uses the ``message_id`` field for both families
        (it carries ``item_id``) so the client pump is identical."""
        accumulated = ""
        settled = False
        finish_reason = None
        try:
            async for chunk in provider.stream(
                messages, model_id, temperature=temperature, max_tokens=max_tokens
            ):
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if not chunk.content:
                    continue
                accumulated += chunk.content
                on_chunk(chunk.content)
                yield _sse("ambassador_chunk", {"message_id": item_id, "text": chunk.content})
            # A thinking model that still hit the cap means reasoning + answer
            # exceeded the budget — surface it so the headroom can be raised.
            if finish_reason == "length":
                logger.warning(
                    f"Ambassador {log_label} hit the token cap (finish_reason=length) "
                    f"for {item_id}; raise ambassador.max_tokens or _THINKING_HEADROOM."
                )
            text = accumulated.strip() or empty_text
            on_done(text)
            settled = True
            yield _sse(
                "ambassador_done",
                {"message_id": item_id, "status": "done", "summary": text},
            )
        except GeneratorExit:
            # Cancelled (or client closed the run). Settle whatever streamed so far
            # as `cancelled`. Re-raise without yielding (can't yield while closing).
            on_cancel()
            settled = True
            raise
        except Exception as e:  # noqa: BLE001 — the sidecar run must never crash
            logger.warning(f"Ambassador {log_label} failed for {item_id}: {e}")
            on_error(str(e)[:500])
            settled = True
            yield _sse("ambassador_error", {"message_id": item_id, "error": str(e)[:500]})
        finally:
            # Belt-and-suspenders: any exotic exit path still settles the record.
            if not settled:
                on_cancel()


# Module-level singleton
_service: Optional[AmbassadorService] = None


def get_ambassador() -> AmbassadorService:
    """Get the global AmbassadorService instance."""
    global _service
    if _service is None:
        _service = AmbassadorService()
    return _service
