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
conversation is never affected. The foundation briefs with a single
``provider.complete``; token-streaming is a drop-in swap later. ``speech``
briefings (OpenRouter TTS) bolt onto the same seam via the profile's
``ambassador.speech_model``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Optional

from ..config import get_config_manager
from ..providers.base import Message, MessageRole
from ..providers.registry import ProviderRegistry, get_registry
from . import ambassador_storage as store
from .conversation_history import load_recent_turns

logger = logging.getLogger(__name__)

# Rough token budget per grounding turn (mirrors conversation_history estimates).
_TOKENS_PER_TURN = 400

# Built-in model floor — used only when neither the settings override nor the
# chosen profile specify a model. resolve_with_fallback degrades from here.
_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

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
        return "\n\n".join(parts)

    def _build_draft_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        base = _persona_override(amb, "draft_persona") or _draft_persona(agent_name)
        parts = [base]
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Personality:\n{profile.system_prompt.strip()}")
        return "\n\n".join(parts)

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
        artifacts_block = self._render_artifacts(artifacts, agent_label)
        if artifacts_block:
            sections.append(artifacts_block)
        sections.append(f"The person asks you:\n{question.strip()}")
        sections.append(
            "Answer their question directly, in your own spoken voice, grounded only in "
            "the conversation above. If the answer isn't there, say so plainly."
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
        return "\n\n".join(parts)

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
