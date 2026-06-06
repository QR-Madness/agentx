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
        """Pick the ambassador profile: configured id → default profile → None."""
        from .profiles import get_profile_manager

        pm = get_profile_manager()
        profile = None
        if profile_id:
            profile = pm.get_profile(profile_id)
        if profile is None:
            profile = pm.get_default_profile()
        return profile

    def _current_run_id(self) -> Optional[str]:
        """Best-effort: the detached runner sets this ambiently per run."""
        try:
            from ..streaming.status import current_run_id

            return current_run_id.get()
        except Exception:
            return None

    def _build_persona(self, profile, agent_name: str = "") -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        parts = [_default_persona(agent_name)]
        # A profile's own system prompt colors the ambassador's voice.
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Persona guidance:\n{profile.system_prompt.strip()}")
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

        # Token-stream the briefing so the panel reveals it progressively (and a
        # re-attach replays the deltas). `settled` guards the sidecar status: if
        # the run is cancelled mid-stream (GeneratorExit from `gen.aclose()`), we
        # must settle it ourselves — otherwise it stays stuck on "streaming" and
        # a reopened panel shows a perpetual spinner.
        accumulated = ""
        settled = False
        finish_reason = None
        try:
            async for chunk in provider.stream(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=self._max_tokens(profile, cfg),
            ):
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if not chunk.content:
                    continue
                accumulated += chunk.content
                store.append_chunk(conversation_id, message_id, chunk.content)
                yield _sse(
                    "ambassador_chunk",
                    {"message_id": message_id, "text": chunk.content},
                )
            # A thinking model that still hit the cap means reasoning + answer
            # exceeded the budget — surface it so the headroom can be raised.
            if finish_reason == "length":
                logger.warning(
                    "Ambassador briefing hit the token cap (finish_reason=length) for "
                    f"{message_id}; raise ambassador.max_tokens or _THINKING_HEADROOM."
                )
            summary = accumulated.strip() or "(The ambassador returned an empty briefing.)"
            store.set_summary(conversation_id, message_id, summary, status="done")
            settled = True
            yield _sse(
                "ambassador_done",
                {"message_id": message_id, "status": "done", "summary": summary},
            )
        except GeneratorExit:
            # Cancelled (or client closed the run). Persist whatever streamed so
            # far as a settled `cancelled` record — never leave it on "streaming".
            # Re-raise without yielding (a generator must not yield while closing).
            store.set_status(conversation_id, message_id, "cancelled", run_id=run_id)
            settled = True
            raise
        except Exception as e:  # noqa: BLE001 — the sidecar run must never crash
            logger.warning(f"Ambassador briefing failed for {message_id}: {e}")
            store.set_status(
                conversation_id, message_id, "error", run_id=run_id, error=str(e)[:500]
            )
            settled = True
            yield _sse(
                "ambassador_error", {"message_id": message_id, "error": str(e)[:500]}
            )
        finally:
            # Belt-and-suspenders: any exotic exit path still settles the record.
            if not settled:
                store.set_status(conversation_id, message_id, "cancelled", run_id=run_id)


# Module-level singleton
_service: Optional[AmbassadorService] = None


def get_ambassador() -> AmbassadorService:
    """Get the global AmbassadorService instance."""
    global _service
    if _service is None:
        _service = AmbassadorService()
    return _service
