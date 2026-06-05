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

_DEFAULT_PERSONA = (
    "You are an Ambassador: a person standing beside the user, quietly telling them "
    "what just happened in a conversation they're watching. You are NOT a participant "
    "— you don't continue or answer the conversation, you just tell the user about it.\n\n"
    "Talk like a human talking out loud. Speak in plain, natural, flowing sentences, "
    "the way you'd lean over and explain something to a friend. Capture what the turn "
    "actually did — the point it made, the question it raised, whatever it left hanging "
    "— and say it in your own words. Never invent anything that wasn't there.\n\n"
    "Hard rules on form: NO markdown, NO headings, NO labels like 'Briefing:', NO bullet "
    "points, NO bold or asterisks, NO numbered lists. Just prose — one flowing voice, as "
    "if spoken. Lean and natural, never stiff or templated. Use the surrounding context "
    "only to understand the turn, not to recap it."
)

_VERBOSITY_HINT = {
    "brief": "Keep it to a sentence or two — just the heart of it, spoken plainly.",
    "normal": "Keep it to a few natural sentences — a short spoken paragraph.",
    "deep": "Take a little longer if it helps — walk through the reasoning and what's left open, still as plain spoken prose.",
}


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
            "max_tokens": config.get("ambassador.max_tokens", 600),
        }

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

    def _build_persona(self, profile) -> str:
        amb = getattr(profile, "ambassador", None) if profile else None
        parts = [_DEFAULT_PERSONA]
        # A profile's own system prompt colors the ambassador's voice.
        if profile and getattr(profile, "system_prompt", None):
            parts.append(f"Persona guidance:\n{profile.system_prompt.strip()}")
        verbosity = getattr(amb, "verbosity", "normal") if amb else "normal"
        parts.append(_VERBOSITY_HINT.get(verbosity, _VERBOSITY_HINT["normal"]))
        if amb and getattr(amb, "briefing_prompt", "").strip():
            parts.append(f"Additional briefing instructions:\n{amb.briefing_prompt.strip()}")
        return "\n\n".join(parts)

    def _build_turn_prompt(
        self, *, user_text: str, assistant_text: str, context: str
    ) -> str:
        sections = []
        if context.strip():
            sections.append(
                "Recent conversation context (for grounding only — do not brief this):\n"
                f"{context.strip()}"
            )
        turn = "--- The turn to brief ---\n"
        if user_text.strip():
            turn += f"The user said:\n{user_text.strip()}\n\n"
        turn += f"The agent replied:\n{assistant_text.strip()}"
        sections.append(turn)
        sections.append("Now write the briefing of that turn for the user.")
        return "\n\n".join(sections)

    def _grounding_context(self, conversation_id: str, max_turns: int) -> str:
        """Read-only recent transcript for grounding. Empty on any failure."""
        try:
            msgs = load_recent_turns(
                conversation_id, token_budget=max_turns * _TOKENS_PER_TURN
            )
        except Exception as e:  # pragma: no cover - DB offline
            logger.debug(f"ambassador grounding load failed: {e}")
            return ""
        lines = []
        for m in msgs[-(max_turns * 2):]:
            role = "User" if m.role == MessageRole.USER else "Agent"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    async def brief_turn(
        self,
        conversation_id: str,
        message_id: str,
        *,
        assistant_text: str,
        user_text: str = "",
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

        context = self._grounding_context(conversation_id, cfg["max_context_turns"])
        messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_persona(profile)),
            Message(
                role=MessageRole.USER,
                content=self._build_turn_prompt(
                    user_text=user_text, assistant_text=assistant_text, context=context
                ),
            ),
        ]

        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=temperature,
                max_tokens=cfg["max_tokens"],
            )
            summary = (result.content or "").strip()
            if not summary:
                summary = "(The ambassador returned an empty briefing.)"
            store.set_summary(conversation_id, message_id, summary, status="done")
            yield _sse("ambassador_chunk", {"message_id": message_id, "text": summary})
            yield _sse(
                "ambassador_done",
                {"message_id": message_id, "status": "done", "summary": summary},
            )
        except Exception as e:  # noqa: BLE001 — the sidecar run must never crash
            logger.warning(f"Ambassador briefing failed for {message_id}: {e}")
            store.set_status(
                conversation_id, message_id, "error", run_id=run_id, error=str(e)[:500]
            )
            yield _sse(
                "ambassador_error", {"message_id": message_id, "error": str(e)[:500]}
            )


# Module-level singleton
_service: Optional[AmbassadorService] = None


def get_ambassador() -> AmbassadorService:
    """Get the global AmbassadorService instance."""
    global _service
    if _service is None:
        _service = AmbassadorService()
    return _service
