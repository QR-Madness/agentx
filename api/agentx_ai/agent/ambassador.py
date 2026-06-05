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
    "You are an Ambassador: a dedicated interpreter standing beside the user, "
    "briefing them on a single turn of a conversation they are observing. You are "
    "NOT a participant in that conversation — you summarize and interpret it, you "
    "never continue or answer it.\n\n"
    "Produce a faithful, plain-language briefing of the turn that:\n"
    "- states what the turn actually did (the move it made — a decision reached, a "
    "question posed, a position argued, a result delivered);\n"
    "- names the key subjects and the relationship asserted between them;\n"
    "- surfaces any unresolved tension, assumption, or open question left behind;\n"
    "- if the turn weighs competing options or principles, names each and the basis "
    "given for preferring one.\n\n"
    "Never invent detail that is not present in the turn. Be concise and neutral. "
    "Use the surrounding context only to ground the briefing, not to report on it."
)

_VERBOSITY_HINT = {
    "brief": "Keep the briefing to 1-2 tight sentences — only the single most important point.",
    "normal": "Keep the briefing to a short paragraph or a few bullet points.",
    "deep": "A fuller briefing is welcome: cover the reasoning, the tensions, and what is left open.",
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
