"""
Aide swarm — cheap, parallel, read-only conversation digesters for the Ambassador.

The Ambassador's tool belt otherwise reads **full transcripts into its own context**
(up to ``_READ_TOKEN_BUDGET`` each), which bloats fast across a cross-conversation
survey. Instead, an *aide* reads/condenses ONE conversation read-only and returns a
short **digest**, so the ambassador only ever ingests condensed summaries — the
orchestrator→worker / map-reduce pattern (aides *map*, the ambassador *reduces*).

This service mirrors :class:`ToolOutputCompressor`: a cheap model tier, fallback-aware
provider resolution, an async core with a loop-detecting sync wrapper (the ambassador's
``execute_tool`` is synchronous but runs inside a live event loop), and **never-raise**
semantics — one bad/timed-out aide returns ``None`` and never sinks the survey.

Invariants (Decisions.md INV-2/INV-1): an aide only *reads* (``_render_transcript``) and
calls a model; it writes nothing to ``conversation_logs``/``conv_summary:``. Digests are
cached in the ambassador sidecar (``amb_aide:`` via :mod:`ambassador_storage`), never the
main transcript.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from ..config import get_config_manager
from ..model_roles import resolve_member_model
from ..providers.base import Message, MessageRole
from ..providers.registry import ProviderRegistry, get_registry

logger = logging.getLogger(__name__)

# The ambassador's cheap floor (mirrors ambassador._DEFAULT_MODEL) — used when the
# aide model is unset, so we never reach for a key that may not exist in config.
_AIDE_FLOOR_MODEL = "anthropic:claude-haiku-4-5-20251001"

# The aide's task description — objective, output shape, and boundaries (a detailed
# brief prevents vague, padded digests). It receives one conversation's transcript.
_AIDE_SYSTEM = (
    "You are an aide to an orchestrator agent. You are given ONE conversation transcript "
    "between a user and an AI agent. In at most 3 sentences, produce a high-level digest: "
    "what the agent set out to do, what it found or decided, and what is still open. "
    "Be concrete and specific to THIS conversation. Do not invent details, do not add "
    "preamble, and do not address the user — output only the digest."
)


class AideService:
    """Run cheap, parallel, read-only model calls that each condense one conversation."""

    def __init__(self) -> None:
        self._registry: ProviderRegistry | None = None

    @property
    def registry(self) -> ProviderRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def _get_config(self) -> dict[str, Any]:
        c = get_config_manager()
        return {
            "enabled": c.get("ambassador.aide.enabled", True),
            # Model-family aware: an unset aide model follows the `fast_utility`
            # role (this is a cheap map tier), then the haiku floor. An explicit
            # `ambassador.aide.model` still wins. See model_roles.ROLE_MEMBERS["aide"].
            "model": resolve_member_model("aide", c.get("ambassador.aide.model", "")) or _AIDE_FLOOR_MODEL,
            "temperature": c.get("ambassador.aide.temperature", 0.2),
            "max_tokens": c.get("ambassador.aide.max_tokens", 220),
            "max_input_chars": c.get("ambassador.aide.max_input_chars", 6000),
            "max_parallel": c.get("ambassador.aide.max_parallel", 4),
            "timeout_seconds": c.get("ambassador.aide.timeout_seconds", 20),
            "max_per_survey": c.get("ambassador.aide.max_per_survey", 8),
            "cache_ttl_seconds": c.get("ambassador.aide.cache_ttl_seconds", 1800),
        }

    @property
    def enabled(self) -> bool:
        return bool(self._get_config()["enabled"])

    # --- single digest ------------------------------------------------------

    async def digest_conversation(
        self,
        conversation_id: str,
        *,
        focus: str = "",
        label: str = "",
        fingerprint: str | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> str | None:
        """Condense ONE conversation into a short digest. Checks/writes the sidecar
        cache when ``fingerprint`` is given. Returns ``None`` on disabled, empty
        conversation, unavailable provider, or any failure (never raises)."""
        cfg = cfg or self._get_config()
        if not cfg["enabled"] or not conversation_id:
            return None

        from .ambassador_storage import get_aide_digest, set_aide_digest

        if fingerprint is not None:
            cached = get_aide_digest(conversation_id, fingerprint, focus)
            if cached:
                return cached

        # Read-only render (lazy import avoids a tools↔aide import cycle).
        from .ambassador_tools import _render_transcript

        transcript = _render_transcript(conversation_id, label)
        if not transcript or transcript.startswith("("):
            # Empty / sentinel note ("(That conversation is empty…)") — nothing to digest.
            return None
        transcript = transcript[: cfg["max_input_chars"]]

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(cfg["model"])
        except Exception as e:  # noqa: BLE001 — provider unavailable ⇒ caller falls back
            logger.debug(f"aide provider unavailable: {e}")
            return None

        user = transcript if not focus else f"Focus the digest on: {focus}\n\n{transcript}"
        messages = [
            Message(role=MessageRole.SYSTEM, content=_AIDE_SYSTEM),
            Message(role=MessageRole.USER, content=user),
        ]
        try:
            result = await provider.complete(
                messages, model_id,
                temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
            )
        except Exception as e:  # noqa: BLE001 — one aide never sinks the survey
            logger.debug(f"aide digest failed for {conversation_id}: {e}")
            return None

        digest = (result.content or "").strip()
        if not digest:
            return None

        self._record_usage(provider, model_id, conversation_id, result.usage)
        if fingerprint is not None:
            set_aide_digest(
                conversation_id, fingerprint, digest, focus, ttl=cfg["cache_ttl_seconds"],
            )
        return digest

    # --- fan-out ------------------------------------------------------------

    async def digest_many(
        self, items: list[tuple[str, str]], *, focus: str = "",
    ) -> dict[str, str]:
        """Digest many conversations in bounded parallel. ``items`` is ``(cid,
        fingerprint)`` pairs. Returns ``{cid: digest}`` for those that succeeded —
        per-aide failures/timeouts are simply absent (never-raise). Caps total aides
        at ``max_per_survey`` and concurrency at ``max_parallel``."""
        cfg = self._get_config()
        if not cfg["enabled"] or not items:
            return {}
        items = items[: cfg["max_per_survey"]]
        sem = asyncio.Semaphore(max(1, int(cfg["max_parallel"])))

        async def _one(cid: str, fp: str) -> tuple[str, str | None]:
            async with sem:
                try:
                    digest = await asyncio.wait_for(
                        self.digest_conversation(cid, focus=focus, fingerprint=fp, cfg=cfg),
                        timeout=cfg["timeout_seconds"],
                    )
                except Exception as e:  # noqa: BLE001 — incl. timeout; never-raise
                    logger.debug(f"aide for {cid} dropped: {e}")
                    return cid, None
                return cid, digest

        results = await asyncio.gather(
            *(_one(cid, fp) for cid, fp in items), return_exceptions=True,
        )
        out: dict[str, str] = {}
        for r in results:
            if isinstance(r, tuple) and r[1]:
                out[r[0]] = r[1]
        return out

    # --- sync wrappers (execute_tool is sync but runs in a live loop) --------

    def digest_conversation_sync(
        self, conversation_id: str, *, focus: str = "", label: str = "",
        fingerprint: str | None = None,
    ) -> str | None:
        return self._run_sync(self.digest_conversation(
            conversation_id, focus=focus, label=label, fingerprint=fingerprint,
        ))

    def digest_many_sync(
        self, items: list[tuple[str, str]], *, focus: str = "",
    ) -> dict[str, str]:
        return self._run_sync(self.digest_many(items, focus=focus)) or {}

    def _run_sync(self, coro: Any) -> Any:
        """Run an aide coroutine from a synchronous caller (the ``compress_sync``
        pattern): in a live loop, offload to a worker thread with a fresh loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        try:
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return asyncio.run(coro)
        except Exception as e:  # noqa: BLE001 — never-raise into the ambassador loop
            logger.debug(f"aide sync run failed: {e}")
            return None

    # --- metering -----------------------------------------------------------

    def _record_usage(self, provider, model_id: str, conversation_id: str, usage) -> None:
        """Record one aide LLM spend event (source ``aide``). Content-free, best-effort."""
        if not usage:
            return
        tokens_in = usage.get("prompt_tokens", 0) or 0
        tokens_out = usage.get("completion_tokens", 0) or 0
        if not (tokens_in or tokens_out):
            return
        try:
            from .usage_ledger import record_usage
            from ..providers.pricing import estimate_cost
            cost = None
            try:
                cost = estimate_cost(provider.get_capabilities(model_id), tokens_in, tokens_out)
            except Exception:  # noqa: BLE001 — pricing is optional
                pass
            record_usage(
                source="aide",
                model=model_id,
                provider=getattr(provider, "name", None),
                conversation_id=conversation_id,
                units={"tokens_in": tokens_in, "tokens_out": tokens_out,
                       "tokens_total": tokens_in + tokens_out},
                cost=cost,
            )
        except Exception as e:  # noqa: BLE001 — metering never breaks a turn
            logger.debug(f"aide usage record skipped: {e}")


_aide_service: AideService | None = None


def get_aide_service() -> AideService:
    """The global AideService singleton (mirrors ``get_compressor()``)."""
    global _aide_service
    if _aide_service is None:
        _aide_service = AideService()
    return _aide_service
