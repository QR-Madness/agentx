"""
Context management for agent conversations.

Handles the context window by:
- Tracking token usage
- Summarizing old context
- Prioritizing relevant information
"""

import logging
from dataclasses import dataclass

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry
from ..kit.agent_memory.models import MemoryBundle

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context management.

    Only the rolling-summary/digest knobs remain here — the legacy token-budget
    knobs were retired in Foundation #6 along with the superseded ``prepare_context``
    assembler (per-turn assembly now flows through the Context Ledger /
    ``assemble_turn_context``).
    """
    summary_model: str = "anthropic:claude-haiku-4-5-20251001"
    # Output budget for one compaction pass. The digest is re-summarized in place
    # each pass, so this bounds the digest's steady-state size (not its coverage).
    summary_max_tokens: int = 800


class ContextManager:
    """
    Manages conversation context to fit within model limits.

    Surviving responsibilities after Foundation #6:
    - :meth:`assemble_turn_context` — sync, budget-fit turn assembly (Context Ledger).
    - :meth:`_summarize_messages` — LLM rolling-summary (used by ``SessionManager``).
    - :meth:`inject_memory` — splice a memory bundle into the message list.
    """

    def __init__(self, config: ContextConfig):
        self.config = config
        self._registry = None

    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def assemble_turn_context(
        self,
        *,
        system_blocks: list[Message],
        history: list[Message],
        new_message: Message,
        context_window: int,
        reserved_tokens: int,
        verbatim_ratio: float = 0.9,
        recent_floor: int = 4,
    ) -> list[Message]:
        """Build the final, budget-fit message list for one turn (sync, no LLM).

        Keeps every ``system_block`` (the base prompt + checkpoints/scratchpad/
        summary/memory/participants — the caller is responsible for bounding their
        size), then fits as much **recent** verbatim ``history`` as the token budget
        allows, dropping the oldest overflow (which a rolling-summary system block,
        if present, already covers). Always keeps at least ``recent_floor`` of the
        most recent turns even under pressure. Output ordering:
        ``system_blocks + fitted_history + [new_message]``.

        The budget is ``min(verbatim_ratio · window, window − reserved_tokens)`` so
        verbatim history can't crowd out the reserved response/tool tokens, and the
        whole prompt stays within a sane fraction of the real context window.

        This is now a thin wrapper over :func:`assemble_ledger` that registers every
        system block as **mandatory** (always emitted full, never dropped) — exactly
        the "keep every system block by fiat" behaviour above. Callers that want
        priority-based shrink/drop should build :class:`LedgerBlock`\\ s and call
        ``assemble_ledger`` directly (see the chat-stream path in ``views.py``).
        """
        from .context_ledger import LedgerBlock, assemble_ledger

        blocks = [
            LedgerBlock(
                key=f"system_{i}",
                priority=100,
                content=m.content,
                mandatory=True,
                role=m.role,
            )
            for i, m in enumerate(system_blocks)
        ]
        return assemble_ledger(
            blocks=blocks,
            history=history,
            new_message=new_message,
            context_window=context_window,
            reserved_tokens=reserved_tokens,
            verbatim_ratio=verbatim_ratio,
            recent_floor=recent_floor,
        ).messages

    async def _summarize_messages(self, messages: list[Message]) -> str:
        """Fold messages into a rolling compaction digest (one LLM call).

        Serves both compaction targets — the conversation-state ``digest`` and the
        legacy prose rolling summary; the caller rolls the prior digest/summary in
        as a leading SYSTEM message. The instruction lives in
        ``system_prompts.yaml`` (``compression.compaction_digest``) — recall-first,
        per INV-CTX-1: this digest is the only surviving view of the aged-out turns.
        """
        if not messages:
            return ""

        try:
            provider, model_id, _ = self.registry.resolve_with_fallback(
                self.config.summary_model
            )

            # Build conversation text
            conversation = []
            for msg in messages:
                role_name = msg.role.value.capitalize()
                conversation.append(f"{role_name}: {msg.content}")

            conversation_text = "\n\n".join(conversation)

            try:
                from ..prompts.loader import get_prompt_loader
                digest_prompt = get_prompt_loader().get("compression.compaction_digest")
            except Exception:  # pragma: no cover — YAML missing/corrupt
                digest_prompt = (
                    "Summarize the following conversation concisely, preserving key "
                    "information and context needed for future responses."
                )

            summary_messages = [
                Message(role=MessageRole.SYSTEM, content=digest_prompt),
                Message(
                    role=MessageRole.USER,
                    content=conversation_text
                ),
            ]
            
            result = await provider.complete(
                summary_messages,
                model_id,
                temperature=0.3,
                max_tokens=self.config.summary_max_tokens,
            )
            
            return result.content
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fall back to truncation
            return "Previous conversation context (details truncated)"
    
    def inject_memory(
        self,
        context: list[Message],
        memories: MemoryBundle | list[str],
        max_memory_tokens: int = 500,
    ) -> list[Message]:
        """
        Inject relevant memories into the context.
        
        Args:
            context: Current context messages
            memories: MemoryBundle or list of memory strings to inject
            max_memory_tokens: Maximum tokens for memories
            
        Returns:
            Context with memories injected
        """
        if not memories:
            return context
        
        # Convert MemoryBundle to formatted string
        if isinstance(memories, MemoryBundle):
            memory_text = memories.to_context_string()
            if not memory_text:
                return context
        else:
            # Legacy list[str] format
            memory_text = "\n".join(f"- {m}" for m in memories)
        
        # Truncate if needed
        if len(memory_text) > max_memory_tokens * 4:
            memory_text = memory_text[:max_memory_tokens * 4] + "..."
        
        memory_message = Message(
            role=MessageRole.SYSTEM,
            content=f"Relevant information from memory:\n{memory_text}"
        )
        
        # Insert after system messages, before conversation
        result = []
        inserted = False
        
        for msg in context:
            result.append(msg)
            if msg.role == MessageRole.SYSTEM and not inserted:
                result.append(memory_message)
                inserted = True
        
        if not inserted:
            result.insert(0, memory_message)
        
        return result
