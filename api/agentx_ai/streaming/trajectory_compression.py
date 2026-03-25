"""
Intra-Trajectory Compression — consolidate older tool-call rounds into a
Knowledge block to free context window space during multi-round tool loops.

When context usage crosses a configurable threshold, older assistant+tool
round pairs are replaced with a single SYSTEM "Knowledge" message that
summarises the findings. The most recent N rounds stay intact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..providers.base import Message, MessageRole

logger = logging.getLogger(__name__)

# Cap per-tool-result text in the serialisation sent to the LLM
_MAX_RESULT_CHARS_IN_PROMPT = 1000


@dataclass
class ToolRound:
    """A single tool-call round: one ASSISTANT message + one or more TOOL messages."""
    start_idx: int
    end_idx: int            # inclusive
    assistant_msg: Message
    tool_msgs: list[Message] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Round identification
# ---------------------------------------------------------------------------

def identify_tool_rounds(messages: list[Message]) -> list[ToolRound]:
    """
    Identify tool-call rounds in a message list.

    A round is an ASSISTANT message with ``tool_calls`` followed by one or
    more TOOL messages.  Returns them in order with indices into *messages*.
    """
    rounds: list[ToolRound] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            current_round = ToolRound(
                start_idx=i,
                end_idx=i,
                assistant_msg=msg,
            )
            j = i + 1
            while j < len(messages) and messages[j].role == MessageRole.TOOL:
                current_round.tool_msgs.append(messages[j])
                current_round.end_idx = j
                j += 1
            rounds.append(current_round)
            i = j
        else:
            i += 1
    return rounds


# ---------------------------------------------------------------------------
# Serialisation for the LLM prompt
# ---------------------------------------------------------------------------

def rounds_to_text(rounds: list[ToolRound]) -> str:
    """Serialise rounds into a readable text block for the compression prompt."""
    parts: list[str] = []
    for idx, rnd in enumerate(rounds, 1):
        lines = [f"Round {idx}:"]
        # Summarise each tool call in the assistant message
        for tc in (rnd.assistant_msg.tool_calls or []):
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
            # Truncate long argument strings
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            lines.append(f"  Called: {name}({args_str})")
        # Append each tool result (capped)
        for tm in rnd.tool_msgs:
            content = (tm.content or "")[:_MAX_RESULT_CHARS_IN_PROMPT]
            if len(tm.content or "") > _MAX_RESULT_CHARS_IN_PROMPT:
                content += f"\n  ...[{len(tm.content) - _MAX_RESULT_CHARS_IN_PROMPT:,} more chars]"
            lines.append(f"  Result ({tm.name or 'tool'}): {content}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM-based knowledge block generation
# ---------------------------------------------------------------------------

def _generate_knowledge_block(
    rounds_text: str,
    task_context: str,
    config: dict[str, Any],
) -> Optional[str]:
    """
    Call the LLM to produce a knowledge summary from serialised rounds.

    Returns the knowledge text, or ``None`` on failure.
    """
    from ..prompts.loader import get_prompt_loader
    from ..providers.registry import get_registry

    try:
        loader = get_prompt_loader()
        prompt = loader.get(
            "compression.trajectory",
            task_context=task_context or "(No specific task context provided)",
            rounds_text=rounds_text,
            max_chars=str(config.get("max_knowledge_chars", 3000)),
        )

        registry = get_registry()
        provider, model_id = registry.get_provider_for_model(config["model"])

        messages = [Message(role=MessageRole.USER, content=prompt)]

        # Async provider call — bridge to sync context
        async def _call():
            return await provider.complete(
                messages,
                model_id,
                temperature=config.get("temperature", 0.2),
                max_tokens=config.get("max_tokens", 1500),
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _call())
                result = future.result(timeout=30)
        else:
            result = asyncio.run(_call())

        return result.content.strip() if result and result.content else None

    except Exception as e:
        logger.warning(f"Trajectory compression LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compress_trajectory(
    messages: list[Message],
    context_limit_tokens: int,
    task_context: str = "",
) -> bool:
    """
    Compress older tool-call rounds in *messages* if context is too large.

    Modifies *messages* **in place**: removes older round messages and inserts
    a Knowledge SYSTEM message after the last system message.

    Args:
        messages: The full message list (modified in place).
        context_limit_tokens: Token budget for context.
        task_context: The user's current task/query for relevance.

    Returns:
        ``True`` if compression was performed, ``False`` otherwise.
    """
    from ..config import get_config_manager
    from .helpers import estimate_tokens

    cfg_mgr = get_config_manager()
    config = {
        "enabled": cfg_mgr.get("trajectory_compression.enabled", True),
        "threshold_ratio": cfg_mgr.get("trajectory_compression.threshold_ratio", 0.75),
        "preserve_recent_rounds": cfg_mgr.get("trajectory_compression.preserve_recent_rounds", 2),
        "model": cfg_mgr.get("trajectory_compression.model", "claude-3-5-haiku-latest"),
        "temperature": cfg_mgr.get("trajectory_compression.temperature", 0.2),
        "max_tokens": cfg_mgr.get("trajectory_compression.max_tokens", 1500),
        "max_knowledge_chars": cfg_mgr.get("trajectory_compression.max_knowledge_chars", 3000),
    }

    if not config["enabled"]:
        return False

    # Check whether context is large enough to warrant compression
    current_tokens = estimate_tokens(messages)
    threshold = int(context_limit_tokens * config["threshold_ratio"])
    if current_tokens <= threshold:
        return False

    # Identify rounds
    rounds = identify_tool_rounds(messages)
    preserve_n = config["preserve_recent_rounds"]
    if len(rounds) <= preserve_n:
        return False  # Nothing old enough to compress

    rounds_to_compress = rounds[:-preserve_n] if preserve_n > 0 else rounds
    rounds_text = rounds_to_text(rounds_to_compress)

    # Generate knowledge block via LLM
    knowledge_text = _generate_knowledge_block(rounds_text, task_context, config)
    if not knowledge_text:
        logger.warning("Trajectory compression: LLM call failed, skipping")
        return False

    # Build the Knowledge SYSTEM message
    knowledge_msg = Message(
        role=MessageRole.SYSTEM,
        content=(
            f"[KNOWLEDGE - Consolidated from {len(rounds_to_compress)} tool-call round(s)]\n\n"
            f"{knowledge_text}"
        ),
    )

    # Remove compressed round messages (reverse order to preserve indices)
    indices_to_remove: list[int] = []
    for rnd in rounds_to_compress:
        indices_to_remove.extend(range(rnd.start_idx, rnd.end_idx + 1))
    for idx in sorted(indices_to_remove, reverse=True):
        del messages[idx]

    # Insert Knowledge message after the last SYSTEM message
    insert_pos = 0
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.SYSTEM:
            insert_pos = i + 1
    messages.insert(insert_pos, knowledge_msg)

    new_tokens = estimate_tokens(messages)
    logger.info(
        f"Trajectory compression: {len(rounds_to_compress)} round(s) → Knowledge block, "
        f"~{current_tokens:,} → ~{new_tokens:,} tokens"
    )
    return True
