"""
Pure builders that shape a finished chat turn into `Turn` records for storage.

Extracted from the inline `_store_turns` closure in `views.py` so the shaping
logic — turn ordering/indexing, the steer metadata, and the skip-empty assistant
rule — is unit-testable and shared by both the normal-completion and the
hard-stop (cancel) persistence paths. These functions do **no** I/O: they return
`Turn` objects the caller persists via `AgentMemory.store_turn`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def _turn_id(conv_id: str, suffix: str) -> str:
    return f"{conv_id}-{uuid.uuid4().hex[:8]}-{suffix}"


def build_user_turn(conv_id: str, content: str, index: int, *, turn_id: str | None = None):
    from ..kit.agent_memory.models import Turn

    return Turn(
        id=turn_id or _turn_id(conv_id, "user"),
        conversation_id=conv_id,
        role="user",
        content=content,
        index=index,
    )


def build_tool_turns(
    conv_id: str, tool_turns_data: list[dict], start_index: int
) -> tuple[list, int]:
    """Build tool_call / tool_result Turns from captured loop data.

    Returns the turns plus the next free index. Preserves the per-type metadata
    (incl. the `delegation` payload that lets a restored conversation rebuild a
    delegation card).
    """
    from ..kit.agent_memory.models import Turn

    turns: list = []
    idx = start_index
    for td in tool_turns_data:
        if td["type"] == "tool_call":
            turn = Turn(
                id=_turn_id(conv_id, "tool_call"),
                conversation_id=conv_id,
                role="tool_call",
                content=json.dumps(td.get("arguments", {})),
                index=idx,
                metadata={"tool": td["tool"], "tool_call_id": td["tool_call_id"]},
            )
        else:  # tool_result
            tr_metadata: dict[str, Any] = {
                "tool": td["tool"],
                "tool_call_id": td["tool_call_id"],
                "success": td.get("success", True),
                "duration_ms": td.get("duration_ms"),
            }
            if td.get("delegation"):
                tr_metadata["delegation"] = td["delegation"]
            turn = Turn(
                id=_turn_id(conv_id, "tool_result"),
                conversation_id=conv_id,
                role="tool_result",
                content=td.get("content", ""),
                index=idx,
                metadata=tr_metadata,
            )
        turns.append(turn)
        idx += 1
    return turns, idx


def build_steer_turns(
    conv_id: str,
    steers: list[dict],
    start_index: int,
    *,
    agent_id: str | None = None,
    agent_name: str | None = None,
) -> list:
    """Build `user` Turns for mid-turn steers, carrying a procedural-ready marker.

    Each steer records which round/tools it followed so a future consolidation job
    can mine user-intervention patterns ("redirected after web_search in round 2").
    The steered agent is recorded in metadata (a user turn has no producing agent,
    so `Turn.agent_id` stays None).
    """
    from ..kit.agent_memory.models import Turn

    turns: list = []
    for offset, s in enumerate(steers):
        metadata: dict[str, Any] = {
            "steered": True,
            "steer_round": s.get("round"),
            "after_tools": s.get("after_tools") or [],
            "phase": s.get("phase"),
        }
        if agent_id:
            metadata["steered_agent_id"] = agent_id
        if agent_name:
            metadata["steered_agent_name"] = agent_name
        turns.append(
            Turn(
                id=_turn_id(conv_id, "user-steer"),
                conversation_id=conv_id,
                role="user",
                content=s.get("content", ""),
                index=start_index + offset,
                metadata=metadata,
            )
        )
    return turns


def build_assistant_turn(
    conv_id: str,
    content: str,
    index: int,
    *,
    metadata: dict,
    token_count: int | None = None,
    model: str | None = None,
    agent_id: str | None = None,
    turn_id: str | None = None,
):
    """Build the assistant Turn, or None for blank content (the skip-empty rule).

    Blank assistant rows make the final message "disappear" on conversation
    restore (e.g. a supervisor that delegated without wrap-up), so they're
    skipped — the caller passes the already-assembled metadata dict. Exception:
    a turn carrying a plan card (``metadata["plan"]``) is meaningful even with
    no text (an interrupted plan stopped before synthesis), so it's kept — else
    the card + its resume offer would be lost on restore.
    """
    from ..kit.agent_memory.models import Turn

    if not content.strip() and not metadata.get("plan"):
        return None
    return Turn(
        id=turn_id or _turn_id(conv_id, "asst"),
        conversation_id=conv_id,
        role="assistant",
        content=content,
        index=index,
        token_count=token_count,
        model=model,
        metadata=metadata,
        agent_id=agent_id,
    )
