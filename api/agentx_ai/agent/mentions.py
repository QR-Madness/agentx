"""@-mention parsing for inline agent routing (Phase 16.5).

A user can address an agent inline — ``@bright-grand-fern`` (Docker-style
agent_id) or ``@Mobius`` (single-word display name) — to route that turn to it.
These helpers are pure (no DB/view coupling): the streaming chat endpoint owns
resolving tokens against the ProfileManager and applying the routing override.
"""

import re
from typing import Callable, Optional

# Match @token where token is hyphen/word chars (agent_ids are adj-adj-noun;
# single-word names also fit). The negative lookbehind skips emails
# (user@host) and paths (a/@b) so only standalone mentions are caught.
_MENTION_RE = re.compile(r"(?<![\w/@])@([\w-]+)")


def extract_mentions(text: str) -> list[str]:
    """Return the @-mention tokens (without the leading '@'), ordered and deduped."""
    seen: list[str] = []
    for m in _MENTION_RE.finditer(text or ""):
        tok = m.group(1)
        if tok not in seen:
            seen.append(tok)
    return seen


def resolve_first_mention(
    text: str,
    resolve_fn: Callable[[str], Optional[str]],
) -> tuple[Optional[str], str]:
    """Resolve the first @-mention that maps to an agent.

    Args:
        text: the raw user message.
        resolve_fn: ``token -> agent_id | None`` (e.g. a ProfileManager lookup).

    Returns:
        ``(agent_id, stripped_text)`` where ``agent_id`` is the first token that
        resolved (or ``None`` if none did) and ``stripped_text`` has the resolved
        ``@token`` removed. Unresolved tokens are left verbatim.
    """
    resolved_agent_id: Optional[str] = None
    for tok in extract_mentions(text):
        agent_id = resolve_fn(tok)
        if agent_id is None:
            continue
        resolved_agent_id = agent_id
        # Strip just this mention token (collapse the surrounding space).
        stripped = re.sub(rf"(?<![\w/@])@{re.escape(tok)}\b", "", text)
        stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
        return resolved_agent_id, stripped
    return None, text
