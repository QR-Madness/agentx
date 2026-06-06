"""
Prompt placeholders — a small, safe whitelist of `{token}` substitutions applied
when a system prompt is composed. Only these exact tokens are replaced, so any
other braces in a prompt (JSON examples, literal `{}`) are left untouched.

The client mirrors this list in `lib/promptPlaceholders.ts` for the editor's
"Insert placeholder" affordance + preview highlighting — keep them in sync.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

# Source of truth for the supported tokens (token, label, description).
PROMPT_PLACEHOLDERS: list[dict[str, str]] = [
    {"token": "{agent_name}", "label": "Agent name", "description": "This agent's display name"},
    {"token": "{date}", "label": "Date", "description": "Today's date (YYYY-MM-DD)"},
    {"token": "{time}", "label": "Time", "description": "Current time (HH:MM, 24h)"},
]


def substitute_placeholders(
    text: str, *, agent_name: str = "", now: Optional[datetime] = None
) -> str:
    """Replace the whitelisted `{token}`s in ``text``. Unknown braces are untouched."""
    if not text or "{" not in text:
        return text
    when = now or datetime.now()
    replacements = {
        "{agent_name}": agent_name or "",
        "{date}": when.strftime("%Y-%m-%d"),
        "{time}": when.strftime("%H:%M"),
    }
    for token, value in replacements.items():
        if token in text:
            text = text.replace(token, value)
    return text
