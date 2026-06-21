"""Command policy — a deny-list of obviously destructive/privilege commands.

This is **defense-in-depth, not the primary control** — the bubblewrap jail (no `data/`,
no network, jailed CWD) is. Deny-lists are easy to bypass; this just stops casual footguns.
"""

from __future__ import annotations

import re

# Conservative defaults; operators can extend via ``shell.deny_patterns``.
DEFAULT_DENY_PATTERNS: list[str] = [
    r"\bsudo\b",
    r"\bsu\b",
    r"rm\s+-rf?\s+(/|~|\$HOME)(\s|$)",   # rm -rf / | ~ | $HOME
    r"\bmkfs\b",
    r"\bdd\b.*\bof=/dev/",
    r">\s*/dev/sd",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bchown\b.*\s/(\s|$)",
    r":\s*\(\s*\)\s*\{",                  # fork bomb :(){
]


def check_command(command: str, deny_patterns: list[str] | None = None) -> str | None:
    """Return a reason string if ``command`` matches a deny pattern, else ``None``."""
    text = (command or "").strip()
    if not text:
        return "empty command"
    for pat in (deny_patterns if deny_patterns is not None else DEFAULT_DENY_PATTERNS):
        try:
            if re.search(pat, text, re.IGNORECASE):
                return f"blocked by shell policy (matched {pat!r})"
        except re.error:  # pragma: no cover - bad operator-supplied pattern
            continue
    return None


def cap_output(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_chars``; returns ``(text, truncated)``."""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"\n…[truncated to {max_chars} chars]", True
    return text, False
