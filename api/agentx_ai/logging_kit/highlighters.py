"""The "log parser that drills text and colors different things".

A :class:`rich.highlighter.RegexHighlighter` that paints semantic tokens inside a
log message — model ids, token/cost figures, durations, tags, urls, ids, quoted
strings. Adding a rule is one regex line. Highlighting is **console-only**; the
ring buffer and archive store the raw (redacted) message.
"""

from __future__ import annotations

from rich.highlighter import RegexHighlighter

# Style name (sans the ``agentx.`` base prefix) → rich style. Consumed when the
# Console theme is built in ``handler.py``.
STYLES: dict[str, str] = {
    "agentx.model": "bold cyan",
    "agentx.tokens": "yellow",
    "agentx.cost": "bold green",
    "agentx.dur": "magenta",
    "agentx.pct": "yellow",
    "agentx.tag": "bold blue",
    "agentx.url": "underline bright_blue",
    "agentx.path": "bright_black",
    "agentx.uuid": "bright_black",
    "agentx.str": "green",
    "agentx.num": "cyan",
    "agentx.bool": "italic bright_magenta",
}


class AgentXHighlighter(RegexHighlighter):
    """Order: generic rules first, most-specific last (later spans override)."""

    base_style = "agentx."
    highlights = [
        r"(?P<str>\"[^\"]*\")",
        r"(?P<bool>\b(?:True|False|None|true|false|null)\b)",
        r"(?P<num>\b\d[\d,]*\.?\d*\b)",
        r"(?P<path>(?:/[\w.\-]+){2,}/?)",
        r"(?P<url>https?://[^\s\]\)]+)",
        r"(?P<uuid>\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b)",
        r"(?P<model>\b[a-z][\w]*:[\w./\-]+)",
        r"(?P<tag>\[[\w :.\-/]+\])",
        r"(?P<pct>\b\d+(?:\.\d+)?%)",
        r"(?P<dur>\b\d+(?:\.\d+)?(?:ms|s)\b)",
        r"(?P<cost>\$\d+(?:\.\d+)+)",
        r"(?P<tokens>~?[\d.,]+\s*[KkMm]?\s*(?:tokens|tok)\b)",
    ]
