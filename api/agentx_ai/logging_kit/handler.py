"""Console handlers.

Three console renderings, chosen by flags:

* **pretty + decorations** → :class:`AgentXRichHandler` (color, category badge,
  run tag, semantic highlighting, rich tracebacks).
* **json** → newline-delimited JSON (for log aggregators in production).
* **plain** (decorations off) → a stock ``StreamHandler`` whose format string is
  byte-identical to the historical ``verbose`` formatter, so opting out restores
  exactly today's output.

``rich`` is imported lazily inside the builder so importing this module from
``settings`` stays cheap.
"""

from __future__ import annotations

import json
import logging

from .categories import category_for
from .flags import LogFlags

_PLAIN_FMT = "{levelname} {asctime} {module} {message}"


def build_plain_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_PLAIN_FMT, style="{"))
    return handler


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "category": category_for(record.name).key,
            "run_id": getattr(record, "run_id", None),
            "conversation_id": getattr(record, "conversation_id", None),
            "agent_id": getattr(record, "agent_id", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def build_json_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    return handler


def build_rich_handler(flags: LogFlags) -> logging.Handler:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.text import Text
    from rich.theme import Theme

    from .highlighters import STYLES, AgentXHighlighter

    console = Console(
        stderr=True,
        theme=Theme(STYLES),
        force_terminal=True if flags.force_color else None,
    )

    class AgentXRichHandler(RichHandler):
        def render_message(self, record: logging.LogRecord, message: str):  # type: ignore[override]
            rendered = super().render_message(record, message)
            cat = category_for(record.name)
            prefix = Text()
            prefix.append(f"{cat.emoji} ", style=cat.color)
            prefix.append(f"{cat.label} ", style=f"bold {cat.color}")
            run_tag = getattr(record, "run_tag", None)
            if run_tag:
                prefix.append(f"run:{run_tag} ", style="bright_black")
            # With markup=False + a highlighter, the base returns a Text we can
            # prepend to inline; fall back gracefully for any other renderable.
            if isinstance(rendered, Text):
                return Text.assemble(prefix, rendered)
            return rendered

    return AgentXRichHandler(
        console=console,
        highlighter=AgentXHighlighter(),
        markup=False,
        rich_tracebacks=True,
        show_path=True,
        log_time_format="%H:%M:%S",
    )


def build_console_handler(flags: LogFlags) -> logging.Handler:
    if not flags.decorations:
        return build_plain_handler()
    if flags.fmt == "json":
        return build_json_handler()
    return build_rich_handler(flags)
