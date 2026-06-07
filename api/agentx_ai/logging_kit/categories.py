"""Log categories — the shared contract between subsystems, the console renderer
and the client Log panel.

A category is derived from the logger *name* (i.e. the module path), so existing
``logging.getLogger(__name__)`` call sites get a category for free — zero per-file
changes. Each category carries a stable ``key`` (the contract the client mirrors
in ``client/src/lib/logCategories.ts`` — keep the two in sync), a short label, an
emoji badge, and a rich console color.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class LogCategory:
    key: str
    label: str
    emoji: str
    color: str  # a rich color/style name


# Ordered most-specific prefix first; the first match on the dotted logger name
# wins. All names are matched with the leading ``agentx_ai.`` stripped.
CATEGORIES: tuple[LogCategory, ...] = (
    LogCategory("provider", "PROVIDER", "🧠", "cyan"),
    LogCategory("stream", "STREAM", "📡", "magenta"),
    LogCategory("plan", "PLAN", "🗺️", "dark_orange"),
    LogCategory("reason", "REASON", "💭", "purple"),
    LogCategory("ambassador", "AMBASS", "🤝", "bright_cyan"),
    LogCategory("memory", "MEMORY", "🧩", "yellow"),
    LogCategory("mcp", "MCP", "🔌", "blue"),
    LogCategory("jobs", "JOBS", "⚙️", "grey62"),
    LogCategory("agent", "AGENT", "🤖", "green"),
    LogCategory("translation", "TRANS", "🌐", "bright_green"),
    LogCategory("logkit", "LOGS", "🪵", "grey50"),
    LogCategory("core", "CORE", "•", "white"),
)

_BY_KEY = {c.key: c for c in CATEGORIES}
DEFAULT = _BY_KEY["core"]

# Dotted-prefix → category key. Checked in order; first (most specific) hit wins,
# so e.g. ``agent.ambassador`` and ``agent.planner`` are routed before the broad
# ``agent`` fallback.
_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("providers", "provider"),
    ("streaming", "stream"),
    ("agent.ambassador", "ambassador"),
    ("agent.plan", "plan"),
    ("agent.planner", "plan"),
    ("reasoning", "reason"),
    ("kit.agent_memory", "memory"),
    ("consolidation", "memory"),
    ("memory", "memory"),
    ("mcp", "mcp"),
    ("background", "jobs"),
    ("jobs", "jobs"),
    ("kit.translation", "translation"),
    ("logging_kit", "logkit"),
    ("agent", "agent"),
)


@lru_cache(maxsize=1024)
def category_for(logger_name: str) -> LogCategory:
    """Resolve a dotted logger name to its category (cached)."""
    name = logger_name or ""
    if name.startswith("agentx_ai."):
        name = name[len("agentx_ai.") :]
    elif name == "agentx_ai":
        name = ""
    for prefix, key in _PREFIX_RULES:
        if name == prefix or name.startswith(prefix + "."):
            return _BY_KEY[key]
    return DEFAULT


def category_by_key(key: str) -> LogCategory:
    return _BY_KEY.get(key, DEFAULT)


def serializable_catalog() -> list[dict[str, str]]:
    """The category registry as plain dicts (for the ``/api/logs/categories`` endpoint)."""
    return [
        {"key": c.key, "label": c.label, "emoji": c.emoji, "color": c.color}
        for c in CATEGORIES
    ]
