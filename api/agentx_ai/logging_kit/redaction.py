"""Secret redaction for log records.

The Log panel exposes records over HTTP and the archive writes them to disk, so
secrets must be scrubbed *at capture time* — auth-gating the endpoint is not
enough. The filter runs on the QueueHandler (emitting thread) and rewrites the
record's final message in place; it also exposes ``redact()`` for the LLM cards
to scrub request params before rendering.
"""

from __future__ import annotations

import logging
import re

_PLACEHOLDER = "«redacted»"

# Each pattern keeps a leading group (the key/prefix) and replaces the secret tail.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # key=value / "key": "value" for sensitive key names. The value group also
    # swallows a leading Bearer/Basic scheme so "Authorization: Bearer <tok>"
    # redacts the whole credential, not just the scheme word.
    re.compile(
        r'(?i)("?\b(?:api[_-]?key|secret|token|password|passwd|authorization|'
        r'access[_-]?key|client[_-]?secret|refresh[_-]?token)\b"?\s*[:=]\s*"?)'
        r'((?:Bearer\s+|Basic\s+)?[^\s",})\]]+)'
    ),
    # Bearer tokens
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)"),
    # Common provider key shapes (sk-..., sk-ant-..., AIza...)
    re.compile(r"\b(sk-(?:ant-)?)[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\b(AIza)[A-Za-z0-9_\-]{20,}"),
)


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _PATTERNS:
        out = pat.sub(lambda m: m.group(1) + _PLACEHOLDER, out)
    return out


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — malformed % args shouldn't drop the line
            return True
        record.msg = redact(msg)
        record.args = ()
        return True
