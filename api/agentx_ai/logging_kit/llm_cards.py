"""Render LLM request logs.

Replaces the old "dump the whole JSON payload" behavior with three modes driven
by ``AGENTX_LLM_LOG_LEVEL`` (see :mod:`flags`):

* ``off``     — nothing.
* ``summary`` — a compact one-line card (model · msgs · ~tokens · tools · temp).
* ``full``    — the full request as pretty, **redacted** JSON (the highlighter
  colors it on the console).

Returns a string so the caller logs it through its own (provider-categorized)
logger; ``None`` means "don't log".
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .flags import read_flags
from .redaction import redact


def _approx_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    chars = 0
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, str):
                chars += len(content)
            elif content is not None:
                chars += len(str(content))
    return chars // 4  # rough chars→tokens heuristic, no tokenizer needed


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"~{n / 1000:.1f}K tok"
    return f"~{n} tok"


def _summary(provider_name: str, params: dict[str, Any]) -> str:
    model = params.get("model", "?")
    messages = params.get("messages") or []
    tools = params.get("tools") or []
    n_msgs = len(messages) if isinstance(messages, list) else 0
    n_tools = len(tools) if isinstance(tools, list) else 0
    parts = [
        f"🤖 LLM ▸ {provider_name} {model}",
        f"{n_msgs} msgs",
        _fmt_tokens(_approx_tokens(messages)),
    ]
    if n_tools:
        parts.append(f"{n_tools} tools")
    temp = params.get("temperature")
    if temp is not None:
        parts.append(f"temp {temp}")
    max_tokens = params.get("max_tokens")
    if max_tokens is not None:
        parts.append(f"max_tokens {max_tokens}")
    return " · ".join(parts)


def _full(provider_name: str, params: dict[str, Any]) -> str:
    try:
        dumped = json.dumps(params, indent=2, default=str)
    except (TypeError, ValueError):
        dumped = str(params)
    return f"🤖 LLM REQUEST ▸ {provider_name}:\n{redact(dumped)}"


def render_llm_request(provider_name: str, request_params: dict[str, Any]) -> Optional[str]:
    level = read_flags().llm_level
    if level == "off":
        return None
    if level == "summary":
        return _summary(provider_name, request_params)
    return _full(provider_name, request_params)
