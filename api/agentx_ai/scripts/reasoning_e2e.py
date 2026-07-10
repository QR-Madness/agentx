#!/usr/bin/env python
# pyright: reportAttributeAccessIssue=false
"""Self-driven end-to-end harness for Thinking Patterns (chat reasoning).

Drives REAL streamed chat turns through /api/agent/chat/stream — one tiny
scenario per pattern — and asserts the pattern telemetry on the done event.
Live LM calls (deliberately small prompts, ≤ a few cents total) on the cheap
adversarial reasoner:

    openrouter:nvidia/nemotron-3-ultra-550b-a55b

Run:  uv run python api/agentx_ai/scripts/reasoning_e2e.py [--model provider:id]

Exits 0 on PASS, 1 on FAIL, 2 on SKIP (Redis or an OpenRouter key unavailable —
the detached chat run rides a Redis stream). Memory is left untouched
(use_memory=false) and config mutations are restored in-process on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_API_DIR = _REPO_ROOT / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

DEFAULT_MODEL = "openrouter:nvidia/nemotron-3-ultra-550b-a55b"
SCENARIO_TIMEOUT_S = 180


def _log(msg: str) -> None:
    print(f"[reasoning-e2e] {msg}", flush=True)


def _stream_turn(client, payload: dict) -> tuple[dict | None, list[dict], str]:
    """POST one streamed turn; return (done_event, status_events, visible_text).

    The chat-stream view is async — its ``streaming_content`` is an async
    iterator even under the sync test Client, so drain it inside asyncio.
    """
    import asyncio

    started = time.monotonic()
    resp = client.post(
        "/api/agent/chat/stream", data=json.dumps(payload),
        content_type="application/json",
    )
    if resp.status_code != 200:
        raise AssertionError(f"stream HTTP {resp.status_code}: {resp.content[:300]!r}")

    done: dict | None = None
    statuses: list[dict] = []
    text_parts: list[str] = []

    async def _drain() -> None:
        nonlocal done
        buffer = ""
        async for raw in resp.streaming_content:
            if time.monotonic() - started > SCENARIO_TIMEOUT_S:
                raise AssertionError("scenario timed out")
            buffer += bytes(raw).decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                event, data = "", ""
                for line in block.split("\n"):
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                if not event:
                    continue
                try:
                    parsed = json.loads(data) if data else {}
                except ValueError:
                    continue
                if event == "chunk":
                    text_parts.append(str(parsed.get("content") or ""))
                elif event == "status":
                    statuses.append(parsed)
                elif event == "done":
                    done = parsed
            if done is not None:
                return

    asyncio.run(_drain())
    return done, statuses, "".join(text_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Thinking Patterns e2e harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="provider:model to drive")
    args = parser.parse_args()

    import django

    django.setup()
    from django.conf import settings as dj_settings
    from django.test import Client

    dj_settings.AGENTX_AUTH_ENABLED = False
    if "testserver" not in dj_settings.ALLOWED_HOSTS:
        dj_settings.ALLOWED_HOSTS = [*dj_settings.ALLOWED_HOSTS, "testserver"]

    # --- Preconditions ------------------------------------------------------
    from agentx_ai.kit.agent_memory.connections import RedisConnection

    try:
        RedisConnection.get_client().ping()
    except Exception as e:  # noqa: BLE001
        _log(f"SKIP: Redis unavailable ({e}). Run `task db:up` — the chat run bus needs it.")
        return 2

    from agentx_ai.config import get_config_manager

    cfg = get_config_manager()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY") or cfg.get("providers.openrouter.api_key"))
    if not has_key:
        _log("SKIP: no OpenRouter key (env OPENROUTER_API_KEY or providers.openrouter.api_key).")
        return 2

    client = Client()
    failures: list[str] = []
    # In-process config tweaks for determinism; restored in finally (never saved).
    saved = {
        "reasoning.sc_k": cfg.get("reasoning.sc_k", 3),
        "reasoning.auto_classifier_enabled": cfg.get("reasoning.auto_classifier_enabled", True),
    }

    def turn(name: str, message: str, pattern: str | None, *, expect_patterns: set,
             expect_status_phase: str | None = None, expect_thinking: bool | None = None):
        payload: dict = {
            "message": message,
            "model": args.model,
            "use_memory": False,
            "session_id": f"reasoning-e2e-{name}-{int(time.time())}",
        }
        if pattern:
            payload["thinking_pattern"] = pattern
        _log(f"— {name}: pattern={pattern or 'auto'} …")
        done, statuses, text = _stream_turn(client, payload)
        if done is None:
            failures.append(f"{name}: no done event")
            return
        got = done.get("thinking_pattern")
        if got not in expect_patterns:
            failures.append(f"{name}: thinking_pattern={got!r}, expected one of {expect_patterns}")
        if expect_thinking is True and not done.get("has_thinking"):
            failures.append(f"{name}: expected thinking, got none")
        if expect_status_phase is not None and not any(
            s.get("phase") == expect_status_phase for s in statuses
        ):
            failures.append(f"{name}: no status with phase={expect_status_phase!r}")
        visible = (text or "").strip()
        # The visible stream may include think tags; strip for the emptiness check.
        from agentx_ai.agent.output_parser import parse_output
        if not parse_output(visible).content.strip():
            failures.append(f"{name}: empty visible answer")
        _log(f"  ok: pattern={got} thinking={done.get('has_thinking')} "
             f"tokens_out={done.get('tokens_output')} chars={len(visible)}")

    try:
        cfg.set("reasoning.sc_k", 2)

        # 1. Explicit native — no scaffold, telemetry rides through.
        turn("native_passthrough", "Reply with exactly the word: ok",
             "native", expect_patterns={"native"})

        # 2. Explicit CoT directive.
        turn("cot_directive", "What is 12*17? Answer in one short sentence.",
             "cot", expect_patterns={"cot"}, expect_thinking=True)

        # 3. step_back — the pre-call emits the reasoning_step status; on
        #    pre-call failure it degrades honestly to cot.
        turn("step_back_precall", "Why does a dropped ball bounce lower each time?",
             "step_back", expect_patterns={"step_back", "cot"})

        # 4. Single-pass reflection directive.
        turn("reflection_single_pass", "Improve this sentence: 'me like code good'.",
             "reflection", expect_patterns={"reflection"})

        # 5. Auto with the LLM tiebreak off — heuristics only.
        cfg.set("reasoning.auto_classifier_enabled", False)
        turn("auto_heuristics_only", "Calculate 3+4 and answer in one word.",
             None, expect_patterns={"cot", "native", "self_consistency"})
        cfg.set("reasoning.auto_classifier_enabled", saved["reasoning.auto_classifier_enabled"])

        # 6. Self-consistency (k=2) — samples surface as thinking, judged final.
        turn("self_consistency_k2", "Is 51 a prime number? Answer yes or no with one reason.",
             "self_consistency", expect_patterns={"self_consistency"}, expect_thinking=True)

        # 7. deep_reflection — draft+critique live in the thinking bubble.
        turn("deep_reflection_multipass", "In one sentence, what makes a good error message?",
             "deep_reflection", expect_patterns={"deep_reflection"}, expect_thinking=True,
             expect_status_phase="reasoning_step")
    finally:
        for key, value in saved.items():
            cfg.set(key, value)

    if failures:
        _log("FAIL:")
        for f in failures:
            _log(f"  ✗ {f}")
        return 1
    _log("PASS: all thinking-pattern scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
