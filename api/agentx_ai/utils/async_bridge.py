"""Run async coroutines from synchronous code.

Several synchronous entry points (the sync ``Agent.run()`` reasoning call, the
RecallLayer's LLM techniques, trajectory compression, extraction convenience
wrappers) need to call ``async def`` provider methods. This is the single,
loop-aware bridge they share: if an event loop is already running we execute the
coroutine in a dedicated thread (a fresh loop), otherwise we run it directly.

This is a transitional helper — as the agent runtime moves async-native for
multi-agent workflows, sync callers should be converted and their bridge calls
removed rather than multiplied.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any


def run_coro_sync(coro: Any, timeout: float = 30.0) -> Any:
    """Run ``coro`` to completion from a synchronous context.

    If called while an event loop is running, the coroutine is executed in a
    dedicated worker thread with its own loop (so we never block or nest the
    running loop); otherwise it runs directly via ``asyncio.run``.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=timeout)
    return asyncio.run(coro)
