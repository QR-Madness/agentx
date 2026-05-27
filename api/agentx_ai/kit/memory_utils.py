"""
AgentX Agent Memory - Lazy-loaded memory system interface.

The memory system is initialized on first access to avoid startup delays
when database services are not running.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging
import threading

if TYPE_CHECKING:
    from .agent_memory.memory.interface import AgentMemory

logger = logging.getLogger(__name__)

_memory_instance: Optional[AgentMemory] = None
_memory_error: Optional[Exception] = None

# Timeout for health checks in seconds
HEALTH_CHECK_TIMEOUT = 5


def get_agent_memory(user_id: str = "default", conversation_id: Optional[str] = None, channel: str = "_default", agent_id: Optional[str] = None):
    """
    Get or create an AgentMemory instance.

    This function lazily initializes the memory system on first call.
    If database connections fail, it logs the error and returns None.

    Args:
        user_id: User identifier for memory isolation
        conversation_id: Optional conversation context
        channel: Memory channel for scoping (default "_default")
        agent_id: Optional agent identifier for self-memory channel

    Returns:
        AgentMemory instance or None if initialization fails
    """
    global _memory_instance, _memory_error

    if _memory_error is not None:
        logger.warning(f"Memory system unavailable: {_memory_error}")
        return None

    if _memory_instance is None:
        try:
            from .agent_memory.memory.interface import AgentMemory
            _memory_instance = AgentMemory(user_id=user_id, conversation_id=conversation_id, channel=channel, agent_id=agent_id)
            logger.info("Agent memory system initialized successfully")
        except Exception as e:
            _memory_error = e
            logger.error(f"Failed to initialize agent memory: {e}")
            return None

    return _memory_instance


def check_memory_health() -> dict:
    """
    Check health of memory system connections with timeouts.

    Probes run in parallel daemon threads (bounded by HEALTH_CHECK_TIMEOUT) and
    delegate to each connection manager's `health_check()` — the single source
    of truth for per-store liveness.

    Returns:
        Dict with status of each connection (neo4j, postgres, redis)
    """
    import queue
    from .agent_memory.connections import (
        Neo4jConnection, PostgresConnection, RedisConnection,
    )

    health = {
        "neo4j": {"status": "unknown", "error": None},
        "postgres": {"status": "unknown", "error": None},
        "redis": {"status": "unknown", "error": None},
    }

    results = queue.Queue()

    # name -> connection manager whose health_check() to call
    probes = {
        "neo4j": Neo4jConnection,
        "postgres": PostgresConnection,
        "redis": RedisConnection,
    }

    def _probe(name, manager):
        result = manager.health_check()
        results.put((name, result["status"], result["error"]))

    # Start daemon threads for each check
    threads = []
    for name, manager in probes.items():
        t = threading.Thread(target=_probe, args=(name, manager), daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete or timeout
    for t in threads:
        t.join(timeout=HEALTH_CHECK_TIMEOUT)
    
    # Collect results that completed
    completed = set()
    while not results.empty():
        name, status, error = results.get_nowait()
        health[name]["status"] = status
        health[name]["error"] = error
        completed.add(name)
    
    # Mark any that didn't complete as timed out
    for name in ["neo4j", "postgres", "redis"]:
        if name not in completed:
            health[name]["status"] = "unhealthy"
            health[name]["error"] = f"Connection timed out after {HEALTH_CHECK_TIMEOUT}s"
    
    return health