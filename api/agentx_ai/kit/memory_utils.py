"""
AgentX Agent Memory - Lazy-loaded memory system interface.

The memory system is initialized on first access to avoid startup delays
when database services are not running.
"""

from typing import Optional
import logging
import threading

logger = logging.getLogger(__name__)

_memory_instance: Optional["AgentMemory"] = None
_memory_error: Optional[Exception] = None

# Timeout for health checks in seconds
HEALTH_CHECK_TIMEOUT = 5


def get_agent_memory(user_id: str = "default", conversation_id: Optional[str] = None):
    """
    Get or create an AgentMemory instance.
    
    This function lazily initializes the memory system on first call.
    If database connections fail, it logs the error and returns None.
    
    Args:
        user_id: User identifier for memory isolation
        conversation_id: Optional conversation context
        
    Returns:
        AgentMemory instance or None if initialization fails
    """
    global _memory_instance, _memory_error
    
    if _memory_error is not None:
        logger.warning(f"Memory system unavailable: {_memory_error}")
        return None
    
    if _memory_instance is None:
        try:
            from .agent_memory.interface import AgentMemory
            _memory_instance = AgentMemory(user_id=user_id, conversation_id=conversation_id)
            logger.info("Agent memory system initialized successfully")
        except Exception as e:
            _memory_error = e
            logger.error(f"Failed to initialize agent memory: {e}")
            return None
    
    return _memory_instance


def check_memory_health() -> dict:
    """
    Check health of memory system connections with timeouts.
    
    Returns:
        Dict with status of each connection (neo4j, postgres, redis)
    """
    from sqlalchemy import text
    import queue
    
    health = {
        "neo4j": {"status": "unknown", "error": None},
        "postgres": {"status": "unknown", "error": None},
        "redis": {"status": "unknown", "error": None},
    }
    
    results = queue.Queue()
    
    def _check_neo4j():
        try:
            from .agent_memory.connections import Neo4jConnection
            with Neo4jConnection.session() as session:
                session.run("RETURN 1").consume()
            results.put(("neo4j", "healthy", None))
        except Exception as e:
            results.put(("neo4j", "unhealthy", str(e)))
    
    def _check_postgres():
        try:
            from .agent_memory.connections import get_postgres_session
            with get_postgres_session() as session:
                session.execute(text("SELECT 1"))
            results.put(("postgres", "healthy", None))
        except Exception as e:
            results.put(("postgres", "unhealthy", str(e)))
    
    def _check_redis():
        try:
            from .agent_memory.connections import RedisConnection
            client = RedisConnection.get_client()
            client.ping()
            results.put(("redis", "healthy", None))
        except Exception as e:
            results.put(("redis", "unhealthy", str(e)))
    
    # Start daemon threads for each check
    threads = []
    for check_fn in [_check_neo4j, _check_postgres, _check_redis]:
        t = threading.Thread(target=check_fn, daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete or timeout
    deadline = threading.Event()
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