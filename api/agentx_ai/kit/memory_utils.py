"""
AgentX Agent Memory - Lazy-loaded memory system interface.

The memory system is initialized on first access to avoid startup delays
when database services are not running.
"""

from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)

_memory_instance: Optional["AgentMemory"] = None
_memory_error: Optional[Exception] = None


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
    Check health of memory system connections.
    
    Returns:
        Dict with status of each connection (neo4j, postgres, redis)
    """
    from sqlalchemy import text
    
    health = {
        "neo4j": {"status": "unknown", "error": None},
        "postgres": {"status": "unknown", "error": None},
        "redis": {"status": "unknown", "error": None},
    }
    
    # Check Neo4j
    try:
        from .agent_memory.connections import Neo4jConnection
        with Neo4jConnection.session() as session:
            session.run("RETURN 1")
        health["neo4j"]["status"] = "healthy"
    except Exception as e:
        health["neo4j"]["status"] = "unhealthy"
        health["neo4j"]["error"] = str(e)
    
    # Check PostgreSQL
    try:
        from .agent_memory.connections import get_postgres_session
        with get_postgres_session() as session:
            session.execute(text("SELECT 1"))
        health["postgres"]["status"] = "healthy"
    except Exception as e:
        health["postgres"]["status"] = "unhealthy"
        health["postgres"]["error"] = str(e)
    
    # Check Redis
    try:
        from .agent_memory.connections import RedisConnection
        client = RedisConnection.get_client()
        client.ping()
        health["redis"]["status"] = "healthy"
    except Exception as e:
        health["redis"]["status"] = "unhealthy"
        health["redis"]["error"] = str(e)
    
    return health