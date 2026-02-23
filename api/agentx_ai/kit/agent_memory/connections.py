"""Database connection management for Neo4j, PostgreSQL, and Redis."""

import atexit
import threading
from contextlib import contextmanager
from typing import Generator, Optional
import redis
from neo4j import GraphDatabase, Driver, Session, NotificationMinimumSeverity
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .config import get_settings

settings = get_settings()

# Thread locks for singleton initialization (reentrant to avoid deadlocks)
_neo4j_lock = threading.RLock()
_postgres_lock = threading.RLock()
_redis_lock = threading.RLock()


# Neo4j Connection Manager
class Neo4jConnection:
    """Neo4j graph database connection manager."""

    _driver: Driver = None

    @classmethod
    def get_driver(cls) -> Driver:
        """Get or create Neo4j driver instance (thread-safe)."""
        if cls._driver is None:
            with _neo4j_lock:
                # Double-check pattern
                if cls._driver is None:
                    cls._driver = GraphDatabase.driver(
                        settings.neo4j_uri,
                        auth=(settings.neo4j_user, settings.neo4j_password),
                        connection_timeout=settings.connection_timeout,
                        max_connection_lifetime=300,
                        connection_acquisition_timeout=settings.connection_timeout,
                        # Suppress warnings about missing labels/properties (expected for empty DB)
                        notifications_min_severity=NotificationMinimumSeverity.OFF,
                    )
        return cls._driver

    @classmethod
    @contextmanager
    def session(cls) -> Generator[Session, None, None]:
        """Context manager for Neo4j sessions."""
        driver = cls.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            session.close()

    @classmethod
    def close(cls):
        """Close the Neo4j driver connection."""
        with _neo4j_lock:
            if cls._driver:
                cls._driver.close()
                cls._driver = None


# PostgreSQL Connection Manager (lazy initialization)
class PostgresConnection:
    """PostgreSQL connection manager with lazy initialization."""
    
    _engine = None
    _session_factory = None
    
    @classmethod
    def get_engine(cls):
        """Get or create SQLAlchemy engine (thread-safe)."""
        if cls._engine is None:
            with _postgres_lock:
                # Double-check pattern
                if cls._engine is None:
                    cls._engine = create_engine(
                        settings.postgres_uri, 
                        pool_size=10, 
                        max_overflow=20,
                        connect_args={"connect_timeout": settings.connection_timeout}
                    )
        return cls._engine
    
    @classmethod
    def get_session_factory(cls):
        """Get or create session factory (thread-safe)."""
        if cls._session_factory is None:
            with _postgres_lock:
                if cls._session_factory is None:
                    cls._session_factory = sessionmaker(
                        bind=cls.get_engine(), 
                        autocommit=False, 
                        autoflush=False
                    )
        return cls._session_factory
    
    @classmethod
    def close(cls):
        """Close the engine and dispose connections."""
        with _postgres_lock:
            if cls._engine:
                cls._engine.dispose()
                cls._engine = None
                cls._session_factory = None


@contextmanager
def get_postgres_session() -> Generator[SQLSession, None, None]:
    """Context manager for PostgreSQL sessions."""
    session = PostgresConnection.get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Redis Connection Manager
class RedisConnection:
    """Redis in-memory data store connection manager."""

    _client: Optional[redis.Redis] = None

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get or create Redis client instance (thread-safe)."""
        if cls._client is None:
            with _redis_lock:
                # Double-check pattern
                if cls._client is None:
                    cls._client = redis.from_url(
                        settings.redis_uri,
                        decode_responses=True,
                        socket_timeout=settings.connection_timeout,
                        socket_connect_timeout=settings.connection_timeout
                    )
        return cls._client

    @classmethod
    def close(cls):
        """Close the Redis client connection."""
        with _redis_lock:
            if cls._client:
                cls._client.close()
                cls._client = None


def close_all_connections():
    """Close all database connections with timeout. Called on process shutdown."""
    import queue
    
    results = queue.Queue()
    
    def _close_neo4j():
        try:
            Neo4jConnection.close()
            results.put("neo4j")
        except Exception:
            pass
    
    def _close_postgres():
        try:
            PostgresConnection.close()
            results.put("postgres")
        except Exception:
            pass
    
    def _close_redis():
        try:
            RedisConnection.close()
            results.put("redis")
        except Exception:
            pass
    
    # Close connections in parallel with daemon threads
    threads = []
    for close_fn in [_close_neo4j, _close_postgres, _close_redis]:
        t = threading.Thread(target=close_fn, daemon=True)
        t.start()
        threads.append(t)
    
    # Wait briefly for cleanup (2 seconds max)
    for t in threads:
        t.join(timeout=2)


def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance, returning None if connection fails.

    This is a convenience function for cases where Redis is optional
    (e.g., caching) and failure should not raise an exception.

    Returns:
        Redis client or None if unavailable
    """
    try:
        client = RedisConnection.get_client()
        # Test connection with a ping
        client.ping()
        return client
    except Exception:
        return None


# Register cleanup function to run on process exit
atexit.register(close_all_connections)
