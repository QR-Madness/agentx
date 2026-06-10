"""Database connection management for Neo4j, PostgreSQL, and Redis."""

import atexit
import logging
import threading
import time
from contextlib import contextmanager
from typing import ClassVar, TypeVar
from collections.abc import Callable, Generator
import redis
from neo4j import GraphDatabase, Driver, Session, NotificationMinimumSeverity
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Thread locks for singleton initialization (reentrant to avoid deadlocks)
_neo4j_lock = threading.RLock()
_postgres_lock = threading.RLock()
_redis_lock = threading.RLock()

# Transient Neo4j errors worth retrying with backoff.
_NEO4J_TRANSIENT_ERRORS = (ServiceUnavailable, SessionExpired, TransientError)

T = TypeVar("T")


def with_neo4j_retry[T](
    fn: Callable[[], T],
    *,
    retries: int = 3,
    base_delay: float = 0.1,
) -> T:
    """Run `fn`, retrying transient Neo4j errors with exponential backoff.

    Retries on ServiceUnavailable / SessionExpired / TransientError (cluster
    failover, leader election, dropped connections). Non-transient errors
    propagate immediately. After `retries` attempts the last error is re-raised.

    `fn` should be self-contained (open its own session) so each attempt starts
    clean. For simple one-shot queries prefer `Neo4jConnection.execute_query`,
    which uses the driver's own managed retry.
    """
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except _NEO4J_TRANSIENT_ERRORS as e:
            last_exc = e
            if attempt == retries - 1:
                break
            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Transient Neo4j error (attempt {attempt + 1}/{retries}), "
                f"retrying in {delay:.2f}s: {e}"
            )
            time.sleep(delay)
    assert last_exc is not None  # only reached after a transient failure
    raise last_exc


# Neo4j Connection Manager
class Neo4jConnection:
    """Neo4j graph database connection manager."""

    _driver: ClassVar[Driver | None] = None

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
                        max_connection_lifetime=settings.neo4j_max_connection_lifetime,
                        connection_acquisition_timeout=settings.connection_timeout,
                        # Suppress warnings about missing labels/properties (expected for empty DB)
                        notifications_min_severity=NotificationMinimumSeverity.OFF,
                    )
        return cls._driver

    @classmethod
    @contextmanager
    def session(cls) -> Generator[Session]:
        """Context manager for Neo4j sessions."""
        driver = cls.get_driver()
        session = driver.session()
        try:
            yield session
        finally:
            session.close()

    @classmethod
    def health_check(cls) -> dict:
        """Probe Neo4j connectivity. Returns {"status", "error"}.

        A single fast probe (no retry/backoff) so callers with their own timeout
        budget get a prompt liveness answer. For operational queries that should
        survive a cluster failover, wrap the session work in `with_neo4j_retry`.
        """
        try:
            with cls.session() as session:
                session.run("RETURN 1").consume()
            return {"status": "healthy", "error": None}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

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

    _engine: ClassVar[Engine | None] = None
    _session_factory: ClassVar[sessionmaker[SQLSession] | None] = None

    @classmethod
    def get_engine(cls) -> Engine:
        """Get or create SQLAlchemy engine (thread-safe)."""
        if cls._engine is None:
            with _postgres_lock:
                # Double-check pattern
                if cls._engine is None:
                    cls._engine = create_engine(
                        settings.postgres_uri,
                        pool_size=settings.postgres_pool_size,
                        max_overflow=settings.postgres_pool_max_overflow,
                        connect_args={"connect_timeout": settings.connection_timeout}
                    )
        return cls._engine  # type: ignore[return-value]

    @classmethod
    def get_session_factory(cls) -> sessionmaker[SQLSession]:
        """Get or create session factory (thread-safe)."""
        if cls._session_factory is None:
            with _postgres_lock:
                if cls._session_factory is None:
                    cls._session_factory = sessionmaker(
                        bind=cls.get_engine(),
                        autocommit=False,
                        autoflush=False
                    )
        return cls._session_factory  # type: ignore[return-value]

    @classmethod
    def health_check(cls) -> dict:
        """Probe PostgreSQL connectivity. Returns {"status", "error"}."""
        try:
            with get_postgres_session() as session:
                session.execute(text("SELECT 1"))
            return {"status": "healthy", "error": None}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @classmethod
    def close(cls) -> None:
        """Close the engine and dispose connections."""
        with _postgres_lock:
            if cls._engine:
                cls._engine.dispose()
                cls._engine = None
                cls._session_factory = None


@contextmanager
def get_postgres_session() -> Generator[SQLSession]:
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

    _client: ClassVar[redis.Redis | None] = None

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
        return cls._client  # type: ignore[return-value]

    @classmethod
    def health_check(cls) -> dict:
        """Probe Redis connectivity. Returns {"status", "error"}."""
        try:
            cls.get_client().ping()
            return {"status": "healthy", "error": None}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @classmethod
    def close(cls) -> None:
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
        except Exception as e:
            logger.warning(f"Failed to close Neo4j connection on shutdown: {e}", exc_info=True)

    def _close_postgres():
        try:
            PostgresConnection.close()
            results.put("postgres")
        except Exception as e:
            logger.warning(f"Failed to close PostgreSQL connection on shutdown: {e}", exc_info=True)

    def _close_redis():
        try:
            RedisConnection.close()
            results.put("redis")
        except Exception as e:
            logger.warning(f"Failed to close Redis connection on shutdown: {e}", exc_info=True)
    
    # Close connections in parallel with daemon threads
    threads = []
    for close_fn in [_close_neo4j, _close_postgres, _close_redis]:
        t = threading.Thread(target=close_fn, daemon=True)
        t.start()
        threads.append(t)
    
    # Wait briefly for cleanup (2 seconds max)
    for t in threads:
        t.join(timeout=2)


def get_redis_client() -> redis.Redis | None:
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
    except Exception as e:
        logger.debug(f"Redis unavailable, returning None: {e}")
        return None


# Register cleanup function to run on process exit
atexit.register(close_all_connections)
