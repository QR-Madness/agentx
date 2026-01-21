"""Database connection management for Neo4j, PostgreSQL, and Redis."""

import atexit
import threading
from contextlib import contextmanager
from typing import Generator, Optional
import redis
from neo4j import GraphDatabase, Driver, Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .config import get_settings

settings = get_settings()

# Thread locks for singleton initialization
_neo4j_lock = threading.Lock()
_postgres_lock = threading.Lock()
_redis_lock = threading.Lock()


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
                        auth=(settings.neo4j_user, settings.neo4j_password)
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
                        max_overflow=20
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
                        decode_responses=True
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
    """Close all database connections. Called on process shutdown."""
    Neo4jConnection.close()
    PostgresConnection.close()
    RedisConnection.close()


# Register cleanup function to run on process exit
atexit.register(close_all_connections)
