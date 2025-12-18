"""Database connection management for Neo4j, PostgreSQL, and Redis."""

from contextlib import contextmanager
from typing import Generator
import redis
from neo4j import GraphDatabase, Driver, Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLSession

from .config import get_settings

settings = get_settings()


# Neo4j Connection Manager
class Neo4jConnection:
    """Neo4j graph database connection manager."""

    _driver: Driver = None

    @classmethod
    def get_driver(cls) -> Driver:
        """Get or create Neo4j driver instance."""
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
        if cls._driver:
            cls._driver.close()
            cls._driver = None


# PostgreSQL Connection Manager
engine = create_engine(settings.postgres_uri, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_postgres_session() -> Generator[SQLSession, None, None]:
    """Context manager for PostgreSQL sessions."""
    session = SessionLocal()
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

    _client: redis.Redis = None

    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get or create Redis client instance."""
        if cls._client is None:
            cls._client = redis.from_url(
                settings.redis_uri,
                decode_responses=True
            )
        return cls._client

    @classmethod
    def close(cls):
        """Close the Redis client connection."""
        if cls._client:
            cls._client.close()
            cls._client = None
