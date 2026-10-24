"""
Shared test utilities, base classes, and helper functions.

This module provides common infrastructure for tests.py and tests_memory.py:
- Skip-condition helpers (Docker, providers, models, embeddings)
- Base test classes (APITestBase, MemoryTestBase, MockRedisTestBase)
- Mock factory functions (Neo4j, PostgreSQL, Redis)
- Shared mock constants (compressor config)
"""

import os
import socket
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from django.test import Client, TestCase


# =============================================================================
# Skip-Condition Helpers
# =============================================================================

def docker_services_running() -> bool:
    """Check if Docker services (Neo4j, PostgreSQL, Redis) are reachable."""
    services = [("localhost", 7687), ("localhost", 5432), ("localhost", 6379)]
    for host, port in services:
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except (TimeoutError, OSError):
            return False
    return True


def has_configured_provider() -> bool:
    """Check if any model provider is configured."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("LMSTUDIO_BASE_URL")
    )


def translation_models_available() -> bool:
    """Check the translation models can load without a download.

    Online, an uncached model downloads on first use (slow, but it works — that's
    ``task test``). Offline (``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``, which
    ``task test:ci`` sets), each model must already be in the local HF cache, so
    the tests that need one skip instead of failing on a fresh CI runner.
    """
    try:
        from huggingface_hub import constants, try_to_load_from_cache

        from agentx_ai.kit.translation import TranslationKit
    except ImportError:
        return False
    if not constants.HF_HUB_OFFLINE:  # also set by TRANSFORMERS_OFFLINE
        return True

    def cached(repo: str, filename: str) -> bool:
        # A str is a cache hit; None / the no-exist sentinel are misses.
        return isinstance(try_to_load_from_cache(repo, filename), str)

    repos = {TranslationKit.language_detection_model_name, TranslationKit.level_ii_translation_model_name}
    return all(
        cached(repo, "config.json")
        and (cached(repo, "model.safetensors") or cached(repo, "pytorch_model.bin"))
        for repo in repos
    )


def embeddings_compatible() -> bool:
    """
    Check if embedding model output dimensions match configured dimensions.

    Loads the embedding provider and compares its actual output dimensions
    against the configured embedding_dimensions setting.
    """
    try:
        from agentx_ai.kit.agent_memory.embeddings import get_embedder
        embedder = get_embedder()
        _, _, match = embedder.validate_dimensions()
        return match
    except Exception:
        return False


# =============================================================================
# Mock Factory Functions
# =============================================================================

def create_mock_neo4j_session() -> MagicMock:
    """Create a mock Neo4j session for testing."""
    mock = MagicMock()
    mock.run.return_value = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def create_mock_postgres_session() -> MagicMock:
    """Create a mock PostgreSQL session for testing."""
    mock = MagicMock()
    mock.execute.return_value = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def create_mock_redis_client() -> MagicMock:
    """Create a mock Redis client for testing."""
    return MagicMock()


# =============================================================================
# Shared Mock Constants
# =============================================================================

COMPRESSOR_CONFIG: dict[str, Any] = {
    "enabled": True,
    "model": "test-model",
    "temperature": 0.2,
    "max_tokens": 1000,
    "max_summary_chars": 2000,
}

COMPRESSOR_CONFIG_DISABLED: dict[str, Any] = {
    **COMPRESSOR_CONFIG,
    "enabled": False,
}


# =============================================================================
# Base Test Classes
# =============================================================================

class APITestBase(TestCase):
    """Base for API endpoint tests with pre-initialized client."""

    def setUp(self) -> None:
        self.client: Client = Client()


class MemoryTestBase(TestCase):
    """Base for memory tests with UUID fixtures."""

    def setUp(self) -> None:
        self.test_user_id: str = str(uuid4())
        self.test_conversation_id: str = str(uuid4())


class MockRedisTestBase(TestCase):
    """Base for tests needing mocked Redis."""

    def setUp(self) -> None:
        self.mock_redis: MagicMock = MagicMock()
        self.redis_patcher = patch(
            "agentx_ai.kit.agent_memory.connections.RedisConnection.get_client"
        )
        self.mock_get_client: MagicMock = self.redis_patcher.start()
        self.mock_get_client.return_value = self.mock_redis

    def tearDown(self) -> None:
        self.redis_patcher.stop()
