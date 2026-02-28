"""
Common decorators for AgentX.

Provides reusable patterns for lazy initialization, caching, etc.
"""

from functools import wraps
from threading import Lock
from typing import Callable, Optional, Protocol, TypeVar, cast
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class LazySingletonCallable(Protocol[T_co]):
    """Protocol for lazy_singleton decorated functions."""

    def __call__(self) -> T_co:
        """Get the singleton instance (creates on first call)."""
        ...

    def reset(self) -> None:
        """Reset the singleton instance (for testing)."""
        ...

    def is_initialized(self) -> bool:
        """Check if the singleton has been initialized without triggering init."""
        ...

    def get_if_initialized(self) -> Optional[T_co]:
        """Get the instance if initialized, or None without triggering init."""
        ...


def lazy_singleton(init_func: Callable[[], T]) -> LazySingletonCallable[T]:
    """
    Decorator for thread-safe lazy singleton initialization.

    Consolidates the repeated pattern of:
        _instance = None
        def get_instance():
            global _instance
            if _instance is None:
                _instance = create_instance()
            return _instance

    Usage:
        @lazy_singleton
        def get_my_service():
            return MyService(expensive_setup=True)

        # Then use:
        service = get_my_service()  # Created on first call
        service = get_my_service()  # Returns cached instance

    The decorator provides helper methods:
        get_my_service.reset()  # Clears cached instance (for testing)
        get_my_service.is_initialized()  # Check without triggering init
        get_my_service.get_if_initialized()  # Get instance or None
    """
    instance: Optional[T] = None
    lock = Lock()

    @wraps(init_func)
    def wrapper() -> T:
        nonlocal instance
        if instance is None:
            with lock:
                # Double-check locking pattern
                if instance is None:
                    logger.debug(f"Initializing singleton: {init_func.__name__}")
                    instance = init_func()
        return instance  # type: ignore[return-value]

    def reset() -> None:
        """Reset the singleton instance (for testing)."""
        nonlocal instance
        with lock:
            instance = None
            logger.debug(f"Reset singleton: {init_func.__name__}")

    def is_initialized() -> bool:
        """Check if the singleton has been initialized without triggering init."""
        return instance is not None

    def get_if_initialized() -> Optional[T]:
        """Get the instance if initialized, or None without triggering init."""
        return instance

    wrapper.reset = reset  # type: ignore[attr-defined]
    wrapper.is_initialized = is_initialized  # type: ignore[attr-defined]
    wrapper.get_if_initialized = get_if_initialized  # type: ignore[attr-defined]
    return cast(LazySingletonCallable[T], wrapper)


def lazy_singleton_with_fallback(
    init_func: Callable[[], T],
    fallback: Optional[T] = None,
) -> Callable[[], Optional[T]]:
    """
    Lazy singleton that returns fallback on initialization failure.

    Useful for optional services that may not be available (e.g., memory system
    when databases are down).

    Usage:
        @lazy_singleton_with_fallback
        def get_memory():
            return AgentMemory()  # May fail if DB is down

        memory = get_memory()  # Returns None if initialization failed
        if memory:
            memory.store_turn(...)

    Args:
        init_func: Initialization function
        fallback: Value to return on failure (default: None)
    """
    instance: Optional[T] = None
    error: Optional[Exception] = None
    lock = Lock()

    @wraps(init_func)
    def wrapper() -> Optional[T]:
        nonlocal instance, error

        # If we previously failed, return fallback without retrying
        if error is not None:
            return fallback

        if instance is None:
            with lock:
                if instance is None and error is None:
                    try:
                        logger.debug(f"Initializing singleton: {init_func.__name__}")
                        instance = init_func()
                    except Exception as e:
                        error = e
                        logger.warning(
                            f"Failed to initialize {init_func.__name__}: {e}"
                        )
                        return fallback
        return instance

    def reset() -> None:
        """Reset the singleton instance and error state (for testing)."""
        nonlocal instance, error
        with lock:
            instance = None
            error = None
            logger.debug(f"Reset singleton: {init_func.__name__}")

    wrapper.reset = reset
    return wrapper
