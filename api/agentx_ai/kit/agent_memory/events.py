"""
Event system for memory lifecycle hooks.

Provides a simple callback registry for memory events without
external framework dependencies.

Usage:
    from agentx_ai.kit.agent_memory.events import MemoryEventEmitter, TurnStoredPayload

    events = MemoryEventEmitter()

    # Register a callback
    def on_fact(payload: FactLearnedPayload):
        print(f"Learned: {payload.claim}")

    events.on("fact_learned", on_fact)

    # Emit an event
    events.emit("fact_learned", FactLearnedPayload(
        event_name="fact_learned",
        fact_id="123",
        claim="User prefers Python"
    ))
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Event callback types
SyncCallback = Callable[..., None]
AsyncCallback = Callable[..., Any]  # Coroutine function
Callback = Union[SyncCallback, AsyncCallback]


@dataclass
class EventPayload:
    """Base payload for memory events."""

    event_name: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnStoredPayload(EventPayload):
    """Payload for on_turn_stored event."""

    turn_id: str = ""
    conversation_id: str = ""
    role: str = ""
    content: str = ""
    user_id: str = ""
    channel: str = "_global"


@dataclass
class FactLearnedPayload(EventPayload):
    """Payload for on_fact_learned event."""

    fact_id: str = ""
    claim: str = ""
    confidence: float = 0.0
    source: str = ""
    user_id: str = ""
    channel: str = "_global"


@dataclass
class EntityCreatedPayload(EventPayload):
    """Payload for on_entity_created event."""

    entity_id: str = ""
    name: str = ""
    entity_type: str = ""
    user_id: str = ""
    channel: str = "_global"


@dataclass
class RetrievalCompletePayload(EventPayload):
    """Payload for on_retrieval_complete event."""

    query: str = ""
    result_count: int = 0
    latency_ms: float = 0.0
    user_id: str = ""
    channel: str = "_global"
    # Summary of results (avoid passing large data)
    turns_count: int = 0
    facts_count: int = 0
    entities_count: int = 0


class MemoryEventEmitter:
    """
    Simple event emitter for memory lifecycle hooks.

    Supports both sync and async callbacks. Events are fire-and-forget;
    errors in handlers are logged but don't block operations.

    Predefined events:
        - TURN_STORED: Emitted after a turn is stored
        - FACT_LEARNED: Emitted after a fact is learned
        - ENTITY_CREATED: Emitted after an entity is created/updated
        - RETRIEVAL_COMPLETE: Emitted after a retrieval operation

    Example:
        events = MemoryEventEmitter()

        # Register a sync callback
        def on_fact(payload: FactLearnedPayload):
            print(f"Learned: {payload.claim}")

        unsubscribe = events.on("fact_learned", on_fact)

        # Register an async callback
        async def async_handler(payload):
            await some_async_operation(payload)

        events.on("turn_stored", async_handler)

        # Emit events
        events.emit("fact_learned", payload)  # Sync emission
        await events.emit_async("turn_stored", payload)  # Async emission

        # Unsubscribe
        unsubscribe()
    """

    # Predefined event names
    TURN_STORED = "turn_stored"
    FACT_LEARNED = "fact_learned"
    ENTITY_CREATED = "entity_created"
    RETRIEVAL_COMPLETE = "retrieval_complete"

    def __init__(self):
        self._handlers: Dict[str, List[Callback]] = {}
        self._enabled = True

    def on(self, event: str, callback: Callback) -> Callable[[], None]:
        """
        Register a callback for an event.

        Args:
            event: Event name
            callback: Function to call when event is emitted

        Returns:
            Unsubscribe function
        """
        if event not in self._handlers:
            self._handlers[event] = []

        self._handlers[event].append(callback)

        def unsubscribe():
            self.off(event, callback)

        return unsubscribe

    def off(self, event: str, callback: Callback) -> bool:
        """
        Remove a callback for an event.

        Args:
            event: Event name
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if event not in self._handlers:
            return False

        try:
            self._handlers[event].remove(callback)
            return True
        except ValueError:
            return False

    def emit(self, event: str, payload: Optional[EventPayload] = None) -> int:
        """
        Emit an event synchronously.

        Sync callbacks are called immediately. Async callbacks are
        scheduled but not awaited (fire-and-forget).

        Args:
            event: Event name
            payload: Event payload

        Returns:
            Number of handlers called
        """
        if not self._enabled:
            return 0

        handlers = self._handlers.get(event, [])
        called = 0

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Schedule async handler without awaiting
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(payload))
                    except RuntimeError:
                        # No running loop, skip async handler
                        logger.debug(f"Skipping async handler for {event}: no event loop")
                else:
                    handler(payload)
                called += 1
            except Exception as e:
                logger.warning(f"Event handler error for {event}: {e}")

        return called

    async def emit_async(
        self, event: str, payload: Optional[EventPayload] = None
    ) -> int:
        """
        Emit an event asynchronously.

        Both sync and async callbacks are supported. Async callbacks
        are awaited.

        Args:
            event: Event name
            payload: Event payload

        Returns:
            Number of handlers called
        """
        if not self._enabled:
            return 0

        handlers = self._handlers.get(event, [])
        called = 0

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
                called += 1
            except Exception as e:
                logger.warning(f"Event handler error for {event}: {e}")

        return called

    def clear(self, event: Optional[str] = None) -> None:
        """
        Clear handlers for an event or all events.

        Args:
            event: Event name, or None to clear all
        """
        if event is None:
            self._handlers.clear()
        elif event in self._handlers:
            self._handlers[event].clear()

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission (handlers are kept but not called)."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if events are enabled."""
        return self._enabled

    def handler_count(self, event: Optional[str] = None) -> int:
        """
        Get the number of registered handlers.

        Args:
            event: Event name, or None for total count

        Returns:
            Number of handlers
        """
        if event is None:
            return sum(len(h) for h in self._handlers.values())
        return len(self._handlers.get(event, []))
