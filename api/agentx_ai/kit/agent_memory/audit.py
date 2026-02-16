"""Audit logging for memory operations.

This module provides the MemoryAuditLogger class for logging all memory
operations to PostgreSQL for traceability and debugging.
"""

import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

from sqlalchemy import text

from .config import Settings, get_settings
from .connections import get_postgres_session

logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


class AuditLogLevel(str, Enum):
    """Audit logging verbosity levels."""

    OFF = "off"
    WRITES = "writes"
    READS = "reads"
    VERBOSE = "verbose"


class OperationType(str, Enum):
    """Types of memory operations."""

    # Write operations
    STORE = "store"
    UPDATE = "update"
    DELETE = "delete"

    # Read operations
    RETRIEVE = "retrieve"
    SEARCH = "search"

    # Procedural operations
    RECORD = "record"

    # Cross-channel operations
    PROMOTE = "promote"

    # Consolidation operations
    JOB_RUN = "job_run"


class MemoryType(str, Enum):
    """Types of memory stores."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    COMPOSITE = "composite"
    CONSOLIDATION = "consolidation"


def _is_write_operation(operation: str) -> bool:
    """Check if an operation is a write operation."""
    return operation in {
        OperationType.STORE.value,
        OperationType.UPDATE.value,
        OperationType.DELETE.value,
        OperationType.RECORD.value,
        OperationType.PROMOTE.value,
        OperationType.JOB_RUN.value,
    }


def _hash_query(query_text: Optional[str]) -> Optional[str]:
    """Create a hash of the query text for deduplication."""
    if not query_text:
        return None
    return hashlib.sha256(query_text.encode()).hexdigest()[:16]


class MemoryAuditLogger:
    """Logs memory operations to PostgreSQL audit table.

    The audit logger respects the configured log level:
    - off: No logging
    - writes: Only log write operations (store, update, delete, record, promote)
    - reads: Log reads and writes
    - verbose: Full logging including working memory and detailed payloads
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the audit logger.

        Args:
            settings: Memory configuration settings. Uses default if not provided.
        """
        self._settings = settings or get_settings()
        self._enabled = self._settings.audit_log_level != AuditLogLevel.OFF.value

    @property
    def log_level(self) -> str:
        """Get current audit log level."""
        return self._settings.audit_log_level

    def _should_log(self, operation: str, memory_type: str) -> bool:
        """Determine if this operation should be logged based on settings.

        Args:
            operation: The operation type (store, retrieve, etc.)
            memory_type: The memory type (episodic, semantic, etc.)

        Returns:
            True if the operation should be logged, False otherwise.
        """
        if not self._enabled:
            return False

        level = self._settings.audit_log_level

        # Verbose logs everything
        if level == AuditLogLevel.VERBOSE.value:
            return True

        # Working memory only logged at verbose level
        if memory_type == MemoryType.WORKING.value:
            return False

        # Writes level logs only write operations
        if level == AuditLogLevel.WRITES.value:
            return _is_write_operation(operation)

        # Reads level logs everything except working memory
        if level == AuditLogLevel.READS.value:
            # Apply sampling for read operations
            if not _is_write_operation(operation):
                if random.random() > self._settings.audit_sample_rate:
                    return False
            return True

        return False

    def _get_config_snapshot(self) -> Dict[str, Any]:
        """Get current configuration values for snapshot."""
        return {
            "audit_log_level": self._settings.audit_log_level,
            "fact_confidence_threshold": self._settings.fact_confidence_threshold,
            "salience_decay_rate": self._settings.salience_decay_rate,
            "default_top_k": self._settings.default_top_k,
            "reranking_enabled": self._settings.reranking_enabled,
            "extraction_enabled": self._settings.extraction_enabled,
        }

    def _log_to_db(
        self,
        operation: str,
        memory_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[Union[str, UUID]] = None,
        source_channel: Optional[str] = None,
        target_channels: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        result_count: Optional[int] = None,
        latency_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        promoted_from_channel: Optional[str] = None,
        promotion_confidence: Optional[float] = None,
        promotion_access_count: Optional[int] = None,
        promotion_conversation_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write audit log entry to PostgreSQL.

        Args:
            operation: Operation type (store, retrieve, update, etc.)
            memory_type: Memory type (episodic, semantic, procedural, etc.)
            user_id: User identifier
            session_id: Session identifier
            conversation_id: Conversation UUID
            source_channel: Channel where operation originated
            target_channels: Channels queried or affected
            query_text: Query text for retrieval operations
            result_count: Number of results returned
            latency_ms: Operation latency in milliseconds
            success: Whether operation succeeded
            error_message: Error message if operation failed
            promoted_from_channel: Source channel for promotions
            promotion_confidence: Confidence threshold for promotion
            promotion_access_count: Access count threshold for promotion
            promotion_conversation_count: Conversation count threshold
            metadata: Additional operation metadata
        """
        try:
            with get_postgres_session() as session:
                # Convert UUID to string if needed
                conv_id = str(conversation_id) if conversation_id else None

                insert_sql = text("""
                    INSERT INTO memory_audit_log (
                        operation, memory_type, user_id, session_id, conversation_id,
                        source_channel, target_channels, query_text, result_count,
                        latency_ms, success, error_message, promoted_from_channel,
                        promotion_confidence, promotion_access_count,
                        promotion_conversation_count, config_snapshot, metadata
                    ) VALUES (
                        :operation, :memory_type, :user_id, :session_id,
                        :conversation_id::uuid, :source_channel, :target_channels,
                        :query_text, :result_count, :latency_ms, :success,
                        :error_message, :promoted_from_channel, :promotion_confidence,
                        :promotion_access_count, :promotion_conversation_count,
                        :config_snapshot, :metadata
                    )
                """)

                session.execute(
                    insert_sql,
                    {
                        "operation": operation,
                        "memory_type": memory_type,
                        "user_id": user_id,
                        "session_id": session_id,
                        "conversation_id": conv_id,
                        "source_channel": source_channel,
                        "target_channels": target_channels,
                        "query_text": query_text,
                        "result_count": result_count,
                        "latency_ms": latency_ms,
                        "success": success,
                        "error_message": error_message,
                        "promoted_from_channel": promoted_from_channel,
                        "promotion_confidence": promotion_confidence,
                        "promotion_access_count": promotion_access_count,
                        "promotion_conversation_count": promotion_conversation_count,
                        "config_snapshot": json.dumps(self._get_config_snapshot()),
                        "metadata": json.dumps(metadata) if metadata else None,
                    },
                )
        except Exception as e:
            # Audit logging should never break the main operation
            logger.warning(f"Failed to write audit log: {e}")

    def log_write(
        self,
        operation: str,
        memory_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[Union[str, UUID]] = None,
        channel: Optional[str] = None,
        record_ids: Optional[List[str]] = None,
        latency_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a write operation (store, update, delete).

        Args:
            operation: Write operation type
            memory_type: Memory store type
            user_id: User identifier
            session_id: Session identifier
            conversation_id: Conversation UUID
            channel: Channel for the operation
            record_ids: IDs of affected records
            latency_ms: Operation latency
            success: Whether operation succeeded
            error_message: Error message if failed
            metadata: Additional metadata
        """
        if not self._should_log(operation, memory_type):
            return

        full_metadata = metadata or {}
        if record_ids:
            full_metadata["record_ids"] = record_ids

        self._log_to_db(
            operation=operation,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            source_channel=channel,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=full_metadata if full_metadata else None,
        )

    def log_read(
        self,
        operation: str,
        memory_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[Union[str, UUID]] = None,
        channels: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        result_count: int = 0,
        latency_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a read operation (retrieve, search).

        Args:
            operation: Read operation type
            memory_type: Memory store type
            user_id: User identifier
            session_id: Session identifier
            conversation_id: Conversation UUID
            channels: Channels searched
            query_text: Query text used
            result_count: Number of results returned
            latency_ms: Operation latency
            success: Whether operation succeeded
            error_message: Error message if failed
            metadata: Additional metadata (e.g., per-strategy breakdown)
        """
        if not self._should_log(operation, memory_type):
            return

        self._log_to_db(
            operation=operation,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            conversation_id=conversation_id,
            source_channel=channels[0] if channels else None,
            target_channels=channels,
            query_text=query_text,
            result_count=result_count,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )

    def log_promotion(
        self,
        source_channel: str,
        promoted_ids: List[str],
        promoted_type: str,
        confidence: float,
        access_count: int,
        conversation_count: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a cross-channel promotion operation.

        Args:
            source_channel: Channel facts/entities were promoted from
            promoted_ids: IDs of promoted records
            promoted_type: Type of promoted items (entity, fact)
            confidence: Confidence threshold that was met
            access_count: Access count threshold that was met
            conversation_count: Conversation count threshold that was met
            user_id: User identifier
            session_id: Session identifier
            latency_ms: Operation latency
            metadata: Additional metadata
        """
        if not self._should_log(OperationType.PROMOTE.value, MemoryType.SEMANTIC.value):
            return

        full_metadata = metadata or {}
        full_metadata["promoted_ids"] = promoted_ids
        full_metadata["promoted_type"] = promoted_type

        self._log_to_db(
            operation=OperationType.PROMOTE.value,
            memory_type=MemoryType.SEMANTIC.value,
            user_id=user_id,
            session_id=session_id,
            source_channel=source_channel,
            target_channels=["_global"],
            result_count=len(promoted_ids),
            latency_ms=latency_ms,
            promoted_from_channel=source_channel,
            promotion_confidence=confidence,
            promotion_access_count=access_count,
            promotion_conversation_count=conversation_count,
            metadata=full_metadata,
        )

    def log_job(
        self,
        job_name: str,
        items_processed: int = 0,
        latency_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a consolidation job execution.

        Args:
            job_name: Name of the consolidation job
            items_processed: Number of items processed
            latency_ms: Job execution time
            success: Whether job completed successfully
            error_message: Error message if failed
            metadata: Additional job metadata
        """
        if not self._should_log(OperationType.JOB_RUN.value, MemoryType.CONSOLIDATION.value):
            return

        full_metadata = metadata or {}
        full_metadata["job_name"] = job_name

        self._log_to_db(
            operation=OperationType.JOB_RUN.value,
            memory_type=MemoryType.CONSOLIDATION.value,
            result_count=items_processed,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=full_metadata,
        )

    @contextmanager
    def timed_operation(
        self,
        operation: str,
        memory_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[Union[str, UUID]] = None,
        channel: Optional[str] = None,
    ):
        """Context manager for timing operations.

        Usage:
            with audit_logger.timed_operation("store", "episodic", user_id) as ctx:
                result = do_something()
                ctx["result_count"] = len(result)
                ctx["record_ids"] = [r.id for r in result]

        Args:
            operation: Operation type
            memory_type: Memory type
            user_id: User identifier
            session_id: Session identifier
            conversation_id: Conversation UUID
            channel: Channel for operation

        Yields:
            Dict to populate with operation results (result_count, record_ids, etc.)
        """
        context = {
            "result_count": None,
            "record_ids": None,
            "query_text": None,
            "channels": None,
            "metadata": None,
            "success": True,
            "error_message": None,
        }

        start_time = time.perf_counter()

        try:
            yield context
        except Exception as e:
            context["success"] = False
            context["error_message"] = str(e)
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            if _is_write_operation(operation):
                self.log_write(
                    operation=operation,
                    memory_type=memory_type,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    channel=channel,
                    record_ids=context.get("record_ids"),
                    latency_ms=elapsed_ms,
                    success=context["success"],
                    error_message=context.get("error_message"),
                    metadata=context.get("metadata"),
                )
            else:
                self.log_read(
                    operation=operation,
                    memory_type=memory_type,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    channels=context.get("channels") or ([channel] if channel else None),
                    query_text=context.get("query_text"),
                    result_count=context.get("result_count") or 0,
                    latency_ms=elapsed_ms,
                    success=context["success"],
                    error_message=context.get("error_message"),
                    metadata=context.get("metadata"),
                )


def audited(
    operation: str,
    memory_type: str,
    user_id_param: str = "user_id",
    session_id_param: str = "session_id",
    conversation_id_param: str = "conversation_id",
    channel_param: str = "channel",
) -> Callable[[F], F]:
    """Decorator for automatic audit logging of memory operations.

    This decorator wraps a method to automatically log its execution
    with timing and result information.

    Args:
        operation: Operation type for logging
        memory_type: Memory type for logging
        user_id_param: Name of the user_id parameter in the method
        session_id_param: Name of the session_id parameter
        conversation_id_param: Name of the conversation_id parameter
        channel_param: Name of the channel parameter

    Returns:
        Decorated function with audit logging

    Example:
        @audited(operation="store", memory_type="episodic")
        def store_turn(self, turn: Turn, user_id: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract audit logger from self if available
            audit_logger = getattr(self, "_audit_logger", None)
            if audit_logger is None:
                # No audit logger, just call the function
                return func(self, *args, **kwargs)

            # Extract parameters for logging
            user_id = kwargs.get(user_id_param)
            session_id = kwargs.get(session_id_param)
            conversation_id = kwargs.get(conversation_id_param)
            channel = kwargs.get(channel_param)

            # Also check self for these attributes
            if user_id is None:
                user_id = getattr(self, "_user_id", None)
            if session_id is None:
                session_id = getattr(self, "_session_id", None)
            if conversation_id is None:
                conversation_id = getattr(self, "_conversation_id", None)
            if channel is None:
                channel = getattr(self, "_channel", None)

            with audit_logger.timed_operation(
                operation=operation,
                memory_type=memory_type,
                user_id=user_id,
                session_id=session_id,
                conversation_id=conversation_id,
                channel=channel,
            ) as ctx:
                result = func(self, *args, **kwargs)

                # Try to extract result count from common return types
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        ctx["result_count"] = len(result)
                    elif isinstance(result, dict) and "count" in result:
                        ctx["result_count"] = result["count"]
                    elif isinstance(result, str):
                        # Single ID returned
                        ctx["record_ids"] = [result]
                        ctx["result_count"] = 1

                return result

        return wrapper  # type: ignore

    return decorator
