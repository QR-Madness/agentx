"""
Query utilities for the memory system.

Centralizes filter building, pagination, and type conversions for
Neo4j Cypher and PostgreSQL queries.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CypherFilterBuilder:
    """
    Fluent builder for Neo4j Cypher WHERE clauses.

    Consolidates the repeated filter logic across episodic, semantic,
    and procedural memory classes (15+ occurrences).

    Usage:
        builder = CypherFilterBuilder("f")
        builder.add_user_filter(user_id).add_channel_filter(channel)

        # For WHERE clause:
        where = builder.build()  # "WHERE f.user_id = $user_id AND ..."

        # For inline (after existing WHERE):
        inline = builder.build_inline()  # " AND f.user_id = $user_id AND ..."

        # In query:
        query = f'''
            MATCH (f:Fact)
            WHERE f.confidence >= $min_confidence {builder.build_inline()}
            RETURN f
        '''
    """

    def __init__(self, node_alias: str = "n"):
        """
        Initialize the filter builder.

        Args:
            node_alias: The alias used for the node in Cypher (e.g., "f" for Fact)
        """
        self.node_alias = node_alias
        self._conditions: List[str] = []

    def add_user_filter(self, user_id: Optional[str]) -> "CypherFilterBuilder":
        """
        Add user_id filter if provided.

        Args:
            user_id: User ID to filter by, or None to skip

        Returns:
            self for chaining
        """
        if user_id:
            self._conditions.append(f"{self.node_alias}.user_id = $user_id")
        return self

    def add_channel_filter(
        self,
        channel: Optional[str],
        include_global: bool = True,
    ) -> "CypherFilterBuilder":
        """
        Add channel filter with standard AgentX semantics.

        Standard behavior:
        - If channel is None or "_all": no filter (all channels)
        - If channel is "_global": only _global channel
        - Otherwise: specified channel OR _global (if include_global=True)

        Args:
            channel: Channel to filter by
            include_global: If True, also includes _global channel (default)

        Returns:
            self for chaining
        """
        if channel == "_all" or channel is None:
            # No filter - show all channels
            pass
        elif channel == "_global":
            # Only global channel
            self._conditions.append(f"{self.node_alias}.channel = '_global'")
        else:
            # Specific channel, optionally including global
            if include_global:
                self._conditions.append(
                    f"({self.node_alias}.channel = $channel OR "
                    f"{self.node_alias}.channel = '_global')"
                )
            else:
                self._conditions.append(f"{self.node_alias}.channel = $channel")
        return self

    def add_time_filter(
        self,
        hours: Optional[int],
        timestamp_field: str = "timestamp",
        max_hours: int = 8760,  # 1 year
    ) -> "CypherFilterBuilder":
        """
        Add time window filter.

        Args:
            hours: Hours to look back, or None to skip
            timestamp_field: Name of the timestamp field
            max_hours: Maximum allowed hours (default: 1 year)

        Returns:
            self for chaining
        """
        if hours is not None:
            # Validate bounds
            validated_hours = max(1, min(int(hours), max_hours))
            self._conditions.append(
                f"{self.node_alias}.{timestamp_field} > "
                f"datetime() - duration('PT{validated_hours}H')"
            )
        return self

    def add_confidence_filter(
        self,
        min_confidence: Optional[float],
        confidence_field: str = "confidence",
    ) -> "CypherFilterBuilder":
        """
        Add minimum confidence filter.

        Args:
            min_confidence: Minimum confidence value, or None to skip
            confidence_field: Name of the confidence field

        Returns:
            self for chaining
        """
        if min_confidence is not None:
            self._conditions.append(
                f"{self.node_alias}.{confidence_field} >= $min_confidence"
            )
        return self

    def add_custom(self, condition: str) -> "CypherFilterBuilder":
        """
        Add a custom condition.

        Args:
            condition: Raw Cypher condition (e.g., "n.type IN $types")

        Returns:
            self for chaining
        """
        if condition:
            self._conditions.append(condition)
        return self

    def build(self, prefix: str = "WHERE") -> str:
        """
        Build the complete WHERE clause.

        Args:
            prefix: Clause prefix (default: "WHERE")

        Returns:
            Complete WHERE clause or "WHERE true" if no conditions
        """
        if not self._conditions:
            return f"{prefix} true" if prefix else ""
        return f"{prefix} " + " AND ".join(self._conditions)

    def build_inline(self) -> str:
        """
        Build conditions for inline use (after existing WHERE).

        Returns:
            " AND condition1 AND condition2..." or empty string
        """
        if not self._conditions:
            return ""
        return " AND " + " AND ".join(self._conditions)

    def has_conditions(self) -> bool:
        """Check if any conditions have been added."""
        return len(self._conditions) > 0

    def clear(self) -> "CypherFilterBuilder":
        """Clear all conditions."""
        self._conditions.clear()
        return self


def neo4j_datetime_to_iso(dt: Any) -> Optional[str]:
    """
    Convert Neo4j DateTime object to ISO string.

    Handles the 8+ occurrences of this conversion pattern:
        if entity.get("last_accessed"):
            entity["last_accessed"] = entity["last_accessed"].isoformat()

    Args:
        dt: Neo4j DateTime object, Python datetime, or None

    Returns:
        ISO format string or None
    """
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


# Common datetime fields found in memory records
COMMON_DATETIME_FIELDS = [
    "timestamp",
    "created_at",
    "last_accessed",
    "first_seen",
    "last_used",
    "started_at",
    "updated_at",
    "consolidated_at",
]


def convert_record_datetimes(
    record: dict[str, Any],
    fields: Optional[List[str]] = None,
) -> dict[str, Any]:
    """
    Convert all DateTime fields in a record to ISO strings.

    Useful for preparing Neo4j query results for JSON serialization.

    Args:
        record: Dictionary from Neo4j query result
        fields: Specific fields to convert (default: common datetime fields)

    Returns:
        Record with converted datetime fields (mutates original)

    Usage:
        result = session.run("MATCH (e:Entity) RETURN e")
        entities = [convert_record_datetimes(dict(r)) for r in result]
    """
    if fields is None:
        fields = COMMON_DATETIME_FIELDS

    for field_name in fields:
        if field_name in record and record[field_name] is not None:
            record[field_name] = neo4j_datetime_to_iso(record[field_name])

    return record


def validate_pagination(
    offset: int = 0,
    limit: int = 20,
    max_limit: int = 100,
) -> tuple[int, int]:
    """
    Validate and clamp pagination parameters.

    Args:
        offset: Requested offset
        limit: Requested limit
        max_limit: Maximum allowed limit

    Returns:
        Tuple of (validated_offset, validated_limit)
    """
    return (
        max(0, int(offset)),
        min(max(1, int(limit)), max_limit),
    )


def validate_time_window(hours: int, max_hours: int = 8760) -> int:
    """
    Validate time window in hours.

    Args:
        hours: Requested hours
        max_hours: Maximum allowed (default: 1 year)

    Returns:
        Validated hours value
    """
    return max(1, min(int(hours), max_hours))


@dataclass
class SQLFilterBuilder:
    """
    Builder for PostgreSQL WHERE clauses.

    Similar to CypherFilterBuilder but for SQL queries.

    Usage:
        builder = SQLFilterBuilder()
        builder.add_user_filter(user_id).add_channel_filter(channel)
        where = builder.build()  # "WHERE user_id = :user_id AND ..."
        params = builder.params  # {"user_id": "...", "channel": "..."}
    """

    _conditions: List[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def add_user_filter(self, user_id: Optional[str]) -> "SQLFilterBuilder":
        """Add user_id filter if provided."""
        if user_id:
            self._conditions.append("user_id = :user_id")
            self.params["user_id"] = user_id
        return self

    def add_channel_filter(
        self,
        channel: Optional[str],
        include_global: bool = True,
    ) -> "SQLFilterBuilder":
        """Add channel filter with standard semantics."""
        if channel == "_all" or channel is None:
            pass
        elif channel == "_global":
            self._conditions.append("channel = '_global'")
        else:
            if include_global:
                self._conditions.append(
                    "(channel = :channel OR channel = '_global')"
                )
            else:
                self._conditions.append("channel = :channel")
            self.params["channel"] = channel
        return self

    def add_custom(self, condition: str, **params: Any) -> "SQLFilterBuilder":
        """Add custom condition with parameters."""
        if condition:
            self._conditions.append(condition)
            self.params.update(params)
        return self

    def build(self, prefix: str = "WHERE") -> str:
        """Build the WHERE clause."""
        if not self._conditions:
            return ""
        return f"{prefix} " + " AND ".join(self._conditions)

    def build_inline(self) -> str:
        """Build for inline use after existing WHERE."""
        if not self._conditions:
            return ""
        return " AND " + " AND ".join(self._conditions)
