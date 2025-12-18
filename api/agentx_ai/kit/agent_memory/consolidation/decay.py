"""Memory decay functions and utilities."""

from typing import Optional
from datetime import datetime, timedelta
import math


def calculate_decay(
    initial_value: float,
    time_since_access: timedelta,
    decay_rate: float = 0.95
) -> float:
    """
    Calculate decayed value based on time since last access.

    Uses exponential decay: value = initial * (decay_rate ^ days)

    Args:
        initial_value: Initial value
        time_since_access: Time since last access
        decay_rate: Daily decay rate (0-1)

    Returns:
        Decayed value
    """
    days = time_since_access.total_seconds() / (24 * 3600)
    return initial_value * (decay_rate ** days)


def calculate_importance_boost(
    base_importance: float,
    access_count: int,
    recency_hours: Optional[float] = None
) -> float:
    """
    Calculate importance score with access frequency and recency boost.

    Args:
        base_importance: Base importance score (0-1)
        access_count: Number of times accessed
        recency_hours: Hours since last access

    Returns:
        Boosted importance score
    """
    # Frequency boost (logarithmic)
    frequency_boost = math.log(access_count + 1) / 10

    # Recency boost (inverse exponential)
    recency_boost = 0
    if recency_hours is not None:
        recency_boost = math.exp(-recency_hours / 24)  # Decay over days

    # Combine boosts
    boosted = base_importance + frequency_boost + (recency_boost * 0.2)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, boosted))


def should_archive(
    created_at: datetime,
    last_accessed: datetime,
    salience: float,
    retention_days: int = 90,
    min_salience: float = 0.1
) -> bool:
    """
    Determine if a memory should be archived.

    Args:
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        salience: Salience score (0-1)
        retention_days: Retention period in days
        min_salience: Minimum salience threshold

    Returns:
        True if should be archived
    """
    age_days = (datetime.utcnow() - created_at).days
    days_since_access = (datetime.utcnow() - last_accessed).days

    # Archive if too old
    if age_days > retention_days:
        return True

    # Archive if low salience and not accessed recently
    if salience < min_salience and days_since_access > 30:
        return True

    return False


def calculate_consolidation_priority(
    conversation_turn_count: int,
    time_since_end: timedelta,
    has_user_feedback: bool = False
) -> float:
    """
    Calculate priority for consolidating a conversation.

    Args:
        conversation_turn_count: Number of turns in conversation
        time_since_end: Time since conversation ended
        has_user_feedback: Whether user provided feedback

    Returns:
        Priority score (higher = more urgent)
    """
    # Base priority from conversation length
    length_priority = min(conversation_turn_count / 20, 1.0)

    # Time priority (consolidate recent conversations first)
    hours = time_since_end.total_seconds() / 3600
    time_priority = math.exp(-hours / 24)  # Decay over days

    # Feedback boost
    feedback_boost = 0.5 if has_user_feedback else 0.0

    return length_priority * 0.4 + time_priority * 0.4 + feedback_boost * 0.2
