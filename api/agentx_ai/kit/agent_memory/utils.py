"""Utility functions for the agent memory system."""

from typing import Any, Dict, List
from datetime import datetime
import hashlib
import json


def generate_content_hash(content: str) -> str:
    """
    Generate a hash for content deduplication.

    Args:
        content: Text content to hash

    Returns:
        SHA-256 hex digest
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime for display.

    Args:
        dt: Datetime object

    Returns:
        Formatted timestamp string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        updates: Updates to apply

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def sanitize_cypher_string(text: str) -> str:
    """
    Sanitize string for use in Cypher queries.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    return text.replace("'", "\\'").replace('"', '\\"')


def deduplicate_by_id(items: List[Dict[str, Any]], id_key: str = "id") -> List[Dict[str, Any]]:
    """
    Deduplicate list of dictionaries by ID key.

    Args:
        items: List of dictionary items
        id_key: Key to use for deduplication

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for item in items:
        item_id = item.get(id_key)
        if item_id and item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result
