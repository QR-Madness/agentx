"""
AgentX Utilities Package.

Common utilities, decorators, and helpers for the API layer.
"""

from .decorators import lazy_singleton
from .responses import (
    json_error,
    json_success,
    parse_json_body,
    paginate_request,
    require_methods,
    handle_options,
    PaginationInfo,
)

__all__ = [
    # Decorators
    "lazy_singleton",
    # Response helpers
    "json_error",
    "json_success",
    "parse_json_body",
    "paginate_request",
    "require_methods",
    "handle_options",
    "PaginationInfo",
]
