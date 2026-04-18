"""AgentX Authentication Module.

Provides session-based authentication for the AgentX API with:
- Root user authentication stored in PostgreSQL
- Session tokens stored in Redis with TTL
- Middleware for route protection
"""

from .service import AuthService, get_auth_service
from .middleware import AgentXAuthMiddleware

__all__ = [
    "AuthService",
    "get_auth_service",
    "AgentXAuthMiddleware",
]
