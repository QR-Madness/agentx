"""Authentication middleware for AgentX.

Gates all /api/* routes behind authentication unless:
- Auth is disabled via settings
- Localhost bypass is active in DEBUG mode (mirrors /api/auth/status)
- Route is in PUBLIC_ROUTES list

Tracks client IP for rate-limiting and request auditing.
"""

import logging
from django.conf import settings
from django.http import JsonResponse

from .service import get_auth_service

logger = logging.getLogger(__name__)

LOCALHOST_IPS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_auth_bypass_active(client_ip: str) -> bool:
    """
    Whether auth should be bypassed for this client.

    Must match the logic in auth/views.py::auth_status so the client and
    middleware agree on whether a token is required. If they disagree, the
    client renders the app while every protected endpoint 401s.
    """
    auth_enabled = getattr(settings, 'AGENTX_AUTH_ENABLED', False)
    if not auth_enabled:
        return True

    bypass_localhost = getattr(settings, 'AGENTX_AUTH_BYPASS_LOCALHOST', True)
    if settings.DEBUG and bypass_localhost and client_ip in LOCALHOST_IPS:
        return True

    return False


class AgentXAuthMiddleware:
    """
    Middleware that enforces authentication on API routes.

    Checks for X-Auth-Token header and validates session.
    Attaches user info and client IP to request on successful auth.
    """

    # Routes that don't require authentication
    PUBLIC_ROUTES = [
        "/api/health",
        "/api/version",
        "/api/auth/login",
        "/api/auth/status",
        "/api/auth/setup",  # Only works when setup is required
    ]

    # Routes that should always be public (even if not in list)
    PUBLIC_PREFIXES = [
        "/admin/",  # Django admin
        "/static/",  # Static files
    ]

    def __init__(self, get_response):
        """Initialize middleware."""
        self.get_response = get_response
        self._auth_service = None

    @property
    def auth_service(self):
        """Lazy-load auth service to avoid import cycles."""
        if self._auth_service is None:
            session_ttl = getattr(settings, 'AGENTX_SESSION_TTL', 86400)
            self._auth_service = get_auth_service(session_ttl=session_ttl)
        return self._auth_service

    def __call__(self, request):
        """Process request through authentication check."""
        # Always track client IP for rate-limiting and auditing
        request.agentx_client_ip = self._get_client_ip(request)

        # OPTIONS preflight requests must never be blocked by auth — the actual
        # request carries the token, not the preflight. Pass through so
        # CorsMiddleware can annotate the response on the way back.
        if request.method == 'OPTIONS':
            return self.get_response(request)

        # Honor the same bypass logic that /api/auth/status reports — auth
        # disabled globally, or localhost bypass active in DEBUG mode.
        if is_auth_bypass_active(request.agentx_client_ip):
            return self.get_response(request)

        # Check if route is public
        if self._is_public_route(request.path):
            # Special handling for /api/auth/setup - only public if setup required
            if request.path == "/api/auth/setup":
                if not self.auth_service.is_setup_required():
                    return self._unauthorized("Setup already completed")
            return self.get_response(request)

        # Extract token from header
        token = request.headers.get("X-Auth-Token")
        if not token:
            logger.debug(f"Auth required - no token from {request.agentx_client_ip}")
            return self._unauthorized("Authentication required")

        # Validate session
        session_data = self.auth_service.validate_session(token, extend_ttl=True)
        if not session_data:
            logger.debug(f"Invalid session from {request.agentx_client_ip}")
            return self._unauthorized("Invalid or expired session")

        # Attach user info to request
        request.agentx_user = {
            "user_id": session_data["user_id"],
            "username": session_data["username"],
            "session_created": session_data["created_at"],
        }
        request.agentx_token = token

        return self.get_response(request)

    def _is_auth_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return getattr(settings, 'AGENTX_AUTH_ENABLED', False)

    def _get_client_ip(self, request) -> str:
        """
        Extract client IP address from request.

        Handles X-Forwarded-For header for proxied requests (e.g., Cloudflare).
        Used for rate-limiting and request auditing.
        """
        # Check X-Forwarded-For for proxied requests
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            # Take the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()

        # Fall back to REMOTE_ADDR
        return request.META.get("REMOTE_ADDR", "unknown")

    def _is_public_route(self, path: str) -> bool:
        """Check if the path is a public route."""
        # Check exact matches
        if path in self.PUBLIC_ROUTES:
            return True

        # Check prefixes
        for prefix in self.PUBLIC_PREFIXES:
            if path.startswith(prefix):
                return True

        # Non-API routes are public (e.g., / root)
        if not path.startswith("/api/"):
            return True

        return False

    def _unauthorized(self, message: str) -> JsonResponse:
        """Return 401 Unauthorized response."""
        return JsonResponse(
            {"error": message, "status": "unauthorized"},
            status=401
        )
