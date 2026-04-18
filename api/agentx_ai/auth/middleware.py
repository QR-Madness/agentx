"""Authentication middleware for AgentX.

Gates all /api/* routes behind authentication unless:
- Auth is disabled via settings
- Route is in PUBLIC_ROUTES list
- Request is from localhost and bypass is enabled
"""

import logging
from django.conf import settings
from django.http import JsonResponse

from .service import get_auth_service

logger = logging.getLogger(__name__)


class AgentXAuthMiddleware:
    """
    Middleware that enforces authentication on API routes.

    Checks for X-Auth-Token header and validates session.
    Attaches user info to request on successful auth.
    """

    # Routes that don't require authentication
    PUBLIC_ROUTES = [
        "/api/health",
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
        # Check if auth should be bypassed
        if self._should_bypass(request):
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
            return self._unauthorized("Authentication required")

        # Validate session
        session_data = self.auth_service.validate_session(token, extend_ttl=True)
        if not session_data:
            return self._unauthorized("Invalid or expired session")

        # Attach user info to request
        request.agentx_user = {
            "user_id": session_data["user_id"],
            "username": session_data["username"],
            "session_created": session_data["created_at"],
        }
        request.agentx_token = token

        return self.get_response(request)

    def _should_bypass(self, request) -> bool:
        """
        Check if authentication should be bypassed.

        Bypass conditions:
        1. AGENTX_AUTH_ENABLED is False
        2. DEBUG is True AND request is from localhost AND bypass_localhost is enabled
        """
        # Check if auth is disabled globally
        auth_enabled = getattr(settings, 'AGENTX_AUTH_ENABLED', False)
        if not auth_enabled:
            return True

        # Check localhost bypass in debug mode
        bypass_localhost = getattr(settings, 'AGENTX_AUTH_BYPASS_LOCALHOST', True)
        if settings.DEBUG and bypass_localhost and self._is_localhost(request):
            return True

        return False

    def _is_localhost(self, request) -> bool:
        """Check if request is from localhost."""
        # Check various ways the client IP might be identified
        forwarded_for = request.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.META.get("REMOTE_ADDR", "")

        localhost_ips = {"127.0.0.1", "::1", "localhost"}
        return client_ip in localhost_ips

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
