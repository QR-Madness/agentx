"""Authentication views for AgentX API."""

import logging
from datetime import datetime, timezone, timedelta
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ..utils.responses import json_error, json_success, parse_json_body, require_methods
from .service import get_auth_service

logger = logging.getLogger(__name__)


def _get_client_ip(request) -> str:
    """Extract client IP from request."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


@csrf_exempt
@require_methods("GET")
def auth_status(request):
    """
    Check authentication status.

    Returns whether auth is required and if initial setup is needed.

    Response:
        {
            "auth_required": bool,
            "setup_required": bool,
            "auth_bypass_active": bool
        }
    """
    auth_enabled = getattr(settings, 'AGENTX_AUTH_ENABLED', False)
    bypass_localhost = getattr(settings, 'AGENTX_AUTH_BYPASS_LOCALHOST', True)

    # Check if bypass is currently active
    bypass_active = False
    if not auth_enabled:
        bypass_active = True
    elif settings.DEBUG and bypass_localhost:
        # Check if request is from localhost
        client_ip = _get_client_ip(request)
        if client_ip in {"127.0.0.1", "::1", "localhost"}:
            bypass_active = True

    # Check if setup is required
    setup_required = False
    if auth_enabled:
        try:
            auth_service = get_auth_service()
            setup_required = auth_service.is_setup_required()
        except Exception as e:
            logger.error(f"Error checking setup status: {e}")
            # If we can't check, don't require setup (might be DB issue)
            setup_required = False

    return json_success({
        "auth_required": auth_enabled and not bypass_active,
        "setup_required": setup_required,
        "auth_bypass_active": bypass_active,
    })


@csrf_exempt
@require_methods("POST")
def auth_login(request):
    """
    Authenticate user and create session.

    Request body:
        {
            "username": "root",
            "password": "..."
        }

    Response:
        {
            "token": "...",
            "expires_at": "ISO timestamp",
            "username": "root"
        }
    """
    data = parse_json_body(request)
    if data is None:
        return json_error("Invalid JSON body", status=400)

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return json_error("Username and password required", status=400)

    auth_service = get_auth_service()

    # Check if setup is required
    if auth_service.is_setup_required():
        return json_error("Initial setup required. Use /api/auth/setup first.", status=403)

    # Attempt login
    result = auth_service.login(
        username=username,
        password=password,
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("User-Agent"),
    )

    if result is None:
        return json_error("Invalid username or password", status=401)

    token, session_data = result

    # Calculate expiration
    session_ttl = getattr(settings, 'AGENTX_SESSION_TTL', 86400)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=session_ttl)

    return json_success({
        "token": token,
        "expires_at": expires_at.isoformat(),
        "username": session_data["username"],
    })


@csrf_exempt
@require_methods("POST")
def auth_logout(request):
    """
    Invalidate current session.

    Requires X-Auth-Token header.

    Response:
        {"message": "Logged out successfully"}
    """
    token = getattr(request, 'agentx_token', None)
    if not token:
        # Also check header directly in case middleware bypassed
        token = request.headers.get("X-Auth-Token")

    if not token:
        return json_error("No active session", status=400)

    auth_service = get_auth_service()
    auth_service.logout(token)

    return json_success({"message": "Logged out successfully"})


@csrf_exempt
@require_methods("GET")
def auth_session(request):
    """
    Get current session information.

    Requires authentication.

    Response:
        {
            "user_id": int,
            "username": str,
            "session_created": str,
            "last_active": str
        }
    """
    user = getattr(request, 'agentx_user', None)
    token = getattr(request, 'agentx_token', None)

    if not user or not token:
        return json_error("Not authenticated", status=401)

    auth_service = get_auth_service()
    session_data = auth_service.validate_session(token, extend_ttl=False)

    if not session_data:
        return json_error("Session not found", status=401)

    return json_success({
        "user_id": session_data["user_id"],
        "username": session_data["username"],
        "session_created": session_data["created_at"],
        "last_active": session_data["last_active"],
    })


@csrf_exempt
@require_methods("POST")
def auth_change_password(request):
    """
    Change user password.

    Requires authentication.

    Request body:
        {
            "old_password": "...",
            "new_password": "..."
        }

    Response:
        {"message": "Password changed successfully"}
    """
    user = getattr(request, 'agentx_user', None)
    token = getattr(request, 'agentx_token', None)

    if not user:
        return json_error("Not authenticated", status=401)

    data = parse_json_body(request)
    if data is None:
        return json_error("Invalid JSON body", status=400)

    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not old_password or not new_password:
        return json_error("Both old and new passwords required", status=400)

    if len(new_password) < 8:
        return json_error("New password must be at least 8 characters", status=400)

    auth_service = get_auth_service()

    try:
        success = auth_service.change_password(
            user_id=user["user_id"],
            old_password=old_password,
            new_password=new_password
        )
    except ValueError as e:
        return json_error(str(e), status=400)

    if not success:
        return json_error("Invalid old password", status=401)

    # Invalidate all other sessions (keep current)
    auth_service.invalidate_all_sessions(except_token=token)

    return json_success({"message": "Password changed successfully"})


@csrf_exempt
@require_methods("POST")
def auth_setup(request):
    """
    Initial root password setup.

    Only works if no password is currently set.

    Request body:
        {
            "password": "...",
            "confirm_password": "..."
        }

    Response:
        {"message": "Root password configured successfully"}
    """
    auth_service = get_auth_service()

    # Check if setup is actually required
    if not auth_service.is_setup_required():
        return json_error("Setup already completed", status=403)

    data = parse_json_body(request)
    if data is None:
        return json_error("Invalid JSON body", status=400)

    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")

    if not password:
        return json_error("Password required", status=400)

    if password != confirm_password:
        return json_error("Passwords do not match", status=400)

    if len(password) < 8:
        return json_error("Password must be at least 8 characters", status=400)

    try:
        success = auth_service.setup_root_password(password)
    except ValueError as e:
        return json_error(str(e), status=400)
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return json_error("Failed to configure password", status=500)

    if not success:
        return json_error("Setup already completed", status=403)

    return json_success({"message": "Root password configured successfully"})
