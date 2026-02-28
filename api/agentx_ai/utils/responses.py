"""
API response utilities for Django views.

Consolidates JSON error responses, pagination, and request handling patterns.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Tuple
import json
import logging

from django.http import JsonResponse, HttpRequest

logger = logging.getLogger(__name__)


@dataclass
class PaginationInfo:
    """
    Pagination metadata for list endpoints.

    Usage:
        pagination = paginate_request(request)
        items = query_items(offset=pagination.offset, limit=pagination.limit)
        return json_success({
            "items": items,
            **pagination.to_dict(total=len(all_items))
        })
    """

    page: int
    limit: int
    offset: int
    total: int = 0

    @property
    def has_next(self) -> bool:
        """Check if there are more pages."""
        return (self.page * self.limit) < self.total

    @property
    def has_prev(self) -> bool:
        """Check if there are previous pages."""
        return self.page > 1

    def to_dict(self, total: Optional[int] = None) -> dict[str, Any]:
        """
        Convert to dictionary for JSON response.

        Args:
            total: Override total count (useful when computed after query)
        """
        actual_total = total if total is not None else self.total
        return {
            "page": self.page,
            "limit": self.limit,
            "total": actual_total,
            "has_next": (self.page * self.limit) < actual_total,
        }


def paginate_request(
    request: HttpRequest,
    default_limit: int = 20,
    max_limit: int = 100,
) -> PaginationInfo:
    """
    Extract and validate pagination parameters from request.

    Consolidates the repeated pagination calculation pattern:
        page = max(1, int(request.GET.get('page', 1)))
        limit = min(100, max(1, int(request.GET.get('limit', 20))))
        offset = (page - 1) * limit

    Args:
        request: Django request object
        default_limit: Default items per page (default: 20)
        max_limit: Maximum allowed limit (default: 100)

    Returns:
        PaginationInfo with validated parameters
    """
    try:
        page_param = request.GET.get("page", "1")
        # Handle case where param is a list (shouldn't happen but Django allows it)
        page_str = page_param[0] if isinstance(page_param, list) else page_param
        page = max(1, int(str(page_str)))
    except (ValueError, TypeError):
        page = 1

    try:
        limit_param = request.GET.get("limit", str(default_limit))
        limit_str = limit_param[0] if isinstance(limit_param, list) else limit_param
        limit = min(max_limit, max(1, int(str(limit_str))))
    except (ValueError, TypeError):
        limit = default_limit

    offset = (page - 1) * limit

    return PaginationInfo(page=page, limit=limit, offset=offset)


def json_error(
    message: str,
    status: int = 400,
    **extra: Any,
) -> JsonResponse:
    """
    Create a standardized JSON error response.

    Consolidates the 20+ error response patterns:
        return JsonResponse({'error': 'Some error'}, status=400)

    Args:
        message: Error message
        status: HTTP status code (default: 400)
        **extra: Additional fields to include in response

    Returns:
        JsonResponse with error structure
    """
    data = {"error": message}
    data.update(extra)
    return JsonResponse(data, status=status)


def json_success(
    data: dict[str, Any],
    status: int = 200,
) -> JsonResponse:
    """
    Create a success JSON response.

    Args:
        data: Response data dictionary
        status: HTTP status code (default: 200)

    Returns:
        JsonResponse with data
    """
    return JsonResponse(data, status=status)


def handle_options() -> JsonResponse:
    """
    Handle CORS preflight (OPTIONS) request.

    Returns empty 200 response for OPTIONS requests.
    """
    return JsonResponse({}, status=200)


def method_not_allowed(allowed: list[str]) -> JsonResponse:
    """
    Return 405 Method Not Allowed response.

    Args:
        allowed: List of allowed HTTP methods

    Returns:
        JsonResponse with 405 status
    """
    return json_error(
        f"Only {', '.join(allowed)} requests allowed",
        status=405,
    )


def parse_json_body(
    request: HttpRequest,
) -> Tuple[Optional[dict], Optional[JsonResponse]]:
    """
    Parse JSON body from request with error handling.

    Consolidates the repeated pattern:
        try:
            content = request.body.decode('utf-8')
            if not content:
                return JsonResponse({'error': 'No content'}, status=400)
            data = json.loads(content)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'Invalid JSON: {e}'}, status=400)

    Args:
        request: Django request object

    Returns:
        Tuple of (parsed_data, None) on success
        Tuple of (None, error_response) on failure

    Usage:
        data, error = parse_json_body(request)
        if error:
            return error
        # Use data...
    """
    try:
        content = request.body.decode("utf-8")
        if not content:
            return None, json_error("No content provided")
        return json.loads(content), None
    except json.JSONDecodeError as e:
        return None, json_error(f"Invalid JSON: {str(e)}")
    except UnicodeDecodeError as e:
        return None, json_error(f"Invalid encoding: {str(e)}")


def require_field(
    data: dict,
    field_name: str,
    field_type: Optional[type] = None,
) -> Optional[JsonResponse]:
    """
    Validate that a required field exists in data.

    Args:
        data: Dictionary to check
        field_name: Name of required field
        field_type: Optional type to validate against

    Returns:
        None if field exists and is valid
        JsonResponse error if field is missing or invalid type
    """
    if field_name not in data or data[field_name] is None:
        return json_error(f"Missing required field: {field_name}")

    if field_type is not None and not isinstance(data[field_name], field_type):
        return json_error(
            f"Field '{field_name}' must be of type {field_type.__name__}"
        )

    return None


def require_methods(*methods: str) -> Callable:
    """
    Decorator to enforce allowed HTTP methods with CORS support.

    Consolidates the repeated pattern:
        if request.method == 'OPTIONS':
            return JsonResponse({}, status=200)
        if request.method != 'POST':
            return JsonResponse({'error': 'Only POST...'}, status=405)

    Usage:
        @csrf_exempt
        @require_methods("POST")
        def my_endpoint(request):
            # Only POST requests reach here
            # OPTIONS is handled automatically
            ...

        @csrf_exempt
        @require_methods("GET", "POST")
        def flexible_endpoint(request):
            if request.method == "GET":
                ...
            else:  # POST
                ...

    Args:
        *methods: Allowed HTTP methods (e.g., "GET", "POST")

    Returns:
        Decorator function
    """
    allowed = [m.upper() for m in methods]

    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapped(request: HttpRequest, *args, **kwargs):
            # Always allow OPTIONS for CORS preflight
            if request.method == "OPTIONS":
                return handle_options()

            # Check if method is allowed
            if request.method not in allowed:
                return method_not_allowed(allowed)

            return view_func(request, *args, **kwargs)

        return wrapped

    return decorator
