"""OAuth token + client-registration persistence for remote MCP servers.

Implements the python-sdk ``TokenStorage`` protocol (``mcp.client.auth``) over
per-server JSON files at ``data/mcp_oauth/{server}.json`` (0600) — the same
``data/`` bind-mount pattern as ``mcp_servers.json``. Each file holds the
OAuth tokens plus the dynamically-registered (RFC 7591) client information so
re-connects skip both the consent and registration steps until expiry.

Pre-registered credentials: providers without dynamic registration (e.g.
Google) put ``client_id``/``client_secret`` in the server config's ``auth``
block; ``FileTokenStorage`` seeds ``get_client_info`` from those when no
registration is stored.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)


def oauth_data_dir() -> Path:
    """`data/mcp_oauth/` next to `data/mcp_servers.json` (created on demand)."""
    base = Path(os.environ.get("AGENTX_DB_DIR") or Path(__file__).parents[3] / "data")
    d = base / "mcp_oauth"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(server_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", server_name) or "_unnamed"


def _token_path(server_name: str) -> Path:
    return oauth_data_dir() / f"{_safe_name(server_name)}.json"


def clear_oauth_state(server_name: str) -> bool:
    """Remove a server's persisted tokens + registration (the "Reset auth" action)."""
    path = _token_path(server_name)
    if path.exists():
        path.unlink()
        logger.info(f"Cleared OAuth state for MCP server '{server_name}'")
        return True
    return False


def has_oauth_state(server_name: str) -> bool:
    return _token_path(server_name).exists()


def _read_state(server_name: str) -> dict[str, Any]:
    """Best-effort read of a server's token file ({} when missing/corrupt)."""
    try:
        with open(_token_path(server_name)) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def has_oauth_tokens(server_name: str) -> bool:
    """True only when actual OAuth *tokens* are stored (not just a registration).

    The SDK writes the per-server file at RFC 7591 dynamic-registration time
    (``client_info`` only, *before* consent), so mere file existence is not
    proof of sign-in — a cancelled/denied attempt leaves a token-less file
    behind. This drives the "signed in" indicator; ``has_oauth_state`` (file
    existence) is only for the reset/cleanup path.
    """
    return bool(_read_state(server_name).get("tokens"))


def oauth_token_status(server_name: str) -> dict[str, Any]:
    """Static lifecycle facts about a server's stored tokens, for status display.

    ``expired`` is a tri-state: True/False when the persisted absolute
    ``expires_at`` decides it, None when the expiry is unknown (legacy file or
    a provider that issues tokens without ``expires_in``) — matching
    ``_RestoringOAuthProvider``, which trusts such tokens as-is. ``refreshable``
    reports a stored ``refresh_token``: an expired-but-refreshable session
    still connects headlessly; expired **and** unrefreshable is the state that
    genuinely needs a new interactive sign-in.
    """
    data = _read_state(server_name)
    tokens = data.get("tokens")
    if not tokens:
        return {"has_tokens": False, "expired": None, "refreshable": False}
    expires_at = data.get("expires_at")
    expired = (time.time() >= float(expires_at)) if expires_at is not None else None
    refreshable = bool(isinstance(tokens, dict) and tokens.get("refresh_token"))
    return {"has_tokens": True, "expired": expired, "refreshable": refreshable}


class FileTokenStorage:
    """``mcp.client.auth.TokenStorage`` over a per-server JSON file.

    File shape: ``{"tokens": {...} | null, "client_info": {...} | null}``.
    Written 0600 — it contains bearer/refresh tokens.
    """

    def __init__(self, server_name: str, preregistered: dict[str, Any] | None = None):
        self._path = _token_path(server_name)
        self._server_name = server_name
        # Optional pre-registered client credentials from the server config's
        # `auth` block (providers without RFC 7591 dynamic registration).
        self._preregistered = preregistered or {}

    def _read(self) -> dict[str, Any]:
        try:
            with open(self._path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            logger.warning(f"Unreadable OAuth state for '{self._server_name}': {e}")
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, 0o600)
        tmp.replace(self._path)

    # --- TokenStorage protocol (async by contract, I/O is tiny local files) ---

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        if not raw:
            return None
        try:
            return OAuthToken.model_validate(raw)
        except ValueError as e:
            logger.warning(f"Invalid stored tokens for '{self._server_name}': {e}")
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read()
        # No exclude_none: required-but-nullable fields (e.g. client_info's
        # redirect_uris) must survive the round-trip or validation fails.
        data["tokens"] = tokens.model_dump(mode="json")
        # Persist the ABSOLUTE expiry. `expires_in` is relative to issue time,
        # so on its own it can't tell a restarted process whether the token is
        # still good — and the SDK's `_initialize` doesn't restore expiry. We
        # capture it here (tokens are freshly issued at set-time) so the loader
        # can drive a proactive, headless refresh instead of a stale bearer →
        # 401 → interactive re-auth. See `read_token_expiry`.
        if tokens.expires_in is not None:
            data["expires_at"] = time.time() + int(tokens.expires_in)
        else:
            data.pop("expires_at", None)
        self._write(data)

    def read_token_expiry(self) -> float | None:
        """Absolute Unix expiry persisted alongside the tokens (None if unknown).

        `None` means either no tokens or a legacy file written before expiry was
        persisted — the loader treats that as "refresh proactively" rather than
        trusting a possibly-stale access token.
        """
        return self._read().get("expires_at")

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        if raw:
            try:
                return OAuthClientInformationFull.model_validate(raw)
            except ValueError as e:
                logger.warning(f"Invalid stored client info for '{self._server_name}': {e}")
        client_id = self._preregistered.get("client_id")
        if client_id:
            # Seed from pre-registered credentials — the SDK then skips DCR.
            return OAuthClientInformationFull(
                client_id=str(client_id),
                client_secret=(str(self._preregistered["client_secret"])
                               if self._preregistered.get("client_secret") else None),
                redirect_uris=None,
            )
        return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)
