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
        self._write(data)

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
