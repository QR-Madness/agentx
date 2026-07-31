"""OpenRouter account linking over OAuth PKCE.

OpenRouter mints a **user-controlled** API key in one round trip: send the user
to ``/auth`` with a PKCE challenge, receive a code on the callback, exchange it
at ``POST /api/v1/auth/keys``. This module owns the half that must not leak — the
verifier, the pending-flow bookkeeping, and the callback address.

Three constraints shape the design, all verified against the live service:

1. **``/auth`` accepts no ``state`` parameter.** It takes only ``callback_url``,
   ``code_challenge`` and ``code_challenge_method``. The usual CSRF/correlation
   channel simply doesn't exist, so the nonce rides the callback **path**
   (``…/callback/<nonce>``) — the one part of the URL guaranteed to survive the
   redirect. Security-wise it plays the same role as ``state``: unguessable,
   single-use, and short-lived.
2. **The callback base is derived, not observed.** ``request.build_absolute_uri``
   returns ``http://`` behind the cluster's TLS-terminating proxy (Django has no
   ``SECURE_PROXY_SSL_HEADER`` configured), and OpenRouter requires https for any
   non-localhost callback. So the base comes from ``AGENTX_PUBLIC_HOST`` — the
   same knob the MCP redirect uses — falling back to loopback for a local install.
3. **The minted key never touches the browser.** The exchange happens server-side
   in the callback view; the client only learns that a flow finished.

Single-flight: starting a link cancels any pending one. Flows expire after
``FLOW_TTL_S``, and each nonce resolves exactly once.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: A consent screen left open longer than this is abandoned.
FLOW_TTL_S = 600.0

AUTH_URL = "https://openrouter.ai/auth"
KEY_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
#: Where a user manages (and revokes) the keys we mint for them.
KEYS_PAGE_URL = "https://openrouter.ai/settings/keys"

CALLBACK_PREFIX = "/api/providers/openrouter/oauth/callback/"

#: Default when nothing indicates a public host — the desktop app's local API.
LOCAL_API_BASE = "http://localhost:12319"


@dataclass
class LinkFlow:
    """One pending account link, keyed by an unguessable nonce."""

    nonce: str
    verifier: str
    callback_url: str
    authorization_url: str
    created_at: float = field(default_factory=time.monotonic)
    #: 'pending' until the callback resolves, then 'linked' or 'error'.
    status: str = "pending"
    error: str | None = None
    user_id: str | None = None

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created_at > FLOW_TTL_S


_LOCK = threading.Lock()
_FLOWS: dict[str, LinkFlow] = {}


def _prune_locked() -> None:
    for nonce in [n for n, flow in _FLOWS.items() if flow.expired]:
        _FLOWS.pop(nonce, None)


def _pkce_pair() -> tuple[str, str]:
    """A PKCE ``(verifier, challenge)`` using S256.

    The verifier is 43–128 unreserved characters per RFC 7636; the challenge is
    its base64url-encoded SHA-256 with padding stripped.
    """
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def callback_base() -> str:
    """The origin OpenRouter should redirect back to.

    ``AGENTX_PUBLIC_HOST`` (the cluster reverse-proxy knob) implies https;
    ``AGENTX_OPENROUTER_CALLBACK_BASE`` overrides everything for odd topologies.
    Otherwise this is a local install and loopback is correct — OpenRouter
    explicitly supports localhost callbacks on any port.
    """
    explicit = os.environ.get("AGENTX_OPENROUTER_CALLBACK_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")
    public_host = os.environ.get("AGENTX_PUBLIC_HOST", "").strip()
    if public_host:
        return f"https://{public_host}"
    return LOCAL_API_BASE


def is_local_callback(base: str | None = None) -> bool:
    """Whether the callback lands on loopback.

    Drives an honest UI note: OpenRouter titles localhost apps by host:port, so
    the consent screen reads ``localhost:12319`` rather than "AgentX". Telling
    the user that *before* the hop keeps it from reading as a phishing smell.
    """
    resolved = (base or callback_base()).lower()
    return "localhost" in resolved or "127.0.0.1" in resolved


def begin_flow() -> LinkFlow:
    """Start a link, superseding any pending one (single-flight)."""
    from urllib.parse import quote, urlencode

    verifier, challenge = _pkce_pair()
    nonce = secrets.token_urlsafe(32)
    callback_url = f"{callback_base()}{CALLBACK_PREFIX}{quote(nonce, safe='')}"

    query = urlencode({
        "callback_url": callback_url,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    flow = LinkFlow(
        nonce=nonce,
        verifier=verifier,
        callback_url=callback_url,
        authorization_url=f"{AUTH_URL}?{query}",
    )

    with _LOCK:
        _prune_locked()
        # Single-flight: a second "Link account" press abandons the first attempt
        # rather than leaving two live nonces that could each mint a key.
        _FLOWS.clear()
        _FLOWS[nonce] = flow
    return flow


def get_flow(nonce: str) -> LinkFlow | None:
    with _LOCK:
        _prune_locked()
        return _FLOWS.get(nonce)


def take_flow(nonce: str) -> LinkFlow | None:
    """Claim a **pending** flow for resolution. Returns None if already settled.

    Popping alone is not enough to make a nonce single-use: ``record_result``
    re-registers the flow under the same nonce so the client's status poll can
    read the outcome. Without the ``pending`` check a replayed callback would
    find that settled flow, claim it again, and mint a **second** key from the
    same authorization code (caught by
    ``test_replayed_callback_does_not_mint_a_second_key``).
    """
    with _LOCK:
        _prune_locked()
        flow = _FLOWS.get(nonce)
        if flow is None or flow.status != "pending":
            return None
        return _FLOWS.pop(nonce)


def record_result(flow: LinkFlow, *, status: str, error: str | None = None,
                  user_id: str | None = None) -> None:
    """Publish a settled flow so the client's status poll can read it once."""
    flow.status = status
    flow.error = error
    flow.user_id = user_id
    with _LOCK:
        _FLOWS[flow.nonce] = flow


def cancel_flow(nonce: str) -> bool:
    """Abandon a pending link (the user closed the consent tab)."""
    with _LOCK:
        return _FLOWS.pop(nonce, None) is not None


def exchange_code(code: str, verifier: str, timeout: float = 20.0) -> dict[str, Any]:
    """Trade an authorization code for a user-controlled API key.

    Synchronous on purpose: the caller is a browser-redirect view that also
    writes config and reloads the provider registry (which bridges async
    internally). Keeping the whole callback sync avoids nesting event loops for
    what is a single one-shot request.

    Returns the parsed ``{key, user_id}`` payload. Raises ``ValueError`` with a
    message written for the person who clicked, mapping OpenRouter's documented
    failures (400 wrong challenge method, 403 bad code/verifier).
    """
    import httpx

    payload = {
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(KEY_EXCHANGE_URL, json=payload)

    # The docs split these (400 = wrong challenge method, 403 = bad code/verifier),
    # but the live service answers **400 for an invalid code** too — so naming a
    # specific cause would be wrong in the common case. One accurate message, with
    # the real status kept in the log for whoever is debugging.
    if response.status_code in (400, 403):
        logger.warning(
            f"OpenRouter rejected the key exchange with {response.status_code}: "
            f"{response.text[:200]}"
        )
        raise ValueError(
            "OpenRouter couldn't verify this sign-in. Make sure you're signed in there, "
            "then start the link again."
        )
    if response.status_code >= 400:
        raise ValueError(f"OpenRouter returned {response.status_code} while issuing the key.")

    try:
        data = response.json()
    except ValueError as e:
        raise ValueError("OpenRouter's response wasn't readable.") from e

    key = data.get("key")
    if not key:
        raise ValueError("OpenRouter didn't return a key.")
    return {"key": str(key), "user_id": data.get("user_id")}
