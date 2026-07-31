"""Endpoint egress guard for user-supplied provider URLs.

Custom providers let a user hand the API an arbitrary base URL that the server
then fetches — a server-side request forgery primitive if left open. On a hosted
cluster that URL could name the container network, the Docker daemon, or a cloud
metadata endpoint (``169.254.169.254``), so the API would happily proxy a
credential dump back to whoever registered the provider.

The policy differs by deployment shape, because so does the threat:

- **Local / desktop** — the user already owns the machine, and private addresses
  are the *point* (LM Studio on the LAN, Ollama on ``localhost``). Allowed.
- **Cluster-exposed** — the API is reachable beyond the machine it runs on, so
  private space is off limits by default.

``AGENTX_AUTH_ENABLED`` is *not* the discriminator: it defaults to true on a
plain local install, so keying off it would block the LAN LM Studio setup that
is the whole point of the local case. The real signal that this API is exposed is
the cluster wiring — ``AGENTX_PUBLIC_HOST`` (the reverse-proxy knob) or
``AGENTX_GATEWAY_TOKEN`` (the nginx shared secret).

``providers.policy.allow_private_endpoints`` is the operator's override and always
wins; the deployment shape only picks its default.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlsplit

from ..config import ConfigManager, get_config_manager

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})


class EndpointNotAllowed(ValueError):
    """A provider base URL failed the egress policy."""


def is_cluster_exposed() -> bool:
    """Whether this API is fronted by the cluster gateway / a public host."""
    return bool(
        os.environ.get("AGENTX_PUBLIC_HOST", "").strip()
        or os.environ.get("AGENTX_GATEWAY_TOKEN", "").strip()
    )


def private_endpoints_allowed(cfg: ConfigManager | None = None) -> bool:
    """Whether private/loopback provider endpoints are permitted here.

    An explicit ``providers.policy.allow_private_endpoints`` wins outright. With
    nothing configured the default follows the deployment: allowed locally,
    blocked once the API is cluster-exposed.
    """
    configured = (cfg or get_config_manager()).get("providers.policy.allow_private_endpoints")
    if configured is not None:
        return bool(configured)
    return not is_cluster_exposed()


def _resolved_addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every IP a host resolves to (empty when resolution fails).

    All records are checked, not just the first — a DNS name that returns both a
    public and a private answer must not slip through on the public one.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return []
    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return addresses


def _is_private(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Loopback, link-local (incl. cloud metadata), private, or otherwise reserved."""
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def assert_public_endpoint(url: str, cfg: ConfigManager | None = None) -> str:
    """Validate a provider base URL against the egress policy.

    Returns the URL unchanged when it passes; raises ``EndpointNotAllowed`` with a
    message written for the person who typed it.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise EndpointNotAllowed("Enter a base URL, like https://api.groq.com/openai/v1.")

    parts = urlsplit(candidate)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise EndpointNotAllowed(
            f"'{parts.scheme or candidate}' isn't a supported scheme — use http or https."
        )
    if not parts.hostname:
        raise EndpointNotAllowed(f"'{candidate}' has no host — include one, like https://host/v1.")

    if private_endpoints_allowed(cfg):
        return candidate

    addresses = _resolved_addresses(parts.hostname)
    if not addresses:
        raise EndpointNotAllowed(f"Couldn't resolve '{parts.hostname}' — check the address.")
    if any(_is_private(address) for address in addresses):
        raise EndpointNotAllowed(
            f"'{parts.hostname}' resolves to a private address. This server only "
            "reaches public endpoints."
        )
    return candidate
