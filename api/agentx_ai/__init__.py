"""
AgentX AI - Core API module

Version constants for API compatibility checking.
"""

# Semantic version of the API (follows semver)
VERSION = "0.17.0"

# Protocol version - integer that increments on breaking API changes
# Clients must match this exactly to connect
PROTOCOL_VERSION = 1

# Minimum compatible client version (semver)
# Clients below this version will be prompted to upgrade
MIN_CLIENT_VERSION = "0.17.0"

# Version tuple for programmatic comparison
VERSION_INFO = tuple(int(x) for x in VERSION.split('.'))


def get_version_info() -> dict:
    """Return complete version information for API responses."""
    return {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "min_client_version": MIN_CLIENT_VERSION,
    }
