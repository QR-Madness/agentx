"""
AgentX AI - Core API module

Version constants loaded from versions.yaml (single source of truth).
"""

from pathlib import Path
import yaml

# Load version info from versions.yaml
_versions_path = Path(__file__).parent.parent.parent / "versions.yaml"

try:
    with open(_versions_path) as f:
        _versions = yaml.safe_load(f)

    VERSION = _versions["api"]["version"]
    PROTOCOL_VERSION = _versions["api"]["protocol_version"]
    MIN_CLIENT_VERSION = _versions["api"]["min_client_version"]
except (FileNotFoundError, KeyError, TypeError) as e:
    # Fallback for edge cases (e.g., running from unusual locations)
    import warnings
    warnings.warn(f"Could not load versions.yaml: {e}. Using fallback values.")
    VERSION = "0.0.0"
    PROTOCOL_VERSION = 0
    MIN_CLIENT_VERSION = "0.0.0"

# Version tuple for programmatic comparison
VERSION_INFO = tuple(int(x) for x in VERSION.split('.'))


def get_version_info() -> dict:
    """Return complete version information for API responses."""
    return {
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "min_client_version": MIN_CLIENT_VERSION,
    }
