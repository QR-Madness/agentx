"""Minimal environment for sandboxed commands.

The API process env holds secrets (ANTHROPIC/OPENAI keys, POSTGRES_PASSWORD, …), so a
shell command must NEVER inherit ``os.environ``. This builds a small, fixed env instead.
"""

from __future__ import annotations

import os

_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def minimal_env(home: str) -> dict[str, str]:
    """A fixed, secret-free environment for a sandboxed command.

    ``home``/``TMPDIR`` point at the jailed work dir so tools write only inside the jail.
    Locale is passed through (display only), everything else is omitted by design.
    """
    return {
        "PATH": _SAFE_PATH,
        "HOME": home,
        "TMPDIR": "/tmp",  # noqa: S108 - jailed tmpfs inside the sandbox, not a host temp path
        "TERM": "dumb",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    }
