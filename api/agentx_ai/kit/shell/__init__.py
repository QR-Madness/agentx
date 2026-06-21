"""Agent shells — sandboxed command execution scoped to a workspace.

Opt-in (``shell.enabled``, off by default): LLM-driven command execution is arbitrary
code execution, so v1 runs every command in a **bubblewrap jail** — filesystem limited
to a per-conversation work dir materialized from the workspace, network off, env scrubbed
of secrets (see the threat-model note in Development-Notes). A bare subprocess is used
only behind an explicit ``shell.allow_unsandboxed`` flag when bubblewrap is unavailable.

Layout:
  - ``sandbox.py``  — ``Sandbox`` protocol, ``BubblewrapSandbox`` / ``LocalSubprocessSandbox``, ``get_sandbox``
  - ``env.py``      — ``minimal_env`` (no API keys / secrets)
  - ``workdir.py``  — materialize a workspace into a work dir + GC + path-jail
  - ``policy.py``   — deny-list + limits (defense-in-depth; the jail is the primary control)
"""
