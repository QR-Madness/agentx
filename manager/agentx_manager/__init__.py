"""AgentX deployment manager.

Cluster lifecycle (create/up/down/destroy/status) over Docker Compose, one
codepath for both flavors of the isolation axis:

- **source** clusters (dev, repo checkout): build the API from the local
  Dockerfile via the build overlay; live under ``clusters/<name>/``.
- **image** deployments (isolated, deploy bundle): pull the published image;
  the bundle directory itself is the deployment root.

The CLI (``agentx-manager``) and the web GUI (``agentx_manager.server``) are
thin layers over the same core functions, so everything the GUI does is unit
testable without Docker.
"""

__version__ = "0.1.0"
