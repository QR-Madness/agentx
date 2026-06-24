"""Shared image-generation helper.

One place that turns a prompt into a stored, served image: call the provider, persist
the bytes as workspace media, and meter the cost. Used by both the ``generate_image``
internal tool (a text agent calling a tool) and the direct image-generation conversation
flow (an image-output model used *as* the agent — see ``views.agent_chat_stream``), so the
two never drift.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# content-type → file extension for the stored blob's filename.
IMAGE_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


async def generate_and_store_image(
    prompt: str,
    *,
    provider: Any,
    model: str,
    workspace_id: str | None = None,
    user_id: str = "default",
    conversation_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Generate an image, store it as workspace media, and record usage.

    ``provider.generate_image`` is awaited (it parses the base64 image off the
    completion). The bytes are stored under the conversation's attached workspace if
    given, else the user's **Home** workspace, behind the ``generated/`` prefix. Cost
    metering is best-effort. Returns ``{url, doc_id, workspace_id, content_type, prompt}``;
    raises on a provider/store failure (callers decide how to surface it).
    """
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import store_media

    result = await provider.generate_image(prompt, model=model)

    ws_id = workspace_id or repository.ensure_home_workspace(user_id)["id"]
    ext = IMAGE_EXT.get(result.content_type, "png")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    doc = store_media(
        workspace_id=ws_id,
        filename=f"generated/{stamp}.{ext}",
        content_type=result.content_type,
        raw=result.image,
    )

    try:  # metering is best-effort — never fail the generation over it
        from ..providers.pricing import estimate_image_cost
        from .usage_ledger import record_usage

        record_usage(
            source="image",
            model=model,
            provider=getattr(provider, "name", None),
            conversation_id=conversation_id,
            agent_id=agent_id,
            units={"images": 1},
            cost=estimate_image_cost(model=model, images=1),
        )
    except Exception as _uerr:  # noqa: BLE001
        logger.debug(f"image usage record skipped: {_uerr}")

    return {
        "url": f"/api/workspaces/{ws_id}/documents/{doc['id']}/raw",
        "doc_id": doc["id"],
        "workspace_id": ws_id,
        "content_type": result.content_type,
        "prompt": prompt,
    }
