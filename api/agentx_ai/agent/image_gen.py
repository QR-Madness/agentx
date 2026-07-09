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


async def model_outputs_image(provider: Any, model_id: str, caps: Any = None) -> bool:
    """True when the model can output images (``output_modalities`` includes ``image``).

    Neutral home (shared by the direct chat path and the delegation executor) so the
    two never disagree on what counts as an image model. An uncached model reports the
    default ``["text"]``, so — mirroring ``core._model_supports_tools`` — warm the
    provider catalog once when caps look cold and re-check. ``caps`` may be passed in
    when the caller already has it. Never raises: returns ``False`` on any probe error
    (degrade to the normal text path rather than wrongly routing to image generation).
    """
    try:
        if caps is None:
            caps = provider.get_capabilities(model_id)
        mods = [str(m).lower() for m in (getattr(caps, "output_modalities", None) or [])]
        if "image" in mods:
            return True
        warm = getattr(provider, "fetch_models", None)
        if warm is not None:
            await warm()
            caps = provider.get_capabilities(model_id)
            mods = [str(m).lower() for m in (getattr(caps, "output_modalities", None) or [])]
        return "image" in mods
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[image-detect] capability probe failed, treating as non-image: {e}")
        return False


async def generate_image_exhibit(
    prompt: str,
    *,
    provider: Any,
    model: str,
    exhibit_id: str,
    workspace_id: str | None = None,
    user_id: str = "default",
    conversation_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Generate an image and build the displayable seam shared by every image path.

    Runs ``generate_and_store_image`` then wraps the stored blob in an ``image``
    exhibit. This is the ONE place the direct chat turn (``views``) and a delegated
    image-only specialist (``AlloyExecutor``) both call, so a change to how a generated
    image surfaces updates both. Never raises — a failure comes back as an error note.

    Returns ``{ok, exhibit_wire | None, note, workspace_id | None, doc_id | None}``.
    """
    from ..streaming.exhibits import image_exhibit_from_generate

    try:
        info = await generate_and_store_image(
            prompt, provider=provider, model=model, workspace_id=workspace_id,
            user_id=user_id, conversation_id=conversation_id, agent_id=agent_id,
        )
    except Exception as e:  # noqa: BLE001 — never break the calling turn
        logger.warning(f"Image generation failed: {e}")
        return {
            "ok": False, "exhibit_wire": None,
            "note": f"Image generation failed: {str(e)[:200]}",
            "workspace_id": None, "doc_id": None,
        }

    exhibit = image_exhibit_from_generate(info["url"], exhibit_id=exhibit_id, alt=prompt)
    return {
        "ok": True,
        "exhibit_wire": exhibit.model_dump() if exhibit is not None else None,
        "note": "🖼️ Generated an image.",
        "workspace_id": info["workspace_id"],
        "doc_id": info["doc_id"],
    }
