"""Shared image-generation helper.

One place that turns a prompt into a stored, served image: call the provider, persist
the bytes as workspace media, and meter the cost. Used by both the ``generate_image``
internal tool (a text agent calling a tool) and the direct image-generation conversation
flow (an image-output model used *as* the agent — see ``views.agent_chat_stream``), so the
two never drift.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# content-type → file extension for the stored blob's filename.
IMAGE_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def _slug_from_prompt(prompt: str, *, max_len: int = 40) -> str:
    """A readable, filesystem-safe stem from the image prompt (lowercase, hyphenated,
    truncated). Falls back to ``image`` when the prompt has nothing usable — so a
    generated file reads like ``a-web-banner-for-docs`` instead of a bare timestamp."""
    slug = re.sub(r"[^a-z0-9]+", "-", (prompt or "").lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "image"


async def generate_and_store_image(
    prompt: str,
    *,
    provider: Any,
    model: str,
    workspace_id: str | None = None,
    user_id: str = "default",
    conversation_id: str | None = None,
    agent_id: str | None = None,
    input_refs: list[Any] | None = None,
) -> dict[str, Any]:
    """Generate an image, store it as workspace media, and record usage.

    ``provider.generate_image`` is awaited (it parses the base64 image off the
    completion). ``input_refs`` (MediaRefs of source images) make it
    **image-to-image** — resolved to base64 via the shared ``resolve_media_data``
    (stale refs drop silently; the prompt still runs as text-to-image). The bytes
    are stored under the conversation's attached workspace if given, else the
    user's **Home** workspace, behind the ``generated/`` prefix. Cost metering is
    best-effort. Returns ``{url, doc_id, workspace_id, content_type, prompt}``;
    raises on a provider/store failure (callers decide how to surface it).
    """
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import store_media
    from ..providers.base import resolve_media_data

    input_images: list[tuple[str, str]] = []
    for ref in input_refs or []:
        resolved = resolve_media_data(ref)
        if resolved is not None:
            input_images.append(resolved)

    result = await provider.generate_image(
        prompt, model=model, input_images=input_images or None,
    )

    ws_id = workspace_id or repository.ensure_home_workspace(user_id)["id"]
    ext = IMAGE_EXT.get(result.content_type, "png")
    # Prompt-derived name + a short unique tail (store_media doesn't collision-check).
    suffix = datetime.now(UTC).strftime("%H%M%S%f")[-6:]
    doc = store_media(
        workspace_id=ws_id,
        filename=f"generated/{_slug_from_prompt(prompt)}-{suffix}.{ext}",
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
    two never disagree on what counts as an image model. One-line predicate over the
    shared warm-once probe (``providers.capabilities``). Never raises: ``False`` on
    any probe error (degrade to the normal text path rather than wrongly routing to
    image generation).
    """
    from ..providers.capabilities import has_output_modality, probe_model_capability

    return await probe_model_capability(
        provider, model_id, lambda c: has_output_modality(c, "image"),
        caps, tag="image-detect",
    )


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
    input_refs: list[Any] | None = None,
) -> dict[str, Any]:
    """Generate an image and build the displayable seam shared by every image path.

    Runs ``generate_and_store_image`` then wraps the stored blob in an ``image``
    exhibit. ``input_refs`` (source-image MediaRefs) make it image-to-image. This is
    the ONE place the direct chat turn (``views``) and a delegated image-only
    specialist (``AlloyExecutor``) both call, so a change to how a generated image
    surfaces updates both. Never raises — a failure comes back as an error note.

    Returns ``{ok, exhibit_wire | None, note, workspace_id | None, doc_id | None}``.
    """
    from ..streaming.exhibits import image_exhibit_from_generate

    try:
        info = await generate_and_store_image(
            prompt, provider=provider, model=model, workspace_id=workspace_id,
            user_id=user_id, conversation_id=conversation_id, agent_id=agent_id,
            input_refs=input_refs,
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
