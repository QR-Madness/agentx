"""Shared speech-generation helper.

One place that turns text into a stored, served audio clip: synthesize via the
Ambassador's speech seam (ONE model resolution: profile ``ambassador.speech_model``
→ global config → shipped default — never forked), persist the bytes as workspace
media, and return the served-blob info. Used by the ``generate_speech`` internal
tool; mirrors :mod:`agent.image_gen` so the two media-generation paths never drift.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# content-type → file extension for the stored clip's filename.
AUDIO_EXT = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/ogg": "ogg"}


def _slug_from_text(text: str, *, max_len: int = 40) -> str:
    """A readable, filesystem-safe stem from the spoken text (mirrors image_gen's
    prompt slug) — so a clip reads like ``welcome-to-the-briefing`` not a timestamp."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or "speech"


async def generate_and_store_speech(
    text: str,
    *,
    voice: str | None = None,
    model: str | None = None,
    workspace_id: str | None = None,
    user_id: str = "default",
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Synthesize speech and store it as workspace media.

    Synthesis goes through :meth:`Ambassador.synthesize` (strict TTS resolution +
    per-character metering, ``usage_source="chat_tts"``); raises
    ``SpeechUnavailable`` on an unconfigured/unsupported voice so the calling tool
    can surface a clean message. Bytes land under the conversation's attached
    workspace if given, else the user's **Home** workspace, behind ``generated/``.
    Returns ``{url, doc_id, workspace_id, content_type, text}``.
    """
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import store_media
    from .ambassador import get_ambassador

    result = await get_ambassador().synthesize(
        text, voice=voice, model=model, usage_source="chat_tts",
    )

    ws_id = workspace_id or repository.ensure_home_workspace(user_id)["id"]
    ext = AUDIO_EXT.get(result.content_type, "mp3")
    # Text-derived name + a short unique tail (store_media doesn't collision-check).
    suffix = datetime.now(UTC).strftime("%H%M%S%f")[-6:]
    doc = store_media(
        workspace_id=ws_id,
        filename=f"generated/{_slug_from_text(text)}-{suffix}.{ext}",
        content_type=result.content_type or "audio/mpeg",
        raw=result.audio,
    )

    return {
        "url": f"/api/workspaces/{ws_id}/documents/{doc['id']}/raw",
        "doc_id": doc["id"],
        "workspace_id": ws_id,
        "content_type": result.content_type,
        "text": text,
    }
