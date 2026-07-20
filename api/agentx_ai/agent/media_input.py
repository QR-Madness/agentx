"""Media input as a capability — parse, gate, and resolve attachments for ANY surface.

Extracted from views (ADR-11): the chat turn was the only consumer of the
vision/audio input gates, so delegation couldn't reuse them — and a delegated
specialist (especially a **direct-input** one with no tool loop to fetch
anything) received text only. Now the chat turn, the delegation executor, and
the image tools all speak through here.

The pieces:
- ``parse_image_refs`` / ``parse_audio_refs`` — wire dicts → validated MediaRefs.
- ``model_accepts_vision`` / ``model_accepts_audio`` — warm-once capability gates
  (one-line predicates over ``providers.capabilities``).
- ``gate_vision_images`` / ``gate_audio_input`` — the capability-driven split:
  what rides the model natively vs. what degrades (strip + notice for images;
  STT transcript lines for audio, cached onto the ref so nothing re-bills).
- ``strip_message_images`` / ``strip_message_audio`` — history hygiene for
  mid-conversation model switches.
- ``resolve_media_docs`` — document ids → typed MediaRefs with view_image-style
  access checks (attached workspace or the user's Home) — the seam that lets a
  supervisor hand media to a specialist by id.
"""

from __future__ import annotations

import logging

from ..providers.base import MediaRef

logger = logging.getLogger(__name__)

# Native `input_audio` clips beyond this raw size fall back to STT (a base64
# multi-MB clip per round trip melts context + latency; transcripts don't).
NATIVE_AUDIO_MAX_BYTES = 10 * 1024 * 1024

_AUDIO_FMT = {
    "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav",
    "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/webm": "webm",
    "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/aac": "aac",
    "audio/flac": "flac",
}


def parse_image_refs(raw_images) -> list[MediaRef]:
    """Validate wire vision-input refs into MediaRefs. Malformed / unsupported-type
    entries are dropped (a bad attachment must never 500 the turn)."""
    from ..kit.workspaces.service import IMAGE_CONTENT_TYPES

    refs = []
    for item in raw_images or []:
        if not isinstance(item, dict):
            continue
        ws, doc, mt = item.get("workspace_id"), item.get("doc_id"), item.get("media_type")
        if ws and doc and mt in IMAGE_CONTENT_TYPES:
            refs.append(MediaRef(workspace_id=ws, doc_id=doc, media_type=mt))
    return refs


def parse_audio_refs(raw_audio) -> list[MediaRef]:
    """Validate wire audio-input refs into MediaRefs (mirror of ``parse_image_refs``).
    A wire ref may carry a cached ``transcript`` (from a prior turn's STT fallback) —
    kept so a regenerate never re-transcribes."""
    from ..kit.workspaces.service import AUDIO_CONTENT_TYPES

    refs = []
    for item in raw_audio or []:
        if not isinstance(item, dict):
            continue
        ws, doc, mt = item.get("workspace_id"), item.get("doc_id"), item.get("media_type")
        if ws and doc and mt in AUDIO_CONTENT_TYPES:
            t = item.get("transcript")
            refs.append(MediaRef(
                workspace_id=ws, doc_id=doc, media_type=mt,
                transcript=t if isinstance(t, str) and t.strip() else None,
            ))
    return refs


async def model_accepts_vision(provider, model_id, caps=None) -> bool:
    """True when the model accepts image *input*: ``supports_vision`` or
    ``input_modalities`` includes ``image``. ``False`` on any probe error
    (degrade to text-only rather than sending blocks a model would 400 on)."""
    from ..providers.capabilities import has_input_modality, probe_model_capability

    return await probe_model_capability(
        provider, model_id,
        lambda c: getattr(c, "supports_vision", False) or has_input_modality(c, "image"),
        caps, tag="vision-detect",
    )


async def model_accepts_audio(provider, model_id, caps=None) -> bool:
    """True when the model accepts audio *input* (``input_modalities`` includes
    ``audio``). ``False`` on any probe error — degrade to the STT-transcript
    fallback rather than sending audio blocks a model would 400 on."""
    from ..providers.capabilities import has_input_modality, probe_model_capability

    return await probe_model_capability(
        provider, model_id, lambda c: has_input_modality(c, "audio"),
        caps, tag="audio-detect",
    )


def gate_vision_images(user_images, accepts_vision, emit) -> list[MediaRef]:
    """Return the images to send the model: the full set when it can see them, else
    empty (with a status notice). The user's images still persist + render regardless.
    ``emit`` is an ``emit_status``-shaped callable (or a no-op for surfaces without
    a status feed)."""
    if user_images and not accepts_vision:
        emit("vision_unsupported", "the selected model can't see images — sent as text only")
        return []
    return user_images


async def gate_audio_input(user_audio, accepts_audio, emit):
    """Split attached audio into (native_refs, transcript_lines).

    A clip rides **native** ``input_audio`` only when the model accepts audio AND
    its container is in the OpenRouter-native set AND it's ≤ the native size cap.
    Everything else is transcribed via the neutral speech seam (``kit.speech``)
    into a clearly-labeled transcript line; the transcript is cached onto the ref
    so persistence/regeneration never re-bills. A clip that can't transcribe
    degrades to a short notice line — never a dead turn.
    """
    from ..providers.base import INPUT_AUDIO_FORMATS

    native, lines = [], []
    if not user_audio:
        return native, lines

    for i, ref in enumerate(user_audio, start=1):
        label = f"Audio attachment {i}"
        try:
            from ..kit.workspaces import repository

            doc = repository.get_document(ref.doc_id) or {}
            fname = (doc.get("filename") or "").rsplit("/", 1)[-1]
            if fname:
                label = f'Audio attachment "{fname}"'
            size_ok = (doc.get("size_bytes") or 0) <= NATIVE_AUDIO_MAX_BYTES
        except Exception:  # noqa: BLE001
            size_ok = True

        if accepts_audio and ref.media_type in INPUT_AUDIO_FORMATS and size_ok:
            native.append(ref)
            continue

        if not ref.transcript:
            emit("transcribing", f"Transcribing audio… ({label})")
            ref.transcript = await transcribe_ref(ref)
        if ref.transcript:
            lines.append(f"[{label} — transcript]: {ref.transcript}")
        else:
            lines.append(f"[{label}: attached, but it couldn't be transcribed "
                         "and this model can't hear audio]")
    if lines and not accepts_audio:
        emit("audio_fallback", "the selected model can't hear audio — transcripts sent instead")
    return native, lines


async def transcribe_ref(ref) -> str | None:
    """Transcribe one stored clip via the neutral speech seam (shared model
    resolution; ADR-11). Returns None on any failure (callers degrade to a notice)."""
    try:
        from ..kit.speech import transcribe_audio
        from ..kit.workspaces import repository, storage

        doc = repository.get_document(ref.doc_id)
        if not doc or doc.get("workspace_id") != ref.workspace_id:
            return None
        raw = storage.read_blob(doc["storage_key"])
        if not raw:
            return None
        fmt = _AUDIO_FMT.get(ref.media_type, "webm")
        result = await transcribe_audio(raw, audio_format=fmt, usage_source="chat_stt")
        return (result.text or "").strip() or None
    except Exception as e:  # noqa: BLE001 — fallback is best-effort
        logger.warning(f"Audio transcription failed for doc {ref.doc_id}: {e}")
        return None


def strip_message_images(messages, accepts_vision):
    """Drop image blocks from every message when the model can't see them (a
    mid-conversation switch to a non-vision model can leave images on history turns).
    Non-mutating — copies so the warm session keeps its images for a later vision turn."""
    if accepts_vision:
        return messages
    return [m.model_copy(update={"images": None}) if m.images else m for m in messages]


def strip_message_audio(messages, accepts_audio):
    """Audio twin of ``strip_message_images`` — drop audio refs from history
    when the current model can't hear them (transcript lines already ride the
    turn text, so nothing is lost)."""
    if accepts_audio:
        return messages
    return [m.model_copy(update={"audio": None}) if m.audio else m for m in messages]


def resolve_media_docs(
    document_ids, *, user_id: str = "default", workspace_id: str | None = None,
) -> tuple[list[MediaRef], list[MediaRef], list[str]]:
    """Resolve document ids into typed input refs: ``(image_refs, audio_refs, notes)``.

    The seam behind media-carrying delegation and image-editing tools: a model
    references media **by document id** (ids are what catalogs/manifests/attachment
    lines show it), and this applies the same access rule as ``view_image`` — the
    doc must live in the attached workspace or the user's Home (no cross-workspace
    peeking). Unusable entries (missing, out of scope, video/other types) become
    human-readable ``notes`` instead of exceptions — a bad id must never kill a
    delegation.
    """
    from ..kit.workspaces import repository
    from ..kit.workspaces.service import AUDIO_CONTENT_TYPES, IMAGE_CONTENT_TYPES

    images: list[MediaRef] = []
    audio: list[MediaRef] = []
    notes: list[str] = []

    home_id = None
    for raw in document_ids or []:
        doc_id = (str(raw) if raw is not None else "").strip()
        if not doc_id:
            continue
        doc = repository.get_document(doc_id)
        if doc is None:
            notes.append(f"media {doc_id}: not found")
            continue
        if home_id is None:
            home_id = repository.ensure_home_workspace(user_id)["id"]
        if doc.get("workspace_id") not in {workspace_id, home_id}:
            notes.append(f"media {doc_id}: not in this project's files or Home")
            continue
        mt = doc.get("content_type")
        ref = MediaRef(workspace_id=doc["workspace_id"], doc_id=doc_id, media_type=mt or "")
        if mt in IMAGE_CONTENT_TYPES:
            images.append(ref)
        elif mt in AUDIO_CONTENT_TYPES:
            audio.append(ref)
        else:
            notes.append(f"media {doc_id}: {mt} can't be used as model input")
    return images, audio, notes


def attachment_reference_line(images, audio) -> str | None:
    """A model-visible line naming this turn's attachment document ids, so the
    agent can ``view_image`` them or hand them to a specialist via the delegation
    ``media`` param. Returns None when there's nothing attached."""
    parts = [f"{r.doc_id} ({r.media_type})" for r in (images or [])]
    parts += [f"{r.doc_id} ({r.media_type})" for r in (audio or [])]
    if not parts:
        return None
    return (
        "[Attached media on this message — document_ids (usable with view_image "
        "and the delegation `media` parameter): " + ", ".join(parts) + "]"
    )
