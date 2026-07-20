"""Content blocks — the internal multi-modal payload vocabulary.

One small, neutral module defining the typed payload units that flow between
providers, the MCP client, the streaming tool loop, and exhibits. The shapes
deliberately mirror the **MCP ContentBlock** union (which Zed's Agent Client
Protocol reuses verbatim): ``text`` / ``image`` / ``audio`` / ``resource_link``
/ ``resource``, with base64 ``data`` + ``mimeType``. Speaking that vocabulary
internally means MCP tool outputs forward without transformation and any future
ACP/A2A bridge is a serializer, not a redesign.

This is a *seam*, not a framework: existing paths keep their shapes; new
payload handling (non-text provider output, MCP media passthrough) speaks
blocks. Wire dicts use the MCP camelCase key (``mimeType``) via aliases —
``to_wire()`` / ``block_from_wire`` round-trip that form.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class _Block(BaseModel):
    """Base config: accept snake_case or MCP camelCase on input."""

    model_config = ConfigDict(populate_by_name=True)

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the MCP/ACP wire shape (camelCase keys, no Nones)."""
        return self.model_dump(by_alias=True, exclude_none=True)


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(_Block):
    """Base64 image payload (e.g. an image-output model's completion, or an
    MCP tool's ImageContent)."""

    type: Literal["image"] = "image"
    data: str  # base64
    mime_type: str = Field(alias="mimeType")


class AudioBlock(_Block):
    """Base64 audio payload (e.g. an audio-output model's completion, or an
    MCP tool's AudioContent)."""

    type: Literal["audio"] = "audio"
    data: str  # base64
    mime_type: str = Field(alias="mimeType")


class ResourceLinkBlock(_Block):
    """A reference to a resource the client/agent may fetch — not embedded bytes."""

    type: Literal["resource_link"] = "resource_link"
    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")


class ResourceContents(_Block):
    """The contents of an embedded resource: ``text`` or base64 ``blob``."""

    type: Literal["resource_contents"] = "resource_contents"
    uri: str
    text: str | None = None
    blob: str | None = None  # base64
    mime_type: str | None = Field(default=None, alias="mimeType")


class ResourceBlock(_Block):
    """An embedded resource (MCP ``EmbeddedResource``)."""

    type: Literal["resource"] = "resource"
    resource: ResourceContents


ContentBlock = Annotated[
    TextBlock | ImageBlock | AudioBlock | ResourceLinkBlock | ResourceBlock,
    Field(discriminator="type"),
]


class _WireAdapter(BaseModel):
    """Internal: validates one wire dict into the discriminated union."""

    model_config = ConfigDict(populate_by_name=True)
    block: ContentBlock


def block_from_wire(raw: dict[str, Any]) -> Any | None:
    """Parse one wire dict into a typed block. Returns ``None`` (logged) on an
    unknown/malformed shape — payload plumbing must never break a turn."""
    try:
        return _WireAdapter(block=raw).block  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"content block unparsed ({raw.get('type') if isinstance(raw, dict) else type(raw)}): {e}")
        return None


def block_from_mcp(item: Any) -> Any | None:
    """Convert one ``mcp.types`` content item into a typed block.

    Handles TextContent / ImageContent / AudioContent / ResourceLink /
    EmbeddedResource; anything else returns ``None`` so the caller can apply
    its own fallback (the tool executor stringifies unknowns).
    """
    from mcp.types import (
        AudioContent,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
    )

    if isinstance(item, TextContent):
        return TextBlock(text=item.text)
    if isinstance(item, ImageContent):
        return ImageBlock(data=item.data, mimeType=item.mimeType)
    if isinstance(item, AudioContent):
        return AudioBlock(data=item.data, mimeType=item.mimeType)
    if isinstance(item, ResourceLink):
        return ResourceLinkBlock(
            uri=str(item.uri),
            name=item.name,
            description=item.description,
            mimeType=item.mimeType,
        )
    if isinstance(item, EmbeddedResource):
        res = item.resource
        return ResourceBlock(
            resource=ResourceContents(
                uri=str(res.uri),
                text=getattr(res, "text", None),
                blob=getattr(res, "blob", None),
                mimeType=getattr(res, "mimeType", None),
            )
        )
    return None
