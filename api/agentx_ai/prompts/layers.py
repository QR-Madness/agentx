"""
Layered system-prompt stack ("Prompt Stack").

The agent's global system prompt is composed from an ordered stack of editable
:class:`~.models.PromptLayer` blocks. Built-in layers ship a ``default`` (the
sidecar, owned by the app and updated by releases); the user's edit is stored
separately as an ``override``. Effective content is the override if present, else
the default — so untouched built-ins keep receiving release improvements while
edited layers are pinned to the user's text and never silently overwritten.

Persistence: only the user's *deltas* are stored, via :class:`ConfigManager`
(``data/config.json`` under ``prompts.layers``) — per built-in
``{override?, enabled, order, base_version}`` and whole records for custom layers.
Built-in default content + versions always come from code (``BUILTIN_LAYERS``), so
bumping a default in a release is enough to offer the update.

This module is the durable store + model. Wiring it into live composition (and
migrating the legacy global-prompt/sections) is a separate step.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..config import get_config_manager
from .models import PromptLayer

logger = logging.getLogger(__name__)

# Where the user's layer deltas live (dot-notation key in data/config.json).
_CONFIG_KEY = "prompts.layers"


# =============================================================================
# Built-in layers (the shipped defaults — bump `default_version` when content
# changes so users with an override get an update prompt + diff).
# =============================================================================

BUILTIN_LAYERS: list[PromptLayer] = [
    PromptLayer(
        id="core-principles",
        title="Core Principles",
        kind="builtin",
        default_version=1,
        order=0,
        default=(
            "You are an intelligent AI assistant with advanced reasoning capabilities.\n\n"
            "Core Principles:\n"
            "- Be helpful, accurate, and thoughtful in your responses\n"
            "- Explain your reasoning when it adds value\n"
            "- Acknowledge uncertainty when you're not sure about something\n"
            "- Be concise but thorough - don't omit important details\n"
            "- Adapt your communication style to the user's needs"
        ),
    ),
    PromptLayer(
        id="reasoning-vs-results",
        title="Reasoning vs. What's Kept",
        kind="builtin",
        default_version=1,
        order=10,
        default=(
            "Your reasoning vs. what's kept:\n"
            "- Your reasoning is ephemeral — it's shown while you work, but it is NOT "
            "saved and is NOT processed into your long-term memory. Only your results "
            "(your responses and tool outputs) are consolidated into memory, along with "
            "anything you deliberately store with your memory tools.\n"
            "- So if a thought, insight, or intermediate finding is worth keeping, don't "
            "leave it buried in your reasoning where it will be lost — write it down: use "
            "the scratchpad (the `checkpoint` tool) to carry it through this conversation, "
            "or your memory tools to commit it to long-term memory."
        ),
    ),
    PromptLayer(
        id="citing-sources",
        title="Citing Sources",
        kind="builtin",
        default_version=1,
        order=20,
        default=(
            "Citing sources:\n"
            "- When you reference external sources (web pages, docs, or memory), present "
            "them as a citation exhibit, not as inline links in your prose.\n"
            "- Web search results are recorded as sources automatically — don't repeat "
            "them as inline links. To spotlight a key one, mark it as an active citation "
            "with a short quote."
        ),
    ),
]

_BUILTIN_BY_ID = {layer.id: layer for layer in BUILTIN_LAYERS}


class LayerStore:
    """Durable, delta-backed store for the prompt-layer stack.

    Reads compose the live built-ins (from code) overlaid with the user's stored
    deltas, plus any custom layers. Every mutation writes through to the config
    file so edits survive a restart (the legacy in-memory global prompt did not).
    """

    def __init__(self) -> None:
        self._config = get_config_manager()

    # --- persistence helpers ------------------------------------------------

    def _raw(self) -> dict:
        """The stored deltas dict (id -> partial state). Never the source of truth
        for built-in default content — only the user's overrides/order/enable."""
        data = self._config.get(_CONFIG_KEY, {})
        return data if isinstance(data, dict) else {}

    def _write(self, raw: dict) -> None:
        self._config.set(_CONFIG_KEY, raw)
        self._config.save()

    # --- reads --------------------------------------------------------------

    def list_layers(self) -> list[PromptLayer]:
        """All layers (built-ins overlaid with deltas + custom), sorted by order."""
        raw = self._raw()
        layers: list[PromptLayer] = []

        # Built-ins: start from code, apply stored delta if any.
        for builtin in BUILTIN_LAYERS:
            delta = raw.get(builtin.id, {})
            layers.append(
                PromptLayer(
                    id=builtin.id,
                    title=builtin.title,
                    kind="builtin",
                    default=builtin.default,
                    default_version=builtin.default_version,
                    override=delta.get("override"),
                    base_version=delta.get("base_version"),
                    enabled=delta.get("enabled", builtin.enabled),
                    order=delta.get("order", builtin.order),
                )
            )

        # Custom layers: stored whole.
        for layer_id, rec in raw.items():
            if rec.get("kind") != "custom":
                continue
            layers.append(
                PromptLayer(
                    id=layer_id,
                    title=rec.get("title", "Untitled layer"),
                    kind="custom",
                    default=None,
                    override=rec.get("override", ""),
                    enabled=rec.get("enabled", True),
                    order=rec.get("order", 1000),
                )
            )

        layers.sort(key=lambda layer: layer.order)
        return layers

    def get(self, layer_id: str) -> Optional[PromptLayer]:
        for layer in self.list_layers():
            if layer.id == layer_id:
                return layer
        return None

    def compose(self) -> str:
        """The composed global stack: enabled layers' effective content, in order."""
        parts = [layer.effective for layer in self.list_layers() if layer.enabled and layer.effective.strip()]
        return "\n\n".join(parts)

    # --- mutations (write-through) -----------------------------------------

    def _delta(self, raw: dict, layer_id: str) -> dict:
        return raw.setdefault(layer_id, {})

    def set_override(self, layer_id: str, content: str) -> Optional[PromptLayer]:
        """Set a layer's override. For a built-in this pins it to the user's text and
        stamps base_version to the current default (so later default changes diff)."""
        raw = self._raw()
        builtin = _BUILTIN_BY_ID.get(layer_id)
        if builtin is not None:
            delta = self._delta(raw, layer_id)
            delta["override"] = content
            delta["base_version"] = builtin.default_version
        elif raw.get(layer_id, {}).get("kind") == "custom":
            raw[layer_id]["override"] = content
        else:
            return None
        self._write(raw)
        return self.get(layer_id)

    def reset(self, layer_id: str) -> Optional[PromptLayer]:
        """Clear a built-in's override (back to riding the shipped default)."""
        raw = self._raw()
        if layer_id not in raw or layer_id not in _BUILTIN_BY_ID:
            return self.get(layer_id)
        raw[layer_id].pop("override", None)
        raw[layer_id].pop("base_version", None)
        if not raw[layer_id]:
            del raw[layer_id]
        self._write(raw)
        return self.get(layer_id)

    def acknowledge(self, layer_id: str) -> Optional[PromptLayer]:
        """Keep the user's override but mark the new default as seen (clears the
        'update available' badge without adopting the new default)."""
        raw = self._raw()
        builtin = _BUILTIN_BY_ID.get(layer_id)
        if builtin is None or layer_id not in raw or "override" not in raw[layer_id]:
            return self.get(layer_id)
        raw[layer_id]["base_version"] = builtin.default_version
        self._write(raw)
        return self.get(layer_id)

    def set_enabled(self, layer_id: str, enabled: bool) -> Optional[PromptLayer]:
        raw = self._raw()
        if layer_id in _BUILTIN_BY_ID:
            self._delta(raw, layer_id)["enabled"] = enabled
        elif raw.get(layer_id, {}).get("kind") == "custom":
            raw[layer_id]["enabled"] = enabled
        else:
            return None
        self._write(raw)
        return self.get(layer_id)

    def create_custom(self, title: str, content: str = "") -> PromptLayer:
        import uuid

        raw = self._raw()
        layer_id = f"custom-{uuid.uuid4().hex[:10]}"
        max_order = max((layer.order for layer in self.list_layers()), default=0)
        raw[layer_id] = {
            "kind": "custom",
            "title": title or "Untitled layer",
            "override": content,
            "enabled": True,
            "order": max_order + 10,
        }
        self._write(raw)
        return self.get(layer_id)  # type: ignore[return-value]

    def update_custom(
        self, layer_id: str, *, title: Optional[str] = None, content: Optional[str] = None
    ) -> Optional[PromptLayer]:
        raw = self._raw()
        rec = raw.get(layer_id)
        if not rec or rec.get("kind") != "custom":
            return None
        if title is not None:
            rec["title"] = title
        if content is not None:
            rec["override"] = content
        self._write(raw)
        return self.get(layer_id)

    def delete_custom(self, layer_id: str) -> bool:
        raw = self._raw()
        rec = raw.get(layer_id)
        if not rec or rec.get("kind") != "custom":
            return False
        del raw[layer_id]
        self._write(raw)
        return True

    def reorder(self, ordered_ids: list[str]) -> list[PromptLayer]:
        """Apply a new order from a full/partial list of ids (index * 10)."""
        raw = self._raw()
        for index, layer_id in enumerate(ordered_ids):
            if layer_id in _BUILTIN_BY_ID:
                self._delta(raw, layer_id)["order"] = index * 10
            elif raw.get(layer_id, {}).get("kind") == "custom":
                raw[layer_id]["order"] = index * 10
        self._write(raw)
        return self.list_layers()


_store: Optional[LayerStore] = None


def get_layer_store() -> LayerStore:
    global _store
    if _store is None:
        _store = LayerStore()
    return _store
