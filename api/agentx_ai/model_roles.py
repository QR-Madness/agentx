"""
Model roles — an implicit resolution tier over the platform's utility models.

The platform has ~15 scattered "which model runs this internal job?" settings
(memory extraction stages, recall HyDE/self-query, compression, recaps, prompt
enhancement, …). Roles cluster them into three intelligible knobs, stored at
``models.roles.*`` in ConfigManager:

- ``fast_utility``   — quick classification & extraction; speed first
- ``deep_reasoning`` — consolidation & distillation; quality + cost-efficiency
- ``summarizer``     — cheap, reliable compression & recaps

Semantics (implicit tier — settings overhaul D1):

- A role is an **overlay, not a rewrite**: setting a role never touches member
  keys. A member follows its role only while its own value is empty/"inherit";
  an explicit member value always wins. Clearing the role restores the
  pre-roles chain exactly (unset role ⇒ this tier is a no-op).
- ``role:<name>`` is also a valid *explicit* value for any model setting
  (including non-members like the planner) and resolves through that role.
- A role holds ONE concrete ``provider:model``; provider fallback (ADR-5) runs
  after resolution, unchanged. The registry expands ``role:`` refs defensively
  so the sentinel can never reach a provider lookup.
- Role values must be concrete — a ``role:`` ref *inside* a role is ignored
  (no role-to-role chains).

This module is the single source of truth for role names, copy, and membership
— resolution, the API (``GET /api/models/roles``), and the client all read it.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ROLE_PREFIX = "role:"

ROLE_NAMES: tuple[str, ...] = ("fast_utility", "deep_reasoning", "summarizer")

ROLES: dict[str, dict[str, str]] = {
    "fast_utility": {
        "label": "Fast Utility",
        "description": "Quick classification & extraction — speed first.",
    },
    "deep_reasoning": {
        "label": "Deep Reasoning",
        "description": "Consolidation & distillation — quality and cost-efficiency.",
    },
    "summarizer": {
        "label": "Summarizer",
        "description": "Cheap, reliable compression & recaps.",
    },
}

# Membership. `kind` says where the member's explicit value lives: "memory" = a
# kit Settings field, "config" = a ConfigManager dot-path.
# Deliberately EXCLUDED (bucket b — see INHERITS_AGENT_MODEL): planner,
# ambassador answerer, images — for those, empty means "follow the agent model",
# a different semantic than "inherit a role" (they can still be set to
# `role:<name>` explicitly).
ROLE_MEMBERS: dict[str, dict[str, str]] = {
    # fast_utility
    "extraction": {
        "role": "fast_utility", "label": "Extraction",
        "kind": "memory", "source": "extraction_model",
    },
    "relevance_filter": {
        "role": "fast_utility", "label": "Relevance filter",
        "kind": "memory", "source": "relevance_filter_model",
    },
    "contradiction": {
        "role": "fast_utility", "label": "Contradiction check",
        "kind": "memory", "source": "contradiction_model",
    },
    "correction": {
        "role": "fast_utility", "label": "Correction detection",
        "kind": "memory", "source": "correction_model",
    },
    "entity_linking": {
        "role": "fast_utility", "label": "Entity linking",
        "kind": "memory", "source": "entity_linking_model",
    },
    "recall_hyde": {
        "role": "fast_utility", "label": "Recall — HyDE",
        "kind": "memory", "source": "recall_hyde_model",
    },
    "recall_self_query": {
        "role": "fast_utility", "label": "Recall — self-query",
        "kind": "memory", "source": "recall_self_query_model",
    },
    "aide": {
        "role": "fast_utility", "label": "Ambassador aide digests",
        "kind": "config", "source": "ambassador.aide.model",
    },
    "thinking_classifier": {
        "role": "fast_utility", "label": "Thinking-pattern classifier",
        "kind": "config", "source": "reasoning.classifier_model",
    },
    # deep_reasoning
    "combined_extraction": {
        "role": "deep_reasoning", "label": "Combined extraction",
        "kind": "memory", "source": "combined_extraction_model",
    },
    "procedural_distill": {
        "role": "deep_reasoning", "label": "Procedural distillation",
        "kind": "memory", "source": "procedural_distill_model",
    },
    # summarizer
    "workspace_summary": {
        "role": "summarizer", "label": "Document summaries",
        "kind": "memory", "source": "workspace_summary_model",
    },
    "rolling_summary": {
        "role": "summarizer", "label": "Rolling summary & recap",
        "kind": "config", "source": "session.rolling_summary.model",
    },
    "compression": {
        "role": "summarizer", "label": "Tool-output compression",
        "kind": "config", "source": "compression.model",
    },
    "trajectory_compression": {
        "role": "summarizer", "label": "Trajectory compression",
        "kind": "config", "source": "trajectory_compression.model",
    },
    "prompt_enhancement": {
        "role": "summarizer", "label": "Prompt enhancement",
        "kind": "config", "source": "prompt_enhancement.model",
    },
}


# --- Model-family coverage invariant (enforced by test_model_family_coverage) --
#
# Every general-purpose LLM feature-model setting must resolve through a family
# (a ROLE_MEMBERS `source` above). There are exactly two documented escape
# hatches; a `*_model` setting that is in NEITHER a family NOR one of these lists
# fails the guard. Keys are the setting's identifier — a memory Settings field
# name, a ConfigManager dot-path, or an `AgentConfig`/`SpeculativeConfig` field.

# Bucket (b): falls through to the *calling agent/profile's own model*
# (`complete_with_fallback`, Foundation #4). Empty here means "use the active
# agent model", NOT "inherit a role" — so these are intentionally family-less
# (they can still be pointed at a role explicitly via `role:<name>`).
INHERITS_AGENT_MODEL: dict[str, str] = {
    "preferences.default_model": "the primary agent model itself (the inheritance root)",
    "feature_default_model": "bulk memory root; empty → the default chat model",
    "planner.model": "the chat planner runs on the active agent's model",
    "ambassador.model": "the ambassador answers on its own profile's model",
    "reasoning_model": "AgentConfig: `reasoning_model or default_model`",
    "drafting_model": "AgentConfig: `drafting_model or default_model`",
    "reasoning.step_back_model": "step_back pre-call thinks on the active turn's model",
    "reasoning.sc_model": "self-consistency samples on the active turn's model",
}

# Bucket (c): not a general-purpose text LLM (a family can't describe it), or a
# paired constraint a single family can't express.
EXEMPT_SPECIALIZED: dict[str, str] = {
    "embedding_model": "embeddings (OpenAI) — not a chat model",
    "local_embedding_model": "embeddings (local sentence-transformers)",
    "cross_encoder_model": "cross-encoder reranker — not a chat model",
    "images.default_model": "image-generation model — not a text LLM",
    "draft_model": "speculative draft — matched pair with target_model",
    "target_model": "speculative target — matched pair with draft_model",
}


def configured_role_model(role: str) -> str:
    """The concrete model configured for a role ("" when unset/unknown).

    Literal-default reads: old installs whose config.json predates
    `models.roles` get "" without any migration (ConfigManager never merges
    new defaults into an existing file).
    """
    from .config import get_config_manager

    cfg = get_config_manager()
    if role == "fast_utility":
        value = cfg.get("models.roles.fast_utility", "")
    elif role == "deep_reasoning":
        value = cfg.get("models.roles.deep_reasoning", "")
    elif role == "summarizer":
        value = cfg.get("models.roles.summarizer", "")
    else:
        return ""
    value = (str(value) if value else "").strip()
    # No role-to-role chains: a `role:` ref inside a role is ignored.
    if value.lower().startswith(_ROLE_PREFIX):
        logger.warning(f"models.roles.{role} holds a role: ref ({value!r}) — ignored")
        return ""
    return value


def expand_role_ref(value: Any) -> Any:
    """Expand an explicit ``role:<name>`` value to the role's concrete model.

    Non-role values (including None/"") pass through unchanged. An unknown or
    unset role expands to ``None`` — callers fall through their existing
    default chain; nothing ever raises.
    """
    if not isinstance(value, str):
        return value
    v = value.strip()
    if not v.lower().startswith(_ROLE_PREFIX):
        return value
    name = v[len(_ROLE_PREFIX):].strip()
    if name not in ROLE_NAMES:
        logger.warning(f"unknown model role ref {value!r} — falling through")
        return None
    return configured_role_model(name) or None


def role_model_for(member_key: str) -> str | None:
    """Implicit tier: the configured model of the member's role, or None."""
    member = ROLE_MEMBERS.get(member_key)
    if not member:
        return None
    return configured_role_model(member["role"]) or None


def resolve_model_pref(*candidates: Any) -> str | None:
    """First usable concrete model among candidates: expands ``role:`` refs,
    skips empty/None/"inherit". Returns None when nothing resolves."""
    for candidate in candidates:
        expanded = expand_role_ref(candidate)
        if not expanded or not isinstance(expanded, str):
            continue
        c = expanded.strip()
        if c and c.lower() != "inherit":
            return c
    return None


def resolve_member_model(member_key: str, explicit: Any) -> str | None:
    """Resolve one member through the implicit tier.

    Chain: explicit value (``role:`` expanded) → the member's role model →
    None. Callers keep their existing default chain after None, so with roles
    unset the requested model is byte-identical to the pre-roles behavior —
    the canonical call shape is::

        model = resolve_member_model("<member>", explicit) or explicit
    """
    resolved = resolve_model_pref(explicit)
    if resolved:
        return resolved
    return role_model_for(member_key)


def effective_member_chain(member_key: str) -> dict[str, Any]:
    """Explain a member's current resolution for the API/UI preview.

    Returns {member, label, role, explicit, role_model, effective, following}
    where `following` is "explicit" | "role" | "fallback".
    """
    member = ROLE_MEMBERS[member_key]
    explicit: Any = None
    if member["kind"] == "memory":
        from .kit.agent_memory.config import get_settings
        explicit = getattr(get_settings(), member["source"], "")
    else:
        from .config import get_config_manager
        cfg = get_config_manager()
        explicit = cfg.get(member["source"], None)

    explicit_resolved = resolve_model_pref(explicit)
    role_model = role_model_for(member_key)
    if explicit_resolved:
        following = "explicit"
        effective = explicit_resolved
    elif role_model:
        following = "role"
        effective = role_model
    else:
        following = "fallback"
        effective = None
    return {
        "member": member_key,
        "label": member["label"],
        "role": member["role"],
        "kind": member["kind"],
        "source": member["source"],
        "explicit": explicit if isinstance(explicit, str) else (explicit or ""),
        "role_model": role_model or "",
        "effective": effective or "",
        "following": following,
    }
