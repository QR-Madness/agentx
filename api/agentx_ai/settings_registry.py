"""
Settings Registry — the one declaration behind every user-tunable setting.

Before this module the config write surface lived in three hand-synced places:
``views.config_update``'s per-section key tuples, ``settings_manifest``'s
``_CONFIG_WRITE_ROUTES`` mirror (carrying a "keep in lockstep" comment), and the
prose in ``OpenApi.yaml`` / ``endpoints.md``. They drifted, and the drift was
invisible: a section could claim a key was writable while the handler silently
dropped every save (the Images regression), or the reverse (``research`` and
``web_research`` were writable but reported read-only).

Now the write path, the manifest, and the generated docs all derive from
``CONFIG_SECTIONS`` below. Adding a setting means declaring it here — nothing
else needs editing, and the coverage tests fail if the two ever disagree.

**Deltas only.** Type and default come from ``DEFAULT_CONFIG`` (and, for the
memory store, from the kit's pydantic ``Settings``) at build time. A section
declares just what a generic walker cannot infer: which keys are writable, which
accept an explicit null, which dict leaves are written whole, and the
presentation axes (constraints, tier) the UI and docs render. A key that isn't
declared is read-only — visible in the manifest, never silently writable.

**Two phases.** ``plan_config_update`` is pure: it validates the whole payload
and returns the operations it *would* apply. ``apply_ops`` mutates. A payload
that fails validation halfway through therefore leaves the process-global
ConfigManager untouched, instead of half-written until the next reload.

Sections whose accept-set is computed per request (``providers`` — the catalog's
built-ins plus registered custom endpoints) or that carry a verb no other
section has (``context_limits`` — a wildcard model subtree with delete) declare
a ``planner`` instead of a key list. Those planners live here too, so the write
surface stays in one file even where it can't be a static list.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

# A planned write: ("set", dotted.path, value) or ("unset", dotted.path, None).
Op = tuple[str, str, Any]

Tier = Literal["essential", "advanced", "experimental"]

# Any key whose name smells like a credential is redacted wherever it surfaces.
# URIs redact too — connection strings embed passwords (postgres_uri). This is
# the single definition; views' redactor and the manifest both import it.
SECRET_MARKERS: tuple[str, ...] = ("key", "password", "token", "secret", "uri")

# Free-form sub-trees that are redacted wholesale regardless of leaf names —
# a custom provider's `headers` map can carry an Authorization value.
SECRET_SUBTREES: tuple[str, ...] = ("headers",)


def is_secret_path(path: str) -> bool:
    """True when a dotted config path (or bare memory key) holds a credential.

    Markers match whole words, not substrings. A plain substring test reads
    ``max_tokens`` as a credential because "token" is inside it — which redacted
    the default of every ``*_max_tokens`` setting on the platform (24 of them),
    so the settings UI and the generated reference would both print ``***``
    where a number belongs.

    The ``endswith`` arm keeps unseparated names like ``apikey`` covered: it is
    better to redact a non-secret than to leak one.
    """
    parts = path.lower().split(".")
    if any(part in SECRET_SUBTREES for part in parts):
        return True
    leaf = parts[-1]
    words = set(re.split(r"[^a-z0-9]+", leaf))
    if words & set(SECRET_MARKERS):
        return True
    return leaf.endswith(SECRET_MARKERS)


@dataclass(frozen=True)
class KeySpec:
    """Per-key deltas. Every axis is optional; an absent KeySpec means "plain
    writable key: skip nulls, write as given, no declared constraints"."""

    #: Explicit ``null`` is a real write (e.g. planner.model = "no override").
    nullable: bool = False
    #: What an empty string means here — metadata for the UI/docs, never the
    #: walker. "" is a *meaningful* value for role-following keys, so the walker
    #: must never treat it as absent.
    empty_means: str | None = None
    #: Dict leaf written atomically. A partial patch would drop sibling keys, so
    #: the manifest advertises one entry rather than per-leaf writability.
    whole_write: bool = False
    #: Applied at plan time (e.g. bool for a checkbox that may arrive as 0/1).
    coerce: Callable[[Any], Any] | None = None

    # Declared constraints. Only keys that declare them are validated on write —
    # everything else keeps today's pass-through semantics.
    min: float | None = None
    max: float | None = None
    step: float | None = None
    enum: tuple[str, ...] | None = None
    unit: str | None = None

    tier: Tier | None = None
    #: Override the name-based secret heuristic.
    secret: bool | None = None
    #: Override the section's ``ui_section``. ``None`` inherits it; ``""`` means
    #: this key has no control on any screen, so nothing should claim it does.
    ui_section: str | None = None

    def has_constraints(self) -> bool:
        return self.min is not None or self.max is not None or self.enum is not None


#: A planner takes (config, section payload) and returns the ops it would apply
#: plus per-key errors. It must not mutate anything.
PlannerFn = Callable[[Any, Any], tuple[list[Op], dict[str, str]]]


@dataclass(frozen=True)
class SectionSpec:
    """One ``DEFAULT_CONFIG`` root."""

    #: Writable keys, relative to the root, dotted for nesting
    #: ("rolling_summary.model"). ``None`` means every leaf under the root is
    #: writable (the historical no-whitelist sections).
    keys: tuple[str, ...] | None = None
    #: Bespoke sections supply a planner instead of relying on the generic walk.
    planner: PlannerFn | None = None
    #: Writable surface a bespoke section exposes to the manifest. ``ALL_SUBTREE``
    #: marks "every leaf under this root" (dynamic ids can't be enumerated).
    manifest_keys: Any = None
    #: SECTION_HIERARCHY id this root renders under, for docs grouping and the
    #: settings Overview digest.
    ui_section: str | None = None
    overrides: dict[str, KeySpec] = field(default_factory=dict)

    def spec_for(self, rel_path: str) -> KeySpec | None:
        return self.overrides.get(rel_path)


#: Sentinel for a bespoke section whose writable leaves can't be listed ahead of
#: time (dynamic provider ids, arbitrary model ids).
ALL_SUBTREE = object()

WRITE_ROUTE = "/api/config/update"


# ---------------------------------------------------------------------------
# Bespoke planners — sections a static key list cannot express.
# ---------------------------------------------------------------------------


def _plan_providers(config: Any, payload: Any) -> tuple[list[Op], dict[str, str]]:
    """Provider credentials/URLs.

    The accepted set is the catalog's — built-ins plus any registered custom
    endpoint — so it can only be known per request. Built-ins keep their
    historical ``providers.<id>.*`` path; custom entries live one level deeper
    under ``providers.custom.<id>.*``. Custom entries are created and removed via
    /api/providers/custom, never here; this only updates ids that already exist.
    """
    from .providers.catalog import BUILTIN_IDS, known_ids

    if not isinstance(payload, dict):
        return [], {}

    accepted = known_ids(config)
    ops: list[Op] = []
    for provider, provider_settings in payload.items():
        if provider not in accepted or not isinstance(provider_settings, dict):
            continue
        base = (
            f"providers.{provider}"
            if provider in BUILTIN_IDS
            else f"providers.custom.{provider}"
        )
        for key, value in provider_settings.items():
            if value is not None:
                ops.append(("set", f"{base}.{key}", value))
    return ops, {}


def _plan_context_limits(config: Any, payload: Any) -> tuple[list[Op], dict[str, str]]:
    """Per-provider and per-model context-window overrides.

    Two levels with a root whitelist and a free-form second level (arbitrary
    model ids), so it stays hand-written. Note /api/config/context-limits is the
    richer route for this section — it also supports deletion; this handler is
    the set-only path kept for the combined config payload.
    """
    if not isinstance(payload, dict):
        return [], {}

    ops: list[Op] = []
    for root, limit_settings in payload.items():
        if root not in ("lmstudio", "models") or not isinstance(limit_settings, dict):
            continue
        for key, value in limit_settings.items():
            if value is not None:
                ops.append(("set", f"context_limits.{root}.{key}", value))
    return ops, {}


def _plan_model_roles(config: Any, payload: Any) -> tuple[list[Op], dict[str, str]]:
    """Model roles — only ``models.roles.<known>`` is writable; the rest of the
    ``models`` section is read-only.

    A value must be "" (clear the role, fall back to the chain) or a concrete
    ``provider:model``. ``role:`` references are rejected so roles can't chain to
    each other. Clearing sends "" — the ``is not None`` idiom the other sections
    use would drop an explicit null, which is why "" is the clear verb here.
    """
    from .model_roles import ROLE_NAMES

    roles = (payload or {}).get("roles", {}) if isinstance(payload, dict) else {}
    if not isinstance(roles, dict):
        return [], {}

    ops: list[Op] = []
    errors: dict[str, str] = {}
    for key, value in roles.items():
        if key not in ROLE_NAMES or value is None:
            continue
        value = str(value).strip()
        if value and (":" not in value or value.lower().startswith("role:")):
            errors[f"models.roles.{key}"] = (
                f'must be "" or a concrete provider:model (got {value!r})'
            )
            continue
        ops.append(("set", f"models.roles.{key}", value))
    return ops, errors


_AIDE_KEYS = (
    "enabled", "model", "temperature", "max_tokens", "max_input_chars",
    "max_parallel", "timeout_seconds", "max_per_survey", "cache_ttl_seconds",
)
_AMBASSADOR_FLAT_KEYS = (
    "enabled", "profile_id", "model", "max_context_turns", "max_tokens",
    "speech_model", "voice", "transcription_model",
)
#: These fall back to "the default profile / the model floor" when explicitly
#: null, so null is a real write rather than "leave unchanged".
_AMBASSADOR_NULLABLE = ("profile_id", "model", "speech_model", "voice", "transcription_model")


def _plan_ambassador(config: Any, payload: Any) -> tuple[list[Op], dict[str, str]]:
    """Ambassador settings.

    Two nested groups merge rather than replace, so editing one sub-key never
    wipes its siblings (absent sub-keys fall back to defaults at read time).
    """
    if not isinstance(payload, dict):
        return [], {}

    ops: list[Op] = []
    for key, value in payload.items():
        if key == "aide":
            if isinstance(value, dict):
                for sub, sub_val in value.items():
                    if sub in _AIDE_KEYS and sub_val is not None:
                        ops.append(("set", f"ambassador.aide.{sub}", sub_val))
            continue
        if key == "dispatch":
            if isinstance(value, dict):
                for sub, sub_val in value.items():
                    if sub == "enabled" and sub_val is not None:
                        ops.append(("set", "ambassador.dispatch.enabled", bool(sub_val)))
            continue
        if key not in _AMBASSADOR_FLAT_KEYS:
            continue
        if value is None and key not in _AMBASSADOR_NULLABLE:
            continue
        ops.append(("set", f"ambassador.{key}", value))
    return ops, {}


_AMBASSADOR_MANIFEST_KEYS = (
    *_AMBASSADOR_FLAT_KEYS,
    *(f"aide.{k}" for k in _AIDE_KEYS),
    "dispatch.enabled",
)


# ---------------------------------------------------------------------------
# The declaration.
#
# Order matters: it is the order writes are applied and reported in `updated`.
# ---------------------------------------------------------------------------

CONFIG_SECTIONS: dict[str, SectionSpec] = {
    "providers": SectionSpec(
        planner=_plan_providers,
        manifest_keys=ALL_SUBTREE,
        ui_section="providers",
    ),
    "preferences": SectionSpec(keys=None),
    "llm_settings": SectionSpec(keys=None, ui_section="models"),
    "context_limits": SectionSpec(
        planner=_plan_context_limits,
        manifest_keys=ALL_SUBTREE,
        ui_section="models",
    ),
    "context": SectionSpec(
        keys=(
            "summary_trigger_ratio", "verbatim_budget_ratio", "recent_floor",
            "preassembly_summary_enabled", "conversation_state_enabled",
            "conversation_state_compaction_enabled", "rehydrate_max_turns",
            "max_input_tokens",
        ),
        ui_section="context",
        overrides={
            "verbatim_budget_ratio": KeySpec(min=0.5, max=0.98, step=0.01,
                                             unit="of the context window",
                                             tier="essential"),
            "summary_trigger_ratio": KeySpec(min=0.5, max=0.98, step=0.01,
                                             unit="of the history budget",
                                             tier="essential"),
            "recent_floor": KeySpec(min=1, max=50, step=1, unit="turns",
                                    tier="essential"),
            "conversation_state_enabled": KeySpec(tier="essential"),
            "preassembly_summary_enabled": KeySpec(tier="advanced"),
            "conversation_state_compaction_enabled": KeySpec(tier="advanced"),
            "rehydrate_max_turns": KeySpec(min=20, max=2000, step=1, unit="turns",
                                           tier="advanced"),
            "max_input_tokens": KeySpec(min=0, max=1_000_000, step=1, unit="tokens",
                                        tier="advanced"),
        },
    ),
    "session": SectionSpec(
        keys=("rolling_summary.enabled", "rolling_summary.model", "rolling_summary.max_tokens"),
        ui_section="context",
        overrides={
            "rolling_summary.enabled": KeySpec(tier="essential"),
            "rolling_summary.model": KeySpec(empty_means="follow_role"),
            "rolling_summary.max_tokens": KeySpec(min=200, max=4000, step=1,
                                                  unit="tokens", tier="advanced"),
        },
    ),
    "trajectory_compression": SectionSpec(
        keys=("enabled", "threshold_ratio", "preserve_recent_rounds", "model",
              "max_knowledge_chars"),
        ui_section="context",
        overrides={
            "model": KeySpec(empty_means="follow_role"),
            "threshold_ratio": KeySpec(min=0.5, max=0.95, step=0.05,
                                       unit="of the turn budget", tier="advanced"),
            "preserve_recent_rounds": KeySpec(min=1, max=5, step=1, unit="rounds",
                                              tier="advanced"),
            "max_knowledge_chars": KeySpec(min=500, max=10_000, step=1,
                                           unit="characters", tier="advanced"),
        },
    ),
    "compression": SectionSpec(
        keys=("enabled", "model", "max_summary_chars"),
        ui_section="context",
        overrides={
            "model": KeySpec(empty_means="follow_role"),
            "max_summary_chars": KeySpec(min=500, max=10_000, step=1,
                                         unit="characters", tier="advanced"),
        },
    ),
    "memory": SectionSpec(
        keys=("episodic_leads_enabled", "project_channels"),
        ui_section="context",
        overrides={
            "episodic_leads_enabled": KeySpec(tier="advanced"),
            # Projects/workspace channel scoping — writable, but no screen shows
            # it. It inherited "context" from this root and claimed a home on the
            # Conversation Context page that has never rendered it.
            "project_channels": KeySpec(ui_section=""),
        },
    ),
    "reasoning": SectionSpec(
        keys=(
            "chat_patterns_enabled", "auto_classifier_enabled", "classifier_model",
            "classifier_min_chars", "step_back_model", "step_back_timeout_seconds",
            "cot_enabled", "step_back_enabled", "reflection_enabled",
            "self_consistency_enabled", "sc_model", "sc_k", "min_output_tokens",
        ),
        ui_section="thinking",
        overrides={
            # --- Availability: what Auto and explicit selection may pick ---
            "chat_patterns_enabled": KeySpec(tier="essential"),
            "cot_enabled": KeySpec(tier="essential"),
            "step_back_enabled": KeySpec(tier="essential"),
            "reflection_enabled": KeySpec(tier="essential"),
            "self_consistency_enabled": KeySpec(tier="essential"),
            "auto_classifier_enabled": KeySpec(tier="essential"),
            # --- Per-pattern models and budgets ---
            # Only the classifier follows a role. The other two fall back to the
            # *conversation's own* model — which is what the code does and what
            # the UI has always said, but not what this declared until now.
            "classifier_model": KeySpec(empty_means="follow_role", tier="advanced"),
            "step_back_model": KeySpec(empty_means="active_turn_model", tier="advanced"),
            "sc_model": KeySpec(empty_means="active_turn_model", tier="advanced"),
            "classifier_min_chars": KeySpec(min=0, max=5000, step=1,
                                            unit="characters", tier="advanced"),
            "step_back_timeout_seconds": KeySpec(min=5, max=120, step=1,
                                                 unit="seconds", tier="advanced"),
            "sc_k": KeySpec(min=2, max=5, step=1, unit="samples", tier="advanced"),
            "min_output_tokens": KeySpec(min=0, max=65536, step=1,
                                         unit="tokens", tier="advanced"),
        },
    ),
    "prompt_enhancement": SectionSpec(
        keys=None,
        ui_section="prompts",
        # The prompt body is edited on Feature Prompts alongside the other
        # feature prompts, not on the Prompt Enhancement screen.
        overrides={"system_prompt": KeySpec(ui_section="feature-prompts")},
    ),
    "planner": SectionSpec(
        keys=None,
        ui_section="planner",
        overrides={
            # Explicit null clears the override and falls back to the default model.
            "model": KeySpec(nullable=True, tier="advanced"),
            "enabled": KeySpec(tier="essential"),
            "complexity_threshold": KeySpec(
                enum=("simple", "moderate", "complex"), tier="essential"),
            "max_subtasks": KeySpec(min=1, max=20, step=1,
                                    unit="subtasks", tier="essential"),
            "temperature": KeySpec(min=0, max=1, step=0.1, tier="advanced"),
            "max_tokens": KeySpec(min=100, max=4000, step=1,
                                  unit="tokens", tier="advanced"),
            # Edited on Feature Prompts with the other feature prompts — the
            # Task Planner screen says so in as many words and doesn't show it.
            "prompt_override": KeySpec(empty_means="built_in_prompt", tier="advanced",
                                       ui_section="feature-prompts"),
        },
    ),
    "search": SectionSpec(
        keys=(
            "backend", "fallback_enabled", "max_results",
            "cache_ttl_seconds", "timeout", "tavily_api_key", "brave_api_key",
            # Per-turn budgets: call counts + the dollar envelopes.
            "per_turn_limit", "research_per_turn_limit",
            "per_turn_cost_usd", "research_per_turn_cost_usd",
            # Search defaults the operator owns.
            "default_search_depth", "default_chunks_per_source", "safesearch",
            "country", "search_lang",
            # Brave grounding + deep research.
            "brave_grounding_default", "brave_context_max_tokens",
            "brave_context_max_tokens_per_url", "brave_context_threshold",
            "brave_answers_enabled",
            "source_policy",
        ),
        ui_section="search",
        overrides={
            # --- Backend ---
            "backend": KeySpec(enum=("tavily", "brave"), tier="essential"),
            "tavily_api_key": KeySpec(tier="essential"),
            "brave_api_key": KeySpec(tier="essential"),
            "fallback_enabled": KeySpec(tier="advanced"),
            # --- Budgets: the operator's cost levers ---
            "per_turn_limit": KeySpec(min=0, max=100, step=1,
                                      unit="searches per turn", tier="essential"),
            "per_turn_cost_usd": KeySpec(min=0, max=100, step=0.05,
                                         unit="USD per turn", tier="essential"),
            "research_per_turn_cost_usd": KeySpec(min=0, max=100, step=0.05,
                                                  unit="USD per turn", tier="advanced"),
            # Lives on the Research Mode screen with the rest of that budget,
            # not on Web Search — it is a `search.*` key by storage only.
            "research_per_turn_limit": KeySpec(min=0, max=200, step=1,
                                               unit="searches per turn",
                                               tier="essential", ui_section="research"),
            "timeout": KeySpec(min=5, max=120, step=1, unit="seconds", tier="advanced"),
            "cache_ttl_seconds": KeySpec(min=0, max=3600, step=1,
                                         unit="seconds", tier="advanced"),
            # --- Defaults the operator owns ---
            "max_results": KeySpec(min=1, max=20, step=1,
                                   unit="results per search", tier="essential"),
            "default_search_depth": KeySpec(
                empty_means="provider_default", tier="advanced",
                enum=("ultra-fast", "fast", "basic", "advanced")),
            "default_chunks_per_source": KeySpec(min=0, max=5, step=1, tier="advanced"),
            "safesearch": KeySpec(empty_means="provider_default", tier="advanced",
                                  enum=("off", "moderate", "strict")),
            "country": KeySpec(empty_means="provider_default", tier="advanced"),
            "search_lang": KeySpec(empty_means="provider_default", tier="advanced"),
            # --- Brave grounding ---
            "brave_grounding_default": KeySpec(tier="advanced"),
            "brave_context_max_tokens": KeySpec(min=1024, max=32768, step=1,
                                                unit="tokens", tier="advanced"),
            "brave_context_max_tokens_per_url": KeySpec(min=128, max=8192, step=1,
                                                        unit="tokens", tier="advanced"),
            "brave_context_threshold": KeySpec(
                enum=("strict", "balanced", "lenient"), tier="advanced"),
            "brave_answers_enabled": KeySpec(tier="advanced"),
            # Written as one dict — a per-leaf patch would drop the siblings.
            "source_policy": KeySpec(whole_write=True, tier="essential"),
        },
    ),
    "research": SectionSpec(
        keys=("enabled", "max_tool_rounds", "default_depth", "min_max_tokens"),
        ui_section="research",
        overrides={
            "enabled": KeySpec(tier="essential"),
            "default_depth": KeySpec(enum=("mini", "auto", "pro"), tier="essential"),
            "max_tool_rounds": KeySpec(min=10, max=80, step=1,
                                       unit="tool rounds", tier="advanced"),
            "min_max_tokens": KeySpec(min=1024, max=200_000, step=1,
                                      unit="tokens", tier="advanced"),
        },
    ),
    "web_research": SectionSpec(
        keys=("enabled", "cache_ttl_seconds", "budget_weight",
              "poll_timeout_seconds", "poll_interval_seconds"),
        ui_section="research",
        overrides={
            "enabled": KeySpec(tier="essential"),
            "budget_weight": KeySpec(min=1, max=20, step=1,
                                     unit="budget units per call", tier="advanced"),
            "cache_ttl_seconds": KeySpec(min=0, max=86400, step=1,
                                         unit="seconds", tier="advanced"),
            "poll_timeout_seconds": KeySpec(min=30, max=900, step=1,
                                            unit="seconds", tier="advanced"),
            "poll_interval_seconds": KeySpec(min=1, max=60, step=1,
                                             unit="seconds", tier="advanced"),
        },
    ),
    "alloy": SectionSpec(
        keys=("allow_adhoc_delegation", "max_parallel_delegations",
              "max_delegation_depth", "delegation_timeout_seconds",
              "non_blocking_delegations", "chain_of_command"),
        ui_section="alloy",
        overrides={
            "allow_adhoc_delegation": KeySpec(tier="essential"),
            "chain_of_command": KeySpec(tier="essential"),
            "max_delegation_depth": KeySpec(min=1, max=5, step=1,
                                            unit="levels deep", tier="essential"),
            "max_parallel_delegations": KeySpec(min=1, max=8, step=1,
                                                unit="at once", tier="advanced"),
            "delegation_timeout_seconds": KeySpec(min=30, max=3600, step=1,
                                                  unit="seconds", tier="advanced"),
            "non_blocking_delegations": KeySpec(tier="advanced"),
        },
    ),
    "ambassador": SectionSpec(
        planner=_plan_ambassador,
        manifest_keys=_AMBASSADOR_MANIFEST_KEYS,
        ui_section="ambassador",
        overrides={
            "model": KeySpec(nullable=True),
            "profile_id": KeySpec(nullable=True),
            "speech_model": KeySpec(nullable=True),
            "voice": KeySpec(nullable=True),
            "transcription_model": KeySpec(nullable=True),
            "aide.model": KeySpec(empty_means="follow_role"),
        },
    ),
    "images": SectionSpec(
        keys=("enabled", "default_model", "avatar_model", "avatar_style_prompt"),
        ui_section="images",
    ),
    "vision": SectionSpec(
        keys=("enabled", "refeed_recent_turns"),
        ui_section="images",
    ),
    "models": SectionSpec(
        planner=_plan_model_roles,
        manifest_keys=None,  # filled below from ROLE_NAMES
        ui_section="model-roles",
    ),
}


def _models_manifest_keys() -> tuple[str, ...]:
    from .model_roles import ROLE_NAMES

    return tuple(f"roles.{name}" for name in ROLE_NAMES)


# ---------------------------------------------------------------------------
# Memory store — a sidecar, not a rewrite.
#
# The kit's pydantic `Settings` stays the type/default/validation truth. Adding
# Field(ge=…, le=…) to 150 fields would change validation behaviour for keys
# that have always been unconstrained, so the presentation axes live here and
# apply only where declared. A test pins every key below to a real Settings
# field so a rename can't rot this quietly.
# ---------------------------------------------------------------------------

MEMORY_KEY_SPECS: dict[str, KeySpec] = {
    # -- Retrieval techniques (the five recall strategies) -------------------
    "recall_enable_hybrid": KeySpec(tier="essential"),
    "recall_enable_entity_centric": KeySpec(tier="essential"),
    "recall_enable_query_expansion": KeySpec(tier="essential"),
    "recall_enable_hyde": KeySpec(tier="essential"),
    "recall_enable_self_query": KeySpec(tier="essential"),

    # -- Two-stage rerank ----------------------------------------------------
    "cross_encoder_enabled": KeySpec(tier="essential"),
    "cross_encoder_model": KeySpec(tier="essential"),
    "recall_candidate_pool": KeySpec(min=10, max=200, step=1, unit="candidates",
                                     tier="essential"),
    "recall_ce_max_demotion": KeySpec(min=0, max=20, step=1, unit="places",
                                      tier="essential"),

    # -- Hybrid search weights ----------------------------------------------
    "recall_hybrid_bm25_weight": KeySpec(min=0, max=1, step=0.1, tier="essential"),
    "recall_hybrid_vector_weight": KeySpec(min=0, max=1, step=0.1, tier="essential"),

    # -- Entity-centric ------------------------------------------------------
    "recall_entity_similarity_threshold": KeySpec(min=0.3, max=0.95, step=0.05,
                                                  tier="essential"),
    "recall_entity_max_entities": KeySpec(min=1, max=20, step=1, unit="entities",
                                          tier="essential"),

    # -- Query expansion -----------------------------------------------------
    "recall_expansion_max_variants": KeySpec(min=1, max=10, step=1, unit="variants",
                                             tier="essential"),

    # -- HyDE ----------------------------------------------------------------
    "recall_hyde_model": KeySpec(empty_means="follow_role", tier="essential"),
    "recall_hyde_temperature": KeySpec(min=0, max=1, step=0.1, tier="essential"),
    "recall_hyde_max_tokens": KeySpec(min=50, max=2000, step=10, unit="tokens",
                                      tier="essential"),

    # -- Self-query ----------------------------------------------------------
    "recall_self_query_model": KeySpec(empty_means="follow_role", tier="essential"),
    "recall_self_query_temperature": KeySpec(min=0, max=1, step=0.05, tier="essential"),
    "recall_self_query_max_tokens": KeySpec(min=50, max=2000, step=10, unit="tokens",
                                            tier="essential"),

    # -- Cross-technique tuning ---------------------------------------------
    "recall_min_confidence": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "recall_hybrid_rrf_k": KeySpec(min=1, max=200, step=1, tier="advanced"),
    "recall_entity_graph_depth": KeySpec(min=1, max=5, step=1, unit="hops",
                                         tier="advanced"),

    # -- Experimental --------------------------------------------------------
    "recall_first_person_guard": KeySpec(tier="experimental"),
    "recall_first_person_penalty": KeySpec(min=0, max=1, step=0.05, tier="experimental"),

    # ========================================================================
    # Consolidation — the background pipeline that turns finished turns into
    # facts, entities and procedures. Stage models take "inherit", which walks
    # a chain rather than meaning one thing, so none of them declare bounds.
    # ========================================================================

    # -- The bulk default every stage falls back to --------------------------
    "feature_default_model": KeySpec(empty_means="follow_default_model",
                                     tier="essential"),

    # -- Extraction ----------------------------------------------------------
    "extraction_enabled": KeySpec(tier="essential"),
    "extraction_model": KeySpec(empty_means="inherit", tier="essential"),
    "extraction_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "extraction_max_tokens": KeySpec(min=200, max=8000, step=50,
                                     unit="tokens", tier="advanced"),
    "extraction_condense_facts": KeySpec(tier="advanced"),
    # Edited on Feature Prompts with the other feature prompts, not here.
    "extraction_system_prompt": KeySpec(empty_means="built_in_prompt",
                                        tier="advanced",
                                        ui_section="feature-prompts"),

    # -- Relevance filter ----------------------------------------------------
    "relevance_filter_enabled": KeySpec(tier="essential"),
    "relevance_filter_model": KeySpec(empty_means="inherit", tier="advanced"),
    "relevance_filter_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "relevance_filter_max_tokens": KeySpec(min=100, max=4000, step=50,
                                           unit="tokens", tier="advanced"),
    "relevance_filter_prompt": KeySpec(empty_means="built_in_prompt",
                                       tier="advanced",
                                       ui_section="feature-prompts"),

    # -- Contradiction detection ---------------------------------------------
    "contradiction_detection_enabled": KeySpec(tier="essential"),
    "contradiction_model": KeySpec(empty_means="inherit", tier="advanced"),
    "contradiction_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "contradiction_max_tokens": KeySpec(min=100, max=4000, step=50,
                                        unit="tokens", tier="advanced"),
    "semantic_duplicate_threshold": KeySpec(min=0.5, max=1.0, step=0.01,
                                            unit="cosine similarity",
                                            tier="advanced"),
    "contradiction_similarity_threshold": KeySpec(min=0.1, max=1.0, step=0.05,
                                                  unit="cosine similarity",
                                                  tier="advanced"),
    "contradiction_max_candidates": KeySpec(min=1, max=50, step=1,
                                            unit="candidates", tier="advanced"),

    # -- Fact → entity linking at write time ---------------------------------
    "link_autocreate_stub_entities": KeySpec(tier="advanced"),

    # -- Correction handling -------------------------------------------------
    "correction_detection_enabled": KeySpec(tier="essential"),
    "correction_model": KeySpec(empty_means="inherit", tier="advanced"),
    "correction_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "correction_max_tokens": KeySpec(min=100, max=4000, step=50,
                                     unit="tokens", tier="advanced"),

    # -- Combined relevance + extraction -------------------------------------
    "combined_extraction_model": KeySpec(empty_means="inherit", tier="essential"),
    "combined_extraction_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "combined_extraction_max_tokens": KeySpec(min=200, max=8000, step=50,
                                              unit="tokens", tier="advanced"),

    # -- Procedural distillation ---------------------------------------------
    "procedural_distill_model": KeySpec(empty_means="inherit", tier="advanced"),
    "procedural_distill_temperature": KeySpec(min=0, max=1, step=0.05, tier="advanced"),
    "procedural_distill_max_tokens": KeySpec(min=200, max=8000, step=50,
                                             unit="tokens", tier="advanced"),
    "procedural_distill_timeout_s": KeySpec(min=10, max=600, step=1,
                                            unit="seconds", tier="advanced"),
    "procedural_dedupe_threshold": KeySpec(min=0.5, max=1.0, step=0.01,
                                           unit="cosine similarity", tier="advanced"),
    "procedural_distill_batch_limit": KeySpec(min=10, max=1000, step=10,
                                              unit="candidates per run",
                                              tier="advanced"),

    # -- Always-on cores -----------------------------------------------------
    "reflex_core_enabled": KeySpec(tier="essential"),
    "reflex_core_limit": KeySpec(min=0, max=20, step=1, unit="procedures",
                                 tier="advanced"),
    "salient_core_enabled": KeySpec(tier="essential"),
    # Capped at 10 to match the render cap in `to_context_string`.
    "salient_core_limit": KeySpec(min=0, max=10, step=1, unit="facts",
                                  tier="advanced"),
    "salient_core_min_salience": KeySpec(min=0, max=1, step=0.05, tier="advanced"),

    # -- Entity linking backfill ---------------------------------------------
    "entity_linking_enabled": KeySpec(tier="essential"),
    "entity_linking_similarity_threshold": KeySpec(min=0.3, max=1.0, step=0.01,
                                                   unit="cosine similarity",
                                                   tier="advanced"),
    "entity_linking_use_llm_disambiguation": KeySpec(tier="advanced"),
    "entity_linking_model": KeySpec(empty_means="inherit", tier="advanced"),
    "entity_linking_max_facts": KeySpec(min=100, max=100_000, step=100,
                                        unit="facts per run", tier="advanced"),
    "entity_linking_max_ngram": KeySpec(min=1, max=8, step=1, unit="words",
                                        tier="advanced"),

    # -- Quality thresholds --------------------------------------------------
    "fact_confidence_threshold": KeySpec(min=0, max=1, step=0.05, tier="essential"),
    "promotion_min_confidence": KeySpec(min=0, max=1, step=0.05, tier="advanced"),

    # -- Job intervals (minutes) ---------------------------------------------
    "job_consolidate_interval": KeySpec(min=1, max=1440, step=1, unit="minutes",
                                        tier="essential"),
    "job_promote_interval": KeySpec(min=1, max=1440, step=1, unit="minutes",
                                    tier="advanced"),
    "job_entity_linking_interval": KeySpec(min=1, max=1440, step=1, unit="minutes",
                                           tier="advanced"),
    "job_distill_procedures_interval": KeySpec(min=1, max=1440, step=1,
                                               unit="minutes", tier="advanced"),
}


def memory_key_spec(key: str) -> KeySpec | None:
    return MEMORY_KEY_SPECS.get(key)


# ---------------------------------------------------------------------------
# Derivation helpers — the manifest and the docs generator read these.
# ---------------------------------------------------------------------------


def _section_and_rel(path: str) -> tuple[str, str]:
    root, _, rest = path.partition(".")
    return root, rest


def config_writable_via(path: str) -> str | None:
    """The route that can write this dotted config path, or None if read-only."""
    root, rest = _section_and_rel(path)
    section = CONFIG_SECTIONS.get(root)
    if section is None or not rest:
        return None

    if section.planner is not None:
        keys = section.manifest_keys
        if keys is None and root == "models":
            keys = _models_manifest_keys()
        if keys is ALL_SUBTREE:
            return WRITE_ROUTE
        if isinstance(keys, tuple):
            # Match the full relative path, or a parent of it for whole-written
            # dicts (ambassador.aide.* is declared leaf-wise).
            if rest in keys or any(rest.startswith(f"{k}.") for k in keys):
                return WRITE_ROUTE
        return None

    if section.keys is None:
        return WRITE_ROUTE
    if rest in section.keys:
        return WRITE_ROUTE
    # A leaf under a declared whole-written dict is reachable only via its
    # parent, so it is not independently writable.
    return None


def config_key_spec(path: str) -> KeySpec | None:
    root, rest = _section_and_rel(path)
    section = CONFIG_SECTIONS.get(root)
    if section is None or not rest:
        return None
    return section.spec_for(rest)


def config_ui_section(path: str) -> str | None:
    root, rel = _section_and_rel(path)
    section = CONFIG_SECTIONS.get(root)
    if not section:
        return None
    spec = section.spec_for(rel)
    if spec is not None and spec.ui_section is not None:
        # "" = declared as having no screen; distinct from inheriting the root's.
        return spec.ui_section or None
    return section.ui_section


#: Where each ``ui_section`` id sits in the settings screen, for prose that has
#: to name it. The **id** is the contract (the client's SECTION_HIERARCHY keys
#: off it); these labels exist so the generated reference can say
#: "Memory → Recall" instead of printing a slug at the reader.
UI_SECTION_LABELS: dict[str, str] = {
    "providers": "Infrastructure → Model Providers",
    "models": "Infrastructure → Model Limits",
    "model-roles": "Infrastructure → Model Roles",
    "search": "Infrastructure → Web Search",
    "images": "Infrastructure → Images & Audio",
    "planner": "Intelligence → Task Planner",
    "thinking": "Intelligence → Thinking Patterns",
    "alloy": "Intelligence → Agent Teams",
    "ambassador": "Intelligence → Ambassador",
    "research": "Intelligence → Research Mode",
    "prompts": "Prompts → Prompt Enhancement",
    # Owns no config *root* of its own — it edits the prompt bodies belonging to
    # other features (planner decomposition, prompt enhancement, and two memory
    # prompts), which is why they declare their way here rather than inherit.
    "feature-prompts": "Prompts → Feature Prompts",
    "context": "Memory → Conversation Context",
    "memory-recall": "Memory → Recall",
    "memory-consolidation": "Memory → Consolidation",
}

#: Settings screens that own no key in either store — they configure their own
#: subsystem through a dedicated endpoint (prompt layers, the template library,
#: themes) or only report state. They appear here so section blurbs can cover
#: the whole nav without the manifest pretending these screens hold settings.
UI_ONLY_SECTIONS: dict[str, str] = {
    "overview": "Overview",
    "prompt-stack": "Prompts → System Prompt",
    "prompt-templates": "Prompts → Template Library",
    "memory-overview": "Memory → Overview",
    "translation": "Tools → Translation",
    "appearance": "Interface → Appearance",
}

#: Every settings screen, keyed by id. Asserted against the client's
#: SECTION_HIERARCHY in both directions, so a screen cannot be added on either
#: side without the other noticing — which is what keeps every screen blurbed.
ALL_SECTIONS: dict[str, str] = {**UI_SECTION_LABELS, **UI_ONLY_SECTIONS}


def ui_section_label(section_id: str | None) -> str:
    """Human-readable home for a ui_section id."""
    if not section_id:
        return "Not shown in Settings"
    return ALL_SECTIONS.get(section_id, section_id)


def whole_write_paths() -> tuple[str, ...]:
    """Dotted paths written as one dict — the manifest emits a single entry for
    each rather than advertising its leaves as independently writable."""
    out: list[str] = []
    for root, section in CONFIG_SECTIONS.items():
        for rel, spec in section.overrides.items():
            if spec.whole_write:
                out.append(f"{root}.{rel}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Validation + the two-phase write.
# ---------------------------------------------------------------------------


def _check_constraints(label: str, spec: KeySpec | None, value: Any) -> str | None:
    """Validate a single declared value. Undeclared keys always pass — this is
    what keeps existing configurations frictionless."""
    if spec is None or value is None or not spec.has_constraints():
        return None

    if spec.enum is not None:
        if not isinstance(value, str) or (value not in spec.enum and value != ""):
            allowed = ", ".join(spec.enum)
            return f"must be one of: {allowed} (got {value!r})"
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"must be a number (got {value!r})"
    if spec.min is not None and value < spec.min:
        return f"must be at least {spec.min} (got {value})"
    if spec.max is not None and value > spec.max:
        return f"must be at most {spec.max} (got {value})"
    return None


def check_declared_constraints(store: str, payload: dict[str, Any]) -> dict[str, str]:
    """Validate a flat memory-store payload against declared constraints.

    Config-store payloads are validated inside ``plan_config_update``; this is
    the entry point the two /api/memory/* views use, where keys are flat.
    """
    if store != "memory":
        return {}
    errors: dict[str, str] = {}
    for key, value in payload.items():
        message = _check_constraints(key, MEMORY_KEY_SPECS.get(key), value)
        if message:
            errors[key] = message
    return errors


def plan_config_update(config: Any, data: dict[str, Any]) -> tuple[list[Op], dict[str, str]]:
    """Work out every write a config payload implies, without applying any.

    Returns ``(ops, errors)``. When ``errors`` is non-empty the caller must
    reject the whole request — nothing has been mutated, so the in-memory
    ConfigManager is still consistent with what is on disk.
    """
    ops: list[Op] = []
    errors: dict[str, str] = {}

    for root, section in CONFIG_SECTIONS.items():
        if root not in data:
            continue
        payload = data.get(root)

        if section.planner is not None:
            section_ops, section_errors = section.planner(config, payload)
            ops.extend(section_ops)
            errors.update(section_errors)
            continue

        if not isinstance(payload, dict):
            continue

        for rel_path, value in _walk_section(payload, section):
            spec = section.spec_for(rel_path)
            if value is None and not (spec and spec.nullable):
                continue
            if spec and spec.coerce is not None:
                value = spec.coerce(value)
            message = _check_constraints(rel_path, spec, value)
            if message:
                errors[f"{root}.{rel_path}"] = message
                continue
            ops.append(("set", f"{root}.{rel_path}", value))

    return ops, errors


def _walk_section(payload: dict[str, Any], section: SectionSpec):
    """Yield ``(relative_path, value)`` for the writable keys in a payload.

    Nesting is driven by the declaration: a key declared as ``a.b`` is looked up
    one level down. Undeclared keys are skipped; sections declaring ``keys=None``
    accept every top-level key, matching their historical behaviour.
    """
    if section.keys is None:
        for key, value in payload.items():
            yield key, value
        return

    nested_roots = {k.split(".", 1)[0] for k in section.keys if "." in k}

    for key, value in payload.items():
        if key in nested_roots and isinstance(value, dict):
            for sub, sub_value in value.items():
                rel = f"{key}.{sub}"
                if rel in section.keys:
                    yield rel, sub_value
            continue
        if key in section.keys:
            yield key, value


def apply_ops(config: Any, ops: list[Op]) -> list[str]:
    """Apply planned operations in order. Returns the touched paths, with
    deletions prefixed ``-`` so a caller can tell them apart."""
    updated: list[str] = []
    for verb, path, value in ops:
        if verb == "unset":
            if config.unset(path):
                updated.append(f"-{path}")
            continue
        config.set(path, value)
        updated.append(path)
    return updated
