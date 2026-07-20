"""
Configuration manager for runtime settings.

Provides a centralized configuration system that:
- Persists to data/config.json
- Supports hot-reload without server restart
- Falls back to environment variables when config values are not set
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# Image-generation defaults. Module constants (not just DEFAULT_CONFIG keys) so callers can
# fall back to them on installs whose pre-existing config.json predates the `images` block —
# `_load` does NOT merge new defaults into an existing file.
DEFAULT_IMAGE_MODEL = "openrouter:black-forest-labs/flux.2-klein-4b"
# Avatars get a dedicated model (portrait quality matters more than speed here);
# blank the setting to fall through to `images.default_model`.
DEFAULT_AVATAR_MODEL = "openrouter:microsoft/mai-image-2.5"
# Template, not a prefix: `<SUBJECT>` is substituted at generation time
# (views._render_avatar_prompt); an empty subject routes into the
# invent-a-synthetic-face branch. Overrides without the marker degrade to the
# old append behavior.
DEFAULT_AVATAR_STYLE_PROMPT = """\
Create one square avatar portrait.

COMPOSITION — a single subject, centered, head-and-shoulders framing with a slight
three-quarter turn toward the viewer; the subject fills about 70% of the frame; keep
everything important inside a centered circular safe zone — the image will be cropped
to a circle, so nothing essential may sit near the corners or edges; corners and edges
fade smoothly into the background.

BACKGROUND — a deep dark-cosmos field: near-black space, faint desaturated nebula haze,
sparse pinpoint stars; keep it low-contrast and neutral so it never competes with the
subject; no planets, no lens flares, no light beams.

SUBJECT — <SUBJECT>
If no subject is given, invent a friendly synthetic being: a robot or android face with
real character — vary the silhouette between generations (visor, antennae, plating,
glowing eyes, sculpted chrome), never the same design twice.

STYLE — polished digital illustration with soft 3D shading; rim light from the upper
left in cool violet, one warm gold key accent; crisp focus on the face, shallow depth
behind it; a consistent series look, as if all avatars in a team were drawn by the
same artist.

RULES — exactly one subject; no text, letters, numbers, logos, watermarks, frames, or
UI elements; no photorealistic human faces; the dark background must reach every edge.
"""

# Default configuration structure
DEFAULT_CONFIG = {
    "providers": {
        "lmstudio": {
            "base_url": None,
            "timeout": 300,
        },
        "anthropic": {
            "api_key": None,
            "base_url": None,
        },
        "openai": {
            "api_key": None,
            "base_url": None,
        },
        "vercel": {
            "api_key": None,
            "base_url": None,
        },
        "openrouter": {
            "api_key": None,
            "site_url": None,
            "app_name": None,
        },
    },
    "models": {
        "defaults": {
            "chat": None,
            "reasoning": None,
            "extraction": None,
        },
        "overrides": {},
        # Model roles (settings overhaul D1) — one model per workload cluster;
        # members with an empty/"inherit" value follow their role (implicit
        # tier, see agentx_ai/model_roles.py). "" = role unset = tier is a
        # no-op. Read with literal "" defaults so old installs need no
        # migration (`_load` does NOT merge new defaults).
        "roles": {
            "fast_utility": "",
            "deep_reasoning": "",
            "summarizer": "",
        },
        # When a feature's configured model is unavailable (provider unconfigured
        # or unreachable), fall back to the active agent model / default model
        # instead of hard-failing the turn. Kill-switch for the universal fallback.
        "fallback_enabled": True,
    },
    "llm_settings": {
        "default_temperature": 0.7,
        "default_max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "context_limits": {
        # Provider-level defaults — only for local providers (LM Studio)
        # API providers (Anthropic, OpenAI, OpenRouter) use their own per-model capabilities
        "lmstudio": {
            "context_window": 32768,  # Conservative default for local models
            "max_output_tokens": 8192,
        },
        # Model-specific overrides (escape hatch for any provider)
        "models": {
            # Example: "claude-3-opus-20240229": {"context_window": 1000000, "max_output_tokens": 32000}
        },
    },
    "compression": {
        "enabled": True,
        "model": "",  # empty ⇒ summarizer role (see model_roles.py); reader floors to haiku
        "temperature": 0.2,
        "max_tokens": 1000,
        "max_summary_chars": 2000,
    },
    "trajectory_compression": {
        "enabled": True,
        "threshold_ratio": 0.75,       # Compress when context > 75% of limit
        "preserve_recent_rounds": 2,   # Keep last N tool-call rounds intact
        "model": "",  # empty ⇒ summarizer role (see model_roles.py); reader floors
        "temperature": 0.2,
        "max_tokens": 1500,
        "max_knowledge_chars": 3000,
    },
    "prompt_enhancement": {
        "enabled": True,
        "model": "",  # empty ⇒ summarizer role (see model_roles.py); reader floors to haiku
        "temperature": 0.7,
        "max_tokens": 1000,
        "system_prompt": "",  # Empty = use default hardcoded prompt
    },
    "planner": {
        "enabled": True,
        "model": None,                       # None → falls back to agent default model
        "temperature": 0.3,
        "max_tokens": 1000,
        "prompt_override": "",               # Empty = use planner.decompose from system_prompts.yaml
        "complexity_threshold": "complex",   # "simple" | "moderate" | "complex"
        "max_subtasks": 6,                   # Hard cap on decomposed subtasks
    },
    "reasoning": {
        # Thinking Patterns — reasoning compiled into the streaming chat turn
        # (reasoning/chat_patterns.py). Master kill-switch restores pre-feature
        # behavior in one flip.
        "chat_patterns_enabled": True,
        # Auto mode: LLM tiebreak when the keyword heuristics are unconfident
        # (fast_utility role, ≤150 output tokens, 5s timeout). Heuristics always
        # run first — the common path costs zero extra calls.
        "auto_classifier_enabled": True,
        "classifier_model": "",   # empty ⇒ fast_utility role (see model_roles.py)
        # Below this message length the tiebreak never fires (trivial turns).
        "classifier_min_chars": 240,
        # step_back pre-call (principles extracted before the turn). Empty
        # model ⇒ the active turn model (role: refs still expand).
        "step_back_model": "",
        "step_back_timeout_seconds": 20,
        # Per-pattern availability (auto + explicit selection both respect these).
        "cot_enabled": True,
        "step_back_enabled": True,
        "reflection_enabled": True,
        "self_consistency_enabled": True,
        # Self-consistency: k parallel short samples + a judged final answer.
        # Empty model ⇒ the active turn model (samples are the k× cost).
        "sc_model": "",
        "sc_k": 3,
        # Thinking output floor (tokens). 0 = auto: floor at the reasoning
        # minimum whenever a pattern is active or the model reasons natively.
        "min_output_tokens": 0,
    },
    "shell": {
        # Agent shells = LLM-driven command execution (arbitrary code), so it's OFF by
        # default and **enabled per-workspace** (set `allow_shell` on the workspace, not a
        # global flag). Commands run in a bubblewrap jail (FS limited to the conversation
        # work dir, no secret access; network off) — see kit/shell. These are the global
        # sandbox knobs; enablement lives on the workspace.
        "allow_network": False,          # jail keeps network off unless True (exfil risk)
        "allow_unsandboxed": False,      # DANGEROUS: bare subprocess if bubblewrap is missing
        "timeout_seconds": 20,
        "max_output_chars": 20000,
        "max_materialize_bytes": 134217728,  # 128 MiB — cap workspace materialization
        "workdir_cleanup_days": 7,
        "deny_patterns": [],             # extends the built-in deny-list (kit/shell/policy.py)
        "require_confirmation": False,   # future hook
        # Container backend (workspace.shell_backend='container'): a persistent per-workspace
        # Docker container the agent can install into. Needs a reachable Docker daemon
        # (dev: host Docker; prod: the dind sidecar via DOCKER_HOST). Off unless enabled.
        "docker": {
            "enabled": False,
            # Sandbox default only — independent of the API's own Dockerfile/requires-python;
            # a future API Python bump shouldn't sweep this value along with it.
            "image": "python:3.14-slim",
            "memory": "2g",
            "cpus": "2",
            "pids_limit": 512,
            "network": "agentx-shell-net",
            "idle_ttl_days": 7,
        },
    },
    "preferences": {
        "default_model": None,
        "default_reasoning_strategy": "auto",
        "enable_memory_by_default": True,
    },
    "pricing": {
        # Audio (TTS/STT) rate overrides, keyed by `provider:model` →
        # {per_1k_chars, per_minute}. Shipped defaults live in providers/pricing.py
        # (an added default here wouldn't reach installs with an existing config.json);
        # this is the user-override slot the usage ledger reads. Foundation #5.
        "audio": {},
        # Image rate overrides, keyed by `provider:model` → {per_image}. Shipped
        # defaults in providers/pricing.py; this is the user-override slot.
        "images": {},
    },
    "images": {
        # Image generation (avatars first; multi-modal pipelines later). OpenRouter-only
        # today (chat-completions + modalities:["image","text"]).
        "enabled": True,
        "default_model": DEFAULT_IMAGE_MODEL,
        # Dedicated avatar model; blank falls through to default_model.
        "avatar_model": DEFAULT_AVATAR_MODEL,
        # App-level avatar STYLE template; `<SUBJECT>` is substituted with the
        # per-profile subject at generation time (empty subject → the template's
        # synthetic-face branch).
        "avatar_style_prompt": DEFAULT_AVATAR_STYLE_PROMPT,
    },
    "vision": {
        # Image INPUT (the user attaches a picture; a vision-capable model sees it).
        # On by default with this opt-out — a non-vision model degrades to text-only.
        "enabled": True,
        # How many recent image-bearing user turns to re-feed to the model when a
        # cold session rehydrates from history (re-feeding base64 every turn is
        # expensive). 0 disables multi-turn re-feed (single-turn vision still works).
        "refeed_recent_turns": 2,
    },
    "session": {
        # Master switch + model for automatic conversation compaction (both
        # targets — the conversation-state digest AND the legacy prose summary
        # gate on `enabled`). Named for the original prose mechanism; the state
        # digest superseded it as the default target (Slice 1c).
        "rolling_summary": {
            "enabled": True,
            "model": "",  # empty ⇒ summarizer role (see model_roles.py); readers floor to haiku
            # Output budget for one compaction pass; the digest is re-summarized
            # in place each pass, so this bounds its steady-state size.
            "max_tokens": 800,
        },
    },
    "context": {
        # Verbatim CEILING: fraction of the model's real context window the
        # verbatim transcript may use (assembly still hard-caps at
        # window − reserved output). Raised 0.7 → 0.9 so compression kicks in
        # just before the context limit, not lossily at 70%. The compaction
        # trigger is a SEPARATE knob now (summary_trigger_ratio) — the old
        # double duty is gone.
        "verbatim_budget_ratio": 0.9,
        # Post-turn compaction pre-warm threshold — fraction of the turn's REAL
        # history budget (input budget − granted preamble blocks). Fires slightly
        # BELOW the JIT backstop so the digest is usually fresh before the next
        # turn's pre-assembly check ever has to pay for it.
        "summary_trigger_ratio": 0.85,
        # JIT pre-assembly compaction (INV-CTX-1 backstop): when a turn's
        # history exceeds its real budget, refresh the digest BEFORE assembly
        # so dropped turns are always covered (deterministic history_digest
        # fallback when the summarizer is unavailable).
        "preassembly_summary_enabled": True,
        # Floor of most-recent turns always kept verbatim.
        "recent_floor": 4,
        # Max turns pulled from durable history when rehydrating a cold session.
        "rehydrate_max_turns": 400,
        # Structured conversation state (Slice 1a): the goals/decisions/
        # open-threads/artifacts/narrative slots + the rolling digest, rendered
        # as a ledger block and writable via the update_conversation_state tool.
        "conversation_state_enabled": True,
        # Slice 1c: compaction targets ConversationState.digest (the default).
        # Off ⇒ aged-out turns fold into the legacy prose rolling summary instead.
        "conversation_state_compaction_enabled": True,
        # Optional per-turn INPUT spend guard for the tool loop (tokens). 0 = off
        # (the in-turn ceiling derives from the model's real window). Replaces the
        # old flat 32k MAX_INPUT_TOKENS cap that strangled big-window models.
        "max_input_tokens": 0,
    },
    "memory": {
        # Episodic "threads to pull" (Slice 2): pointer leads into past
        # conversations on episodic intent, expanded on demand via read_thread.
        "episodic_leads_enabled": True,
        # Project (workspace) memory channels — `_project_{ws_id}` scoping.
        "project_channels": True,
    },
    "alloy": {
        "max_delegation_depth": 3,
        # Max specialists a supervisor may run concurrently when it emits several
        # delegate_to calls in one turn (fan-out). Bounds combinatorial blow-up.
        "max_parallel_delegations": 3,
        # Per-delegation wall-clock cap, enforced in the streaming tool loop for
        # both blocking (`delegate_to`) and background (`delegate_start`) runs.
        "delegation_timeout_seconds": 300,
        # Phase 16.4: when true, agents in a normal (non-workflow) conversation
        # get a `delegate_to` tool + roster prompt targeting opted-in profiles.
        # Default ON (ship-experimental-on): safe because the roster is opt-in
        # per profile (`available_for_delegation` defaults off) — nothing
        # delegates until the user puts profiles on the roster.
        "allow_adhoc_delegation": True,
        # Non-blocking work orders: offers `delegate_start` alongside
        # `delegate_to` — dispatch returns a receipt immediately and the report
        # folds in automatically later in the same turn. Default ON
        # (ship-experimental-on).
        "non_blocking_delegations": True,
        # Agentic Organizations: strict adjacency-only delegation for org
        # participants (manager → lead of an owned team, lead → own member,
        # member ↑ own lead). Enforced at BOTH the delegate_to tool enum and
        # the executor (INV: dual enforcement). Structural: only agents whose
        # teams carry a manager_agent_id are "in the org" — org-free installs
        # keep the flat opt-in roster byte-identical. Default ON
        # (ship-experimental-on).
        "chain_of_command": True,
    },
    "search": {
        "backend": "tavily",          # "tavily" | "brave"
        "fallback_enabled": True,      # fall back to the other backend on error/empty
        "max_results": 5,
        "cache_ttl_seconds": 300,      # short-TTL in-process cache of identical queries
        # Hard per-call wall-clock cap (seconds). web_search runs synchronously in
        # the tool loop, so an unbounded call (Tavily's SDK default is ~60s) blocks
        # the turn and stalls Stop until it returns. Cap both backends here.
        "timeout": 15,
        # Per-turn search budget (Foundation #5). Max web_search/web_research calls
        # one user turn may make before the tool returns a budget error instead of
        # hitting the backend. Bounds a runaway tool loop's Tavily spend. 0 = unlimited.
        # A soft cap: high enough to never bite normal use, only clips loops.
        "per_turn_limit": 8,
        # Elevated per-turn budget used when a conversation is in Research Mode
        # (research_mode=true on the chat request). Bounded, not unlimited, by
        # default; 0 = unlimited. Research turns need many searches + a few deep
        # web_research calls; this cap (not tool-rounds) governs how deep research goes.
        "research_per_turn_limit": 40,
        # Ledger cost estimate only (search spend is logged to usage_events).
        # Tavily bills per credit: a basic search = 1 credit, advanced = 2.
        "cost_per_credit_usd": 0.008,
        # Brave Search bills ~$5 per 1,000 requests. Used to cost Brave-backend
        # spend in the usage ledger (previously logged as free, which was wrong).
        "brave_cost_per_request_usd": 0.005,
        # API keys (env fallback: TAVILY_API_KEY / BRAVE_API_KEY). Redacted on GET /api/config.
        "tavily_api_key": None,
        "brave_api_key": None,
    },
    "web_research": {
        # Tavily agentic deep-research tool (slow; minutes). Experimental — a
        # genuinely long research wants the background-job path. Off ⇒ the tool
        # returns a disabled error rather than blocking the turn.
        "enabled": True,
        # TTL for caching identical deep-research (query, depth) results. Deep
        # research burns 5–20 credits/call, so caching a repeated query is a big
        # cost saver. Longer than web_search's cache (research is expensive + stable).
        "cache_ttl_seconds": 1800,
        # Tavily's Research API is async: research() only initiates a task
        # ({request_id, status}); the report arrives via get_research(request_id)
        # polling. These bound the poll: total wall-clock wait and per-poll gap.
        # mini completes in ~30-60s, auto ~1-2min, pro can take minutes — on
        # deadline the tool errors with advice to narrow the query or use mini.
        "poll_timeout_seconds": 240,
        "poll_interval_seconds": 5,
        # How many units a single web_research call charges against the per-turn
        # search budget. Deep research is "one call, but a costly one" (5–20
        # credits), so it weighs more than a basic web_search (1).
        "budget_weight": 3,
    },
    "research": {
        # Research Mode (per-conversation `research_mode` flag on the chat request):
        # elevated search budget + a rigorous, evidence-grounded, self-reviewing
        # research prompt that lands a durable, cited report in the attached Project.
        # Ship-experimental-on: enabled by default, opt-out in settings.
        "enabled": True,
        # Tool-round cap for a research turn (vs the standard DEFAULT_MAX_TOOL_ROUNDS).
        # Generous on purpose: the *search budget*, not tool-rounds, should govern
        # depth. If rounds bind first, the loop force-answers mid-research.
        "max_tool_rounds": 40,
        # Default web_research effort tier the prompt steers toward. The cost/quality
        # dial: mini/auto/pro ≈ $0.04/$0.08/$0.16 per deep call at $0.008/credit.
        "default_depth": "auto",
        # Output-budget floor for a research turn's completions. A research turn
        # must fit thinking + a full report in one call — the chat-sized adaptive
        # budget starves it. Bounded by the model's effective output cap (set
        # Model Limits for catalog-miss models to unlock the full floor).
        "min_max_tokens": 16384,
    },
    "ambassador": {
        # Phase 16.6 — a dedicated agent that runs *parallel* to a conversation
        # and briefs the user on a turn without entering (polluting) the main
        # transcript. Identity/model/persona come from the chosen agent profile;
        # these are feature-level knobs only.
        "enabled": True,
        # Global default ambassador profile id. None ⇒ fall back to the default
        # agent profile. (An ambassador is any profile with an `ambassador` section.)
        "profile_id": None,
        # Explicit ambassador model override (authoritative when set). None ⇒ use
        # the chosen profile's model, else a built-in floor. Set via Settings → Ambassador.
        "model": None,
        # How many recent user/assistant turns to read (read-only) for grounding.
        "max_context_turns": 8,
        "max_tokens": 600,
        # Voice knobs — None ⇒ the resolution chain (explicit arg → profile
        # `ambassador.{speech_model,voice,transcription_model}` → these →
        # shipped code default) picks the effective value. Declared here (as
        # None, behavior-preserving) so the config surface is complete for the
        # Settings Manifest / config UI.
        "speech_model": None,
        "voice": None,
        "transcription_model": None,
        # Aide swarm (Phase 16.7) — instead of reading full transcripts into the
        # ambassador's own context, fan out cheap "aide" model calls that each
        # condense ONE conversation read-only and return a short digest (map-reduce:
        # aides map, the ambassador reduces). Keeps its context lean across a
        # cross-conversation survey. Read-only + never-raise; OFF ⇒ today's behavior.
        "aide": {
            "enabled": True,
            # Aide model. "" ⇒ the ambassador's cheap floor (haiku) via fallback.
            "model": "",
            "temperature": 0.2,
            "max_tokens": 220,          # a tight digest, not a transcript
            "max_input_chars": 6000,    # cap transcript fed to an aide
            "max_parallel": 4,          # concurrent aides (3–5 is the sweet spot)
            "timeout_seconds": 20,      # per-aide wall-clock
            "max_per_survey": 8,        # cap aides spawned by one survey
            "cache_ttl_seconds": 1800,  # digest cache TTL (auto-refresh on growth)
        },
        # Dispatch (Phase 16.7, write-side) — hand a task to a chosen worker by minting
        # a new conversation and running it headless (you pick + confirm). Off ⇒ the
        # Dispatch affordance hides and the endpoint 422s.
        "dispatch": {
            "enabled": True,
        },
    },
}


class ConfigManager:
    """
    Manages runtime configuration with file persistence.

    Thread-safe singleton that loads config from data/config.json
    and supports partial updates with automatic persistence.
    """

    CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "config.json"

    def __init__(self):
        self._config: dict = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        """Load configuration from file or use defaults."""
        with self._lock:
            if self.CONFIG_PATH.exists():
                try:
                    with open(self.CONFIG_PATH) as f:
                        self._config = json.load(f)
                    logger.info(f"Loaded config from {self.CONFIG_PATH}")
                except (OSError, json.JSONDecodeError) as e:
                    logger.error(f"Failed to load config: {e}, using defaults")
                    self._config = self._deep_copy(DEFAULT_CONFIG)
            else:
                logger.info("No config file found, using defaults")
                self._config = self._deep_copy(DEFAULT_CONFIG)

    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of a nested dict/list structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj

    def _get_nested(self, data: dict, key: str, default: Any = None) -> Any:
        """Get a nested value using dot notation (e.g., 'providers.lmstudio.base_url')."""
        keys = key.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def _set_nested(self, data: dict, key: str, value: Any) -> None:
        """Set a nested value using dot notation."""
        keys = key.split(".")
        current = data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Examples:
            config.get("providers.lmstudio.base_url")
            config.get("preferences.default_model")
        """
        with self._lock:
            return self._get_nested(self._config, key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a config value by dot-notation key.

        Note: Call save() to persist changes to disk.
        """
        with self._lock:
            self._set_nested(self._config, key, value)

    def unset(self, key: str) -> bool:
        """Remove a config value by dot-notation key. No-op if absent.

        Returns True if a key was removed. Call save() to persist.
        """
        with self._lock:
            keys = key.split(".")
            current = self._config
            for k in keys[:-1]:
                nxt = current.get(k) if isinstance(current, dict) else None
                if not isinstance(nxt, dict):
                    return False
                current = nxt
            if isinstance(current, dict) and keys[-1] in current:
                del current[keys[-1]]
                return True
            return False

    def save(self) -> bool:
        """
        Persist current configuration to disk.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            try:
                # Ensure directory exists
                self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write: serialize to a unique temp file in the same dir,
                # then os.replace() (atomic rename) into place. A reader always
                # sees a complete old-or-new file, and two concurrent writers
                # (e.g. API startup + worker startup both seeding on first boot)
                # can't truncate/interleave into a corrupt config.json — they
                # just race on the final rename (last complete write wins).
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.CONFIG_PATH.parent, prefix=".config.", suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(self._config, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self.CONFIG_PATH)
                except BaseException:
                    # Don't leave the temp file behind on any failure.
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

                logger.info(f"Saved config to {self.CONFIG_PATH}")
                return True
            except OSError as e:
                logger.error(f"Failed to save config: {e}")
                return False

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load()

    def get_all(self) -> dict:
        """Get the entire configuration (for debugging only)."""
        with self._lock:
            return self._deep_copy(self._config)

    def get_provider_value(
        self,
        provider: str,
        key: str,
        env_var: str | None = None,
        default: Any = None
    ) -> Any:
        """
        Get a provider config value with env var fallback.

        Args:
            provider: Provider name (lmstudio, anthropic, openai)
            key: Config key within provider (api_key, base_url, etc.)
            env_var: Environment variable to fall back to
            default: Default value if neither config nor env var is set

        Returns:
            Config value, env var value, or default (in that priority)
        """
        config_value = self.get(f"providers.{provider}.{key}")
        if config_value is not None:
            return config_value

        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                return env_value

        return default

    # Provider api_key → env var, mirroring registry.py's fallbacks. Settings
    # (config.json) is the source of truth; `.env` only seeds it on first boot.
    _PROVIDER_KEY_ENV = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "vercel": "AI_GATEWAY_API_KEY",
    }

    def seed_provider_keys_from_env(self) -> list[str]:
        """Backfill provider api keys from env into config.json when Settings is
        empty — so config.json (the source of truth) is populated once, and
        every process (API + worker) reads keys from Settings rather than `.env`
        at runtime. A key rotated in Settings is never overwritten from `.env`.

        Idempotent + best-effort (logs and swallows failures). Returns the list
        of providers seeded this call.
        """
        seeded: list[str] = []
        try:
            for provider, env_var in self._PROVIDER_KEY_ENV.items():
                # Never clobber an existing Settings value (a rotated key wins).
                if self.get(f"providers.{provider}.api_key"):
                    continue
                env_value = os.environ.get(env_var)
                if env_value:
                    self.set(f"providers.{provider}.api_key", env_value)
                    seeded.append(provider)
            if seeded:
                self.save()
                logger.info(f"Seeded provider keys from env into config: {', '.join(seeded)}")
        except Exception as e:  # noqa: BLE001 — seeding must never break boot
            logger.warning(f"Provider key seed from env failed: {e}")
        return seeded


# Global singleton instance
_config_manager: ConfigManager | None = None
_config_lock = Lock()


def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager singleton."""
    global _config_manager
    with _config_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
        return _config_manager


def set_config_manager(manager: ConfigManager | None) -> None:
    """Inject the global ConfigManager (or `None` to clear).

    Dependency-injection seam: lets tests swap in a fake manager instead of
    patching the module global. Production code keeps calling
    `get_config_manager()`.
    """
    global _config_manager
    with _config_lock:
        _config_manager = manager


def reset_config_manager() -> None:
    """Clear the global ConfigManager so the next access rebuilds it."""
    set_config_manager(None)


def reload_config() -> None:
    """Reload the global config from disk."""
    manager = get_config_manager()
    manager.reload()


def get_context_limit_overrides(model_id: str, provider_name: str) -> dict[str, int]:
    """
    Get context limit overrides from config.

    This returns ONLY user-configured overrides. Provider capabilities
    should be used as the primary source, with these overrides applied on top.

    Provider-level overrides only apply to local providers (lmstudio) where
    hardware constraints may limit context. API providers (anthropic, openai,
    openrouter) already know their per-model capabilities.

    Priority:
    1. Model-specific override in context_limits.models.{model_id}
    2. Provider override in context_limits.{provider_name} (local providers only)

    Args:
        model_id: The model identifier
        provider_name: The provider name (lmstudio, anthropic, openai, etc.)

    Returns:
        Dict with overrides (may be empty, or have context_window and/or max_output_tokens)
    """
    config = get_config_manager()

    # Check for model-specific override first (works for all providers)
    model_override = config.get(f"context_limits.models.{model_id}")
    if model_override:
        return dict(model_override)

    # Provider-level overrides only apply to local providers
    LOCAL_PROVIDERS = {"lmstudio"}
    if provider_name in LOCAL_PROVIDERS:
        provider_config = config.get(f"context_limits.{provider_name}")
        if provider_config:
            return dict(provider_config)

    # No overrides — use provider's own per-model capabilities
    return {}


# Backwards compatibility alias
def get_context_limits(model_id: str, provider_name: str) -> dict[str, int]:
    """Deprecated: Use get_context_limit_overrides instead."""
    overrides = get_context_limit_overrides(model_id, provider_name)
    return {
        "context_window": overrides.get("context_window", 32768),
        "max_output_tokens": overrides.get("max_output_tokens", 4096),
    }
