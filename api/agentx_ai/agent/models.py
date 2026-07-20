"""
Pydantic models for agent profiles.

Agent profiles define customizable agent identities with names,
model settings, and behavior configuration.
"""

from datetime import datetime
from enum import Enum
import random
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Docker-style agent ID generation
# ---------------------------------------------------------------------------

_ADJECTIVES = [
    "bold", "brave", "bright", "calm", "clever", "cool", "cosmic", "crisp",
    "daring", "eager", "fast", "fierce", "gentle", "giddy", "gleaming",
    "golden", "grand", "happy", "hidden", "humble", "keen", "kind", "lively",
    "lucky", "merry", "mighty", "nimble", "noble", "patient", "plucky",
    "proud", "quiet", "radiant", "rising", "sharp", "silent", "smooth",
    "steady", "swift", "vivid", "warm", "wild", "witty", "zen",
]

_NOUNS = [
    "aurora", "beacon", "breeze", "cedar", "comet", "coral", "crane",
    "crystal", "dawn", "dune", "echo", "ember", "falcon", "fern", "flare",
    "frost", "gale", "grove", "hawk", "iris", "jade", "lark", "lotus",
    "maple", "meadow", "nebula", "oak", "olive", "osprey", "pearl",
    "phoenix", "pine", "quartz", "reef", "ridge", "sage", "spark",
    "spruce", "summit", "thistle", "tide", "vale", "wren",
]


def generate_agent_id() -> str:
    """Generate a Docker-style human-friendly agent ID (e.g., 'bold-cosmic-falcon')."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies / chat thinking patterns for a profile.

    In chat, values compile to thinking patterns (`reasoning/chat_patterns.py`);
    `tot`/`react` degrade honestly there (tot→cot; react→native+tool narration —
    the tool loop IS ReAct) while keeping their full offline behavior on
    `/api/agent/run`.
    """
    AUTO = "auto"
    NATIVE = "native"                    # model thinks natively, no scaffold
    COT = "cot"                          # Chain of Thought
    STEP_BACK = "step_back"              # principles-first pre-call
    TOT = "tot"                          # Tree of Thought (offline; chat: →cot)
    REACT = "react"                      # (offline; chat: →native + narration)
    REFLECTION = "reflection"            # single-pass draft→critique→final
    DEEP_REFLECTION = "deep_reflection"  # multi-pass streamed reflection
    SELF_CONSISTENCY = "self_consistency"  # k samples + judged consensus


class AmbassadorConfig(BaseModel):
    """Extra profile section turning a normal agent into a conversation
    **ambassador** — a dedicated agent that runs *parallel* to a conversation
    and briefs the user on it without ever entering (polluting) the main
    transcript. An ambassador is configured like any other agent profile
    (model, temperature, persona) plus this block, which tunes *how* it briefs.

    Beyond the briefing knobs, the **voice** block (``voice_mode``/``speech_model``
    /``voice``/``speech_speed``) drives spoken briefings via OpenRouter TTS — the
    ambassador speaks its briefings + Q&A aloud, with an opt-in immersive voice
    surface in the panel. ``speech_model``/``voice`` left null fall back to the
    shipped default (``openrouter:microsoft/mai-voice-2``) in ``ambassador.py``.
    """
    enabled: bool = Field(
        default=False, description="Legacy marker (pre-`kind`); an ambassador is now any profile with kind='ambassador'"
    )
    briefing_prompt: str = Field(
        default="",
        description="Extra instructions layered onto the briefing: what to surface / how to brief",
    )
    verbosity: Literal["brief", "normal", "deep"] = Field(
        default="normal", description="How detailed the briefing should be"
    )
    # Functional persona overrides — the ambassador's load-bearing voices. None =
    # use the shipped code default (`_default_persona`/`_qa_persona`/`_draft_persona`
    # in ambassador.py). The *primary* customizable voice is the profile's
    # `system_prompt` (the "Communications"/personality prompt); these are advanced.
    briefing_persona: str | None = Field(
        default=None, description="Override for the per-turn briefing persona (None = shipped default)"
    )
    qa_persona: str | None = Field(
        default=None, description="Override for the free-form Q&A persona (None = shipped default)"
    )
    draft_persona: str | None = Field(
        default=None, description="Override for the outbound-message draft persona (None = shipped default)"
    )
    voice_persona: str | None = Field(
        default=None,
        description="Override for the voice-command router persona (None = shipped default)",
    )
    # Voice (TTS) — spoken briefings via OpenRouter's /audio/speech. STT (the
    # user-speaks half of two-way voice mode) is a separate, later seam.
    voice_mode: bool = Field(
        default=False,
        description="Opt-in: enable the immersive two-way voice surface for this ambassador",
    )
    speech_model: str | None = Field(
        default=None,
        description="TTS/speech model id for spoken briefings (None = shipped default microsoft/mai-voice-2)",
    )
    voice: str | None = Field(
        default=None, description="Voice id for spoken briefings (None = the speech model's default voice)"
    )
    speech_speed: float | None = Field(
        default=None, description="Optional playback-speed multiplier for spoken briefings (provider-dependent)"
    )
    transcription_model: str | None = Field(
        default=None,
        description="STT/transcription model id for voice input (None = shipped default openai/whisper-1)",
    )


class AgentProfile(BaseModel):
    """
    Configuration profile for an agent persona.

    Defines the agent's identity, default settings, and behavior configuration.
    The agent name is injected into system prompts as "Your name is {name}."
    """
    # Identity
    id: str = Field(..., description="Unique identifier for the profile")
    name: str = Field(..., description="Display name for the agent (used in prompts and UI)")
    # Profile kind. 'agent' = a normal chat agent; 'ambassador' = a parallel,
    # non-polluting conversation interpreter (hidden from chat/delegation/@-mention).
    # Named `kind` for consistency with PromptLayer.kind; extensible for future kinds.
    kind: Literal["agent", "ambassador"] = Field(
        "agent", description="Profile kind: 'agent' (chat) or 'ambassador' (parallel interpreter)"
    )
    agent_id: str = Field(default_factory=generate_agent_id, description="Unique human-friendly agent identifier (Docker-style)")
    avatar: str | None = Field(None, description="Avatar icon name (e.g., 'sparkles', 'brain')")
    description: str | None = Field(None, description="Description of this profile's purpose")
    # Short, free-form labels surfacing the agent's traits/roles (e.g. "research",
    # "fast", "writer"). Shown as chips in the agent selector. Capped at 4 so the
    # selector row stays readable; each is trimmed and length-limited.
    tags: list[str] = Field(
        default_factory=list,
        description="Up to 4 short trait/role labels shown in the agent selector",
    )

    # Model settings
    default_model: str | None = Field(None, description="Default model to use for this profile")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature (0.0-2.0)")

    # Prompt configuration
    prompt_profile_id: str | None = Field(
        None,
        description="ID of the PromptProfile to use for system prompt composition"
    )
    system_prompt: str | None = Field(
        None,
        description="Custom system prompt for this agent (prepended to conversations)"
    )

    # Behavior settings
    reasoning_strategy: ReasoningStrategy = Field(
        ReasoningStrategy.AUTO,
        description="Default reasoning strategy for task execution"
    )
    enable_memory: bool = Field(True, description="Whether to use the memory system")
    memory_channel: str = Field("_global", description="Memory channel to use for this profile")
    enable_tools: bool = Field(True, description="Whether to enable MCP tools")
    # Direct mode: bypass the whole agent harness for this turn — no global/layered
    # system prompt, no memory (core + recall), no tools. The model receives only the
    # user's message. The right primitive for a "transform-only" model (a fast
    # classifier/rewriter) and *required* for an image-only model (e.g. flux), which
    # can't act on a system prompt or call tools and otherwise just returns nothing.
    # Auto-forced at request time when the resolved model is image-output-only, so it
    # can't be misconfigured; this flag is the manual opt-in for other cases.
    direct_mode: bool = Field(
        False,
        description="Bypass system prompt + memory + tools; send only the user message (auto-on for image-only models)",
    )
    # Phase 18.2 / 18.9.x: per-profile tool gating. Entries are fully-qualified
    # `server.tool` keys — built-in tools are `_internal.<tool_name>` (e.g.
    # `_internal.checkpoint`); MCP tools are `<server_name>.<tool_name>`. The
    # matching in `Agent._get_tools_for_provider` is exact-string and case-
    # sensitive. Bare names (no `.`) won't match anything under the FQ scheme
    # and are flagged with a startup warning; see profiles.py.
    allowed_tools: list[str] | None = Field(
        None,
        description=(
            "If set, only these fully-qualified tools (`server.tool`; "
            "`_internal.<name>` for built-ins) are exposed to this agent. "
            "None = all enabled."
        ),
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Fully-qualified tools (`server.tool`; `_internal.<name>` for "
            "built-ins) explicitly hidden from this agent. Wins over allowed_tools."
        ),
    )

    # Delegation (Phase 16.4): when true, this profile is offered as an ad-hoc
    # `delegate_to` target to other agents. Default FALSE — the team roster is
    # opt-in (pairs with `alloy.allow_adhoc_delegation` defaulting ON: the
    # feature is live, but nothing delegates until profiles join the roster).
    # Profiles saved before this flip carry their explicit persisted value.
    available_for_delegation: bool = Field(
        False,
        description="Whether other agents may delegate to this profile (ad-hoc delegation)",
    )
    # One-line specialty surfaced to teammates deciding whom to delegate to — shown
    # in the ad-hoc roster prompt and the `delegate_to` tool's target list. Falls
    # back to `description` when unset.
    delegation_hint: str | None = Field(
        None,
        description="One-line specialty shown to teammates deciding whom to delegate to",
    )

    # Ambassador (Phase 16.6): optional extra section turning this profile into a
    # parallel, non-polluting conversation interpreter. None = not an ambassador.
    ambassador: AmbassadorConfig | None = Field(
        None,
        description="Extra section configuring this profile as a conversation ambassador",
    )

    # Metadata
    is_default: bool = Field(False, description="Whether this is the default *agent* profile")
    is_default_ambassador: bool = Field(
        False, description="Whether this is the default *ambassador* (per-kind default; briefings use it)"
    )
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default_factory=datetime.utcnow)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: object) -> list[str]:
        """Trim, drop blanks, length-limit, de-dupe (case-insensitive), cap at 4."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)):
            return []
        seen: set[str] = set()
        out: list[str] = []
        for item in v:
            tag = str(item).strip()[:24]
            if not tag:
                continue
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tag)
            if len(out) >= 4:
                break
        return out

    @field_validator("delegation_hint", mode="before")
    @classmethod
    def _normalize_delegation_hint(cls, v: object) -> str | None:
        """Trim; empty → None; cap at 200 chars (it's a one-liner for teammates)."""
        if v is None:
            return None
        hint = str(v).strip()[:200]
        return hint or None

    @property
    def self_channel(self) -> str:
        """Memory channel for this agent's self-knowledge."""
        return f"_self_{self.agent_id}"

    class Config:
        use_enum_values = True
