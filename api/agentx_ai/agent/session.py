"""
Session management for agent conversations.

Sessions maintain conversation state across multiple interactions,
enabling contextual responses and long-running dialogs.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from ..providers.base import Message, MessageRole
from ..tokens import estimate_tokens

if TYPE_CHECKING:
    from .models import AgentProfile

logger = logging.getLogger(__name__)


def compaction_uses_state(cfg=None) -> bool:
    """Whether compaction targets the conversation-state digest (default) or the
    legacy prose summary.

    ONE definition shared by every compaction call site (streaming JIT backstop,
    streaming post-turn pre-warm, background `Agent.chat`) so no two sites can
    compact into different targets — the drift Decisions.md INV-CTX-1 rule (c)
    forbids.
    """
    if cfg is None:
        from ..config import get_config_manager
        cfg = get_config_manager()
    return (
        bool(cfg.get("context.conversation_state_enabled", True))
        and bool(cfg.get("context.conversation_state_compaction_enabled", True))
    )


@dataclass
class Session:
    """
    A conversation session with the agent.
    
    Maintains:
    - Message history
    - Session metadata
    - Summarized context for long conversations
    """
    
    id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    # Message history
    messages: list[Message] = field(default_factory=list)
    
    # Context summary for long conversations
    summary: str | None = None
    
    # Metadata
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Multi-agent conversations (Phase 16.2): agents that have spoken in this
    # conversation, keyed by Docker-style agent_id. Derived from the durable
    # conversation_logs.agent_id attribution (16.1) plus the active agent.
    participants: dict[str, AgentProfile] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the session.

        Growth is bounded by the token-based rolling summary
        (:meth:`SessionManager.maybe_update_summary`), which ages old turns into
        ``summary`` and trims the in-memory tail — so there's no fixed message-count
        cap here (the legacy ``max_messages``/``auto_summarize_at`` knobs were retired
        in Foundation #6).
        """
        self.messages.append(message)
        self.last_active = time.time()

    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages from the session."""
        if limit:
            return self.messages[-limit:]
        return list(self.messages)
    
    def clear(self) -> None:
        """Clear the session messages."""
        self.messages = []
        self.summary = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "message_count": len(self.messages),
            "has_summary": self.summary is not None,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


class SessionManager:
    """
    Manages multiple agent sessions.
    
    Provides:
    - Session creation and retrieval
    - Session cleanup for inactive sessions
    - Session persistence (future)
    """
    
    def __init__(self, max_sessions: int = 1000, session_timeout: float = 3600.0):
        self.sessions: dict[str, Session] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # 1 hour default
    
    def create(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        **metadata: Any,
    ) -> Session:
        """Create a new session.

        If ``session_id`` is supplied, the session is stored under that id
        (so client-supplied ids round-trip across requests). Otherwise a
        fresh UUID is generated.
        """
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()

        sid = session_id or str(uuid.uuid4())
        session = Session(
            id=sid,
            user_id=user_id,
            metadata=metadata,
        )

        self.sessions[sid] = session
        return session
    
    def get(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        
        if session:
            # Check if expired
            if time.time() - session.last_active > self.session_timeout:
                del self.sessions[session_id]
                return None
            
            session.last_active = time.time()
        
        return session
    
    def get_or_create(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        **metadata: Any,
    ) -> Session:
        """Get an existing session or create a new one."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session

        return self.create(user_id=user_id, session_id=session_id, **metadata)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user."""
        result = []
        
        for session in self.sessions.values():
            if user_id and session.user_id != user_id:
                continue
            result.append(session.to_dict())
        
        return result
    
    @staticmethod
    def _split_aged_out(
        session: Session, *, token_threshold: int, recent_floor: int
    ) -> tuple[list[Message], list[Message]] | None:
        """Shared newest→oldest budget walk for both compaction targets.

        Keeps the most-recent turns within ``token_threshold`` (always ≥
        ``recent_floor``); everything older is the aged-out overflow. Returns
        ``(aged_out, kept_tail)`` or ``None`` when nothing needs to age out —
        one implementation so the state-digest and legacy prose paths can't
        drift apart on what "aged out" means.
        """
        non_system = [m for m in session.messages if m.role != MessageRole.SYSTEM]
        if len(non_system) <= recent_floor:
            return None

        used = 0
        keep = 0
        for m in reversed(non_system):
            used += estimate_tokens(m.content)
            keep += 1
            if keep >= recent_floor and used >= token_threshold:
                break
        if keep >= len(non_system):
            return None  # everything fits within budget — nothing to summarize

        aged_out = non_system[: len(non_system) - keep]
        if not aged_out:
            return None
        return aged_out, non_system[len(non_system) - keep:]

    @staticmethod
    def _summary_context_manager():
        """A ContextManager on the compaction summarizer (shared by both targets).

        Empty config ⇒ follow the `summarizer` model role; the concrete model is a
        last-resort floor only when neither an explicit value nor the role is set.
        """
        from ..config import get_config_manager
        from ..model_roles import resolve_member_model
        from .context import ContextConfig, ContextManager

        cfg = get_config_manager()
        explicit_model = cfg.get("session.rolling_summary.model", "")
        summary_model = (
            resolve_member_model("rolling_summary", explicit_model)
            or "anthropic:claude-haiku-4-5-20251001"
        )
        return ContextManager(ContextConfig(
            summary_model=summary_model,
            summary_max_tokens=int(cfg.get("session.rolling_summary.max_tokens", 800)),
        ))

    async def maybe_update_summary(
        self,
        session_id: str,
        *,
        token_threshold: int,
        recent_floor: int = 4,
    ) -> bool:
        """
        Refresh the **legacy prose** rolling summary when the verbatim transcript
        exceeds the token budget — context-window-based, not a fixed message count.

        The default compaction target is the conversation-state digest
        (:meth:`maybe_compact_to_state`); this path serves installs that disabled
        conversation state or its compaction. Aged-out overflow is folded into
        ``session.summary`` via an LLM call (rolling the prior summary in),
        **persisted** so it survives a cold rebuild, and trimmed from the in-memory
        session so it stays lean. Returns True if the summary was updated.
        """
        from ..config import get_config_manager

        cfg = get_config_manager()
        if not cfg.get("session.rolling_summary.enabled", True):
            return False

        session = self.sessions.get(session_id)
        if session is None:
            return False

        split = self._split_aged_out(
            session, token_threshold=token_threshold, recent_floor=recent_floor
        )
        if split is None:
            return False
        aged_out, kept_tail = split

        manager = self._summary_context_manager()

        # Roll the previous summary into the new one so the summary stays bounded.
        to_summarize: list[Message] = []
        if session.summary:
            to_summarize.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Prior summary: {session.summary}",
            ))
        to_summarize.extend(aged_out)

        try:
            new_summary = await manager._summarize_messages(to_summarize)
        except Exception as e:
            logger.warning(f"Rolling summary update failed for {session_id}: {e}")
            return False

        if not new_summary:
            return False

        session.summary = new_summary
        # Drop the summarized turns from the live session (durable copy stays in
        # conversation_logs); keep only the recent verbatim tail.
        session.messages = kept_tail
        try:
            from .conversation_summary_storage import set_summary
            set_summary(session_id, new_summary)
        except Exception as e:  # pragma: no cover - Redis offline
            logger.debug(f"summary persist skipped: {e}")
        return True

    async def maybe_compact_to_state(
        self,
        session_id: str,
        *,
        token_threshold: int,
        recent_floor: int = 4,
    ) -> str | None:
        """Compact aged-out turns into the structured conversation-state object's
        rolling ``digest`` (Slice 1c) — the state object, not the prose summary, is
        the compaction target.

        Same newest→oldest budget walk as :meth:`maybe_update_summary`, but the
        older overflow is folded into ``ConversationState.digest`` via ONE LLM pass
        that rolls the prior digest in (so it's re-summarized in place, never
        appended-and-truncated — no silent loss). ``session_id`` is the conversation
        id (the chat path keys them identically).

        **Returns the digest it wrote** (or ``None`` when nothing was compacted). The
        caller surfaces that value directly as the turn's INV-CTX-1 coverage, so the
        just-evicted turns are guaranteed to be represented **this** turn even if the
        later Redis read behind the ``conversation_state`` block hiccups — the digest
        is also persisted for the next cold read.
        """
        from ..config import get_config_manager

        cfg = get_config_manager()
        if not cfg.get("session.rolling_summary.enabled", True):
            return None

        session = self.sessions.get(session_id)
        if session is None:
            return None

        split = self._split_aged_out(
            session, token_threshold=token_threshold, recent_floor=recent_floor
        )
        if split is None:
            return None
        aged_out, kept_tail = split

        from .conversation_state_storage import get_state, update_digest

        manager = self._summary_context_manager()

        # Roll the prior digest in so it stays bounded (re-summarized, not appended).
        prior = get_state(session_id).digest
        to_summarize: list[Message] = []
        if prior:
            to_summarize.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Summary of the conversation so far: {prior}",
            ))
        to_summarize.extend(aged_out)

        try:
            new_digest = await manager._summarize_messages(to_summarize)
        except Exception as e:
            logger.warning(f"State compaction failed for {session_id}: {e}")
            return None

        if not new_digest:
            return None

        # Persist the digest to the state object, then trim the live session to the
        # recent tail (durable copy of the turns stays in conversation_logs).
        try:
            update_digest(session_id, new_digest)
        except Exception as e:  # pragma: no cover - Redis offline
            logger.warning(f"State digest persist failed for {session_id}: {e}")
            return None
        session.messages = kept_tail
        return new_digest

    def _cleanup_old_sessions(self) -> int:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_active > self.session_timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        # If still at limit, remove oldest
        if len(self.sessions) >= self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_active
            )
            to_remove = len(self.sessions) - self.max_sessions + 10  # Remove 10 extra
            for sid, _ in sorted_sessions[:to_remove]:
                del self.sessions[sid]
                expired.append(sid)

        return len(expired)


_SESSION_MANAGER: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Process-wide SessionManager singleton.

    Per-request agents are constructed fresh (with per-profile config), so
    a per-agent SessionManager would always be empty on the next request.
    Sessions are keyed by client-supplied session_id, so a single shared
    manager serves all agent instances correctly.
    """
    global _SESSION_MANAGER
    if _SESSION_MANAGER is None:
        _SESSION_MANAGER = SessionManager()
    return _SESSION_MANAGER
