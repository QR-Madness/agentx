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

if TYPE_CHECKING:
    from .models import AgentProfile

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough char→token estimate (mirrors ContextManager.estimate_tokens)."""
    return len(text or "") // 4 + 10


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

    # Settings
    max_messages: int = 100
    auto_summarize_at: int = 50
    
    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.last_active = time.time()
        
        # Check if we need to summarize
        if len(self.messages) >= self.auto_summarize_at and not self.summary:
            # Mark for summarization (actual summarization done externally)
            pass
        
        # Trim if too long
        if len(self.messages) > self.max_messages:
            # Keep system messages and recent history
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            recent = self.messages[-(self.max_messages - len(system_messages)):]
            self.messages = system_messages + recent
    
    def get_messages(self, limit: int | None = None) -> list[Message]:
        """Get messages from the session."""
        if limit:
            return self.messages[-limit:]
        return list(self.messages)
    
    def get_context_messages(self, max_messages: int = 20) -> list[Message]:
        """Get messages suitable for context, including summary if available."""
        messages = []
        
        # Include summary as system message if available
        if self.summary:
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self.summary}"
            ))
        
        # Include recent messages
        messages.extend(self.get_messages(max_messages))
        
        return messages
    
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
    
    async def maybe_update_summary(
        self,
        session_id: str,
        *,
        token_threshold: int,
        recent_floor: int = 4,
    ) -> bool:
        """
        Refresh the rolling summary when the verbatim transcript exceeds the token
        budget — **context-window-based**, not a fixed message count.

        Walks newest→oldest keeping recent turns within ``token_threshold`` (always
        ≥ ``recent_floor``); the older overflow is folded into ``session.summary``
        via an LLM call (rolling the prior summary in), **persisted** so it survives
        a cold rebuild, and trimmed from the in-memory session so it stays lean.
        Returns True if the summary was updated.
        """
        from ..config import get_config_manager

        cfg = get_config_manager()
        if not cfg.get("session.rolling_summary.enabled", True):
            return False

        session = self.sessions.get(session_id)
        if session is None:
            return False

        non_system = [m for m in session.messages if m.role != MessageRole.SYSTEM]
        if len(non_system) <= recent_floor:
            return False

        # Keep the most-recent turns that fit the token budget (>= floor); the rest
        # age out into the summary.
        used = 0
        keep = 0
        for m in reversed(non_system):
            used += _estimate_tokens(m.content)
            keep += 1
            if keep >= recent_floor and used >= token_threshold:
                break
        if keep >= len(non_system):
            return False  # everything fits within budget — nothing to summarize

        aged_out = non_system[: len(non_system) - keep]
        if not aged_out:
            return False

        from .context import ContextManager, ContextConfig

        summary_model = cfg.get(
            "session.rolling_summary.model",
            "anthropic:claude-haiku-4-5-20251001",
        )
        manager = ContextManager(ContextConfig(summary_model=summary_model))

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
        session.messages = non_system[len(non_system) - keep:]
        try:
            from .conversation_summary_storage import set_summary
            set_summary(session_id, new_summary)
        except Exception as e:  # pragma: no cover - Redis offline
            logger.debug(f"summary persist skipped: {e}")
        return True

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
