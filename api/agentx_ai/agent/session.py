"""
Session management for agent conversations.

Sessions maintain conversation state across multiple interactions,
enabling contextual responses and long-running dialogs.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..providers.base import Message, MessageRole


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
    summary: Optional[str] = None
    
    # Metadata
    user_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
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
    
    def get_messages(self, limit: Optional[int] = None) -> list[Message]:
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
    
    def create(self, user_id: Optional[str] = None, **metadata: Any) -> Session:
        """Create a new session."""
        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()
        
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            user_id=user_id,
            metadata=metadata,
        )
        
        self.sessions[session_id] = session
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
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
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **metadata: Any,
    ) -> Session:
        """Get an existing session or create a new one."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        
        return self.create(user_id=user_id, **metadata)
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list(self, user_id: Optional[str] = None) -> list[dict[str, Any]]:
        """List sessions, optionally filtered by user."""
        result = []
        
        for session in self.sessions.values():
            if user_id and session.user_id != user_id:
                continue
            result.append(session.to_dict())
        
        return result
    
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
