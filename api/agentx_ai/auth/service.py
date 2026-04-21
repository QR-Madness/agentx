"""Authentication service for AgentX.

Handles user authentication with PostgreSQL for user storage
and Redis for session management.
"""

import json
import logging
import secrets
import threading
from datetime import datetime, timezone
from typing import ClassVar, Optional, TypedDict

import bcrypt
from sqlalchemy import text

from ..kit.agent_memory.connections import RedisConnection, get_postgres_session

logger = logging.getLogger(__name__)

# Thread lock for singleton initialization
_service_lock = threading.RLock()


class SessionData(TypedDict):
    """Session data stored in Redis."""
    user_id: int
    username: str
    created_at: str
    last_active: str
    ip_address: Optional[str]
    user_agent: Optional[str]


class AuthService:
    """
    Authentication service using PostgreSQL for users and Redis for sessions.

    Session tokens are stored in Redis with a configurable TTL (default 24h).
    Sessions are extended on each validated request.
    """

    _instance: ClassVar[Optional["AuthService"]] = None

    SESSION_PREFIX = "agentx:session:"

    def __init__(self, session_ttl: int = 86400):
        """
        Initialize AuthService.

        Args:
            session_ttl: Session TTL in seconds (default 24 hours)
        """
        self.session_ttl = session_ttl

    @classmethod
    def get_instance(cls, session_ttl: int = 86400) -> "AuthService":
        """Get or create singleton AuthService instance."""
        if cls._instance is None:
            with _service_lock:
                if cls._instance is None:
                    cls._instance = cls(session_ttl=session_ttl)
        return cls._instance

    def _get_redis(self):
        """Get Redis client, raising if unavailable."""
        try:
            client = RedisConnection.get_client()
            client.ping()
            return client
        except Exception as e:
            logger.error(f"Redis unavailable for auth: {e}")
            raise RuntimeError("Authentication service unavailable: Redis connection failed")

    def is_setup_required(self) -> bool:
        """
        Check if initial password setup is required.

        Returns True if the agentx_auth table has no users or the root user
        has no password hash.
        """
        try:
            with get_postgres_session() as session:
                result = session.execute(
                    text("SELECT COUNT(*) FROM agentx_auth WHERE password_hash IS NOT NULL")
                )
                count = result.scalar()
                return count == 0
        except Exception as e:
            logger.error(f"Error checking setup status: {e}")
            # If we can't check, assume setup is required
            return True

    def setup_root_password(self, password: str) -> bool:
        """
        Set up the initial root password.

        This should only work if no password is currently set.

        Args:
            password: The password to set

        Returns:
            True if password was set, False otherwise
        """
        if not self.is_setup_required():
            logger.warning("Attempted to run setup when password already exists")
            return False

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12))

        try:
            with get_postgres_session() as session:
                # Check if root user exists
                result = session.execute(
                    text("SELECT id FROM agentx_auth WHERE username = 'root'")
                )
                row = result.fetchone()

                if row:
                    # Update existing root user
                    session.execute(
                        text("UPDATE agentx_auth SET password_hash = :hash, updated_at = NOW() WHERE username = 'root'"),
                        {"hash": password_hash.decode('utf-8')}
                    )
                else:
                    # Insert new root user
                    session.execute(
                        text("INSERT INTO agentx_auth (username, password_hash) VALUES ('root', :hash)"),
                        {"hash": password_hash.decode('utf-8')}
                    )

                logger.info("Root password configured successfully")
                return True
        except Exception as e:
            logger.error(f"Error setting up root password: {e}")
            raise

    def login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[tuple[str, SessionData]]:
        """
        Authenticate user and create session.

        Args:
            username: Username (usually 'root')
            password: Password to verify
            ip_address: Client IP address for session tracking
            user_agent: Client user agent for session tracking

        Returns:
            Tuple of (token, session_data) if successful, None otherwise
        """
        try:
            with get_postgres_session() as session:
                result = session.execute(
                    text("SELECT id, password_hash FROM agentx_auth WHERE username = :username AND is_active = TRUE"),
                    {"username": username}
                )
                row = result.fetchone()

                if not row:
                    logger.warning(f"Login attempt for unknown user: {username}")
                    return None

                user_id, password_hash = row

                # Verify password
                if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                    logger.warning(f"Invalid password for user: {username}")
                    return None

                # Update last login
                session.execute(
                    text("UPDATE agentx_auth SET last_login = NOW() WHERE id = :user_id"),
                    {"user_id": user_id}
                )

            # Create session token
            token = secrets.token_urlsafe(32)
            now = datetime.now(timezone.utc).isoformat()

            session_data: SessionData = {
                "user_id": user_id,
                "username": username,
                "created_at": now,
                "last_active": now,
                "ip_address": ip_address,
                "user_agent": user_agent,
            }

            # Store session in Redis
            redis = self._get_redis()
            session_key = f"{self.SESSION_PREFIX}{token}"
            redis.setex(session_key, self.session_ttl, json.dumps(session_data))

            logger.info(f"User {username} logged in successfully")
            return (token, session_data)

        except Exception as e:
            logger.error(f"Login error: {e}")
            return None

    def logout(self, token: str) -> bool:
        """
        Invalidate session token.

        Args:
            token: Session token to invalidate

        Returns:
            True if session was invalidated, False otherwise
        """
        try:
            redis = self._get_redis()
            session_key = f"{self.SESSION_PREFIX}{token}"
            deleted = redis.delete(session_key)

            if deleted:
                logger.info("Session invalidated")
            return deleted > 0
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

    def validate_session(self, token: str, extend_ttl: bool = True) -> Optional[SessionData]:
        """
        Validate session token and optionally extend TTL.

        Args:
            token: Session token to validate
            extend_ttl: Whether to extend session TTL on successful validation

        Returns:
            Session data if valid, None otherwise
        """
        try:
            redis = self._get_redis()
            session_key = f"{self.SESSION_PREFIX}{token}"

            data = redis.get(session_key)
            if not data:
                return None

            session_data: SessionData = json.loads(data)

            if extend_ttl:
                # Update last_active and extend TTL
                session_data["last_active"] = datetime.now(timezone.utc).isoformat()
                redis.setex(session_key, self.session_ttl, json.dumps(session_data))

            return session_data

        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None

    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password for verification
            new_password: New password to set

        Returns:
            True if password was changed, False otherwise
        """
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters")

        try:
            with get_postgres_session() as session:
                # Get current password hash
                result = session.execute(
                    text("SELECT password_hash FROM agentx_auth WHERE id = :user_id"),
                    {"user_id": user_id}
                )
                row = result.fetchone()

                if not row:
                    return False

                current_hash = row[0]

                # Verify old password
                if not bcrypt.checkpw(old_password.encode('utf-8'), current_hash.encode('utf-8')):
                    logger.warning(f"Invalid old password for user_id: {user_id}")
                    return False

                # Hash new password
                new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(12))

                # Update password
                session.execute(
                    text("UPDATE agentx_auth SET password_hash = :hash, updated_at = NOW() WHERE id = :user_id"),
                    {"hash": new_hash.decode('utf-8'), "user_id": user_id}
                )

                logger.info(f"Password changed for user_id: {user_id}")
                return True

        except Exception as e:
            logger.error(f"Change password error: {e}")
            return False

    def invalidate_all_sessions(self, except_token: Optional[str] = None) -> int:
        """
        Invalidate all sessions, optionally keeping one.

        Useful after password change.

        Args:
            except_token: Token to keep (usually current session)

        Returns:
            Number of sessions invalidated
        """
        try:
            redis = self._get_redis()
            pattern = f"{self.SESSION_PREFIX}*"

            count = 0
            for key in redis.scan_iter(pattern):
                if except_token and key == f"{self.SESSION_PREFIX}{except_token}":
                    continue
                redis.delete(key)
                count += 1

            if count:
                logger.info(f"Invalidated {count} sessions")
            return count

        except Exception as e:
            logger.error(f"Error invalidating sessions: {e}")
            return 0


def get_auth_service(session_ttl: int = 86400) -> AuthService:
    """Get AuthService singleton instance."""
    return AuthService.get_instance(session_ttl=session_ttl)
