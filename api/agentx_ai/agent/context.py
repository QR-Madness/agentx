"""
Context management for agent conversations.

Handles the context window by:
- Tracking token usage
- Summarizing old context
- Prioritizing relevant information
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..providers.base import Message, MessageRole
from ..providers.registry import get_registry

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_tokens: int = 8000
    summarize_threshold: int = 6000
    min_recent_messages: int = 4
    summary_model: str = "gpt-3.5-turbo"
    tokens_per_message_estimate: int = 100


class ContextManager:
    """
    Manages conversation context to fit within model limits.
    
    Strategies:
    1. Sliding window: Keep most recent messages
    2. Summarization: Summarize old context
    3. Priority selection: Keep important messages
    
    Example usage:
        manager = ContextManager(ContextConfig(max_tokens=8000))
        context = await manager.prepare_context(messages, new_message)
    """
    
    def __init__(self, config: ContextConfig):
        self.config = config
        self._registry = None
    
    @property
    def registry(self):
        """Lazy-load the provider registry."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry
    
    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for messages."""
        # Rough estimate: ~4 chars per token on average
        total_chars = sum(len(m.content) for m in messages)
        # Add overhead for role markers, formatting
        overhead = len(messages) * 10
        return (total_chars // 4) + overhead
    
    async def prepare_context(
        self,
        history: list[Message],
        new_message: Message,
        reserved_tokens: int = 1000,
    ) -> list[Message]:
        """
        Prepare context that fits within token limits.
        
        Args:
            history: Previous messages
            new_message: The new user message
            reserved_tokens: Tokens to reserve for the response
            
        Returns:
            List of messages that fit within context window
        """
        available_tokens = self.config.max_tokens - reserved_tokens
        
        # Start with system messages (always keep)
        system_messages = [m for m in history if m.role == MessageRole.SYSTEM]
        other_messages = [m for m in history if m.role != MessageRole.SYSTEM]
        
        # Estimate tokens
        system_tokens = self.estimate_tokens(system_messages)
        new_message_tokens = self.estimate_tokens([new_message])
        
        remaining = available_tokens - system_tokens - new_message_tokens
        
        if remaining <= 0:
            # Even system + new message is too big
            logger.warning("Context window too small for system messages")
            return system_messages + [new_message]
        
        # Check if all history fits
        history_tokens = self.estimate_tokens(other_messages)
        
        if history_tokens <= remaining:
            # All fits
            return system_messages + other_messages + [new_message]
        
        # Need to trim - try summarization first
        if history_tokens > self.config.summarize_threshold:
            summary = await self._summarize_messages(other_messages[:-self.config.min_recent_messages])
            recent = other_messages[-self.config.min_recent_messages:]
            
            summary_message = Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {summary}"
            )
            
            return system_messages + [summary_message] + recent + [new_message]
        
        # Fall back to sliding window
        messages_to_include = []
        tokens_used = 0
        
        for msg in reversed(other_messages):
            msg_tokens = self.estimate_tokens([msg])
            if tokens_used + msg_tokens > remaining:
                break
            messages_to_include.insert(0, msg)
            tokens_used += msg_tokens
        
        return system_messages + messages_to_include + [new_message]
    
    async def _summarize_messages(self, messages: list[Message]) -> str:
        """Summarize a list of messages."""
        if not messages:
            return ""
        
        try:
            provider, model_id = self.registry.get_provider_for_model(
                self.config.summary_model
            )
            
            # Build conversation text
            conversation = []
            for msg in messages:
                role_name = msg.role.value.capitalize()
                conversation.append(f"{role_name}: {msg.content}")
            
            conversation_text = "\n\n".join(conversation)
            
            summary_messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="Summarize the following conversation concisely, preserving key information and context needed for future responses."
                ),
                Message(
                    role=MessageRole.USER,
                    content=conversation_text
                ),
            ]
            
            result = await provider.complete(
                summary_messages,
                model_id,
                temperature=0.3,
                max_tokens=500,
            )
            
            return result.content
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fall back to truncation
            return "Previous conversation context (details truncated)"
    
    def inject_memory(
        self,
        context: list[Message],
        memories: list[str],
        max_memory_tokens: int = 500,
    ) -> list[Message]:
        """
        Inject relevant memories into the context.
        
        Args:
            context: Current context messages
            memories: Relevant memories to inject
            max_memory_tokens: Maximum tokens for memories
            
        Returns:
            Context with memories injected
        """
        if not memories:
            return context
        
        # Combine memories
        memory_text = "\n".join(f"- {m}" for m in memories)
        
        # Truncate if needed
        if len(memory_text) > max_memory_tokens * 4:
            memory_text = memory_text[:max_memory_tokens * 4] + "..."
        
        memory_message = Message(
            role=MessageRole.SYSTEM,
            content=f"Relevant information from memory:\n{memory_text}"
        )
        
        # Insert after system messages, before conversation
        result = []
        inserted = False
        
        for msg in context:
            result.append(msg)
            if msg.role == MessageRole.SYSTEM and not inserted:
                result.append(memory_message)
                inserted = True
        
        if not inserted:
            result.insert(0, memory_message)
        
        return result
