"""
Tool Output Compressor — LLM-based task-aware compression of large tool outputs.

When tool outputs exceed the size threshold, this service creates a compressed
summary that preserves task-relevant information and provides a structural index
of the full output. Falls back gracefully to empty result when no provider is
available or compression fails.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..config import get_config_manager
from ..prompts.loader import get_prompt_loader
from ..providers.base import Message, MessageRole
from ..providers.registry import ProviderRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of tool output compression."""
    compressed_text: str = ""
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    original_chars: int = 0
    compressed_chars: int = 0


class ToolOutputCompressor:
    """
    LLM-based compression of large tool outputs.

    Uses a fast/cheap model to create task-aware summaries that preserve
    the most relevant information for the agent's current task.
    """

    def __init__(self):
        self._registry: Optional[ProviderRegistry] = None

    @property
    def registry(self) -> ProviderRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    def _get_config(self) -> dict[str, Any]:
        """Get compression config from ConfigManager."""
        config = get_config_manager()
        return {
            "enabled": config.get("compression.enabled", True),
            "model": config.get("compression.model", "claude-3-5-haiku-latest"),
            "temperature": config.get("compression.temperature", 0.2),
            "max_tokens": config.get("compression.max_tokens", 1000),
            "max_summary_chars": config.get("compression.max_summary_chars", 2000),
        }

    def _get_provider(self, model: str) -> Tuple[Any, str]:
        """Get provider and resolved model_id for compression."""
        return self.registry.get_provider_for_model(model)

    async def compress(
        self,
        tool_name: str,
        tool_output: str,
        task_context: str = "",
        max_input_chars: int = 8000,
    ) -> CompressionResult:
        """
        Compress a large tool output into a task-aware summary.

        Args:
            tool_name: Name of the tool that produced the output
            tool_output: The full tool output content
            task_context: The user's current query/task (for relevance)
            max_input_chars: Max chars of tool_output to send to LLM

        Returns:
            CompressionResult with compressed text, or success=False on failure
        """
        cfg = self._get_config()
        original_chars = len(tool_output)

        if not cfg["enabled"]:
            return CompressionResult(
                success=False,
                error="compression_disabled",
                original_chars=original_chars,
            )

        try:
            provider, model_id = self._get_provider(cfg["model"])
        except ValueError as e:
            logger.warning(f"Compression provider unavailable: {e}")
            return CompressionResult(
                success=False,
                error=f"provider_unavailable: {e}",
                original_chars=original_chars,
            )

        # Truncate tool output for the LLM — we're compressing, not forwarding everything
        truncated_output = tool_output[:max_input_chars]
        if len(tool_output) > max_input_chars:
            truncated_output += f"\n...[{len(tool_output) - max_input_chars:,} more chars]"

        # Build prompt from YAML template
        loader = get_prompt_loader()
        prompt = loader.get(
            "compression.summarize",
            task_context=task_context or "(No specific task context provided)",
            tool_name=tool_name,
            tool_output=truncated_output,
            max_chars=str(cfg["max_summary_chars"]),
        )

        messages = [
            Message(role=MessageRole.USER, content=prompt),
        ]

        try:
            result = await provider.complete(
                messages,
                model_id,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )

            compressed = result.content.strip()
            tokens_used = 0
            if result.usage:
                tokens_used = result.usage.get("total_tokens", 0)

            return CompressionResult(
                compressed_text=compressed,
                success=True,
                tokens_used=tokens_used,
                original_chars=original_chars,
                compressed_chars=len(compressed),
            )

        except Exception as e:
            logger.warning(f"Compression failed for '{tool_name}': {e}")
            return CompressionResult(
                success=False,
                error=str(e),
                original_chars=original_chars,
            )

    def compress_sync(
        self,
        tool_name: str,
        tool_output: str,
        task_context: str = "",
        max_input_chars: int = 8000,
    ) -> CompressionResult:
        """Synchronous wrapper for compress()."""
        coro = self.compress(tool_name, tool_output, task_context, max_input_chars)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a thread to avoid nested loop
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return asyncio.run(coro)


# Module-level singleton
_compressor: Optional[ToolOutputCompressor] = None


def get_compressor() -> ToolOutputCompressor:
    """Get the global ToolOutputCompressor instance."""
    global _compressor
    if _compressor is None:
        _compressor = ToolOutputCompressor()
    return _compressor
