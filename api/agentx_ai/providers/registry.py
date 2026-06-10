"""
Provider registry for managing model providers and configurations.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .base import ModelProvider, ProviderConfig
from ..config import ConfigManager, get_config_manager
from ..exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: str
    model_id: str | None = None  # If different from the key
    context_window: int = 8192
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    local: bool = False
    default_temperature: float = 0.7
    max_output_tokens: int | None = None


class ProviderRegistry:
    """
    Registry for model providers.
    
    Manages provider instances and model configurations,
    providing a unified interface to get providers and models.
    """
    
    def __init__(
        self,
        config_path: Path | None = None,
        config_manager: ConfigManager | None = None,
    ):
        self._providers: dict[str, ModelProvider] = {}
        self._model_configs: dict[str, ModelConfig] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        # Injectable ConfigManager (defaults to the global singleton on use).
        self._config_manager = config_manager
        # Best-effort provider health cache for the fallback path: name ->
        # (expiry_epoch, is_healthy). Populated by health_check() and by
        # complete_with_fallback() observing real call outcomes. Only used to
        # *skip* a known-down provider; the execution-time retry is the real
        # guarantee for the "configured but unreachable" case.
        self._provider_health: dict[str, tuple[float, bool]] = {}

        # Load configuration if provided
        if config_path and config_path.exists():
            self.load_config(config_path)
        else:
            # Load from default location or environment
            self._load_default_config()

    def _load_default_config(self) -> None:
        """Load configuration from ConfigManager with environment variable fallback."""
        config = self._config_manager or get_config_manager()

        # LM Studio (local, OpenAI-compatible) - primary local provider
        lmstudio_url = config.get_provider_value(
            "lmstudio", "base_url", env_var="LMSTUDIO_BASE_URL"
        )
        if lmstudio_url:
            lmstudio_timeout = config.get_provider_value(
                "lmstudio", "timeout", env_var="LMSTUDIO_TIMEOUT", default=300
            )
            self._provider_configs["lmstudio"] = ProviderConfig(
                base_url=lmstudio_url,
                timeout=float(lmstudio_timeout),
            )

        # Anthropic (cloud) - primary cloud provider
        anthropic_key = config.get_provider_value(
            "anthropic", "api_key", env_var="ANTHROPIC_API_KEY"
        )
        if anthropic_key:
            anthropic_url = config.get_provider_value(
                "anthropic", "base_url", env_var="ANTHROPIC_BASE_URL"
            )
            self._provider_configs["anthropic"] = ProviderConfig(
                api_key=anthropic_key,
                base_url=anthropic_url,
            )

        # OpenAI (cloud) - experimental
        openai_key = config.get_provider_value(
            "openai", "api_key", env_var="OPENAI_API_KEY"
        )
        if openai_key:
            openai_url = config.get_provider_value(
                "openai", "base_url", env_var="OPENAI_BASE_URL"
            )
            self._provider_configs["openai"] = ProviderConfig(
                api_key=openai_key,
                base_url=openai_url,
            )

        # OpenRouter (cloud aggregator - 100+ models)
        openrouter_key = config.get_provider_value(
            "openrouter", "api_key", env_var="OPENROUTER_API_KEY"
        )
        if openrouter_key:
            self._provider_configs["openrouter"] = ProviderConfig(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                extra={
                    "site_url": os.environ.get("OPENROUTER_SITE_URL", ""),
                    "app_name": os.environ.get("OPENROUTER_APP_NAME", "AgentX"),
                },
            )

        # Vercel AI Gateway (cloud aggregator - 100+ models, high availability)
        vercel_key = config.get_provider_value(
            "vercel", "api_key", env_var="AI_GATEWAY_API_KEY"
        )
        if vercel_key:
            vercel_url = config.get_provider_value(
                "vercel", "base_url", env_var="AI_GATEWAY_BASE_URL"
            )
            self._provider_configs["vercel"] = ProviderConfig(
                api_key=vercel_key,
                base_url=vercel_url or "https://ai-gateway.vercel.sh/v1",
            )
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from a YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Load provider configurations
        for provider_name, provider_data in config.get("providers", {}).items():
            self._provider_configs[provider_name] = ProviderConfig(
                api_key=provider_data.get("api_key") or os.environ.get(
                    provider_data.get("api_key_env", f"{provider_name.upper()}_API_KEY")
                ),
                base_url=provider_data.get("base_url"),
                # Unset → None so each provider applies its own default.
                timeout=provider_data.get("timeout"),
                max_retries=provider_data.get("max_retries", 3),
            )
        
        # Load model configurations
        for model_name, model_data in config.get("models", {}).items():
            self._model_configs[model_name] = ModelConfig(**model_data)
    
    def get_provider(self, name: str) -> ModelProvider:
        """
        Get or create a provider instance.

        Args:
            name: Provider name ("lmstudio", "anthropic", "openai", "openrouter")

        Returns:
            Configured ModelProvider instance

        Raises:
            ValueError: If provider is not supported or not configured
        """
        if name in self._providers:
            return self._providers[name]

        if name not in self._provider_configs:
            raise ModelNotFoundError(
                f"Provider '{name}' not configured. "
                f"Configure it in Settings or set {name.upper()}_BASE_URL / {name.upper()}_API_KEY.",
                provider=name,
            )

        config = self._provider_configs[name]

        if name == "lmstudio":
            from .lmstudio_provider import LMStudioProvider
            provider = LMStudioProvider(config)
        elif name == "anthropic":
            from .anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
        elif name == "openai":
            from .openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
        elif name == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            provider = OpenRouterProvider(config)
        elif name == "vercel":
            from .vercel_provider import VercelProvider
            provider = VercelProvider(config)
        else:
            raise ModelNotFoundError(f"Unknown provider: {name}", provider=name)

        self._providers[name] = provider
        return provider
    
    def get_model_config(self, model: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        return self._model_configs.get(model)
    
    def get_provider_for_model(self, model: str) -> tuple[ModelProvider, str]:
        """
        Get the appropriate provider for a model.

        Models must use provider:model format (e.g., "anthropic:claude-3-opus").

        Args:
            model: Model in provider:model_id format

        Returns:
            Tuple of (provider, model_id)

        Raises:
            ValueError: If format is invalid or provider not configured
        """
        if ":" not in model:
            raise ValueError(
                f"Invalid model format '{model}'. "
                "Use provider:model format (e.g., 'anthropic:claude-3-5-sonnet-latest', 'lmstudio:llama3.2')."
            )

        provider_name, model_id = model.split(":", 1)

        if provider_name not in self._provider_configs:
            available = ", ".join(self._provider_configs.keys()) if self._provider_configs else "none"
            raise ModelNotFoundError(
                f"Provider '{provider_name}' not configured. "
                f"Available providers: {available}",
                model=model,
                provider=provider_name,
            )

        return self.get_provider(provider_name), model_id

    # ──────────────────────────────────────────────
    #  Model fallback — never hard-fail a feature turn
    # ──────────────────────────────────────────────

    _HEALTH_TTL = 30.0  # seconds a cached health verdict stays authoritative

    def _config(self) -> ConfigManager:
        return self._config_manager or get_config_manager()

    def _fallback_enabled(self) -> bool:
        return bool(self._config().get("models.fallback_enabled", True))

    def _default_chat_model(self) -> str | None:
        """The global default chat model (floor when there's no active turn).

        Reads either latent global-default key (`preferences.default_model` or
        `models.defaults.chat`) — neither has a settings UI yet (Todo backlog), so
        on live turns the real floor is the agent profile's model, threaded in as
        `preferred_fallback`; this only matters for profile-less background work.
        """
        cfg = self._config()
        val = cfg.get("preferences.default_model") or cfg.get("models.defaults.chat")
        return str(val) if val else None

    def mark_provider_health(self, name: str, healthy: bool) -> None:
        """Record a provider's reachability (observed via a real call or a ping)."""
        self._provider_health[name] = (time.time() + self._HEALTH_TTL, healthy)

    def _is_cached_unhealthy(self, name: str) -> bool:
        entry = self._provider_health.get(name)
        return entry is not None and entry[0] > time.time() and not entry[1]

    def _fallback_chain(self, model: str, preferred_fallback: str | None) -> list[str]:
        """Ordered, de-duped `provider:model` candidates: requested → the active
        agent model (caller's known-good) → global default model. Empty/invalid
        entries are dropped."""
        chain: list[str] = []
        for m in (model, preferred_fallback, self._default_chat_model()):
            if m and ":" in m and m not in chain:
                chain.append(m)
        return chain

    def resolve_with_fallback(
        self, model: str, *, preferred_fallback: str | None = None
    ) -> tuple[ModelProvider, str, str | None]:
        """Resolve a `provider:model`, falling back when its provider is
        unconfigured or cached-unhealthy.

        Returns ``(provider, model_id, note)`` where ``note`` is a human string
        describing a substitution (None when the requested model was used). Honors
        the ``models.fallback_enabled`` kill-switch (strict resolve when off).
        """
        if not self._fallback_enabled():
            provider, model_id = self.get_provider_for_model(model)
            return provider, model_id, None

        for cand in self._fallback_chain(model, preferred_fallback):
            provider_name, model_id = cand.split(":", 1)
            if provider_name not in self._provider_configs:
                continue
            if self._is_cached_unhealthy(provider_name):
                continue
            note = None if cand == model else f"requested '{model}' unavailable; using '{cand}'"
            if note:
                logger.warning(f"model fallback: {note}")
            return self.get_provider(provider_name), model_id, note

        # Nothing configured/healthy in the chain — strict resolve gives a clear error.
        provider, model_id = self.get_provider_for_model(model)
        return provider, model_id, None

    async def complete_with_fallback(
        self,
        model: str,
        messages: list,
        *,
        preferred_fallback: str | None = None,
        **kwargs: Any,
    ):
        """Complete with the requested model, transparently retrying down the
        fallback chain on an unconfigured **or unreachable** provider.

        This is the universal "never crash the turn" wrapper: it catches both
        resolution failures and runtime provider errors (timeout/5xx/connection),
        updating the health cache as it goes. Raises only if every candidate fails.
        """
        if not self._fallback_enabled():
            provider, model_id = self.get_provider_for_model(model)
            return await provider.complete(messages, model_id, **kwargs)

        chain = self._fallback_chain(model, preferred_fallback)
        last_exc: Exception | None = None
        tried: list[str] = []
        for cand in chain:
            provider_name, model_id = cand.split(":", 1)
            if provider_name not in self._provider_configs:
                continue
            try:
                provider = self.get_provider(provider_name)
                result = await provider.complete(messages, model_id, **kwargs)
            except Exception as e:  # noqa: BLE001 - try the next candidate
                self.mark_provider_health(provider_name, False)
                last_exc = e
                tried.append(f"{cand}: {e}")
                logger.warning(f"model '{cand}' failed ({e}); trying fallback")
                continue
            self.mark_provider_health(provider_name, True)
            if cand != model:
                logger.warning(f"model fallback: requested '{model}' unavailable; used '{cand}'")
            return result

        if last_exc is not None:
            raise last_exc
        raise ModelNotFoundError(
            f"No usable model for '{model}' (no configured providers in the fallback chain).",
            model=model,
        )

    def list_providers(self) -> list[str]:
        """List configured providers."""
        return list(self._provider_configs.keys())
    
    def list_models(self) -> list[str]:
        """List all configured models."""
        models = list(self._model_configs.keys())
        
        # Also include models from active providers
        for provider in self._providers.values():
            models.extend(provider.list_models())
        
        return list(set(models))
    
    async def health_check(self, per_provider_timeout: float = 4.0) -> dict[str, Any]:
        """Check health of all configured providers.

        Pings every provider in parallel with a per-provider timeout so a single
        unreachable backend (e.g. LM Studio not running) can't stall the whole
        endpoint. The dashboard polls this; with a 4 s cap the worst-case wall
        time is ~4 s regardless of how many providers are configured.
        """
        import asyncio

        names = list(self._provider_configs)

        async def _check(name: str) -> dict[str, Any]:
            try:
                provider = self.get_provider(name)
                return await asyncio.wait_for(
                    provider.health_check(),
                    timeout=per_provider_timeout,
                )
            except TimeoutError:
                return {
                    "status": "unhealthy",
                    "error": f"timeout after {per_provider_timeout:.0f}s",
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

        results = await asyncio.gather(*[_check(n) for n in names])
        health = dict(zip(names, results, strict=False))
        # Feed the fallback path's best-effort health cache.
        for name, res in health.items():
            self.mark_provider_health(name, res.get("status") == "healthy")
        return health

    async def aclose(self) -> None:
        """Close all cached provider instances, releasing their HTTP/SDK clients.

        Safe to call multiple times. Individual close failures are logged and
        do not abort the rest.
        """
        for name, provider in list(self._providers.items()):
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider '{name}': {e}")
        self._providers.clear()

    def reload(self) -> None:
        """
        Reload configuration and clear cached providers.

        Call this after config changes to apply new settings.
        Running requests will continue using old providers;
        new requests will use the updated configuration.
        """
        # Close + clear cached provider instances so evicted clients (OpenAI/
        # Anthropic hold a long-lived AsyncOpenAI/AsyncAnthropic) don't leak
        # their connection pools. reload() is sync, so bridge the async close.
        from ..utils.async_bridge import run_coro_sync
        try:
            run_coro_sync(self.aclose())
        except Exception as e:
            logger.warning(f"Error closing providers during reload: {e}")
            self._providers.clear()
        self._provider_configs.clear()

        # Reload config from the (possibly injected) ConfigManager
        (self._config_manager or get_config_manager()).reload()

        # Re-read provider configs
        self._load_default_config()

        logger.info("Provider registry reloaded")


# Global registry instance
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def set_registry(registry: ProviderRegistry | None) -> None:
    """Inject the global provider registry (or `None` to clear).

    Dependency-injection seam: lets tests swap in a fake registry instead of
    patching the module global. Production code keeps calling `get_registry()`.
    """
    global _registry
    _registry = registry


def reset_registry() -> None:
    """Clear the global provider registry so the next access rebuilds it."""
    set_registry(None)


def get_provider(name: str) -> ModelProvider:
    """Get a provider by name from the global registry."""
    return get_registry().get_provider(name)


def get_model_config(model: str) -> ModelConfig | None:
    """Get model configuration from the global registry."""
    return get_registry().get_model_config(model)


def reload_providers() -> None:
    """Reload provider configuration from ConfigManager."""
    registry = get_registry()
    registry.reload()
