"""Embedding generation for vector similarity search."""

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger("agentx")


class EmbeddingProvider:
    """Unified embedding interface supporting multiple providers."""

    def __init__(self):
        # The provider choice (and the dispatcher's queue settings below) are
        # frozen at first build of the process-wide singleton — a provider
        # change needs a restart. Everything else reads settings live.
        self.provider = get_settings().embedding_provider
        self._client = None
        self._model = None
        self._dimensions_validated = False
        self._dispatcher = None

    def _init_openai(self):
        """Initialize the OpenAI(-compatible) client for embeddings.

        ``embedding_base_url`` points this same code path at any OpenAI-compatible
        endpoint (OpenRouter, TEI, vLLM, LiteLLM…). Boot-frozen like the provider
        choice — the client is built once per process.
        """
        from openai import OpenAI

        settings = get_settings()
        api_key = settings.embedding_api_key or settings.openai_api_key
        if settings.embedding_base_url:
            # TEI/vLLM commonly run keyless; the SDK requires a non-None key.
            self._client = OpenAI(
                api_key=api_key or "sk-no-auth",
                base_url=settings.embedding_base_url,
            )
        else:
            self._client = OpenAI(api_key=api_key)

    def _init_local(self):
        """Initialize local embedding model on the resolved compute device."""
        from sentence_transformers import SentenceTransformer
        from ..device import resolve_device

        device = resolve_device()
        model_name = get_settings().local_embedding_model
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Local embedding model '%s' loaded on device '%s'.",
            model_name,
            device,
        )

    @property
    def output_dimensions(self) -> int:
        """Auto-detect actual output dimensions from the loaded model."""
        if self.provider == "openai":
            return get_settings().embedding_dimensions

        if self._model is None:
            self._init_local()
            assert self._model is not None

        dims = self._model.get_sentence_embedding_dimension()
        assert dims is not None, "Model did not report embedding dimensions"
        return dims

    def validate_dimensions(self) -> tuple[int, int, bool]:
        """
        Compare actual model output dimensions against configured dimensions.

        Returns:
            (actual, configured, match) tuple
        """
        actual = self.output_dimensions
        configured = get_settings().embedding_dimensions
        return actual, configured, actual == configured

    def _check_dimensions(self):
        """Log a warning once if actual dimensions don't match config."""
        if self._dimensions_validated:
            return
        self._dimensions_validated = True
        actual, configured, match = self.validate_dimensions()
        if not match:
            logger.warning(
                f"Embedding dimension mismatch: model produces {actual}-dim vectors "
                f"but config says {configured}. Vector index queries will fail. "
                f"Update EMBEDDING_DIMENSIONS={actual} or re-initialize schemas."
            )

    def embed(self, texts: str | list[str], timeout: float | None = None) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of texts
            timeout: Optional wait override (seconds) for the queued request —
                background callers (document ingestion) pass a generous value;
                None uses ``embedding_request_timeout`` (sized for chat recall).

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        return self._get_dispatcher().embed(texts, timeout=timeout)

    def _get_dispatcher(self):
        """Lazily build the process-wide cache+queue dispatcher for this provider."""
        if self._dispatcher is None:
            from .embedding_queue import EmbeddingDispatcher

            settings = get_settings()
            model = (
                settings.embedding_model
                if self.provider == "openai"
                else settings.local_embedding_model
            )
            self._dispatcher = EmbeddingDispatcher(
                self._compute, settings, namespace=f"{self.provider}:{model}"
            )
        return self._dispatcher

    def _compute(self, texts: list[str]) -> list[list[float]]:
        """Raw embedding compute — invoked serially by the dispatcher's worker."""
        if self.provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the OpenAI(-compatible) API, chunking large
        input lists to stay under per-request caps."""
        if self._client is None:
            self._init_openai()
            assert self._client is not None

        settings = get_settings()
        max_inputs = max(1, settings.embedding_remote_max_inputs)
        out: list[list[float]] = []
        for start in range(0, len(texts), max_inputs):
            response = self._client.embeddings.create(
                model=settings.embedding_model,
                input=texts[start:start + max_inputs],
            )
            # Defensive: honor the response's index field rather than assuming
            # every compatible endpoint preserves input order.
            out.extend(
                item.embedding
                for item in sorted(response.data, key=lambda d: d.index)
            )
        self._check_remote_dimensions(out)
        return out

    def _check_remote_dimensions(self, vectors: list[list[float]]) -> None:
        """Warn once if the remote model's vectors don't match configured dims
        (mirrors the local ``_check_dimensions``; a mismatch silently breaks
        vector-index queries)."""
        if self._dimensions_validated or not vectors:
            return
        self._dimensions_validated = True
        actual = len(vectors[0])
        configured = get_settings().embedding_dimensions
        if actual != configured:
            logger.warning(
                f"Embedding dimension mismatch: remote endpoint returns {actual}-dim "
                f"vectors but config says {configured}. Vector index queries will "
                f"fail. Update EMBEDDING_DIMENSIONS={actual} or change the model."
            )

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local model."""
        if self._model is None:
            self._init_local()
            assert self._model is not None

        self._check_dimensions()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """
        Convenience method for single text embedding.

        Args:
            text: Text to embed

        Returns:
            Single embedding vector
        """
        return self.embed(text)[0]


@lru_cache
def get_embedder() -> EmbeddingProvider:
    """Get cached embedding provider instance."""
    return EmbeddingProvider()
