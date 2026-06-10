"""Embedding generation for vector similarity search."""

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger("agentx")
settings = get_settings()


class EmbeddingProvider:
    """Unified embedding interface supporting multiple providers."""

    def __init__(self):
        self.provider = settings.embedding_provider
        self._client = None
        self._model = None
        self._dimensions_validated = False
        self._dispatcher = None

    def _init_openai(self):
        """Initialize OpenAI client for embeddings."""
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)

    def _init_local(self):
        """Initialize local embedding model on the resolved compute device."""
        from sentence_transformers import SentenceTransformer
        from ..device import resolve_device

        device = resolve_device()
        self._model = SentenceTransformer(settings.local_embedding_model, device=device)
        logger.info(
            "Local embedding model '%s' loaded on device '%s'.",
            settings.local_embedding_model,
            device,
        )

    @property
    def output_dimensions(self) -> int:
        """Auto-detect actual output dimensions from the loaded model."""
        if self.provider == "openai":
            return settings.embedding_dimensions

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
        configured = settings.embedding_dimensions
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

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of texts

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        return self._get_dispatcher().embed(texts)

    def _get_dispatcher(self):
        """Lazily build the process-wide cache+queue dispatcher for this provider."""
        if self._dispatcher is None:
            from .embedding_queue import EmbeddingDispatcher

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
        """Generate embeddings using OpenAI API."""
        if self._client is None:
            self._init_openai()
            assert self._client is not None

        response = self._client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]

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
