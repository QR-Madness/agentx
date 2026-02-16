"""
Abstract base classes for memory system extensibility.

These ABCs define extension points for customizing the memory system:
- MemoryStore: Custom storage backends
- Embedder: Custom embedding providers
- Extractor: Custom extraction logic
- Reranker: Custom result reranking

Usage:
    from agentx_ai.kit.agent_memory.abc import MemoryStore, Embedder

    class PineconeStore(MemoryStore):
        def store(self, key, data, embedding=None, **kwargs):
            # Custom implementation
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class ScoredResult:
    """A result with a relevance score."""

    item: Any
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Health check result for a component."""

    healthy: bool
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class MemoryStore(ABC):
    """
    Abstract base class for memory storage backends.

    Implement this to create custom storage backends (e.g., different
    vector databases, alternative graph stores).

    Example:
        class PineconeMemoryStore(MemoryStore):
            def store(self, key: str, data: dict, embedding: list[float]) -> str:
                # Store in Pinecone
                ...

            def retrieve(self, query_embedding, filters, top_k) -> list[ScoredResult]:
                # Query Pinecone
                ...
    """

    @abstractmethod
    def store(
        self,
        key: str,
        data: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store data in the memory store.

        Args:
            key: Unique identifier for the data
            data: Data to store
            embedding: Optional vector embedding for similarity search
            **kwargs: Additional store-specific parameters

        Returns:
            The key/ID of the stored item
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[ScoredResult]:
        """
        Retrieve data from the memory store.

        Args:
            query_embedding: Vector for similarity search
            filters: Filter criteria (varies by implementation)
            top_k: Maximum number of results
            **kwargs: Additional store-specific parameters

        Returns:
            List of scored results
        """
        pass

    @abstractmethod
    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete data from the memory store.

        Args:
            key: Key of the item to delete
            **kwargs: Additional store-specific parameters

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """
        Check the health of the memory store.

        Returns:
            HealthStatus indicating store health
        """
        pass


class Embedder(ABC):
    """
    Abstract base class for embedding providers.

    Implement this to add custom embedding models or services.

    Example:
        class CohereEmbedder(Embedder):
            @property
            def dimensions(self) -> int:
                return 1024

            def embed(self, texts: list[str]) -> list[list[float]]:
                # Call Cohere API
                ...
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Default implementation calls embed() with single item.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed([text])[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Generate embeddings in batches (for large inputs).

        Default implementation processes batches sequentially.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embedding vectors
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.embed(batch))
        return results


class Extractor(ABC, Generic[T]):
    """
    Abstract base class for text extractors.

    Implement this to add custom extraction logic for entities,
    facts, relationships, or other structured data.

    Example:
        class SpaCyEntityExtractor(Extractor[Entity]):
            @property
            def extraction_type(self) -> str:
                return "entity"

            def extract(self, text: str, context: dict) -> list[Entity]:
                # Use spaCy NER
                ...
    """

    @property
    @abstractmethod
    def extraction_type(self) -> str:
        """Return the type of data this extractor produces (e.g., 'entity', 'fact')."""
        pass

    @abstractmethod
    def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """
        Extract structured data from text.

        Args:
            text: Text to extract from
            context: Optional context (e.g., previous extractions, user info)

        Returns:
            List of extracted items
        """
        pass

    async def extract_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """
        Async version of extract.

        Default implementation calls sync extract().
        Override for truly async implementations.

        Args:
            text: Text to extract from
            context: Optional context

        Returns:
            List of extracted items
        """
        return self.extract(text, context)


class Reranker(ABC):
    """
    Abstract base class for result rerankers.

    Implement this to add custom reranking logic (e.g., cross-encoders,
    diversity filters, business rules).

    Example:
        class CrossEncoderReranker(Reranker):
            def rerank(self, query: str, results: list[ScoredResult]) -> list[ScoredResult]:
                # Score with cross-encoder model
                ...
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[ScoredResult],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ScoredResult]:
        """
        Rerank results based on the query.

        Args:
            query: The original query text
            results: List of results to rerank
            top_k: Maximum number of results to return (None = all)
            **kwargs: Additional reranker-specific parameters

        Returns:
            Reranked list of results
        """
        pass

    async def rerank_async(
        self,
        query: str,
        results: List[ScoredResult],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ScoredResult]:
        """
        Async version of rerank.

        Default implementation calls sync rerank().
        Override for model-based rerankers.

        Args:
            query: The original query text
            results: List of results to rerank
            top_k: Maximum number of results to return
            **kwargs: Additional parameters

        Returns:
            Reranked list of results
        """
        return self.rerank(query, results, top_k, **kwargs)
