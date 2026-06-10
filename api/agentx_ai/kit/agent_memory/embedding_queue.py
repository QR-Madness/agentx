"""Process-wide embedding serializer + cache.

Embeddings are generated from many places — chat recall, consolidation jobs,
strategy/entity indexing, tool-output chunking — on whatever thread happens to
call them. The local ``sentence-transformers`` model isn't safe to call
concurrently and the remote OpenAI path is rate-limited, so uncoordinated bursts
can collide or get dropped.

This module funnels every embedding request through a single daemon worker
(serialized, with opportunistic batching) and fronts it with an LRU+TTL cache so
identical queries skip the work entirely. The public ``EmbeddingDispatcher.embed``
keeps a synchronous signature, so the ~40 existing call sites are unchanged.

The worker is started lazily on first use (never at import) to match the lazy
database-connection convention and avoid spinning a thread the process doesn't
need (e.g. during tests that mock the embedder).
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from collections.abc import Callable

logger = logging.getLogger("agentx")

ComputeFn = Callable[[list[str]], list[list[float]]]

# Substrings that mark a failure as worth retrying (remote rate-limit / transient
# network), vs a permanent error (e.g. a torch RuntimeError) that should fail fast.
_TRANSIENT_NAME_HINTS = ("ratelimit", "timeout", "connection", "apiconnection", "serviceunavailable")
_TRANSIENT_MSG_HINTS = (
    "rate limit", "ratelimit", "429", "timeout", "timed out", "connection",
    "temporarily", "503", "502", "overloaded", "try again",
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(h in name for h in _TRANSIENT_NAME_HINTS):
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _TRANSIENT_MSG_HINTS)


@dataclass
class _Request:
    texts: list[str]
    future: Future[list[list[float]]]


class EmbeddingCache:
    """Thread-safe LRU cache with per-entry TTL, keyed on (namespace, text)."""

    def __init__(self, max_size: int, ttl_seconds: float):
        self._max = max(1, max_size)
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(namespace: str, text: str) -> str:
        return f"{namespace}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"  # noqa: S324 — non-security cache key

    def get(self, namespace: str, text: str) -> list[float] | None:
        key = self._key(namespace, text)
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            ts, vec = item
            if self._ttl and (now - ts) > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return vec

    def put(self, namespace: str, text: str, vec: list[float]) -> None:
        key = self._key(namespace, text)
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now, vec)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class _EmbeddingQueue:
    """Single daemon worker that serializes + batches embedding compute calls."""

    def __init__(
        self,
        compute: ComputeFn,
        *,
        max_size: int,
        batch_max: int,
        window_ms: int,
        timeout: float,
        max_retries: int,
    ):
        self._compute = compute
        self._batch_max = max(1, batch_max)
        self._window = max(0, window_ms) / 1000.0
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._q: queue.Queue[_Request] = queue.Queue(maxsize=max(1, max_size))
        self._worker: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def submit(self, texts: list[str]) -> list[list[float]]:
        """Enqueue a request and block until its embeddings are ready."""
        self._ensure_worker()
        fut: Future[list[list[float]]] = Future()
        # Bounded put provides backpressure; raises queue.Full on timeout.
        self._q.put(_Request(texts=texts, future=fut), timeout=self._timeout)
        return fut.result(timeout=self._timeout)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        with self._start_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run, name="embedding-queue", daemon=True
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                first = self._q.get()
            except Exception:  # pragma: no cover - defensive
                continue
            batch = [first]
            total = len(first.texts)
            deadline = time.monotonic() + self._window
            # Coalesce additional immediately-available requests within the window.
            while total < self._batch_max:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                batch.append(nxt)
                total += len(nxt.texts)
            self._process(batch)

    def _process(self, batch: list[_Request]) -> None:
        all_texts: list[str] = []
        for req in batch:
            all_texts.extend(req.texts)
        try:
            vectors = self._compute_with_retry(all_texts)
        except BaseException as exc:  # noqa: BLE001 - propagate to every waiter
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return
        idx = 0
        for req in batch:
            n = len(req.texts)
            if not req.future.done():
                req.future.set_result(vectors[idx : idx + n])
            idx += n

    def _compute_with_retry(self, texts: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return self._compute(texts)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > self._max_retries or not _is_transient(exc):
                    raise
                backoff = min(2 ** (attempt - 1), 8) * 0.5
                logger.warning(
                    "Embedding compute failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)


class EmbeddingDispatcher:
    """Cache → queue → compute orchestration with a synchronous ``embed`` API."""

    def __init__(self, compute: ComputeFn, settings, namespace: str):
        self.namespace = namespace
        self._compute = compute
        self._cache = (
            EmbeddingCache(settings.embedding_cache_max_size, settings.embedding_cache_ttl_seconds)
            if getattr(settings, "embedding_cache_enabled", True)
            else None
        )
        self._queue = (
            _EmbeddingQueue(
                compute,
                max_size=settings.embedding_queue_max_size,
                batch_max=settings.embedding_batch_max_size,
                window_ms=settings.embedding_batch_window_ms,
                timeout=settings.embedding_request_timeout,
                max_retries=settings.embedding_max_retries,
            )
            if getattr(settings, "embedding_queue_enabled", True)
            else None
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._cache is None:
            return self._compute_or_queue(texts)

        results: list[list[float] | None] = [self._cache.get(self.namespace, t) for t in texts]
        miss_slots = [i for i, r in enumerate(results) if r is None]
        if miss_slots:
            computed = self._compute_or_queue([texts[i] for i in miss_slots])
            for slot, vec in zip(miss_slots, computed, strict=False):
                results[slot] = vec
                self._cache.put(self.namespace, texts[slot], vec)
        return [r for r in results if r is not None]

    def _compute_or_queue(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._queue is not None:
            return self._queue.submit(texts)
        return self._compute(texts)

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
