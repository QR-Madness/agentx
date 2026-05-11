"""
Background chat job queue.

Fire-and-forget chat runs that surface in a per-user inbox when complete.
Uses Redis Streams for the queue and a Redis hash per job for state.

Job lifecycle:
    queued -> running -> done | failed

Storage keys:
    Stream:   background_chat_jobs           (Redis Stream)
    Hash:     agentx:bg_chat:{job_id}        (TTL 24h)
    Index:    agentx:bg_chat:index:{user_id} (ZSET, score=created_at_ms, capped at MAX_INDEX)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

STREAM_KEY = "background_chat_jobs"
JOB_TTL_SECONDS = 24 * 60 * 60
MAX_INDEX = 50
CONSUMER_GROUP = "bg_chat_workers"
CONSUMER_NAME = "bg_chat_worker_1"


def _job_key(job_id: str) -> str:
    return f"agentx:bg_chat:{job_id}"


def _index_key(user_id: str) -> str:
    return f"agentx:bg_chat:index:{user_id}"


def _redis():
    from ..kit.agent_memory.connections import RedisConnection
    return RedisConnection.get_client()


def enqueue_background_chat(
    *,
    user_id: str,
    message: str,
    session_id: Optional[str] = None,
    profile_id: Optional[str] = None,
    agent_profile_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    model: Optional[str] = None,
    use_memory: bool = True,
) -> str:
    """Enqueue a background chat run. Returns the new job_id."""
    job_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    now_ms = int(time.time() * 1000)

    payload = {
        "job_id": job_id,
        "user_id": user_id,
        "message": message,
        "session_id": session_id or "",
        "profile_id": profile_id or "",
        "agent_profile_id": agent_profile_id or "",
        "workflow_id": workflow_id or "",
        "model": model or "",
        "use_memory": "1" if use_memory else "0",
        "status": "queued",
        "created_at": now,
        "response": "",
        "error": "",
    }

    redis = _redis()
    pipe = redis.pipeline()
    pipe.hset(_job_key(job_id), mapping=payload)
    pipe.expire(_job_key(job_id), JOB_TTL_SECONDS)
    pipe.zadd(_index_key(user_id), {job_id: now_ms})
    # Cap index at MAX_INDEX (drop oldest)
    pipe.zremrangebyrank(_index_key(user_id), 0, -(MAX_INDEX + 1))
    pipe.expire(_index_key(user_id), JOB_TTL_SECONDS)
    pipe.xadd(STREAM_KEY, {"data": json.dumps({"job_id": job_id})})
    pipe.execute()

    logger.info(f"Enqueued background chat job {job_id} for user {user_id}")
    return job_id


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def get_background_chat(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single job by id, or None if missing/expired."""
    raw = _redis().hgetall(_job_key(job_id))
    if not raw:
        return None
    return {_decode(k): _decode(v) for k, v in raw.items()}  # type: ignore[union-attr]


def list_background_chats(user_id: str, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent jobs for a user, most recent first."""
    redis = _redis()
    ids = redis.zrevrange(_index_key(user_id), 0, max(0, limit - 1))
    out: List[Dict[str, Any]] = []
    for raw_id in ids:  # type: ignore[union-attr]
        jid = _decode(raw_id)
        rec = get_background_chat(jid)
        if rec is None:
            # Clean up stale index entry
            redis.zrem(_index_key(user_id), jid)
            continue
        out.append(rec)
    return out


def dismiss_background_chat(job_id: str, *, user_id: str) -> bool:
    """Delete the job hash and remove from the user index. Returns True if deleted."""
    redis = _redis()
    deleted = redis.delete(_job_key(job_id))
    redis.zrem(_index_key(user_id), job_id)
    return bool(deleted)


def _update_job(job_id: str, **fields: str) -> None:
    redis = _redis()
    redis.hset(_job_key(job_id), mapping=fields)
    redis.expire(_job_key(job_id), JOB_TTL_SECONDS)


def _run_job(job: Dict[str, Any]) -> None:
    """Execute a single chat job and persist the result."""
    job_id = job["job_id"]
    _update_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        from ..agent import Agent, AgentConfig

        config_kwargs: Dict[str, Any] = {
            "enable_memory": job.get("use_memory") != "0",
        }
        if job.get("model"):
            config_kwargs["default_model"] = job["model"]

        agent = Agent(AgentConfig(**config_kwargs))
        result = agent.chat(
            job["message"],
            session_id=job.get("session_id") or None,
            profile_id=job.get("profile_id") or None,
        )

        _update_job(
            job_id,
            status="done",
            completed_at=datetime.now(timezone.utc).isoformat(),
            response=result.answer or "",
            session_id=job.get("session_id") or result.task_id,
            total_tokens=str(getattr(result, "total_tokens", 0) or 0),
            total_time_ms=str(getattr(result, "total_time_ms", 0) or 0),
        )
        logger.info(f"Background chat job {job_id} completed")
    except Exception as exc:  # noqa: BLE001 — worker must not crash
        logger.exception(f"Background chat job {job_id} failed: {exc}")
        _update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc)[:1000],
        )


_worker_thread: Optional[threading.Thread] = None
_worker_started = threading.Lock()


def _worker_loop(stop_event: threading.Event) -> None:
    logger.info("Background chat worker started")
    redis = _redis()

    # Ensure consumer group exists (idempotent)
    try:
        redis.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="$", mkstream=True)
    except Exception as exc:  # group already exists
        if "BUSYGROUP" not in str(exc):
            logger.warning(f"xgroup_create: {exc}")

    while not stop_event.is_set():
        try:
            entries = redis.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=1,
                block=2000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"xreadgroup failed: {exc}")
            time.sleep(2)
            continue

        if not entries:
            continue

        for _stream, msgs in entries:  # type: ignore[union-attr]
            for msg_id, fields in msgs:
                try:
                    raw = fields.get(b"data") or fields.get("data")
                    payload = json.loads(_decode(raw))
                    job_id = payload["job_id"]
                    job = get_background_chat(job_id)
                    if job is None:
                        logger.warning(f"Job {job_id} missing from store, skipping")
                    else:
                        _run_job(job)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(f"Worker error on msg {msg_id}: {exc}")
                finally:
                    try:
                        redis.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                    except Exception:
                        pass

    logger.info("Background chat worker stopped")


_stop_event = threading.Event()


def start_worker() -> None:
    """Start the daemon worker thread (idempotent)."""
    global _worker_thread
    with _worker_started:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            args=(_stop_event,),
            name="bg-chat-worker",
            daemon=True,
        )
        _worker_thread.start()
