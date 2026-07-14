#!/usr/bin/env python
"""Freeze a chat run into a golden stream fixture.

Captures BOTH halves of a conversation turn's truth so the client test suite
can assert they agree (the live-vs-restored parity invariant):

  events.jsonl       - the run's full SSE stream, verbatim, straight off the
                       Redis event bus (``chat_run:{run_id}:events``) - exactly
                       what a live or re-attaching client consumes. One line
                       per event: {"id": <stream id>, "sse": <raw SSE string>}.
                       First line is {"_state": <run state hash>}.
  conversation.json  - the persisted turns in the exact shape
                       ``GET /api/conversations/{id}/messages`` serves (the
                       client restore path's input), captured server-side with
                       the same query + metadata folding as the view.
  meta.json          - run state, event census, capture timestamp.

Fixtures are written VERBATIM (no scrubbing) - chunk boundaries are load-
bearing for parse-boundary bugs (e.g. split <think> tags). The replay tests
normalize ids/timestamps at assertion time instead.

Usage (dev cluster up; the bus holds a run for ~2h after its last event):
  uv run python scripts/capture_stream_fixture.py --list
  uv run python scripts/capture_stream_fixture.py <run_id> --name work-order-background
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "client" / "src" / "test" / "fixtures" / "streams"


def _bootstrap_django() -> None:
    sys.path.insert(0, str(REPO_ROOT / "api"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")
    import django

    django.setup()


def _redis():
    import redis

    return redis.Redis.from_url(
        os.environ.get("REDIS_URI", "redis://localhost:6379"),
        decode_responses=True,
    )


def list_runs(user_id: str) -> None:
    r = _redis()
    run_ids = r.zrevrange(f"chat_run:index:{user_id}", 0, 19)
    if not run_ids:
        print(f"no runs indexed for user {user_id!r}")
        return
    for rid in run_ids:
        state = r.hgetall(f"chat_run:{rid}")
        n = r.xlen(f"chat_run:{rid}:events")
        print(
            f"{rid}  [{state.get('status', '?'):9}]  {n:5} events  "
            f"{state.get('created_at', '')[:19]}  {state.get('message', '')[:60]!r}"
        )


def fetch_conversation(conversation_id: str) -> list[dict]:
    """Persisted turns, serialized exactly like views.conversations_messages."""
    from agentx_ai.agent.profiles import get_profile_manager
    from agentx_ai.kit.agent_memory.connections import PostgresConnection

    pg = PostgresConnection.get_engine().raw_connection()
    try:
        with pg.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content, timestamp, turn_index, metadata, model, agent_id
                FROM conversation_logs
                WHERE conversation_id = %s
                ORDER BY turn_index ASC
                """,
                (conversation_id,),
            )
            rows = cursor.fetchall()
    finally:
        pg.close()

    agent_names: dict[str, str] = {}
    try:
        for p in get_profile_manager().list_profiles():
            if getattr(p, "agent_id", None):
                agent_names[p.agent_id] = p.name
    except Exception:
        pass  # attribution is best-effort; fall back to agent_id

    messages = []
    for role, content, timestamp, turn_index, metadata, model, agent_id in rows:
        msg: dict = {
            "role": role,
            "content": content,
            "timestamp": timestamp.isoformat() if timestamp else None,
            "turn_index": turn_index,
        }
        if metadata:
            msg["metadata"] = metadata if isinstance(metadata, dict) else {}
        if model:
            msg.setdefault("metadata", {})["model"] = model
        if agent_id:
            msg.setdefault("metadata", {})["agent_id"] = agent_id
            msg["metadata"]["agent_name"] = agent_names.get(agent_id, agent_id)
        messages.append(msg)
    return messages


def _event_name(sse: str) -> str:
    if sse.startswith("event: "):
        return sse.split("\n", 1)[0].removeprefix("event: ").strip()
    return "?"


def capture(run_id: str, name: str, *, force: bool) -> int:
    r = _redis()
    state = r.hgetall(f"chat_run:{run_id}")
    if not state:
        print(f"error: run {run_id} not found (bus TTL expired?)", file=sys.stderr)
        return 1
    entries = r.xrange(f"chat_run:{run_id}:events", "-", "+")
    if not entries:
        print(f"error: run {run_id} has no events on the bus", file=sys.stderr)
        return 1

    out_dir = FIXTURES_DIR / name
    if out_dir.exists() and not force:
        print(f"error: {out_dir} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    census: dict[str, int] = {}
    with open(out_dir / "events.jsonl", "w") as f:
        f.write(json.dumps({"_state": state}) + "\n")
        for entry_id, fields in entries:
            sse = fields.get("data", "")
            census[_event_name(sse)] = census.get(_event_name(sse), 0) + 1
            f.write(json.dumps({"id": entry_id, "sse": sse}) + "\n")

    conversation_id = state.get("session_id", "")
    conversation = fetch_conversation(conversation_id) if conversation_id else []
    with open(out_dir / "conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)

    meta = {
        "scenario": name,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "status": state.get("status"),
        "message": state.get("message"),
        "created_at": state.get("created_at"),
        "captured_at": datetime.now(UTC).isoformat(),
        "event_count": len(entries),
        "event_census": dict(sorted(census.items(), key=lambda kv: -kv[1])),
        "turn_count": len(conversation),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"froze {name}: {len(entries)} events, {len(conversation)} turns -> {out_dir}")
    print(f"  census: {meta['event_census']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", nargs="?", help="run id to freeze")
    ap.add_argument("--name", help="scenario name (fixture directory)")
    ap.add_argument("--list", action="store_true", help="list recent runs on the bus")
    ap.add_argument("--user", default="default", help="user id for --list")
    ap.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    args = ap.parse_args()

    if args.list:
        list_runs(args.user)
        return 0
    if not args.run_id or not args.name:
        ap.error("either --list, or <run_id> --name <scenario>")

    _bootstrap_django()
    return capture(args.run_id, args.name, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
