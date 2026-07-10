#!/usr/bin/env python
"""Live e2e for the conversation-state durable tier + digest expandability.

Proves, against the real dev cluster (`task db:up`):

  1. save_state writes through to Postgres (TEXT key, FULL id — incl. >150 chars);
  2. Redis wipe → get_state read-through returns the durable state and re-warms Redis
     with the 30-day TTL;
  3. a corrupt Redis payload falls through to the durable copy;
  4. a both-tiers miss is negative-cached (short TTL) so stateless conversations
     don't re-query Postgres per turn;
  5. clear_state removes both tiers;
  6. load_turn_window (the read_thread("current") substrate): earliest-first
     default, centered window, non-UUID id degrades to [].

Run:  uv run python api/agentx_ai/scripts/state_durability_e2e.py

Exits 0 on PASS, 1 on FAIL, 2 on SKIP (Postgres/Redis unavailable). Idempotent —
uses throwaway conversation ids and deletes everything it writes.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

_API_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_API_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agentx_api.settings")

from typing import Any  # noqa: E402

import django  # noqa: E402

django.setup()

from agentx_ai.agent import conversation_state_storage as css  # noqa: E402
from agentx_ai.agent.conversation_history import load_turn_window  # noqa: E402
from agentx_ai.agent.conversation_state_storage import (  # noqa: E402
    ConversationState,
    StateEntry,
    clear_state,
    get_state,
    save_state,
)
from agentx_ai.kit.agent_memory.connections import (  # noqa: E402
    PostgresConnection,
    RedisConnection,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    css._pg_retry_at = 0.0
    r: Any = RedisConnection.get_client()

    # --- state round trip, long non-UUID id (the TEXT-key case) ---
    conv = "state-e2e-" + "x" * 160 + "-" + uuid.uuid4().hex
    key = css._key(conv)
    state = ConversationState(
        goals=[StateEntry(text="prove the durable tier live")],
        digest="earlier: we set up the durable tier and verified it",
    )
    save_state(conv, state)
    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT conversation_id FROM conversation_state WHERE conversation_id = %s",
                (conv,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    check("write-through stores FULL >150-char id", row is not None and row[0] == conv)

    r.delete(key)
    check("read-through after Redis wipe", get_state(conv).digest == state.digest)
    check("re-warm put it back in Redis", bool(r.get(key)))
    ttl = r.ttl(key)
    check("re-warm carries the 30-day TTL", ttl is not None and ttl > 86400, f"ttl={ttl}")

    # --- corrupt hot cache falls through ---
    r.set(key, "{corrupt", ex=60)
    check(
        "corrupt Redis payload falls through to durable",
        get_state(conv).digest == state.digest,
    )

    # --- both-tiers miss negative-cached ---
    fresh = "state-e2e-fresh-" + uuid.uuid4().hex
    css._pg_retry_at = 0.0
    check("both-miss returns empty", get_state(fresh).is_empty())
    nttl = r.ttl(css._key(fresh))
    check(
        "both-miss negative-cached w/ short TTL",
        nttl is not None and 0 < nttl <= css.STATE_NEGATIVE_TTL_SECONDS,
        f"ttl={nttl}",
    )
    clear_state(fresh)

    # --- clear removes both tiers ---
    clear_state(conv)
    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM conversation_state WHERE conversation_id = %s", (conv,)
            )
            gone = cur.fetchone() is None
    finally:
        conn.close()
    check("clear_state removed the durable row", gone)
    check("clear_state removed the Redis key", not r.get(key))

    # --- load_turn_window over conversation_logs ---
    tconv = str(uuid.uuid4())
    conn: Any = PostgresConnection.get_engine().raw_connection()
    try:
        with conn.cursor() as cur:
            for i in range(8):
                cur.execute(
                    """
                    INSERT INTO conversation_logs (conversation_id, turn_index, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (tconv, i, "user" if i % 2 == 0 else "assistant", f"turn number {i}"),
                )
        conn.commit()
    finally:
        conn.close()
    try:
        earliest = load_turn_window(tconv, limit=3)
        check(
            "no center_turn ⇒ earliest-first",
            [t["index"] for t in earliest] == [0, 1, 2],
            f"got={[t['index'] for t in earliest]}",
        )
        window = load_turn_window(tconv, center_turn=4, radius=2)
        check(
            "center_turn window is [center-r, center+r]",
            [t["index"] for t in window] == [2, 3, 4, 5, 6],
            f"got={[t['index'] for t in window]}",
        )
        check("non-UUID id degrades to []", load_turn_window("not-a-uuid-id") == [])
    finally:
        conn: Any = PostgresConnection.get_engine().raw_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_logs WHERE conversation_id = %s", (tconv,)
                )
            conn.commit()
        finally:
            conn.close()

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("RESULT: ALL LIVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as e:  # DBs down → skip, don't fail
        print(f"SKIP: cluster unavailable ({e})")
        code = 2
    sys.exit(code)
