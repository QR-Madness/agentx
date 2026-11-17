"""Counts across the memory types for one user: what the memory stats view shows.

Each per-type class answers questions about its own records; this one answers the
dashboard's question about all of them at once (totals, and a per-channel breakdown),
in one session. Its PostgreSQL twin (Decisions.md ADR-21, Slice 3) counts the same
families from the memory tables.
"""

from typing import Any

from ..connections import Neo4jConnection


class MemoryOverview:
    """Totals and per-channel counts for one user's memory."""

    def counts(self, user_id: str) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
        """``(totals, by_channel)``. Totals cover entities, facts, strategies, procedures
        and turns; the breakdown covers entities, facts, strategies and turns, keyed by
        channel (a node without one counts as ``_global``)."""
        with Neo4jConnection.session() as session:
            # Get totals - query each type separately since they have different structures
            # Entity, Fact, Strategy have user_id property
            # Turn is linked via User -> Conversation -> Turn
            totals = {"entities": 0, "facts": 0, "strategies": 0, "procedures": 0, "turns": 0}

            entity_count = session.run("""
                MATCH (e:Entity {user_id: $user_id})
                RETURN count(e) AS cnt
            """, user_id=user_id).single()
            totals["entities"] = entity_count["cnt"] if entity_count else 0

            fact_count = session.run("""
                MATCH (f:Fact {user_id: $user_id})
                RETURN count(f) AS cnt
            """, user_id=user_id).single()
            totals["facts"] = fact_count["cnt"] if fact_count else 0

            strategy_count = session.run("""
                MATCH (s:Strategy {user_id: $user_id})
                RETURN count(s) AS cnt
            """, user_id=user_id).single()
            totals["strategies"] = strategy_count["cnt"] if strategy_count else 0

            # Distilled procedures (Slice 1 — procedural memory).
            procedure_count = session.run("""
                MATCH (p:Procedure {user_id: $user_id})
                RETURN count(p) AS cnt
            """, user_id=user_id).single()
            totals["procedures"] = procedure_count["cnt"] if procedure_count else 0

            # Turn is linked through Conversation, not directly to user_id property
            turn_count = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)-[:HAS_TURN]->(t:Turn)
                RETURN count(t) AS cnt
            """, user_id=user_id).single()
            totals["turns"] = turn_count["cnt"] if turn_count else 0

            # Get per-channel breakdown
            by_channel: dict[str, dict[str, int]] = {}

            def bucket(record: Any) -> dict[str, int]:
                ch = record["channel"] or "_global"
                if ch not in by_channel:
                    by_channel[ch] = {"entities": 0, "facts": 0, "strategies": 0, "turns": 0}
                return by_channel[ch]

            # Query each type separately for accurate per-channel counts
            entity_result = session.run("""
                MATCH (e:Entity {user_id: $user_id})
                RETURN e.channel AS channel, count(e) AS cnt
            """, user_id=user_id)
            for record in entity_result:
                bucket(record)["entities"] = record["cnt"]

            fact_result = session.run("""
                MATCH (f:Fact {user_id: $user_id})
                RETURN f.channel AS channel, count(f) AS cnt
            """, user_id=user_id)
            for record in fact_result:
                bucket(record)["facts"] = record["cnt"]

            strategy_result = session.run("""
                MATCH (s:Strategy {user_id: $user_id})
                RETURN s.channel AS channel, count(s) AS cnt
            """, user_id=user_id)
            for record in strategy_result:
                bucket(record)["strategies"] = record["cnt"]

            # Turn linked through Conversation
            turn_result = session.run("""
                MATCH (u:User {id: $user_id})-[:HAS_CONVERSATION]->(c:Conversation)-[:HAS_TURN]->(t:Turn)
                RETURN t.channel AS channel, count(t) AS cnt
            """, user_id=user_id)
            for record in turn_result:
                bucket(record)["turns"] = record["cnt"]

        return totals, by_channel
