"""Seeding, readouts and wipes for the eval and debug harnesses.

``eval_consolidation``, ``eval_recall`` and ``debug_attribution`` seed memory, read
back what the pipeline stored and clean up after themselves. They run against a
sterile cluster (or a unique test user and channel prefix), and most of what they
do spans users: prefix wipes, channel readouts, conversations seeded without a user.
So this class is reachable only through ``CrossUserStore.eval_harness()``: the
cross-user store stays the one way into work across users (Decisions.md ADR-21).
Nothing in the running app calls it.

Writes consume their results before the session closes, so an error reaches the
harness instead of being swallowed when the session closes.
"""

from typing import Any

from ..connections import Neo4jConnection


class EvalHarness:
    """The harnesses' store operations. Obtain it from ``CrossUserStore.eval_harness()``."""

    # -- one user ----------------------------------------------------------------------

    def wipe_user(self, user_id: str) -> None:
        """DESTROY every node that carries this user id."""
        with Neo4jConnection.session() as s:
            s.run("MATCH (n) WHERE n.user_id = $uid DETACH DELETE n", uid=user_id).consume()

    def reset_access_stats(self, user_id: str) -> None:
        """Zero the access counts and last-accessed times on the user's nodes (recall
        reinforces them as a side effect)."""
        with Neo4jConnection.session() as s:
            s.run(
                "MATCH (n) WHERE n.user_id = $uid "
                "SET n.access_count = 0, n.last_accessed = null",
                uid=user_id,
            ).consume()

    def live_facts_about(self, user_id: str, channel: str) -> list[dict[str, Any]]:
        """The user's live facts in `channel`, by claim, each with the names of the
        entities it is about (``claim``, ``about``; ``about`` may hold nulls)."""
        with Neo4jConnection.session() as s:
            return s.run(
                """
                MATCH (f:Fact {user_id: $u, channel: $c})
                WHERE f.superseded_at IS NULL
                OPTIONAL MATCH (f)-[:ABOUT]->(e:Entity)
                RETURN f.claim AS claim, collect(e.name) AS about
                ORDER BY claim
                """, u=user_id, c=channel).data()

    # -- channels and seeds (span users) --------------------------------------------

    def wipe_channel_prefix(self, prefix: str) -> None:
        """DESTROY every node whose channel starts with `prefix`, for every user."""
        with Neo4jConnection.session() as s:
            s.run("MATCH (n) WHERE n.channel STARTS WITH $p DETACH DELETE n",
                  p=prefix).consume()

    def seed_conversation(self, conv_id: str, channel: str) -> None:
        """Create (or reuse) a conversation with no user, in `channel`."""
        with Neo4jConnection.session() as s:
            s.run("MERGE (c:Conversation {id:$id}) SET c.channel = $ch",
                  id=conv_id, ch=channel).consume()

    def seed_outcome(self, conv_id: str, success: bool, task_type: str, channel: str) -> None:
        """Record the conversation's outcome, which pattern detection gates on."""
        with Neo4jConnection.session() as s:
            s.run(
                """
                MATCH (c:Conversation {id:$id})
                CREATE (o:Outcome {success:$success, task_type:$task_type, channel:$ch})
                MERGE (c)-[:RESULTED_IN]->(o)
                """,
                id=conv_id, success=success,
                task_type=task_type, ch=channel,
            ).consume()

    def channel_contents(self, channel: str) -> tuple[list[Any], list[Any], list[Any]]:
        """Everything consolidation stored in `channel`, for every user: facts
        (``c`` claim, ``st`` source turn id), entities (``id``, ``n`` name, ``a``
        aliases) and relationships between them (``src``, ``t`` type, ``tgt``)."""
        with Neo4jConnection.session() as s:
            facts = list(s.run(
                "MATCH (f:Fact {channel:$ch}) RETURN f.claim AS c, "
                "f.source_turn_id AS st", ch=channel))
            entities = list(s.run(
                "MATCH (e:Entity {channel:$ch}) RETURN e.id AS id, "
                "e.name AS n, coalesce(e.aliases, []) AS a", ch=channel))
            relationships = list(s.run(
                "MATCH (a:Entity {channel:$ch})-[r:RELATES_TO]->(b:Entity {channel:$ch}) "
                "RETURN a.name AS src, r.type AS t, b.name AS tgt", ch=channel))
        return facts, entities, relationships

    def learned_strategy(self, channel: str) -> Any | None:
        """A strategy learned from a conversation in `channel` (``tools``, ``d``
        description), if any. Strategies carry no channel, so this goes through the
        conversation they succeeded in."""
        with Neo4jConnection.session() as s:
            return s.run(
                """
                MATCH (c:Conversation {channel:$ch})<-[:SUCCEEDED_IN]-(s:Strategy)
                RETURN s.tool_sequence AS tools, s.description AS d LIMIT 1
                """,
                ch=channel,
            ).single()

    def wipe_eval_run(self, prefix: str, user_id: str, tool_names: list[str]) -> None:
        """DESTROY one consolidation eval run, in order: strategies learned from the
        run's conversations (they carry no channel, so before those conversations
        go), every node in a `prefix` channel, the conversations themselves, the eval
        user, and the named tools (tools are global)."""
        with Neo4jConnection.session() as s:
            s.run("""
                MATCH (c:Conversation) WHERE c.channel STARTS WITH $p
                MATCH (c)<-[:SUCCEEDED_IN]-(s:Strategy) DETACH DELETE s
            """, p=prefix).consume()
            s.run("MATCH (n) WHERE n.channel STARTS WITH $p DETACH DELETE n", p=prefix).consume()
            s.run("MATCH (c:Conversation) WHERE c.channel STARTS WITH $p DETACH DELETE c", p=prefix).consume()
            s.run("MATCH (u:User {id:$uid}) DETACH DELETE u", uid=user_id).consume()
            if tool_names:
                s.run("MATCH (t:Tool) WHERE t.name IN $names DETACH DELETE t", names=tool_names).consume()
