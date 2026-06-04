"""Procedural memory - tool usage patterns and successful strategies."""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, TYPE_CHECKING, cast
import json
import logging
import re

from typing_extensions import LiteralString
from sqlalchemy import text

from ..models import Strategy, Procedure
from ..connections import Neo4jConnection, get_postgres_session
from ..embeddings import get_embedder
from ..query_utils import CypherFilterBuilder, convert_record_datetimes, convert_all_datetimes

if TYPE_CHECKING:
    from ..audit import MemoryAuditLogger

logger = logging.getLogger(__name__)


# Cheap, LLM-free detector for an explicit standing rule/preference stated in a
# user message — a high-signal procedural candidate (the "encode" loop). Lossy-
# permissive by design: Slice 1's baseline-deviation filter prunes false positives.
_RULE_TRIGGERS = re.compile(
    r"\b("
    r"from now on|going forward|in (?:the )?future|"
    r"always|never|"
    r"make sure(?: to)?|be sure to|ensure that|"
    r"remember to|don'?t forget to|"
    r"don'?t ever|do not ever|"
    r"i(?:'d| would)? prefer|i prefer|we prefer|we usually|"
    r"please always|please never|you should always|you should never"
    r")\b",
    re.IGNORECASE,
)


def detect_explicit_rule(message: str) -> Optional[str]:
    """Return the rule clause if a message states a standing rule/preference, else None.

    Heuristic only (no LLM): matches imperative/preference phrasing and returns the
    clause from the trigger to the end of its sentence (capped at 500 chars).
    """
    if not message:
        return None
    m = _RULE_TRIGGERS.search(message)
    if not m:
        return None
    # Clause = from the trigger phrase to the next sentence boundary (or end).
    tail = message[m.start():]
    clause = re.split(r"(?<=[.!?\n])\s", tail, maxsplit=1)[0].strip()
    return clause[:500] or None


class ProceduralMemory:
    """Handles tool usage patterns and successful strategies."""

    def __init__(self, audit_logger: Optional["MemoryAuditLogger"] = None):
        """Initialize procedural memory.

        Args:
            audit_logger: Optional audit logger for operation tracking.
        """
        self.embedder = get_embedder()
        self._audit_logger = audit_logger

    def record_invocation(
        self,
        conversation_id: str,
        turn_id: Optional[str],
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        success: bool,
        latency_ms: int,
        channel: str = "_global",
        turn_index: Optional[int] = None,
    ) -> None:
        """
        Record a tool invocation.

        Args:
            conversation_id: Conversation ID
            turn_id: Turn ID (optional)
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output
            success: Whether invocation was successful
            latency_ms: Latency in milliseconds
            channel: Memory channel
            turn_index: Turn index within conversation (optional)
        """
        # PostgreSQL for audit log
        with get_postgres_session() as session:
            session.execute(
                text(
                    """
                INSERT INTO tool_invocations
                (conversation_id, turn_index, tool_name, tool_input, tool_output, success, latency_ms, channel)
                VALUES (:conv_id, :turn_idx, :tool, :input, :output, :success, :latency, :channel)
            """
                ),
                {
                    "conv_id": conversation_id,
                    "turn_idx": turn_index if turn_index is not None else 0,
                    "tool": tool_name,
                    "input": json.dumps(tool_input),
                    "output": json.dumps(tool_output) if tool_output else None,
                    "success": success,
                    "latency": latency_ms,
                    "channel": channel,
                },
            )

        # Neo4j for graph relationships
        # FIX: Calculate running average correctly
        # Formula: new_avg = old_avg + (new_value - old_avg) / new_count
        # We need to increment count AFTER calculating the average update
        with Neo4jConnection.session() as neo_session:
            neo_session.run(
                """
                MERGE (tool:Tool {name: $tool_name})
                ON CREATE SET tool.usage_count = 0,
                              tool.success_count = 0,
                              tool.avg_latency_ms = $latency
                SET tool.avg_latency_ms = CASE
                        WHEN tool.usage_count = 0 THEN $latency
                        ELSE tool.avg_latency_ms + ($latency - tool.avg_latency_ms) / (tool.usage_count + 1)
                    END,
                    tool.usage_count = tool.usage_count + 1,
                    tool.success_count = tool.success_count + CASE WHEN $success THEN 1 ELSE 0 END

                WITH tool
                MATCH (c:Conversation {id: $conv_id})
                CREATE (inv:ToolInvocation {
                    timestamp: datetime(),
                    tool_name: $tool_name,
                    success: $success,
                    latency_ms: $latency,
                    channel: $channel,
                    turn_index: $turn_index
                })
                MERGE (c)-[:USED_TOOL]->(inv)
                MERGE (inv)-[:INVOKED]->(tool)
            """,
                conv_id=conversation_id,
                tool_name=tool_name,
                success=success,
                latency=latency_ms,
                channel=channel,
                turn_index=turn_index,
            )

    def stage_candidate(
        self,
        conversation_id: str,
        signal: str,
        content: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        channel: str = "_global",
        agent_id: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> None:
        """Stage a raw procedural-memory candidate (the encode loop).

        A cheap PG insert of a high-signal event — a steer/correction or an
        explicit user rule. Slice 1 consolidation distills `pending` rows into
        scoped Procedures. Mirrors `record_invocation`'s write pattern.
        """
        with get_postgres_session() as session:
            session.execute(
                text(
                    """
                INSERT INTO procedure_candidates
                (conversation_id, turn_index, signal, content, context, channel, agent_id)
                VALUES (:conv_id, :turn_idx, :signal, :content, :context, :channel, :agent_id)
            """
                ),
                {
                    "conv_id": conversation_id,
                    "turn_idx": turn_index,
                    "signal": signal,
                    "content": content,
                    "context": json.dumps(context) if context else None,
                    "channel": channel,
                    "agent_id": agent_id,
                },
            )

    def count_candidates(self, *, status: str = "pending", channel: Optional[str] = None) -> int:
        """Count staged candidates (for observability / stats)."""
        sql = "SELECT COUNT(*) FROM procedure_candidates WHERE status = :status"
        params: Dict[str, Any] = {"status": status}
        if channel:
            sql += " AND channel = :channel"
            params["channel"] = channel
        with get_postgres_session() as session:
            return int(session.execute(text(sql), params).scalar() or 0)

    # ------------------------------------------------------------------
    # Procedures (Slice 1) — distilled, scoped rules (the "how we work here"
    # delta). Distinct from the tool-sequence ``Strategy`` machinery above:
    # a Procedure has an NL trigger + replayable body and is strengthened, not
    # duplicated, as the same pattern recurs.
    # ------------------------------------------------------------------

    def _safe_embed(self, text_value: str) -> Optional[List[float]]:
        """Embed text, degrading to None if no embedder is available.

        Keeps procedural writes working (sans vector dedupe/search) on a box
        with no embedding model — consistent with the rest of the kit.
        """
        try:
            return self.embedder.embed_single(text_value)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Procedure embedding skipped (no embedder?): {e}")
            return None

    def mark_candidates(
        self,
        candidate_ids: List[int],
        status: str,
        *,
        distilled_into: Optional[str] = None,
    ) -> None:
        """Bulk-update staged candidates' lifecycle (``distilled`` | ``discarded``)."""
        if not candidate_ids:
            return
        with get_postgres_session() as session:
            session.execute(
                text(
                    """
                UPDATE procedure_candidates
                SET status = :status, distilled_into = :distilled_into
                WHERE id = ANY(:ids)
            """
                ),
                {
                    "status": status,
                    "distilled_into": distilled_into,
                    "ids": candidate_ids,
                },
            )

    def learn_procedure(
        self,
        *,
        trigger: str,
        body: str,
        rationale: str = "",
        scope: str = "_global",
        agent_id: Optional[str] = None,
        signal_kinds: Optional[List[str]] = None,
        evidence_refs: Optional[List[str]] = None,
        conversation_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Procedure:
        """Write a distilled Procedure node (Slice 1 — the encode→distill output).

        Repurposes the write pattern of the dead ``learn_strategy`` for the new
        richer Procedure model: NL trigger + replayable body, scoped, with
        evidence back-references.
        """
        procedure = Procedure(
            trigger=trigger,
            body=body,
            rationale=rationale,
            scope=scope,
            agent_id=agent_id,
            signal_kinds=signal_kinds or [],
            evidence_refs=evidence_refs or [],
            embedding=self._safe_embed(f"{trigger}\n{body}"),
            created_at=datetime.now(timezone.utc),
            last_reinforced=datetime.now(timezone.utc),
        )

        with Neo4jConnection.session() as session:
            session.run(
                """
                CREATE (p:Procedure {
                    id: $id,
                    trigger: $trigger,
                    trigger_features: $trigger_features,
                    body: $body,
                    rationale: $rationale,
                    channel: $scope,
                    scope: $scope,
                    agent_id: $agent_id,
                    strength: 1,
                    evidence_refs: $evidence_refs,
                    signal_kinds: $signal_kinds,
                    embedding: $embedding,
                    user_id: $user_id,
                    created_at: datetime(),
                    last_reinforced: datetime()
                })

                // Link to user
                WITH p
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_PROCEDURE]->(p)

                // Link to evidence conversations
                WITH p
                UNWIND $conversation_ids AS conv_id
                OPTIONAL MATCH (c:Conversation {id: conv_id})
                FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (p)-[:DISTILLED_FROM]->(c)
                )
            """,
                id=procedure.id,
                trigger=procedure.trigger,
                trigger_features=json.dumps(procedure.trigger_features),
                body=procedure.body,
                rationale=procedure.rationale,
                scope=procedure.scope,
                agent_id=procedure.agent_id,
                evidence_refs=procedure.evidence_refs,
                signal_kinds=procedure.signal_kinds,
                embedding=procedure.embedding,
                user_id=user_id,
                conversation_ids=conversation_ids or [],
            )

        return procedure

    def reinforce_procedure(
        self,
        procedure_id: str,
        *,
        evidence_refs: Optional[List[str]] = None,
        signal_kinds: Optional[List[str]] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """Strengthen an existing Procedure (replay): ``strength += 1``, merge
        evidence + signal kinds, bump ``last_reinforced``. Repurposes the dead
        ``reinforce_strategy`` write pattern."""
        with Neo4jConnection.session() as session:
            filters = CypherFilterBuilder("p")
            filters.add_channel_filter(channel)
            result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (p:Procedure {{id: $id}})
                WHERE true {filters.build_inline()}
                SET p.strength = coalesce(p.strength, 1) + 1,
                    p.last_reinforced = datetime(),
                    p.evidence_refs = coalesce(p.evidence_refs, []) +
                        [x IN $evidence_refs WHERE NOT x IN coalesce(p.evidence_refs, [])],
                    p.signal_kinds = coalesce(p.signal_kinds, []) +
                        [x IN $signal_kinds WHERE NOT x IN coalesce(p.signal_kinds, [])]
                RETURN p.id AS updated_id
            """,
                ),
                id=procedure_id,
                channel=channel,
                evidence_refs=evidence_refs or [],
                signal_kinds=signal_kinds or [],
            )
            record = result.single()
            return record is not None and record["updated_id"] is not None

    def find_procedures(
        self,
        query: str,
        *,
        channels: Optional[List[str]] = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Procedure]:
        """Vector-search procedures by ``trigger + body`` similarity (used for
        dedupe-on-write and the deferred deliberate-recall mode). ``min_score``
        filters to cosine-similar matches (dedupe). Returns [] when no embedder is
        available."""
        embedding = self._safe_embed(query)
        if embedding is None:
            return []

        with Neo4jConnection.session() as session:
            conditions = []
            if channels:
                conditions.append("p.channel IN $channels")
            if min_score is not None:
                conditions.append("score >= $min_score")
            where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            try:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    CALL db.index.vector.queryNodes('procedure_embeddings', $k, $embedding)
                    YIELD node AS p, score
                    {where_clause}
                    RETURN p.id AS id,
                           p.trigger AS trigger,
                           p.body AS body,
                           p.rationale AS rationale,
                           p.scope AS scope,
                           p.agent_id AS agent_id,
                           p.strength AS strength,
                           p.evidence_refs AS evidence_refs,
                           p.signal_kinds AS signal_kinds,
                           score
                    ORDER BY score DESC
                """,
                    ),
                    k=top_k * 2,
                    embedding=embedding,
                    channels=channels,
                    min_score=min_score,
                )
                procedures = []
                for record in result:
                    procedures.append(
                        Procedure(
                            id=record["id"],
                            trigger=record["trigger"] or "",
                            body=record["body"] or "",
                            rationale=record["rationale"] or "",
                            scope=record["scope"] or "_global",
                            agent_id=record["agent_id"],
                            strength=record["strength"] or 1,
                            evidence_refs=record["evidence_refs"] or [],
                            signal_kinds=record["signal_kinds"] or [],
                        )
                    )
                return procedures[:top_k]
            except Exception as e:  # noqa: BLE001
                # Most commonly the vector index doesn't exist yet (schema not
                # initialized). Degrade to "no match" so distillation still creates
                # the procedure rather than aborting the whole job.
                logger.warning(f"find_procedures skipped (vector search unavailable): {e}")
                return []

    def get_reflex_procedures(
        self,
        channels: List[str],
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """The reflex core: top-``strength`` procedures across the recall channels.

        Non-vector and *maintained, not searched* — injected every turn regardless
        of query content (procedural recall is trigger-conditional, not content-
        similar). Returns lightweight dicts for prompt rendering.
        """
        if not channels:
            return []
        with Neo4jConnection.session() as session:
            result = session.run(
                """
                MATCH (p:Procedure)
                WHERE p.channel IN $channels
                RETURN p.trigger AS trigger,
                       p.body AS body,
                       p.scope AS scope,
                       coalesce(p.strength, 1) AS strength
                ORDER BY strength DESC, p.last_reinforced DESC
                LIMIT $limit
            """,
                channels=channels,
                limit=limit,
            )
            return [dict(record) for record in result]

    def count_procedures(self, *, channel: Optional[str] = None) -> int:
        """Count stored procedures (observability / stats)."""
        with Neo4jConnection.session() as session:
            filters = CypherFilterBuilder("p")
            filters.add_channel_filter(channel)
            result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (p:Procedure)
                WHERE true {filters.build_inline()}
                RETURN count(p) AS total
            """,
                ),
                channel=channel,
            )
            record = result.single()
            return int(record["total"]) if record else 0

    def list_procedures(
        self, user_id: str, channel: str = "_global", offset: int = 0, limit: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """List procedures with pagination (mirrors ``list_strategies``)."""
        with Neo4jConnection.session() as session:
            conditions = ["p.user_id = $user_id"]
            if channel == "_all":
                pass
            elif channel and channel != "_global":
                conditions.append("(p.channel = $channel OR p.channel = '_global')")
            else:
                conditions.append("p.channel = '_global'")
            where_clause = " AND ".join(conditions)

            count_result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (p:Procedure)
                WHERE {where_clause}
                RETURN count(p) AS total
            """,
                ),
                user_id=user_id,
                channel=channel,
            )
            record = count_result.single()
            total = record["total"] if record else 0

            result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (p:Procedure)
                WHERE {where_clause}
                RETURN p.id AS id,
                       p.trigger AS trigger,
                       p.body AS body,
                       p.rationale AS rationale,
                       p.scope AS scope,
                       p.agent_id AS agent_id,
                       coalesce(p.strength, 1) AS strength,
                       p.signal_kinds AS signal_kinds,
                       p.evidence_refs AS evidence_refs,
                       p.channel AS channel,
                       p.last_reinforced AS last_reinforced
                ORDER BY p.strength DESC, p.last_reinforced DESC
                SKIP $offset
                LIMIT $limit
            """,
                ),
                user_id=user_id,
                channel=channel,
                offset=offset,
                limit=limit,
            )
            # convert_all_datetimes (not the allowlist variant) so last_reinforced
            # and any future temporal field serialize cleanly.
            procedures = [convert_all_datetimes(dict(record)) for record in result]
            return procedures, total

    def learn_strategy(
        self,
        description: str,
        context_pattern: str,
        tool_sequence: List[str],
        from_conversation_id: Optional[str] = None,
        success: bool = True,
        user_id: Optional[str] = None,
        channel: str = "_global",
    ) -> Strategy:
        """
        Record a successful (or failed) strategy pattern.

        Args:
            description: Strategy description
            context_pattern: Context pattern for matching
            tool_sequence: Sequence of tools used
            from_conversation_id: Source conversation ID
            success: Whether strategy was successful
            user_id: User ID for linking
            channel: Memory channel

        Returns:
            Strategy object
        """
        embedding = self.embedder.embed_single(description)

        strategy = Strategy(
            description=description,
            context_pattern=context_pattern,
            tool_sequence=tool_sequence,
            embedding=embedding,
            success_count=1 if success else 0,
            failure_count=0 if success else 1,
            last_used=datetime.now(timezone.utc),
        )

        with Neo4jConnection.session() as session:
            session.run(
                """
                CREATE (s:Strategy {
                    id: $id,
                    description: $description,
                    context_pattern: $context_pattern,
                    tool_sequence: $tool_sequence,
                    embedding: $embedding,
                    success_count: $success_count,
                    failure_count: $failure_count,
                    user_id: $user_id,
                    channel: $channel,
                    last_used: datetime()
                })

                // Link to user
                WITH s
                MERGE (u:User {id: $user_id})
                MERGE (u)-[:HAS_STRATEGY]->(s)

                // Link to tools
                WITH s
                UNWIND $tool_sequence AS tool_name
                MATCH (t:Tool {name: tool_name})
                MERGE (s)-[:USES_TOOL]->(t)

                // Link to conversation if provided
                WITH s
                OPTIONAL MATCH (c:Conversation {id: $conv_id})
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND $success THEN [1] ELSE [] END |
                    MERGE (s)-[:SUCCEEDED_IN]->(c)
                )
                FOREACH (_ IN CASE WHEN c IS NOT NULL AND NOT $success THEN [1] ELSE [] END |
                    MERGE (s)-[:FAILED_IN]->(c)
                )
            """,
                id=strategy.id,
                description=strategy.description,
                context_pattern=strategy.context_pattern,
                tool_sequence=strategy.tool_sequence,
                embedding=strategy.embedding,
                success_count=strategy.success_count,
                failure_count=strategy.failure_count,
                conv_id=from_conversation_id,
                success=success,
                user_id=user_id,
                channel=channel,
            )

        return strategy

    def find_strategies(
        self,
        task_description: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Strategy]:
        """
        Find strategies that worked for similar tasks.

        Args:
            task_description: Description of the task
            top_k: Number of strategies to return
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of Strategy objects
        """
        embedding = self.embedder.embed_single(task_description)

        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            result = session.run(
                cast(LiteralString, f"""
                CALL db.index.vector.queryNodes('strategy_embeddings', $k, $embedding)
                YIELD node AS s, score
                WHERE s.success_count > 0 {filters.build_inline()}
                RETURN s.id AS id,
                       s.description AS description,
                       s.context_pattern AS context_pattern,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       s.channel AS channel,
                       CASE WHEN (s.success_count + s.failure_count) > 0
                            THEN s.success_count * 1.0 / (s.success_count + s.failure_count)
                            ELSE 0.5
                       END AS success_rate,
                       score
                ORDER BY success_rate * score DESC
            """),
                k=top_k * 2,
                embedding=embedding,
                user_id=user_id,
                channel=channel,
            )

            strategies = []
            for record in result:
                strategies.append(
                    Strategy(
                        id=record["id"],
                        description=record["description"],
                        context_pattern=record["context_pattern"],
                        tool_sequence=record["tool_sequence"],
                        success_count=record["success_count"],
                        failure_count=record["failure_count"],
                    )
                )

            return strategies[:top_k]

    def reinforce_strategy(
        self,
        strategy_id: str,
        success: bool,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> bool:
        """
        Update strategy success/failure counts.

        Args:
            strategy_id: Strategy ID
            success: Whether execution was successful
            user_id: User ID (for validation)
            channel: Channel scope (validates strategy is accessible)

        Returns:
            True if strategy was found and updated, False otherwise
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            if success:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy {{id: $id}})
                    WHERE true {filters.build_inline()}
                    SET s.success_count = s.success_count + 1,
                        s.last_used = datetime()
                    RETURN s.id AS updated_id
                """,
                    ),
                    id=strategy_id,
                    user_id=user_id,
                    channel=channel,
                )
            else:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy {{id: $id}})
                    WHERE true {filters.build_inline()}
                    SET s.failure_count = s.failure_count + 1,
                        s.last_used = datetime()
                    RETURN s.id AS updated_id
                """,
                    ),
                    id=strategy_id,
                    user_id=user_id,
                    channel=channel,
                )

            record = result.single()
            return record is not None and record["updated_id"] is not None

    def get_tool_stats(
        self,
        task_type: Optional[str] = None,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get tool usage statistics, optionally filtered by task type.

        Args:
            task_type: Optional task type filter
            user_id: Filter by user ID
            channel: Filter by channel (searches channel + _global)

        Returns:
            List of tool statistics
        """
        with Neo4jConnection.session() as session:
            # Build filters using utility
            filters = CypherFilterBuilder("s")
            filters.add_user_filter(user_id).add_channel_filter(channel)

            if task_type:
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE s.context_pattern CONTAINS $task_type
                          {filters.build_inline()}
                    WITH t,
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures
                    RETURN t.name AS tool,
                           successes,
                           failures,
                           CASE WHEN (successes + failures) > 0
                                THEN successes * 1.0 / (successes + failures)
                                ELSE 0.5
                           END AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY success_rate DESC
                """,
                    ),
                    task_type=task_type,
                    user_id=user_id,
                    channel=channel,
                )
            else:
                # For global stats without task_type, still scope to user's strategies
                result = session.run(
                    cast(
                        LiteralString,
                        f"""
                    MATCH (s:Strategy)-[:USES_TOOL]->(t:Tool)
                    WHERE true {filters.build_inline()}
                    WITH t,
                         sum(s.success_count) AS successes,
                         sum(s.failure_count) AS failures,
                         count(DISTINCT s) AS strategy_count
                    RETURN t.name AS tool,
                           successes + failures AS usage_count,
                           successes AS success_count,
                           CASE WHEN (successes + failures) > 0
                                THEN successes * 1.0 / (successes + failures)
                                ELSE 0.5
                           END AS success_rate,
                           t.avg_latency_ms AS avg_latency
                    ORDER BY usage_count DESC
                """,
                    ),
                    user_id=user_id,
                    channel=channel,
                )

            return [dict(record) for record in result]

    def list_strategies(
        self, user_id: str, channel: str = "_global", offset: int = 0, limit: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List strategies with pagination.

        Args:
            user_id: User ID to filter by
            channel: Filter by channel (searches channel + _global)
            offset: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (strategies list, total count)
        """
        with Neo4jConnection.session() as session:
            # Build WHERE conditions
            conditions = ["s.user_id = $user_id"]

            # Channel filter - _all means no filter, otherwise search both specified channel and _global
            if channel == "_all":
                pass  # No channel filter - show all channels
            elif channel and channel != "_global":
                conditions.append("(s.channel = $channel OR s.channel = '_global')")
            else:
                conditions.append("s.channel = '_global'")

            where_clause = " AND ".join(conditions)

            # Get total count
            count_result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (s:Strategy)
                WHERE {where_clause}
                RETURN count(s) AS total
            """,
                ),
                user_id=user_id,
                channel=channel,
            )
            record = count_result.single()
            total = record["total"] if record else 0

            # Get paginated results with success rate
            result = session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (s:Strategy)
                WHERE {where_clause}
                RETURN s.id AS id,
                       s.description AS description,
                       s.tool_sequence AS tool_sequence,
                       s.success_count AS success_count,
                       s.failure_count AS failure_count,
                       CASE WHEN (s.success_count + s.failure_count) > 0
                            THEN s.success_count * 1.0 / (s.success_count + s.failure_count)
                            ELSE 0.5
                       END AS success_rate,
                       s.channel AS channel,
                       s.last_used AS last_used
                ORDER BY s.success_count DESC, s.last_used DESC
                SKIP $offset
                LIMIT $limit
            """,
                ),
                user_id=user_id,
                channel=channel,
                offset=offset,
                limit=limit,
            )

            strategies = []
            for record in result:
                strategy = convert_record_datetimes(dict(record))
                strategies.append(strategy)
            return strategies, total
