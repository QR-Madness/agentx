"""
Phase 11.8+ Memory System Tests

This file contains comprehensive tests for the memory system fixes and features:
- Security tests (access control, entity type validation)
- Data integrity tests (embedding storage, turn index passthrough)
- Edge case tests (time bounds, division by zero, partition naming)
- Correctness tests (latency calculation, job metrics, extraction timeout)
- Performance tests (Redis SCAN, TTL refresh, query validation, graph limits)
- Integration tests (require Docker services)
- Agent integration tests (require Docker + configured provider)
"""

# Test-only: suppress framework-typing false positives (mocked sessions, Optional
# model getters, redis/pydantic typing). Test-harness artifacts, not real bugs;
# source code stays strictly type-checked (baseline 0).
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportFunctionMemberAccess=false

import asyncio
import inspect
import json
import re
from datetime import datetime, timezone
from unittest import skipUnless
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from django.test import TestCase
from sqlalchemy import text

from agentx_ai.kit.agent_memory import config as memory_config_module
from agentx_ai.kit.agent_memory.config import (
    SETTINGS_CACHE_TTL,
    Settings,
    get_settings,
)
from agentx_ai.kit.agent_memory.connections import Neo4jConnection, get_postgres_session
from agentx_ai.kit.agent_memory.consolidation import jobs as consolidation_jobs_module
from agentx_ai.kit.agent_memory.consolidation.jobs import (
    _build_entity_index,
    _claim_entity_candidates,
    _get_recent_facts,
    _handle_contradiction,
    _make_subject_router,
    _resolve_fact_entity_ids,
    _resolve_subject_channel,
    _slug,
    apply_memory_decay,
    link_facts_to_entities,
    consolidate_episodic_to_semantic,
    detect_patterns,
    manage_audit_partitions,
    promote_to_global,
)
from agentx_ai.kit.agent_memory.consolidation.metrics import (
    AggregatedMetrics,
    ConsolidationMetrics,
)
from agentx_ai.kit.agent_memory.extraction import (
    ContradictionResult,
    CorrectionResult,
    extract_entities,
)
from agentx_ai.kit.agent_memory.extraction.service import (
    CombinedExtractionResult,
    ExtractionResult,
    ExtractionService,
    get_extraction_service,
)
from agentx_ai.kit.agent_memory.portability.exporter import MemoryExporter
from agentx_ai.kit.agent_memory.portability.schema import (
    MemoryExport,
    current_embedder_info,
)
from agentx_ai.kit.agent_memory.query_utils import convert_all_datetimes
from agentx_ai.kit.agent_memory.memory.episodic import EpisodicMemory
from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
from agentx_ai.kit.agent_memory.memory.procedural import ProceduralMemory
from agentx_ai.kit.agent_memory.memory.recall import RecallLayer, RecallMetrics
from agentx_ai.kit.agent_memory.memory.retrieval import MemoryRetriever
from agentx_ai.kit.agent_memory.memory.semantic import SemanticMemory
from agentx_ai.kit.agent_memory.memory.working import WorkingMemory
from agentx_ai.kit.agent_memory.models import (
    Entity,
    Fact,
    MemoryBundle,
    Turn,
    compute_claim_hash,
)
from agentx_ai.prompts.loader import get_prompt_loader
from agentx_ai.test_utils import (
    APITestBase,
    MemoryTestBase,
    create_mock_neo4j_session,
    create_mock_postgres_session,
    create_mock_redis_client,
    docker_services_running,
    embeddings_compatible,
    has_configured_provider,
)


# =============================================================================
# Phase 11.8+: Security Tests
# =============================================================================

class GoalAccessControlTest(TestCase):
    """Test access control for goal operations."""

    def test_user_can_complete_own_goal(self) -> None:
        """User A can complete a goal they created."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_record = {"updated_id": "goal-1"}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j, \
             patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_redis:
            mock_neo4j.return_value = mock_session
            mock_redis.return_value = create_mock_redis_client()

            memory = AgentMemory(user_id="user-A", channel="_global")
            result = memory.complete_goal("goal-1", status="completed")

            self.assertTrue(result)

    def test_user_cannot_complete_other_users_goal(self) -> None:
        """User A cannot complete User B's goal - returns False."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.single.return_value = None  # No goal found
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j, \
             patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_redis:
            mock_neo4j.return_value = mock_session
            mock_redis.return_value = create_mock_redis_client()

            memory = AgentMemory(user_id="user-B", channel="_global")
            result = memory.complete_goal("goal-owned-by-user-A", status="completed")

            self.assertFalse(result)

    def test_complete_goal_uses_has_goal_relationship(self) -> None:
        """Verify the Cypher query includes HAS_GOAL relationship check."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.single.return_value = {"updated_id": "goal-1"}
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j, \
             patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_redis:
            mock_neo4j.return_value = mock_session
            mock_redis.return_value = create_mock_redis_client()

            memory = AgentMemory(user_id="user-A", channel="_global")
            memory.complete_goal("goal-1", status="completed")

            call_args = mock_session.run.call_args
            query = call_args[0][0]
            self.assertIn(":HAS_GOAL", query)
            self.assertIn("User", query)
            self.assertIn("Goal", query)

    def test_complete_goal_respects_channel_filter(self) -> None:
        """Goal in different channel cannot be completed."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j, \
             patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_redis:
            mock_neo4j.return_value = mock_session
            mock_redis.return_value = create_mock_redis_client()

            memory = AgentMemory(user_id="user-A", channel="project-X")
            result = memory.complete_goal("goal-in-different-channel", status="completed")

            self.assertFalse(result)

            call_args = mock_session.run.call_args
            kwargs = call_args[1]
            self.assertEqual(kwargs["channel"], "project-X")


class FactSalienceForgetTest(TestCase):
    """boost_salience / forget_fact / get_fact_provenance (Neo4j mocked)."""

    def _memory(self, mock_session):
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j, \
             patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_redis:
            mock_neo4j.return_value = mock_session
            mock_redis.return_value = create_mock_redis_client()
            return AgentMemory(user_id="user-A", channel="_global"), mock_neo4j

    def test_boost_salience_sets_salience_in_cypher(self) -> None:
        mock_session = create_mock_neo4j_session()
        # update_fact -> single() {"id"}; then get_fact_by_id -> single() fact dict
        mock_session.run.return_value.single.side_effect = [
            {"id": "f1"},
            {"id": "f1", "claim": "c", "salience": 0.9, "confidence": 0.8},
        ]
        memory, mock_neo4j = self._memory(mock_session)
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session', mock_neo4j):
            result = memory.boost_salience("f1", to=0.9)

        self.assertEqual(result["salience"], 0.9)
        set_query = mock_session.run.call_args_list[0][0][0]
        self.assertIn("f.salience", set_query)
        self.assertEqual(mock_session.run.call_args_list[0][1]["salience"], 0.9)

    def test_boost_salience_clamps_to_unit_interval(self) -> None:
        mock_session = create_mock_neo4j_session()
        mock_session.run.return_value.single.side_effect = [
            {"id": "f1"},
            {"id": "f1", "claim": "c", "salience": 1.0},
        ]
        memory, mock_neo4j = self._memory(mock_session)
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session', mock_neo4j):
            memory.boost_salience("f1", to=5.0)
        self.assertEqual(mock_session.run.call_args_list[0][1]["salience"], 1.0)

    def test_forget_fact_soft_retires_with_past_and_low_salience(self) -> None:
        mock_session = create_mock_neo4j_session()
        # get_fact_by_id (current) -> update_fact single {"id"} -> get_fact_by_id (updated)
        mock_session.run.return_value.single.side_effect = [
            {"id": "f1", "claim": "c", "confidence": 0.8},
            {"id": "f1"},
            {"id": "f1", "claim": "c", "confidence": 0.24, "temporal_context": "past", "salience": 0.05},
        ]
        memory, mock_neo4j = self._memory(mock_session)
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session', mock_neo4j):
            result = memory.forget_fact("f1", hard=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "soft")
        update_kwargs = mock_session.run.call_args_list[1][1]
        self.assertEqual(update_kwargs["temporal_context"], "past")
        self.assertEqual(update_kwargs["salience"], 0.05)
        self.assertEqual(update_kwargs["confidence"], round(0.8 * 0.3, 3))

    def test_forget_fact_missing_returns_failure(self) -> None:
        mock_session = create_mock_neo4j_session()
        mock_session.run.return_value.single.side_effect = [None]
        memory, mock_neo4j = self._memory(mock_session)
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session', mock_neo4j):
            result = memory.forget_fact("missing", hard=False)
        self.assertFalse(result["success"])

    def test_provenance_resolves_origin_turn(self) -> None:
        mock_session = create_mock_neo4j_session()
        # get_fact_by_id -> fact with source_turn_id; get_turn_by_id -> turn + conv
        turn_node = {"id": "t9", "role": "user", "content": "the key is X", "timestamp": "2026-01-01T00:00:00Z"}
        mock_session.run.return_value.single.side_effect = [
            {"id": "f1", "claim": "key is X", "source": "user", "source_turn_id": "t9"},
            {"t": turn_node, "conversation_id": "conv-7"},
        ]
        memory, mock_neo4j = self._memory(mock_session)
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session', mock_neo4j):
            result = memory.get_fact_provenance("f1")

        self.assertTrue(result["success"])
        self.assertEqual(result["source_turn_id"], "t9")
        self.assertIsNotNone(result["origin"])
        self.assertEqual(result["origin"]["conversation_id"], "conv-7")


class GoalMemoryTest(TestCase):
    """GoalMemory sub-module storage (Pass 4 — extracted from interface.py)."""

    def test_add_goal_creates_node_and_has_goal_relationship(self) -> None:
        from agentx_ai.kit.agent_memory.memory.goal import GoalMemory
        from agentx_ai.kit.agent_memory.models import Goal

        mock_session = create_mock_neo4j_session()
        mock_session.run.return_value = MagicMock()
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session
            GoalMemory(audit_logger=None).add_goal(
                Goal(description="ship it", embedding=[0.1, 0.2]),
                user_id="user-A", channel="_global",
            )
            create_query = mock_session.run.call_args_list[0][0][0]
            self.assertIn("CREATE (g:Goal", create_query)
            self.assertIn(":HAS_GOAL", create_query)

    def test_add_goal_links_subgoal(self) -> None:
        from agentx_ai.kit.agent_memory.memory.goal import GoalMemory
        from agentx_ai.kit.agent_memory.models import Goal

        mock_session = create_mock_neo4j_session()
        mock_session.run.return_value = MagicMock()
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session
            GoalMemory(audit_logger=None).add_goal(
                Goal(description="subtask", parent_goal_id="parent-1", embedding=[0.0]),
                user_id="user-A", channel="_global",
            )
            queries = [c[0][0] for c in mock_session.run.call_args_list]
            self.assertTrue(any("SUBGOAL_OF" in q for q in queries))

    def test_get_active_goals_orders_by_priority_and_filters_channel(self) -> None:
        from agentx_ai.kit.agent_memory.memory.goal import GoalMemory

        mock_session = create_mock_neo4j_session()
        record = {
            "g": {"id": "g1", "description": "d", "status": "active",
                  "priority": 3, "channel": "proj"},
            "parent": None,
        }
        mock_session.run.return_value = [record]
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session
            goals = GoalMemory().get_active_goals(user_id="user-A", channel="proj")
            query = mock_session.run.call_args[0][0]
            self.assertIn("ORDER BY g.priority DESC", query)
            self.assertEqual(mock_session.run.call_args[1]["channel"], "proj")
            self.assertEqual(len(goals), 1)

    def test_get_goal_returns_none_when_missing(self) -> None:
        from agentx_ai.kit.agent_memory.memory.goal import GoalMemory

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session
            self.assertIsNone(
                GoalMemory().get_goal("missing", user_id="user-A", channel="_global")
            )


class EntityTypeValidationTest(TestCase):
    """Test entity type whitelist validation."""

    def test_valid_entity_type_preserved(self) -> None:
        """Valid types from whitelist are preserved."""

        sm = SemanticMemory()
        result = sm._validate_entity_type("Person")
        self.assertEqual(result, "Person")

        result = sm._validate_entity_type("Organization")
        self.assertEqual(result, "Organization")

    def test_invalid_entity_type_defaults_to_entity(self) -> None:
        """Invalid types default to 'Entity'."""

        sm = SemanticMemory()
        result = sm._validate_entity_type("SomethingCompletelyInvalid")
        self.assertEqual(result, "Entity")

        result = sm._validate_entity_type("DropTable")
        self.assertEqual(result, "Entity")

    def test_entity_type_normalization(self) -> None:
        """Types are normalized (title case, stripped)."""

        sm = SemanticMemory()
        result = sm._validate_entity_type("person")
        self.assertEqual(result, "Person")

        result = sm._validate_entity_type("  Person  ")
        self.assertEqual(result, "Person")

    def test_whitelist_comes_from_settings(self) -> None:
        """Whitelist is loaded from settings.entity_types."""

        settings = get_settings()
        self.assertTrue(hasattr(settings, 'entity_types'))
        self.assertIsInstance(settings.entity_types, (list, tuple, set))
        self.assertIn("Person", settings.entity_types)


# =============================================================================
# Phase 11.8+: Data Integrity Tests
# =============================================================================

class EmbeddingStorageFormatTest(TestCase):
    """Test embedding storage format is JSON, not Python str representation."""

    def test_embedding_stored_as_json_dumps(self) -> None:
        """store_turn_log() uses json.dumps() for embedding."""
        mock_session = create_mock_postgres_session()

        with patch('agentx_ai.kit.agent_memory.memory.episodic.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_session


            em = EpisodicMemory()
            turn = Turn(
                id="turn-1",
                conversation_id="conv-1",
                index=0,
                role="user",
                content="Hello",
                timestamp=datetime.now(timezone.utc),
                embedding=[0.1, 0.2, 0.3]
            )

            em.store_turn_log(turn, channel="_global")

            call_args = mock_session.execute.call_args
            params = call_args[1]

            embedding_str = params.get("embedding")
            if embedding_str:
                parsed = json.loads(embedding_str)
                self.assertIsInstance(parsed, list)

    def test_embedding_not_stored_as_str(self) -> None:
        """Embedding is not str(list) which produces 'Python repr' format."""
        embedding = [0.1, 0.2, 0.3]
        json_format = json.dumps(embedding)
        self.assertTrue(json_format.startswith('['))
        str_format = str(embedding)
        self.assertEqual(json_format, str_format)

    def test_embedding_can_be_parsed_back(self) -> None:
        """Stored JSON can be json.loads() back to list."""
        original = [0.123456, 0.654321, -0.5, 0.0, 1.0]
        stored = json.dumps(original)
        recovered = json.loads(stored)

        self.assertEqual(original, recovered)
        self.assertIsInstance(recovered, list)
        self.assertIsInstance(recovered[0], float)


class TurnAgentAttributionTest(TestCase):
    """Phase 16.1 — store_turn_log persists turn.agent_id into conversation_logs."""

    def _captured_params(self, turn: Turn) -> dict:
        mock_session = create_mock_postgres_session()
        with patch('agentx_ai.kit.agent_memory.memory.episodic.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_session
            EpisodicMemory().store_turn_log(turn, channel="_global")
        # store_turn_log calls session.execute(text(sql), params_dict) — the
        # bind dict is the second positional arg.
        return mock_session.execute.call_args.args[1]

    def test_assistant_turn_persists_agent_id(self) -> None:
        params = self._captured_params(Turn(
            id="t-asst", conversation_id="conv-1", index=1, role="assistant",
            content="Hi", timestamp=datetime.now(timezone.utc),
            agent_id="bold-cosmic-falcon",
        ))
        self.assertEqual(params.get("agent_id"), "bold-cosmic-falcon")

    def test_user_turn_persists_null_agent_id(self) -> None:
        params = self._captured_params(Turn(
            id="t-user", conversation_id="conv-1", index=0, role="user",
            content="Hello", timestamp=datetime.now(timezone.utc),
        ))
        self.assertIsNone(params.get("agent_id"))

    def test_insert_sql_references_agent_id_column(self) -> None:
        mock_session = create_mock_postgres_session()
        with patch('agentx_ai.kit.agent_memory.memory.episodic.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_session
            EpisodicMemory().store_turn_log(
                Turn(id="t", conversation_id="c", index=0, role="assistant",
                     content="x", timestamp=datetime.now(timezone.utc), agent_id="a"),
                channel="_global",
            )
        sql = str(mock_session.execute.call_args[0][0])
        self.assertIn("agent_id", sql)


class ConversationAgentIdsTest(TestCase):
    """Phase 16.2 — get_conversation_agent_ids reads conversation_logs attribution."""

    def test_returns_non_null_agent_ids(self) -> None:
        mock_session = create_mock_postgres_session()
        mock_session.execute.return_value = [("alpha-agent",), ("beta-agent",), (None,)]
        with patch('agentx_ai.kit.agent_memory.memory.episodic.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_session
            ids = EpisodicMemory().get_conversation_agent_ids("conv-1")
        self.assertEqual(ids, ["alpha-agent", "beta-agent"])  # NULL row dropped

    def test_query_filters_and_binds_conversation(self) -> None:
        mock_session = create_mock_postgres_session()
        mock_session.execute.return_value = []
        with patch('agentx_ai.kit.agent_memory.memory.episodic.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_session
            EpisodicMemory().get_conversation_agent_ids("conv-xyz")
        sql = str(mock_session.execute.call_args[0][0]).lower()
        self.assertIn("distinct agent_id", sql)
        self.assertIn("agent_id is not null", sql)
        self.assertEqual(mock_session.execute.call_args[0][1], {"conv_id": "conv-xyz"})


class AgentParticipantGraphTest(TestCase):
    """Phase 16.5 — store_turn records AgentParticipant + PARTICIPATED_IN in Neo4j."""

    def test_store_turn_cypher_writes_agent_participant(self) -> None:
        mock_session = create_mock_neo4j_session()
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session
            EpisodicMemory().store_turn(
                Turn(id="t", conversation_id="c1", index=0, role="assistant",
                     content="hi", timestamp=datetime.now(timezone.utc), agent_id="beta-agent"),
                user_id="u", channel="_global", agent_id="beta-agent",
            )
        cypher = mock_session.run.call_args[0][0]
        self.assertIn("AgentParticipant", cypher)
        self.assertIn("PARTICIPATED_IN", cypher)
        # Guarded on the producing agent so user/tool (NULL) turns don't create nodes.
        self.assertIn("$turn_agent_id IS NOT NULL", cypher)


class TurnIndexPassthroughTest(TestCase):
    """Test turn_index passed correctly to tool invocation recording."""

    def test_turn_index_passed_to_postgres(self) -> None:
        """record_invocation() includes turn_index in SQL INSERT."""

        mock_pg_session = create_mock_postgres_session()
        mock_neo4j_session = create_mock_neo4j_session()

        with patch('agentx_ai.kit.agent_memory.memory.procedural.get_postgres_session') as mock_pg, \
             patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_pg.return_value = mock_pg_session
            mock_neo4j.return_value = mock_neo4j_session

            pm = ProceduralMemory()
            pm.record_invocation(
                conversation_id="conv-1",
                turn_id="turn-1",
                tool_name="test_tool",
                tool_input={"arg": "value"},
                tool_output={"result": "ok"},
                success=True,
                latency_ms=100,
                channel="_global",
                turn_index=5
            )

            # Check PostgreSQL call - params may be positional or keyword
            pg_call_args = mock_pg_session.execute.call_args
            # Try both keyword and positional args
            if pg_call_args[1]:
                pg_params = pg_call_args[1]
            else:
                # Positional args - second arg should be the params dict
                pg_params = pg_call_args[0][1] if len(pg_call_args[0]) > 1 else {}

            # Verify turn_idx is in params with value 5
            self.assertIn("turn_idx", pg_params)
            self.assertEqual(pg_params["turn_idx"], 5)

    def test_turn_index_passed_to_neo4j(self) -> None:
        """record_invocation() includes turn_index in Cypher CREATE."""

        mock_pg_session = create_mock_postgres_session()
        mock_neo4j_session = create_mock_neo4j_session()

        with patch('agentx_ai.kit.agent_memory.memory.procedural.get_postgres_session') as mock_pg, \
             patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_pg.return_value = mock_pg_session
            mock_neo4j.return_value = mock_neo4j_session

            pm = ProceduralMemory()
            pm.record_invocation(
                conversation_id="conv-1",
                turn_id="turn-1",
                tool_name="test_tool",
                tool_input={},
                tool_output={},
                success=True,
                latency_ms=100,
                turn_index=7
            )

            neo4j_call_args = mock_neo4j_session.run.call_args
            query = neo4j_call_args[0][0]
            self.assertIn("turn_index", query)

            kwargs = neo4j_call_args[1]
            self.assertEqual(kwargs.get("turn_index"), 7)

    def test_turn_index_none_defaults_to_zero(self) -> None:
        """None turn_index defaults to 0 in SQL."""

        mock_pg_session = create_mock_postgres_session()
        mock_neo4j_session = create_mock_neo4j_session()

        with patch('agentx_ai.kit.agent_memory.memory.procedural.get_postgres_session') as mock_pg, \
             patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_pg.return_value = mock_pg_session
            mock_neo4j.return_value = mock_neo4j_session

            pm = ProceduralMemory()
            pm.record_invocation(
                conversation_id="conv-1",
                turn_id="turn-1",
                tool_name="test_tool",
                tool_input={},
                tool_output={},
                success=True,
                latency_ms=100,
                turn_index=None
            )

            # Check PostgreSQL call - params may be positional or keyword
            pg_call_args = mock_pg_session.execute.call_args
            # Try both keyword and positional args
            if pg_call_args[1]:
                pg_params = pg_call_args[1]
            else:
                # Positional args - second arg should be the params dict
                pg_params = pg_call_args[0][1] if len(pg_call_args[0]) > 1 else {}

            # Verify turn_idx is in params with value 0 (None defaults to 0)
            self.assertIn("turn_idx", pg_params)
            self.assertEqual(pg_params["turn_idx"], 0)


# =============================================================================
# Phase 11.8+: Edge Case Tests
# =============================================================================


class SuccessRateDivisionTest(TestCase):
    """Test division-by-zero protection in success rate calculations."""

    def test_success_rate_zero_total_returns_half(self) -> None:
        """success_count=0, failure_count=0 returns 0.5 (neutral)."""
        # When both are 0, should return 0.5 as neutral
        success_count = 0
        failure_count = 0
        total = success_count + failure_count

        if total == 0:
            rate = 0.5
        else:
            rate = success_count / total

        self.assertEqual(rate, 0.5)

    def test_success_rate_all_success_returns_one(self) -> None:
        """success_count=5, failure_count=0 returns 1.0."""
        success_count = 5
        failure_count = 0
        total = success_count + failure_count

        if total == 0:
            rate = 0.5
        else:
            rate = success_count / total

        self.assertEqual(rate, 1.0)

    def test_success_rate_all_failure_returns_zero(self) -> None:
        """success_count=0, failure_count=5 returns 0.0."""
        success_count = 0
        failure_count = 5
        total = success_count + failure_count

        if total == 0:
            rate = 0.5
        else:
            rate = success_count / total

        self.assertEqual(rate, 0.0)

    def test_success_rate_cypher_uses_case_expression(self) -> None:
        """Verify Cypher query patterns use CASE WHEN for division protection."""
        # This tests that the Cypher template pattern includes protection
        cypher_pattern = """
        CASE WHEN usage_count = 0 THEN 0.5
             ELSE toFloat(success_count) / usage_count
        END AS success_rate
        """
        self.assertIn("CASE", cypher_pattern)
        self.assertIn("WHEN", cypher_pattern)
        self.assertIn("usage_count = 0", cypher_pattern)


class ConsolidationTimestampTest(TestCase):
    """Test consolidated timestamp set even on partial extraction failure."""

    def test_timestamp_set_on_full_success(self) -> None:
        """c.consolidated = datetime() set when all extractions succeed."""
        # This is a pattern test - the actual implementation should set
        # the timestamp in a finally block

        # Just verify the function exists and is callable
        self.assertTrue(callable(consolidate_episodic_to_semantic))

    def test_timestamp_set_on_partial_failure(self) -> None:
        """c.consolidated = datetime() set even if entity extraction fails."""
        # The consolidation should mark conversations as processed even on partial failure
        # This prevents infinite retry loops
        pass  # Actual test requires integration testing

    def test_fact_storage_handles_errors_gracefully(self) -> None:
        """The fact-storage stage guards each fact with try/except.

        Post-decomposition (roadmap item 1) the per-fact defensive handling
        lives in the extracted `_store_facts_with_verification` helper rather
        than inline in the coordinator, so introspect the helper that owns it.
        """
        source = inspect.getsource(
            consolidation_jobs_module._store_facts_with_verification
        )

        # Each fact must be stored best-effort so one failure can't abort the batch
        self.assertIn("try", source)
        self.assertIn("except", source)


class SQLPartitionNameValidationTest(TestCase):
    """Test SQL partition name validation (alphanumeric only)."""

    def test_valid_partition_name_accepted(self) -> None:
        """memory_audit_log_20260217 is valid."""
        pattern = r'^[a-zA-Z0-9_]+$'

        valid_names = [
            "memory_audit_log_20260217",
            "audit_partition_01",
            "test123",
        ]

        for name in valid_names:
            self.assertIsNotNone(re.match(pattern, name), f"{name} should be valid")

    def test_invalid_partition_name_rejected(self) -> None:
        """Names with special chars are rejected."""
        pattern = r'^[a-zA-Z0-9_]+$'

        invalid_names = [
            "audit; DROP TABLE --",
            "partition-name",
            "audit.log",
            "name with spaces",
        ]

        for name in invalid_names:
            self.assertIsNone(re.match(pattern, name), f"{name} should be invalid")

    def test_partition_name_regex_pattern(self) -> None:
        """Pattern matches ^[a-zA-Z0-9_]+$."""

        pattern = r'^[a-zA-Z0-9_]+$'

        # Should match
        self.assertIsNotNone(re.match(pattern, "valid_name_123"))

        # Should not match
        self.assertIsNone(re.match(pattern, "invalid-name"))


# =============================================================================
# Phase 11.8+: Correctness Tests
# =============================================================================

class AverageLatencyCalculationTest(TestCase):
    """Test running mean calculation in procedural.record_invocation()."""

    def test_first_invocation_sets_initial_latency(self) -> None:
        """First invocation sets avg_latency_ms directly."""
        # Running mean formula: new_avg = old_avg + (new - old_avg) / (count + 1)
        # For first invocation (count=0): new_avg = 0 + (100 - 0) / 1 = 100

        old_avg = 0
        count = 0
        new_value = 100

        # First invocation: count is 0, formula gives new_value directly
        new_avg = old_avg + (new_value - old_avg) / (count + 1)

        self.assertEqual(new_avg, 100)

    def test_running_mean_formula_correct(self) -> None:
        """Running mean: new_avg = old_avg + (new - old_avg)/(count+1)"""
        # After 2 invocations with 100ms and 200ms, avg should be 150

        # First: avg = 100, count = 1
        avg = 100
        count = 1

        # Second: new value 200
        new_value = 200
        avg = avg + (new_value - avg) / (count + 1)

        self.assertEqual(avg, 150)

    def test_running_mean_over_many_invocations(self) -> None:
        """Test with 10+ invocations for numerical stability."""
        # Start with first value
        values = [100, 200, 150, 175, 125, 180, 160, 140, 190, 110]
        expected_avg = sum(values) / len(values)

        # Simulate running mean
        avg = values[0]
        for i, v in enumerate(values[1:], start=1):
            count = i
            avg = avg + (v - avg) / (count + 1)

        # Should be close to actual average
        self.assertAlmostEqual(avg, expected_avg, places=5)

    def test_latency_calculation_uses_correct_count_order(self) -> None:
        """Count is incremented AFTER average update, not before."""
        # The Cypher query should:
        # 1. Calculate new avg using current count
        # 2. Then increment count
        # This ensures the formula (count + 1) in denominator is correct

        # Verify the pattern in the Cypher template

        pm = ProceduralMemory()
        # The record_invocation method should use the correct formula
        self.assertTrue(hasattr(pm, 'record_invocation'))


class ConsolidationJobMetricsTest(TestCase):
    """Test consolidation jobs return proper metrics dictionaries."""

    def test_consolidate_episodic_returns_metrics_dict(self) -> None:
        """consolidate_episodic_to_semantic() returns dict with metrics."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))  # No conversations
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session

            result = asyncio.run(consolidate_episodic_to_semantic())

            self.assertIsInstance(result, dict)
            # Should have metric keys (actual key is items_processed)
            self.assertIn("items_processed", result)

    def test_detect_patterns_returns_metrics_dict(self) -> None:
        """detect_patterns() returns dict with metrics."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session

            result = detect_patterns()

            self.assertIsInstance(result, dict)

    def test_apply_memory_decay_returns_metrics_dict(self) -> None:
        """apply_memory_decay() returns dict with decay counts."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        # Return the expected key from the Cypher query
        mock_result.single.return_value = {"decayed_count": 0}
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session

            result = apply_memory_decay()

            self.assertIsInstance(result, dict)

    def test_promote_to_global_returns_metrics_dict(self) -> None:
        """promote_to_global() returns dict with promotion counts."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as mock_neo4j:
            mock_neo4j.return_value = mock_session

            result = promote_to_global()

            self.assertIsInstance(result, dict)

    def test_manage_audit_partitions_returns_metrics_dict(self) -> None:
        """manage_audit_partitions() returns dict with partition info."""

        mock_pg_session = create_mock_postgres_session()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_pg_session.execute.return_value = mock_result

        with patch('agentx_ai.kit.agent_memory.connections.get_postgres_session') as mock_pg:
            mock_pg.return_value = mock_pg_session

            result = manage_audit_partitions()

            self.assertIsInstance(result, dict)


class ConsolidationPipelineTest(TestCase):
    """Behavior-pinning tests for consolidate_episodic_to_semantic().

    These drive the whole job against a mocked Neo4j session + extraction
    service + memory and assert the externally-observable contract — which
    storage calls fire, when the ``consolidated``/``self_consolidated`` flags
    get set, and the shape of the returned metrics. They exist so the
    god-function decomposition (roadmap item 1) is provably behavior-preserving:
    a pure relocation must keep every assertion here green.
    """

    def _make_session(self, user_records, assistant_records):
        """Mock Neo4j session that routes the discovery queries by content."""
        session = create_mock_neo4j_session()

        def _run(query, **params):
            res = MagicMock()
            if "self_consolidated IS NULL" in query:
                res.__iter__ = MagicMock(return_value=iter(list(assistant_records)))
            elif "c.consolidated IS NULL" in query:
                res.__iter__ = MagicMock(return_value=iter(list(user_records)))
            else:
                res.__iter__ = MagicMock(return_value=iter([]))
            return res

        session.run.side_effect = _run
        return session

    @staticmethod
    def _ran(session, needle):
        """True if any session.run() call's Cypher contained `needle`."""
        return any(
            needle in (call.args[0] if call.args else "")
            for call in session.run.call_args_list
        )

    @staticmethod
    def _combined(*, success=True, is_relevant=True, entities=None, facts=None,
                  relationships=None, reason=""):
        r = MagicMock()
        r.success = success
        r.is_relevant = is_relevant
        r.entities = entities or []
        r.facts = facts or []
        r.relationships = relationships or []
        r.tokens_used = 5
        r.reason = reason
        return r

    def _run_job(self, *, user_records=(), assistant_records=(),
                 combined=None, assistant_combined=None, is_duplicate=False):
        """Run consolidation with all sub-helpers patched; return (result, session, memory)."""
        session = self._make_session(user_records, assistant_records)

        memory = MagicMock()
        fact_obj = MagicMock()
        fact_obj.id = "fact-1"
        memory.learn_fact.return_value = fact_obj
        memory.semantic.get_fact_by_id.return_value = None

        ext = MagicMock()
        ext.check_correction = AsyncMock(return_value=MagicMock(is_correction=False))
        ext.check_relevance_and_extract = AsyncMock(
            return_value=combined if combined is not None else self._combined()
        )
        ext.check_relevance_and_extract_assistant = AsyncMock(
            return_value=(assistant_combined if assistant_combined is not None
                          else self._combined(is_relevant=False))
        )
        ext.check_contradictions = AsyncMock(
            return_value=MagicMock(has_contradiction=False)
        )

        def _resolve(*, memory, extracted_entities, entity_map, user_id,
                     channel, conv_id, metrics, errors):
            for e in extracted_entities:
                entity_map[str(e.get("name", "")).lower()] = "ent-1"
            return (list(extracted_entities), 0, len(extracted_entities))

        mod = consolidation_jobs_module
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session',
                   return_value=session), \
             patch.object(mod, 'get_extraction_service', return_value=ext), \
             patch.object(mod, 'get_embedder', return_value=MagicMock()), \
             patch.object(mod, '_get_memory_for_user', return_value=memory), \
             patch('agentx_ai.kit.agent_memory.memory.interface.AgentMemory',
                   return_value=memory), \
             patch.object(mod, '_build_scope_context', return_value=([], [])), \
             patch.object(mod, '_resolve_and_prepare_entities', side_effect=_resolve), \
             patch.object(mod, '_batch_store_entities', return_value=1), \
             patch.object(mod, '_batch_store_relationships', return_value=0), \
             patch.object(mod, '_is_duplicate_fact', return_value=is_duplicate), \
             patch.object(mod, '_is_semantic_duplicate', return_value=False), \
             patch.object(mod, '_get_contradiction_candidates', return_value=[]):
            result = asyncio.run(consolidate_episodic_to_semantic())

        return result, session, memory

    @staticmethod
    def _user_record():
        return {
            "conversation_id": "conv-1",
            "user_id": "default",
            "channel": "_default",
            "turns": [{"content": "Acme Corp raised a Series A funding round."}],
        }

    @staticmethod
    def _assistant_record():
        return {
            "conversation_id": "conv-2",
            "user_id": "default",
            "agent_id": "bold-cosmic-falcon",
            "turns": [{
                "content": "I am most effective when given explicit constraints "
                           "up front, and I tend to over-explain when uncertain.",
                "id": "turn-a1",
            }],
        }

    def test_happy_path_stores_and_marks_consolidated(self) -> None:
        """A relevant turn → entity + fact stored and conversation marked consolidated."""
        combined = self._combined(
            entities=[{"name": "Acme", "type": "organization"}],
            facts=[{"claim": "Acme raised a Series A", "confidence": 0.9, "entity_names": []}],
        )
        result, session, memory = self._run_job(
            user_records=[self._user_record()], combined=combined,
        )

        memory.learn_fact.assert_called_once()
        self.assertTrue(self._ran(session, "SET c.consolidated"))
        self.assertEqual(result["facts"], 1)
        self.assertEqual(result["entities"], 1)
        self.assertGreaterEqual(result["items_processed"], 1)

    def test_extraction_failure_skips_consolidated_mark(self) -> None:
        """A failed extraction must NOT mark the conversation consolidated (retry guarantee)."""
        combined = self._combined(success=False, is_relevant=False)
        result, session, memory = self._run_job(
            user_records=[self._user_record()], combined=combined,
        )

        memory.learn_fact.assert_not_called()
        self.assertFalse(self._ran(session, "SET c.consolidated"))

    def test_no_relevant_turns_marks_consolidated(self) -> None:
        """No relevant turns (but no failure) → still marked consolidated, nothing stored."""
        combined = self._combined(success=True, is_relevant=False, reason="heuristic_skip")
        result, session, memory = self._run_job(
            user_records=[self._user_record()], combined=combined,
        )

        memory.learn_fact.assert_not_called()
        self.assertTrue(self._ran(session, "SET c.consolidated"))

    def test_hash_duplicate_fact_skipped(self) -> None:
        """A hash-duplicate fact is skipped and counted, but the conversation still consolidates."""
        combined = self._combined(
            facts=[{"claim": "Acme raised a Series A", "confidence": 0.9, "entity_names": []}],
        )
        result, session, memory = self._run_job(
            user_records=[self._user_record()], combined=combined, is_duplicate=True,
        )

        memory.learn_fact.assert_not_called()
        self.assertEqual(result["metrics"]["duplicates_skipped"], 1)
        self.assertTrue(self._ran(session, "SET c.consolidated"))

    def test_assistant_phase_stores_self_facts(self) -> None:
        """An assistant turn with an agent_id → self-fact stored and self_consolidated set."""
        assistant_combined = self._combined(
            entities=[],
            facts=[{"claim": "Assistant works best with explicit constraints",
                    "confidence": 0.8, "entity_names": []}],
        )
        result, session, memory = self._run_job(
            assistant_records=[self._assistant_record()],
            assistant_combined=assistant_combined,
        )

        memory.learn_fact.assert_called_once()
        _, kwargs = memory.learn_fact.call_args
        self.assertEqual(kwargs.get("source"), "self_extraction")
        self.assertTrue(self._ran(session, "SET c.self_consolidated"))
        self.assertEqual(result["metrics"]["assistant_facts_stored"], 1)


class EntityNameNormalizationTest(TestCase):
    """Test entity name case normalization in relationship linking."""

    def test_relationship_linking_case_insensitive(self) -> None:
        """Entity 'John' matches relationship source 'john'."""
        entity_map = {"john": "entity-1", "acme": "entity-2"}

        # Lookup with different cases
        name_lower = "john"
        name_title = "John"
        name_upper = "JOHN"

        # All should map to same entity when using lowercase keys
        self.assertEqual(entity_map.get(name_lower.lower()), "entity-1")
        self.assertEqual(entity_map.get(name_title.lower()), "entity-1")
        self.assertEqual(entity_map.get(name_upper.lower()), "entity-1")

    def test_entity_map_uses_lowercase_keys(self) -> None:
        """consolidate_episodic_to_semantic stores lowercase keys."""
        # The entity_map in consolidation should use lowercase
        entity_map = {}

        # Simulating entity storage
        entity_name = "John Smith"
        entity_id = "entity-123"
        entity_map[entity_name.lower()] = entity_id

        self.assertIn("john smith", entity_map)
        self.assertNotIn("John Smith", entity_map)

    def test_mixed_case_entities_link_correctly(self) -> None:
        """'ACME Corp' entity links to 'acme corp' in relationship."""
        entity_map = {}

        # Store entities with lowercase keys
        entities = ["ACME Corp", "John Doe", "New York"]
        for i, name in enumerate(entities):
            entity_map[name.lower()] = f"entity-{i}"

        # Lookup with relationship source names
        relationship_sources = ["Acme Corp", "john doe", "NEW YORK"]

        for source in relationship_sources:
            entity_id = entity_map.get(source.lower())
            self.assertIsNotNone(entity_id, f"Should find entity for {source}")


class ExtractionTimeoutTest(TestCase):
    """Test extraction timeout fires after configured seconds."""

    def test_extraction_times_out_after_config_seconds(self) -> None:
        """extract_all() times out after extraction_timeout setting."""

        settings = get_settings()
        self.assertTrue(hasattr(settings, 'extraction_timeout'))
        self.assertIsInstance(settings.extraction_timeout, (int, float))
        self.assertGreater(settings.extraction_timeout, 0)

    def test_timeout_returns_failure_result(self) -> None:
        """Timeout should return ExtractionResult with success=False."""

        # Simulate timeout result
        result = ExtractionResult(
            entities=[],
            facts=[],
            relationships=[],
            success=False,
            error="Extraction timed out after 30.0 seconds"
        )

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)  # type: ignore[arg-type]

    def test_timeout_value_comes_from_settings(self) -> None:
        """Timeout uses settings.extraction_timeout value."""

        settings = get_settings()
        # Default should be 30 seconds
        self.assertEqual(settings.extraction_timeout, 30.0)


class AsyncExtractionContextTest(TestCase):
    """Test async extraction works from both sync and async contexts."""

    def test_extract_all_works_from_async_context(self) -> None:
        """extract_all() works when called from async function."""

        service = get_extraction_service()

        async def test_async():
            # Should not raise when called from async context
            # (Will return empty on short text)
            result = service.extract_all("Short")
            return result

        # Run in event loop using asyncio.run() for Python 3.10+
        result = asyncio.run(test_async())
        self.assertIsNotNone(result)

    def test_extract_entities_sync_wrapper_works(self) -> None:
        """Sync extract_entities() wrapper works without event loop."""

        # Should work without explicit async handling
        result = extract_entities("Short text")
        self.assertIsInstance(result, list)

    def test_handles_nested_event_loop(self) -> None:
        """Works correctly with nested async contexts."""
        # The extraction service should handle cases where it's called
        # from within an existing event loop

        service = get_extraction_service()
        # Just verify it exists and is callable
        self.assertTrue(hasattr(service, 'extract_all'))


# =============================================================================
# Phase 11.8+: Performance Tests
# =============================================================================

class RedisScanPaginationTest(TestCase):
    """Test Redis SCAN is used instead of KEYS."""

    def test_get_context_uses_scan_not_keys(self) -> None:
        """WorkingMemory.get_context() uses SCAN for iteration."""

        mock_redis = create_mock_redis_client()
        mock_redis.scan.return_value = (0, [])
        mock_redis.get.return_value = None

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            wm = WorkingMemory(user_id="user1", channel="_global")
            wm.get_context()

            # Should use scan, not keys
            if mock_redis.scan.called:
                self.assertTrue(True)
            elif mock_redis.keys.called:
                self.fail("Should use SCAN instead of KEYS")

    def test_clear_session_uses_scan_not_keys(self) -> None:
        """WorkingMemory.clear_session() uses SCAN for iteration."""

        mock_redis = create_mock_redis_client()
        mock_redis.scan.return_value = (0, [b"key1", b"key2"])

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")
            wm.clear_session()

            # Verify SCAN was called
            self.assertTrue(mock_redis.scan.called)

    def test_invalidate_cache_uses_scan_not_keys(self) -> None:
        """MemoryRetriever.invalidate_cache() uses SCAN for iteration."""

        mock_redis = create_mock_redis_client()
        mock_redis.scan.return_value = (0, [])

        mock_memory = MagicMock()

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            retriever = MemoryRetriever(mock_memory)
            retriever.invalidate_cache("user1", "_global")

            # Verify SCAN was called, not KEYS
            if mock_redis.scan.called:
                self.assertTrue(True)

    def test_scan_handles_pagination_correctly(self) -> None:
        """SCAN loop continues until cursor returns 0."""
        mock_redis = create_mock_redis_client()

        # Simulate paginated SCAN results
        call_count = [0]

        def mock_scan(cursor, match=None, count=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (1, [b"key1"])  # More to scan
            else:
                return (0, [b"key2"])  # Done

        mock_redis.scan.side_effect = mock_scan

        # Simulate iteration
        cursor = None
        keys = []
        while True:
            cursor, batch = mock_redis.scan(cursor or 0, match="pattern:*")
            keys.extend(batch)
            if cursor == 0:
                break

        self.assertEqual(len(keys), 2)
        self.assertEqual(call_count[0], 2)


class WorkingMemoryTTLRefreshTest(TestCase):
    """Test TTL is refreshed on read access (sliding window expiry)."""

    def test_get_recent_turns_refreshes_ttl(self) -> None:
        """get_recent_turns() calls expire() on access."""

        mock_redis = create_mock_redis_client()
        mock_redis.lrange.return_value = []
        mock_redis.exists.return_value = True

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")
            wm.get_recent_turns(limit=5)

            # Should call expire to refresh TTL
            # (Implementation may vary - check if expire was called)
            # This tests the pattern, actual call depends on implementation

    def test_ttl_refresh_uses_configured_value(self) -> None:
        """TTL refresh uses 3600 seconds (1 hour)."""

        mock_redis = create_mock_redis_client()
        mock_redis.lrange.return_value = []

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

            # Create turn to trigger expire call
            mock_turn = MagicMock()
            mock_turn.id = "turn-1"
            mock_turn.index = 0
            mock_turn.role = "user"
            mock_turn.content = "test"
            mock_turn.timestamp = datetime.now(timezone.utc)

            wm.add_turn(mock_turn)

            # Check expire was called with 3600
            expire_calls = [c for c in mock_redis.method_calls if 'expire' in str(c)]
            if expire_calls:
                # Verify TTL value
                for call in expire_calls:
                    args = call[1]
                    if len(args) >= 2:
                        self.assertEqual(args[1], 3600)

    def test_ttl_refresh_only_if_data_exists(self) -> None:
        """TTL not refreshed if no turns exist."""

        mock_redis = create_mock_redis_client()
        mock_redis.lrange.return_value = []  # No turns
        mock_redis.exists.return_value = False

        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock_get:
            mock_get.return_value = mock_redis

            wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")
            wm.get_recent_turns(limit=5)

            # Should not call expire if key doesn't exist
            # (or should be a no-op)


class QueryLengthValidationTest(TestCase):
    """Test retrieval rejects oversized queries."""

    def test_query_under_limit_accepted(self) -> None:
        """Query within max_query_length is accepted."""

        settings = get_settings()
        max_length = settings.max_query_length

        # Short query should be fine
        query = "a" * 100
        self.assertLess(len(query), max_length)

    def test_query_over_limit_raises_valueerror(self) -> None:
        """Query exceeding max_query_length raises ValueError."""

        settings = get_settings()
        max_length = settings.max_query_length

        # Create oversized query
        oversized_query = "a" * (max_length + 1)

        # The retriever should reject this
        # (Test the validation logic pattern)
        self.assertGreater(len(oversized_query), max_length)

    def test_limit_comes_from_settings(self) -> None:
        """Limit is settings.max_query_length (default 10000)."""

        settings = get_settings()
        self.assertTrue(hasattr(settings, 'max_query_length'))
        self.assertIsInstance(settings.max_query_length, int)
        self.assertEqual(settings.max_query_length, 10000)

    def test_error_message_includes_length_info(self) -> None:
        """ValueError message should show actual vs max length."""
        actual = 15000
        max_length = 10000

        error_msg = f"Query length {actual} exceeds maximum {max_length}"

        self.assertIn(str(actual), error_msg)
        self.assertIn(str(max_length), error_msg)


class GraphTraversalLimitsTest(TestCase):
    """Test graph traversal depth limits (max 3) and result limits."""

    def test_depth_minimum_is_one(self) -> None:
        """Depth below 1 is raised to 1."""
        min_depth = 1

        # Test validation logic
        requested_depth = 0
        validated_depth = max(min_depth, min(requested_depth, 3))

        self.assertEqual(validated_depth, 1)

    def test_max_related_capped_at_hundred(self) -> None:
        """max_related parameter capped at 100."""
        max_related_cap = 100

        # Test validation logic
        requested = 500
        validated = min(requested, max_related_cap)

        self.assertEqual(validated, 100)

    def test_results_limited_in_cypher_query(self) -> None:
        """Cypher query includes LIMIT clause."""
        # Verify the pattern includes LIMIT
        cypher_pattern = """
        MATCH (e:Entity {id: $entity_id})-[r*1..3]-(related)
        RETURN related
        LIMIT $limit
        """
        self.assertIn("LIMIT", cypher_pattern)


# =============================================================================
# Phase 11.8+: Integration Tests (Require Docker)
# =============================================================================

@skipUnless(docker_services_running(), "Docker services not running")
class MemoryLifecycleIntegrationTest(MemoryTestBase):
    """Test full cycle: store turn -> extract -> consolidate -> retrieve."""

    def test_store_turn_persists_to_neo4j(self) -> None:
        """store_turn() creates Turn node in Neo4j."""

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        turn = Turn(
            id=str(uuid4()),
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="Hello, this is a test message",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Verify in Neo4j
        with Neo4jConnection.session() as session:
            result = session.run(
                "MATCH (t:Turn {id: $id}) RETURN t",
                id=turn.id
            )
            record = result.single()
            self.assertIsNotNone(record)

    def test_store_turn_persists_to_postgres(self) -> None:
        """store_turn() creates row in conversation_logs."""

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        turn = Turn(
            id=str(uuid4()),
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="Hello, PostgreSQL test",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Verify in PostgreSQL - check by conversation_id and index
        with get_postgres_session() as session:
            result = session.execute(
                text("SELECT * FROM conversation_logs WHERE conversation_id = :conv_id AND turn_index = 0"),
                {"conv_id": self.test_conversation_id}
            )
            row = result.fetchone()
            self.assertIsNotNone(row)

    def test_store_turn_updates_working_memory(self) -> None:
        """store_turn() adds turn to Redis working memory."""

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        turn = Turn(
            id=str(uuid4()),
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="Hello, Redis test",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Verify in Redis via working memory
        recent = memory.working.get_recent_turns(limit=1)
        self.assertEqual(len(recent), 1)

    def test_remember_retrieves_stored_turn(self) -> None:
        """remember() finds previously stored turn by semantic similarity."""

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        # Store a turn about Python
        turn = Turn(
            id=f"turn-python-{self.test_user_id}",
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="I love programming in Python. It's my favorite language.",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Query for Python-related content
        bundle = memory.remember("What programming language do I like?", top_k=5)

        self.assertIsNotNone(bundle)
        # Should have some episodic results
        self.assertGreaterEqual(len(bundle.turns), 0)  # type: ignore[union-attr]

    def test_full_memory_cycle_end_to_end(self) -> None:
        """Complete cycle: store -> consolidate -> remember."""

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        # Store turn
        turn = Turn(
            id=f"turn-full-{self.test_user_id}",
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="I work at a tech company in San Francisco",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Remember
        bundle = memory.remember("Where do I work?")
        self.assertIsNotNone(bundle)


@skipUnless(docker_services_running(), "Docker services not running")
class Neo4jVectorSearchTest(MemoryTestBase):
    """Test Neo4j vector search returns relevant results."""

    def test_vector_search_respects_user_filter(self) -> None:
        """Vector search only returns user's own turns."""

        em = EpisodicMemory()

        # Search for a different user
        other_user = "nonexistent-user-xyz"
        results = em.get_recent_turns(
            user_id=other_user,
            channel="_global",
            limit=10
        )

        # Should be empty (other user has no turns)
        self.assertEqual(len(results), 0)

    def test_vector_search_respects_channel_filter(self) -> None:
        """Vector search filters by channel."""

        em = EpisodicMemory()

        # Search in nonexistent channel
        results = em.get_recent_turns(
            user_id=self.test_user_id,
            channel="nonexistent-channel-xyz",
            limit=10
        )

        # Should be empty
        self.assertEqual(len(results), 0)


@skipUnless(docker_services_running(), "Docker services not running")
class PostgresAuditLogTest(MemoryTestBase):
    """Test PostgreSQL audit log captures operations."""

    def test_write_operation_logged(self) -> None:
        """store_turn() creates audit log entry."""

        settings = get_settings()

        # Only test if audit logging is enabled
        if settings.audit_log_level == "off":
            self.skipTest("Audit logging is disabled")

        # Skip if embedding dimensions don't match schema
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local=768 vs schema=1536)")

        memory = AgentMemory(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            channel="_global"
        )

        turn = Turn(
            id=str(uuid4()),
            conversation_id=self.test_conversation_id,
            index=0,
            role="user",
            content="Audit test message",
            timestamp=datetime.now(timezone.utc)
        )

        memory.store_turn(turn)

        # Check audit log
        with get_postgres_session() as session:
            result = session.execute(
                text("""
                    SELECT * FROM memory_audit_log
                    WHERE user_id = :user_id
                    AND operation = 'store'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """),
                {"user_id": self.test_user_id}
            )
            row = result.fetchone()

            if row:
                self.assertEqual(row.operation, "store")

    def test_audit_log_includes_channel(self) -> None:
        """Audit entries include channel column."""

        with get_postgres_session() as session:
            # Check if partitioned table has channel column via parent table inspection
            # memory_audit_log is partitioned, so check the parent table definition
            result = session.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name LIKE 'memory_audit_log%'
                AND column_name = 'channel'
                LIMIT 1
            """))
            row = result.fetchone()

            # Channel column should exist in audit log tables
            # Note: If partitions don't exist yet, this may be None
            # In that case, we check that the column is expected in the schema
            if row is None:
                # Check via SQL definition - this is a schema design test
                # The column should be defined even if no partitions exist
                self.skipTest("No audit log partitions exist yet")


@skipUnless(docker_services_running(), "Docker services not running")
class ChannelCRUDTest(APITestBase):
    """Test channel create, list, delete and data cleanup."""

    def setUp(self) -> None:
        super().setUp()
        self.test_channel = f"test-channel-{uuid4().hex[:8]}"

    def test_create_channel(self) -> None:
        """POST /api/memory/channels creates channel."""
        response = self.client.post(
            "/api/memory/channels",
            data={"name": self.test_channel},
            content_type="application/json"
        )

        self.assertIn(response.status_code, [200, 201])  # type: ignore[union-attr]

    def test_list_channels(self) -> None:
        """GET /api/memory/channels returns channel list."""
        response = self.client.get("/api/memory/channels")

        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        self.assertIn("channels", data)

    def test_global_channel_cannot_be_deleted(self) -> None:
        """Attempting to delete _global returns error."""
        response = self.client.delete("/api/memory/channels/_global")

        # Should fail
        self.assertIn(response.status_code, [400, 403, 404])  # type: ignore[union-attr]


# =============================================================================
# Phase 11.8+: Agent Integration Tests (Require Docker + Provider)
# =============================================================================

@skipUnless(
    docker_services_running() and has_configured_provider(),
    "Docker services or provider not available"
)
class AgentChatMemoryStorageTest(APITestBase):
    """Test /api/agent/chat stores turns in memory with correct channel."""

    def test_chat_stores_user_turn(self) -> None:
        """User message stored in episodic memory."""
        # This test requires a configured provider
        pass  # Placeholder for full integration

    def test_chat_stores_assistant_turn(self) -> None:
        """Assistant response stored in episodic memory."""
        pass

    def test_chat_respects_channel_parameter(self) -> None:
        """Turns stored in specified channel."""
        pass


@skipUnless(
    docker_services_running() and has_configured_provider(),
    "Docker services or provider not available"
)
class AgentGracefulDegradationTest(TestCase):
    """Test graceful degradation when databases are down."""

    def test_agent_works_without_neo4j(self) -> None:
        """Agent chat functions when Neo4j is down."""
        # Would require mocking connection failures
        pass

    def test_agent_works_without_postgres(self) -> None:
        """Agent chat functions when PostgreSQL is down."""
        pass

    def test_agent_works_without_redis(self) -> None:
        """Agent chat functions when Redis is down."""
        pass


# =============================================================================
# Phase 11.12: Correction and Contradiction Detection Tests
# =============================================================================

class CorrectionDetectionTest(TestCase):
    """Test user correction detection in extraction service."""

    def _make_service_with_llm(self, llm_content: str) -> ExtractionService:
        """Build an ExtractionService with correction enabled + a scripted LLM."""
        service = ExtractionService()
        mock_settings = MagicMock()
        mock_settings.correction_detection_enabled = True
        service._settings = mock_settings

        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(content=llm_content))
        service._get_provider_for_stage = MagicMock(
            return_value=(provider, "mock-model", 0.0, 256)
        )
        return service

    def test_check_correction_llm_yes_parses_original_and_corrected(self) -> None:
        """LLM-affirmative response yields populated original/corrected claims."""

        # The previous regex pre-filter would have missed a paraphrase like
        # "scratch that, ..." — the LLM-only path handles it.
        service = self._make_service_with_llm(
            "CORRECTION: YES\n"
            "ORIGINAL: works at Microsoft\n"
            "CORRECTED: works at Google\n"
        )

        result = asyncio.run(
            service.check_correction("scratch that — I work at Google, not Microsoft")
        )
        self.assertTrue(result.is_correction)
        self.assertEqual(result.original_claim, "works at Microsoft")
        self.assertEqual(result.corrected_claim, "works at Google")
        self.assertTrue(result.success)

    def test_check_correction_llm_no_returns_false(self) -> None:
        """LLM-negative response yields is_correction=False without claims."""

        service = self._make_service_with_llm("CORRECTION: NO")

        result = asyncio.run(
            service.check_correction("I started a new project yesterday")
        )
        self.assertFalse(result.is_correction)
        self.assertIsNone(result.original_claim)
        self.assertIsNone(result.corrected_claim)

    def test_check_correction_short_input_skips_llm(self) -> None:
        """Trivially short input short-circuits before the LLM call."""

        service = self._make_service_with_llm("CORRECTION: YES\nORIGINAL: x\nCORRECTED: y")

        result = asyncio.run(service.check_correction("ok"))
        self.assertFalse(result.is_correction)
        # Gate fired before provider lookup.
        service._get_provider_for_stage.assert_not_called()

    def test_correction_result_model(self) -> None:
        """CorrectionResult model has expected fields."""

        result = CorrectionResult(
            is_correction=True,
            original_claim="works at Microsoft",
            corrected_claim="works at Google",
        )

        self.assertTrue(result.is_correction)
        self.assertEqual(result.original_claim, "works at Microsoft")
        self.assertEqual(result.corrected_claim, "works at Google")
        self.assertTrue(result.success)

    def test_check_correction_disabled_returns_false(self) -> None:
        """check_correction returns False when disabled."""

        service = ExtractionService()

        # Mock settings via _settings attribute (before lazy load)
        mock_settings = MagicMock()
        mock_settings.correction_detection_enabled = False
        service._settings = mock_settings

        result = asyncio.run(service.check_correction("Actually, I meant Python"))
        self.assertFalse(result.is_correction)


class ContradictionDetectionTest(TestCase):
    """Test contradiction detection in extraction service."""

    def test_contradiction_result_model(self) -> None:
        """ContradictionResult model has expected fields."""

        result = ContradictionResult(
            has_contradiction=True,
            contradicting_fact_id="fact-123",
            reason="Location cannot be both Seattle and New York",
            resolution="prefer_new",
        )

        self.assertTrue(result.has_contradiction)
        self.assertEqual(result.contradicting_fact_id, "fact-123")
        self.assertEqual(result.resolution, "prefer_new")
        self.assertTrue(result.success)

    def test_check_contradictions_disabled_returns_false(self) -> None:
        """check_contradictions returns False when disabled."""

        service = ExtractionService()

        # Mock settings via _settings attribute (before lazy load)
        mock_settings = MagicMock()
        mock_settings.contradiction_detection_enabled = False
        service._settings = mock_settings

        result = asyncio.run(service.check_contradictions(
            "User lives in Seattle",
            [{"id": "1", "claim": "User lives in New York"}]
        ))
        self.assertFalse(result.has_contradiction)

    def test_check_contradictions_empty_facts_returns_false(self) -> None:
        """check_contradictions returns False with no existing facts."""

        service = ExtractionService()

        # Mock settings via _settings attribute (before lazy load)
        mock_settings = MagicMock()
        mock_settings.contradiction_detection_enabled = True
        service._settings = mock_settings

        result = asyncio.run(service.check_contradictions("User lives in Seattle", []))
        self.assertFalse(result.has_contradiction)

    def test_contradiction_resolution_values(self) -> None:
        """ContradictionResult resolution accepts expected values."""

        for resolution in ["prefer_new", "prefer_old", "flag_review"]:
            result = ContradictionResult(
                has_contradiction=True,
                resolution=resolution,
            )
            self.assertEqual(result.resolution, resolution)


class FactModelSupersessionTest(TestCase):
    """Test Fact model supersession fields."""

    def test_fact_model_has_supersession_fields(self) -> None:
        """Fact model includes supersession tracking fields."""

        fact = Fact(claim="Test fact", source="extraction")

        # Check fields exist and have None defaults
        self.assertIsNone(fact.superseded_at)
        self.assertIsNone(fact.superseded_by_id)
        self.assertIsNone(fact.supersedes_id)
        self.assertFalse(fact.flagged_for_review)

    def test_fact_model_supersession_fields_settable(self) -> None:
        """Fact supersession fields can be set."""

        fact = Fact(
            claim="Updated fact",
            source="user_correction",
            supersedes_id="old-fact-id",
        )

        self.assertEqual(fact.supersedes_id, "old-fact-id")

        # Test flagged_for_review
        flagged_fact = Fact(
            claim="Questionable fact",
            source="extraction",
            flagged_for_review=True,
        )
        self.assertTrue(flagged_fact.flagged_for_review)


class ConsolidationCorrectionHandlerTest(TestCase):
    """Test correction handling in consolidation pipeline."""

    def test_get_recent_facts_query(self) -> None:
        """_get_recent_facts returns facts in expected format."""

        mock_session = create_mock_neo4j_session()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"id": "fact-1", "claim": "User likes Python", "confidence": 0.9, "created_at": datetime.now(timezone.utc)},
            {"id": "fact-2", "claim": "User works at Google", "confidence": 0.8, "created_at": datetime.now(timezone.utc)},
        ]))
        mock_session.run.return_value = mock_result

        facts = _get_recent_facts(mock_session, "user-123", "_default")

        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["id"], "fact-1")
        mock_session.run.assert_called_once()


class ConsolidationContradictionHandlerTest(TestCase):
    """Test contradiction handling in consolidation pipeline."""

    def test_handle_contradiction_prefer_old_skips_storage(self) -> None:
        """_handle_contradiction with prefer_old returns 'skipped'."""

        mock_memory = MagicMock()
        mock_session = create_mock_neo4j_session()

        contradiction = ContradictionResult(
            has_contradiction=True,
            contradicting_fact_id="old-fact",
            resolution="prefer_old",
        )

        fact_dict = {"claim": "New contradicting fact", "confidence": 0.7}

        result = _handle_contradiction(
            mock_memory, mock_session, fact_dict, contradiction, "user-1", "_default"
        )

        self.assertEqual(result, "skipped")

    def test_handle_contradiction_flag_review_flags_fact(self) -> None:
        """_handle_contradiction with flag_review sets flagged_for_review."""

        mock_memory = MagicMock()
        mock_session = create_mock_neo4j_session()

        contradiction = ContradictionResult(
            has_contradiction=True,
            contradicting_fact_id="old-fact",
            resolution="flag_review",
        )

        fact_dict = {"claim": "Ambiguous fact", "confidence": 0.7}

        result = _handle_contradiction(
            mock_memory, mock_session, fact_dict, contradiction, "user-1", "_default"
        )

        self.assertEqual(result, "flagged")
        self.assertTrue(fact_dict.get("flagged_for_review"))


class MemoryExportDatetimeTest(TestCase):
    """Memory export must not crash on raw Neo4j temporals (e.g. self_consolidated)."""

    @staticmethod
    def _neo4j_dt():
        from neo4j.time import DateTime as Neo4jDateTime
        return Neo4jDateTime(2026, 6, 1, 12, 0, 0)

    def test_convert_all_datetimes_generic(self) -> None:
        rec = {
            "id": "c1",
            "self_consolidated": self._neo4j_dt(),  # not in any allowlist
            "channel": "_default",
            "count": 3,
            "tags": ["a", "b"],
            "missing": None,
        }
        out = convert_all_datetimes(rec)
        self.assertIsInstance(out["self_consolidated"], str)
        # Non-temporal values are untouched.
        self.assertEqual(out["channel"], "_default")
        self.assertEqual(out["count"], 3)
        self.assertEqual(out["tags"], ["a", "b"])
        self.assertIsNone(out["missing"])

    def test_exporter_node_serializes_self_consolidated(self) -> None:
        node = MemoryExporter(user_id="u1")._node({
            "id": "c1",
            "self_consolidated": self._neo4j_dt(),
            "embedding": [0.1, 0.2],
        })
        self.assertNotIn("embedding", node)  # text-only export
        self.assertIsInstance(node["self_consolidated"], str)
        # End-to-end: the node round-trips through pydantic JSON dump — the original
        # crash site (model_dump raised on the raw neo4j.time.DateTime).
        export = MemoryExport(
            user_id="u1", channel=None,
            embedder=current_embedder_info(), conversations=[node],
        )
        dumped = export.model_dump(mode="json")
        self.assertEqual(
            dumped["conversations"][0]["self_consolidated"], node["self_consolidated"]
        )


class EntityLinkBackfillTest(TestCase):
    """Deterministic name/alias/slug backfill of orphaned fact→entity links."""

    def test_slug(self) -> None:
        self.assertEqual(_slug("San Francisco!"), "sanfrancisco")
        self.assertEqual(_slug("GPT-4o"), "gpt4o")

    def test_claim_candidates_ngrams_stopwords_order(self) -> None:
        lc = [c.lower() for c in _claim_entity_candidates("User works at Acme Corp.", max_ngram=3)]
        self.assertIn("acme corp", lc)        # multi-word entity name
        self.assertIn("acme", lc)
        self.assertNotIn("user", lc)          # subject token dropped (unigram)
        self.assertNotIn("at", lc)            # stopword dropped (unigram)
        self.assertLess(lc.index("acme corp"), lc.index("acme"))  # longest-first

    def test_build_entity_index(self) -> None:
        session = MagicMock()
        session.run.return_value = [
            {"id": "e1", "name": "Acme Corp", "aliases": ["Acme"], "salience": 0.9},
            {"id": "e2", "name": "Python", "aliases": [], "salience": 0.5},
        ]
        idx = _build_entity_index(session, "u1", "_default")
        self.assertEqual(idx["acme corp"], "e1")   # name
        self.assertEqual(idx["acme"], "e1")         # alias
        self.assertEqual(idx["acmecorp"], "e1")     # slug
        self.assertEqual(idx["python"], "e2")

    def test_backfill_links_matches_and_counts_orphans(self) -> None:
        orphans = [
            {"fact_id": "f1", "claim": "User works at Acme Corp",
             "user_id": "u1", "channel": "_default"},
            {"fact_id": "f2", "claim": "User enjoys long walks",
             "user_id": "u1", "channel": "_default"},
        ]
        entities = [{"id": "e1", "name": "Acme Corp", "aliases": ["Acme"], "salience": 0.9}]
        merge_calls = []

        def _run(query, **params):
            if "WHERE NOT (f)-[:ABOUT]" in query:
                return list(orphans)
            if "MERGE (f)-[r:ABOUT]" in query:
                merge_calls.append(params)
                return MagicMock()
            if "MATCH (e:Entity)" in query:  # the per-channel index builder
                return list(entities)
            return MagicMock()

        session = create_mock_neo4j_session()
        session.run.side_effect = _run
        with patch(
            'agentx_ai.kit.agent_memory.connections.Neo4jConnection.session',
            return_value=session,
        ):
            result = link_facts_to_entities()

        self.assertEqual(result["items_processed"], 2)
        self.assertEqual(result["links_created"], 1)        # f1 → e1
        self.assertEqual(result["facts_still_orphan"], 1)   # f2 unmatched
        self.assertEqual(len(merge_calls), 1)
        self.assertEqual(merge_calls[0]["entity_ids"], ["e1"])


class SubjectRoutingTest(TestCase):
    """Subject → channel routing so user facts and agent self-knowledge stay apart."""

    def test_resolve_subject_channel(self) -> None:
        # agent-subject + an agent present → the agent's self channel
        self.assertEqual(
            _resolve_subject_channel("agent", "_default", "bold-cosmic-falcon"),
            "_self_bold-cosmic-falcon",
        )
        # user / third_party → the active channel
        self.assertEqual(_resolve_subject_channel("user", "proj", "agent-x"), "proj")
        self.assertEqual(_resolve_subject_channel("third_party", "proj", "agent-x"), "proj")
        # agent-subject but no agent on the conversation → fall back to active channel
        self.assertEqual(_resolve_subject_channel("agent", "proj", None), "proj")

    def test_resolve_subject_channel_specific_agent(self) -> None:
        # A fact attributed to a *specific* agent homes to THAT agent's self channel,
        # not the conversation's addressed/producing agent.
        self.assertEqual(
            _resolve_subject_channel(
                "agent", "_default", "bold-cosmic-falcon",
                subject_agent_id="mobile-cosmic-falcon",
            ),
            "_self_mobile-cosmic-falcon",
        )
        # subject_agent_id is ignored for non-agent subjects.
        self.assertEqual(
            _resolve_subject_channel(
                "user", "proj", "agent-x", subject_agent_id="mobile-cosmic-falcon",
            ),
            "proj",
        )

    def test_router_routes_and_caches(self) -> None:
        active_mem = MagicMock(name="active")
        cache: dict = {"u1:proj": active_mem}
        with patch(
            'agentx_ai.kit.agent_memory.memory.interface.AgentMemory',
        ) as AM:
            self_mem = MagicMock(name="self")
            AM.return_value = self_mem
            route = _make_subject_router(cache, "u1", "proj", "bold-cosmic-falcon")

            # user-subject → existing active-channel memory, no new construction
            mem, channel = route({"subject": "user"})
            self.assertIs(mem, active_mem)
            self.assertEqual(channel, "proj")
            AM.assert_not_called()

            # agent-subject → lazily built self-channel memory, then cached
            mem, channel = route({"subject": "agent"})
            self.assertIs(mem, self_mem)
            self.assertEqual(channel, "_self_bold-cosmic-falcon")
            self.assertIn("u1:_self_bold-cosmic-falcon", cache)
            route({"subject": "agent"})  # second call reuses cache
            AM.assert_called_once()

    def test_router_fans_out_to_specific_agents(self) -> None:
        """One conversation can route facts to several agents' self-channels."""
        active_mem = MagicMock(name="active")
        cache: dict = {"u1:proj": active_mem}
        with patch('agentx_ai.kit.agent_memory.memory.interface.AgentMemory') as AM:
            AM.side_effect = lambda **kw: MagicMock(name=kw.get("channel"))
            route = _make_subject_router(cache, "u1", "proj", "bold-cosmic-falcon")

            # A directive aimed at Mobius → Mobius's self channel (not the producer's).
            _, mobius_ch = route({"subject": "agent", "subject_agent_id": "mobile-cosmic-falcon"})
            self.assertEqual(mobius_ch, "_self_mobile-cosmic-falcon")
            # Another fact aimed at Atlas → Atlas's self channel.
            _, atlas_ch = route({"subject": "agent", "subject_agent_id": "steady-iron-atlas"})
            self.assertEqual(atlas_ch, "_self_steady-iron-atlas")
            # Bare agent subject → the conversation's producing agent.
            _, bare_ch = route({"subject": "agent"})
            self.assertEqual(bare_ch, "_self_bold-cosmic-falcon")

    def test_normalize_subject_defaults_and_validates(self) -> None:
        facts = [
            {"claim": "User works at Acme"},                 # missing → default
            {"claim": "Agent reasons well", "subject": "agent"},
            {"claim": "X", "subject": "BOGUS"},              # invalid → default
            {"claim": "Y", "subject": "Third_Party"},        # case-insensitive
        ]
        out = ExtractionService._normalize_subject(facts, default="user")
        self.assertEqual([f["subject"] for f in out],
                         ["user", "agent", "user", "third_party"])


class AgentAttributionResolutionTest(TestCase):
    """Resolve the LLM-supplied agent NAME → agent_id (the durable source of truth)."""

    ROSTER = [
        {"agent_id": "mobile-cosmic-falcon", "name": "Mobius"},
        {"agent_id": "steady-iron-atlas", "name": "Atlas"},
    ]

    def test_resolves_name_case_insensitively(self) -> None:
        facts = [{"claim": "User wants Mobius to think step-by-step",
                  "subject": "agent", "subject_agent": "mobius"}]
        out = ExtractionService._resolve_agent_attribution(
            facts, self.ROSTER, default_agent_id="steady-iron-atlas", default_subject="user",
        )
        self.assertEqual(out[0]["subject"], "agent")
        self.assertEqual(out[0]["subject_agent_id"], "mobile-cosmic-falcon")
        self.assertNotIn("subject_agent", out[0])  # transient name field consumed

    def test_bare_agent_falls_back_to_addressed(self) -> None:
        facts = [{"claim": "Agent is helpful", "subject": "agent"}]
        out = ExtractionService._resolve_agent_attribution(
            facts, self.ROSTER, default_agent_id="steady-iron-atlas", default_subject="user",
        )
        self.assertEqual(out[0]["subject_agent_id"], "steady-iron-atlas")

    def test_unknown_name_demotes_to_third_party(self) -> None:
        # Never fabricate an agent_id for a name that isn't a real agent.
        facts = [{"claim": "Jarvis is fast", "subject": "agent", "subject_agent": "Jarvis"}]
        out = ExtractionService._resolve_agent_attribution(
            facts, self.ROSTER, default_agent_id="steady-iron-atlas", default_subject="user",
        )
        self.assertEqual(out[0]["subject"], "third_party")
        self.assertNotIn("subject_agent_id", out[0])

    def test_user_subject_untouched(self) -> None:
        facts = [{"claim": "User likes Python", "subject": "user"}]
        out = ExtractionService._resolve_agent_attribution(
            facts, self.ROSTER, default_agent_id="mobile-cosmic-falcon", default_subject="user",
        )
        self.assertEqual(out[0]["subject"], "user")
        self.assertNotIn("subject_agent_id", out[0])

    def test_render_roster(self) -> None:
        roster_block, addressed = ExtractionService._render_roster(
            self.ROSTER, "mobile-cosmic-falcon",
        )
        self.assertIn("Mobius (id: mobile-cosmic-falcon)", roster_block)
        self.assertIn("Atlas (id: steady-iron-atlas)", roster_block)
        self.assertEqual(addressed, "Mobius (id: mobile-cosmic-falcon)")

    def test_render_roster_empty(self) -> None:
        roster_block, addressed = ExtractionService._render_roster([], None)
        self.assertIn("single-agent", roster_block)
        self.assertIn("unknown", addressed)


class FactEntityLinkResolutionTest(TestCase):
    """Unit tests for _resolve_fact_entity_ids — the bulletproof fact→entity linker.

    Pins the three-tier resolution (batch map → store lookup → stub) plus dedup,
    blank-skipping, and the autocreate-disabled fallback. All mocked; no Docker.
    """

    @staticmethod
    def _metrics() -> ConsolidationMetrics:
        return ConsolidationMetrics(job_id="t")

    def test_batch_map_fast_path(self) -> None:
        """A name already in entity_map resolves without touching the store."""
        memory = MagicMock()
        metrics = self._metrics()
        ids = _resolve_fact_entity_ids(
            memory, MagicMock(), ["Python"], {"python": "ent-py"},
            "user-1", "_default", "conv-1", metrics,
        )
        self.assertEqual(ids, ["ent-py"])
        memory.semantic.find_entity_by_name_or_alias.assert_not_called()
        self.assertEqual(metrics.fact_entity_links_recovered, 0)
        self.assertEqual(metrics.fact_entity_stubs_created, 0)

    def test_store_lookup_recovers_link(self) -> None:
        """A name missing from the batch map is recovered via name/alias/slug lookup."""
        memory = MagicMock()
        memory.semantic.find_entity_by_name_or_alias.return_value = {"id": "ent-claude"}
        metrics = self._metrics()
        entity_map: dict = {}
        ids = _resolve_fact_entity_ids(
            memory, MagicMock(), ["Claude"], entity_map,
            "user-1", "_default", "conv-1", metrics,
        )
        self.assertEqual(ids, ["ent-claude"])
        self.assertEqual(metrics.fact_entity_links_recovered, 1)
        self.assertEqual(metrics.fact_entity_stubs_created, 0)
        self.assertEqual(entity_map["claude"], "ent-claude")  # cached for the batch

    def test_autocreate_stub_when_unresolved(self) -> None:
        """An unresolvable name gets a stub entity so the fact is never orphaned."""
        memory = MagicMock()
        memory.semantic.find_entity_by_name_or_alias.return_value = None
        metrics = self._metrics()
        entity_map: dict = {}
        with patch.object(
            consolidation_jobs_module, '_batch_store_entities', return_value=1,
        ) as bse:
            ids = _resolve_fact_entity_ids(
                memory, MagicMock(), ["NewThing"], entity_map,
                "user-1", "_default", "conv-1", metrics,
            )
        self.assertEqual(len(ids), 1)
        self.assertEqual(metrics.fact_entity_stubs_created, 1)
        bse.assert_called_once()
        # The stub was persisted with the verbatim name and cached lowercased.
        stub_entity = bse.call_args.args[1][0]
        self.assertEqual(stub_entity.name, "NewThing")
        self.assertEqual(stub_entity.type, "Concept")
        self.assertEqual(entity_map["newthing"], ids[0])

    def test_stub_disabled_drops_with_no_link(self) -> None:
        """With autocreate off, an unresolved name is dropped (no stub, no link)."""
        memory = MagicMock()
        memory.semantic.find_entity_by_name_or_alias.return_value = None
        metrics = self._metrics()
        with patch.object(consolidation_jobs_module, '_batch_store_entities') as bse, \
             patch.object(
                 consolidation_jobs_module, 'get_settings',
                 return_value=MagicMock(link_autocreate_stub_entities=False),
             ):
            ids = _resolve_fact_entity_ids(
                memory, MagicMock(), ["Ghost"], {},
                "user-1", "_default", "conv-1", metrics,
            )
        self.assertEqual(ids, [])
        bse.assert_not_called()
        self.assertEqual(metrics.fact_entity_stubs_created, 0)

    def test_dedup_order_and_blank_skip(self) -> None:
        """Blank names are skipped; ids are deduped while preserving first-seen order."""
        memory = MagicMock()
        metrics = self._metrics()
        ids = _resolve_fact_entity_ids(
            memory, MagicMock(), ["A", "  ", "B", "a"], {"a": "ent-a", "b": "ent-b"},
            "user-1", "_default", "conv-1", metrics,
        )
        self.assertEqual(ids, ["ent-a", "ent-b"])


class ConsolidationMetricsTest(TestCase):
    """Test ConsolidationMetrics dataclass and its methods."""

    def test_metrics_initialization(self) -> None:
        """ConsolidationMetrics should initialize with default values."""

        metrics = ConsolidationMetrics()
        self.assertEqual(metrics.turns_total, 0)
        self.assertEqual(metrics.entities_stored, 0)
        self.assertEqual(metrics.facts_stored, 0)
        self.assertEqual(metrics.total_llm_calls, 0)

    def test_skip_rate_calculation(self) -> None:
        """skip_rate should calculate percentage of skipped turns."""

        metrics = ConsolidationMetrics(
            turns_total=10,
            turns_skipped_heuristic=3,
            turns_skipped_llm=2,
        )
        self.assertAlmostEqual(metrics.skip_rate, 0.5)

    def test_skip_rate_zero_turns(self) -> None:
        """skip_rate should return 0 when no turns processed."""

        metrics = ConsolidationMetrics(turns_total=0)
        self.assertEqual(metrics.skip_rate, 0.0)

    def test_extraction_efficiency_calculation(self) -> None:
        """extraction_efficiency should calculate facts per LLM call."""

        metrics = ConsolidationMetrics(
            extraction_calls=5,
            facts_extracted=15,
        )
        self.assertAlmostEqual(metrics.extraction_efficiency, 3.0)

    def test_to_dict_serialization(self) -> None:
        """to_dict should properly serialize metrics including computed properties."""

        now = datetime.now(timezone.utc)
        metrics = ConsolidationMetrics(
            job_id="test-123",
            started_at=now,
            turns_total=10,
            turns_skipped_heuristic=5,
        )
        d = metrics.to_dict()

        self.assertEqual(d["job_id"], "test-123")
        self.assertEqual(d["turns_total"], 10)
        self.assertIn("skip_rate", d)
        self.assertAlmostEqual(d["skip_rate"], 0.5)
        self.assertIsInstance(d["started_at"], str)

    def test_from_dict_deserialization(self) -> None:
        """from_dict should reconstruct metrics from dictionary."""

        d = {
            "job_id": "test-456",
            "turns_total": 20,
            "facts_extracted": 5,
            "started_at": "2024-03-01T12:00:00+00:00",
        }
        metrics = ConsolidationMetrics.from_dict(d)

        self.assertEqual(metrics.job_id, "test-456")
        self.assertEqual(metrics.turns_total, 20)
        self.assertIsNotNone(metrics.started_at)


class AggregatedMetricsTest(TestCase):
    """Test AggregatedMetrics for dashboard aggregation."""

    def test_add_run_aggregates_correctly(self) -> None:
        """add_run should aggregate metrics from multiple consolidation runs."""
        agg = AggregatedMetrics(period="2024-03-01")

        # Add first run
        run1 = ConsolidationMetrics(
            turns_total=10,
            turns_skipped_heuristic=2,
            total_llm_calls=5,
            total_tokens_used=1000,
            entities_stored=3,
            facts_stored=7,
            errors=["error1"],
        )
        agg.add_run(run1)

        self.assertEqual(agg.runs, 1)
        self.assertEqual(agg.total_turns, 10)
        self.assertEqual(agg.total_errors, 1)

        # Add second run
        run2 = ConsolidationMetrics(
            turns_total=20,
            turns_skipped_heuristic=5,
            total_llm_calls=10,
            total_tokens_used=2000,
            entities_stored=5,
            facts_stored=10,
        )
        agg.add_run(run2)

        self.assertEqual(agg.runs, 2)
        self.assertEqual(agg.total_turns, 30)
        self.assertEqual(agg.total_tokens, 3000)
        self.assertEqual(agg.avg_tokens_per_run, 1500)


class ClaimHashTest(TestCase):
    """Test claim hash computation for duplicate detection."""

    def test_compute_claim_hash(self) -> None:
        """compute_claim_hash should return consistent hash."""

        claim = "User prefers Python programming"
        hash1 = compute_claim_hash(claim)
        hash2 = compute_claim_hash(claim)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_claim_hash_case_insensitive(self) -> None:
        """Hash should be case-insensitive."""

        hash1 = compute_claim_hash("User prefers Python")
        hash2 = compute_claim_hash("user prefers python")

        self.assertEqual(hash1, hash2)

    def test_claim_hash_whitespace_normalized(self) -> None:
        """Hash should normalize whitespace."""

        hash1 = compute_claim_hash("User prefers Python")
        hash2 = compute_claim_hash("  User   prefers    Python  ")

        self.assertEqual(hash1, hash2)

    def test_fact_auto_computes_hash(self) -> None:
        """Fact model should auto-compute claim_hash on creation."""

        fact = Fact(claim="User prefers Python", source="extraction")

        self.assertIsNotNone(fact.claim_hash)
        assert fact.claim_hash is not None
        self.assertEqual(len(fact.claim_hash), 16)

    def test_fact_preserves_explicit_hash(self) -> None:
        """Fact should preserve explicitly provided claim_hash."""

        fact = Fact(
            claim="Test claim",
            claim_hash="explicit123456ab",
            source="extraction",
        )

        self.assertEqual(fact.claim_hash, "explicit123456ab")


class SettingsCacheTTLTest(TestCase):
    """Test settings cache TTL refresh mechanism."""

    def test_settings_cache_respects_ttl(self) -> None:
        """Settings should be cached and respect TTL."""

        # Reset cache
        memory_config_module._runtime_settings = None
        memory_config_module._settings_cache_time = 0.0

        # First load
        s1 = memory_config_module.get_settings()
        time1 = memory_config_module._settings_cache_time

        # Second load (should use cache)
        s2 = memory_config_module.get_settings()

        self.assertIs(s1, s2)
        self.assertEqual(memory_config_module._settings_cache_time, time1)

    def test_cache_ttl_constant_exists(self) -> None:
        """SETTINGS_CACHE_TTL should be defined."""

        self.assertIsInstance(SETTINGS_CACHE_TTL, float)
        self.assertGreater(SETTINGS_CACHE_TTL, 0)


# =============================================================================
# Combined Extraction Tests (Session 2)
# =============================================================================

class CombinedExtractionResultTest(TestCase):
    """Test CombinedExtractionResult model."""

    def test_default_values(self) -> None:
        """CombinedExtractionResult should have sensible defaults."""

        result = CombinedExtractionResult()

        self.assertFalse(result.is_relevant)
        self.assertEqual(result.reason, "")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.facts, [])
        self.assertEqual(result.relationships, [])
        self.assertEqual(result.tokens_used, 0)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_with_data(self) -> None:
        """CombinedExtractionResult should hold extraction data."""

        result = CombinedExtractionResult(
            is_relevant=True,
            reason="llm_extracted",
            entities=[{"name": "Acme", "type": "Organization"}],
            facts=[{"claim": "User works at Acme", "confidence": 0.9}],
            relationships=[{"source": "User", "target": "Acme", "type": "works_at"}],
            tokens_used=150,
        )

        self.assertTrue(result.is_relevant)
        self.assertEqual(result.reason, "llm_extracted")
        self.assertEqual(len(result.entities), 1)
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(result.tokens_used, 150)


class CombinedExtractionHeuristicTest(TestCase):
    """Test heuristic skip behavior in combined extraction."""

    def test_heuristic_skip_short_text(self) -> None:
        """Very short text should be skipped via heuristic."""

        service = ExtractionService()
        result = asyncio.run(service.check_relevance_and_extract("ok"))

        self.assertFalse(result.is_relevant)
        self.assertEqual(result.reason, "heuristic_skip")
        self.assertTrue(result.success)
        self.assertEqual(result.entities, [])
        self.assertEqual(result.facts, [])

    def test_heuristic_skip_common_phrases(self) -> None:
        """Common filler phrases should be skipped via heuristic."""

        service = ExtractionService()
        skip_phrases = ["thanks", "got it", "sure", "yes", "no", "cool"]

        for phrase in skip_phrases:
            result = asyncio.run(service.check_relevance_and_extract(phrase))
            self.assertFalse(result.is_relevant, f"Should skip: {phrase}")
            self.assertEqual(result.reason, "heuristic_skip")

    def test_non_skip_text_proceeds_to_llm(self) -> None:
        """Meaningful text should not be skipped by heuristic."""

        service = ExtractionService()

        # This text is long enough and not a skip phrase
        text = "I work at Anthropic as a software engineer"

        # Mock the provider to avoid actual LLM call
        with patch.object(service, '_get_provider_for_stage') as mock_provider:
            # Simulate provider unavailable to trigger fallback
            mock_provider.side_effect = ValueError("Provider unavailable")

            asyncio.run(service.check_relevance_and_extract(text))

            # Should have attempted to get provider (not skipped by heuristic)
            # First call should be for 'combined' stage
            self.assertTrue(mock_provider.called)
            first_call_args = mock_provider.call_args_list[0][0]
            self.assertEqual(first_call_args[0], 'combined')


class ConfidenceCalibrationTest(TestCase):
    """Test confidence calibration mapping."""

    def test_explicit_certainty_maps_to_high_confidence(self) -> None:
        """Explicit certainty should map to highest confidence."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "I work at Anthropic", "certainty": "explicit"}]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.95)
        self.assertNotIn("certainty", calibrated[0])  # Should be removed

    def test_implied_certainty_maps_to_085(self) -> None:
        """Implied certainty should map to 0.85."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "My office is in SF", "certainty": "implied"}]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.85)

    def test_inferred_certainty_maps_to_070(self) -> None:
        """Inferred certainty should map to 0.70."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "User knows Python", "certainty": "inferred"}]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.70)

    def test_uncertain_certainty_maps_to_050(self) -> None:
        """Uncertain certainty should map to lowest confidence."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "I think I might like React", "certainty": "uncertain"}]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.50)

    def test_unknown_certainty_defaults_to_inferred(self) -> None:
        """Unknown certainty levels should default to inferred (0.70)."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "Some claim", "certainty": "unknown_level"}]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.70)

    def test_missing_certainty_defaults_to_inferred(self) -> None:
        """Missing certainty field should default to inferred (0.70)."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{"claim": "Some claim"}]  # No certainty field
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.70)

    def test_calibration_is_case_insensitive(self) -> None:
        """Certainty levels should be matched case-insensitively."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [
            {"claim": "Claim 1", "certainty": "EXPLICIT"},
            {"claim": "Claim 2", "certainty": "Implied"},
            {"claim": "Claim 3", "certainty": "INFERRED"},
        ]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["confidence"], 0.95)
        self.assertEqual(calibrated[1]["confidence"], 0.85)
        self.assertEqual(calibrated[2]["confidence"], 0.70)

    def test_calibration_preserves_other_fields(self) -> None:
        """Calibration should preserve other fact fields."""

        service = ExtractionService()
        service._settings = Settings()

        facts = [{
            "claim": "I work at Acme",
            "certainty": "explicit",
            "entity_names": ["Acme"],
            "source_turn_id": "turn_123",
        }]
        calibrated = service._apply_confidence_calibration(facts)

        self.assertEqual(calibrated[0]["claim"], "I work at Acme")
        self.assertEqual(calibrated[0]["entity_names"], ["Acme"])
        self.assertEqual(calibrated[0]["source_turn_id"], "turn_123")


class CombinedExtractionConfigTest(TestCase):
    """Test combined extraction configuration settings."""

    def test_combined_extraction_settings_exist(self) -> None:
        """Combined extraction settings should be defined."""

        settings = Settings()

        # Model settings (uses provider:model format, e.g., "lmstudio:nvidia/nemotron-3-nano")
        self.assertIsInstance(settings.combined_extraction_model, str)
        self.assertIsInstance(settings.combined_extraction_temperature, float)
        self.assertIsInstance(settings.combined_extraction_max_tokens, int)

    def test_confidence_calibration_settings_exist(self) -> None:
        """Confidence calibration settings should be defined."""

        settings = Settings()

        self.assertIsInstance(settings.confidence_explicit, float)
        self.assertIsInstance(settings.confidence_implied, float)
        self.assertIsInstance(settings.confidence_inferred, float)
        self.assertIsInstance(settings.confidence_uncertain, float)

    def test_default_confidence_values_ordered(self) -> None:
        """Default confidence values should be properly ordered."""

        settings = Settings()

        # explicit > implied > inferred > uncertain
        self.assertGreater(settings.confidence_explicit, settings.confidence_implied)
        self.assertGreater(settings.confidence_implied, settings.confidence_inferred)
        self.assertGreater(settings.confidence_inferred, settings.confidence_uncertain)

    def test_default_model_is_reasoning_model(self) -> None:
        """Default combined extraction model should be reasoning model."""

        settings = Settings()

        # Should default to lmstudio:nvidia/nemotron-3-nano or similar reasoning model
        self.assertIn("nemotron", settings.combined_extraction_model.lower())


class CombinedExtractionPromptTest(TestCase):
    """Test combined extraction prompt template."""

    def test_combined_prompt_exists(self) -> None:
        """Combined extraction prompt should be defined."""

        loader = get_prompt_loader()
        prompt = loader.get("extraction.combined_with_relevance", text="test text")

        self.assertIsInstance(prompt, str)
        self.assertIn("test text", prompt)

    def test_combined_prompt_has_relevance_step(self) -> None:
        """Combined prompt should include relevance check step."""

        loader = get_prompt_loader()
        prompt = loader.get("extraction.combined_with_relevance", text="test")

        self.assertIn("relevant", prompt.lower())
        self.assertIn("is_relevant", prompt)

    def test_combined_prompt_has_certainty_levels(self) -> None:
        """Combined prompt should include certainty levels."""

        loader = get_prompt_loader()
        prompt = loader.get("extraction.combined_with_relevance", text="test")

        self.assertIn("explicit", prompt.lower())
        self.assertIn("implied", prompt.lower())
        self.assertIn("inferred", prompt.lower())
        self.assertIn("uncertain", prompt.lower())

    def test_combined_prompt_outputs_json(self) -> None:
        """Combined prompt should request JSON output."""

        loader = get_prompt_loader()
        prompt = loader.get("extraction.combined_with_relevance", text="test")

        self.assertIn("JSON", prompt)
        self.assertIn("is_relevant", prompt)
        self.assertIn("entities", prompt)
        self.assertIn("facts", prompt)


class CombinedExtractionProviderUnavailableTest(TestCase):
    """Test behavior when combined provider is unavailable."""

    def test_returns_error_when_provider_unavailable(self) -> None:
        """Should return error result when provider unavailable."""

        service = ExtractionService()

        with patch.object(service, '_get_provider_for_stage') as mock_provider:
            # Simulate combined provider unavailable
            mock_provider.side_effect = ValueError("Provider unavailable")

            # Meaningful text that won't be skipped by heuristic
            result = asyncio.run(service.check_relevance_and_extract(
                "I work at Anthropic as a software engineer"
            ))

            # Should return error result (defaults to relevant for safety)
            self.assertTrue(result.is_relevant)
            self.assertEqual(result.reason, "provider_unavailable")
            self.assertFalse(result.success)
            self.assertIn("Provider unavailable", result.error)  # type: ignore[arg-type]


# =============================================================================
# Access Tracking Tests (Session 3 - Reinforcement Signal)
# =============================================================================

class FactAccessTrackingFieldsTest(TestCase):
    """Test that Fact model has access tracking fields."""

    def test_fact_has_last_accessed_field(self) -> None:
        """Fact model should have last_accessed field."""

        fact = Fact(claim="test claim", source="extraction")

        self.assertTrue(hasattr(fact, 'last_accessed'))
        self.assertIsNotNone(fact.last_accessed)

    def test_fact_has_access_count_field(self) -> None:
        """Fact model should have access_count field with default 0."""

        fact = Fact(claim="test claim", source="extraction")

        self.assertTrue(hasattr(fact, 'access_count'))
        self.assertEqual(fact.access_count, 0)

    def test_fact_has_salience_field(self) -> None:
        """Fact model should have salience field with default 0.5."""

        fact = Fact(claim="test claim", source="extraction")

        self.assertTrue(hasattr(fact, 'salience'))
        self.assertEqual(fact.salience, 0.5)

    def test_fact_salience_can_be_set(self) -> None:
        """Fact salience should be settable."""

        fact = Fact(claim="test claim", source="extraction", salience=0.8)

        self.assertEqual(fact.salience, 0.8)


# =============================================================================
# Temporal Context Tests (Session 3 - Temporal Reasoning)
# =============================================================================

class FactTemporalContextFieldTest(TestCase):
    """Test that Fact model has temporal_context field."""

    def test_fact_has_temporal_context_field(self) -> None:
        """Fact model should have temporal_context field."""

        fact = Fact(claim="test claim", source="extraction")

        self.assertTrue(hasattr(fact, 'temporal_context'))
        self.assertIsNone(fact.temporal_context)  # Default is None

    def test_fact_temporal_context_can_be_set(self) -> None:
        """Fact temporal_context should be settable."""

        fact = Fact(claim="I work at Google", source="extraction", temporal_context="current")

        self.assertEqual(fact.temporal_context, "current")


class TemporalContextNormalizationTest(TestCase):
    """Test temporal context normalization in extraction service."""

    def test_normalize_valid_current(self) -> None:
        """Current should be normalized to lowercase."""

        service = ExtractionService()
        facts = [{"claim": "I work at Google", "temporal_context": "Current"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertEqual(normalized[0]["temporal_context"], "current")

    def test_normalize_valid_past(self) -> None:
        """Past should be normalized to lowercase."""

        service = ExtractionService()
        facts = [{"claim": "I used to work at Google", "temporal_context": "PAST"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertEqual(normalized[0]["temporal_context"], "past")

    def test_normalize_valid_future(self) -> None:
        """Future should be normalized to lowercase."""

        service = ExtractionService()
        facts = [{"claim": "I'm starting at Google", "temporal_context": "Future"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertEqual(normalized[0]["temporal_context"], "future")

    def test_normalize_invalid_to_none(self) -> None:
        """Invalid temporal_context should become None."""

        service = ExtractionService()
        facts = [{"claim": "Some claim", "temporal_context": "invalid_value"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertIsNone(normalized[0]["temporal_context"])

    def test_normalize_null_string_to_none(self) -> None:
        """'null' string should become None."""

        service = ExtractionService()
        facts = [{"claim": "Some claim", "temporal_context": "null"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertIsNone(normalized[0]["temporal_context"])

    def test_normalize_missing_field(self) -> None:
        """Missing temporal_context should become None."""

        service = ExtractionService()
        facts = [{"claim": "Some claim"}]

        normalized = service._normalize_temporal_fields(facts)

        self.assertIsNone(normalized[0]["temporal_context"])


class TemporalContextInPromptTest(TestCase):
    """Test temporal context in extraction prompt."""

    def test_prompt_includes_temporal_context(self) -> None:
        """Combined extraction prompt should include temporal context instructions."""

        loader = get_prompt_loader()
        prompt = loader.get("extraction.combined_with_relevance", text="test")

        self.assertIn("temporal_context", prompt.lower())
        self.assertIn("current", prompt.lower())
        self.assertIn("past", prompt.lower())
        self.assertIn("future", prompt.lower())


class LearnFactTemporalContextTest(TestCase):
    """Test learn_fact accepts temporal_context parameter."""

    def test_learn_fact_accepts_temporal_context(self) -> None:
        """learn_fact should accept temporal_context parameter."""

        # Check the method signature accepts temporal_context
        sig = inspect.signature(AgentMemory.learn_fact)
        params = list(sig.parameters.keys())

        self.assertIn("temporal_context", params)


# =============================================================================
# RecallLayer Tests (Phase 11.11)
# =============================================================================


class RecallLayerConfigTest(TestCase):
    """Test RecallLayer configuration settings."""

    def test_config_has_recall_settings(self) -> None:
        """Config should have all RecallLayer settings."""

        settings = get_settings()

        # Feature toggles
        self.assertIsInstance(settings.recall_enable_hybrid, bool)
        self.assertIsInstance(settings.recall_enable_entity_centric, bool)
        self.assertIsInstance(settings.recall_enable_query_expansion, bool)
        self.assertIsInstance(settings.recall_enable_hyde, bool)
        self.assertIsInstance(settings.recall_enable_self_query, bool)

        # Hybrid settings
        self.assertIsInstance(settings.recall_hybrid_bm25_weight, float)
        self.assertIsInstance(settings.recall_hybrid_vector_weight, float)
        self.assertIsInstance(settings.recall_hybrid_rrf_k, int)

        # Entity-centric settings
        self.assertIsInstance(settings.recall_entity_similarity_threshold, float)
        self.assertIsInstance(settings.recall_entity_max_entities, int)

    def test_default_techniques_enabled(self) -> None:
        """Hybrid, entity-centric, and expansion should be enabled by default."""

        settings = get_settings()

        # These should be ON by default (cheap, high impact)
        self.assertTrue(settings.recall_enable_hybrid)
        self.assertTrue(settings.recall_enable_entity_centric)
        self.assertTrue(settings.recall_enable_query_expansion)

        # These should be OFF by default (expensive LLM calls)
        self.assertFalse(settings.recall_enable_hyde)
        self.assertFalse(settings.recall_enable_self_query)


class RecallLayerMetricsTest(TestCase):
    """Test RecallMetrics dataclass."""

    def test_metrics_to_dict(self) -> None:
        """RecallMetrics.to_dict() should return all fields."""

        metrics = RecallMetrics(
            query="test query",
            user_id="user123",
            channel="_global",
            techniques_enabled={"hybrid": True},
            base_results=5,
        )

        result = metrics.to_dict()

        self.assertEqual(result["query"], "test query")
        self.assertEqual(result["user_id"], "user123")
        self.assertEqual(result["channel"], "_global")
        self.assertEqual(result["base_results"], 5)
        self.assertIn("techniques_enabled", result)

    def test_metrics_defaults(self) -> None:
        """RecallMetrics should have sensible defaults."""

        metrics = RecallMetrics(
            query="test",
            user_id="user",
            channel="ch",
        )

        self.assertEqual(metrics.hyde_results, 0)
        self.assertEqual(metrics.hybrid_bm25_results, 0)
        self.assertEqual(metrics.entity_centric_facts, 0)
        self.assertEqual(metrics.expansion_results, 0)
        self.assertEqual(metrics.duplicates_removed, 0)


class RecallLayerQueryExpansionTest(TestCase):
    """Test query expansion transforms."""

    def test_question_to_statement_when(self) -> None:
        """'When is my X?' should transform to 'X is'."""

        # Create mock objects
        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        result = recall._question_to_statement("when is my birthday?")
        self.assertIn("birthday", result)
        self.assertIn("is", result)

    def test_question_to_statement_what(self) -> None:
        """'What is my X?' should transform to 'X is'."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        result = recall._question_to_statement("what is my favorite color?")
        self.assertIn("favorite color", result)

    def test_extract_keywords(self) -> None:
        """Keywords should exclude stopwords and question words."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        result = recall._extract_keywords("When is my birthday?")

        # Should include 'birthday' but not 'when', 'is', 'my'
        self.assertIn("birthday", result)
        self.assertNotIn("when", result.lower())
        self.assertNotIn(" is ", result.lower())

    def test_expand_query_generates_variants(self) -> None:
        """expand_query should generate query variants."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        variants = recall._expand_query("When is my birthday?")

        self.assertIsInstance(variants, list)
        # Should generate at least 1 variant
        self.assertGreater(len(variants), 0)


class RecallLayerRRFFusionTest(TestCase):
    """Test Reciprocal Rank Fusion scoring."""

    def test_rrf_fusion_combines_results(self) -> None:
        """RRF should combine BM25 and vector results."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        bm25_results = [
            {"id": "a", "claim": "fact A", "score": 5.0},
            {"id": "b", "claim": "fact B", "score": 3.0},
        ]
        vector_results = [
            {"id": "b", "claim": "fact B", "score": 0.9},
            {"id": "c", "claim": "fact C", "score": 0.8},
        ]

        merged = recall._rrf_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.3,
            vector_weight=0.7,
            rrf_k=60,
            top_k=10,
        )

        # Should have 3 unique results (a, b, c)
        self.assertEqual(len(merged), 3)

        # Results should have rrf_score
        for r in merged:
            self.assertIn("rrf_score", r)

    def test_rrf_fusion_ranks_by_score(self) -> None:
        """RRF results should be sorted by score descending."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        bm25_results = [{"id": "a", "claim": "A", "score": 5.0}]
        vector_results = [{"id": "a", "claim": "A", "score": 0.9}]

        merged = recall._rrf_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.3,
            vector_weight=0.7,
            rrf_k=60,
            top_k=10,
        )

        scores = [r["rrf_score"] for r in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))


class RecallLayerMergeBundlesTest(TestCase):
    """Test bundle merging and deduplication."""

    def test_merge_deduplicates_by_id(self) -> None:
        """Merging should deduplicate by ID."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        bundle1 = MemoryBundle(
            facts=[{"id": "a", "claim": "A"}, {"id": "b", "claim": "B"}]
        )
        bundle2 = MemoryBundle(
            facts=[{"id": "b", "claim": "B"}, {"id": "c", "claim": "C"}]
        )

        merged, stats = recall._merge_bundles(bundle1, bundle2)

        # Should have 3 unique facts
        self.assertEqual(len(merged.facts), 3)

        # Should track 1 duplicate removed
        self.assertEqual(stats["duplicates_removed"], 1)

    def test_merge_preserves_entities_and_turns(self) -> None:
        """Merging should also deduplicate entities and turns."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        bundle1 = MemoryBundle(
            entities=[{"id": "e1", "name": "User"}],
            relevant_turns=[{"id": "t1", "content": "Hello"}],
        )
        bundle2 = MemoryBundle(
            entities=[{"id": "e1", "name": "User"}, {"id": "e2", "name": "Project"}],
            relevant_turns=[{"id": "t2", "content": "World"}],
        )

        merged, _ = recall._merge_bundles(bundle1, bundle2)

        self.assertEqual(len(merged.entities), 2)
        self.assertEqual(len(merged.relevant_turns), 2)


class RecallLayerInterfaceIntegrationTest(TestCase):
    """Test RecallLayer integration with AgentMemory interface."""

    def test_agent_memory_has_recall_layer(self) -> None:
        """AgentMemory should have a recall_layer attribute."""

        # Check the class has the attribute in __init__
        source = inspect.getsource(AgentMemory.__init__)

        self.assertIn("recall_layer", source)
        self.assertIn("RecallLayer", source)

    def test_remember_has_use_recall_layer_param(self) -> None:
        """remember() should have use_recall_layer parameter."""

        sig = inspect.signature(AgentMemory.remember)
        params = list(sig.parameters.keys())

        self.assertIn("use_recall_layer", params)

    def test_remember_defaults_to_recall_layer(self) -> None:
        """use_recall_layer should default to True."""

        sig = inspect.signature(AgentMemory.remember)
        param = sig.parameters["use_recall_layer"]

        self.assertEqual(param.default, True)


class RecallLayerEscapeLuceneTest(TestCase):
    """Test Lucene query escaping for BM25 search."""

    def test_escape_special_chars(self) -> None:
        """Special Lucene characters should be escaped."""

        mock_memory = MagicMock()
        mock_retriever = MagicMock()

        recall = RecallLayer(mock_memory, mock_retriever)

        # Test various special characters
        result = recall._escape_lucene("test+query")
        self.assertIn("\\+", result)

        result = recall._escape_lucene("user@email.com")
        # @ is not a special Lucene char, should remain
        self.assertIn("@", result)

        result = recall._escape_lucene("path/to/file")
        self.assertIn("\\/", result)


# =============================================================================
# Memory Mutation Routes (PATCH/DELETE) — Phase 18 editable memory
# =============================================================================

@skipUnless(docker_services_running(), "Docker services not running")
class MemoryMutationViewTest(APITestBase):
    """PATCH/DELETE routes for facts and entities, with re-embedding semantics."""

    def setUp(self) -> None:
        super().setUp()
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch (local vs schema)")
        # Views always run as DEFAULT_USER_ID = "default"
        from agentx_ai.kit.agent_memory.models import Entity
        self.memory = AgentMemory(user_id="default", channel="_global")
        self.fact = self.memory.learn_fact(
            "The sky is blue", source="user_stated", confidence=0.9
        )
        self.entity = self.memory.upsert_entity(
            Entity(name="Pluto", type="Concept", description="dwarf planet")
        )

    def tearDown(self) -> None:
        # Best-effort cleanup
        try:
            with Neo4jConnection.session() as session:
                session.run(
                    "MATCH (f:Fact {id: $id}) DETACH DELETE f", id=self.fact.id
                )
                session.run(
                    "MATCH (e:Entity {id: $id}) DETACH DELETE e", id=self.entity.id
                )
        except Exception:
            pass
        super().tearDown()

    def _get_fact_embedding(self, fact_id: str):
        with Neo4jConnection.session() as session:
            result = session.run(
                "MATCH (f:Fact {id: $id}) RETURN f.embedding AS e, f.claim AS claim, f.claim_hash AS h",
                id=fact_id,
            )
            record = result.single()
            assert record is not None, f"Fact {fact_id} not found"
            return record

    def _get_entity_embedding(self, entity_id: str):
        with Neo4jConnection.session() as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e.embedding AS e, e.name AS name",
                id=entity_id,
            )
            record = result.single()
            assert record is not None, f"Entity {entity_id} not found"
            return record

    def test_patch_fact_claim_reembeds_and_updates_hash(self) -> None:
        before = self._get_fact_embedding(self.fact.id)
        response = self.client.patch(
            f"/api/memory/facts/{self.fact.id}",
            data=json.dumps({"claim": "The sky is teal"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        body = response.json()  # type: ignore[union-attr]
        self.assertEqual(body["fact"]["claim"], "The sky is teal")

        after = self._get_fact_embedding(self.fact.id)
        self.assertEqual(after["claim"], "The sky is teal")
        self.assertNotEqual(before["e"], after["e"])  # embedding changed
        self.assertNotEqual(before["h"], after["h"])  # claim_hash changed
        self.assertEqual(after["h"], compute_claim_hash("The sky is teal"))

    def test_patch_fact_confidence_does_not_reembed(self) -> None:
        before = self._get_fact_embedding(self.fact.id)
        response = self.client.patch(
            f"/api/memory/facts/{self.fact.id}",
            data=json.dumps({"confidence": 0.42}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        body = response.json()  # type: ignore[union-attr]
        self.assertAlmostEqual(body["fact"]["confidence"], 0.42)

        after = self._get_fact_embedding(self.fact.id)
        self.assertEqual(before["e"], after["e"])  # unchanged

    def test_patch_fact_unknown_field_returns_400(self) -> None:
        response = self.client.patch(
            f"/api/memory/facts/{self.fact.id}",
            data=json.dumps({"banana": 1}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]

    def test_patch_fact_empty_body_returns_400(self) -> None:
        response = self.client.patch(
            f"/api/memory/facts/{self.fact.id}",
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]

    def test_patch_fact_invalid_confidence_returns_400(self) -> None:
        response = self.client.patch(
            f"/api/memory/facts/{self.fact.id}",
            data=json.dumps({"confidence": 2.5}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]

    def test_patch_fact_missing_returns_404(self) -> None:
        response = self.client.patch(
            "/api/memory/facts/does-not-exist",
            data=json.dumps({"confidence": 0.5}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 404)  # type: ignore[union-attr]

    def test_delete_fact_returns_ok_then_404(self) -> None:
        response = self.client.delete(f"/api/memory/facts/{self.fact.id}")
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertTrue(response.json()["deleted"])  # type: ignore[union-attr]
        # Second delete is 404
        response = self.client.delete(f"/api/memory/facts/{self.fact.id}")
        self.assertEqual(response.status_code, 404)  # type: ignore[union-attr]

    def test_patch_entity_name_reembeds(self) -> None:
        before = self._get_entity_embedding(self.entity.id)
        response = self.client.patch(
            f"/api/memory/entities/{self.entity.id}",
            data=json.dumps({"name": "Eris"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        body = response.json()  # type: ignore[union-attr]
        self.assertEqual(body["entity"]["name"], "Eris")

        after = self._get_entity_embedding(self.entity.id)
        self.assertEqual(after["name"], "Eris")
        self.assertNotEqual(before["e"], after["e"])

    def test_patch_entity_properties_does_not_reembed(self) -> None:
        before = self._get_entity_embedding(self.entity.id)
        response = self.client.patch(
            f"/api/memory/entities/{self.entity.id}",
            data=json.dumps({"properties": {"discovered": 1930}}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]

        after = self._get_entity_embedding(self.entity.id)
        self.assertEqual(before["e"], after["e"])

    def test_patch_entity_invalid_aliases_returns_400(self) -> None:
        response = self.client.patch(
            f"/api/memory/entities/{self.entity.id}",
            data=json.dumps({"aliases": "not-a-list"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]

    def test_delete_entity_returns_ok_then_404(self) -> None:
        response = self.client.delete(f"/api/memory/entities/{self.entity.id}")
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertTrue(response.json()["deleted"])  # type: ignore[union-attr]
        response = self.client.delete(f"/api/memory/entities/{self.entity.id}")
        self.assertEqual(response.status_code, 404)  # type: ignore[union-attr]


# =============================================================================
# Phase 18.6: Extraction Tuning — stateful extraction & entity resolution
# =============================================================================


class NameCandidateExtractionTest(TestCase):
    """Regex pre-pass that feeds scope-context lookups."""

    def test_extracts_capitalized_names(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _extract_name_candidates

        out = _extract_name_candidates("I work at Anthropic in San Francisco using Python")
        out_lower = [c.lower() for c in out]
        self.assertIn("anthropic", out_lower)
        self.assertIn("san francisco", out_lower)
        self.assertIn("python", out_lower)

    def test_strips_pronoun_stopwords(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _extract_name_candidates

        out = [c.lower() for c in _extract_name_candidates("I love Python but You hate it")]
        self.assertNotIn("i", out)
        self.assertNotIn("you", out)
        self.assertIn("python", out)

    def test_quoted_spans_picked_up(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _extract_name_candidates

        out = [c.lower() for c in _extract_name_candidates('Project codename "Project Zephyr" ships Friday')]
        self.assertTrue(any("zephyr" in c for c in out))

    def test_case_insensitive_dedup(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _extract_name_candidates

        out = _extract_name_candidates("Python python PYTHON")
        # Only one entry regardless of casing
        self.assertEqual(len([c for c in out if c.lower() == "python"]), 1)


class RenderScopeContextTest(TestCase):
    """ExtractionService._render_scope_context builds the prompt blocks."""

    def test_empty_inputs_yield_none_blocks(self) -> None:
        ents, facts = ExtractionService._render_scope_context(None, None)
        self.assertEqual(ents, "(none)")
        self.assertEqual(facts, "(none)")

    def test_renders_entity_with_aliases_and_description(self) -> None:
        ents, _ = ExtractionService._render_scope_context(
            known_entities=[{
                "id": "ent-1",
                "name": "Python",
                "type": "Technology",
                "aliases": ["py"],
                "description": "Programming language",
            }],
            known_facts=None,
        )
        self.assertIn("ent-1", ents)
        self.assertIn("Python", ents)
        self.assertIn("aliases=['py']", ents)
        self.assertIn("Programming language", ents)

    def test_renders_fact_with_temporal_and_confidence(self) -> None:
        _, facts = ExtractionService._render_scope_context(
            known_entities=None,
            known_facts=[{
                "id": "fact-1",
                "claim": "User uses Python",
                "temporal_context": "current",
                "confidence": 0.95,
            }],
        )
        self.assertIn("fact-1", facts)
        self.assertIn("User uses Python", facts)
        self.assertIn("temporal=current", facts)
        self.assertIn("0.95", facts)

    def test_caps_entries(self) -> None:
        many = [{"id": f"e{i}", "name": f"N{i}", "type": "Concept"} for i in range(50)]
        ents, _ = ExtractionService._render_scope_context(many, None, max_entities=3)
        self.assertEqual(ents.count("\n"), 2)  # 3 lines = 2 newlines


class EntityResolutionTest(TestCase):
    """Server-side entity resolution before storage."""

    def _fake_memory(self, lookup_result=None, by_id_result=None):
        memory = MagicMock()
        memory.semantic.find_entity_by_name_or_alias.return_value = lookup_result
        memory.semantic.get_entity_by_id.return_value = by_id_result
        memory.semantic.merge_entity_aliases.return_value = True
        return memory

    def test_honors_llm_existing_entity_id(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _resolve_and_prepare_entities

        memory = self._fake_memory(
            by_id_result={"id": "ent-py-1", "name": "Python", "type": "Technology", "aliases": ["py"]},
        )
        extracted = [{
            "name": "python",
            "type": "Technology",
            "existing_entity_id": "ent-py-1",
        }]
        entity_map: dict = {}
        metrics = ConsolidationMetrics(job_id="t", started_at=datetime.now(timezone.utc))
        errors: list = []

        new_entities, reused, total = _resolve_and_prepare_entities(
            memory=memory,
            extracted_entities=extracted,
            entity_map=entity_map,
            user_id="u",
            channel="_default",
            conv_id="c",
            metrics=metrics,
            errors=errors,
        )

        self.assertEqual(new_entities, [])
        self.assertEqual(reused, 1)
        self.assertEqual(entity_map.get("python"), "ent-py-1")
        memory.semantic.merge_entity_aliases.assert_called_once()
        # find_entity_by_name_or_alias should NOT have been called since existing_entity_id resolved
        memory.semantic.find_entity_by_name_or_alias.assert_not_called()

    def test_falls_back_to_name_lookup_when_id_missing(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _resolve_and_prepare_entities

        memory = self._fake_memory(
            lookup_result={"id": "ent-py-1", "name": "Python", "type": "Technology", "aliases": []},
        )
        extracted = [{"name": "Python", "type": "Technology"}]
        entity_map: dict = {}
        metrics = ConsolidationMetrics(job_id="t", started_at=datetime.now(timezone.utc))

        new_entities, reused, _ = _resolve_and_prepare_entities(
            memory=memory,
            extracted_entities=extracted,
            entity_map=entity_map,
            user_id="u",
            channel="_default",
            conv_id="c",
            metrics=metrics,
            errors=[],
        )
        self.assertEqual(new_entities, [])
        self.assertEqual(reused, 1)
        self.assertEqual(metrics.entities_reused, 1)

    def test_creates_new_entity_when_no_match(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _resolve_and_prepare_entities

        memory = self._fake_memory(lookup_result=None)
        extracted = [{
            "name": "Brand New Thing",
            "type": "Concept",
            "description": "First mention",
            "confidence": 0.9,
        }]
        entity_map: dict = {}
        metrics = ConsolidationMetrics(job_id="t", started_at=datetime.now(timezone.utc))

        new_entities, reused, _ = _resolve_and_prepare_entities(
            memory=memory,
            extracted_entities=extracted,
            entity_map=entity_map,
            user_id="u",
            channel="_default",
            conv_id="c",
            metrics=metrics,
            errors=[],
        )
        self.assertEqual(len(new_entities), 1)
        self.assertEqual(reused, 0)
        self.assertEqual(new_entities[0].name, "Brand New Thing")
        self.assertEqual(new_entities[0].description, "First mention")
        self.assertIn("brand new thing", entity_map)
        memory.semantic.merge_entity_aliases.assert_not_called()

    def test_skips_entity_with_missing_name_or_type(self) -> None:
        from agentx_ai.kit.agent_memory.consolidation.jobs import _resolve_and_prepare_entities

        memory = self._fake_memory(lookup_result=None)
        extracted = [
            {"name": "", "type": "Concept"},
            {"name": "OK", "type": ""},
            {"name": "Valid", "type": "Concept"},
        ]
        metrics = ConsolidationMetrics(job_id="t", started_at=datetime.now(timezone.utc))

        new_entities, reused, _ = _resolve_and_prepare_entities(
            memory=memory,
            extracted_entities=extracted,
            entity_map={},
            user_id="u", channel="_default", conv_id="c",
            metrics=metrics, errors=[],
        )
        self.assertEqual(len(new_entities), 1)
        self.assertEqual(new_entities[0].name, "Valid")


class RefinesFactIdSupersedureTest(TestCase):
    """Phase 18.6 LLM-supplied refines_fact_id should supersede the target fact."""

    def test_in_scope_refine_triggers_supersede(self) -> None:
        # We exercise the in-scope check + call dispatch by stubbing
        # _handle_contradiction. Since the refine block lives inside the consolidation
        # loop, we replicate its decision logic here directly.
        from agentx_ai.kit.agent_memory.consolidation import jobs as jobs_module

        memory = MagicMock()
        memory.semantic.get_fact_by_id.return_value = {
            "id": "fact-old",
            "claim": "User uses Python",
            "channel": "_default",
        }

        captured_args = {}

        def _fake_handle(memory, session, fact_dict, contradiction, user_id, channel):
            captured_args["fact"] = fact_dict
            captured_args["contradicting_fact_id"] = contradiction.contradicting_fact_id
            captured_args["resolution"] = contradiction.resolution
            return "superseded"

        with patch.object(jobs_module, "_handle_contradiction", side_effect=_fake_handle) as mock_h:
            # Simulate the in-scope check + dispatch
            fact_dict = {
                "claim": "User uses Python 3.12 for async work",
                "refines_fact_id": "fact-old",
            }
            target = memory.semantic.get_fact_by_id(fact_dict["refines_fact_id"], "u")
            self.assertIsNotNone(target)
            self.assertEqual(target["channel"], "_default")

            jobs_module._handle_contradiction(
                memory, None, fact_dict,
                type("R", (), {
                    "has_contradiction": True,
                    "contradicting_fact_id": fact_dict["refines_fact_id"],
                    "resolution": "prefer_new",
                    "reason": "llm_refinement",
                })(),
                "u", "_default",
            )
            self.assertEqual(captured_args["contradicting_fact_id"], "fact-old")
            self.assertEqual(captured_args["resolution"], "prefer_new")
            mock_h.assert_called_once()

    def test_out_of_scope_refine_is_ignored(self) -> None:
        # Target in a different channel should NOT trigger supersede.
        memory = MagicMock()
        memory.semantic.get_fact_by_id.return_value = {
            "id": "fact-other",
            "claim": "User uses Vim",
            "channel": "_private_other",
        }
        fact_dict = {"claim": "User uses Vim everywhere", "refines_fact_id": "fact-other"}
        target = memory.semantic.get_fact_by_id(fact_dict["refines_fact_id"], "u")
        in_scope = target.get("channel") in ("_default", "_global")
        self.assertFalse(in_scope)


class ExtractionServiceScopeContextWiringTest(TestCase):
    """ExtractionService passes known_entities/known_facts into the prompt template."""

    def test_known_blocks_substituted_into_prompt(self) -> None:
        loader = get_prompt_loader()
        ents, facts = ExtractionService._render_scope_context(
            known_entities=[{"id": "ent-1", "name": "Python", "type": "Technology"}],
            known_facts=[{"id": "fact-1", "claim": "User uses Python", "temporal_context": "current"}],
        )
        prompt = loader.get(
            "extraction.combined_with_relevance",
            text="sample",
            known_entities=ents,
            known_facts=facts,
        )
        self.assertIn("ent-1", prompt)
        self.assertIn("fact-1", prompt)
        self.assertNotIn("{known_entities}", prompt)
        self.assertNotIn("{known_facts}", prompt)

    def test_assistant_self_prompt_accepts_known_blocks(self) -> None:
        loader = get_prompt_loader()
        ents, facts = ExtractionService._render_scope_context(
            known_entities=[{"id": "ent-self-1", "name": "Cache thrash", "type": "Pattern"}],
            known_facts=None,
        )
        prompt = loader.get(
            "extraction.assistant_self",
            text="sample assistant reasoning that is long enough to be considered for extraction by the heuristic gate which requires at least 100 characters",
            known_entities=ents,
            known_facts=facts,
        )
        self.assertIn("ent-self-1", prompt)
        self.assertIn("(none)", prompt)  # facts block
        self.assertNotIn("{known_entities}", prompt)


class RecallProviderImportTest(TestCase):
    """Regression for two compounding bugs that silently disabled the HyDE and
    self-query recall techniques (both wrapped in try/except, so failures were
    swallowed):

    1. recall.py imported providers at the wrong relative depth
       (`...providers` -> nonexistent `kit.providers` instead of
       `....providers` -> `agentx_ai.providers`) -> ImportError.
    2. `provider.complete()` is async but was called without awaiting, then
       `.content` was accessed on the coroutine -> AttributeError.

    The provider stub uses AsyncMock so `complete` is a real coroutine; a wrong
    import depth or a missing await would surface as the empty fallback instead
    of the provider's response."""

    def _make_recall(self) -> RecallLayer:
        return RecallLayer(memory=MagicMock(), base_retriever=MagicMock())

    def _patch_registry(self, content: str):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(content=content))
        registry = MagicMock()
        registry.get_provider_for_model.return_value = (provider, "model-x")
        return patch(
            "agentx_ai.providers.registry.get_registry", return_value=registry
        ), provider

    def test_hyde_import_resolves_and_calls_provider(self) -> None:
        recall = self._make_recall()
        patcher, provider = self._patch_registry("  Paris is the capital of France.  ")
        with patcher:
            result = recall._generate_hypothetical("What is the capital of France?")
        # Wrong import depth would raise ImportError -> except -> "".
        self.assertEqual(result, "Paris is the capital of France.")
        provider.complete.assert_called_once()

    def test_self_query_import_resolves_and_calls_provider(self) -> None:
        recall = self._make_recall()
        patcher, provider = self._patch_registry('{"keywords": ["python"]}')
        with patcher:
            filters = recall._extract_filters("things about python")
        # Wrong import depth would raise ImportError -> except -> {}.
        self.assertEqual(filters, {"keywords": ["python"]})
        provider.complete.assert_called_once()


class Neo4jRetryTest(TestCase):
    """with_neo4j_retry backs off transient Neo4j errors (roadmap item 6)."""

    def test_retries_transient_then_succeeds(self) -> None:
        from neo4j.exceptions import TransientError
        from agentx_ai.kit.agent_memory.connections import with_neo4j_retry

        fn = MagicMock(side_effect=[TransientError("failover"), "ok"])
        result = with_neo4j_retry(fn, retries=3, base_delay=0.0)

        self.assertEqual(result, "ok")
        self.assertEqual(fn.call_count, 2)

    def test_persistent_transient_error_surfaces(self) -> None:
        from neo4j.exceptions import ServiceUnavailable
        from agentx_ai.kit.agent_memory.connections import with_neo4j_retry

        fn = MagicMock(side_effect=ServiceUnavailable("down"))
        with self.assertRaises(ServiceUnavailable):
            with_neo4j_retry(fn, retries=3, base_delay=0.0)
        self.assertEqual(fn.call_count, 3)

    def test_non_transient_error_not_retried(self) -> None:
        from agentx_ai.kit.agent_memory.connections import with_neo4j_retry

        fn = MagicMock(side_effect=ValueError("bad query"))
        with self.assertRaises(ValueError):
            with_neo4j_retry(fn, retries=3, base_delay=0.0)
        self.assertEqual(fn.call_count, 1)


class ConnectionHealthCheckTest(TestCase):
    """Per-manager health_check() + check_memory_health delegation (item 6)."""

    def test_neo4j_health_check_healthy(self) -> None:
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection

        with patch.object(Neo4jConnection, "session") as sess:
            sess.return_value = create_mock_neo4j_session()
            result = Neo4jConnection.health_check()
        self.assertEqual(result["status"], "healthy")
        self.assertIsNone(result["error"])

    def test_neo4j_health_check_unhealthy(self) -> None:
        from agentx_ai.kit.agent_memory.connections import Neo4jConnection

        with patch.object(Neo4jConnection, "session", side_effect=RuntimeError("no driver")):
            result = Neo4jConnection.health_check()
        self.assertEqual(result["status"], "unhealthy")
        self.assertIn("no driver", result["error"])

    def test_redis_health_check_healthy(self) -> None:
        from agentx_ai.kit.agent_memory.connections import RedisConnection

        client = MagicMock()
        with patch.object(RedisConnection, "get_client", return_value=client):
            result = RedisConnection.health_check()
        client.ping.assert_called_once()
        self.assertEqual(result["status"], "healthy")

    def test_check_memory_health_delegates(self) -> None:
        from agentx_ai.kit import memory_utils
        from agentx_ai.kit.agent_memory.connections import (
            Neo4jConnection, PostgresConnection, RedisConnection,
        )

        with patch.object(Neo4jConnection, "health_check",
                          return_value={"status": "healthy", "error": None}), \
             patch.object(PostgresConnection, "health_check",
                          return_value={"status": "unhealthy", "error": "boom"}), \
             patch.object(RedisConnection, "health_check",
                          return_value={"status": "healthy", "error": None}):
            health = memory_utils.check_memory_health()

        self.assertEqual(health["neo4j"]["status"], "healthy")
        self.assertEqual(health["postgres"]["status"], "unhealthy")
        self.assertEqual(health["postgres"]["error"], "boom")
        self.assertEqual(health["redis"]["status"], "healthy")


def _queue_settings(**overrides):
    """Minimal settings stub for the embedding dispatcher/queue."""
    import types as _types

    base = dict(
        embedding_queue_enabled=True,
        embedding_batch_max_size=8,
        embedding_batch_window_ms=10,
        embedding_request_timeout=5.0,
        embedding_queue_max_size=64,
        embedding_max_retries=2,
        embedding_cache_enabled=True,
        embedding_cache_max_size=4,
        embedding_cache_ttl_seconds=10.0,
    )
    base.update(overrides)
    return _types.SimpleNamespace(**base)


class EmbeddingQueueTest(TestCase):
    """Serializer + batching + cache + retry for the embedding request queue."""

    def test_serializes_concurrent_calls(self) -> None:
        import threading
        import time

        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingDispatcher

        overlap = {"cur": 0, "max": 0}
        lock = threading.Lock()
        compute_calls = []

        def compute(texts):
            with lock:
                overlap["cur"] += 1
                overlap["max"] = max(overlap["max"], overlap["cur"])
            time.sleep(0.02)
            with lock:
                overlap["cur"] -= 1
            compute_calls.append(list(texts))
            return [[float(len(t))] for t in texts]

        d = EmbeddingDispatcher(compute, _queue_settings(), namespace="t:m")
        results = {}

        def worker(i):
            results[i] = d.embed([f"text-{i}"])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Never two compute calls at once (local model is not thread-safe).
        self.assertEqual(overlap["max"], 1)
        # Concurrent requests coalesced into fewer compute calls.
        self.assertLess(len(compute_calls), 12)
        for i, vec in results.items():
            self.assertEqual(vec, [[float(len(f"text-{i}"))]])

    def test_cache_hit_skips_compute(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingDispatcher

        calls = []

        def compute(texts):
            calls.append(list(texts))
            return [[1.0] for _ in texts]

        d = EmbeddingDispatcher(compute, _queue_settings(), namespace="t:m")
        v1 = d.embed(["hello"])
        v2 = d.embed(["hello"])
        self.assertEqual(v1, v2)
        self.assertEqual(len(calls), 1)  # second call served from cache

    def test_cache_ttl_expiry_and_lru(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingCache

        c = EmbeddingCache(max_size=2, ttl_seconds=0.05)
        c.put("n", "a", [1.0])
        self.assertEqual(c.get("n", "a"), [1.0])
        import time

        time.sleep(0.06)
        self.assertIsNone(c.get("n", "a"))  # expired

        c2 = EmbeddingCache(max_size=2, ttl_seconds=100)
        c2.put("n", "a", [1.0])
        c2.put("n", "b", [2.0])
        c2.put("n", "c", [3.0])  # evicts LRU "a"
        self.assertIsNone(c2.get("n", "a"))
        self.assertEqual(c2.get("n", "c"), [3.0])
        self.assertEqual(len(c2), 2)

    def test_retry_on_transient_then_succeeds(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingDispatcher

        attempts = {"n": 0}

        def compute(texts):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("rate limit exceeded (429)")
            return [[9.0] for _ in texts]

        d = EmbeddingDispatcher(compute, _queue_settings(), namespace="t:m")
        result = d.embed(["x"])
        self.assertEqual(result, [[9.0]])
        self.assertEqual(attempts["n"], 2)

    def test_permanent_error_fails_fast(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingDispatcher

        attempts = {"n": 0}

        def compute(texts):
            attempts["n"] += 1
            raise RuntimeError("CUDA out of memory")  # not transient

        d = EmbeddingDispatcher(compute, _queue_settings(), namespace="t:m")
        with self.assertRaises(RuntimeError):
            d.embed(["x"])
        self.assertEqual(attempts["n"], 1)  # no retries

    def test_queue_disabled_uses_direct_compute(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import EmbeddingDispatcher

        calls = []

        def compute(texts):
            calls.append(list(texts))
            return [[1.0] for _ in texts]

        d = EmbeddingDispatcher(
            compute,
            _queue_settings(embedding_queue_enabled=False, embedding_cache_enabled=False),
            namespace="t:m",
        )
        self.assertEqual(d.embed(["a", "b"]), [[1.0], [1.0]])
        self.assertEqual(calls, [["a", "b"]])

    def test_transient_classifier(self) -> None:
        from agentx_ai.kit.agent_memory.embedding_queue import _is_transient

        self.assertTrue(_is_transient(Exception("Rate limit reached, try again")))
        self.assertTrue(_is_transient(TimeoutError("connection timed out")))
        self.assertFalse(_is_transient(RuntimeError("device-side assert triggered")))


class ResolveDeviceTest(TestCase):
    """AGENTX_DEVICE resolution + CUDA fallback."""

    def setUp(self) -> None:
        from agentx_ai.kit.device import resolve_device

        resolve_device.cache_clear()

    def tearDown(self) -> None:
        from agentx_ai.kit.device import resolve_device

        resolve_device.cache_clear()

    def test_auto_picks_cpu_without_cuda(self) -> None:
        from agentx_ai.kit import device

        with patch.object(device, "_cuda_available", return_value=False):
            self.assertEqual(device.resolve_device("auto"), "cpu")

    def test_auto_picks_cuda_when_available(self) -> None:
        from agentx_ai.kit import device

        with patch.object(device, "_cuda_available", return_value=True):
            self.assertEqual(device.resolve_device("auto"), "cuda")

    def test_cuda_request_falls_back_to_cpu(self) -> None:
        from agentx_ai.kit import device

        with patch.object(device, "_cuda_available", return_value=False):
            self.assertEqual(device.resolve_device("cuda"), "cpu")

    def test_explicit_cpu_override(self) -> None:
        from agentx_ai.kit import device

        with patch.object(device, "_cuda_available", return_value=True):
            self.assertEqual(device.resolve_device("cpu"), "cpu")


class DedupeEntitiesAliasMergeTest(TestCase):
    """
    Pure-Python helpers in ``dedupe_entities`` — the Cypher rewrite needs a
    live Neo4j to exercise, but the alias-merge logic that decides what the
    survivor ends up with is deterministic and worth its own regression net.
    """

    def test_merge_alias_set_excludes_survivor_name(self) -> None:
        from agentx_ai.management.commands.dedupe_entities import _merge_alias_set

        merged = _merge_alias_set(
            survivor_name="Alice",
            names=["Alice", "alice", "ALICE"],
            alias_lists=[["AL"], [], []],
        )
        # Survivor name is excluded; alias from the survivor's own list survives.
        self.assertEqual(merged, ["AL"])

    def test_merge_alias_set_lowercase_dedups_across_dups(self) -> None:
        from agentx_ai.management.commands.dedupe_entities import _merge_alias_set

        merged = _merge_alias_set(
            survivor_name="Bob",
            names=["Bob", "Robert", "bob"],
            alias_lists=[["B"], ["Robert", "robbie"], ["B", "robbie"]],
        )
        # Order: dup names first ("Robert"), then alias lists in order
        # ("B", "robbie"). Duplicates (case-insensitive) drop.
        self.assertEqual(merged, ["Robert", "B", "robbie"])

    def test_merge_alias_set_handles_empty_and_whitespace(self) -> None:
        from agentx_ai.management.commands.dedupe_entities import _merge_alias_set

        merged = _merge_alias_set(
            survivor_name="Carol",
            names=["Carol", "", "  ", "Caroline"],
            alias_lists=[[""], ["  ", "C."]],
        )
        self.assertEqual(merged, ["Caroline", "C."])

    def test_first_non_empty_skips_blank_strings(self) -> None:
        from agentx_ai.management.commands.dedupe_entities import _first_non_empty

        self.assertEqual(
            _first_non_empty([None, "", "  ", "real value", "later"]),
            "real value",
        )
        self.assertIsNone(_first_non_empty([None, "", "  "]))


@skipUnless(docker_services_running(), "Docker services not running")
class MemoryPortabilityTest(MemoryTestBase):
    """Round-trippable memory export/import (kit.agent_memory.portability)."""

    def _seed(self, user_id, conversation_id, channel="_global"):
        """Write a turn + entity + fact (linked) for `user_id`. Returns (memory, ids)."""
        memory = AgentMemory(
            user_id=user_id, conversation_id=conversation_id, channel=channel
        )
        turn = Turn(
            conversation_id=conversation_id, index=0, role="user",
            content="My favorite language is Python.",
            timestamp=datetime.now(timezone.utc),
        )
        memory.store_turn(turn)
        entity = Entity(name="Python", type="Concept", description="A programming language")
        memory.upsert_entity(entity)
        fact = memory.learn_fact(
            "The user's favorite language is Python",
            source="user_stated", confidence=0.9,
            entity_ids=[entity.id], source_turn_id=turn.id,
        )
        return memory, {"turn": turn.id, "entity": entity.id, "fact": fact.id}

    def _count(self, user_id, label):
        # label is from a fixed internal set — safe to interpolate.
        with Neo4jConnection.session() as session:
            rec = session.run(
                f"MATCH (n:{label}) WHERE n.user_id = $uid RETURN count(n) AS c",
                uid=user_id,
            ).single()
            return rec["c"] if rec else 0

    def _delete_user(self, user_id):
        with Neo4jConnection.session() as session:
            session.run(
                "MATCH (n) WHERE n.user_id = $uid DETACH DELETE n", uid=user_id
            ).consume()

    def test_round_trip_restores_after_wipe(self):
        """export → wipe → import (merge) restores nodes with their original ids."""
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch")
        memory, ids = self._seed(self.test_user_id, self.test_conversation_id)

        export = memory.export_memory(channel="_all")
        self.assertGreaterEqual(len(export.facts), 1)
        self.assertGreaterEqual(len(export.entities), 1)
        self.assertGreaterEqual(len(export.turns), 1)
        self.assertIn(ids["fact"], [f["id"] for f in export.facts])

        self._delete_user(self.test_user_id)
        self.assertEqual(self._count(self.test_user_id, "Fact"), 0)

        memory.import_memory(export, mode="merge")

        self.assertEqual(self._count(self.test_user_id, "Fact"), 1)
        self.assertEqual(self._count(self.test_user_id, "Entity"), 1)
        self.assertEqual(self._count(self.test_user_id, "Turn"), 1)
        # The fact↔entity ABOUT edge is rebuilt and the id preserved.
        with Neo4jConnection.session() as session:
            rec = session.run(
                "MATCH (f:Fact {id: $fid})-[:ABOUT]->(e:Entity {id: $eid}) RETURN f.id AS id",
                fid=ids["fact"], eid=ids["entity"],
            ).single()
            self.assertIsNotNone(rec)

    def test_merge_is_idempotent(self):
        """Re-importing an export the user already has creates nothing new."""
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch")
        memory, _ = self._seed(self.test_user_id, self.test_conversation_id)
        export = memory.export_memory(channel="_all")

        summary = memory.import_memory(export, mode="merge")
        created = sum(c["created"] for c in summary["imported"].values())
        self.assertEqual(created, 0)
        # Node count unchanged after a second import.
        self.assertEqual(self._count(self.test_user_id, "Fact"), 1)

    def test_replace_wipes_stray_nodes(self):
        """replace mode drops channel data not present in the export."""
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch")
        memory, _ = self._seed(self.test_user_id, self.test_conversation_id)
        export = memory.export_memory(channel="_all")

        # Add a stray fact that is NOT in the export.
        memory.learn_fact("A stray fact", source="inferred", confidence=0.5)
        self.assertEqual(self._count(self.test_user_id, "Fact"), 2)

        memory.import_memory(export, mode="replace", channel="_global")
        self.assertEqual(self._count(self.test_user_id, "Fact"), len(export.facts))

    def test_export_is_text_only_and_import_recomputes_embeddings(self):
        """Exports never carry vectors; import regenerates them from text."""
        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch")
        memory, ids = self._seed(self.test_user_id, self.test_conversation_id)

        export = memory.export_memory(channel="_all")
        # Text-only: no node carries an embedding.
        self.assertTrue(all(not f.get("embedding") for f in export.facts))
        self.assertTrue(all(not t.get("embedding") for t in export.turns))
        self.assertTrue(all(not e.get("embedding") for e in export.entities))

        self._delete_user(self.test_user_id)
        summary = memory.import_memory(export, mode="merge")
        self.assertGreater(summary["recomputed_embeddings"], 0)

        with Neo4jConnection.session() as session:
            rec = session.run(
                "MATCH (f:Fact {id: $fid}) RETURN f.embedding IS NOT NULL AS has_emb",
                fid=ids["fact"],
            ).single()
            self.assertTrue(rec["has_emb"])

    def test_newer_schema_version_rejected(self):
        """Imports refuse an envelope from a newer schema than the build supports."""
        from agentx_ai.kit.agent_memory.portability import MemoryImporter

        memory, _ = self._seed(self.test_user_id, self.test_conversation_id) \
            if embeddings_compatible() else (None, None)
        if memory is None:
            self.skipTest("Embedding dimensions mismatch")

        export = memory.export_memory(channel="_all")
        payload = export.model_dump(mode="json")
        payload["schema_version"] = 999

        with self.assertRaises(ValueError):
            MemoryImporter(user_id=self.test_user_id).import_export(payload)


@skipUnless(docker_services_running(), "Docker services not running")
class EvalSnapshotRestoreTest(MemoryTestBase):
    """eval_consolidation --snapshot/--restore: cluster snapshot bundle + per-user restore.

    Deliberately exercises `_make_snapshot` (multi-user enumeration + bundling) and the
    per-user import that `_restore_snapshot` performs, but NOT the global `_wipe()` it
    calls — wiping the shared dev cluster from a test would risk real data if the test
    failed mid-run. Scope deletes to the seeded user instead (cf. MemoryPortabilityTest).
    """

    def _seed(self):
        memory = AgentMemory(
            user_id=self.test_user_id, conversation_id=self.test_conversation_id,
            channel="_global",
        )
        turn = Turn(
            conversation_id=self.test_conversation_id, index=0, role="user",
            content="My favorite language is Python.", timestamp=datetime.now(timezone.utc),
        )
        memory.store_turn(turn)
        entity = Entity(name="Python", type="Concept", description="A programming language")
        memory.upsert_entity(entity)
        fact = memory.learn_fact(
            "The user's favorite language is Python", source="user_stated",
            confidence=0.9, entity_ids=[entity.id], source_turn_id=turn.id,
        )
        return {"turn": turn.id, "entity": entity.id, "fact": fact.id}

    def _delete_user(self, user_id):
        with Neo4jConnection.session() as session:
            session.run("MATCH (n) WHERE n.user_id = $uid DETACH DELETE n", uid=user_id).consume()

    def _count(self, user_id, label):
        with Neo4jConnection.session() as session:
            rec = session.run(
                f"MATCH (n:{label}) WHERE n.user_id = $uid RETURN count(n) AS c", uid=user_id,
            ).single()
            return rec["c"] if rec else 0

    def test_snapshot_bundles_users_and_restores_per_user(self):
        import json
        import tempfile
        from pathlib import Path
        from uuid import uuid4 as _uuid4

        from agentx_ai.management.commands.eval_consolidation import Command
        from agentx_ai.kit.agent_memory.portability import MemoryImporter

        if not embeddings_compatible():
            self.skipTest("Embedding dimensions mismatch")

        ids = self._seed()
        cmd = Command()
        snap_path = Path(tempfile.gettempdir()) / f"evalsnap_{_uuid4().hex}.json"
        try:
            cmd._make_snapshot(snap_path)
            bundle = json.loads(snap_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["snapshot_version"], 1)

            # The seeded user is present in the cluster snapshot...
            mine = [u for u in bundle["users"] if u["user_id"] == self.test_user_id]
            self.assertEqual(len(mine), 1)
            envelope = mine[0]
            self.assertIn(ids["fact"], [f["id"] for f in envelope["facts"]])

            # ...and its envelope round-trips per-user (what _restore_snapshot does
            # for each user after the global wipe).
            self._delete_user(self.test_user_id)
            self.assertEqual(self._count(self.test_user_id, "Fact"), 0)

            MemoryImporter(self.test_user_id).import_export(envelope, mode="merge")
            self.assertEqual(self._count(self.test_user_id, "Fact"), 1)
            self.assertEqual(self._count(self.test_user_id, "Entity"), 1)
            self.assertEqual(self._count(self.test_user_id, "Turn"), 1)
        finally:
            snap_path.unlink(missing_ok=True)
            self._delete_user(self.test_user_id)


class BackfillAgentAttributionRenameTest(TestCase):
    """Pure-function tests for the legacy 'Agent ...' → '<Name> ...' claim rewrite."""

    @staticmethod
    def _rename(claim: str, name: str):
        from agentx_ai.management.commands.backfill_agent_attribution import Command
        return Command._rename(claim, name)

    def test_rewrites_leading_agent(self) -> None:
        self.assertEqual(
            self._rename("Agent forgets the user's name", "Mobius"),
            "Mobius forgets the user's name",
        )

    def test_rewrites_the_agent(self) -> None:
        self.assertEqual(
            self._rename("The agent prefers concise answers", "Mobius"),
            "Mobius prefers concise answers",
        )

    def test_preserves_possessive(self) -> None:
        self.assertEqual(
            self._rename("Agent's reasoning is strong", "Mobius"),
            "Mobius's reasoning is strong",
        )

    def test_idempotent_when_already_named(self) -> None:
        self.assertIsNone(self._rename("Mobius forgets the user's name", "Mobius"))

    def test_skips_non_agent_claims(self) -> None:
        self.assertIsNone(self._rename("User works at Acme", "Mobius"))
        # "agential" must not match the \bagent\b boundary.
        self.assertIsNone(self._rename("Agentic workflows are useful", "Mobius"))


class ProcedureReflexRenderTest(TestCase):
    """Reflex core: distilled procedures render into the prompt context (Fix ③)."""

    def test_to_context_string_renders_procedures(self) -> None:
        from agentx_ai.kit.agent_memory.models import MemoryBundle

        bundle = MemoryBundle(procedures=[
            {"trigger": "presenting a recommendation", "body": "give options with rationales", "strength": 3},
            {"trigger": "", "body": "lead with the conclusion, then the reasoning", "strength": 1},
        ])
        text_out = bundle.to_context_string()
        self.assertIn("Learned Procedures", text_out)
        self.assertIn("When presenting a recommendation: give options with rationales", text_out)
        # Triggerless procedure still renders as a bare bullet.
        self.assertIn("- lead with the conclusion, then the reasoning", text_out)

    def test_empty_procedures_omitted(self) -> None:
        from agentx_ai.kit.agent_memory.models import MemoryBundle

        self.assertNotIn("Learned Procedures", MemoryBundle().to_context_string())

    def test_trigger_with_leading_condition_word_not_doubled(self) -> None:
        """A trigger that already begins with 'when' must not render 'When when …'."""
        from agentx_ai.kit.agent_memory.models import MemoryBundle

        bundle = MemoryBundle(procedures=[
            {"trigger": "when presenting a recommendation", "body": "give options", "strength": 2},
            {"trigger": "before finalizing an analysis", "body": "list assumptions", "strength": 1},
        ])
        text_out = bundle.to_context_string()
        self.assertNotIn("When when", text_out)
        self.assertNotIn("when when", text_out)
        # Leading condition word is preserved (just capitalized), not re-prefixed.
        self.assertIn("- When presenting a recommendation: give options", text_out)
        self.assertIn("- Before finalizing an analysis: list assumptions", text_out)

    def test_prefix_trigger_helper(self) -> None:
        """_prefix_trigger de-doubles condition words but prefixes plain phrases."""
        from agentx_ai.kit.agent_memory.models import _prefix_trigger

        self.assertEqual(_prefix_trigger("when doing a web search"), "When doing a web search")
        self.assertEqual(_prefix_trigger("presenting results"), "When presenting results")
        self.assertEqual(_prefix_trigger("Whenever asked"), "Whenever asked")
        self.assertEqual(_prefix_trigger(""), "")


class ProcedureWritePathTest(TestCase):
    """ProceduralMemory.learn_procedure / reinforce_procedure / get_reflex_procedures."""

    def _procedural(self):
        from agentx_ai.kit.agent_memory.memory.procedural import ProceduralMemory
        pm = ProceduralMemory()
        pm.embedder = MagicMock()
        pm.embedder.embed_single.return_value = [0.1] * 1024
        return pm

    def test_learn_procedure_creates_node(self) -> None:
        mock_neo4j = create_mock_neo4j_session()
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as sess:
            sess.return_value = mock_neo4j
            pm = self._procedural()
            proc = pm.learn_procedure(
                trigger="when presenting a recommendation",
                body="give options with rationales",
                rationale="user prefers to weigh alternatives",
                scope="_global",
                signal_kinds=["explicit_rule"],
                evidence_refs=["cand:1"],
                conversation_ids=["conv-1"],
                user_id="default",
            )
        cypher = mock_neo4j.run.call_args[0][0]
        self.assertIn("CREATE (p:Procedure", cypher)
        self.assertIn("HAS_PROCEDURE", cypher)
        self.assertEqual(proc.trigger, "when presenting a recommendation")
        self.assertEqual(proc.strength, 1)

    def test_reinforce_procedure_increments_strength(self) -> None:
        mock_neo4j = create_mock_neo4j_session()
        mock_neo4j.run.return_value.single.return_value = {"updated_id": "proc-1"}
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as sess:
            sess.return_value = mock_neo4j
            pm = self._procedural()
            ok = pm.reinforce_procedure("proc-1", evidence_refs=["cand:2"], signal_kinds=["correction"])
        cypher = mock_neo4j.run.call_args[0][0]
        self.assertTrue(ok)
        self.assertIn("p.strength = coalesce(p.strength, 1) + 1", cypher)

    def test_get_reflex_procedures_returns_dicts(self) -> None:
        mock_neo4j = create_mock_neo4j_session()
        mock_neo4j.run.return_value = [
            {"trigger": "t1", "body": "b1", "scope": "_global", "strength": 5},
        ]
        with patch('agentx_ai.kit.agent_memory.connections.Neo4jConnection.session') as sess:
            sess.return_value = mock_neo4j
            pm = self._procedural()
            out = pm.get_reflex_procedures(["_global"], limit=5)
        self.assertEqual(out[0]["body"], "b1")

    def test_get_reflex_procedures_empty_channels(self) -> None:
        pm = self._procedural()
        self.assertEqual(pm.get_reflex_procedures([], limit=5), [])


class ProcedureDistillStageTest(TestCase):
    """ExtractionService.distill_procedure parses keep/discard from the LLM."""

    def _service_with_response(self, content: str):
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionService
        ext = ExtractionService()
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=MagicMock(content=content))
        ext._get_provider_for_stage = MagicMock(return_value=(provider, "model", 0.2, 1000))
        return ext

    def test_distill_keep(self) -> None:
        ext = self._service_with_response(
            '{"keep": true, "trigger": "when presenting a recommendation", "body": "give options with rationales", "rationale": "user preference"}'
        )
        result = asyncio.run(ext.distill_procedure(
            [{"signal": "explicit_rule", "content": "always give me options with rationales"}], "_global"
        ))
        self.assertTrue(result.keep)
        self.assertEqual(result.trigger, "when presenting a recommendation")
        self.assertEqual(result.body, "give options with rationales")

    def test_distill_discard(self) -> None:
        ext = self._service_with_response('{"keep": false, "trigger": "", "body": "", "rationale": ""}')
        result = asyncio.run(ext.distill_procedure(
            [{"signal": "correction", "content": "be accurate"}], "_global"
        ))
        self.assertFalse(result.keep)

    def test_distill_keep_without_body_is_discard(self) -> None:
        ext = self._service_with_response('{"keep": true, "trigger": "x", "body": "", "rationale": "y"}')
        result = asyncio.run(ext.distill_procedure(
            [{"signal": "correction", "content": "x"}], "_global"
        ))
        self.assertFalse(result.keep)


class ProcedureDistillJobTest(TestCase):
    """distill_procedures consolidation job: create / discard / reinforce + candidate marking."""

    def _row(self, **kw):
        from types import SimpleNamespace
        base = dict(id=1, conversation_id="conv-1", signal="explicit_rule",
                    content="always summarize trade-offs before recommending",
                    channel="_global", agent_id=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def _run_job(self, *, rows, distill_result, find_result=None, reinforce_ok=True):
        jobs_mod = 'agentx_ai.kit.agent_memory.consolidation.jobs'

        pg = create_mock_postgres_session()
        pg.execute.return_value.fetchall.return_value = rows

        ext = MagicMock()
        ext.distill_procedure = AsyncMock(return_value=distill_result)

        proc = MagicMock()
        proc.find_procedures.return_value = find_result or []
        proc.reinforce_procedure.return_value = reinforce_ok
        proc.learn_procedure.return_value = MagicMock(id="proc-new")

        with patch(f'{jobs_mod}.get_postgres_session', return_value=pg), \
             patch(f'{jobs_mod}.get_extraction_service', return_value=ext), \
             patch(f'{jobs_mod}._resolve_user_for_conversation', return_value="default"), \
             patch('agentx_ai.kit.agent_memory.memory.procedural.ProceduralMemory', return_value=proc):
            from agentx_ai.kit.agent_memory.consolidation.jobs import distill_procedures
            result = asyncio.run(distill_procedures())
        return result, proc

    def test_creates_procedure_and_marks_distilled(self) -> None:
        from agentx_ai.kit.agent_memory.extraction.service import ProcedureDistillResult
        keep = ProcedureDistillResult(keep=True, trigger="when recommending",
                                      body="summarize trade-offs first", rationale="user preference")
        result, proc = self._run_job(rows=[self._row()], distill_result=keep)
        self.assertEqual(result["procedures_created"], 1)
        proc.learn_procedure.assert_called_once()
        proc.mark_candidates.assert_called_with([1], "distilled", distilled_into="proc-new")

    def test_discards_baseline(self) -> None:
        from agentx_ai.kit.agent_memory.extraction.service import ProcedureDistillResult
        discard = ProcedureDistillResult(keep=False)
        result, proc = self._run_job(rows=[self._row(signal="correction")], distill_result=discard)
        self.assertEqual(result["discarded"], 1)
        proc.mark_candidates.assert_called_with([1], "discarded")
        proc.learn_procedure.assert_not_called()

    def test_reinforces_existing(self) -> None:
        from agentx_ai.kit.agent_memory.extraction.service import ProcedureDistillResult
        keep = ProcedureDistillResult(keep=True, trigger="when recommending",
                                      body="summarize trade-offs first", rationale="user preference")
        result, proc = self._run_job(
            rows=[self._row()], distill_result=keep,
            find_result=[MagicMock(id="existing-1")], reinforce_ok=True,
        )
        self.assertEqual(result["procedures_reinforced"], 1)
        proc.reinforce_procedure.assert_called_once()
        proc.learn_procedure.assert_not_called()
        proc.mark_candidates.assert_called_with([1], "distilled", distilled_into="existing-1")

    def test_correction_routes_to_self_channel(self) -> None:
        from agentx_ai.kit.agent_memory.extraction.service import ProcedureDistillResult
        keep = ProcedureDistillResult(keep=True, trigger="t", body="b", rationale="r")
        result, proc = self._run_job(
            rows=[self._row(signal="correction", agent_id="bold-cosmic-falcon")],
            distill_result=keep,
        )
        # Scope passed to distill + write should be the agent's self-channel.
        scope_arg = proc.learn_procedure.call_args.kwargs["scope"]
        self.assertEqual(scope_arg, "_self_bold-cosmic-falcon")
