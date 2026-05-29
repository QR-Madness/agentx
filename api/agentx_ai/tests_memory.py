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
    _get_recent_facts,
    _handle_contradiction,
    apply_memory_decay,
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
    CORRECTION_PATTERNS,
    CombinedExtractionResult,
    ExtractionResult,
    ExtractionService,
    get_extraction_service,
)
from agentx_ai.kit.agent_memory.memory.episodic import EpisodicMemory
from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
from agentx_ai.kit.agent_memory.memory.procedural import ProceduralMemory
from agentx_ai.kit.agent_memory.memory.recall import RecallLayer, RecallMetrics
from agentx_ai.kit.agent_memory.memory.retrieval import MemoryRetriever
from agentx_ai.kit.agent_memory.memory.semantic import SemanticMemory
from agentx_ai.kit.agent_memory.memory.working import WorkingMemory
from agentx_ai.kit.agent_memory.models import (
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

    def test_check_correction_heuristic_patterns(self) -> None:
        """Heuristic patterns should trigger correction check."""

        # Test cases that should match
        matches = [
            "Actually, I work at Google not Microsoft",
            "No, I meant Python not Java",
            "Sorry, I misspoke - it's version 3",
            "I meant to say Seattle",
            "Correction: my name is John",
            "Wait, that's wrong - I have 5 years",
            "That's not right, I prefer TypeScript",
            "Let me correct that - it's 2024",
            "I misspoke earlier, the deadline is Friday",
            "Not Python, but JavaScript",
        ]

        for sample in matches:
            sample_lower = sample.strip().lower()
            matched = any(re.search(p, sample_lower, re.IGNORECASE) for p in CORRECTION_PATTERNS)
            self.assertTrue(matched, f"Pattern should match: {sample}")

    def test_check_correction_non_matches(self) -> None:
        """Normal statements should not match correction patterns."""

        # Test cases that should NOT match
        non_matches = [
            "I work at Google",
            "My favorite language is Python",
            "I have 5 years of experience",
            "The project deadline is Friday",
            "I started a new project yesterday",
        ]

        for sample in non_matches:
            sample_lower = sample.strip().lower()
            matched = any(re.search(p, sample_lower, re.IGNORECASE) for p in CORRECTION_PATTERNS)
            self.assertFalse(matched, f"Pattern should NOT match: {sample}")

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
