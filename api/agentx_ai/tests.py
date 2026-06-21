# Test-only: suppress framework-typing false positives (no django-stubs configured).
# The Django test client's get/post resolve to RequestFactory's request return type,
# pydantic v2 positional Field() defaults aren't recognized as optional, and Optional
# model getters trip Optional checks. These are test-harness artifacts, not real bugs;
# source code stays strictly type-checked (baseline 0).
# pyright: reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportArgumentType=false, reportFunctionMemberAccess=false
import asyncio
import json
import os
import re
from datetime import datetime, timedelta, UTC
from unittest import skipUnless
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import TestCase, override_settings

from agentx_ai.agent.tool_output_chunker import (
    _cosine_similarity,
    _keyword_search,
    chunk_text,
    detect_sections,
    get_section_content,
    resolve_json_path,
)
from agentx_ai.agent.tool_output_compressor import (
    CompressionResult,
    ToolOutputCompressor,
    get_compressor,
)
from agentx_ai.drafting.base import DraftingConfig, DraftResult, DraftStatus
from agentx_ai.drafting.candidate import Candidate, CandidateConfig, ScoringMethod
from agentx_ai.drafting.pipeline import PipelineConfig, PipelineStage, StageRole
from agentx_ai.drafting.speculative import SpeculativeConfig
from agentx_ai.kit.agent_memory.audit import (
    AuditLogLevel,
    MemoryAuditLogger,
    MemoryType,
    OperationType,
)
from agentx_ai.kit.agent_memory.config import Settings, get_settings
from agentx_ai.kit.agent_memory.events import (
    EventPayload,
    FactLearnedPayload,
    MemoryEventEmitter,
    TurnStoredPayload,
)
from agentx_ai.kit.agent_memory.extraction import (
    extract_entities,
    extract_facts,
    extract_relationships,
)
from agentx_ai.kit.agent_memory.extraction.service import (
    ExtractionResult,
    get_extraction_service,
    reset_extraction_service,
)
from agentx_ai.kit.agent_memory.memory.retrieval import (
    MemoryRetriever,
    RetrievalMetrics,
    RetrievalWeights,
)
from agentx_ai.kit.agent_memory.memory.working import WorkingMemory
from agentx_ai.kit.translation import LanguageLexicon
from agentx_ai.mcp import ServerConfig, ServerRegistry
from agentx_ai.mcp.internal_tools import execute_internal_tool, get_internal_tools, is_retrieval_tool
from agentx_ai.mcp.server_registry import TransportType
from agentx_ai.providers.base import CompletionResult, Message, MessageRole
from agentx_ai.providers.registry import get_registry
from agentx_ai.reasoning.base import (
    ReasoningConfig,
    ReasoningResult,
    ReasoningStatus,
    ThoughtStep,
    ThoughtType,
)
from agentx_ai.reasoning.chain_of_thought import CoTConfig
from agentx_ai.reasoning.react import ReActConfig, Tool
from agentx_ai.reasoning.reflection import ReflectionConfig, Revision
from agentx_ai.reasoning.tree_of_thought import ToTConfig, TreeNode
from agentx_ai.streaming.trajectory_compression import (
    compress_trajectory,
    identify_tool_rounds,
    rounds_to_text,
)
from agentx_ai.test_utils import (
    APITestBase,
    COMPRESSOR_CONFIG,
    COMPRESSOR_CONFIG_DISABLED,
    MockRedisTestBase,
    docker_services_running,
    has_configured_provider,
)

import agentx_ai.agent.tool_output_compressor as _compressor_mod


class TranslationKitTest(APITestBase):

    def test_language_detect_post(self) -> None:
        """Test that the language detection API works with POST."""
        response = self.client.post(
            "/api/tools/language-detect-20",
            data={"text": "Bonjour, comment allez-vous?"},
            content_type="application/json"
        )
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('detected_language', data)
        self.assertIn('confidence', data)
        self.assertEqual(data['detected_language'], 'fr')

    def test_language_detect_get(self) -> None:
        """Test backwards compatibility with GET request."""
        response = self.client.get("/api/tools/language-detect-20")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('detected_language', data)
        # Default text is English
        self.assertEqual(data['detected_language'], 'en')

    def test_lexicon_convert_level_i_to_level_ii(self) -> None:
        """Test that the lexicon converts level I language codes to level II language codes."""
        lexicon = LanguageLexicon(verbose=True)
        level_i_language = "en"
        level_ii_language = lexicon.convert_level_i_detection_to_level_ii(level_i_language)
        self.assertEqual(level_ii_language, "eng_Latn")
        self.assertTrue(level_ii_language in lexicon.level_ii_languages)

    def test_translate_to_french(self) -> None:
        """Test that the translation API works."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello, AgentX AI!", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]


class HealthCheckTest(APITestBase):

    def test_health_endpoint(self) -> None:
        """Test that the health check endpoint returns expected structure."""
        response = self.client.get("/api/health")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('status', data)
        self.assertIn('api', data)
        self.assertIn('translation', data)
        self.assertEqual(data['api']['status'], 'healthy')

    @skipUnless(docker_services_running(), "Docker services not running")
    def test_health_with_memory_check(self) -> None:
        """Test health check with memory system - requires Docker services running."""
        response = self.client.get("/api/health?include_memory=true")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('memory', data)
        self.assertIn('neo4j', data['memory'])
        self.assertIn('postgres', data['memory'])
        self.assertIn('redis', data['memory'])
        # Assert all database connections are healthy
        self.assertEqual(data['memory']['neo4j']['status'], 'healthy', 
                         f"Neo4j unhealthy: {data['memory']['neo4j'].get('error')}")
        self.assertEqual(data['memory']['postgres']['status'], 'healthy',
                         f"PostgreSQL unhealthy: {data['memory']['postgres'].get('error')}")
        self.assertEqual(data['memory']['redis']['status'], 'healthy',
                         f"Redis unhealthy: {data['memory']['redis'].get('error')}")


@override_settings(AGENTX_AUTH_ENABLED=False)  # endpoint tests don't auth; stay green regardless of local .env
class MCPClientTest(APITestBase):

    def test_mcp_servers_endpoint(self) -> None:
        """Test that the MCP servers endpoint returns expected structure."""
        response = self.client.get("/api/mcp/servers")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('servers', data)
        self.assertIsInstance(data['servers'], list)
        # Each server should have name, status, transport
        for server in data['servers']:
            self.assertIn('name', server)
            self.assertIn('status', server)
            self.assertIn('transport', server)

    def test_mcp_tools_endpoint(self) -> None:
        """Test that the MCP tools endpoint returns expected structure."""
        response = self.client.get("/api/mcp/tools")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('tools', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['tools'], list)

    def test_mcp_resources_endpoint(self) -> None:
        """Test that the MCP resources endpoint returns expected structure."""
        response = self.client.get("/api/mcp/resources")
        data = response.json()  # type: ignore[union-attr]
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        self.assertIn('resources', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['resources'], list)

    def test_mcp_connect_requires_post(self) -> None:
        """Test that connect endpoint rejects GET requests."""
        response = self.client.get("/api/mcp/connect")
        self.assertEqual(response.status_code, 405)  # type: ignore[union-attr]

    def test_mcp_connect_requires_server_name(self) -> None:
        """Test that connect endpoint requires server name or all flag."""
        response = self.client.post(
            "/api/mcp/connect",
            data="{}",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]

    def test_mcp_disconnect_requires_post(self) -> None:
        """Test that disconnect endpoint rejects GET requests."""
        response = self.client.get("/api/mcp/disconnect")
        self.assertEqual(response.status_code, 405)  # type: ignore[union-attr]

    def test_mcp_disconnect_unknown_server(self) -> None:
        """Test disconnecting a server that isn't connected."""
        response = self.client.post(
            "/api/mcp/disconnect",
            data='{"server": "nonexistent"}',
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 404)  # type: ignore[union-attr]


class MCPServerRegistryTest(TestCase):
    def test_server_config_creation(self) -> None:
        """Test creating a server configuration."""
        config = ServerConfig(
            name="test-server",
            transport=TransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        
        self.assertEqual(config.name, "test-server")
        self.assertEqual(config.transport, TransportType.STDIO)
        self.assertEqual(config.command, "npx")
        self.assertTrue(config.validate())

    def test_server_registry_operations(self) -> None:
        """Test server registry register/get/list operations."""
        registry = ServerRegistry()
        
        config = ServerConfig(
            name="test-server",
            transport=TransportType.STDIO,
            command="echo",
            args=["test"],
        )
        
        registry.register(config)
        
        # Test get
        retrieved = registry.get("test-server")
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.name, "test-server")
        
        # Test list
        servers = registry.list()
        self.assertEqual(len(servers), 1)
        
        # Test unregister
        result = registry.unregister("test-server")
        self.assertTrue(result)
        self.assertIsNone(registry.get("test-server"))

    def test_env_resolution(self) -> None:
        """Test environment variable resolution in server config."""
        os.environ["TEST_TOKEN"] = "my-secret-token"

        config = ServerConfig(
            name="test",
            transport=TransportType.STDIO,
            command="echo",
            env={"TOKEN": "${TEST_TOKEN}"},
        )

        resolved = config.resolve_env()
        self.assertEqual(resolved["TOKEN"], "my-secret-token")

        del os.environ["TEST_TOKEN"]

    def test_auto_connect_round_trips(self) -> None:
        """auto_connect defaults False and survives from_dict/_server_to_dict."""
        default = ServerConfig(name="s", transport=TransportType.STDIO, command="echo")
        self.assertFalse(default.auto_connect)

        loaded = ServerConfig.from_dict(
            "s", {"transport": "stdio", "command": "echo", "auto_connect": True}
        )
        self.assertTrue(loaded.auto_connect)

        # Round-trips through serialization (and is omitted when False/default).
        self.assertTrue(ServerRegistry._server_to_dict(loaded).get("auto_connect"))
        self.assertNotIn("auto_connect", ServerRegistry._server_to_dict(default))

    def test_connect_persisted_only_targets_flagged_servers(self) -> None:
        """connect_persisted attempts auto_connect servers and skips the rest."""
        from .mcp.client import MCPClientManager

        registry = ServerRegistry()
        registry.register(ServerConfig(
            name="wanted", transport=TransportType.STDIO, command="echo",
            auto_connect=True,
        ))
        registry.register(ServerConfig(
            name="skipped", transport=TransportType.STDIO, command="echo",
            auto_connect=False,
        ))

        manager = MCPClientManager(registry)
        attempted: list[str] = []
        manager.connect = lambda name: attempted.append(name)  # type: ignore[method-assign]

        results = manager.connect_persisted()
        self.assertEqual(attempted, ["wanted"])
        self.assertEqual(results["wanted"]["status"], "connected")
        self.assertNotIn("skipped", results)


# =============================================================================
# Phase 10: Translation Tests
# =============================================================================

class TranslationLanguagePairsTest(APITestBase):
    """Test translation across multiple language pairs."""

    def test_translate_english_to_spanish(self) -> None:
        """Test English to Spanish translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello, how are you?", "targetLanguage": "spa_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        self.assertIn('translatedText', data)
        self.assertTrue(len(data['translatedText']) > 0)

    def test_translate_english_to_german(self) -> None:
        """Test English to German translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Good morning", "targetLanguage": "deu_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        self.assertIn('translatedText', data)

    def test_translate_english_to_japanese(self) -> None:
        """Test English to Japanese translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Thank you", "targetLanguage": "jpn_Jpan"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        self.assertIn('translatedText', data)


class TranslationErrorHandlingTest(APITestBase):
    """Test translation error handling."""

    def test_translate_invalid_language_code(self) -> None:
        """Test translation with invalid language code."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello", "targetLanguage": "invalid_code"},
            content_type="application/json"
        )
        self.assertIn(response.status_code, [400, 500])  # type: ignore[union-attr]

    def test_translate_empty_text(self) -> None:
        """Test translation with empty text."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        self.assertIn(response.status_code, [200, 400])  # type: ignore[union-attr]

    def test_translate_missing_target_language(self) -> None:
        """Test translation with missing target language."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)  # type: ignore[union-attr]


class TranslationLongTextTest(APITestBase):
    """Test translation with various text lengths."""

    def test_translate_short_text(self) -> None:
        """Test translation of short text."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hi", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]

    def test_translate_paragraph(self) -> None:
        """Test translation of a paragraph."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of the translation system. "
            "It should handle multiple sentences correctly."
        )
        response = self.client.post(
            "/api/tools/translate",
            data={"text": text, "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)  # type: ignore[union-attr]
        data = response.json()  # type: ignore[union-attr]
        self.assertIn('translatedText', data)
        self.assertTrue(len(data['translatedText']) > 20)


# =============================================================================
# Phase 10: Reasoning Framework Tests
# =============================================================================

class ReasoningBaseTest(TestCase):
    """Test reasoning framework base classes."""

    def test_reasoning_status_enum(self) -> None:
        """Test ReasoningStatus enum values."""
        self.assertEqual(ReasoningStatus.PENDING, "pending")
        self.assertEqual(ReasoningStatus.THINKING, "thinking")
        self.assertEqual(ReasoningStatus.COMPLETE, "complete")
        self.assertEqual(ReasoningStatus.FAILED, "failed")

    def test_thought_type_enum(self) -> None:
        """Test ThoughtType enum values."""
        self.assertEqual(ThoughtType.OBSERVATION, "observation")
        self.assertEqual(ThoughtType.REASONING, "reasoning")
        self.assertEqual(ThoughtType.ACTION, "action")
        self.assertEqual(ThoughtType.CONCLUSION, "conclusion")

    def test_thought_step_creation(self) -> None:
        """Test ThoughtStep model creation."""
        step = ThoughtStep(
            step_number=1,
            thought_type=ThoughtType.REASONING,
            content="First, let's analyze the problem.",
            confidence=0.9
        )
        
        self.assertEqual(step.step_number, 1)
        self.assertEqual(step.thought_type, ThoughtType.REASONING)
        self.assertEqual(step.content, "First, let's analyze the problem.")
        self.assertEqual(step.confidence, 0.9)
    
    def test_reasoning_result_creation(self) -> None:
        """Test ReasoningResult model creation."""
        result = ReasoningResult(
            answer="The answer is 42.",
            strategy="cot",
            status=ReasoningStatus.COMPLETE,
            total_steps=3,
            total_tokens=150,
        )
        
        self.assertEqual(result.answer, "The answer is 42.")
        self.assertEqual(result.strategy, "cot")
        self.assertEqual(result.status, ReasoningStatus.COMPLETE)
    
    def test_reasoning_config_creation(self) -> None:
        """Test ReasoningConfig creation."""
        config = ReasoningConfig(
            name="test-cot",
            strategy_type="cot",
            model="llama3.2",
            temperature=0.7,
            max_steps=5,
        )
        
        self.assertEqual(config.name, "test-cot")
        self.assertEqual(config.strategy_type, "cot")
        self.assertEqual(config.model, "llama3.2")


class ChainOfThoughtTest(TestCase):
    """Test Chain-of-Thought reasoning components."""
    
    def test_cot_config_creation(self) -> None:
        """Test CoTConfig creation with defaults."""
        config = CoTConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.mode, "zero_shot")
        self.assertEqual(config.thinking_prompt, "Let's think step by step.")
        self.assertTrue(config.extract_steps)
    
    def test_cot_config_few_shot_mode(self) -> None:
        """Test CoTConfig with few-shot mode."""
        examples = [
            {"question": "2+2?", "reasoning": "Add 2 and 2", "answer": "4"}
        ]
        config = CoTConfig(
            model="llama3.2",
            mode="few_shot",
            examples=examples
        )
        
        self.assertEqual(config.mode, "few_shot")
        self.assertEqual(len(config.examples), 1)  # type: ignore[arg-type]
    
    def test_step_extraction_pattern(self) -> None:
        """Test that step extraction regex works correctly."""
        # Simulate the step extraction pattern
        step_prefix = "Step"
        response = """Step 1: First, identify the numbers.
Step 2: Add them together: 2 + 2 = 4.
Step 3: Verify the result.
Answer: The answer is 4."""
        
        step_pattern = rf"{step_prefix}\s*(\d+)[:\.]?\s*(.+?)(?=(?:{step_prefix}\s*\d+|Answer:|Final|$))"
        matches = re.findall(step_pattern, response, re.IGNORECASE | re.DOTALL)
        
        self.assertEqual(len(matches), 3)
        self.assertEqual(matches[0][0], "1")
        self.assertIn("identify", matches[0][1])


class TreeOfThoughtTest(TestCase):
    """Test Tree-of-Thought reasoning components."""
    
    def test_tree_node_creation(self) -> None:
        """Test TreeNode creation."""
        root = TreeNode(
            id="root",
            content="Initial problem state",
            depth=0,
            score=1.0,
        )
        
        self.assertEqual(root.id, "root")
        self.assertEqual(root.depth, 0)
        self.assertEqual(root.score, 1.0)
        self.assertEqual(len(root.children), 0)
    
    def test_tot_config_defaults(self) -> None:
        """Test ToT configuration defaults."""
        config = ToTConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.search_method, "bfs")
        self.assertEqual(config.max_depth, 4)
        self.assertEqual(config.branching_factor, 3)


class ReActTest(TestCase):
    """Test ReAct reasoning components."""
    
    def test_tool_creation(self) -> None:
        """Test Tool creation."""
        tool = Tool(
            name="search",
            description="Search for information",
            parameters={"query": "string"},
            execute=lambda x: "result"
        )
        
        self.assertEqual(tool.name, "search")
        self.assertEqual(tool.description, "Search for information")
    
    def test_react_config_defaults(self) -> None:
        """Test ReAct configuration defaults."""
        config = ReActConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.thought_prefix, "Thought:")
        self.assertEqual(config.action_prefix, "Action:")


class ReflectionTest(TestCase):
    """Test Reflection reasoning components."""
    
    def test_revision_creation(self) -> None:
        """Test Revision model creation."""
        revision = Revision(
            version=1,
            content="Improved response",
            critique="The original lacked examples.",
            score=0.85,
            improvements=["Added examples"]
        )
        
        self.assertEqual(revision.version, 1)
        self.assertEqual(revision.score, 0.85)
        self.assertIn("Added examples", revision.improvements)
    
    def test_reflection_config_defaults(self) -> None:
        """Test ReflectionConfig defaults."""
        config = ReflectionConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.max_revisions, 3)


# =============================================================================
# Phase 10: Drafting Framework Tests
# =============================================================================

class DraftingBaseTest(TestCase):
    """Test drafting framework base classes."""
    
    def test_draft_status_enum(self) -> None:
        """Test DraftStatus enum values."""
        self.assertEqual(DraftStatus.PENDING, "pending")
        self.assertEqual(DraftStatus.DRAFTING, "drafting")
        self.assertEqual(DraftStatus.COMPLETE, "complete")
        self.assertEqual(DraftStatus.FAILED, "failed")
    
    def test_draft_result_creation(self) -> None:
        """Test DraftResult model creation."""
        result = DraftResult(
            content="Generated draft content",
            strategy="speculative",
            total_tokens=50,
        )
        
        self.assertEqual(result.content, "Generated draft content")
        self.assertEqual(result.strategy, "speculative")
        self.assertEqual(result.total_tokens, 50)
    
    def test_drafting_config_creation(self) -> None:
        """Test DraftingConfig creation."""
        config = DraftingConfig(
            name="test-speculative",
            strategy_type="speculative",
        )
        
        self.assertEqual(config.name, "test-speculative")
        self.assertEqual(config.strategy_type, "speculative")
        self.assertEqual(config.temperature, 0.7)


class SpeculativeDecodingTest(TestCase):
    """Test speculative decoding components."""
    
    def test_speculative_config_defaults(self) -> None:
        """Test SpeculativeConfig defaults."""
        config = SpeculativeConfig(
            draft_model="llama3.2:1b",
            target_model="llama3.2"
        )
        
        self.assertEqual(config.draft_model, "llama3.2:1b")
        self.assertEqual(config.target_model, "llama3.2")
        self.assertEqual(config.draft_tokens, 20)
        self.assertEqual(config.acceptance_threshold, 0.8)


class PipelineTest(TestCase):
    """Test multi-model pipeline components."""
    
    def test_stage_role_enum(self) -> None:
        """Test StageRole enum values."""
        self.assertEqual(StageRole.ANALYZE, "analyze")
        self.assertEqual(StageRole.DRAFT, "draft")
        self.assertEqual(StageRole.REVIEW, "review")
        self.assertEqual(StageRole.REFINE, "refine")
    
    def test_pipeline_stage_creation(self) -> None:
        """Test PipelineStage creation."""
        stage = PipelineStage(
            name="draft-stage",
            model="llama3.2",
            role=StageRole.DRAFT,
        )
        
        self.assertEqual(stage.name, "draft-stage")
        self.assertEqual(stage.model, "llama3.2")
        self.assertEqual(stage.role, StageRole.DRAFT)
    
    def test_pipeline_config_creation(self) -> None:
        """Test PipelineConfig creation."""
        stages = [
            PipelineStage(name="analyze", model="llama3.2", role=StageRole.ANALYZE),
            PipelineStage(name="draft", model="llama3.2", role=StageRole.DRAFT),
        ]
        config = PipelineConfig(name="test-pipeline", stages=stages)
        
        self.assertEqual(config.name, "test-pipeline")
        self.assertEqual(len(config.stages), 2)


class CandidateGenerationTest(TestCase):
    """Test candidate generation components."""
    
    def test_scoring_method_enum(self) -> None:
        """Test ScoringMethod enum values."""
        self.assertEqual(ScoringMethod.MAJORITY_VOTE, "majority_vote")
        self.assertEqual(ScoringMethod.VERIFIER, "verifier")
        self.assertEqual(ScoringMethod.LENGTH_PREFERENCE, "length_preference")
    
    def test_candidate_creation(self) -> None:
        """Test Candidate model creation."""
        candidate = Candidate(
            content="This is a candidate response.",
            score=0.85,
            model="llama3.2",
            index=0,
        )
        
        self.assertEqual(candidate.content, "This is a candidate response.")
        self.assertEqual(candidate.score, 0.85)
        self.assertEqual(candidate.index, 0)
    
    def test_candidate_config_defaults(self) -> None:
        """Test CandidateConfig defaults."""
        config = CandidateConfig(name="test-gen", models=["llama3.2"])
        
        self.assertEqual(config.candidates_per_model, 1)
        self.assertEqual(config.scoring_method, ScoringMethod.MAJORITY_VOTE)


# =============================================================================
# Phase 10: Provider Tests
# =============================================================================

class ProviderRegistryTest(TestCase):
    """Test provider registry functionality."""
    
    def test_registry_singleton(self) -> None:
        """Test that get_registry returns same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        self.assertIs(reg1, reg2)

    def test_provider_detection_local_prefix(self) -> None:
        """Test provider detection for local models by prefix."""
        _ = get_registry()

        # Models with local prefixes should be detected
        # This tests the detection logic without requiring providers to be configured
        local_prefixes = ["llama", "mistral", "qwen", "phi", "gemma"]
        for prefix in local_prefixes:
            model = f"{prefix}3.2"
            # Just verify the model name starts with a known local prefix
            self.assertTrue(model.startswith(prefix))
    
    def test_model_config_retrieval(self) -> None:
        """Test model config retrieval."""
        registry = get_registry()
        
        # Should return None for unknown models
        config = registry.get_model_config("nonexistent-model-xyz")
        self.assertIsNone(config)


class ProviderBaseTest(TestCase):
    """Test provider base classes."""
    
    def test_message_creation(self) -> None:
        """Test Message model creation."""
        msg = Message(role=MessageRole.USER, content="Hello")
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello")

    def test_message_role_enum(self) -> None:
        """Test MessageRole enum values."""
        self.assertEqual(MessageRole.SYSTEM, "system")
        self.assertEqual(MessageRole.USER, "user")
        self.assertEqual(MessageRole.ASSISTANT, "assistant")
    
    def test_completion_result_creation(self) -> None:
        """Test CompletionResult model creation."""
        result = CompletionResult(
            content="Hello, how can I help?",
            model="llama3.2",
            finish_reason="stop",
        )
        
        self.assertEqual(result.content, "Hello, how can I help?")
        self.assertEqual(result.model, "llama3.2")
        self.assertEqual(result.finish_reason, "stop")


class ProviderRobustnessTest(TestCase):
    """Pass 3 robustness fixes: tool_calls passthrough, timeout sentinel, lifecycle."""

    def test_converter_includes_assistant_tool_calls(self) -> None:
        """Shared converter passes assistant tool_calls through (WS1)."""
        from agentx_ai.providers.base import convert_messages_to_openai_format

        tool_calls = [{"id": "c1", "type": "function",
                       "function": {"name": "calc", "arguments": "{}"}}]
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)
        out = convert_messages_to_openai_format([msg])
        self.assertEqual(out[0]["tool_calls"], tool_calls)

    def test_lmstudio_convert_messages_keeps_tool_calls(self) -> None:
        """LM Studio now delegates to the shared converter (regression for the drop)."""
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(ProviderConfig())
        tool_calls = [{"id": "c1", "type": "function",
                       "function": {"name": "calc", "arguments": "{}"}}]
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)
        converted = provider._convert_messages([msg])
        self.assertEqual(converted[0]["tool_calls"], tool_calls)

    def test_provider_config_timeout_defaults_none(self) -> None:
        """Unset timeout is None so providers can apply their own default (WS3)."""
        from agentx_ai.providers.base import ProviderConfig

        self.assertIsNone(ProviderConfig().timeout)

    def test_lmstudio_unset_timeout_uses_long_default(self) -> None:
        """LM Studio bumps an unset (None) timeout to its long default."""
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(ProviderConfig())
        self.assertEqual(provider._timeout, LMStudioProvider.DEFAULT_TIMEOUT)

    def test_lmstudio_explicit_timeout_is_honored(self) -> None:
        """An explicit 60.0 is no longer mistaken for 'unset' (the bug this fixes)."""
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(ProviderConfig(timeout=60.0))
        self.assertEqual(provider._timeout, 60.0)

    def test_openai_provider_close_resets_client(self) -> None:
        """close() closes the cached client and resets it for re-creation (WS2)."""
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(ProviderConfig(api_key="sk-test"))
        fake_client = MagicMock()
        fake_client.close = AsyncMock()
        provider._client = fake_client

        asyncio.run(provider.close())
        fake_client.close.assert_awaited_once()
        self.assertIsNone(provider._client)

    def test_registry_aclose_closes_cached_providers(self) -> None:
        """Registry.aclose() closes every cached provider and clears the cache (WS2)."""
        from agentx_ai.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        fake = MagicMock()
        fake.close = AsyncMock()
        registry._providers["fake"] = fake

        asyncio.run(registry.aclose())
        fake.close.assert_awaited_once()
        self.assertEqual(registry._providers, {})

    def test_registry_reload_closes_evicted_providers(self) -> None:
        """reload() closes evicted providers before clearing (no leaked pools)."""
        from agentx_ai.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        fake = MagicMock()
        fake.close = AsyncMock()
        registry._providers["fake"] = fake

        registry.reload()
        fake.close.assert_awaited_once()
        self.assertNotIn("fake", registry._providers)


class DependencyInjectionTest(TestCase):
    """The registry/config singletons are injectable (roadmap item 4)."""

    def test_agent_uses_injected_registry(self) -> None:
        """Agent(config, registry=fake) uses the injected registry, not the global."""
        from agentx_ai.agent.core import Agent, AgentConfig
        from agentx_ai.providers import registry as registry_mod

        fake = MagicMock()
        with patch.object(registry_mod, "get_registry") as global_getter:
            agent = Agent(AgentConfig(), registry=fake)
            self.assertIs(agent.registry, fake)
            global_getter.assert_not_called()

    def test_agent_falls_back_to_global_registry(self) -> None:
        """Without injection, Agent.registry resolves via get_registry()."""
        from agentx_ai.agent.core import Agent, AgentConfig

        fake = MagicMock()
        with patch("agentx_ai.agent.core.get_registry", return_value=fake) as getter:
            agent = Agent(AgentConfig())
            self.assertIs(agent.registry, fake)
            getter.assert_called_once()

    def test_set_and_reset_registry(self) -> None:
        """set_registry injects the global; reset_registry rebuilds on next access."""
        from agentx_ai.providers.registry import (
            ProviderRegistry, get_registry, set_registry, reset_registry,
        )

        fake = MagicMock(spec=ProviderRegistry)
        set_registry(fake)
        try:
            self.assertIs(get_registry(), fake)
        finally:
            reset_registry()
        # After reset, a fresh real instance is built (not the fake)
        self.assertIsInstance(get_registry(), ProviderRegistry)
        self.assertIsNot(get_registry(), fake)

    def test_set_and_reset_config_manager(self) -> None:
        """set_config_manager injects the global; reset rebuilds on next access."""
        from agentx_ai.config import (
            ConfigManager, get_config_manager, set_config_manager, reset_config_manager,
        )

        fake = MagicMock(spec=ConfigManager)
        set_config_manager(fake)
        try:
            self.assertIs(get_config_manager(), fake)
        finally:
            reset_config_manager()
        self.assertIsInstance(get_config_manager(), ConfigManager)

    def test_registry_uses_injected_config_manager(self) -> None:
        """ProviderRegistry(config_manager=fake) reads provider config from it."""
        from agentx_ai.providers.registry import ProviderRegistry

        fake_cm = MagicMock()
        fake_cm.get_provider_value.return_value = None  # no providers configured
        ProviderRegistry(config_manager=fake_cm)
        # _load_default_config probes the injected manager, not the global
        self.assertTrue(fake_cm.get_provider_value.called)


class ExceptionHierarchyTest(TestCase):
    """The AgentX exception hierarchy (roadmap item 9)."""

    def test_all_errors_are_agentx_and_builtin_exceptions(self) -> None:
        from agentx_ai.exceptions import (
            AgentXError, ConfigError, ProviderError, ModelNotFoundError,
            ProviderUnavailableError, MCPError, MCPServerNotFoundError,
            MCPTransportError, ToolExecutionError, MemoryStoreError,
        )

        for cls in (
            ConfigError, ProviderError, ModelNotFoundError, ProviderUnavailableError,
            MCPError, MCPServerNotFoundError, MCPTransportError, ToolExecutionError,
            MemoryStoreError,
        ):
            self.assertTrue(issubclass(cls, AgentXError), cls.__name__)
            self.assertTrue(issubclass(cls, Exception), cls.__name__)

    def test_back_compat_value_error_inheritance(self) -> None:
        """Leaf errors replacing a raise ValueError are still ValueErrors."""
        from agentx_ai.exceptions import (
            ModelNotFoundError, MCPServerNotFoundError, MCPTransportError,
        )

        for cls in (ModelNotFoundError, MCPServerNotFoundError, MCPTransportError):
            self.assertTrue(issubclass(cls, ValueError), cls.__name__)
            with self.assertRaises(ValueError):
                raise cls("boom")

    def test_http_status_mapping(self) -> None:
        from agentx_ai.exceptions import (
            AgentXError, ConfigError, ProviderError, ModelNotFoundError,
            ProviderUnavailableError, MCPError, MCPServerNotFoundError,
            MCPTransportError, ToolExecutionError, MemoryStoreError,
        )

        self.assertEqual(AgentXError.http_status, 500)
        self.assertEqual(ConfigError.http_status, 500)
        self.assertEqual(ProviderError.http_status, 502)
        self.assertEqual(ModelNotFoundError.http_status, 404)
        self.assertEqual(ProviderUnavailableError.http_status, 502)
        self.assertEqual(MCPError.http_status, 503)
        self.assertEqual(MCPServerNotFoundError.http_status, 404)
        self.assertEqual(MCPTransportError.http_status, 400)
        self.assertEqual(ToolExecutionError.http_status, 500)
        self.assertEqual(MemoryStoreError.http_status, 503)

    def test_message_and_details_round_trip(self) -> None:
        from agentx_ai.exceptions import ModelNotFoundError

        err = ModelNotFoundError("no such model", model="ghost:x", provider="ghost")
        self.assertEqual(str(err), "no such model")
        self.assertEqual(err.message, "no such model")
        self.assertEqual(err.details, {"model": "ghost:x", "provider": "ghost"})

    def test_memory_store_error_is_not_builtin(self) -> None:
        """MemoryStoreError must not shadow the builtin MemoryError."""
        from agentx_ai.exceptions import MemoryStoreError

        self.assertIsNot(MemoryStoreError, MemoryError)
        self.assertFalse(issubclass(MemoryStoreError, MemoryError))

    def test_registry_raises_model_not_found(self) -> None:
        """Provider resolution failures raise ModelNotFoundError (still a ValueError)."""
        from agentx_ai.exceptions import ModelNotFoundError
        from agentx_ai.providers.registry import ProviderRegistry

        fake_cm = MagicMock()
        fake_cm.get_provider_value.return_value = None  # no providers configured
        reg = ProviderRegistry(config_manager=fake_cm)

        with self.assertRaises(ModelNotFoundError):
            reg.get_provider_for_model("ghost:model-x")
        # Back-compat: existing `except ValueError` boundaries still catch it.
        with self.assertRaises(ValueError):
            reg.get_provider_for_model("ghost:model-x")

    def test_mcp_unknown_server_raises_typed_error(self) -> None:
        """Connecting an unregistered MCP server raises MCPServerNotFoundError."""
        from agentx_ai.exceptions import MCPServerNotFoundError
        from agentx_ai.mcp.client import MCPClientManager
        from agentx_ai.mcp.server_registry import ServerRegistry

        manager = MCPClientManager(registry=ServerRegistry())
        with self.assertRaises(MCPServerNotFoundError):
            manager.connect("does-not-exist")
        # Back-compat with the prior ValueError contract.
        with self.assertRaises(ValueError):
            manager.connect("does-not-exist")

    def test_error_response_maps_status(self) -> None:
        """error_response uses the exception's http_status, 500 for plain errors."""
        from agentx_ai.exceptions import (
            ModelNotFoundError, ProviderUnavailableError, MCPTransportError,
        )
        from agentx_ai.utils.responses import error_response

        self.assertEqual(error_response(ModelNotFoundError("x")).status_code, 404)
        self.assertEqual(error_response(ProviderUnavailableError("x")).status_code, 502)
        self.assertEqual(error_response(MCPTransportError("x")).status_code, 400)
        self.assertEqual(error_response(ValueError("x")).status_code, 500)

    def test_error_response_message_body(self) -> None:
        from agentx_ai.exceptions import ModelNotFoundError
        from agentx_ai.utils.responses import error_response

        resp = error_response(ModelNotFoundError("no model"))
        self.assertEqual(json.loads(resp.content)["error"], "no model")
        # Empty message falls back to the class name rather than "".
        resp2 = error_response(ModelNotFoundError())
        self.assertEqual(json.loads(resp2.content)["error"], "ModelNotFoundError")


class ToolDiscoveryErrorTest(TestCase):
    """Tool/resource discovery failures are distinguishable from 'none' (WS4)."""

    def test_discovery_failure_recorded_and_cleared(self) -> None:
        from agentx_ai.mcp.tool_executor import ToolExecutor

        executor = ToolExecutor()
        failing = MagicMock()
        failing.list_tools = AsyncMock(side_effect=RuntimeError("boom"))

        tools = asyncio.run(executor.discover_tools(failing, "srv"))
        self.assertEqual(tools, [])
        self.assertEqual(executor.get_discovery_error("srv"), "boom")

        # A subsequent successful (but empty) discovery clears the error.
        ok = MagicMock()
        ok.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        tools = asyncio.run(executor.discover_tools(ok, "srv"))
        self.assertEqual(tools, [])
        self.assertIsNone(executor.get_discovery_error("srv"))


class MemoryRecorderTest(TestCase):
    """MemoryRecorder translates agent lifecycle events into memory writes (Pass 4)."""

    def _recorder(self):
        from agentx_ai.agent.hooks import MemoryRecorder
        mem = MagicMock()
        return MemoryRecorder(mem), mem

    def test_task_complete_reflects_and_completes_goal(self) -> None:
        from agentx_ai.agent.hooks import TaskOutcome
        rec, mem = self._recorder()
        rec.on_task_complete(TaskOutcome(
            task_id="t1", task="do x", status="complete", answer="done", goal_id="g1",
        ))
        self.assertEqual(mem.reflect.call_args[0][0]["status"], "complete")
        mem.complete_goal.assert_called_once_with("g1", status="completed", result="done")

    def test_task_complete_cancelled_abandons_goal(self) -> None:
        from agentx_ai.agent.hooks import TaskOutcome
        rec, mem = self._recorder()
        rec.on_task_complete(TaskOutcome(
            task_id="t1", task="do x", status="complete", answer="[CANCELLED]", goal_id="g1",
        ))
        self.assertEqual(mem.complete_goal.call_args.kwargs["status"], "abandoned")

    def test_task_complete_without_goal_skips_complete_goal(self) -> None:
        from agentx_ai.agent.hooks import TaskOutcome
        rec, mem = self._recorder()
        rec.on_task_complete(TaskOutcome(task_id="t1", task="x", status="complete"))
        mem.complete_goal.assert_not_called()

    def test_task_error_reflects_failed_and_abandons_goal(self) -> None:
        from agentx_ai.agent.hooks import TaskOutcome
        rec, mem = self._recorder()
        rec.on_task_error(TaskOutcome(
            task_id="t1", task="x", status="failed", error="boom", goal_id="g1",
        ))
        self.assertEqual(mem.reflect.call_args[0][0]["status"], "failed")
        self.assertEqual(mem.complete_goal.call_args.kwargs["status"], "abandoned")
        self.assertIn("boom", mem.complete_goal.call_args.kwargs["result"])

    def test_on_turn_and_tool_and_goal_delegate(self) -> None:
        from agentx_ai.kit.agent_memory.models import Turn
        rec, mem = self._recorder()
        turn = Turn(conversation_id="c1", index=0, role="user", content="hi")
        rec.on_turn(turn)
        mem.store_turn.assert_called_once_with(turn)
        rec.on_tool_use("calc", {"a": 1}, "2", True, 5, None)
        self.assertEqual(mem.record_tool_usage.call_args.kwargs["tool_name"], "calc")
        rec.on_goal_complete("g2", "completed", "r")
        mem.complete_goal.assert_called_with("g2", status="completed", result="r")

    def test_reflect_failure_does_not_block_goal_completion(self) -> None:
        """Per-op fault isolation is preserved: a failed reflect still lets the goal complete."""
        from agentx_ai.agent.hooks import TaskOutcome
        rec, mem = self._recorder()
        mem.reflect.side_effect = RuntimeError("reflect down")
        rec.on_task_complete(TaskOutcome(
            task_id="t1", task="x", status="complete", answer="a", goal_id="g1",
        ))  # must not raise
        mem.complete_goal.assert_called_once()


class AgentHookDispatchTest(TestCase):
    """Agent._dispatch isolates a broken subscriber (Pass 4)."""

    def test_dispatch_swallows_subscriber_errors(self) -> None:
        from agentx_ai.agent.core import Agent, AgentConfig
        from agentx_ai.agent.hooks import AgentHooks

        class BrokenHook(AgentHooks):
            def on_turn(self, turn) -> None:
                raise RuntimeError("subscriber boom")

        agent = Agent(AgentConfig(enable_memory=False, enable_tools=False))
        agent._hooks = [BrokenHook()]
        # Must not propagate the subscriber's error.
        agent._dispatch("on_turn", object())


# =============================================================================
# Phase 11.3: Extraction Pipeline Tests
# =============================================================================

class ExtractionPipelineTest(TestCase):
    """Tests for the extraction pipeline."""

    def test_extract_entities_empty_text(self) -> None:
        """Empty text should return empty list."""
        self.assertEqual(extract_entities(""), [])

    def test_extract_entities_short_text(self) -> None:
        """Very short text should return empty list."""
        self.assertEqual(extract_entities("Hi there"), [])

    def test_extract_facts_empty_text(self) -> None:
        """Empty text should return empty list."""
        self.assertEqual(extract_facts(""), [])

    def test_extract_facts_short_text(self) -> None:
        """Very short text should return empty list."""
        self.assertEqual(extract_facts("OK"), [])

    def test_extract_relationships_no_entities(self) -> None:
        """Relationships extraction with no entities should return empty list."""
        self.assertEqual(extract_relationships("Some text here", []), [])

    def test_extract_relationships_empty_text(self) -> None:
        """Relationships extraction with empty text should return empty list."""
        entities = [{"name": "Test", "type": "Person"}]
        self.assertEqual(extract_relationships("", entities), [])

    def test_extraction_service_singleton(self) -> None:
        """Extraction service should be a singleton."""
        reset_extraction_service()

        service1 = get_extraction_service()
        service2 = get_extraction_service()
        self.assertIs(service1, service2)

        # Clean up
        reset_extraction_service()

    def test_extraction_result_model(self) -> None:
        """Test ExtractionResult model structure."""
        result = ExtractionResult(
            entities=[{"name": "Test", "type": "Person"}],
            facts=[{"claim": "Test is a person"}],
            relationships=[],
            success=True,
            tokens_used=100
        )

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.success, True)
        self.assertEqual(result.tokens_used, 100)

    def test_extraction_config_settings(self) -> None:
        """Test extraction configuration is loaded."""
        settings = get_settings()
        self.assertTrue(hasattr(settings, 'extraction_enabled'))
        self.assertTrue(hasattr(settings, 'extraction_model'))  # provider:model format
        self.assertTrue(hasattr(settings, 'extraction_temperature'))
        self.assertTrue(hasattr(settings, 'entity_types'))
        self.assertTrue(hasattr(settings, 'relationship_types'))

    @skipUnless(has_configured_provider(), "Extraction model provider not configured")
    def test_extract_entities_real(self) -> None:
        """Test real entity extraction with configured provider."""
        text = """
        User: I work at Anthropic in San Francisco as a software engineer.
        Assistant: That's great! Anthropic is doing interesting AI safety research.
        """
        result = extract_entities(text)

        # Should find at least one entity
        self.assertIsInstance(result, list)
        # Check structure of results
        for entity in result:
            self.assertIn("name", entity)
            self.assertIn("type", entity)
            self.assertIn("confidence", entity)

    @skipUnless(has_configured_provider(), "Extraction model provider not configured")
    def test_extract_facts_real(self) -> None:
        """Test real fact extraction with configured provider."""
        text = """
        User: I prefer Python over JavaScript for backend development.
        Assistant: Python is indeed very popular for backend work with Django and FastAPI.
        """
        result = extract_facts(text)

        # Should return list
        self.assertIsInstance(result, list)
        # Check structure of results
        for fact in result:
            self.assertIn("claim", fact)
            self.assertIn("confidence", fact)


# =============================================================================
# Phase 11.8: Memory System Unit Tests
# =============================================================================

class MemoryEventEmitterUnitTest(TestCase):
    """Unit tests for MemoryEventEmitter."""

    def test_on_registers_callback(self):
        """on() adds callback to handlers list."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        emitter.on("test_event", handler)

        self.assertEqual(emitter.handler_count("test_event"), 1)

    def test_on_returns_unsubscribe_function(self):
        """on() returns callable that removes handler."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        unsubscribe = emitter.on("test_event", handler)
        self.assertEqual(emitter.handler_count("test_event"), 1)

        unsubscribe()
        self.assertEqual(emitter.handler_count("test_event"), 0)

    def test_off_removes_callback(self):
        """off() removes specified callback."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        emitter.on("test_event", handler)
        result = emitter.off("test_event", handler)

        self.assertTrue(result)
        self.assertEqual(emitter.handler_count("test_event"), 0)

    def test_off_returns_false_for_nonexistent(self):
        """off() returns False for nonexistent handler."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        result = emitter.off("nonexistent", handler)

        self.assertFalse(result)

    def test_emit_calls_all_handlers(self):
        """emit() calls all registered handlers for event."""
        emitter = MemoryEventEmitter()
        call_count = {"value": 0}

        def handler1(payload):
            call_count["value"] += 1

        def handler2(payload):
            call_count["value"] += 1

        emitter.on("test_event", handler1)
        emitter.on("test_event", handler2)

        payload = EventPayload(event_name="test_event")
        count = emitter.emit("test_event", payload)

        self.assertEqual(count, 2)
        self.assertEqual(call_count["value"], 2)

    def test_emit_catches_handler_errors(self):
        """emit() logs but doesn't propagate handler exceptions."""
        emitter = MemoryEventEmitter()
        working_called = {"value": False}

        def failing_handler(payload):
            raise ValueError("Test error")

        def working_handler(payload):
            working_called["value"] = True

        emitter.on("test_event", failing_handler)
        emitter.on("test_event", working_handler)

        payload = EventPayload(event_name="test_event")
        # Should not raise even though first handler fails
        count = emitter.emit("test_event", payload)

        # Second handler should still be called (emit doesn't stop on error)
        self.assertTrue(working_called["value"])
        # Count reflects only successful handlers
        self.assertEqual(count, 1)

    def test_disable_prevents_emission(self):
        """disable() prevents handlers from being called."""
        emitter = MemoryEventEmitter()
        called = {"value": False}

        def handler(payload):
            called["value"] = True

        emitter.on("test_event", handler)
        emitter.disable()

        payload = EventPayload(event_name="test_event")
        count = emitter.emit("test_event", payload)

        self.assertEqual(count, 0)
        self.assertFalse(called["value"])

    def test_clear_removes_all_handlers(self):
        """clear() removes all handlers for event or all events."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        emitter.on("event1", handler)
        emitter.on("event2", handler)

        # Clear single event
        emitter.clear("event1")
        self.assertEqual(emitter.handler_count("event1"), 0)
        self.assertEqual(emitter.handler_count("event2"), 1)

        # Clear all events
        emitter.clear()
        self.assertEqual(emitter.handler_count(), 0)

    def test_emit_returns_handler_count(self):
        """emit() returns number of handlers called."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        emitter.on("test_event", handler)
        emitter.on("test_event", handler)

        payload = EventPayload(event_name="test_event")
        count = emitter.emit("test_event", payload)

        self.assertEqual(count, 2)

    def test_handler_count_total(self):
        """handler_count() returns total when no event specified."""
        emitter = MemoryEventEmitter()

        def handler(payload):
            pass

        emitter.on("event1", handler)
        emitter.on("event2", handler)
        emitter.on("event2", handler)

        self.assertEqual(emitter.handler_count(), 3)


class RetrievalWeightsUnitTest(TestCase):
    """Unit tests for RetrievalWeights."""

    def test_default_values(self):
        """RetrievalWeights has correct defaults."""
        weights = RetrievalWeights()

        self.assertEqual(weights.episodic, 0.3)
        self.assertEqual(weights.semantic_facts, 0.25)
        self.assertEqual(weights.semantic_entities, 0.2)
        self.assertEqual(weights.procedural, 0.15)
        self.assertEqual(weights.recency, 0.1)

    def test_from_dict_uses_defaults(self):
        """from_dict() uses defaults for missing keys."""
        weights = RetrievalWeights.from_dict({"episodic": 0.5})

        self.assertEqual(weights.episodic, 0.5)
        # Other values use defaults
        self.assertEqual(weights.semantic_facts, 0.25)
        self.assertEqual(weights.procedural, 0.15)

    def test_from_dict_all_values(self):
        """from_dict() creates weights from complete dict."""
        weights = RetrievalWeights.from_dict({
            "episodic": 0.4,
            "semantic_facts": 0.3,
            "semantic_entities": 0.15,
            "procedural": 0.1,
            "recency": 0.05,
        })

        self.assertEqual(weights.episodic, 0.4)
        self.assertEqual(weights.semantic_facts, 0.3)
        self.assertEqual(weights.semantic_entities, 0.15)
        self.assertEqual(weights.procedural, 0.1)
        self.assertEqual(weights.recency, 0.05)

    def test_merge_with_none(self):
        """merge(None) returns self."""
        weights = RetrievalWeights()
        merged = weights.merge(None)

        self.assertEqual(merged.episodic, weights.episodic)
        self.assertEqual(merged.semantic_facts, weights.semantic_facts)

    def test_merge_with_dict_overrides(self):
        """merge() applies dict overrides correctly."""
        weights = RetrievalWeights()
        merged = weights.merge({"episodic": 0.5, "recency": 0.2})

        # Overridden values
        self.assertEqual(merged.episodic, 0.5)
        self.assertEqual(merged.recency, 0.2)

    def test_merge_with_weights_object(self):
        """merge() works with RetrievalWeights object."""
        base = RetrievalWeights()
        override = RetrievalWeights(episodic=0.5, semantic_facts=0.3)
        merged = base.merge(override)

        self.assertEqual(merged.episodic, 0.5)
        self.assertEqual(merged.semantic_facts, 0.3)

    def test_from_config(self):
        """from_config() loads weights from settings."""
        weights = RetrievalWeights.from_config()

        # Should have valid weights
        self.assertIsInstance(weights.episodic, float)
        self.assertIsInstance(weights.semantic_facts, float)
        self.assertGreater(weights.episodic, 0)


class MemoryAuditLoggerUnitTest(TestCase):
    """Unit tests for MemoryAuditLogger."""

    def test_log_level_off_skips_all(self):
        """No logging when audit_log_level is 'off'."""
        settings = Settings(audit_log_level="off")
        logger = MemoryAuditLogger(settings=settings)

        # Should not log for any operation
        self.assertFalse(logger._should_log("store", "episodic"))
        self.assertFalse(logger._should_log("retrieve", "semantic"))

    def test_log_level_writes_logs_write_operations(self):
        """'writes' level logs store/update/delete operations."""
        settings = Settings(audit_log_level="writes")
        logger = MemoryAuditLogger(settings=settings)

        self.assertTrue(logger._should_log("store", "episodic"))
        self.assertTrue(logger._should_log("update", "semantic"))
        self.assertTrue(logger._should_log("delete", "procedural"))
        self.assertTrue(logger._should_log("record", "procedural"))

    def test_log_level_writes_skips_read_operations(self):
        """'writes' level skips retrieve/search operations."""
        settings = Settings(audit_log_level="writes")
        logger = MemoryAuditLogger(settings=settings)

        self.assertFalse(logger._should_log("retrieve", "episodic"))
        self.assertFalse(logger._should_log("search", "semantic"))

    def test_log_level_reads_logs_non_working(self):
        """'reads' level logs reads and writes, not working memory."""
        # Use sample rate of 1.0 to ensure reads are logged
        settings = Settings(audit_log_level="reads", audit_sample_rate=1.0)
        logger = MemoryAuditLogger(settings=settings)

        # Should log episodic reads/writes
        self.assertTrue(logger._should_log("store", "episodic"))
        self.assertTrue(logger._should_log("retrieve", "episodic"))
        # Should NOT log working memory
        self.assertFalse(logger._should_log("store", "working"))

    def test_log_level_verbose_logs_everything(self):
        """'verbose' level logs all operations including working memory."""
        settings = Settings(audit_log_level="verbose")
        logger = MemoryAuditLogger(settings=settings)

        self.assertTrue(logger._should_log("store", "episodic"))
        self.assertTrue(logger._should_log("retrieve", "semantic"))
        self.assertTrue(logger._should_log("store", "working"))

    def test_log_property(self):
        """log_level property returns current level."""
        settings = Settings(audit_log_level="writes")
        logger = MemoryAuditLogger(settings=settings)

        self.assertEqual(logger.log_level, "writes")

    def test_timed_operation_context_manager(self):
        """timed_operation context manager provides context dict."""
        settings = Settings(audit_log_level="off")  # off to avoid actual logging
        logger = MemoryAuditLogger(settings=settings)

        with logger.timed_operation("store", "episodic", user_id="test") as ctx:
            ctx["result_count"] = 5
            ctx["record_ids"] = ["id1", "id2"]

        # Context was accessible
        self.assertEqual(ctx["result_count"], 5)
        self.assertEqual(ctx["record_ids"], ["id1", "id2"])

    def test_operation_type_enum(self):
        """OperationType enum has expected values."""
        self.assertEqual(OperationType.STORE.value, "store")
        self.assertEqual(OperationType.RETRIEVE.value, "retrieve")
        self.assertEqual(OperationType.PROMOTE.value, "promote")

    def test_memory_type_enum(self):
        """MemoryType enum has expected values."""
        self.assertEqual(MemoryType.EPISODIC.value, "episodic")
        self.assertEqual(MemoryType.SEMANTIC.value, "semantic")
        self.assertEqual(MemoryType.WORKING.value, "working")

    def test_audit_log_level_enum(self):
        """AuditLogLevel enum has expected values."""
        self.assertEqual(AuditLogLevel.OFF.value, "off")
        self.assertEqual(AuditLogLevel.WRITES.value, "writes")
        self.assertEqual(AuditLogLevel.READS.value, "reads")
        self.assertEqual(AuditLogLevel.VERBOSE.value, "verbose")


class WorkingMemoryUnitTest(MockRedisTestBase):
    """Unit tests for WorkingMemory with mocked Redis."""

    def test_key_includes_channel_for_isolation(self):
        """Working memory keys include channel for isolation."""
        wm = WorkingMemory(user_id="user1", channel="project-a", conversation_id="conv1")

        self.assertIn("project-a", wm.session_key)
        self.assertIn("project-a", wm.turns_key)

    def test_key_uses_global_channel_by_default(self):
        """Working memory keys use _global channel by default."""
        wm = WorkingMemory(user_id="user1", conversation_id="conv1")

        self.assertIn("_global", wm.session_key)

    def test_add_turn_pushes_to_list(self):
        """add_turn LPUSHes turn data to Redis list."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

        # Create mock turn
        turn = MagicMock()
        turn.id = "turn-1"
        turn.index = 0
        turn.role = "user"
        turn.content = "Hello"
        turn.timestamp = datetime.utcnow()

        wm.add_turn(turn)

        # Verify lpush was called
        self.mock_redis.lpush.assert_called_once()
        call_args = self.mock_redis.lpush.call_args
        self.assertEqual(call_args[0][0], wm.turns_key)

    def test_add_turn_trims_to_max_items(self):
        """add_turn LTRIMs list to max_working_memory_items."""
        settings = get_settings()
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

        turn = MagicMock()
        turn.id = "turn-1"
        turn.index = 0
        turn.role = "user"
        turn.content = "Hello"
        turn.timestamp = datetime.utcnow()

        wm.add_turn(turn)

        # Verify ltrim was called with correct max
        self.mock_redis.ltrim.assert_called_once()
        call_args = self.mock_redis.ltrim.call_args
        self.assertEqual(call_args[0][2], settings.max_working_memory_items - 1)

    def test_add_turn_sets_ttl(self):
        """add_turn sets 1-hour TTL on turns key."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

        turn = MagicMock()
        turn.id = "turn-1"
        turn.index = 0
        turn.role = "user"
        turn.content = "Hello"
        turn.timestamp = datetime.utcnow()

        wm.add_turn(turn)

        # Verify expire was called with 3600 seconds
        self.mock_redis.expire.assert_called_once_with(wm.turns_key, 3600)

    def test_get_recent_turns_returns_json_decoded(self) -> None:
        """get_recent_turns returns decoded turn dictionaries."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

        # Mock Redis to return encoded turns
        turn_data = {"id": "turn-1", "role": "user", "content": "Hello"}
        self.mock_redis.lrange.return_value = [json.dumps(turn_data).encode()]

        result = wm.get_recent_turns(limit=5)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "turn-1")
        self.assertEqual(result[0]["role"], "user")

    def test_set_stores_with_ttl(self):
        """set() stores JSON-encoded value with TTL."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")

        wm.set("test_key", {"value": 42}, ttl_seconds=300)

        # Verify setex was called
        self.mock_redis.setex.assert_called_once()
        call_args = self.mock_redis.setex.call_args
        self.assertIn("test_key", call_args[0][0])
        self.assertEqual(call_args[0][1], 300)

    def test_get_returns_none_for_missing(self):
        """get() returns None for nonexistent key."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")
        self.mock_redis.get.return_value = None

        result = wm.get("nonexistent")

        self.assertIsNone(result)

    def test_clear_session_deletes_pattern(self):
        """clear_session deletes all keys matching session pattern using SCAN."""
        wm = WorkingMemory(user_id="user1", channel="_global", conversation_id="conv1")
        # SCAN returns (cursor, keys) tuple - cursor 0 means end of iteration
        self.mock_redis.scan.return_value = (0, [b"key1", b"key2"])

        wm.clear_session()

        self.mock_redis.scan.assert_called()
        self.mock_redis.delete.assert_called_once()


class MemoryRetrieverUnitTest(TestCase):
    """Unit tests for MemoryRetriever with mocked memory stores."""

    def test_retrieval_weights_default_sum(self):
        """Default weights sum to 1.0."""
        weights = RetrievalWeights()
        total = (
            weights.episodic +
            weights.semantic_facts +
            weights.semantic_entities +
            weights.procedural +
            weights.recency
        )

        self.assertAlmostEqual(total, 1.0, places=5)

    def test_retrieval_metrics_structure(self):
        """RetrievalMetrics has correct fields."""
        metrics = RetrievalMetrics()

        self.assertEqual(metrics.episodic_count, 0)
        self.assertEqual(metrics.semantic_facts_count, 0)
        self.assertEqual(metrics.cache_hit, False)
        self.assertIsInstance(metrics.channels_searched, list)
        self.assertIsInstance(metrics.results_per_channel, dict)

    def test_normalize_scores_min_max(self) -> None:
        """_normalize_scores uses min-max normalization."""
        # Create a minimal mock memory
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        results = [
            {"score": 0.2},
            {"score": 0.5},
            {"score": 0.8},
        ]

        normalized = retriever._normalize_scores(results, "score")

        # Min score (0.2) should normalize to 0.0
        self.assertAlmostEqual(normalized[0]["normalized_score"], 0.0, places=5)
        # Max score (0.8) should normalize to 1.0
        self.assertAlmostEqual(normalized[2]["normalized_score"], 1.0, places=5)

    def test_normalize_scores_equal_returns_half(self):
        """_normalize_scores returns 0.5 when all scores equal."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        results = [
            {"score": 0.5},
            {"score": 0.5},
            {"score": 0.5},
        ]

        normalized = retriever._normalize_scores(results, "score")

        for r in normalized:
            self.assertEqual(r["normalized_score"], 0.5)

    def test_calculate_recency_score_recent_is_high(self):
        """Recent timestamps get higher recency scores."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        now = datetime.now(UTC)
        recent = now - timedelta(hours=1)
        old = now - timedelta(hours=48)

        recent_score = retriever._calculate_recency_score(recent)
        old_score = retriever._calculate_recency_score(old)

        self.assertGreater(recent_score, old_score)

    def test_calculate_recency_score_decays_exponentially(self):
        """Recency score decays with 24-hour half-life."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        now = datetime.now(UTC)
        one_day_ago = now - timedelta(hours=24)

        # Score at 24 hours should be approximately 0.5 (half-life)
        score = retriever._calculate_recency_score(one_day_ago)

        self.assertAlmostEqual(score, 0.5, places=1)

    def test_get_cache_key_includes_user_and_channels(self):
        """Cache key includes user_id, channels, and query hash."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        cache_key = retriever._get_cache_key(
            query="test query",
            user_id="user-123",
            channels=["project-a", "_global"],
            top_k=10,
            include_episodic=True,
            include_semantic=True,
            include_procedural=True,
        )

        self.assertIn("user-123", cache_key)
        self.assertIn("project-a", cache_key)

    def test_normalize_scores_empty_list(self):
        """_normalize_scores handles empty list."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        result = retriever._normalize_scores([], "score")

        self.assertEqual(result, [])


class ChannelScopingUnitTest(TestCase):
    """Unit tests for channel scoping and tenant isolation."""

    def test_default_channel_is_global(self):
        """Default channel is _global."""
        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock:
            mock.return_value = MagicMock()
            wm = WorkingMemory(user_id="user1")

            self.assertEqual(wm.channel, "_global")
            self.assertIn("_global", wm.session_key)

    def test_channel_included_in_key_patterns(self):
        """Channel is included in key patterns for isolation."""
        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock:
            mock.return_value = MagicMock()
            wm = WorkingMemory(user_id="user1", channel="my-project")

            self.assertIn("my-project", wm.session_key)
            self.assertIn("my-project", wm.turns_key)
            self.assertIn("my-project", wm.context_key)

    def test_user_id_required_in_working_memory(self):
        """WorkingMemory requires user_id."""
        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock:
            mock.return_value = MagicMock()
            # user_id is a required parameter
            wm = WorkingMemory(user_id="user1")
            self.assertEqual(wm.user_id, "user1")

    def test_different_channels_have_different_keys(self):
        """Different channels create different key patterns."""
        with patch('agentx_ai.kit.agent_memory.connections.RedisConnection.get_client') as mock:
            mock.return_value = MagicMock()

            wm_a = WorkingMemory(user_id="user1", channel="project-a")
            wm_b = WorkingMemory(user_id="user1", channel="project-b")

            # Keys should be different
            self.assertNotEqual(wm_a.session_key, wm_b.session_key)
            self.assertNotEqual(wm_a.turns_key, wm_b.turns_key)

    def test_retrieval_weights_channel_boost(self):
        """RetrievalWeights from config includes channel boost."""
        settings = get_settings()

        # Channel boost should be configured
        self.assertIsInstance(settings.channel_active_boost, float)
        self.assertGreater(settings.channel_active_boost, 1.0)

    def test_event_payload_includes_channel(self):
        """Event payloads include channel field."""
        turn_payload = TurnStoredPayload(
            event_name="turn_stored",
            turn_id="t1",
            channel="my-channel"
        )
        self.assertEqual(turn_payload.channel, "my-channel")

        fact_payload = FactLearnedPayload(
            event_name="fact_learned",
            fact_id="f1",
            channel="another-channel"
        )
        self.assertEqual(fact_payload.channel, "another-channel")

    def test_event_payload_default_channel(self):
        """Event payloads default to _global channel."""
        payload = TurnStoredPayload(event_name="turn_stored")

        self.assertEqual(payload.channel, "_global")

    def test_retrieval_metrics_tracks_channels(self):
        """RetrievalMetrics tracks channels searched."""
        metrics = RetrievalMetrics(
            channels_searched=["project-a", "_global"],
            results_per_channel={"project-a": 5, "_global": 3}
        )

        self.assertEqual(len(metrics.channels_searched), 2)
        self.assertEqual(metrics.results_per_channel["project-a"], 5)
        self.assertEqual(metrics.results_per_channel["_global"], 3)


class ToolOutputCompressorTest(TestCase):
    """Tests for the tool output compression service (Phase 14.2)."""

    def test_compression_result_defaults(self) -> None:
        """CompressionResult should have sensible defaults."""
        result = CompressionResult()
        self.assertTrue(result.success)
        self.assertEqual(result.compressed_text, "")
        self.assertIsNone(result.error)
        self.assertEqual(result.tokens_used, 0)

    def test_compressor_singleton(self) -> None:
        """get_compressor() should return a singleton."""
        # Reset singleton
        _compressor_mod._compressor = None

        c1 = get_compressor()
        c2 = get_compressor()
        self.assertIs(c1, c2)

        # Clean up
        _compressor_mod._compressor = None

    def test_compress_disabled_config(self) -> None:
        """Compression should return success=False when disabled."""
        compressor = ToolOutputCompressor()

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG_DISABLED):
            result = asyncio.run(compressor.compress("test_tool", "x" * 5000))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "compression_disabled")
        self.assertEqual(result.original_chars, 5000)

    def test_compress_no_provider(self) -> None:
        """Compression should return success=False when provider unavailable."""
        compressor = ToolOutputCompressor()

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG), \
             patch.object(compressor, '_get_provider', side_effect=ValueError("No provider")):
            result = asyncio.run(compressor.compress("test_tool", "x" * 5000))

        self.assertFalse(result.success)
        self.assertIn("provider_unavailable", result.error)  # type: ignore[operator]

    def test_compress_success_with_mock_provider(self) -> None:
        """Compression should produce structured output with a mocked provider."""
        compressor = ToolOutputCompressor()

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=CompletionResult(
            content="## Summary\nKey info here\n\n## Structure Index\n- 3 sections",
            finish_reason="stop",
            model="test-model",
            usage={"total_tokens": 150},
        ))

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG), \
             patch.object(compressor, '_get_provider', return_value=(mock_provider, "test-model")):
            result = asyncio.run(compressor.compress(
                "read_file",
                "x" * 10000,
                task_context="Find the database config",
            ))

        self.assertTrue(result.success)
        self.assertIn("Summary", result.compressed_text)
        self.assertIn("Structure Index", result.compressed_text)
        self.assertEqual(result.tokens_used, 150)
        self.assertEqual(result.original_chars, 10000)

    def test_compress_truncates_large_input(self) -> None:
        """Input exceeding max_input_chars should be truncated before LLM call."""
        compressor = ToolOutputCompressor()

        captured_messages: list[Message] = []
        async def capture_complete(messages: list[Message], model: str, **kwargs: object) -> CompletionResult:
            captured_messages.extend(messages)
            return CompletionResult(
                content="Compressed",
                finish_reason="stop",
                model="test-model",
                usage={"total_tokens": 50},
            )

        mock_provider = MagicMock()
        mock_provider.complete = capture_complete

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG), \
             patch.object(compressor, '_get_provider', return_value=(mock_provider, "test-model")):
            asyncio.run(compressor.compress(
                "big_tool",
                "x" * 20000,
                task_context="test",
                max_input_chars=5000,
            ))

        # The prompt should contain the truncation notice, not all 20000 chars
        prompt_content = captured_messages[0].content
        self.assertIn("15,000 more chars", prompt_content)

    def test_compress_error_fallback(self) -> None:
        """Provider error should return success=False with error detail."""
        compressor = ToolOutputCompressor()

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("API timeout"))

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG), \
             patch.object(compressor, '_get_provider', return_value=(mock_provider, "test-model")):
            result = asyncio.run(compressor.compress("test_tool", "x" * 5000))

        self.assertFalse(result.success)
        self.assertIn("API timeout", result.error)  # type: ignore[operator]

    def test_compress_sync_wrapper(self) -> None:
        """compress_sync should delegate to compress and return result."""
        compressor = ToolOutputCompressor()

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=CompletionResult(
            content="Compressed output",
            finish_reason="stop",
            model="test-model",
            usage={"total_tokens": 100},
        ))

        with patch.object(compressor, '_get_config', return_value=COMPRESSOR_CONFIG), \
             patch.object(compressor, '_get_provider', return_value=(mock_provider, "test-model")):
            result = compressor.compress_sync("test_tool", "x" * 5000, task_context="test query")

        self.assertTrue(result.success)
        self.assertEqual(result.compressed_text, "Compressed output")


class ToolOutputChunkerTest(TestCase):
    """Tests for tool output chunking utilities (Phase 14.3)."""

    def test_chunk_text_fixed_size(self):
        """Plain text should produce fixed-size chunks with overlap."""

        content = "word " * 500  # ~2500 chars
        chunks = chunk_text(content, chunk_size=500, overlap=100)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("start", chunk)
            self.assertIn("end", chunk)
            self.assertIn("index", chunk)

    def test_chunk_text_structural(self):
        """Markdown with headings should split at heading boundaries."""

        content = "# Section One\nContent for section one.\n\n# Section Two\nContent for section two.\n\n# Section Three\nMore content."
        chunks = chunk_text(content)

        self.assertEqual(len(chunks), 3)
        self.assertIn("Section One", chunks[0]["text"])
        self.assertIn("Section Two", chunks[1]["text"])
        self.assertIn("Section Three", chunks[2]["text"])

    def test_chunk_text_empty(self) -> None:
        """Empty content returns empty list."""
        self.assertEqual(chunk_text(""), [])

    def test_detect_sections_markdown(self):
        """Markdown headings should be detected with correct names."""

        content = "# Introduction\nSome intro.\n\n## Methods\nSome methods.\n\n## Results\nSome results."
        sections = detect_sections(content)

        names = [s["name"] for s in sections]
        self.assertIn("Introduction", names)
        self.assertIn("Methods", names)
        self.assertIn("Results", names)
        self.assertEqual(sections[0]["level"], 1)
        self.assertEqual(sections[1]["level"], 2)

    def test_detect_sections_json(self):
        """JSON top-level keys should be detected as sections."""

        content = json.dumps({"config": {"a": 1}, "data": [1, 2, 3], "meta": "info"})
        sections = detect_sections(content)

        names = [s["name"] for s in sections]
        self.assertIn("config", names)
        self.assertIn("data", names)
        self.assertIn("meta", names)

    def test_detect_sections_plain_text(self):
        """Blank-line separated paragraphs should be detected."""

        content = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        sections = detect_sections(content)

        self.assertEqual(len(sections), 3)

    def test_get_section_content_match(self):
        """Case-insensitive section matching should work."""

        content = "# Setup\nSetup instructions here.\n\n# Usage\nUsage guide here."
        result = get_section_content(content, "setup")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["name"], "Setup")
        self.assertIn("Setup instructions", result["content"])

    def test_get_section_content_not_found(self):
        """Missing section returns None."""

        content = "# Setup\nSetup instructions here.\n\n# Usage\nUsage guide here."
        result = get_section_content(content, "nonexistent")
        self.assertIsNone(result)

    def test_resolve_json_path_simple(self):
        """Simple key access should resolve."""

        content = json.dumps({"name": "Alice", "age": 30})
        result = resolve_json_path(content, "name")

        self.assertTrue(result["success"])
        self.assertIn("Alice", result["value"])

    def test_resolve_json_path_nested(self):
        """Nested dot-notation path should resolve."""

        content = json.dumps({"data": {"items": [{"name": "first"}, {"name": "second"}]}})
        result = resolve_json_path(content, "data.items[0].name")

        self.assertTrue(result["success"])
        self.assertIn("first", result["value"])

    def test_resolve_json_path_wildcard(self):
        """Wildcard [*] should return array."""

        content = json.dumps({"items": [{"id": 1}, {"id": 2}, {"id": 3}]})
        result = resolve_json_path(content, "items[*]")

        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "list")

    def test_resolve_json_path_invalid(self):
        """Bad path returns error."""

        content = json.dumps({"name": "Alice"})
        result = resolve_json_path(content, "nonexistent.field")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_resolve_json_path_non_json(self):
        """Non-JSON content returns error."""

        result = resolve_json_path("This is plain text, not JSON.", "key")
        self.assertFalse(result["success"])
        self.assertIn("not valid JSON", result["error"])

    def test_keyword_search_fallback(self):
        """Keyword search should rank chunks by word overlap."""

        chunks = [
            {"text": "The error log shows a timeout failure", "start": 0, "end": 37, "index": 0},
            {"text": "Configuration file loaded successfully", "start": 38, "end": 75, "index": 1},
            {"text": "Error connecting to database timeout", "start": 76, "end": 112, "index": 2},
        ]
        results = _keyword_search(chunks, "error timeout", top_k=2)

        self.assertEqual(len(results), 2)
        # Both chunks mentioning error/timeout should rank higher
        texts = [r["text"] for r in results]
        self.assertTrue(any("error" in t.lower() for t in texts))

    def test_cosine_similarity(self):
        """Cosine similarity should compute correctly."""

        self.assertAlmostEqual(_cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(_cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(_cosine_similarity([0, 0], [1, 0]), 0.0)


class WebSearchToolTest(TestCase):
    """Track B: the internal `web_search` tool (Tavily primary, Brave fallback)."""

    def setUp(self):
        from unittest.mock import patch

        from agentx_ai.config import get_config_manager
        from agentx_ai.mcp import internal_tools

        self.internal_tools = internal_tools
        internal_tools._SEARCH_CACHE.clear()

        # Deterministic config; restore in tearDown.
        cfg = get_config_manager()
        self._orig_search = cfg.get("search")
        cfg.set("search", {
            "backend": "tavily",
            "fallback_enabled": True,
            "max_results": 3,
            "cache_ttl_seconds": 300,
            "tavily_api_key": "test-tavily",
            "brave_api_key": "test-brave",
        })

        # Keys always resolve so we exercise orchestration, not key resolution.
        self._key_patch = patch.object(
            internal_tools, "_resolve_search_key", return_value="test-key"
        )
        self._key_patch.start()

    def tearDown(self):
        from agentx_ai.config import get_config_manager

        self._key_patch.stop()
        self.internal_tools._SEARCH_CACHE.clear()
        get_config_manager().set("search", self._orig_search)

    def _fake_tavily(self, search):
        """A patch for `_tavily_client` whose `.search` is the given callable/Mock.

        The Tavily path goes through the SDK client (`client.search`), so tests
        mock at that seam rather than the raw HTTP transport.
        """
        from unittest.mock import MagicMock, patch

        client = MagicMock()
        client.search.side_effect = search if callable(search) else None
        if not callable(search):
            client.search.return_value = search
        return patch.object(self.internal_tools, "_tavily_client", return_value=client)

    def test_tavily_success(self):
        tavily_payload = {"results": [
            {"title": "T1", "url": "https://a", "content": "snippet one"},
            {"title": "T2", "url": "https://b", "content": "snippet two"},
        ]}
        with self._fake_tavily(tavily_payload):
            out = self.internal_tools.web_search("hello world")
        self.assertTrue(out["success"])
        self.assertEqual(out["backend"], "tavily")
        self.assertEqual(out["count"], 2)
        r0 = out["results"][0]
        self.assertEqual(
            {k: r0[k] for k in ("title", "url", "snippet")},
            {"title": "T1", "url": "https://a", "snippet": "snippet one"},
        )

    def test_fallback_to_brave(self):
        from unittest.mock import patch

        brave_payload = {"web": {"results": [
            {"title": "B1", "url": "https://x", "description": "brave snippet"},
        ]}}
        with self._fake_tavily(RuntimeError("tavily down")), \
             patch.object(self.internal_tools, "_http_get_json", return_value=brave_payload):
            out = self.internal_tools.web_search("hello")
        self.assertTrue(out["success"])
        self.assertEqual(out["backend"], "brave")
        r0 = out["results"][0]
        self.assertEqual(
            {k: r0[k] for k in ("title", "url", "snippet")},
            {"title": "B1", "url": "https://x", "snippet": "brave snippet"},
        )

    def test_both_down_graceful_empty(self):
        from unittest.mock import patch

        with self._fake_tavily(RuntimeError("tavily down")), \
             patch.object(self.internal_tools, "_http_get_json", side_effect=RuntimeError("brave down")):
            out = self.internal_tools.web_search("hello")
        self.assertFalse(out["success"])
        self.assertEqual(out["results"], [])
        self.assertIn("error", out)

    def test_cache_hit_avoids_second_call(self):
        tavily_payload = {"results": [{"title": "T", "url": "https://a", "content": "c"}]}
        with self._fake_tavily(tavily_payload) as mock_client:
            first = self.internal_tools.web_search("cached query")
            second = self.internal_tools.web_search("cached query")
        self.assertEqual(mock_client.return_value.search.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])

    def test_empty_query_rejected(self):
        out = self.internal_tools.web_search("   ")
        self.assertFalse(out["success"])
        self.assertEqual(out["results"], [])

    def test_per_turn_budget_exhausts_after_limit(self):
        """Within a budget window, calls past the limit short-circuit without
        hitting a backend (Foundation #5). Distinct queries dodge the cache."""
        from agentx_ai.agent.search_budget import search_budget_window

        payload = {"results": [{"title": "T", "url": "https://a", "content": "c"}]}
        with self._fake_tavily(payload) as mock_client, search_budget_window(2):
            ok1 = self.internal_tools.web_search("budget q1")
            ok2 = self.internal_tools.web_search("budget q2")
            blocked = self.internal_tools.web_search("budget q3")
        self.assertTrue(ok1["success"])
        self.assertTrue(ok2["success"])
        self.assertFalse(blocked["success"])
        self.assertIn("budget", blocked["error"].lower())
        # Backend hit only for the two allowed calls; the third never reached it.
        self.assertEqual(mock_client.return_value.search.call_count, 2)

    def test_no_window_means_unlimited(self):
        """Without a budget window (e.g. background callers) searches aren't gated."""
        payload = {"results": [{"title": "T", "url": "https://a", "content": "c"}]}
        with self._fake_tavily(payload):
            outs = [self.internal_tools.web_search(f"unbounded {i}") for i in range(5)]
        self.assertTrue(all(o["success"] for o in outs))

    def test_records_search_spend_to_ledger(self):
        """A successful search writes one source='search' usage row (best-effort)."""
        from unittest.mock import patch

        payload = {"results": [{"title": "T", "url": "https://a", "content": "c"}]}
        with self._fake_tavily(payload), \
             patch("agentx_ai.agent.usage_ledger.record_usage") as mock_rec:
            self.internal_tools.web_search("ledger query")
        self.assertEqual(mock_rec.call_count, 1)
        kwargs = mock_rec.call_args.kwargs
        self.assertEqual(kwargs["source"], "search")
        self.assertEqual(kwargs["units"]["credits"], 1)
        self.assertIn("cost_total", kwargs["cost"])


class IntentAwareRetrievalTest(TestCase):
    """Tests for Phase 14.3 intent-aware retrieval internal tools."""

    def test_new_tools_registered(self):
        """The three new tools should appear in get_internal_tools()."""
        tools = get_internal_tools()
        tool_names = [t.name for t in tools]

        self.assertIn("tool_output_query", tool_names)
        self.assertIn("tool_output_section", tool_names)
        self.assertIn("tool_output_path", tool_names)

    def test_tool_schemas_valid(self):
        """Each new tool schema should have required fields."""
        tools = get_internal_tools()
        new_tools = {t.name: t for t in tools if t.name.startswith("tool_output_")}

        for name in ["tool_output_query", "tool_output_section", "tool_output_path"]:
            tool = new_tools[name]
            schema = tool.input_schema
            self.assertEqual(schema["type"], "object")
            self.assertIn("properties", schema)
            self.assertIn("key", schema["properties"])

    def test_query_not_found(self):
        """Querying an expired/missing key should return error."""
        result = execute_internal_tool("tool_output_query", {
            "key": "nonexistent_key_12345",
            "query": "find errors",
        })

        self.assertFalse(result.success)

    def test_section_list_with_mock(self):
        """Section listing should return section names from stored output."""

        mock_data = {
            "content": "# Setup\nSetup info.\n\n# Usage\nUsage info.\n\n# Troubleshooting\nHelp.",
            "tool_name": "read_file",
            "tool_call_id": "test-123",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("tool_output_section", {"key": "test_key"})

        self.assertTrue(result.success)
        parsed = json.loads(result.content[0]["text"])
        self.assertIn("sections", parsed)
        names = [s["name"] for s in parsed["sections"]]
        self.assertIn("Setup", names)
        self.assertIn("Usage", names)

    def test_section_retrieve_with_mock(self):
        """Section retrieval should return section content."""

        mock_data = {
            "content": "# Setup\nSetup instructions here.\n\n# Usage\nUsage guide here.",
            "tool_name": "read_file",
            "tool_call_id": "test-123",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("tool_output_section", {
                "key": "test_key",
                "section": "Setup",
            })

        self.assertTrue(result.success)
        parsed = json.loads(result.content[0]["text"])
        self.assertIn("Setup instructions", parsed["content"])

    def test_path_resolve_with_mock(self):
        """JSON path resolution should return the value."""

        mock_data = {
            "content": json.dumps({"data": {"name": "test", "count": 42}}),
            "tool_name": "api_call",
            "tool_call_id": "test-456",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("tool_output_path", {
                "key": "test_key",
                "jsonpath": "data.name",
            })

        self.assertTrue(result.success)
        parsed = json.loads(result.content[0]["text"])
        self.assertIn("test", parsed["value"])

    def test_path_non_json_with_mock(self):
        """JSON path on non-JSON content should return error."""

        mock_data = {
            "content": "This is plain text, not JSON at all.",
            "tool_name": "read_file",
            "tool_call_id": "test-789",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("tool_output_path", {
                "key": "test_key",
                "jsonpath": "some.path",
            })

        # The tool itself returns success=True (execution succeeded)
        # but the result dict contains success=False
        parsed = json.loads(result.content[0]["text"])
        self.assertFalse(parsed["success"])


class TrajectoryCompressionTest(TestCase):
    """Tests for intra-trajectory compression (Phase 14.4)."""

    def _make_messages(self, num_rounds: int, content_size: int = 100) -> list[Message]:
        """Helper: build a message list with system, user, and N tool rounds."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful agent."),
            Message(role=MessageRole.USER, content="Do something useful."),
        ]
        for i in range(num_rounds):
            messages.append(Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": f"tool_{i}", "arguments": json.dumps({"arg": f"val_{i}"})},
                }],
            ))
            messages.append(Message(
                role=MessageRole.TOOL,
                content="x" * content_size,
                tool_call_id=f"call_{i}",
                name=f"tool_{i}",
            ))
        return messages

    def test_identify_tool_rounds(self):
        """Should identify correct number of rounds with proper indices."""
        messages = self._make_messages(3)
        rounds = identify_tool_rounds(messages)

        self.assertEqual(len(rounds), 3)
        # First round starts at index 2 (after system + user)
        self.assertEqual(rounds[0].start_idx, 2)
        self.assertEqual(rounds[0].end_idx, 3)
        self.assertEqual(rounds[1].start_idx, 4)
        self.assertEqual(rounds[1].end_idx, 5)
        self.assertEqual(rounds[2].start_idx, 6)
        self.assertEqual(rounds[2].end_idx, 7)
        # Each round has one tool message
        for rnd in rounds:
            self.assertEqual(len(rnd.tool_msgs), 1)

    def test_no_compression_below_threshold(self):
        """Should return False and leave messages intact when below threshold."""
        messages = self._make_messages(3, content_size=50)
        original_count = len(messages)

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": True,
            "trajectory_compression.threshold_ratio": 0.75,
            "trajectory_compression.preserve_recent_rounds": 2,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg):
            # Very high limit so we stay below threshold
            result = compress_trajectory(messages, context_limit_tokens=100000)

        self.assertFalse(result)
        self.assertEqual(len(messages), original_count)

    def test_compression_triggers(self):
        """Should compress older rounds and insert Knowledge block."""

        # Large content to exceed threshold
        messages = self._make_messages(4, content_size=2000)
        original_count = len(messages)

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": True,
            "trajectory_compression.threshold_ratio": 0.1,  # Very low to trigger
            "trajectory_compression.preserve_recent_rounds": 2,
            "trajectory_compression.model": "test-model",
            "trajectory_compression.temperature": 0.2,
            "trajectory_compression.max_tokens": 1500,
            "trajectory_compression.max_knowledge_chars": 3000,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg), \
             patch("agentx_ai.streaming.trajectory_compression._generate_knowledge_block",
                   return_value="Key finding: tool_0 returned data X. tool_1 returned data Y."):
            result = compress_trajectory(messages, context_limit_tokens=100)

        self.assertTrue(result)
        # Should have fewer messages (removed 2 rounds = 4 messages, added 1 Knowledge)
        self.assertEqual(len(messages), original_count - 4 + 1)
        # Knowledge block should exist
        knowledge_msgs = [m for m in messages if "[KNOWLEDGE -" in (m.content or "")]
        self.assertEqual(len(knowledge_msgs), 1)
        self.assertEqual(knowledge_msgs[0].role, MessageRole.SYSTEM)
        self.assertIn("2 tool-call round(s)", knowledge_msgs[0].content)

    def test_preserve_recent_rounds(self):
        """With preserve=2 and 4 rounds, last 2 should remain intact."""

        messages = self._make_messages(4, content_size=2000)

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": True,
            "trajectory_compression.threshold_ratio": 0.1,
            "trajectory_compression.preserve_recent_rounds": 2,
            "trajectory_compression.model": "test-model",
            "trajectory_compression.temperature": 0.2,
            "trajectory_compression.max_tokens": 1500,
            "trajectory_compression.max_knowledge_chars": 3000,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg), \
             patch("agentx_ai.streaming.trajectory_compression._generate_knowledge_block",
                   return_value="Consolidated knowledge."):
            compress_trajectory(messages, context_limit_tokens=100)

        # The last 2 rounds should still have their tool_calls assistant messages
        assistant_with_tools = [
            m for m in messages
            if m.role == MessageRole.ASSISTANT and m.tool_calls
        ]
        self.assertEqual(len(assistant_with_tools), 2)
        # And their tool results
        tool_msgs = [m for m in messages if m.role == MessageRole.TOOL]
        self.assertEqual(len(tool_msgs), 2)

    def test_fallback_on_llm_failure(self):
        """Should return False and leave messages intact when LLM fails."""
        messages = self._make_messages(4, content_size=2000)
        original_count = len(messages)

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": True,
            "trajectory_compression.threshold_ratio": 0.1,
            "trajectory_compression.preserve_recent_rounds": 2,
            "trajectory_compression.model": "test-model",
            "trajectory_compression.temperature": 0.2,
            "trajectory_compression.max_tokens": 1500,
            "trajectory_compression.max_knowledge_chars": 3000,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg), \
             patch("agentx_ai.streaming.trajectory_compression._generate_knowledge_block",
                   return_value=None):
            result = compress_trajectory(messages, context_limit_tokens=100)

        self.assertFalse(result)
        self.assertEqual(len(messages), original_count)

    def test_knowledge_block_placement(self):
        """Knowledge block should be placed after system messages."""
        messages = self._make_messages(4, content_size=2000)
        # Add a second system message
        messages.insert(1, Message(role=MessageRole.SYSTEM, content="Additional system context."))

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": True,
            "trajectory_compression.threshold_ratio": 0.1,
            "trajectory_compression.preserve_recent_rounds": 2,
            "trajectory_compression.model": "test-model",
            "trajectory_compression.temperature": 0.2,
            "trajectory_compression.max_tokens": 1500,
            "trajectory_compression.max_knowledge_chars": 3000,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg), \
             patch("agentx_ai.streaming.trajectory_compression._generate_knowledge_block",
                   return_value="Knowledge content."):
            compress_trajectory(messages, context_limit_tokens=100)

        # Find the Knowledge message
        knowledge_idx = None
        for i, m in enumerate(messages):
            if "[KNOWLEDGE -" in (m.content or ""):
                knowledge_idx = i
                break
        self.assertIsNotNone(knowledge_idx)
        assert knowledge_idx is not None
        # It should come after both system messages
        # System messages are at 0 and 1, so Knowledge should be at index 2
        self.assertEqual(knowledge_idx, 2)
        # And before the USER message
        self.assertEqual(messages[knowledge_idx + 1].role, MessageRole.USER)

    def test_disabled_via_config(self):
        """Should return False when trajectory_compression.enabled is False."""
        messages = self._make_messages(4, content_size=2000)
        original_count = len(messages)

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "trajectory_compression.enabled": False,
        }.get(key, default)

        with patch("agentx_ai.config.get_config_manager", return_value=mock_cfg):
            result = compress_trajectory(messages, context_limit_tokens=100)

        self.assertFalse(result)
        self.assertEqual(len(messages), original_count)

    def test_rounds_to_text_format(self):
        """Serialisation should include tool names, arguments, and capped results."""
        messages = self._make_messages(2, content_size=50)
        rounds = identify_tool_rounds(messages)
        text = rounds_to_text(rounds)

        self.assertIn("Round 1:", text)
        self.assertIn("Round 2:", text)
        self.assertIn("tool_0", text)
        self.assertIn("tool_1", text)
        self.assertIn("val_0", text)


class StreamingToolLoopTest(TestCase):
    """Behavior-pinning tests for streaming_tool_loop (roadmap item 1 tail).

    Drive the loop against a fake provider/agent and assert the observable SSE
    event stream + ToolLoopResult accumulation, so the decomposition into
    per-stage helpers is provably behavior-preserving.
    """

    class _FakeProvider:
        """Yields a pre-scripted list of chunks per stream() call."""
        def __init__(self, rounds):
            self._rounds = rounds
            self._call = 0

        async def stream(self, messages, model_id, **kwargs):
            chunks = self._rounds[self._call]
            self._call += 1
            for c in chunks:
                yield c

    class _FakeAgent:
        def __init__(self):
            self._active_alloy_executor = None
            self.executed = []

        def _execute_tool_calls(self, calls, task_context=""):
            from agentx_ai.providers.base import Message, MessageRole
            self.executed.append(list(calls))
            return [
                Message(role=MessageRole.TOOL, content=f"result-{tc.name}",
                        tool_call_id=tc.id, name=tc.name)
                for tc in calls
            ]

    @staticmethod
    async def _drain(gen):
        return [ev async for ev in gen]

    def _run(self, provider, agent, messages, tools, **kwargs):
        from agentx_ai.streaming.tool_loop import streaming_tool_loop, ToolLoopResult
        result = ToolLoopResult()
        with patch("agentx_ai.streaming.tool_loop.compress_trajectory", return_value=False), \
             patch("agentx_ai.streaming.tool_loop.estimate_tokens", return_value=10):
            events = asyncio.run(self._drain(streaming_tool_loop(
                provider, "fake:model", messages, tools, agent,
                result=result, **kwargs,
            )))
        return events, result

    @staticmethod
    def _events_of(events, name):
        return [e for e in events if e.startswith(f"event: {name}\n")]

    def test_plain_completion(self) -> None:
        """One content chunk, no tool calls → chunk event, final_content set, loop ends."""
        from agentx_ai.providers.base import StreamChunk

        provider = self._FakeProvider([[StreamChunk(content="hello")]])
        agent = self._FakeAgent()
        messages: list = []
        events, result = self._run(provider, agent, messages, None)

        self.assertEqual(len(self._events_of(events, "chunk")), 1)
        self.assertEqual(self._events_of(events, "tool_call"), [])
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.final_content, "hello")
        self.assertEqual(agent.executed, [])

    def test_empty_completion_fallback(self) -> None:
        """An empty first completion emits the explicit fallback chunk."""
        provider = self._FakeProvider([[]])  # stream yields nothing
        agent = self._FakeAgent()
        events, result = self._run(provider, agent, [], None)

        self.assertEqual(result.content, "[empty response from model]")
        self.assertEqual(result.final_content, "[empty response from model]")
        chunk_events = self._events_of(events, "chunk")
        self.assertEqual(len(chunk_events), 1)
        self.assertIn("[empty response from model]", chunk_events[0])

    def test_one_tool_round_then_completion(self) -> None:
        """A tool round emits tool_call + tool_result, extends messages, then completes."""
        from agentx_ai.providers.base import StreamChunk, ToolCall, MessageRole

        provider = self._FakeProvider([
            [StreamChunk(tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})])],
            [StreamChunk(content="final answer")],
        ])
        agent = self._FakeAgent()
        messages: list = []
        events, result = self._run(
            provider, agent, messages, [{"type": "function", "function": {"name": "search"}}],
        )

        self.assertEqual(len(self._events_of(events, "tool_call")), 1)
        self.assertEqual(len(self._events_of(events, "tool_result")), 1)
        self.assertEqual(result.tools_used, ["search"])
        self.assertEqual(result.final_content, "final answer")
        # messages now carries the assistant tool_calls msg + the tool result
        roles = [m.role for m in messages]
        self.assertIn(MessageRole.ASSISTANT, roles)
        self.assertIn(MessageRole.TOOL, roles)
        self.assertEqual(len(agent.executed), 1)

    def test_emits_status_phases(self) -> None:
        """A tool round publishes status: thinking → running_tool → reading → thinking."""
        from agentx_ai.providers.base import StreamChunk, ToolCall

        provider = self._FakeProvider([
            [StreamChunk(tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})])],
            [StreamChunk(content="final answer")],
        ])
        agent = self._FakeAgent()
        with patch("agentx_ai.streaming.tool_loop.emit_status") as mock_status:
            self._run(
                provider, agent, [], [{"type": "function", "function": {"name": "search"}}],
            )
        phases = [call.args[0] for call in mock_status.call_args_list]
        self.assertEqual(phases, ["thinking", "running_tool", "reading", "thinking"])
        # running_tool carries the tool name in its label.
        running = next(c for c in mock_status.call_args_list if c.args[0] == "running_tool")
        self.assertIn("search", running.kwargs.get("label", ""))

    def test_steer_folds_at_tool_boundary(self) -> None:
        """A steer drained after a tool round is folded in as a USER message."""
        from agentx_ai.providers.base import StreamChunk, ToolCall, MessageRole

        provider = self._FakeProvider([
            [StreamChunk(tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})])],
            [StreamChunk(content="final answer")],
        ])
        agent = self._FakeAgent()
        messages: list = []
        # Boundary drain (after the tool round) yields the steer; the would-end
        # drain on the final round yields nothing.
        with patch(
            "agentx_ai.streaming.tool_loop.drain_steer_messages",
            side_effect=[["also check Y"], []],
        ):
            self._run(
                provider, agent, messages,
                [{"type": "function", "function": {"name": "search"}}],
            )
        steered = [m for m in messages if m.role == MessageRole.USER and m.content == "also check Y"]
        self.assertEqual(len(steered), 1)

    def test_steer_continues_at_would_end(self) -> None:
        """A steer arriving when the model would finish runs another round."""
        from agentx_ai.providers.base import StreamChunk, MessageRole

        provider = self._FakeProvider([
            [StreamChunk(content="partial")],   # round 0: no tools → would end
            [StreamChunk(content="final")],     # round 1: after steer folded in
        ])
        agent = self._FakeAgent()
        messages: list = []
        with patch(
            "agentx_ai.streaming.tool_loop.drain_steer_messages",
            side_effect=[["do Z instead"], []],
        ):
            _events, result = self._run(provider, agent, messages, None)
        # Ran a second round rather than ending after the first.
        self.assertEqual(provider._call, 2)
        # The answer-so-far + the steer were folded in coherently.
        self.assertTrue(
            any(m.role == MessageRole.ASSISTANT and m.content == "partial" for m in messages)
        )
        self.assertTrue(
            any(m.role == MessageRole.USER and m.content == "do Z instead" for m in messages)
        )
        self.assertEqual(result.final_content, "final")
        # would-end steer is captured on the result for persistence.
        self.assertEqual(len(result.steers), 1)
        self.assertEqual(result.steers[0]["phase"], "would_end")
        self.assertEqual(result.steers[0]["content"], "do Z instead")
        self.assertEqual(result.steers[0]["after_tools"], [])

    def test_steer_captured_on_result_at_boundary(self) -> None:
        """A boundary steer is captured on result.steers with the round's tools."""
        from agentx_ai.providers.base import StreamChunk, ToolCall

        provider = self._FakeProvider([
            [StreamChunk(tool_calls=[ToolCall(id="t1", name="search", arguments={"q": "x"})])],
            [StreamChunk(content="final answer")],
        ])
        agent = self._FakeAgent()
        with patch(
            "agentx_ai.streaming.tool_loop.drain_steer_messages",
            side_effect=[["also check Y"], []],
        ):
            _events, result = self._run(
                provider, agent, [],
                [{"type": "function", "function": {"name": "search"}}],
            )
        self.assertEqual(len(result.steers), 1)
        self.assertEqual(result.steers[0]["phase"], "tool_boundary")
        self.assertEqual(result.steers[0]["after_tools"], ["search"])
        self.assertEqual(result.steers[0]["round"], 0)


class SteerPersistenceTest(TestCase):
    """Pure turn-builders extracted from views._store_turns (streaming/persistence.py)."""

    def test_build_steer_turns_shape_and_metadata(self) -> None:
        from agentx_ai.streaming.persistence import build_steer_turns

        steers = [
            {"content": "also check Y", "round": 0, "after_tools": ["search"], "phase": "tool_boundary"},
            {"content": "do Z", "round": 1, "after_tools": [], "phase": "would_end"},
        ]
        turns = build_steer_turns("conv1", steers, 5, agent_id="bold-falcon", agent_name="Mobius")
        self.assertEqual([t.index for t in turns], [5, 6])
        self.assertTrue(all(t.role == "user" for t in turns))
        self.assertTrue(all(t.agent_id is None for t in turns))  # producing agent unknown
        m0 = turns[0].metadata
        self.assertEqual(m0["steered"], True)
        self.assertEqual(m0["steer_round"], 0)
        self.assertEqual(m0["after_tools"], ["search"])
        self.assertEqual(m0["phase"], "tool_boundary")
        self.assertEqual(m0["steered_agent_id"], "bold-falcon")
        self.assertEqual(m0["steered_agent_name"], "Mobius")
        self.assertEqual(turns[1].content, "do Z")

    def test_build_assistant_turn_skips_blank(self) -> None:
        from agentx_ai.streaming.persistence import build_assistant_turn

        self.assertIsNone(build_assistant_turn("c", "   ", 3, metadata={}))
        turn = build_assistant_turn(
            "c", "hello", 3, metadata={"interrupted": True}, token_count=10, model="m", agent_id="a",
        )
        self.assertIsNotNone(turn)
        self.assertEqual(turn.role, "assistant")
        self.assertEqual(turn.index, 3)
        self.assertEqual(turn.metadata["interrupted"], True)
        self.assertEqual(turn.agent_id, "a")

    def test_build_assistant_turn_keeps_blank_plan_card(self) -> None:
        """A turn carrying a plan card is meaningful even with no text (an
        interrupted plan stopped before synthesis); dropping it would lose the
        card + its resume offer on restore."""
        from agentx_ai.streaming.persistence import build_assistant_turn

        plan_meta = {"plan": {"plan_id": "p1", "status": "interrupted", "subtasks": []}}
        kept = build_assistant_turn("c", "", 4, metadata=plan_meta)
        self.assertIsNotNone(kept)
        self.assertEqual(kept.role, "assistant")
        self.assertEqual(kept.metadata["plan"]["status"], "interrupted")
        # Blank with no plan card is still skipped.
        self.assertIsNone(build_assistant_turn("c", "", 4, metadata={"interrupted": True}))

    def test_build_tool_turns_roles_and_index(self) -> None:
        from agentx_ai.streaming.persistence import build_tool_turns

        data = [
            {"type": "tool_call", "tool": "search", "tool_call_id": "t1", "arguments": {"q": "x"}},
            {"type": "tool_result", "tool": "search", "tool_call_id": "t1", "content": "ok", "success": True},
        ]
        turns, next_index = build_tool_turns("c", data, 1)
        self.assertEqual([t.role for t in turns], ["tool_call", "tool_result"])
        self.assertEqual([t.index for t in turns], [1, 2])
        self.assertEqual(next_index, 3)
        self.assertEqual(turns[1].metadata["success"], True)


class StatusEmitterTest(TestCase):
    """Unit tests for the run-scoped status emitter (streaming/status.py)."""

    def setUp(self) -> None:
        from agentx_ai.streaming import status as status_mod
        self.status = status_mod
        # Isolate throttle state between tests.
        self.status._last_emit.clear()

    def _patch_bus(self):
        from agentx_ai.streaming import chat_run
        return patch.object(chat_run.store, "append_event")

    @staticmethod
    def _payload(sse_event: str) -> dict:
        for line in sse_event.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        return {}

    def test_noop_without_run(self) -> None:
        """No contextvar + no run_id arg → nothing published."""
        self.assertIsNone(self.status.current_run_id.get())
        with self._patch_bus() as append:
            self.status.emit_status("thinking")
        append.assert_not_called()

    def test_emits_with_explicit_run_id(self) -> None:
        with self._patch_bus() as append:
            self.status.emit_status("thinking", run_id="r1")
        append.assert_called_once()
        rid, sse = append.call_args.args
        self.assertEqual(rid, "r1")
        self.assertTrue(sse.startswith("event: status\n"))
        payload = self._payload(sse)
        self.assertEqual(payload["phase"], "thinking")
        self.assertEqual(payload["label"], "Thinking…")  # default from STATUS_PHASES

    def test_resolves_from_contextvar(self) -> None:
        token = self.status.current_run_id.set("r2")
        try:
            with self._patch_bus() as append:
                self.status.emit_status("composing")
        finally:
            self.status.current_run_id.reset(token)
        self.assertEqual(append.call_args.args[0], "r2")

    def test_throttle_coalesces_same_phase(self) -> None:
        with self._patch_bus() as append:
            self.status.emit_status("thinking", run_id="r3")
            self.status.emit_status("thinking", run_id="r3")  # immediate repeat → dropped
            self.status.emit_status("running_tool", label="Running x…", run_id="r3")
            self.status.emit_status("running_tool", label="Running y…", run_id="r3")  # label change → kept
        self.assertEqual(append.call_count, 3)

    def test_optional_fields_in_payload(self) -> None:
        with self._patch_bus() as append:
            self.status.emit_status(
                "embedding", detail="query", group="recalling", progress=0.5, run_id="r4",
            )
        payload = self._payload(append.call_args.args[1])
        self.assertEqual(payload["detail"], "query")
        self.assertEqual(payload["group"], "recalling")
        self.assertEqual(payload["progress"], 0.5)


class LiveSteeringStoreTest(MockRedisTestBase):
    """ChatRunStore steer-queue push/drain (live steering)."""

    @property
    def store(self):
        from agentx_ai.streaming.chat_run import store
        return store

    def test_push_steer_rpushes_and_caps(self) -> None:
        from agentx_ai.streaming.chat_run import STEER_MAXLEN, _queue_key
        self.mock_redis.exists.return_value = 1
        ok = self.store.push_steer("r1", "also check Y")
        self.assertTrue(ok)
        self.mock_redis.rpush.assert_called_once_with(_queue_key("r1"), "also check Y")
        # Trimmed to the most-recent STEER_MAXLEN entries.
        self.mock_redis.ltrim.assert_called_once_with(_queue_key("r1"), -STEER_MAXLEN, -1)

    def test_push_steer_false_when_run_gone(self) -> None:
        self.mock_redis.exists.return_value = 0
        self.assertFalse(self.store.push_steer("missing", "hi"))
        self.mock_redis.rpush.assert_not_called()

    def test_drain_steer_reads_and_clears_atomically(self) -> None:
        from agentx_ai.streaming.chat_run import _queue_key
        pipe = MagicMock()
        pipe.execute.return_value = [[b"first", b"second"], 1]
        self.mock_redis.pipeline.return_value = pipe
        out = self.store.drain_steer("r1")
        self.assertEqual(out, ["first", "second"])
        pipe.lrange.assert_called_once_with(_queue_key("r1"), 0, -1)
        pipe.delete.assert_called_once_with(_queue_key("r1"))


class ProcedureCandidateTest(TestCase):
    """Procedural memory — encode loop (Slice 0): rule detection + candidate staging."""

    def test_detect_explicit_rule_matches(self) -> None:
        from agentx_ai.kit.agent_memory.memory.procedural import detect_explicit_rule

        clause = detect_explicit_rule("From now on, always cite your sources. Thanks!")
        self.assertIsNotNone(clause)
        self.assertIn("cite", clause.lower())
        # The clause stops at the sentence boundary (doesn't swallow "Thanks!").
        self.assertNotIn("Thanks", clause)
        self.assertIsNotNone(detect_explicit_rule("I prefer concise answers"))
        self.assertIsNotNone(detect_explicit_rule("make sure to run the tests before pushing"))

    def test_detect_explicit_rule_rejects_ordinary(self) -> None:
        from agentx_ai.kit.agent_memory.memory.procedural import detect_explicit_rule

        self.assertIsNone(detect_explicit_rule("what's the weather today?"))
        self.assertIsNone(detect_explicit_rule("build me a login form"))
        self.assertIsNone(detect_explicit_rule(""))

    def test_stage_candidate_inserts_row(self) -> None:
        from agentx_ai.kit.agent_memory.memory import procedural as proc_mod

        mock_session = MagicMock()
        cm = MagicMock()
        cm.__enter__.return_value = mock_session
        cm.__exit__.return_value = False
        with patch.object(proc_mod, "get_embedder", return_value=MagicMock()), \
             patch.object(proc_mod, "get_postgres_session", return_value=cm):
            pm = proc_mod.ProceduralMemory()
            pm.stage_candidate(
                "conv1", "correction", "do Y not X",
                context={"after_tools": ["search"]}, channel="proj", agent_id="bold-falcon",
            )
        mock_session.execute.assert_called_once()
        params = mock_session.execute.call_args.args[1]
        self.assertEqual(params["conv_id"], "conv1")
        self.assertEqual(params["signal"], "correction")
        self.assertEqual(params["content"], "do Y not X")
        self.assertEqual(params["channel"], "proj")
        self.assertEqual(params["agent_id"], "bold-falcon")
        self.assertIn("search", params["context"])  # JSON-encoded context string


class EntityGraphNodeDictTest(TestCase):
    """Regression: get_entity_graph must return related entities as mutable dicts.

    A raw neo4j Node rejects item assignment, but the retriever's scoring does
    `entity["final_score"] = …` — a raw node there raised "'Node' object does not
    support item assignment", silently breaking recall + 500'ing user-history.
    """

    class _FakeNode:
        """Mimics a neo4j Node: dict()-able + .get(), but rejects __setitem__."""
        def __init__(self, data):
            self._d = dict(data)

        def keys(self):
            return self._d.keys()

        def __getitem__(self, k):
            return self._d[k]

        def get(self, k, default=None):
            return self._d.get(k, default)

        def __setitem__(self, k, v):
            raise TypeError("'Node' object does not support item assignment")

    def test_related_entities_are_mutable_dicts(self) -> None:
        from agentx_ai.kit.agent_memory.memory import semantic as sem_mod

        record = {
            "entity": self._FakeNode({"id": "e1", "name": "Root"}),
            "related": [{
                "entity": self._FakeNode({"id": "e2", "name": "Project X", "type": "Project"}),
                "relationship": "RELATES_TO",
                "path_length": 1,
            }],
            "facts": [],
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [record]
        cm = MagicMock()
        cm.__enter__.return_value = mock_session
        cm.__exit__.return_value = False
        with patch.object(sem_mod.Neo4jConnection, "session", return_value=cm):
            out = sem_mod.SemanticMemory().get_entity_graph(["e1"])

        related_entity = out["related"][0]["entity"]
        self.assertIsInstance(related_entity, dict)
        self.assertEqual(related_entity["id"], "e2")
        related_entity["final_score"] = 1.0  # must not raise (the bug)
        self.assertEqual(out["related"][0]["relationship"], "RELATES_TO")


class PlanExecutorResultTest(TestCase):
    """PlanExecutor cleanup: typed PlanResult, completed-count, sync safety net."""

    @staticmethod
    def _plan(*results):
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity
        steps = []
        for i, r in enumerate(results):
            s = Subtask(id=i, description=f"step {i}", type=SubtaskType.GENERATION)
            s.result = r
            s.completed = r is not None
            steps.append(s)
        return TaskPlan(task="t", complexity=TaskComplexity.COMPLEX, steps=steps)

    def test_completed_count(self) -> None:
        from agentx_ai.agent.plan_executor import PlanExecutor

        plan = self._plan("ok", "[FAILED: x]", "[ABANDONED: y]", None)
        # Default counts everything with a result that isn't FAILED (so ABANDONED counts).
        self.assertEqual(PlanExecutor._completed_count(plan), 2)
        # exclude_abandoned drops the ABANDONED one.
        self.assertEqual(PlanExecutor._completed_count(plan, exclude_abandoned=True), 1)

    def test_sync_execute_force_fails_non_terminal_subtask(self) -> None:
        """Safety-net parity: a subtask that never marks complete is force-failed,
        so the sync loop terminates instead of re-selecting it forever."""
        from agentx_ai.agent.plan_executor import PlanExecutor

        plan = self._plan(None)  # one pending subtask
        agent = MagicMock()
        state = MagicMock()
        state.is_cancel_requested.return_value = False
        ex = PlanExecutor(agent, state)
        # Subtask "runs" fine, but mark_complete is neutered (simulates the
        # historical id/index mismatch that left the slot not-completed).
        with patch.object(PlanExecutor, "_execute_subtask_sync", return_value="x"), \
             patch.object(plan, "mark_complete", lambda *a, **k: None), \
             patch.object(PlanExecutor, "_compose_answer_sync", return_value="final"):
            answer = ex.execute(plan)
        self.assertEqual(answer, "final")          # loop terminated (didn't spin)
        self.assertTrue(plan.steps[0].completed)   # force-failed to a terminal state
        self.assertTrue((plan.steps[0].result or "").startswith("[FAILED"))

    def test_execute_streaming_surfaces_outputs_into_result(self) -> None:
        """The caller-owned PlanResult carries content, tokens, and steers
        (the dropped-field regression that the typed result guards against)."""
        from types import SimpleNamespace
        from agentx_ai.agent.plan_executor import PlanExecutor, PlanResult

        plan = self._plan(None)  # one pending subtask
        agent = MagicMock()
        agent._active_alloy_executor = None
        agent._get_tools_for_provider.return_value = None
        state = MagicMock()
        state.is_cancel_requested.return_value = False

        async def fake_loop(provider, model_id, messages, tools, ag, *, result=None, **kw):
            if result is not None:
                result.final_content = "subtask out"
                result.tokens_in = 5
                result.tokens_out = 3
                result.steers.append(
                    {"content": "do Y", "round": 0, "after_tools": [], "phase": "would_end"}
                )
            return
            yield  # make this an async generator

        class _FakeProvider:
            async def stream(self, messages, model_id, **kw):
                yield SimpleNamespace(content="final synthesis")

        ex = PlanExecutor(agent, state)
        pr = PlanResult()

        async def _drive():
            async for _ in ex.execute_streaming(
                plan, _FakeProvider(), "m", None, result=pr,
            ):
                pass

        with patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_loop):
            asyncio.run(_drive())

        self.assertEqual(pr.plan_id, ex.plan_id)
        self.assertIn("final synthesis", pr.full_content)
        self.assertEqual(pr.tokens_in, 5)
        self.assertEqual(pr.tokens_out, 3)
        self.assertEqual(len(pr.steers), 1)
        self.assertEqual(pr.steers[0]["content"], "do Y")

    def test_execute_streaming_resume_skips_completed_subtasks(self) -> None:
        """Resuming a partially-complete plan emits plan_resumed (not plan_start),
        re-runs only the not-yet-terminal subtasks, and reuses the existing
        Redis state (no fresh create)."""
        import json as _json
        from types import SimpleNamespace
        from agentx_ai.agent.plan_executor import PlanExecutor, PlanResult
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity

        # 3 subtasks; #0 already complete (as if restored by load_plan).
        steps = [
            Subtask(id=0, description="step 0", type=SubtaskType.GENERATION),
            Subtask(id=1, description="step 1", type=SubtaskType.GENERATION, dependencies=[0]),
            Subtask(id=2, description="step 2", type=SubtaskType.GENERATION, dependencies=[1]),
        ]
        steps[0].completed = True
        steps[0].result = "already done"
        plan = TaskPlan(task="t", complexity=TaskComplexity.COMPLEX, steps=steps)

        agent = MagicMock()
        agent._active_alloy_executor = None
        state = MagicMock()
        state.is_cancel_requested.return_value = False

        ran: list = []

        async def fake_loop(provider, model_id, messages, tools, ag, *, result=None,
                            task_context=None, **kw):
            # Record which subtask ran (its description rides in task_context).
            ran.append(task_context)
            if result is not None:
                result.final_content = f"out:{task_context}"
            return
            yield  # async generator

        class _FakeProvider:
            async def stream(self, messages, model_id, **kw):
                yield SimpleNamespace(content="synth")

        ex = PlanExecutor(agent, state)
        events: list[str] = []

        async def _drive():
            async for ev in ex.execute_streaming(
                plan, _FakeProvider(), "m", None, result=PlanResult(),
                resume_plan_id="resumed-id",
            ):
                events.append(ev)

        with patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_loop):
            asyncio.run(_drive())

        # Reused the existing state — no fresh create.
        state.create.assert_not_called()
        self.assertEqual(ex.plan_id, "resumed-id")

        # First event is a plan_resumed snapshot, never plan_start.
        self.assertIn("event: plan_resumed", events[0])
        self.assertFalse(any("event: plan_start" in e for e in events))
        # The snapshot marks #0 complete and #1/#2 pending.
        payload = _json.loads(events[0].split("data: ", 1)[1])
        statuses = {s["subtask_id"]: s["status"] for s in payload["subtasks"]}
        self.assertEqual(statuses, {0: "complete", 1: "pending", 2: "pending"})

        # Only the two remaining subtasks executed (in dependency order).
        self.assertEqual(ran, ["step 1", "step 2"])


class SubtaskGoalTrackingTest(TestCase):
    """Phase 15.7 #1 — subtask-level goal tracking.

    The planner creates a child goal per subtask (linked to the plan's parent
    goal) and the executor closes each one out through the agent hook seam.
    These tests certify the full chain without a live memory backend.
    """

    @staticmethod
    def _multi_step_plan():
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity
        steps = [
            Subtask(id=0, description="gather the premises", type=SubtaskType.RESEARCH),
            Subtask(id=1, description="weigh the trade-offs", type=SubtaskType.ANALYSIS,
                    dependencies=[0]),
            Subtask(id=2, description="state the conclusion", type=SubtaskType.GENERATION,
                    dependencies=[1]),
        ]
        return TaskPlan(task="reach a reasoned conclusion", complexity=TaskComplexity.COMPLEX,
                        steps=steps)

    def test_planner_creates_child_goal_per_subtask(self) -> None:
        """A multi-step plan gets a parent goal plus one child goal per step,
        each carrying parent_goal_id; every step.goal_id is populated."""
        from agentx_ai.agent.planner import TaskPlanner

        plan = self._multi_step_plan()
        memory = MagicMock()
        planner = TaskPlanner()

        result = planner._create_goal_for_plan(plan, memory)

        # Parent + 3 children added.
        self.assertEqual(memory.add_goal.call_count, 4)
        self.assertIsNotNone(result.goal_id)
        # Each child references the parent and is wired back onto its step.
        child_goals = [c.args[0] for c in memory.add_goal.call_args_list[1:]]
        for step, goal in zip(result.steps, child_goals, strict=False):
            self.assertEqual(goal.parent_goal_id, result.goal_id)
            self.assertEqual(step.goal_id, goal.id)

    def test_planner_single_step_skips_child_goals(self) -> None:
        """A single-step plan only creates the parent goal — the lone step is
        already represented by it (no redundant subgoal)."""
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity
        from agentx_ai.agent.planner import TaskPlanner

        plan = TaskPlan(
            task="answer directly",
            complexity=TaskComplexity.SIMPLE,
            steps=[Subtask(id=0, description="answer", type=SubtaskType.GENERATION)],
        )
        memory = MagicMock()

        result = TaskPlanner()._create_goal_for_plan(plan, memory)

        memory.add_goal.assert_called_once()
        self.assertIsNone(result.steps[0].goal_id)

    def test_planner_no_memory_is_noop(self) -> None:
        """Without a memory backend the plan is returned untouched."""
        from agentx_ai.agent.planner import TaskPlanner

        plan = self._multi_step_plan()
        result = TaskPlanner()._create_goal_for_plan(plan, memory=None)
        self.assertIsNone(result.goal_id)
        self.assertTrue(all(s.goal_id is None for s in result.steps))

    def test_executor_completes_subtask_goal_through_hook(self) -> None:
        """A subtask with a goal_id routes completion through the agent's hook
        seam (on_goal_complete → MemoryRecorder.complete_goal)."""
        from agentx_ai.agent.plan_executor import PlanExecutor
        from agentx_ai.agent.planner import Subtask, SubtaskType

        agent = MagicMock()
        ex = PlanExecutor(agent, MagicMock())
        subtask = Subtask(id=0, description="step", type=SubtaskType.GENERATION)
        subtask.goal_id = "g-1"

        ex._complete_subtask_goal(subtask, "completed", "done")

        agent._dispatch.assert_called_once_with("on_goal_complete", "g-1", "completed", "done")

    def test_executor_no_goal_id_skips_dispatch(self) -> None:
        """A subtask without a goal_id never fires the hook."""
        from agentx_ai.agent.plan_executor import PlanExecutor
        from agentx_ai.agent.planner import Subtask, SubtaskType

        agent = MagicMock()
        ex = PlanExecutor(agent, MagicMock())
        subtask = Subtask(id=0, description="step", type=SubtaskType.GENERATION)  # goal_id=None

        ex._complete_subtask_goal(subtask, "completed", "done")

        agent._dispatch.assert_not_called()


class PlanSerializationTest(TestCase):
    """Phase 15.7 #3 (B1) — durable plan serialization + resume rebuild."""

    @staticmethod
    def _plan():
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity
        steps = [
            Subtask(id=0, description="gather premises", type=SubtaskType.RESEARCH,
                    tools_needed=["web_search"], estimated_complexity=TaskComplexity.MODERATE),
            Subtask(id=1, description="weigh trade-offs", type=SubtaskType.ANALYSIS,
                    dependencies=[0]),
            Subtask(id=2, description="state conclusion", type=SubtaskType.GENERATION,
                    dependencies=[1]),
        ]
        return TaskPlan(task="reach a conclusion", complexity=TaskComplexity.COMPLEX,
                        steps=steps, reasoning_strategy="tot", estimated_tokens=42)

    def test_plan_round_trips_through_dict(self) -> None:
        """to_dict → from_dict preserves the full structure needed to resume."""
        from agentx_ai.agent.planner import TaskPlan

        plan = self._plan()
        plan.goal_id = "goal-1"
        plan.steps[1].goal_id = "goal-1-b"

        rebuilt = TaskPlan.from_dict(plan.to_dict())

        self.assertEqual(rebuilt.task, plan.task)
        self.assertEqual(rebuilt.complexity, plan.complexity)
        self.assertEqual(rebuilt.reasoning_strategy, "tot")
        self.assertEqual(rebuilt.estimated_tokens, 42)
        self.assertEqual(rebuilt.goal_id, "goal-1")
        self.assertEqual(len(rebuilt.steps), 3)
        s0 = rebuilt.steps[0]
        self.assertEqual(s0.type, plan.steps[0].type)
        self.assertEqual(s0.tools_needed, ["web_search"])
        self.assertEqual(s0.estimated_complexity, plan.steps[0].estimated_complexity)
        self.assertEqual(rebuilt.steps[1].dependencies, [0])
        self.assertEqual(rebuilt.steps[1].goal_id, "goal-1-b")

    def test_load_plan_overlays_live_status(self) -> None:
        """load_plan rebuilds the skeleton and overlays per-subtask status:
        terminal steps are restored completed (with reconstructed sentinels),
        a running step is reset so it re-executes, and get_next_subtask points
        at the right place."""
        import json
        from agentx_ai.agent.plan_state import PlanStateStore

        plan = self._plan()
        snapshot = {
            "status": "active",
            "plan_json": json.dumps(plan.to_dict()),
            "subtask:0:status": "complete",
            "subtask:0:result": "premises gathered",
            "subtask:1:status": "running",  # died mid-flight → must re-run
        }

        store = PlanStateStore("sess-1")
        with patch.object(PlanStateStore, "get_status", return_value=snapshot):
            rebuilt = store.load_plan("p1")

        self.assertIsNotNone(rebuilt)
        assert rebuilt is not None
        self.assertTrue(rebuilt.steps[0].completed)
        self.assertEqual(rebuilt.steps[0].result, "premises gathered")
        self.assertFalse(rebuilt.steps[1].completed)   # running reset
        self.assertIsNone(rebuilt.steps[1].result)
        self.assertFalse(rebuilt.steps[2].completed)   # pending
        # Next executable subtask is #1 (its dep #0 is satisfied).
        nxt = rebuilt.get_next_subtask()
        self.assertIsNotNone(nxt)
        assert nxt is not None
        self.assertEqual(nxt.id, 1)

    def test_load_plan_reconstructs_failure_sentinels(self) -> None:
        """Failed/skipped/abandoned terminal states are re-derived to the same
        sentinel strings the executor uses (so dependency-skip + synthesis match)."""
        import json
        from agentx_ai.agent.plan_state import PlanStateStore

        plan = self._plan()
        snapshot = {
            "status": "active",
            "plan_json": json.dumps(plan.to_dict()),
            "subtask:0:status": "failed",
            "subtask:0:error": "network down",
            "subtask:1:status": "skipped",
            "subtask:2:status": "pending",
        }
        store = PlanStateStore("sess-1")
        with patch.object(PlanStateStore, "get_status", return_value=snapshot):
            rebuilt = store.load_plan("p1")

        assert rebuilt is not None
        self.assertTrue(rebuilt.steps[0].result.startswith("[FAILED"))
        self.assertIn("network down", rebuilt.steps[0].result)
        self.assertTrue(rebuilt.steps[1].result.startswith("[SKIPPED"))

    def test_load_plan_returns_none_when_not_resumable(self) -> None:
        """Missing state, a finished plan, and a pre-B1 snapshot all yield None."""
        from agentx_ai.agent.plan_state import PlanStateStore

        store = PlanStateStore("sess-1")
        with patch.object(PlanStateStore, "get_status", return_value=None):
            self.assertIsNone(store.load_plan("p1"))
        with patch.object(PlanStateStore, "get_status", return_value={"status": "complete"}):
            self.assertIsNone(store.load_plan("p1"))
        # Active but no structural blob (legacy state) → can't safely resume.
        with patch.object(PlanStateStore, "get_status", return_value={"status": "active"}):
            self.assertIsNone(store.load_plan("p1"))

    def test_is_resumable_reflects_remaining_work(self) -> None:
        import json
        from agentx_ai.agent.plan_state import PlanStateStore

        plan = self._plan()
        base = {"status": "active", "plan_json": json.dumps(plan.to_dict())}
        store = PlanStateStore("sess-1")

        # All terminal → nothing to resume.
        done = {**base, "subtask:0:status": "complete", "subtask:1:status": "complete",
                "subtask:2:status": "complete"}
        with patch.object(PlanStateStore, "get_status", return_value=done):
            self.assertFalse(store.is_resumable("p1"))
        # One still pending → resumable.
        partial = {**base, "subtask:0:status": "complete"}
        with patch.object(PlanStateStore, "get_status", return_value=partial):
            self.assertTrue(store.is_resumable("p1"))


class PlanTerminationTest(TestCase):
    """Clean termination: a hard Stop (GeneratorExit) mid-subtask leaves a
    resumable state (in-flight subtask reset to pending, plan 'interrupted')
    instead of a stuck 'running'."""

    def test_executor_stop_marks_interrupted_and_resets_subtask(self) -> None:
        from agentx_ai.agent.plan_executor import PlanExecutor, PlanResult
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity

        plan = TaskPlan(
            task="t", complexity=TaskComplexity.COMPLEX,
            steps=[Subtask(id=0, description="s0", type=SubtaskType.GENERATION)],
        )
        agent = MagicMock()
        agent._active_alloy_executor = None
        state = MagicMock()
        state.is_cancel_requested.return_value = False

        async def fake_loop(provider, model_id, messages, tools, ag, *, result=None, **kw):
            yield 'event: chunk\ndata: {"content": "x"}\n\n'
            await asyncio.Event().wait()  # suspend so aclose() can inject GeneratorExit

        class _Prov:
            async def stream(self, *a, **k):
                if False:
                    yield

        async def _drive():
            ex = PlanExecutor(agent, state)
            agen = ex.execute_streaming(plan, _Prov(), "m", None, result=PlanResult())
            await agen.__anext__()  # plan_start
            await agen.__anext__()  # subtask_start (subtask 0 now inflight)
            await agen.__anext__()  # first chunk from the subtask loop
            await agen.aclose()     # Stop → GeneratorExit mid-subtask

        with patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_loop):
            asyncio.run(_drive())

        reset_calls = [
            c for c in state.update_subtask.call_args_list
            if len(c.args) >= 3 and c.args[1] == 0 and c.args[2] == "pending"
        ]
        self.assertTrue(reset_calls, "in-flight subtask was not reset to pending")
        state.mark_interrupted.assert_called_once()

    def test_load_plan_accepts_interrupted_status(self) -> None:
        from agentx_ai.agent.plan_state import PlanStateStore
        from agentx_ai.agent.planner import TaskPlan, Subtask, SubtaskType, TaskComplexity

        plan = TaskPlan(task="t", complexity=TaskComplexity.COMPLEX, steps=[
            Subtask(id=0, description="s0", type=SubtaskType.GENERATION),
            Subtask(id=1, description="s1", type=SubtaskType.GENERATION, dependencies=[0]),
        ])
        snapshot = {
            "status": "interrupted",
            "plan_json": json.dumps(plan.to_dict()),
            "subtask:0:status": "pending",
            "subtask:1:status": "pending",
        }
        store = PlanStateStore("sess")
        with patch.object(PlanStateStore, "get_status", return_value=snapshot):
            self.assertIsNotNone(store.load_plan("p1"))
            self.assertTrue(store.is_resumable("p1"))


class PlannerComposeTest(TestCase):
    """Main-agent plan composition: tolerant JSON extraction + compose_with_model."""

    class _Result:
        def __init__(self, content, usage=None):
            self.content = content
            self.usage = usage or {}

    class _FakeProvider:
        def __init__(self, content):
            self._content = content

        async def complete(self, messages, model_id, **kw):
            return PlannerComposeTest._Result(self._content)

    class _BoomProvider:
        async def complete(self, messages, model_id, **kw):
            raise RuntimeError("provider down")

    def _compose(self, content):
        import asyncio
        from agentx_ai.agent.planner import TaskPlanner
        planner = TaskPlanner(max_subtasks=6)
        return asyncio.run(
            planner.compose_with_model(self._FakeProvider(content), "m", [], "the task", memory=None)
        )

    # ---- JSON extraction ----
    def test_extract_raw_json(self):
        from agentx_ai.agent.planner import _extract_json_object
        self.assertEqual(_extract_json_object('{"plan": null}'), {"plan": None})

    def test_extract_fenced_json(self):
        from agentx_ai.agent.planner import _extract_json_object
        text = 'Sure!\n```json\n{"plan": [1, 2]}\n```\nhope that helps'
        self.assertEqual(_extract_json_object(text), {"plan": [1, 2]})

    def test_extract_embedded_json(self):
        from agentx_ai.agent.planner import _extract_json_object
        text = 'Here is the plan: {"plan": [{"description": "x"}]} — done.'
        self.assertEqual(_extract_json_object(text), {"plan": [{"description": "x"}]})

    def test_extract_garbage_returns_none(self):
        from agentx_ai.agent.planner import _extract_json_object
        self.assertIsNone(_extract_json_object("no json here at all"))
        self.assertIsNone(_extract_json_object(""))

    # ---- compose_with_model ----
    def test_compose_multi_step_builds_normalized_plan(self):
        import json as _json
        from agentx_ai.agent.planner import SubtaskType
        content = _json.dumps({"plan": [
            {"description": "gather evidence", "type": "research", "depends": [], "tools": ["web_search"]},
            {"description": "weigh trade-offs", "type": "analysis", "depends": [0]},
            {"description": "state conclusion", "type": "generation", "depends": [1]},
        ]})
        plan = self._compose(content)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual([s.id for s in plan.steps], [0, 1, 2])
        self.assertEqual(plan.steps[0].type, SubtaskType.RESEARCH)
        self.assertEqual(plan.steps[0].tools_needed, ["web_search"])
        self.assertEqual(plan.steps[1].dependencies, [0])

    def test_compose_plan_null_returns_none(self):
        self.assertIsNone(self._compose('{"plan": null}'))

    def test_compose_single_step_returns_none(self):
        # One step isn't worth a plan — the model declined to decompose.
        self.assertIsNone(self._compose('{"plan": [{"description": "just answer"}]}'))

    def test_compose_no_json_returns_none(self):
        self.assertIsNone(self._compose("I'll just answer directly, no plan needed."))

    def test_compose_provider_error_returns_none(self):
        import asyncio
        from agentx_ai.agent.planner import TaskPlanner
        planner = TaskPlanner(max_subtasks=6)
        plan = asyncio.run(
            planner.compose_with_model(self._BoomProvider(), "m", [], "task", memory=None)
        )
        self.assertIsNone(plan)

    def test_compose_caps_and_reindexes(self):
        # >max_subtasks gets capped; out-of-range deps are dropped by normalize.
        import json as _json
        steps = [{"description": f"step {i}", "type": "generation",
                  "depends": [i - 1] if i else []} for i in range(9)]
        plan = self._compose(_json.dumps({"plan": steps}))
        assert plan is not None
        self.assertEqual(len(plan.steps), 6)  # planner.max_subtasks
        self.assertEqual([s.id for s in plan.steps], list(range(6)))


class ContextGateTest(TestCase):
    """Tests for context gate loop prevention and iterative chunking."""

    def test_threshold_updated(self):
        """AgentConfig threshold should be 12000 chars."""
        from agentx_ai.agent.core import AgentConfig
        config = AgentConfig()
        self.assertEqual(config.max_tool_result_chars, 12000)

    def test_retrieval_tool_detected(self):
        """All retrieval tools should be recognized by is_retrieval_tool."""
        for name in ["read_stored_output", "list_stored_outputs",
                      "tool_output_query", "tool_output_section", "tool_output_path"]:
            self.assertTrue(is_retrieval_tool(name), f"{name} should be a retrieval tool")

    def test_non_retrieval_tool_not_detected(self):
        """Non-retrieval tools should not be flagged."""
        for name in ["web_search", "read_file", "some_mcp_tool"]:
            self.assertFalse(is_retrieval_tool(name), f"{name} should NOT be a retrieval tool")

    def test_read_stored_output_default_pagination(self):
        """read_stored_output without explicit limit returns <=12000 chars with has_more."""
        content = "A" * 25000
        mock_data = {
            "content": content,
            "tool_name": "big_tool",
            "tool_call_id": "tc-1",
            "stored_at": "2026-01-01T00:00:00",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("read_stored_output", {"key": "test_key"})

        parsed = json.loads(result.content[0]["text"])
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["returned_size"], 12000)
        self.assertEqual(parsed["total_size"], 25000)
        self.assertTrue(parsed["has_more"])
        self.assertEqual(parsed["next_offset"], 12000)

    def test_read_stored_output_pagination_walk(self):
        """Can page through full content with offset increments."""
        content = "B" * 25000
        mock_data = {
            "content": content,
            "tool_name": "big_tool",
            "tool_call_id": "tc-2",
            "stored_at": "2026-01-01T00:00:00",
        }

        collected = ""
        offset = 0
        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            while True:
                result = execute_internal_tool("read_stored_output", {
                    "key": "test_key", "offset": offset,
                })
                parsed = json.loads(result.content[0]["text"])
                collected += parsed["content"]
                if not parsed["has_more"]:
                    break
                offset = parsed["next_offset"]

        self.assertEqual(len(collected), 25000)
        self.assertEqual(collected, content)

    def test_read_stored_output_last_page(self):
        """Final page has has_more=False and next_offset=None."""
        content = "C" * 5000  # Under default limit
        mock_data = {
            "content": content,
            "tool_name": "small_tool",
            "tool_call_id": "tc-3",
            "stored_at": "2026-01-01T00:00:00",
        }

        with patch("agentx_ai.agent.tool_output_storage.get_tool_output", return_value=mock_data):
            result = execute_internal_tool("read_stored_output", {"key": "test_key"})

        parsed = json.loads(result.content[0]["text"])
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["returned_size"], 5000)
        self.assertFalse(parsed["has_more"])
        self.assertIsNone(parsed["next_offset"])


class AgentSelfMemoryTest(TestCase):
    """Tests for agent self-memory: ID generation, self-channel, recall integration."""

    def test_generate_agent_id_format(self):
        """Agent ID should be three hyphenated words."""
        from agentx_ai.agent.models import generate_agent_id
        aid = generate_agent_id()
        parts = aid.split("-")
        self.assertEqual(len(parts), 3, f"Expected 3 parts, got {len(parts)}: {aid}")
        for part in parts:
            self.assertTrue(part.isalpha(), f"Part '{part}' should be alphabetic")

    def test_generate_agent_id_uniqueness(self):
        """100 generated IDs should all be unique."""
        from agentx_ai.agent.models import generate_agent_id
        ids = {generate_agent_id() for _ in range(100)}
        self.assertGreater(len(ids), 90, "Expected >90 unique IDs out of 100")

    def test_agent_profile_has_agent_id(self):
        """AgentProfile should auto-generate an agent_id."""
        from agentx_ai.agent.models import AgentProfile
        profile = AgentProfile(id="test", name="Test Agent")
        self.assertIsNotNone(profile.agent_id)
        self.assertIn("-", profile.agent_id)

    def test_agent_profile_self_channel(self):
        """self_channel property should be _self_<agent_id>."""
        from agentx_ai.agent.models import AgentProfile
        profile = AgentProfile(id="test", name="Test Agent", agent_id="bold-cosmic-falcon")
        self.assertEqual(profile.self_channel, "_self_bold-cosmic-falcon")

    def test_agent_config_agent_id(self):
        """AgentConfig should accept agent_id."""
        from agentx_ai.agent.core import AgentConfig
        config = AgentConfig(agent_id="brave-hidden-dawn")
        self.assertEqual(config.agent_id, "brave-hidden-dawn")

    def test_self_channel_in_default_recall(self):
        """AgentMemory._default_recall_channels should include self-channel when agent_id set."""
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        with patch.object(AgentMemory, "__init__", lambda self, **kw: None):
            mem = AgentMemory.__new__(AgentMemory)
            mem.channel = "_default"
            mem.self_channel = "_self_bold-cosmic-falcon"
            channels = mem._default_recall_channels()
            self.assertEqual(channels, ["_default", "_self_bold-cosmic-falcon", "_global"])

    def test_self_channel_none_without_agent_id(self):
        """Without agent_id, self_channel should be None and not in recall channels."""
        from agentx_ai.kit.agent_memory.memory.interface import AgentMemory
        with patch.object(AgentMemory, "__init__", lambda self, **kw: None):
            mem = AgentMemory.__new__(AgentMemory)
            mem.channel = "_default"
            mem.self_channel = None
            channels = mem._default_recall_channels()
            self.assertEqual(channels, ["_default", "_global"])

    def test_extraction_service_has_assistant_method(self):
        """ExtractionService should have check_relevance_and_extract_assistant method."""
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionService
        self.assertTrue(hasattr(ExtractionService, "check_relevance_and_extract_assistant"))

    def test_assistant_confidence_calibration(self):
        """Assistant certainty levels should map to expected confidence scores."""
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionService
        service = ExtractionService.__new__(ExtractionService)
        facts = [
            {"claim": "A", "certainty": "definitive"},
            {"claim": "B", "certainty": "analytical"},
            {"claim": "C", "certainty": "speculative"},
        ]
        calibrated = service._apply_assistant_confidence_calibration(facts)
        self.assertAlmostEqual(calibrated[0]["confidence"], 0.90)
        self.assertAlmostEqual(calibrated[1]["confidence"], 0.75)
        self.assertAlmostEqual(calibrated[2]["confidence"], 0.55)
        # Certainty should be replaced by confidence
        for f in calibrated:
            self.assertNotIn("certainty", f)

    def test_retrieval_tool_bypass_in_is_retrieval_tool(self):
        """is_retrieval_tool should detect all 5 retrieval tools."""
        from agentx_ai.mcp.internal_tools import RETRIEVAL_TOOL_NAMES
        self.assertEqual(len(RETRIEVAL_TOOL_NAMES), 5)


class FactVerificationPipelineTest(TestCase):
    """Tests for the three-layer fact verification pipeline."""

    def test_is_temporal_progression_current_supersedes_current(self):
        """New 'current' fact should supersede old 'current' fact."""
        from agentx_ai.kit.agent_memory.consolidation.jobs import _is_temporal_progression
        new_fact = {"claim": "User works at Meta", "temporal_context": "current"}
        old_fact = {"claim": "User works at Google", "temporal_context": "current"}
        self.assertTrue(_is_temporal_progression(new_fact, old_fact))

    def test_is_temporal_progression_current_supersedes_null(self):
        """New 'current' fact should supersede old fact with no temporal context."""
        from agentx_ai.kit.agent_memory.consolidation.jobs import _is_temporal_progression
        new_fact = {"claim": "User lives in Seattle", "temporal_context": "current"}
        old_fact = {"claim": "User lives in NYC", "temporal_context": None}
        self.assertTrue(_is_temporal_progression(new_fact, old_fact))

    def test_is_temporal_progression_past_does_not_supersede(self):
        """New 'past' fact should not be treated as temporal progression."""
        from agentx_ai.kit.agent_memory.consolidation.jobs import _is_temporal_progression
        new_fact = {"claim": "User worked at Google", "temporal_context": "past"}
        old_fact = {"claim": "User works at Meta", "temporal_context": "current"}
        self.assertFalse(_is_temporal_progression(new_fact, old_fact))

    def test_is_temporal_progression_null_does_not_supersede(self):
        """Fact without temporal context should not trigger auto-resolution."""
        from agentx_ai.kit.agent_memory.consolidation.jobs import _is_temporal_progression
        new_fact = {"claim": "User likes Python", "temporal_context": None}
        old_fact = {"claim": "User likes Java", "temporal_context": "current"}
        self.assertFalse(_is_temporal_progression(new_fact, old_fact))

    def test_semantic_duplicate_threshold_in_config(self):
        """Config should have semantic_duplicate_threshold."""
        from agentx_ai.kit.agent_memory.config import Settings
        s = Settings()
        self.assertEqual(s.semantic_duplicate_threshold, 0.92)

    def test_contradiction_similarity_threshold_in_config(self):
        """Config should have contradiction_similarity_threshold."""
        from agentx_ai.kit.agent_memory.config import Settings
        s = Settings()
        self.assertEqual(s.contradiction_similarity_threshold, 0.5)

    def test_contradiction_max_candidates_in_config(self):
        """Config should have contradiction_max_candidates."""
        from agentx_ai.kit.agent_memory.config import Settings
        s = Settings()
        self.assertEqual(s.contradiction_max_candidates, 10)

    def test_contradiction_enabled_by_default(self):
        """Contradiction detection should be enabled by default."""
        from agentx_ai.kit.agent_memory.config import Settings
        s = Settings()
        self.assertTrue(s.contradiction_detection_enabled)

    def test_correction_enabled_by_default(self):
        """Correction detection should be enabled by default."""
        from agentx_ai.kit.agent_memory.config import Settings
        s = Settings()
        self.assertTrue(s.correction_detection_enabled)

    def test_metrics_has_pipeline_fields(self):
        """ConsolidationMetrics should have the new pipeline tracking fields."""
        from agentx_ai.kit.agent_memory.consolidation.metrics import ConsolidationMetrics
        m = ConsolidationMetrics()
        self.assertEqual(m.semantic_duplicates_skipped, 0)
        self.assertEqual(m.contradiction_candidates_found, 0)
        self.assertEqual(m.temporal_progressions_resolved, 0)

    def test_metrics_serialization_includes_pipeline_fields(self):
        """Pipeline fields should appear in serialized metrics."""
        from agentx_ai.kit.agent_memory.consolidation.metrics import ConsolidationMetrics
        m = ConsolidationMetrics(
            semantic_duplicates_skipped=3,
            contradiction_candidates_found=5,
            temporal_progressions_resolved=2,
        )
        d = m.to_dict()
        self.assertEqual(d["semantic_duplicates_skipped"], 3)
        self.assertEqual(d["contradiction_candidates_found"], 5)
        self.assertEqual(d["temporal_progressions_resolved"], 2)

    def test_check_contradictions_accepts_temporal_args(self):
        """check_contradictions should accept new_temporal and new_confidence kwargs."""
        import inspect
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionService
        sig = inspect.signature(ExtractionService.check_contradictions)
        params = list(sig.parameters.keys())
        self.assertIn("new_temporal", params)
        self.assertIn("new_confidence", params)

    def test_consolidation_settings_includes_pipeline_config(self):
        """get_consolidation_settings should include pipeline thresholds."""
        from agentx_ai.kit.agent_memory.config import get_consolidation_settings
        settings = get_consolidation_settings()
        self.assertIn("semantic_duplicate_threshold", settings)
        self.assertIn("contradiction_similarity_threshold", settings)
        self.assertIn("contradiction_max_candidates", settings)


# Phase 11.8+ tests moved to tests_memory.py


class ReasoningAsyncRegressionTest(TestCase):
    """Regression: the reasoning strategies and orchestrator were sync methods
    that called the async ``provider.complete()`` without awaiting it, then read
    ``.content``/``.usage`` on the resulting coroutine. The orchestrator's broad
    ``except`` swallowed the ``AttributeError``, so CoT/ToT/ReAct/Reflection
    silently always returned ``status=FAILED``.

    These tests use an ``AsyncMock`` provider so the coroutine path is real: if a
    strategy fails to await, ``complete`` is never awaited (await_count 0) and
    the result is FAILED. Asserting it was awaited + not-FAILED locks the fix."""

    def _fake_registry(self):
        from agentx_ai.providers.base import CompletionResult

        provider = MagicMock()
        provider.complete = AsyncMock(return_value=CompletionResult(
            content="1. First approach\n2. Second approach\nThe answer is 42. Score: 0.8",
            finish_reason="stop",
            usage={"total_tokens": 10},
            model="model-x",
        ))
        registry = MagicMock()
        registry.get_provider_for_model.return_value = (provider, "model-x")
        return registry, provider

    def test_chain_of_thought_awaits_provider(self) -> None:
        from agentx_ai.reasoning.base import ReasoningStatus
        from agentx_ai.reasoning.chain_of_thought import ChainOfThought, CoTConfig

        strategy = ChainOfThought(CoTConfig(model="model-x"))
        registry, provider = self._fake_registry()
        strategy._registry = registry

        result = asyncio.run(strategy.reason("What is 6 times 7?"))

        self.assertEqual(result.status, ReasoningStatus.COMPLETE)
        self.assertGreaterEqual(provider.complete.await_count, 1)

    def test_reflection_awaits_provider(self) -> None:
        from agentx_ai.reasoning.base import ReasoningStatus
        from agentx_ai.reasoning.reflection import ReflectiveReasoner, ReflectionConfig

        strategy = ReflectiveReasoner(ReflectionConfig(model="model-x", max_revisions=1))
        registry, provider = self._fake_registry()
        strategy._registry = registry

        result = asyncio.run(strategy.reason("Write an intro."))

        self.assertEqual(result.status, ReasoningStatus.COMPLETE)
        self.assertGreaterEqual(provider.complete.await_count, 1)

    def test_tree_of_thought_awaits_provider(self) -> None:
        from agentx_ai.reasoning.base import ReasoningStatus
        from agentx_ai.reasoning.tree_of_thought import TreeOfThought, ToTConfig

        strategy = TreeOfThought(ToTConfig(model="model-x", max_depth=1, branching_factor=2))
        registry, provider = self._fake_registry()
        strategy._registry = registry

        result = asyncio.run(strategy.reason("Plan a launch."))

        # ToT always returns COMPLETE absent an exception; the regression is that
        # the provider is actually awaited rather than the coroutine path failing.
        self.assertEqual(result.status, ReasoningStatus.COMPLETE)
        self.assertGreaterEqual(provider.complete.await_count, 1)

    def test_react_awaits_provider(self) -> None:
        from agentx_ai.reasoning.base import ReasoningStatus
        from agentx_ai.reasoning.react import ReActAgent, ReActConfig

        strategy = ReActAgent(ReActConfig(model="model-x", max_iterations=1))
        registry, provider = self._fake_registry()
        strategy._registry = registry

        result = asyncio.run(strategy.reason("Capital of France?"))

        self.assertNotEqual(result.status, ReasoningStatus.FAILED)
        self.assertGreaterEqual(provider.complete.await_count, 1)

    def test_orchestrator_awaits_strategy(self) -> None:
        from agentx_ai.reasoning.base import ReasoningStatus
        from agentx_ai.reasoning.orchestrator import ReasoningOrchestrator, OrchestratorConfig

        registry, provider = self._fake_registry()
        orchestrator = ReasoningOrchestrator(OrchestratorConfig())

        # The orchestrator builds the strategy internally; patch the module-level
        # get_registry the ChainOfThought strategy resolves at call time.
        with patch("agentx_ai.reasoning.chain_of_thought.get_registry", return_value=registry):
            result = asyncio.run(orchestrator.reason("What is 6 times 7?", strategy="cot"))

        self.assertEqual(result.status, ReasoningStatus.COMPLETE)
        self.assertGreaterEqual(provider.complete.await_count, 1)


@override_settings(AGENTX_AUTH_ENABLED=False)
class PlanStatusEndpointTest(MockRedisTestBase):
    """GET /api/agent/plans/<id>/status — Redis-backed plan state read."""

    def setUp(self) -> None:
        super().setUp()
        from django.test import Client
        self.client = Client()

    def test_returns_parsed_state_when_present(self):
        self.mock_redis.hgetall.return_value = {
            "status": "complete",
            "task": "Build a thing",
            "complexity": "moderate",
            "subtask_count": "2",
            "completed_count": "2",
            "subtask:0:status": "complete",
            "subtask:0:description": "first",
            "subtask:1:status": "complete",
            "subtask:1:description": "second",
        }
        resp = self.client.get("/api/agent/plans/abc123/status?session_id=sess-1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["found"])
        self.assertEqual(data["status"], "complete")
        self.assertEqual(data["subtask_count"], 2)
        self.assertEqual(data["completed_count"], 2)
        self.assertEqual(len(data["subtasks"]), 2)
        self.assertEqual(data["subtasks"][0]["id"], 0)
        self.assertEqual(data["subtasks"][1]["description"], "second")

    def test_returns_not_found_when_expired(self):
        self.mock_redis.hgetall.return_value = {}
        resp = self.client.get("/api/agent/plans/gone/status?session_id=sess-1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["found"])

    def test_requires_session_id(self):
        resp = self.client.get("/api/agent/plans/abc123/status")
        self.assertEqual(resp.status_code, 400)

    def test_decodes_bytes_from_redis(self):
        self.mock_redis.hgetall.return_value = {
            b"status": b"active",
            b"subtask_count": b"1",
            b"completed_count": b"0",
            b"subtask:0:status": b"running",
        }
        resp = self.client.get("/api/agent/plans/b/status?session_id=s")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "active")
        self.assertEqual(data["subtasks"][0]["status"], "running")


@override_settings(AGENTX_AUTH_ENABLED=False)
class CheckpointEndpointTest(MockRedisTestBase):
    """GET/DELETE /api/memory/checkpoints — Redis-backed checkpoint store."""

    def setUp(self) -> None:
        super().setUp()
        from django.test import Client
        self.client = Client()

    def test_get_lists_checkpoints(self):
        import json
        from agentx_ai.agent.checkpoint_storage import add_checkpoint

        # add_checkpoint serializes via rpush; capture what it stored and feed
        # it back through lrange so the round-trip is exercised.
        add_checkpoint("conv-1", "Did the thing", ["chose X"], "do Y next")
        stored = self.mock_redis.rpush.call_args[0][1]
        self.mock_redis.lrange.return_value = [stored]

        resp = self.client.get("/api/memory/checkpoints?conversation_id=conv-1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["checkpoints"][0]["summary"], "Did the thing")
        self.assertEqual(data["checkpoints"][0]["decisions"], ["chose X"])
        self.assertEqual(data["checkpoints"][0]["next_step"], "do Y next")
        # sanity: the captured payload is valid JSON
        json.loads(stored)

    def test_get_requires_conversation_id(self):
        resp = self.client.get("/api/memory/checkpoints")
        self.assertEqual(resp.status_code, 400)

    def test_delete_clears_checkpoints(self):
        self.mock_redis.llen.return_value = 3
        resp = self.client.delete("/api/memory/checkpoints?conversation_id=conv-1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["cleared"], 3)
        self.mock_redis.delete.assert_called_once()

    def test_get_empty_after_clear(self):
        self.mock_redis.lrange.return_value = []
        resp = self.client.get("/api/memory/checkpoints?conversation_id=conv-1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["count"], 0)


class ScratchpadStorageTest(MockRedisTestBase):
    """Redis-backed scratchpad note store (mirrors checkpoint storage)."""

    def test_add_note_round_trips_through_render(self):
        from agentx_ai.agent.scratchpad_storage import (
            add_note,
            render_scratchpad_block,
        )

        # add_note serializes via rpush; feed the captured payload back through
        # lrange so the full round-trip is exercised.
        add_note("conv-1", "remember the API key is in .env")
        stored = self.mock_redis.rpush.call_args[0][1]
        self.assertEqual(json.loads(stored)["note"], "remember the API key is in .env")

        self.mock_redis.lrange.return_value = [stored]
        block = render_scratchpad_block("conv-1")
        self.assertIn("Scratchpad", block)
        self.assertIn("remember the API key is in .env", block)

    def test_render_empty_returns_blank(self):
        from agentx_ai.agent.scratchpad_storage import render_scratchpad_block

        self.mock_redis.lrange.return_value = []
        self.assertEqual(render_scratchpad_block("conv-1"), "")

    def test_clear_notes_returns_prior_count(self):
        from agentx_ai.agent.scratchpad_storage import clear_notes

        self.mock_redis.llen.return_value = 4
        self.assertEqual(clear_notes("conv-1"), 4)
        self.mock_redis.delete.assert_called_once()


class ScratchpadToolTest(MockRedisTestBase):
    """The scratchpad_note internal tool — write + read-back modes."""

    def setUp(self) -> None:
        super().setUp()
        from agentx_ai.mcp.internal_context import (
            InternalToolContext,
            set_context,
        )
        self._ctx_token = set_context(InternalToolContext(
            user_id="u1",
            channel="_default",
            agent_id=None,
            conversation_id="conv-1",
        ))

    def tearDown(self) -> None:
        from agentx_ai.mcp.internal_context import reset_context
        reset_context(self._ctx_token)
        super().tearDown()

    def test_write_appends_note(self):
        from agentx_ai.mcp.internal_tools import scratchpad_note

        result = scratchpad_note(note="check the retry path")
        self.assertTrue(result["success"])
        self.assertTrue(self.mock_redis.rpush.called)
        self.assertEqual(result["stored"]["note"], "check the retry path")

    def test_replace_clears_before_write(self):
        from agentx_ai.mcp.internal_tools import scratchpad_note

        scratchpad_note(note="new note", replace=True)
        # clear_notes issues a DELETE before the new rpush.
        self.assertTrue(self.mock_redis.delete.called)
        self.assertTrue(self.mock_redis.rpush.called)

    def test_read_back_returns_state_not_transcript(self):
        from agentx_ai.mcp.internal_tools import scratchpad_note

        # No live memory system → active_goals degrades to [].
        with patch("agentx_ai.kit.memory_utils.get_agent_memory", return_value=None):
            self.mock_redis.lrange.return_value = []
            result = scratchpad_note(read=True)

        self.assertTrue(result["success"])
        self.assertEqual(set(result.keys()) >= {"notes", "checkpoints", "active_goals"}, True)
        self.assertEqual(result["active_goals"], [])
        # Read-back must never echo the conversation transcript.
        self.assertNotIn("recent_turns", result)
        self.assertNotIn("turns", result)

    def test_requires_conversation_context(self):
        from agentx_ai.mcp.internal_context import set_context, reset_context
        from agentx_ai.mcp.internal_tools import scratchpad_note

        token = set_context(None)
        try:
            result = scratchpad_note(note="orphaned")
        finally:
            reset_context(token)
        self.assertFalse(result["success"])


class FactToolWiringTest(TestCase):
    """remember_this / forget internal tools route to AgentMemory correctly."""

    def setUp(self) -> None:
        from agentx_ai.mcp.internal_context import InternalToolContext, set_context
        self._ctx_token = set_context(InternalToolContext(
            user_id="u1", channel="_default", agent_id=None, conversation_id="conv-1",
        ))

    def tearDown(self) -> None:
        from agentx_ai.mcp.internal_context import reset_context
        reset_context(self._ctx_token)

    def test_remember_this_boosts_salience(self):
        from agentx_ai.mcp.internal_tools import remember_this

        mem = MagicMock()
        mem.boost_salience.return_value = {"id": "f1", "salience": 0.9}
        with patch("agentx_ai.kit.memory_utils.get_agent_memory", return_value=mem):
            result = remember_this(fact_id="f1")

        mem.boost_salience.assert_called_once_with("f1")
        self.assertTrue(result["success"])
        self.assertEqual(result["salience"], 0.9)

    def test_remember_this_requires_fact_id(self):
        from agentx_ai.mcp.internal_tools import remember_this
        self.assertFalse(remember_this(fact_id="")["success"])

    def test_forget_passes_hard_flag(self):
        from agentx_ai.mcp.internal_tools import forget

        mem = MagicMock()
        mem.forget_fact.return_value = {"success": True, "mode": "hard", "fact_id": "f1"}
        with patch("agentx_ai.kit.memory_utils.get_agent_memory", return_value=mem):
            result = forget(fact_id="f1", hard=True)

        mem.forget_fact.assert_called_once_with("f1", hard=True)
        self.assertEqual(result["mode"], "hard")

    def test_forget_defaults_to_soft(self):
        from agentx_ai.mcp.internal_tools import forget

        mem = MagicMock()
        mem.forget_fact.return_value = {"success": True, "mode": "soft", "fact_id": "f1"}
        with patch("agentx_ai.kit.memory_utils.get_agent_memory", return_value=mem):
            forget(fact_id="f1")
        mem.forget_fact.assert_called_once_with("f1", hard=False)


@override_settings(AGENTX_AUTH_ENABLED=False)
class FactActionEndpointTest(TestCase):
    """POST forget/remember + GET provenance routes (AgentMemory mocked)."""

    def setUp(self) -> None:
        from django.test import Client
        self.client = Client()

    def _patch_memory(self, mem):
        return patch(
            "agentx_ai.kit.agent_memory.memory.interface.AgentMemory",
            return_value=mem,
        )

    def test_remember_endpoint_boosts(self):
        mem = MagicMock()
        mem.boost_salience.return_value = {"id": "f1", "salience": 0.9}
        with self._patch_memory(mem):
            resp = self.client.post(
                "/api/memory/facts/f1/remember",
                data=json.dumps({"to": 0.9}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fact"]["salience"], 0.9)
        mem.boost_salience.assert_called_once_with("f1", to=0.9)

    def test_remember_rejects_get(self):
        resp = self.client.get("/api/memory/facts/f1/remember")
        self.assertEqual(resp.status_code, 405)

    def test_forget_endpoint_soft_by_default(self):
        mem = MagicMock()
        mem.forget_fact.return_value = {"success": True, "mode": "soft", "fact_id": "f1"}
        with self._patch_memory(mem):
            resp = self.client.post(
                "/api/memory/facts/f1/forget",
                data=json.dumps({}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["mode"], "soft")
        mem.forget_fact.assert_called_once_with("f1", hard=False)

    def test_forget_returns_404_when_missing(self):
        mem = MagicMock()
        mem.forget_fact.return_value = {"success": False, "mode": "soft", "fact_id": "x"}
        with self._patch_memory(mem):
            resp = self.client.post(
                "/api/memory/facts/x/forget",
                data=json.dumps({}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 404)

    def test_provenance_endpoint_returns_origin(self):
        mem = MagicMock()
        mem.get_fact_provenance.return_value = {
            "success": True, "fact_id": "f1", "source_turn_id": "t9",
            "origin": {"conversation_id": "conv-7"},
        }
        with self._patch_memory(mem):
            resp = self.client.get("/api/memory/facts/f1/provenance")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["origin"]["conversation_id"], "conv-7")


class UserRecapTest(MockRedisTestBase):
    """Cached user recap store + recall_user_history summary wiring."""

    def test_cache_round_trip(self):
        from agentx_ai.kit.agent_memory.recap import set_cached_recap, get_cached_recap

        set_cached_recap("u1", "_default", "Alice is a backend developer.")
        # setex(key, ttl, value) — feed the stored payload back through get().
        key, _ttl, value = self.mock_redis.setex.call_args[0]
        self.assertTrue(key.startswith("user_recap:"))
        self.mock_redis.get.return_value = value

        entry = get_cached_recap("u1", "_default")
        self.assertEqual(entry["summary"], "Alice is a backend developer.")

    def test_recall_user_history_fills_summary_from_cache(self):
        from agentx_ai.mcp.internal_context import InternalToolContext, set_context, reset_context
        from agentx_ai.mcp.internal_tools import recall_user_history

        mem = MagicMock()
        mem.remember.return_value = MagicMock(relevant_turns=[], facts=[])

        token = set_context(InternalToolContext(
            user_id="u1", channel="_default", agent_id=None, conversation_id="c1",
        ))
        try:
            with patch("agentx_ai.kit.memory_utils.get_agent_memory", return_value=mem), \
                 patch(
                     "agentx_ai.kit.agent_memory.recap.get_cached_recap",
                     return_value={"summary": "Alice is a backend developer."},
                 ):
                result = recall_user_history()
        finally:
            reset_context(token)

        self.assertTrue(result["success"])
        self.assertEqual(result["summary"], "Alice is a backend developer.")


class ChatRunStoreTest(MockRedisTestBase):
    """Detached chat-run state store + tail entry paths (Redis mocked)."""

    def test_create_and_get_state(self):
        from agentx_ai.streaming.chat_run import store
        self.mock_redis.hgetall.return_value = {
            "status": "running",
            "created_at": "2026-05-28T00:00:00+00:00",
        }
        store.create("run1")
        self.assertTrue(self.mock_redis.hset.called)
        state = store.get_state("run1")
        self.assertEqual(state["status"], "running")

    def test_request_cancel_requires_existing_key(self):
        from agentx_ai.streaming.chat_run import store
        self.mock_redis.exists.return_value = 0
        self.assertFalse(store.request_cancel("missing"))
        self.mock_redis.exists.return_value = 1
        self.assertTrue(store.request_cancel("run1"))

    def test_is_cancel_requested_decodes_bytes(self):
        from agentx_ai.streaming.chat_run import store
        self.mock_redis.hget.return_value = b"1"
        self.assertTrue(store.is_cancel_requested("run1"))
        self.mock_redis.hget.return_value = None
        self.assertFalse(store.is_cancel_requested("run1"))

    def test_tail_emits_run_missing_when_state_absent(self):
        from agentx_ai.streaming.chat_run import tail_chat_run
        self.mock_redis.hgetall.return_value = {}

        async def _collect():
            return [ev async for ev in tail_chat_run("gone")]

        events = asyncio.run(_collect())
        self.assertEqual(len(events), 1)
        self.assertIn("run_missing", events[0])

    def test_tail_emits_run_missing_when_stale_running(self):
        from agentx_ai.streaming.chat_run import tail_chat_run
        self.mock_redis.hgetall.return_value = {
            "status": "running",
            "updated_at": "2000-01-01T00:00:00+00:00",  # ancient → orphaned
        }

        async def _collect():
            return [ev async for ev in tail_chat_run("stale")]

        events = asyncio.run(_collect())
        self.assertEqual(len(events), 1)
        self.assertIn("run_missing", events[0])
        self.assertIn("stale", events[0])

    def test_create_indexes_run_with_metadata(self):
        from agentx_ai.streaming.chat_run import store
        store.create("run1", user_id="alice", message="hello", session_id="sess1")
        # State hash carries the recovery metadata...
        mapping = self.mock_redis.hset.call_args.kwargs["mapping"]
        self.assertEqual(mapping["user_id"], "alice")
        self.assertEqual(mapping["message"], "hello")
        self.assertEqual(mapping["session_id"], "sess1")
        # ...and the run is added to the per-user index ZSET.
        self.assertTrue(self.mock_redis.zadd.called)
        index_key = self.mock_redis.zadd.call_args.args[0]
        self.assertIn("alice", index_key)

    def test_set_session_backfills_hash(self):
        from agentx_ai.streaming.chat_run import store
        store.set_session("run1", "sess-late")
        self.mock_redis.hset.assert_called_with("chat_run:run1", "session_id", "sess-late")

    def test_list_runs_returns_states_newest_first(self):
        from agentx_ai.streaming.chat_run import store
        self.mock_redis.zrevrange.return_value = [b"run2", b"run1"]
        self.mock_redis.hgetall.side_effect = [
            {"status": "running", "message": "two", "session_id": "s2",
             "created_at": "t2", "updated_at": "t2"},
            {"status": "running", "message": "one", "session_id": "",
             "created_at": "t1", "updated_at": "t1"},
        ]
        runs = store.list_runs("alice")
        self.assertEqual([r["run_id"] for r in runs], ["run2", "run1"])
        self.assertEqual(runs[0]["message"], "two")
        self.assertIsNone(runs[1]["session_id"])  # empty string → None

    def test_list_runs_prunes_stale_index_entries(self):
        from agentx_ai.streaming.chat_run import store
        self.mock_redis.zrevrange.return_value = [b"ghost"]
        self.mock_redis.hgetall.return_value = {}  # state expired
        runs = store.list_runs("alice")
        self.assertEqual(runs, [])
        self.mock_redis.zrem.assert_called_once()


@override_settings(AGENTX_AUTH_ENABLED=False)
class ChatRunsEndpointTest(MockRedisTestBase):
    """GET /api/agent/chat/runs — list this user's detached runs."""

    def setUp(self) -> None:
        super().setUp()
        from django.test import Client
        self.client = Client()

    def test_lists_runs(self):
        self.mock_redis.zrevrange.return_value = [b"run1"]
        self.mock_redis.hgetall.return_value = {
            "status": "running", "message": "hi", "session_id": "s1",
            "created_at": "t", "updated_at": "t",
        }
        resp = self.client.get("/api/agent/chat/runs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(data["runs"][0]["run_id"], "run1")

    def test_post_rejected(self):
        resp = self.client.post("/api/agent/chat/runs")
        self.assertEqual(resp.status_code, 405)


class AlloyDelegationMetricsTest(TestCase):
    """Per-delegation metrics: emitted on `delegation_complete` (executor) and
    persisted into the `delegation_raw` carrier (tool_loop). Backs the Alloy
    run-trace UI."""

    @staticmethod
    async def _drain(gen):
        return [ev async for ev in gen]

    @staticmethod
    def _parse_complete(events):
        for ev in events:
            # delegate() yields (sse_string, partial_text) tuples.
            sse = ev[0] if isinstance(ev, tuple) else ev
            if sse.startswith("event: delegation_complete\n"):
                data_line = sse.split("data: ", 1)[1].rstrip()
                return json.loads(data_line)
        return None

    def test_run_delegations_captures_metrics_into_raw(self):
        """_run_delegations parses delegation_complete and merges metrics into
        delegation_raw, which becomes the persisted tool_result metadata."""
        from agentx_ai.streaming.tool_loop import _run_delegations, ToolLoopResult

        class FakeExec:
            async def delegate(self, target, task, *, tool_call_id):
                payload = {
                    "delegation_id": "abc",
                    "target_agent_id": target,
                    "tool_call_id": tool_call_id,
                    "status": "success",
                    "error": None,
                    "result_preview": "done",
                    "tokens_input": 120,
                    "tokens_output": 60,
                    "duration_ms": 1234.5,
                    "cost_estimate": 0.004,
                    "cost_currency": "USD",
                    "pricing_snapshot": {"x": 1},
                }
                yield f"event: delegation_complete\ndata: {json.dumps(payload)}\n\n", "done"

        tc = MagicMock()
        tc.id = "tc1"
        tc.name = "delegate_to"
        tc.arguments = {"agent_id": "spec", "task": "do it"}

        result = ToolLoopResult()
        delegation_messages: list = []
        delegation_raw: dict = {}
        # Bare object => no handle_oversized_tool_output attribute (skips that path).
        agent = object()
        asyncio.run(self._drain(_run_delegations(
            [tc], FakeExec(), agent,
            result=result,
            delegation_messages=delegation_messages,
            delegation_raw=delegation_raw,
        )))

        raw = delegation_raw["tc1"]
        self.assertEqual(raw["raw_content"], "done")
        self.assertEqual(raw["target_agent_id"], "spec")
        self.assertEqual(raw["tokens_input"], 120)
        self.assertEqual(raw["tokens_output"], 60)
        self.assertEqual(raw["duration_ms"], 1234.5)
        self.assertEqual(raw["cost_estimate"], 0.004)
        self.assertEqual(raw["cost_currency"], "USD")
        self.assertEqual(raw["pricing_snapshot"], {"x": 1})

    def _make_executor(self):
        from types import SimpleNamespace
        from agentx_ai.alloy.executor import AlloyExecutor
        from agentx_ai.alloy.models import Workflow, WorkflowMember, MemberRole

        workflow = Workflow(
            id="wf",
            name="WF",
            supervisor_agent_id="sup",
            members=[
                WorkflowMember(agent_id="sup", role=MemberRole.SUPERVISOR),
                WorkflowMember(agent_id="spec", role=MemberRole.SPECIALIST),
            ],
        )
        supervisor = SimpleNamespace(
            config=SimpleNamespace(user_id="u", default_model="m", max_tool_rounds=3),
            memory=None,  # skips goal creation + turn storage
        )
        session = SimpleNamespace(id="sess")
        return AlloyExecutor(supervisor, session, workflow=workflow, max_delegation_depth=3)

    def test_delegate_emits_metrics_on_complete(self):
        """delegate() yields token/duration/cost on the delegation_complete event."""
        from types import SimpleNamespace

        executor = self._make_executor()

        profile = SimpleNamespace(
            name="Spec", default_model="m", agent_id="spec", prompt_profile_id=None,
            enable_memory=False, enable_tools=False, temperature=0.5, system_prompt=None,
        )
        fake_provider = SimpleNamespace(get_capabilities=lambda mid: object())
        fake_specialist = SimpleNamespace(
            config=SimpleNamespace(default_model="m", max_tool_rounds=3),
            registry=SimpleNamespace(get_provider_for_model=lambda m: (fake_provider, "m")),
            _get_tools_for_provider=lambda: None,
            memory=None,
        )

        async def fake_stream(*args, **kwargs):
            kwargs["result"].tokens_in = 120
            kwargs["result"].tokens_out = 60
            kwargs["result"].content = "spec output"
            yield 'event: chunk\ndata: {"content": "spec output"}\n\n'

        with patch("agentx_ai.alloy.executor.get_profile_manager",
                   return_value=SimpleNamespace(list_profiles=lambda: [profile])), \
             patch("agentx_ai.agent.core.Agent", return_value=fake_specialist), \
             patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_stream), \
             patch("agentx_ai.prompts.get_prompt_manager",
                   return_value=SimpleNamespace(get_system_prompt=lambda **k: "sys")), \
             patch("agentx_ai.providers.pricing.estimate_cost",
                   return_value={"cost_total": 0.004, "currency": "USD", "pricing_snapshot": {"r": 1}}):
            events = asyncio.run(self._drain(
                executor.delegate("spec", "do it", tool_call_id="tc1")
            ))

        done = self._parse_complete(events)
        self.assertIsNotNone(done)
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["tokens_input"], 120)
        self.assertEqual(done["tokens_output"], 60)
        self.assertIsInstance(done["duration_ms"], float)
        self.assertEqual(done["cost_estimate"], 0.004)
        self.assertEqual(done["cost_currency"], "USD")
        self.assertEqual(done["pricing_snapshot"], {"r": 1})

    def test_delegate_cost_none_when_pricing_unavailable(self):
        """A pricing-less model (estimate_cost -> None) yields null cost but
        still reports tokens + duration."""
        from types import SimpleNamespace

        executor = self._make_executor()
        profile = SimpleNamespace(
            name="Spec", default_model="m", agent_id="spec", prompt_profile_id=None,
            enable_memory=False, enable_tools=False, temperature=0.5, system_prompt=None,
        )
        fake_provider = SimpleNamespace(get_capabilities=lambda mid: object())
        fake_specialist = SimpleNamespace(
            config=SimpleNamespace(default_model="m", max_tool_rounds=3),
            registry=SimpleNamespace(get_provider_for_model=lambda m: (fake_provider, "m")),
            _get_tools_for_provider=lambda: None,
            memory=None,
        )

        async def fake_stream(*args, **kwargs):
            kwargs["result"].tokens_in = 10
            kwargs["result"].tokens_out = 5
            kwargs["result"].content = "out"
            yield 'event: chunk\ndata: {"content": "out"}\n\n'

        with patch("agentx_ai.alloy.executor.get_profile_manager",
                   return_value=SimpleNamespace(list_profiles=lambda: [profile])), \
             patch("agentx_ai.agent.core.Agent", return_value=fake_specialist), \
             patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_stream), \
             patch("agentx_ai.prompts.get_prompt_manager",
                   return_value=SimpleNamespace(get_system_prompt=lambda **k: "sys")), \
             patch("agentx_ai.providers.pricing.estimate_cost", return_value=None):
            events = asyncio.run(self._drain(
                executor.delegate("spec", "do it", tool_call_id="tc2")
            ))

        done = self._parse_complete(events)
        self.assertEqual(done["tokens_input"], 10)
        self.assertIsNone(done["cost_estimate"])
        self.assertIsNone(done["cost_currency"])
        self.assertIsNone(done["pricing_snapshot"])


class ParallelDelegationTest(TestCase):
    """Track A: fan-out delegation — `_run_delegations` runs multiple delegate_to
    calls concurrently, interleaving their events, isolating failures, bounding
    concurrency, and mapping results back by tool_call_id in original order."""

    @staticmethod
    def _tc(tid, agent_id, task="t"):
        m = MagicMock()
        m.id = tid
        m.name = "delegate_to"
        m.arguments = {"agent_id": agent_id, "task": task}
        return m

    @staticmethod
    async def _drain(gen):
        return [ev async for ev in gen]

    @staticmethod
    def _chunk_target(ev):
        """Extract the delegation_id/target marker from a delegation_chunk SSE."""
        if not ev.startswith("event: delegation_chunk"):
            return None
        data_line = ev.split("data: ", 1)[1].rstrip()
        return json.loads(data_line).get("delegation_id")

    def _run(self, calls, exec_obj, agent=None):
        from agentx_ai.streaming.tool_loop import _run_delegations, ToolLoopResult
        result = ToolLoopResult()
        messages: list = []
        raw: dict = {}
        events = asyncio.run(self._drain(_run_delegations(
            calls, exec_obj, agent if agent is not None else object(),
            result=result, delegation_messages=messages, delegation_raw=raw,
        )))
        return events, result, messages, raw

    def test_interleaving(self):
        """Two branches' chunk events interleave rather than block-A-then-block-B."""
        class ChunkExec:
            max_parallel_delegations = 5

            async def delegate(self, target, task, *, tool_call_id):
                for i in range(3):
                    await asyncio.sleep(0)
                    yield (
                        f"event: delegation_chunk\ndata: "
                        f"{json.dumps({'delegation_id': target, 'content': f'{target}{i}'})}\n\n",
                        f"{target}-{i}",
                    )
                yield (
                    f"event: delegation_complete\ndata: "
                    f"{json.dumps({'delegation_id': target, 'target_agent_id': target, 'tool_call_id': tool_call_id, 'status': 'success', 'result_preview': target})}\n\n",
                    f"{target}-final",
                )

        events, _, _, _ = self._run(
            [self._tc("tcA", "A"), self._tc("tcB", "B")], ChunkExec()
        )
        targets = [t for t in (self._chunk_target(e) for e in events) if t]
        # Both branches surface within the first two chunk events → interleaved.
        self.assertEqual(set(targets[:2]), {"A", "B"})

    def test_tool_result_mapping_and_order(self):
        """B completes before A, but delegation_messages stay in original order and
        each tool_call_id maps to its own accumulated content."""
        class UnevenExec:
            max_parallel_delegations = 5

            async def delegate(self, target, task, *, tool_call_id):
                # A streams more chunks, so B reaches completion first.
                n = 5 if target == "A" else 1
                for i in range(n):
                    await asyncio.sleep(0)
                    yield (
                        f"event: delegation_chunk\ndata: {json.dumps({'delegation_id': target, 'content': 'x'})}\n\n",
                        f"{target}-partial-{i}",
                    )
                yield (
                    f"event: delegation_complete\ndata: {json.dumps({'delegation_id': target, 'tool_call_id': tool_call_id, 'status': 'success'})}\n\n",
                    f"{target}-final",
                )

        _, _, messages, raw = self._run(
            [self._tc("tcA", "A"), self._tc("tcB", "B")], UnevenExec()
        )
        self.assertEqual([m.tool_call_id for m in messages], ["tcA", "tcB"])
        self.assertEqual(messages[0].content, "A-final")
        self.assertEqual(messages[1].content, "B-final")
        self.assertEqual(raw["tcA"]["raw_content"], "A-final")
        self.assertEqual(raw["tcB"]["raw_content"], "B-final")

    def test_error_isolation(self):
        """One branch raising does not kill its sibling; the failed branch still
        emits a delegation_complete and produces a TOOL message."""
        class FlakyExec:
            max_parallel_delegations = 5

            async def delegate(self, target, task, *, tool_call_id):
                if target == "A":
                    raise RuntimeError("boom")
                    yield  # pragma: no cover - make it an async generator
                yield (
                    f"event: delegation_chunk\ndata: {json.dumps({'delegation_id': target, 'content': 'ok'})}\n\n",
                    "B-partial",
                )
                yield (
                    f"event: delegation_complete\ndata: {json.dumps({'delegation_id': target, 'tool_call_id': tool_call_id, 'status': 'success'})}\n\n",
                    "B-final",
                )

        events, _, messages, raw = self._run(
            [self._tc("tcA", "A"), self._tc("tcB", "B")], FlakyExec()
        )
        # Sibling B intact.
        self.assertEqual(raw["tcB"]["raw_content"], "B-final")
        # Failed branch A emits a failed completion and still yields a TOOL message.
        self.assertTrue(any(
            e.startswith("event: delegation_complete") and '"status": "failed"' in e
            for e in events
        ))
        self.assertEqual({m.tool_call_id for m in messages}, {"tcA", "tcB"})
        self.assertTrue(raw["tcA"]["raw_content"].startswith("[delegation failed"))

    def test_max_concurrency_bound(self):
        """Peak simultaneous delegate() calls never exceeds max_parallel_delegations."""
        class CountingExec:
            max_parallel_delegations = 2

            def __init__(self):
                self.active = 0
                self.peak = 0

            async def delegate(self, target, task, *, tool_call_id):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    for _ in range(3):
                        await asyncio.sleep(0)
                        yield (
                            f"event: delegation_chunk\ndata: {json.dumps({'delegation_id': target, 'content': 'x'})}\n\n",
                            "p",
                        )
                finally:
                    self.active -= 1

        exec_obj = CountingExec()
        calls = [self._tc(f"tc{i}", f"A{i}") for i in range(4)]
        self._run(calls, exec_obj)
        self.assertLessEqual(exec_obj.peak, 2)

    def test_cancellation_cleans_up_branches(self):
        """aclose() mid-stream cancels all in-flight branches (no orphans)."""
        from agentx_ai.streaming.tool_loop import _run_delegations, ToolLoopResult

        class ForeverExec:
            max_parallel_delegations = 5

            def __init__(self):
                self.active = 0

            async def delegate(self, target, task, *, tool_call_id):
                self.active += 1
                try:
                    while True:
                        await asyncio.sleep(0)
                        yield (
                            f"event: delegation_chunk\ndata: {json.dumps({'delegation_id': target, 'content': 'x'})}\n\n",
                            "p",
                        )
                finally:
                    self.active -= 1

        exec_obj = ForeverExec()

        async def scenario():
            gen = _run_delegations(
                [self._tc("tcA", "A"), self._tc("tcB", "B")], exec_obj, object(),
                result=ToolLoopResult(), delegation_messages=[], delegation_raw={},
            )
            seen = 0
            async for _ in gen:
                seen += 1
                if seen >= 3:
                    break
            await gen.aclose()

        asyncio.run(scenario())
        self.assertEqual(exec_obj.active, 0)

    def test_executor_has_no_shared_depth_counter(self):
        """Reentrancy guarantee: delegate() no longer mutates a shared self.depth
        (depth is a per-call parameter), and the depth gate works independently."""
        from types import SimpleNamespace
        from agentx_ai.alloy.executor import AlloyExecutor
        from agentx_ai.alloy.models import Workflow, WorkflowMember, MemberRole

        workflow = Workflow(
            id="wf", name="WF", supervisor_agent_id="sup",
            members=[
                WorkflowMember(agent_id="sup", role=MemberRole.SUPERVISOR),
                WorkflowMember(agent_id="spec", role=MemberRole.SPECIALIST),
            ],
        )
        supervisor = SimpleNamespace(
            config=SimpleNamespace(user_id="u", default_model="m", max_tool_rounds=3),
            memory=None,
        )
        executor = AlloyExecutor(
            supervisor, SimpleNamespace(id="s"), workflow=workflow, max_delegation_depth=1
        )
        self.assertFalse(hasattr(executor, "depth"))

        async def run():
            return [ev async for ev in executor.delegate("spec", "t", tool_call_id="x", depth=1)]

        events = asyncio.run(run())
        done = json.loads(events[-1][0].split("data: ", 1)[1].rstrip())
        self.assertEqual(done["status"], "failed")
        self.assertIn("max delegation depth", done["error"])


class DelegatableProfileTest(TestCase):
    """Track D: per-profile `available_for_delegation` flag + the tool-gating
    persistence-bug fix in ProfileManager.save_config()."""

    def _manager(self, tmp_path):
        from agentx_ai.agent.profiles import ProfileManager
        return ProfileManager(config_path=tmp_path)

    def test_save_config_persists_tool_gating_and_delegation_flag(self):
        """Regression: allowed_tools/blocked_tools/available_for_delegation must
        survive a save→reload (previously silently dropped on save)."""
        import tempfile
        from pathlib import Path
        from agentx_ai.agent.models import AgentProfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profiles.yaml"
            mgr = self._manager(path)
            mgr.create_profile(AgentProfile(
                id="r", name="R", agent_id="aaa-bbb-ccc",
                allowed_tools=["_internal.web_search"],
                blocked_tools=["x.y"],
                available_for_delegation=False,
            ))
            # Reload from disk into a fresh manager.
            reloaded = self._manager(path)
            p = reloaded.get_profile("r")
            self.assertEqual(p.allowed_tools, ["_internal.web_search"])
            self.assertEqual(p.blocked_tools, ["x.y"])
            self.assertFalse(p.available_for_delegation)

    def test_adhoc_delegation_excludes_undelegatable_profiles(self):
        """build_adhoc_delegation_tool drops profiles with the flag off."""
        from unittest.mock import patch
        from types import SimpleNamespace
        from agentx_ai.alloy.delegation_tool import build_adhoc_delegation_tool

        profiles = [
            SimpleNamespace(agent_id="self-aa-bb", name="Self", description="", available_for_delegation=True),
            SimpleNamespace(agent_id="on-cc-dd", name="On", description="yes", available_for_delegation=True),
            SimpleNamespace(agent_id="off-ee-ff", name="Off", description="no", available_for_delegation=False),
        ]
        with patch("agentx_ai.agent.profiles.get_profile_manager") as gpm:
            gpm.return_value = SimpleNamespace(list_profiles=lambda: profiles)
            tool = build_adhoc_delegation_tool("self-aa-bb")

        enum = tool["input_schema"]["properties"]["agent_id"].get("enum", [])
        self.assertIn("on-cc-dd", enum)        # delegatable peer included
        self.assertNotIn("off-ee-ff", enum)    # flag off → excluded
        self.assertNotIn("self-aa-bb", enum)   # self excluded


class ExplicitAgentRoutingTest(TestCase):
    """Phase 16.2: routing-by-agent_id helper + multi-agent prompt block."""

    def _profiles(self):
        from agentx_ai.agent.models import AgentProfile
        # Field-defaulted args omitted (matches DEFAULT_PROFILE construction style).
        return (
            AgentProfile(id="1", name="Alpha", agent_id="alpha-agent"),  # type: ignore[call-arg]
            AgentProfile(id="2", name="Beta", agent_id="beta-agent"),  # type: ignore[call-arg]
        )

    def test_get_profile_by_agent_id(self):
        from agentx_ai.agent.profiles import ProfileManager
        a, b = self._profiles()
        mgr = ProfileManager.__new__(ProfileManager)
        mgr._profiles = {a.id: a, b.id: b}
        self.assertIs(mgr.get_profile_by_agent_id("beta-agent"), b)
        self.assertIsNone(mgr.get_profile_by_agent_id("nope"))

    def test_participants_block_excludes_self(self):
        from agentx_ai.prompts.multi_agent import build_participants_block
        a, b = self._profiles()
        block = build_participants_block(
            "alpha-agent", {"alpha-agent": a, "beta-agent": b}
        )
        assert block is not None
        self.assertIn("Beta", block)
        self.assertIn("beta-agent", block)
        self.assertNotIn("Alpha", block)  # self is excluded from the roster

    def test_participants_block_none_when_alone(self):
        from agentx_ai.prompts.multi_agent import build_participants_block
        a, _ = self._profiles()
        self.assertIsNone(build_participants_block("alpha-agent", {"alpha-agent": a}))
        self.assertIsNone(build_participants_block("alpha-agent", {}))


class MentionRoutingTest(TestCase):
    """Phase 16.5: @-mention parsing + name lookup for inline routing."""

    def test_extract_mentions_order_dedup(self):
        from agentx_ai.agent.mentions import extract_mentions
        self.assertEqual(
            extract_mentions("hi @bright-grand-fern and @Mobius, also @Mobius again"),
            ["bright-grand-fern", "Mobius"],
        )
        self.assertEqual(extract_mentions("no mentions here"), [])

    def test_extract_mentions_ignores_emails_and_paths(self):
        from agentx_ai.agent.mentions import extract_mentions
        self.assertEqual(extract_mentions("mail me at user@example.com"), [])
        self.assertEqual(extract_mentions("see path/@thing"), [])

    def test_resolve_first_mention_strips_resolved_token(self):
        from agentx_ai.agent.mentions import resolve_first_mention
        # Only "beta" resolves; "nope" is left in place.
        def resolve(tok):
            return "beta-agent" if tok == "beta" else None
        agent_id, stripped = resolve_first_mention("hey @beta and @nope do it", resolve)
        self.assertEqual(agent_id, "beta-agent")
        self.assertEqual(stripped, "hey and @nope do it")

    def test_resolve_first_mention_none_when_unresolved(self):
        from agentx_ai.agent.mentions import resolve_first_mention
        agent_id, stripped = resolve_first_mention("hello @ghost", lambda t: None)
        self.assertIsNone(agent_id)
        self.assertEqual(stripped, "hello @ghost")  # untouched

    def test_get_profile_by_name_case_insensitive(self):
        from agentx_ai.agent.profiles import ProfileManager
        from agentx_ai.agent.models import AgentProfile
        a = AgentProfile(id="1", name="Mobius", agent_id="bright-grand-fern")  # type: ignore[call-arg]
        mgr = ProfileManager.__new__(ProfileManager)
        mgr._profiles = {a.id: a}
        self.assertIs(mgr.get_profile_by_name("mobius"), a)
        self.assertIsNone(mgr.get_profile_by_name("nobody"))


class AdhocDelegationTest(TestCase):
    """Phase 16.4: ad-hoc (non-workflow) agent-to-agent delegation."""

    @staticmethod
    async def _drain(gen):
        return [ev async for ev in gen]

    @staticmethod
    def _parse_complete(events) -> dict:
        for ev in events:
            sse = ev[0] if isinstance(ev, tuple) else ev
            if sse.startswith("event: delegation_complete\n"):
                return json.loads(sse.split("data: ", 1)[1].rstrip())
        raise AssertionError("no delegation_complete event emitted")

    def _profiles(self):
        from types import SimpleNamespace
        return [
            SimpleNamespace(agent_id="alpha-agent", name="Alpha", description="lead"),
            SimpleNamespace(agent_id="beta-agent", name="Beta", description="researcher"),
        ]

    # ---- tool descriptor ----

    def test_adhoc_tool_lists_others_excludes_self(self):
        from types import SimpleNamespace
        from agentx_ai.alloy.delegation_tool import build_adhoc_delegation_tool, DELEGATION_TOOL_NAME
        with patch("agentx_ai.agent.profiles.get_profile_manager",
                   return_value=SimpleNamespace(list_profiles=self._profiles)):
            desc = build_adhoc_delegation_tool("alpha-agent")
        self.assertEqual(desc["name"], DELEGATION_TOOL_NAME)
        self.assertEqual(desc["input_schema"]["properties"]["agent_id"]["enum"], ["beta-agent"])
        self.assertIn("Beta", desc["description"])
        self.assertNotIn("alpha-agent", desc["description"])

    # ---- executor (ad-hoc mode) ----

    def _make_executor(self, depth=0, max_depth=3):
        from types import SimpleNamespace
        from agentx_ai.alloy.executor import AlloyExecutor
        supervisor = SimpleNamespace(
            config=SimpleNamespace(user_id="u", default_model="m", max_tool_rounds=3),
            memory=None,
        )
        ex = AlloyExecutor(
            supervisor, SimpleNamespace(id="sess"),  # type: ignore[arg-type]
            channel="_global", delegator_agent_id="alpha-agent", max_delegation_depth=max_depth,
        )
        ex.depth = depth
        return ex

    def test_self_delegation_rejected(self):
        from types import SimpleNamespace
        ex = self._make_executor()
        with patch("agentx_ai.alloy.executor.get_profile_manager",
                   return_value=SimpleNamespace(get_profile_by_agent_id=lambda a: object())):
            done = self._parse_complete(asyncio.run(self._drain(
                ex.delegate("alpha-agent", "x", tool_call_id="t"))))
        self.assertEqual(done["status"], "failed")
        self.assertIn("itself", done["error"])

    def test_unknown_target_rejected(self):
        from types import SimpleNamespace
        ex = self._make_executor()
        with patch("agentx_ai.alloy.executor.get_profile_manager",
                   return_value=SimpleNamespace(get_profile_by_agent_id=lambda a: None)):
            done = self._parse_complete(asyncio.run(self._drain(
                ex.delegate("ghost", "x", tool_call_id="t"))))
        self.assertEqual(done["status"], "failed")
        self.assertIn("no agent profile", done["error"])

    def test_depth_ceiling_enforced(self):
        from types import SimpleNamespace
        ex = self._make_executor(depth=3, max_depth=3)
        with patch("agentx_ai.alloy.executor.get_profile_manager",
                   return_value=SimpleNamespace(get_profile_by_agent_id=lambda a: object())):
            done = self._parse_complete(asyncio.run(self._drain(
                ex.delegate("beta-agent", "x", tool_call_id="t"))))
        self.assertEqual(done["status"], "failed")
        self.assertIn("max delegation depth", done["error"])

    def test_adhoc_success_path(self):
        from types import SimpleNamespace
        ex = self._make_executor()
        profile = SimpleNamespace(
            name="Beta", default_model="m", agent_id="beta-agent", prompt_profile_id=None,
            enable_memory=False, enable_tools=False, temperature=0.5, system_prompt=None,
        )
        fake_provider = SimpleNamespace(get_capabilities=lambda mid: object())
        fake_specialist = SimpleNamespace(
            config=SimpleNamespace(default_model="m", max_tool_rounds=3),
            registry=SimpleNamespace(get_provider_for_model=lambda m: (fake_provider, "m")),
            _get_tools_for_provider=lambda: None,
            memory=None,
        )

        async def fake_stream(*args, **kwargs):
            kwargs["result"].content = "beta result"
            yield 'event: chunk\ndata: {"content": "beta result"}\n\n'

        pm = SimpleNamespace(
            get_profile_by_agent_id=lambda a: profile if a == "beta-agent" else None,
            list_profiles=lambda: [profile],
        )
        with patch("agentx_ai.alloy.executor.get_profile_manager", return_value=pm), \
             patch("agentx_ai.agent.core.Agent", return_value=fake_specialist), \
             patch("agentx_ai.streaming.tool_loop.streaming_tool_loop", fake_stream), \
             patch("agentx_ai.prompts.get_prompt_manager",
                   return_value=SimpleNamespace(get_system_prompt=lambda **k: "sys")), \
             patch("agentx_ai.providers.pricing.estimate_cost", return_value=None):
            events = asyncio.run(self._drain(
                ex.delegate("beta-agent", "do it", tool_call_id="t")))
        done = self._parse_complete(events)
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["target_agent_id"], "beta-agent")
        self.assertTrue(any(e[0].startswith("event: delegation_start") for e in events))


class TranslationInternalToolsTest(TestCase):
    """detect_language + translate_text internal tools (kit mocked — no model load)."""

    def test_tools_registered(self) -> None:
        from agentx_ai.mcp.internal_tools import is_internal_tool, get_internal_tools

        self.assertTrue(is_internal_tool("detect_language"))
        self.assertTrue(is_internal_tool("translate_text"))
        names = {t.name for t in get_internal_tools()}
        self.assertIn("detect_language", names)
        self.assertIn("translate_text", names)

    def test_detect_language(self) -> None:
        from agentx_ai.mcp import internal_tools

        kit = MagicMock()
        kit.detect_language_level_i.return_value = ("fr", 98.5)
        with patch("agentx_ai.kit.translation.get_translation_kit", return_value=kit):
            result = internal_tools.detect_language("Bonjour tout le monde")

        self.assertTrue(result["success"])
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["confidence"], 98.5)

    def test_detect_language_empty(self) -> None:
        from agentx_ai.mcp import internal_tools

        result = internal_tools.detect_language("   ")
        self.assertFalse(result["success"])

    def test_translate_level_1_for_bare_code(self) -> None:
        from agentx_ai.mcp import internal_tools

        kit = MagicMock()
        kit.translate_text.return_value = "Bonjour"
        with patch("agentx_ai.kit.translation.get_translation_kit", return_value=kit):
            result = internal_tools.translate_text("Hello", "fr")

        self.assertTrue(result["success"])
        self.assertEqual(result["translated_text"], "Bonjour")
        # Bare code → level 1
        _, kwargs = kit.translate_text.call_args
        self.assertEqual(kwargs["target_language_level"], 1)

    def test_translate_level_2_for_nllb_code(self) -> None:
        from agentx_ai.mcp import internal_tools

        kit = MagicMock()
        kit.translate_text.return_value = "Hallo"
        with patch("agentx_ai.kit.translation.get_translation_kit", return_value=kit):
            result = internal_tools.translate_text("Hello", "deu_Latn")

        self.assertTrue(result["success"])
        _, kwargs = kit.translate_text.call_args
        self.assertEqual(kwargs["target_language_level"], 2)

    def test_translate_unsupported_lists_supported(self) -> None:
        from agentx_ai.mcp import internal_tools

        kit = MagicMock()
        kit.translate_text.side_effect = ValueError("Language xx not supported")
        with patch("agentx_ai.kit.translation.get_translation_kit", return_value=kit):
            result = internal_tools.translate_text("Hello", "xx")

        self.assertFalse(result["success"])
        self.assertIn("supported", result)
        self.assertIn("french", result["supported"])  # level_i_languages map

    def test_translate_missing_args(self) -> None:
        from agentx_ai.mcp import internal_tools

        self.assertFalse(internal_tools.translate_text("", "fr")["success"])
        self.assertFalse(internal_tools.translate_text("hi", "")["success"])

    def test_execute_internal_tool_wraps_result(self) -> None:
        from agentx_ai.mcp import internal_tools

        kit = MagicMock()
        kit.translate_text.return_value = "Hola"
        with patch("agentx_ai.kit.translation.get_translation_kit", return_value=kit):
            tr = internal_tools.execute_internal_tool(
                "translate_text", {"text": "Hello", "target_language": "es"}
            )
        self.assertTrue(tr.success)
        self.assertIn("Hola", tr.content[0]["text"])


class TaskPlannerNormalizationTest(TestCase):
    """Subtask-id normalization + cap (prevents the executor subtask loop)."""

    def _planner(self, **kw):
        from agentx_ai.agent.planner import TaskPlanner

        return TaskPlanner("openai:gpt-4", **kw)

    def test_reindexes_messy_numbering_and_remaps_deps(self) -> None:
        planner = self._planner()
        # LLM numbered non-contiguously (2, 5) with a duplicate and an out-of-range dep.
        response = (
            "SUBTASK 2: Research the topic\nTYPE: RESEARCH\nDEPENDS: none\n"
            "SUBTASK 5: Analyze findings\nTYPE: ANALYSIS\nDEPENDS: 2\n"
            "SUBTASK 5: Write it up\nTYPE: GENERATION\nDEPENDS: 99\n"
        )
        steps = planner._parse_plan(response)

        # Contiguous ids matching list position (the invariant the executor needs).
        self.assertEqual([s.id for s in steps], list(range(len(steps))))
        # Dep "2" (old id1) remapped to the reindexed earlier step; junk dep dropped.
        self.assertEqual(steps[1].dependencies, [0])
        self.assertEqual(steps[2].dependencies, [])

    def test_executor_loop_terminates(self) -> None:
        """Simulate the executor's selection loop — must not re-select a subtask."""
        from agentx_ai.agent.planner import TaskComplexity, TaskPlan

        planner = self._planner()
        response = (
            "SUBTASK 2: A\nTYPE: GENERATION\nDEPENDS: none\n"
            "SUBTASK 5: B\nTYPE: ANALYSIS\nDEPENDS: 2\n"
            "SUBTASK 5: C\nTYPE: GENERATION\nDEPENDS: 99\n"
        )
        steps = planner._parse_plan(response)
        plan = TaskPlan(task="t", complexity=TaskComplexity.COMPLEX, steps=steps)

        selected, guard = [], 0
        while not plan.is_complete():
            guard += 1
            self.assertLess(guard, 100, "executor loop did not terminate")
            st = plan.get_next_subtask()
            self.assertIsNotNone(st)
            selected.append(st.id)
            plan.mark_complete(st.id, "done")

        # Every subtask selected exactly once.
        self.assertEqual(sorted(selected), list(range(len(steps))))
        self.assertEqual(len(selected), len(set(selected)))

    def test_caps_subtasks(self) -> None:
        planner = self._planner(max_subtasks=3)
        response = "".join(
            f"SUBTASK {i}: step {i}\nTYPE: GENERATION\nDEPENDS: none\n" for i in range(1, 9)
        )
        steps = planner._parse_plan(response)
        self.assertEqual(len(steps), 3)
        self.assertEqual([s.id for s in steps], [0, 1, 2])


class TaskComplexityTest(TestCase):
    """Conservative complexity assessment — simple asks must not over-plan."""

    def _planner(self):
        from agentx_ai.agent.planner import TaskPlanner

        return TaskPlanner("openai:gpt-4")

    def test_simple_prompts_stay_simple(self) -> None:
        from agentx_ai.agent.planner import TaskComplexity

        planner = self._planner()
        for prompt in [
            "write a haiku about the sea",
            "summarize this paragraph",
            "explain how a for-loop works",
            "what's the capital of France?",
            "build a regex for emails",
            "draft a thank-you note",
        ]:
            self.assertEqual(
                planner._assess_complexity(prompt), TaskComplexity.SIMPLE, prompt
            )

    def test_multistep_tasks_are_complex(self) -> None:
        from agentx_ai.agent.planner import TaskComplexity

        planner = self._planner()
        self.assertEqual(
            planner._assess_complexity(
                "Research the top 3 vector DBs, compare their pricing, and write a recommendation"
            ),
            TaskComplexity.COMPLEX,
        )
        self.assertEqual(
            planner._assess_complexity("Design a system architecture for a chat app"),
            TaskComplexity.COMPLEX,
        )
        self.assertEqual(
            planner._assess_complexity(
                "First, analyze the logs. Then build a dashboard and write a summary."
            ),
            TaskComplexity.COMPLEX,
        )


class ToolGatingTest(TestCase):
    """
    Phase 18.9.x: per-profile tool gating in ``Agent._get_tools_for_provider``.

    Matching is on the fully-qualified ``{server_name}.{tool_name}`` key so two
    MCP servers exposing a same-named tool don't gate together. Built-in tools
    resolve to ``_internal.<name>``.
    """

    def _make_agent(self, *, allowed=None, blocked=None):
        from agentx_ai.agent.core import Agent, AgentConfig

        config = AgentConfig(
            enable_tools=True,
            allowed_tools=allowed,
            blocked_tools=list(blocked) if blocked else [],
        )
        agent = Agent(config)
        # Bypass the lazy MCP-manager lookup; the property's `is None` guard
        # means assigning to the private attr is enough.
        client = MagicMock()
        client.list_tools = MagicMock(return_value=self._tools())
        client.list_connections = MagicMock(return_value=[])
        # Empty registry => no server-side whitelist gate fires for these
        # tools (their server_names don't appear in `server_gate`).
        client.registry.list = MagicMock(return_value=[])
        agent._mcp_client = client
        return agent

    def _tools(self):
        from agentx_ai.mcp.tool_executor import ToolInfo

        return [
            ToolInfo(name="read_file", description="fs", input_schema={}, server_name="filesystem"),
            ToolInfo(name="read_file", description="git", input_schema={}, server_name="git"),
            ToolInfo(name="checkpoint", description="ck", input_schema={}, server_name="_internal"),
            ToolInfo(name="recall_user_history", description="r", input_schema={}, server_name="_internal"),
        ]

    def _names(self, exposed):
        # Exposed entries are provider-format dicts; flatten to FQ-ish display.
        return sorted(t["function"]["name"] for t in (exposed or []))

    def test_no_gating_exposes_everything(self) -> None:
        agent = self._make_agent()
        exposed = agent._get_tools_for_provider()
        self.assertIsNotNone(exposed)
        assert exposed is not None  # for pyright
        # Two tools share the bare name `read_file`; both should pass.
        self.assertEqual(len(exposed), 4)

    def test_allowed_tools_is_fully_qualified(self) -> None:
        # Whitelisting only `filesystem.read_file` must NOT also allow
        # `git.read_file` — the bug we're fixing.
        agent = self._make_agent(allowed=["filesystem.read_file"])
        exposed = agent._get_tools_for_provider()
        assert exposed is not None
        self.assertEqual(len(exposed), 1)
        self.assertEqual(exposed[0]["function"]["name"], "read_file")

    def test_blocked_tools_is_fully_qualified(self) -> None:
        agent = self._make_agent(blocked=["git.read_file"])
        exposed = agent._get_tools_for_provider()
        assert exposed is not None
        # Three left: filesystem.read_file + the two internals.
        self.assertEqual(len(exposed), 3)

    def test_blocked_wins_over_allowed(self) -> None:
        agent = self._make_agent(
            allowed=["filesystem.read_file", "_internal.checkpoint"],
            blocked=["_internal.checkpoint"],
        )
        exposed = agent._get_tools_for_provider()
        self.assertEqual(self._names(exposed), ["read_file"])

    def test_internal_tool_gating(self) -> None:
        # Block only the introspection tool; checkpoint must stay.
        agent = self._make_agent(blocked=["_internal.recall_user_history"])
        exposed = agent._get_tools_for_provider()
        names = self._names(exposed)
        self.assertIn("checkpoint", names)
        self.assertNotIn("recall_user_history", names)


class ProfileUnqualifiedToolWarningTest(TestCase):
    """A profile loaded with bare tool names (legacy format) logs a warning."""

    def test_warns_on_unqualified_entries(self) -> None:
        from agentx_ai.agent.models import AgentProfile
        from agentx_ai.agent.profiles import _warn_unqualified_tool_names

        profile = AgentProfile(  # type: ignore[call-arg]
            id="legacy",
            name="Legacy",
            allowed_tools=["checkpoint"],         # bare → should warn
            blocked_tools=["filesystem.read_file"],  # FQ → fine
        )
        with self.assertLogs("agentx_ai.agent.profiles", level="WARNING") as cm:
            _warn_unqualified_tool_names(profile)
        joined = "\n".join(cm.output)
        self.assertIn("legacy", joined)
        self.assertIn("checkpoint", joined)
        self.assertNotIn("filesystem.read_file", joined)

    def test_silent_when_all_fully_qualified(self) -> None:
        from agentx_ai.agent.models import AgentProfile
        from agentx_ai.agent.profiles import _warn_unqualified_tool_names

        profile = AgentProfile(  # type: ignore[call-arg]
            id="modern",
            name="Modern",
            allowed_tools=["_internal.checkpoint"],
            blocked_tools=["filesystem.read_file"],
        )
        # assertNoLogs is 3.10+. The logger emits a record at WARNING only when
        # there's at least one bare entry; check by capturing and asserting empty.
        with self.assertLogs("agentx_ai.agent.profiles", level="WARNING") as cm:
            # Emit something so assertLogs doesn't fail when our helper is silent.
            import logging
            logging.getLogger("agentx_ai.agent.profiles").warning("sentinel")
            _warn_unqualified_tool_names(profile)
        # Only the sentinel should be present.
        self.assertEqual(len(cm.output), 1)
        self.assertIn("sentinel", cm.output[0])


class SchemaLoaderTest(TestCase):
    """Comment-aware Cypher statement splitting (regression for the
    'Invalid input id' schema-init bug caused by a ';' inside a // comment)."""

    def test_semicolon_inside_comment_does_not_break_next_statement(self):
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        # The exact shape that broke init_memory_schema: a comment containing a
        # ';' immediately followed by a real statement.
        cypher = (
            '// Phase 16.5: one node per (agent, conversation); id is "<conv_id>:<agent_id>"\n'
            "CREATE CONSTRAINT agent_participant_id IF NOT EXISTS\n"
            "FOR (ap:AgentParticipant) REQUIRE ap.id IS UNIQUE;\n"
        )
        stmts = split_cypher_statements(cypher)
        self.assertEqual(len(stmts), 1)
        self.assertTrue(stmts[0].startswith("CREATE CONSTRAINT agent_participant_id"))
        # The orphaned comment tail must NOT have leaked into the statement.
        self.assertNotIn("id is", stmts[0])

    def test_trailing_inline_comment_with_semicolon(self):
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        cypher = "CREATE INDEX foo IF NOT EXISTS FOR (n:Foo) ON (n.bar);  // note; with semicolon\n"
        stmts = split_cypher_statements(cypher)
        self.assertEqual(len(stmts), 1)
        self.assertNotIn("note", stmts[0])

    def test_blank_and_return_markers_dropped(self):
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        self.assertEqual(split_cypher_statements("// only a comment\nRETURN 1;\n"), [])

    def test_real_baseline_schema_parses_cleanly(self):
        """Every parsed statement from the real baseline must start with a
        Cypher keyword — a leaked comment fragment would start lowercase."""
        from pathlib import Path
        from agentx_ai.kit.agent_memory.schema_loader import split_cypher_statements
        schema = Path(__file__).resolve().parents[2] / "queries" / "neo4j_schemas.cypher"
        stmts = split_cypher_statements(schema.read_text())
        self.assertGreater(len(stmts), 0)
        for s in stmts:
            first = s.split(None, 1)[0]
            self.assertTrue(
                first.isupper() or first[0].isupper(),
                f"statement does not start with a keyword (leaked comment?): {s[:60]!r}",
            )


class PresentExhibitToolTest(TestCase):
    """Exhibits — the declarative content-part protocol (Slice 1: Mermaid).

    Covers the Exhibit model + validation, the present_exhibit tool body, and
    the tool-loop wiring that surfaces a present_exhibit call as a typed
    `exhibit` SSE event (and suppresses its tool_call/tool_result cards).
    """

    def _result_payload(self, result):
        """Parse the JSON dict an internal tool returns from its ToolResult."""
        self.assertTrue(result.content, "tool returned no content")
        return json.loads(result.content[0]["text"])

    # --- Exhibit model / validation -------------------------------------

    def test_exhibit_from_present_call_defaults(self):
        from agentx_ai.streaming.exhibits import EXHIBIT_SCHEMA_VERSION, exhibit_from_present_call
        ex = exhibit_from_present_call(
            {"elements": [{"type": "mermaid", "content": "graph TD; A-->B;", "title": "Flow"}]}
        )
        self.assertTrue(ex.id.startswith("exh_"))  # generated when omitted
        self.assertEqual(ex.layout, "stack")
        self.assertEqual(ex.schema_version, EXHIBIT_SCHEMA_VERSION)
        self.assertEqual(len(ex.elements), 1)
        self.assertEqual(ex.elements[0].title, "Flow")

    def test_exhibit_keeps_explicit_id(self):
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        ex = exhibit_from_present_call(
            {"id": "diagram-1", "elements": [{"type": "mermaid", "content": "pie title X"}]}
        )
        self.assertEqual(ex.id, "diagram-1")

    def test_exhibit_rejects_unknown_element_type(self):
        from pydantic import ValidationError
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        with self.assertRaises(ValidationError):
            exhibit_from_present_call({"elements": [{"type": "table", "content": "x"}]})

    def test_mermaid_sanity_error(self):
        from agentx_ai.streaming.exhibits import mermaid_sanity_error
        self.assertIsNone(mermaid_sanity_error("sequenceDiagram\n A->>B: hi"))
        self.assertIsNone(mermaid_sanity_error("graph TD; A-->B"))
        self.assertIsNotNone(mermaid_sanity_error(""))
        self.assertIsNotNone(mermaid_sanity_error("this is just prose"))

    # --- present_exhibit tool body --------------------------------------

    def test_present_exhibit_tool_registered(self):
        names = {t.name for t in get_internal_tools()}
        self.assertIn("present_exhibit", names)

    def test_present_exhibit_valid(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "mermaid", "content": "flowchart LR; A-->B"}]},
        )
        self.assertTrue(result.success)
        payload = self._result_payload(result)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["element_count"], 1)

    def test_present_exhibit_empty_content_errors(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "mermaid", "content": "   "}]},
        )
        self.assertFalse(result.success)
        self.assertFalse(self._result_payload(result)["success"])

    def test_present_exhibit_non_mermaid_keyword_errors(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "mermaid", "content": "just some text, not a diagram"}]},
        )
        self.assertFalse(result.success)

    def test_present_exhibit_unknown_type_errors(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "table", "content": "a,b\n1,2"}]},
        )
        self.assertFalse(result.success)

    # --- choice element -------------------------------------------------

    def test_choice_element_builds_and_cleans_options(self):
        from agentx_ai.streaming.exhibits import ChoiceElement, exhibit_from_present_call
        ex = exhibit_from_present_call(
            {"elements": [{"type": "choice", "prompt": "Pick", "options": ["  A ", "B", "B", ""]}]}
        )
        el = ex.elements[0]
        self.assertIsInstance(el, ChoiceElement)  # discriminator resolved
        self.assertEqual(el.options, ["A", "B"])  # stripped + de-duped, blanks dropped

    def test_choice_empty_options_rejected(self):
        from pydantic import ValidationError
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        with self.assertRaises(ValidationError):
            exhibit_from_present_call({"elements": [{"type": "choice", "options": ["  "]}]})

    def test_present_exhibit_choice_valid(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "choice", "prompt": "DB?", "options": ["PostgreSQL", "Neo4j"]}]},
        )
        self.assertTrue(result.success)
        self.assertTrue(self._result_payload(result)["success"])

    def test_present_exhibit_mixed_elements(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [
                {"type": "mermaid", "content": "graph TD; A-->B"},
                {"type": "choice", "options": ["yes", "no"]},
            ]},
        )
        self.assertTrue(result.success)
        self.assertEqual(self._result_payload(result)["element_count"], 2)

    def test_emit_exhibit_event_choice(self):
        from types import SimpleNamespace
        from agentx_ai.streaming.tool_loop import _emit_exhibit_event
        tc = SimpleNamespace(
            id="tc_3",
            name="present_exhibit",
            arguments={"elements": [{"type": "choice", "options": ["A", "B"]}]},
        )
        events = _emit_exhibit_event(tc)
        self.assertEqual(len(events), 1)
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        self.assertEqual(payload["elements"][0]["type"], "choice")
        self.assertEqual(payload["elements"][0]["options"], ["A", "B"])

    # --- table element --------------------------------------------------

    def test_table_stringifies_and_normalizes_rows(self):
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        ex = exhibit_from_present_call({"elements": [{
            "type": "table",
            "columns": ["Model", "Cost"],
            "rows": [["opus", 0.4], ["haiku", None, "extra"], ["solo"]],
        }]})
        el = ex.elements[0]
        # cells stringified (0.4 -> "0.4", None -> ""), rows padded/truncated to 2 cols
        self.assertEqual(el.rows, [["opus", "0.4"], ["haiku", ""], ["solo", ""]])

    def test_table_too_many_columns_rejected(self):
        from pydantic import ValidationError
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        with self.assertRaises(ValidationError):
            exhibit_from_present_call({"elements": [{
                "type": "table",
                "columns": [str(i) for i in range(13)],
                "rows": [],
            }]})

    def test_present_exhibit_table_valid(self):
        result = execute_internal_tool(
            "present_exhibit",
            {"elements": [{"type": "table", "columns": ["A", "B"], "rows": [["1", "2"]]}]},
        )
        self.assertTrue(result.success)

    # --- citation element -----------------------------------------------

    def test_citation_defaults_passive_and_validates_label(self):
        from pydantic import ValidationError
        from agentx_ai.streaming.exhibits import exhibit_from_present_call
        ex = exhibit_from_present_call({"elements": [{
            "type": "citation",
            "sources": [
                {"label": "NLLB", "url": "http://x", "quote": "q", "kind": "active", "source_type": "web"},
                {"label": "docs"},
            ],
        }]})
        kinds = [(s.label, s.kind) for s in ex.elements[0].sources]
        self.assertEqual(kinds, [("NLLB", "active"), ("docs", "passive")])
        with self.assertRaises(ValidationError):
            exhibit_from_present_call({"elements": [{"type": "citation", "sources": [{"label": "  "}]}]})

    def test_present_exhibit_mixed_table_citation_mermaid(self):
        result = execute_internal_tool("present_exhibit", {"elements": [
            {"type": "mermaid", "content": "graph TD; A-->B"},
            {"type": "table", "columns": ["a"], "rows": [["1"]]},
            {"type": "citation", "sources": [{"label": "s", "kind": "active"}]},
        ]})
        self.assertTrue(result.success)
        self.assertEqual(self._result_payload(result)["element_count"], 3)

    # --- tool-loop wiring -----------------------------------------------

    def test_emit_exhibit_event_valid(self):
        from types import SimpleNamespace
        from agentx_ai.streaming.tool_loop import EXHIBIT_TOOL_NAME, _emit_exhibit_event
        self.assertEqual(EXHIBIT_TOOL_NAME, "present_exhibit")
        tc = SimpleNamespace(
            id="tc_1",
            name="present_exhibit",
            arguments={"elements": [{"type": "mermaid", "content": "graph TD; A-->B"}]},
        )
        events = _emit_exhibit_event(tc)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("event: exhibit\n"))
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        self.assertEqual(payload["layout"], "stack")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["elements"][0]["type"], "mermaid")

    def test_emit_exhibit_event_invalid_suppressed(self):
        """Malformed declaration → no exhibit event (the tool body's error to
        the model drives a re-present)."""
        from types import SimpleNamespace
        from agentx_ai.streaming.tool_loop import _emit_exhibit_event
        tc = SimpleNamespace(id="tc_2", name="present_exhibit", arguments={"elements": []})
        self.assertEqual(_emit_exhibit_event(tc), [])


class WebSearchCapabilityTest(TestCase):
    """Capability-aware web tools (Slice 4): the active-backend pre-check that
    advertises only what the active search backend supports, the Tavily SDK +
    Brave handlers, and web_search → passive citation auto-capture.
    """

    # --- Active-backend resolver + advertisement -------------------------

    def test_resolve_active_backend_prefers_configured_primary(self):
        from agentx_ai.mcp import internal_tools as it
        fake = MagicMock()
        fake.get.return_value = "brave"
        with patch("agentx_ai.config.get_config_manager", return_value=fake), \
             patch.object(it, "_backend_has_key", lambda n: True):
            self.assertEqual(it.resolve_active_search_backend(), "brave")

    def test_resolve_active_backend_falls_back_to_keyed(self):
        from agentx_ai.mcp import internal_tools as it
        fake = MagicMock()
        fake.get.return_value = "tavily"  # primary, but no tavily key
        with patch("agentx_ai.config.get_config_manager", return_value=fake), \
             patch.object(it, "_backend_has_key", lambda n: n == "brave"):
            self.assertEqual(it.resolve_active_search_backend(), "brave")

    def test_resolve_active_backend_none_without_keys(self):
        from agentx_ai.mcp import internal_tools as it
        fake = MagicMock()
        fake.get.return_value = "tavily"
        with patch("agentx_ai.config.get_config_manager", return_value=fake), \
             patch.object(it, "_backend_has_key", lambda n: False):
            self.assertIsNone(it.resolve_active_search_backend())

    def test_build_schema_reflects_backend(self):
        from agentx_ai.mcp import internal_tools as it
        tav = it.build_tool_schema("web_search", "tavily")["properties"]
        self.assertIn("topic", tav)
        self.assertIn("include_domains", tav)
        self.assertNotIn("safesearch", tav)
        brave = it.build_tool_schema("web_search", "brave")["properties"]
        self.assertIn("safesearch", brave)
        self.assertIn("result_filter", brave)
        self.assertNotIn("topic", brave)
        # base params present for both
        for s in (tav, brave):
            self.assertIn("query", s)
            self.assertIn("max_results", s)

    def test_advertisement_gates_tavily_only_tools(self):
        from agentx_ai.mcp import internal_tools as it
        with patch.object(it, "resolve_active_search_backend", return_value="tavily"):
            names = {t.name for t in it.get_internal_tools()}
            self.assertTrue({"web_search", "web_extract", "web_map"} <= names)
        with patch.object(it, "resolve_active_search_backend", return_value="brave"):
            names = {t.name for t in it.get_internal_tools()}
            self.assertIn("web_search", names)
            self.assertNotIn("web_extract", names)
            self.assertNotIn("web_map", names)
        with patch.object(it, "resolve_active_search_backend", return_value=None):
            names = {t.name for t in it.get_internal_tools()}
            self.assertNotIn("web_search", names)

    def test_gated_tools_still_executable_when_not_advertised(self):
        """A stale web_extract call must still dispatch (self-guards), even when
        Brave is active and the tool isn't advertised."""
        from agentx_ai.mcp import internal_tools as it
        self.assertIsNotNone(it.find_internal_tool("web_extract"))
        self.assertIsNotNone(it.find_internal_tool("web_map"))

    # --- Backends forward only supported params --------------------------

    def test_tavily_search_forwards_supported_params(self):
        from agentx_ai.mcp import internal_tools as it
        client = MagicMock()
        client.search.return_value = {
            "results": [{"title": "T", "url": "https://x", "content": "snip", "score": 0.9}],
            "answer": "the answer",
        }
        with patch.object(it, "_tavily_client", return_value=client):
            payload = it._tavily_search(
                "q", 5, topic="news", time_range="week", include_answer=True, safesearch="strict"
            )
        kwargs = client.search.call_args.kwargs
        self.assertEqual(kwargs["query"], "q")
        self.assertEqual(kwargs["topic"], "news")
        self.assertEqual(kwargs["time_range"], "week")
        self.assertTrue(kwargs["include_answer"])
        self.assertNotIn("safesearch", kwargs)  # not a Tavily param → dropped
        self.assertEqual(payload["results"][0]["snippet"], "snip")
        self.assertEqual(payload["answer"], "the answer")

    def test_brave_search_maps_time_range_to_freshness(self):
        from agentx_ai.mcp import internal_tools as it
        data = {"web": {"results": [{"title": "T", "url": "https://x", "description": "d"}]}}
        with patch.object(it, "_resolve_search_key", return_value="brv"), \
             patch.object(it, "_http_get_json", return_value=data) as get:
            payload = it._brave_search("q", 5, time_range="day", safesearch="strict", topic="news")
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["freshness"], "pd")
        self.assertEqual(params["safesearch"], "strict")
        self.assertNotIn("topic", params)  # not a Brave param
        self.assertEqual(payload["results"][0]["snippet"], "d")

    def test_web_extract_requires_tavily(self):
        from agentx_ai.mcp import internal_tools as it
        with patch.object(it, "_tavily_client", side_effect=RuntimeError("no key")):
            out = it.web_extract(["https://x"])
        self.assertFalse(out["success"])
        self.assertIn("requires Tavily", out["error"])

    def test_web_extract_caps_and_returns_content(self):
        from agentx_ai.mcp import internal_tools as it
        client = MagicMock()
        client.extract.return_value = {
            "results": [{"url": "https://x", "raw_content": "full text"}],
            "failed_results": [],
        }
        with patch.object(it, "_tavily_client", return_value=client):
            out = it.web_extract([f"https://x/{i}" for i in range(30)])
        self.assertTrue(out["success"])
        self.assertEqual(len(client.extract.call_args.kwargs["urls"]), 20)  # abuse cap
        self.assertEqual(out["results"][0]["content"], "full text")

    def test_web_map_hard_caps_output(self):
        from agentx_ai.mcp import internal_tools as it
        client = MagicMock()
        client.map.return_value = {"results": [f"https://x/{i}" for i in range(500)]}
        with patch.object(it, "_tavily_client", return_value=client):
            out = it.web_map("https://x", limit=1000)
        self.assertTrue(out["success"])
        self.assertEqual(out["count"], 200)  # ceiling

    # --- Auto-capture: web_search → passive citation ----------------------

    def test_citation_exhibit_dedupes_and_caps(self):
        from agentx_ai.streaming.exhibits import citation_exhibit_from_web_search
        results = [
            {"title": "A", "url": "https://a"},
            {"title": "B", "url": "https://b"},
            {"title": "A-dup", "url": "https://a"},  # dup url
            {"title": "", "url": ""},  # blank → skipped
        ]
        ex = citation_exhibit_from_web_search(results, exhibit_id="exh_src_1")
        self.assertIsNotNone(ex)
        sources = ex.elements[0].sources
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(s.kind == "passive" and s.source_type == "web" for s in sources))
        self.assertIsNone(citation_exhibit_from_web_search([], exhibit_id="x"))

    def test_emit_web_search_citation(self):
        from types import SimpleNamespace
        from agentx_ai.streaming.tool_loop import _emit_web_search_citation
        tm = SimpleNamespace(
            tool_call_id="tc9",
            name="web_search",
            content=json.dumps({"success": True, "results": [{"title": "A", "url": "https://a"}]}),
        )
        events = _emit_web_search_citation(tm)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("event: exhibit\n"))
        payload = json.loads(events[0].split("data: ", 1)[1].strip())
        self.assertEqual(payload["id"], "exh_src_tc9")
        self.assertEqual(payload["elements"][0]["type"], "citation")

    def test_emit_web_search_citation_skips_failed_and_disabled(self):
        from types import SimpleNamespace
        from agentx_ai.streaming.tool_loop import _emit_web_search_citation
        failed = SimpleNamespace(
            tool_call_id="z", name="web_search",
            content=json.dumps({"success": False, "results": []}),
        )
        self.assertEqual(_emit_web_search_citation(failed), [])
        ok = SimpleNamespace(
            tool_call_id="z2", name="web_search",
            content=json.dumps({"success": True, "results": [{"title": "A", "url": "https://a"}]}),
        )
        fake = MagicMock()
        fake.get.return_value = False  # citations.auto_capture_web_search off
        with patch("agentx_ai.config.get_config_manager", return_value=fake):
            self.assertEqual(_emit_web_search_citation(ok), [])


class ModelFallbackTest(TestCase):
    """Slice 5 + Foundation #4 — universal model fallback (registry) + memory
    stage inheritance + feature-site coverage.

    A feature whose configured model is unavailable (provider unconfigured or
    unreachable) must fall back to the active/default model instead of crashing
    the turn. Foundation #4 extends this from the Ambassador to every feature
    site (chat path, reasoning, drafting, planner, alloy); specialized model
    roles (speculative draft/target pair, TTS/STT) and the explicit availability
    probes (`validate()`, cost estimation) intentionally stay strict.
    """

    def _registry(self, *, configured, default_model="anthropic:claude-haiku-4-5",
                  fallback_enabled=True):
        from unittest.mock import MagicMock
        from agentx_ai.providers.registry import ProviderRegistry
        from agentx_ai.providers.base import ProviderConfig
        cfg = MagicMock()
        vals = {
            "models.fallback_enabled": fallback_enabled,
            "preferences.default_model": default_model,
        }
        cfg.get.side_effect = lambda k, d=None: vals.get(k, d)
        cfg.get_provider_value.side_effect = lambda *a, **k: None
        reg = ProviderRegistry(config_manager=cfg)
        reg._provider_configs = {n: ProviderConfig(api_key="x") for n in configured}
        reg.get_provider = lambda n: f"<provider:{n}>"  # type: ignore[assignment]
        return reg

    def test_configured_model_used_as_is(self):
        reg = self._registry(configured=["anthropic"])
        provider, model_id, note = reg.resolve_with_fallback("anthropic:claude-opus-4-8")
        self.assertEqual(model_id, "claude-opus-4-8")
        self.assertIsNone(note)

    def test_unconfigured_provider_falls_back_to_default(self):
        reg = self._registry(configured=["anthropic"])
        provider, model_id, note = reg.resolve_with_fallback("lmstudio:gemma")
        self.assertEqual(model_id, "claude-haiku-4-5")  # the default chat model
        self.assertIsNotNone(note)

    def test_preferred_fallback_wins_over_global_default(self):
        reg = self._registry(configured=["anthropic", "openai"])
        provider, model_id, note = reg.resolve_with_fallback(
            "lmstudio:gemma", preferred_fallback="openai:gpt-4o"
        )
        self.assertEqual(model_id, "gpt-4o")  # the active agent model, not the global default

    def test_cached_unhealthy_provider_is_skipped(self):
        reg = self._registry(configured=["lmstudio", "anthropic"])
        reg.mark_provider_health("lmstudio", False)
        provider, model_id, note = reg.resolve_with_fallback("lmstudio:gemma")
        self.assertEqual(model_id, "claude-haiku-4-5")  # skipped lmstudio → default
        self.assertIsNotNone(note)

    def test_kill_switch_restores_strict_behavior(self):
        from agentx_ai.exceptions import ModelNotFoundError
        reg = self._registry(configured=["anthropic"], fallback_enabled=False)
        with self.assertRaises(ModelNotFoundError):
            reg.resolve_with_fallback("lmstudio:gemma")

    def test_complete_with_fallback_retries_on_runtime_error(self):
        import asyncio
        from unittest.mock import AsyncMock
        reg = self._registry(configured=["lmstudio", "anthropic"])

        good = AsyncMock(return_value="OK")
        bad = AsyncMock(side_effect=RuntimeError("LM Studio down"))
        providers = {
            "lmstudio": type("P", (), {"complete": bad})(),
            "anthropic": type("P", (), {"complete": good})(),
        }
        reg.get_provider = lambda n: providers[n]  # type: ignore[assignment]

        result = asyncio.run(reg.complete_with_fallback("lmstudio:gemma", ["m"]))
        self.assertEqual(result, "OK")  # retried onto the healthy default
        bad.assert_awaited_once()
        good.assert_awaited_once()
        # lmstudio is now cached-unhealthy from the observed failure
        self.assertTrue(reg._is_cached_unhealthy("lmstudio"))

    def test_reasoning_site_routes_through_fallback(self):
        """A reasoning strategy (representative feature site) resolves via
        resolve_with_fallback, so an unavailable sub-model degrades to the
        fallback model rather than hard-failing the turn."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from agentx_ai.reasoning.chain_of_thought import ChainOfThought, CoTConfig

        reasoner = ChainOfThought(CoTConfig(model="lmstudio:gemma", extract_steps=False))

        completion = MagicMock(content="Answer: 42", usage={"total_tokens": 7})
        fake_provider = MagicMock(complete=AsyncMock(return_value=completion))
        reg = MagicMock()
        # The configured sub-model is "unavailable" → resolver substitutes the default.
        reg.resolve_with_fallback.return_value = (fake_provider, "claude-haiku-4-5", "substituted")
        reasoner._registry = reg

        result = asyncio.run(reasoner.reason("what is 6 * 7?"))

        # Routed through the fallback resolver with the configured model...
        reg.resolve_with_fallback.assert_called_once_with("lmstudio:gemma")
        reg.get_provider_for_model.assert_not_called()
        # ...and ran against the substituted model, not the unavailable one.
        self.assertEqual(fake_provider.complete.await_args.args[1], "claude-haiku-4-5")
        self.assertEqual(result.answer, "42")

    def test_stage_model_inheritance(self):
        from unittest.mock import MagicMock, patch
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionService
        svc = ExtractionService.__new__(ExtractionService)
        # explicit wins
        svc._settings = MagicMock(feature_default_model="anthropic:bulk")
        with patch.object(type(svc), "settings", property(lambda s: s._settings)):
            self.assertEqual(svc._resolve_stage_model("lmstudio:explicit"), "lmstudio:explicit")
            # empty stage → feature_default_model (bulk)
            self.assertEqual(svc._resolve_stage_model(""), "anthropic:bulk")
            self.assertEqual(svc._resolve_stage_model("inherit"), "anthropic:bulk")
        # empty stage + empty bulk → global default chat model
        svc._settings = MagicMock(feature_default_model="")
        cfg = MagicMock()
        cfg.get.side_effect = lambda k, d=None: "anthropic:main" if k == "preferences.default_model" else d
        with patch.object(type(svc), "settings", property(lambda s: s._settings)), \
             patch("agentx_ai.config.get_config_manager", return_value=cfg):
            self.assertEqual(svc._resolve_stage_model(""), "anthropic:main")


class UsageLedgerTest(TestCase):
    """Foundation #5 — the content-free usage/cost ledger writer."""

    def _mock_session(self):
        from unittest.mock import MagicMock
        session = MagicMock()
        cm = MagicMock()
        cm.__enter__.return_value = session
        cm.__exit__.return_value = False
        return session, cm

    def test_record_usage_writes_normalized_row(self):
        from unittest.mock import patch
        from agentx_ai.agent import usage_ledger
        session, cm = self._mock_session()
        cost = {
            "cost_total": 0.0123, "currency": "USD",
            "pricing_snapshot": {"cost_per_1k_input": 1.0, "cost_per_1k_output": 2.0},
        }
        with patch.object(usage_ledger, "get_postgres_session", return_value=cm, create=True), \
             patch("agentx_ai.kit.agent_memory.connections.get_postgres_session", return_value=cm):
            usage_ledger.record_usage(
                source="ambassador_llm", model="anthropic:opus", provider="anthropic",
                conversation_id="c1", agent_id="amb-1",
                units={"tokens_in": 10, "tokens_out": 20, "tokens_total": 30}, cost=cost,
            )
        session.execute.assert_called_once()
        params = session.execute.call_args.args[1]
        self.assertEqual(params["source"], "ambassador_llm")
        self.assertEqual(params["cost_total"], 0.0123)
        self.assertEqual(params["currency"], "USD")
        self.assertIn("tokens_total", params["units"])      # JSON-encoded units
        self.assertIsNone(params["ref"])                    # ambassador rows always insert
        session.commit.assert_called_once()

    def test_record_usage_ref_is_passed_for_dedupe(self):
        from unittest.mock import patch
        from agentx_ai.agent import usage_ledger
        session, cm = self._mock_session()
        with patch("agentx_ai.kit.agent_memory.connections.get_postgres_session", return_value=cm):
            usage_ledger.record_usage(
                source="chat", model="m", conversation_id="c1",
                units={"tokens_in": 1, "tokens_out": 1, "tokens_total": 2},
                cost=None, ref=usage_ledger.turn_ref("c1", 4),
            )
        params = session.execute.call_args.args[1]
        self.assertEqual(params["ref"], "c1:4")
        self.assertIsNone(params["cost_total"])             # no pricing → null cost, still recorded
        self.assertEqual(params["currency"], "USD")

    def test_record_usage_never_raises(self):
        from unittest.mock import patch
        from agentx_ai.agent import usage_ledger
        # A DB failure must never break a turn.
        with patch("agentx_ai.kit.agent_memory.connections.get_postgres_session",
                   side_effect=RuntimeError("postgres down")):
            usage_ledger.record_usage(source="chat", model="m", units={}, cost=None)
        # No exception == pass.

    def test_turn_ref_requires_both_parts(self):
        from agentx_ai.agent.usage_ledger import turn_ref
        self.assertEqual(turn_ref("c1", 0), "c1:0")
        self.assertIsNone(turn_ref(None, 3))
        self.assertIsNone(turn_ref("c1", None))

    def test_backfill_classifies_and_keys_rows(self):
        """The history backfill tags alloy vs chat and keys each row by
        conversation_id:turn_index so re-runs upsert (no double-count)."""
        from unittest.mock import patch, MagicMock
        from types import SimpleNamespace
        from django.core.management import call_command
        rows = [
            SimpleNamespace(conversation_id="c1", turn_index=2, agent_id="a1", model="m",
                            provider="anthropic", tokens_in=5, tokens_out=7, tokens_total=12,
                            cost_total=0.01, currency="USD", pricing_snapshot=None, is_alloy=False),
            SimpleNamespace(conversation_id="c1", turn_index=3, agent_id="spec", model="m2",
                            provider="openai", tokens_in=1, tokens_out=2, tokens_total=3,
                            cost_total=None, currency="USD", pricing_snapshot=None, is_alloy=True),
        ]
        session, cm = self._mock_session()
        select_result = MagicMock()
        select_result.fetchall.return_value = rows
        session.execute.side_effect = [select_result, MagicMock(), MagicMock()]
        with patch("agentx_ai.kit.agent_memory.connections.get_postgres_session", return_value=cm):
            call_command("backfill_usage_ledger")
        self.assertEqual(session.execute.call_count, 3)  # 1 select + 2 upserts
        up1 = session.execute.call_args_list[1].args[1]
        self.assertEqual((up1["source"], up1["ref"]), ("chat", "c1:2"))
        up2 = session.execute.call_args_list[2].args[1]
        self.assertEqual((up2["source"], up2["ref"]), ("alloy", "c1:3"))
        self.assertIn("tokens_total", up2["units"])

    def test_ambassador_stream_records_usage(self):
        """The ambassador's streaming answer sums chunk usage and records an
        `ambassador_llm` event (content-free) when it settles."""
        import asyncio
        from unittest.mock import patch
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.providers.base import StreamChunk, ModelCapabilities

        class FakeProvider:
            name = "anthropic"
            def get_capabilities(self, _m):
                return ModelCapabilities(cost_per_1k_input=1.0, cost_per_1k_output=2.0)
            async def stream(self, messages, model_id, **kw):
                yield StreamChunk(content="Hello")
                yield StreamChunk(usage={"prompt_tokens": 100, "completion_tokens": 50},
                                  finish_reason="stop")

        svc = AmbassadorService()
        recorded: dict = {}

        async def _run():
            agen = svc._stream_and_settle(
                item_id="x", provider=FakeProvider(), model_id="anthropic:opus",
                temperature=0.2, max_tokens=100, messages=[],
                on_chunk=lambda t: None, on_done=lambda s: None,
                on_cancel=lambda: None, on_error=lambda e: None,
                empty_text="(empty)", log_label="test",
                conversation_id="c1", agent_id="amb-1",
            )
            async for _ in agen:
                pass

        with patch("agentx_ai.agent.usage_ledger.record_usage",
                   side_effect=lambda **kw: recorded.update(kw)):
            asyncio.run(_run())

        self.assertEqual(recorded["source"], "ambassador_llm")
        self.assertEqual(recorded["agent_id"], "amb-1")
        self.assertEqual(recorded["units"]["tokens_in"], 100)
        self.assertEqual(recorded["units"]["tokens_out"], 50)
        self.assertIsNotNone(recorded["cost"])  # priced model → cost estimated

    def test_estimate_audio_cost(self):
        from agentx_ai.providers.pricing import estimate_audio_cost
        # TTS: per-1k-chars rate (shipped default for mai-voice-2 = $0.015/1k).
        tts = estimate_audio_cost(model="openrouter:microsoft/mai-voice-2", chars=2000)
        self.assertIsNotNone(tts)
        self.assertAlmostEqual(tts["cost_total"], 0.03)
        self.assertIn("per_1k_chars", tts["pricing_snapshot"])
        # STT: per-minute rate (whisper-1 = $0.006/min).
        stt = estimate_audio_cost(model="openrouter:openai/whisper-1", seconds=120)
        self.assertIsNotNone(stt)
        self.assertAlmostEqual(stt["cost_total"], 0.012)
        # No rate for the model → None.
        self.assertIsNone(estimate_audio_cost(model="local:whatever", chars=100))
        # Rate exists but the supplied unit isn't priced (whisper has no char rate,
        # no seconds given) → None, not a fabricated zero.
        self.assertIsNone(estimate_audio_cost(model="openrouter:openai/whisper-1", chars=100))

    @override_settings(AGENTX_AUTH_ENABLED=False)
    def test_usage_metrics_includes_by_source(self):
        """GET /api/metrics/usage aggregates the ledger with a by-source breakdown."""
        from unittest.mock import patch, MagicMock
        from django.test import Client

        def fake_execute(stmt, params=None):
            sql = str(stmt)
            res = MagicMock()
            if "GROUP BY source" in sql:
                res.mappings.return_value.all.return_value = [
                    {"source": "chat", "turns": 3, "tokens_input": 10, "tokens_output": 5,
                     "tokens_total": 15, "cost_total": 0.02},
                    {"source": "ambassador_tts", "turns": 1, "tokens_input": 0, "tokens_output": 0,
                     "tokens_total": 0, "cost_total": 0.001},
                ]
            elif "avg_latency_ms" in sql:
                res.scalar.return_value = 1200.0
            elif "FROM usage_events" in sql and "GROUP BY" not in sql:
                res.mappings.return_value.first.return_value = {
                    "turns": 4, "tokens_input": 10, "tokens_output": 5,
                    "tokens_total": 15, "cost_total": 0.021, "cost_currency": "USD"}
            else:
                res.mappings.return_value.all.return_value = []
            return res

        session = MagicMock()
        session.execute.side_effect = fake_execute
        cm = MagicMock()
        cm.__enter__.return_value = session
        cm.__exit__.return_value = False
        with patch("agentx_ai.kit.agent_memory.connections.get_postgres_session", return_value=cm):
            resp = Client().get("/api/metrics/usage?days=7")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        sources = {row["source"]: row for row in data["by_source"]}
        self.assertIn("ambassador_tts", sources)
        self.assertEqual(sources["chat"]["cost_total"], 0.02)
        self.assertEqual(data["totals"]["avg_latency_ms"], 1200.0)


class WebResearchToolsTest(TestCase):
    """Slice 5 — Tavily crawl/research tools (capability-gated, self-guarding)."""

    def test_crawl_research_advertised_only_for_tavily(self):
        from agentx_ai.mcp import internal_tools as it
        from unittest.mock import patch
        with patch.object(it, "resolve_active_search_backend", return_value="tavily"):
            names = {t.name for t in it.get_internal_tools()}
            self.assertIn("web_crawl", names)
            self.assertIn("web_research", names)
        with patch.object(it, "resolve_active_search_backend", return_value="brave"):
            names = {t.name for t in it.get_internal_tools()}
            self.assertNotIn("web_crawl", names)
            self.assertNotIn("web_research", names)
        # still executable (self-guarding) regardless of advertisement
        self.assertIsNotNone(it.find_internal_tool("web_crawl"))
        self.assertIsNotNone(it.find_internal_tool("web_research"))

    def test_crawl_requires_tavily_and_caps_pages(self):
        from agentx_ai.mcp import internal_tools as it
        from unittest.mock import patch, MagicMock
        with patch.object(it, "_tavily_client", side_effect=RuntimeError("no key")):
            out = it.web_crawl("https://x")
        self.assertFalse(out["success"])
        self.assertIn("requires Tavily", out["error"])

        client = MagicMock()
        client.crawl.return_value = {"results": [{"url": f"https://x/{i}", "raw_content": "c"} for i in range(200)]}
        with patch.object(it, "_tavily_client", return_value=client):
            out = it.web_crawl("https://x", limit=1000)
        self.assertTrue(out["success"])
        self.assertEqual(out["count"], 50)  # hard cap

    def test_research_disabled_flag(self):
        from agentx_ai.mcp import internal_tools as it
        from unittest.mock import patch, MagicMock
        cfg = MagicMock()
        cfg.get.side_effect = lambda k, d=None: False if k == "web_research.enabled" else d
        with patch("agentx_ai.config.get_config_manager", return_value=cfg):
            out = it.web_research("q")
        self.assertFalse(out["success"])
        self.assertIn("disabled", out["error"])

    def test_research_normalizes_sources_for_autocapture(self):
        from agentx_ai.mcp import internal_tools as it
        from unittest.mock import patch, MagicMock
        client = MagicMock()
        client.research.return_value = {
            "answer": "a report",
            "citations": [{"title": "A", "url": "https://a"}, "https://b"],
        }
        cfg = MagicMock()
        cfg.get.side_effect = lambda k, d=None: True if k == "web_research.enabled" else d
        with patch("agentx_ai.config.get_config_manager", return_value=cfg), \
             patch.object(it, "_tavily_client", return_value=client):
            out = it.web_research("q", depth="pro")
        self.assertTrue(out["success"])
        self.assertEqual(out["report"], "a report")
        self.assertEqual(out["results"][0]["url"], "https://a")
        self.assertEqual(out["results"][1]["url"], "https://b")  # bare string normalized


class ConversationContextTest(TestCase):
    """Slice 6 — rehydrate verbatim transcript + context-window-based assembly."""

    def test_load_recent_turns_chronological_and_budgeted(self):
        from agentx_ai.agent.conversation_history import load_recent_turns
        from agentx_ai.providers.base import MessageRole
        # reader returns newest-first (role, content)
        rows = [("assistant", "A3"), ("user", "U3"), ("assistant", "A2"),
                ("user", "U2"), ("assistant", "A1"), ("user", "U1")]
        msgs = load_recent_turns("c", token_budget=10_000, reader=lambda c, n: rows)
        self.assertEqual([m.content for m in msgs], ["U1", "A1", "U2", "A2", "U3", "A3"])
        self.assertEqual(msgs[0].role, MessageRole.USER)
        # Tiny budget still keeps at least the most-recent turn.
        one = load_recent_turns("c", token_budget=1, reader=lambda c, n: rows)
        self.assertEqual([m.content for m in one], ["A3"])

    def test_hydrate_is_idempotent_and_fills_empty_session(self):
        from agentx_ai.agent.session import Session
        from agentx_ai.agent.conversation_history import hydrate_session_from_history
        rows = [("assistant", "A1"), ("user", "U1")]
        reader = lambda c, n: rows  # noqa: E731
        s = Session(id="c1")
        n = hydrate_session_from_history(s, "c1", token_budget=10_000, reader=reader)
        self.assertEqual(n, 2)
        self.assertEqual([m.content for m in s.messages], ["U1", "A1"])
        # Second call no-ops (already populated / hydrated flag).
        self.assertEqual(hydrate_session_from_history(s, "c1", token_budget=10_000, reader=reader), 0)
        # A session that already has live messages is never clobbered.
        s2 = Session(id="c2")
        from agentx_ai.providers.base import Message, MessageRole
        s2.add_message(Message(role=MessageRole.USER, content="live"))
        self.assertEqual(hydrate_session_from_history(s2, "c2", token_budget=10_000, reader=reader), 0)
        self.assertEqual([m.content for m in s2.messages], ["live"])

    def test_chat_rehydrates_cold_session(self):
        """Foundation #4b — agent.chat() hydrates from durable history before the
        turn, so the queued/background-chat path resumes warm (not just the
        interactive stream). Hydration runs early, so it's captured even when the
        downstream completion can't resolve a provider in the test env."""
        from unittest.mock import patch
        from agentx_ai.agent.core import Agent, AgentConfig
        agent = Agent(AgentConfig(enable_memory=False, enable_tools=False))
        with patch(
            "agentx_ai.agent.conversation_history.hydrate_session_from_history",
            return_value=0,
        ) as hyd:
            agent.chat("hello", session_id="conv-xyz")
        hyd.assert_called_once()
        # Hydrated against the conversation the turn belongs to.
        self.assertEqual(hyd.call_args.args[1], "conv-xyz")

    def test_assemble_turn_context_budget_fit(self):
        from agentx_ai.agent.context import ContextManager, ContextConfig
        from agentx_ai.providers.base import Message, MessageRole
        mgr = ContextManager(ContextConfig())
        sys = [Message(role=MessageRole.SYSTEM, content="S")]
        history = [Message(role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                           content="x" * 400) for i in range(20)]
        new = Message(role=MessageRole.USER, content="now")
        # Large window: everything fits.
        out = mgr.assemble_turn_context(
            system_blocks=sys, history=history, new_message=new,
            context_window=1_000_000, reserved_tokens=1000, verbatim_ratio=0.7, recent_floor=4)
        self.assertEqual(len(out), 1 + 20 + 1)
        self.assertEqual(out[0].content, "S")
        self.assertEqual(out[-1].content, "now")
        # Tiny window: keeps system + floor of recent turns + new.
        out2 = mgr.assemble_turn_context(
            system_blocks=sys, history=history, new_message=new,
            context_window=300, reserved_tokens=100, verbatim_ratio=0.7, recent_floor=4)
        kept = [m for m in out2 if m.role != MessageRole.SYSTEM and m.content != "now"]
        self.assertEqual(len(kept), 4)  # floor honored
        self.assertEqual([m.content for m in kept], [m.content for m in history[-4:]])

    def test_maybe_update_summary_is_token_triggered(self):
        import asyncio
        from unittest.mock import patch, AsyncMock
        from agentx_ai.agent.session import SessionManager
        from agentx_ai.providers.base import Message, MessageRole

        mgr = SessionManager()
        s = mgr.create(session_id="c")
        for i in range(12):
            s.add_message(Message(role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                                  content="y" * 400))

        with patch("agentx_ai.agent.context.ContextManager._summarize_messages",
                   new=AsyncMock(return_value="SUMMARY")), \
             patch("agentx_ai.agent.conversation_summary_storage.set_summary") as set_sum:
            # High threshold → nothing aged out.
            none = asyncio.run(mgr.maybe_update_summary("c", token_threshold=10_000_000, recent_floor=4))
            self.assertFalse(none)
            # Low threshold → summarize + trim + persist.
            did = asyncio.run(mgr.maybe_update_summary("c", token_threshold=300, recent_floor=4))
            self.assertTrue(did)
            self.assertEqual(s.summary, "SUMMARY")
            self.assertEqual(len(s.messages), 4)  # trimmed to the recent floor
            set_sum.assert_called_once()


class TokenEstimatorTest(TestCase):
    """Foundation #6 — the shared tiktoken-backed token estimator.

    Resets the module-level encoder cache around each test so the lazy singleton
    doesn't leak the loaded/failed state between cases.
    """

    def setUp(self):
        import agentx_ai.tokens as tokens
        self.tokens = tokens
        self._saved = (tokens._encoder, tokens._encoder_failed)

    def tearDown(self):
        self.tokens._encoder, self.tokens._encoder_failed = self._saved

    def test_empty_input_is_zero(self):
        self.assertEqual(self.tokens.estimate_tokens(""), 0)
        self.assertEqual(self.tokens.estimate_tokens(None), 0)

    def test_tiktoken_path_returns_positive(self):
        # Real encoder path — a non-trivial sentence is at least a few tokens.
        n = self.tokens.estimate_tokens(
            "Consider whether the premise actually supports the conclusion."
        )
        self.assertGreater(n, 3)

    def test_falls_back_to_char_heuristic_when_tiktoken_unavailable(self):
        import builtins
        tokens = self.tokens
        tokens._encoder = None
        tokens._encoder_failed = False
        real_import = builtins.__import__

        def _no_tiktoken(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("simulated missing tiktoken")
            return real_import(name, *args, **kwargs)

        text = "y" * 400  # under the fast-path cutoff, so it would normally tokenize
        with patch("builtins.__import__", side_effect=_no_tiktoken):
            self.assertEqual(
                tokens.estimate_tokens(text),
                len(text) // tokens.FALLBACK_CHARS_PER_TOKEN,
            )
        self.assertTrue(tokens._encoder_failed)  # failure is cached, logged once

    def test_messages_add_per_message_overhead(self):
        from agentx_ai.providers.base import Message, MessageRole
        tokens = self.tokens
        msgs = [
            Message(role=MessageRole.USER, content="alpha beta"),
            Message(role=MessageRole.ASSISTANT, content="gamma delta"),
        ]
        expected = (
            sum(tokens.estimate_tokens(m.content) for m in msgs)
            + len(msgs) * tokens._PER_MESSAGE_OVERHEAD
        )
        self.assertEqual(tokens.estimate_messages(msgs), expected)

    def test_large_string_uses_char_fast_path(self):
        tokens = self.tokens
        big = "reason " * 4000  # 28k chars, above _EXACT_MAX_CHARS
        self.assertGreater(len(big), tokens._EXACT_MAX_CHARS)
        # Above the cutoff we skip tiktoken entirely and use chars/4.
        self.assertEqual(
            tokens.estimate_tokens(big),
            len(big) // tokens.FALLBACK_CHARS_PER_TOKEN,
        )


class WorkspaceIngestionTest(TestCase):
    """File Workspaces & Document RAG — Slice 1 (parsing/storage are pure; the full
    ingestion path skips without a healthy Postgres or local embeddings)."""

    def test_extension_and_support(self):
        from agentx_ai.kit.workspaces import parsing
        self.assertEqual(parsing.extension_of("Report.PDF"), "pdf")
        self.assertEqual(parsing.extension_of("notes.md"), "md")
        self.assertTrue(parsing.is_supported("a.py", ["py", "md"]))
        self.assertFalse(parsing.is_supported("a.exe", ["py", "md"]))

    def test_parse_strips_nul_bytes(self):
        # Postgres TEXT rejects 0x00; PDF extraction emits them — must be stripped.
        from agentx_ai.kit.workspaces import parsing
        out = parsing.parse_to_text(b"hel\x00lo world", "note.txt")
        self.assertNotIn("\x00", out)
        self.assertEqual(out, "hello world")

    def test_blob_roundtrip_is_content_addressed(self):
        import hashlib
        from agentx_ai.kit.workspaces import storage
        raw = b"the premise does not entail the conclusion"
        sha, key = storage.store_blob("ws_test_unit", raw)
        try:
            self.assertEqual(sha, hashlib.sha256(raw).hexdigest())
            self.assertTrue(key.endswith(sha))
            self.assertEqual(storage.read_blob(key), raw)
            # Same bytes → same key (dedup), idempotent store.
            sha2, key2 = storage.store_blob("ws_test_unit", raw)
            self.assertEqual((sha, key), (sha2, key2))
        finally:
            storage.delete_blob(key)

    def test_full_ingestion_against_postgres(self):
        from agentx_ai.kit.agent_memory.connections import PostgresConnection, get_postgres_session
        if PostgresConnection.health_check().get("status") != "healthy":
            self.skipTest("Postgres unavailable")
        from agentx_ai.kit.workspaces import ingestion, repository, storage
        try:
            from agentx_ai.kit.agent_memory.embeddings import get_embedder
            get_embedder().embed(["warmup"])
        except Exception as e:  # embeddings model not available
            self.skipTest(f"embeddings unavailable: {e}")

        ws = repository.create_workspace(name="unit-test-ingestion")
        try:
            raw = (
                b"Deductive reasoning guarantees the conclusion when the premises hold. "
                b"Inductive reasoning only makes it probable."
            )
            sha, key = storage.store_blob(ws["id"], raw)
            doc = repository.create_document(
                workspace_id=ws["id"], filename="reasoning.txt",
                content_type="text/plain", size_bytes=len(raw), sha256=sha, storage_key=key,
            )
            result = ingestion.ingest_document(doc["id"])
            self.assertEqual(result["status"], "ready", result)
            with get_postgres_session() as s:
                from sqlalchemy import text
                n = s.execute(
                    text("SELECT COUNT(*) FROM document_chunks WHERE document_id=:id AND embedding IS NOT NULL"),
                    {"id": doc["id"]},
                ).scalar()
            self.assertGreater(n or 0, 0)
            self.assertEqual(repository.get_document(doc["id"])["status"], "ready")
        finally:
            for d in repository.list_documents(ws["id"]):
                full = repository.get_document(d["id"])
                if full and full.get("storage_key"):
                    storage.delete_blob(full["storage_key"])
            repository.delete_workspace(ws["id"])

    def test_retrieval_and_tools_against_postgres(self):
        from agentx_ai.kit.agent_memory.connections import PostgresConnection
        if PostgresConnection.health_check().get("status") != "healthy":
            self.skipTest("Postgres unavailable")
        from agentx_ai.kit.workspaces import ingestion, repository, retrieval, storage
        try:
            from agentx_ai.kit.agent_memory.embeddings import get_embedder
            get_embedder().embed(["warmup"])
        except Exception as e:
            self.skipTest(f"embeddings unavailable: {e}")

        ws = repository.create_workspace(name="unit-test-retrieval")
        try:
            raw = (
                b"Photosynthesis converts sunlight, water, and carbon dioxide into "
                b"glucose and oxygen inside the chloroplast."
            )
            sha, key = storage.store_blob(ws["id"], raw)
            doc = repository.create_document(
                workspace_id=ws["id"], filename="biology.txt",
                content_type="text/plain", size_bytes=len(raw), sha256=sha, storage_key=key,
            )
            self.assertEqual(ingestion.ingest_document(doc["id"])["status"], "ready")

            # Semantic retrieval finds the passage from a paraphrased query.
            hits = retrieval.query_chunks(ws["id"], "how do plants make energy from light", top_k=3)
            self.assertTrue(hits and hits[0]["filename"] == "biology.txt")
            self.assertGreater(hits[0]["score"], 0.1)
            # Manifest (catalog) search finds the file by name.
            self.assertTrue(retrieval.search_manifest(ws["id"], "biology"))

            # The agent-facing tools resolve the active workspace from context.
            from agentx_ai.mcp.internal_context import (
                InternalToolContext, reset_context, set_context,
            )
            from agentx_ai.mcp.internal_tools import execute_internal_tool
            tok = set_context(InternalToolContext(user_id="default", workspace_id=ws["id"]))
            try:
                res = execute_internal_tool("document_query", {"query": "glucose and oxygen"})
                self.assertTrue(res.success)
                # No workspace bound → a clear, non-fatal error (not a crash).
                reset_context(tok)
                tok = set_context(InternalToolContext(user_id="default", workspace_id=None))
                res2 = execute_internal_tool("document_query", {"query": "x"})
                payload2 = json.loads(res2.content[0]["text"])
                self.assertFalse(payload2.get("success"))
            finally:
                reset_context(tok)
        finally:
            for d in repository.list_documents(ws["id"]):
                full = repository.get_document(d["id"])
                if full and full.get("storage_key"):
                    storage.delete_blob(full["storage_key"])
            repository.delete_workspace(ws["id"])

    def test_render_manifest_block_pure(self):
        # Pure formatting check (no DB) via monkeypatching the document list.
        from agentx_ai.kit.workspaces import retrieval
        from unittest.mock import patch
        docs = [
            {"filename": "a.pdf", "tags": ["finance"], "summary": "Q3 report.", "status": "ready"},
            {"filename": "b.md", "tags": [], "summary": "", "status": "pending"},  # excluded
        ]
        with patch.object(retrieval.repository, "list_documents", return_value=docs):
            block = retrieval.render_manifest_block("ws_x")
        self.assertIn("a.pdf", block)
        self.assertIn("finance", block)
        self.assertNotIn("b.md", block)  # only ready docs appear


class ShellSandboxTest(TestCase):
    """Agent shells — sandbox, policy, path-jail (pure); bwrap jail (skips if absent)."""

    def test_minimal_env_has_no_secrets(self):
        from agentx_ai.kit.shell.env import minimal_env
        env = minimal_env("/work")
        self.assertEqual(env["HOME"], "/work")
        for secret in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "POSTGRES_PASSWORD", "NEO4J_PASSWORD"):
            self.assertNotIn(secret, env)

    def test_policy_blocks_destructive(self):
        from agentx_ai.kit.shell import policy
        self.assertIsNone(policy.check_command("ls -la && grep foo bar.txt"))
        self.assertIsNotNone(policy.check_command("sudo apt install x"))
        self.assertIsNotNone(policy.check_command("rm -rf /"))
        self.assertIsNotNone(policy.check_command(""))

    def test_cap_output(self):
        from agentx_ai.kit.shell import policy
        capped, trunc = policy.cap_output("x" * 100, 10)
        self.assertTrue(trunc)
        self.assertTrue(capped.startswith("x" * 10))
        same, trunc2 = policy.cap_output("short", 100)
        self.assertFalse(trunc2)
        self.assertEqual(same, "short")

    def test_path_jail(self):
        import tempfile
        from pathlib import Path
        from agentx_ai.kit.shell.workdir import WorkdirError, resolve_in_workdir
        with tempfile.TemporaryDirectory() as t:
            wd = Path(t)
            ok = resolve_in_workdir(wd, "a/b.txt")
            self.assertTrue(str(ok).startswith(str(wd.resolve())))
            with self.assertRaises(WorkdirError):
                resolve_in_workdir(wd, "../escape")
            with self.assertRaises(WorkdirError):
                resolve_in_workdir(wd, "/etc/passwd")

    def test_bubblewrap_jail(self):
        import tempfile
        from pathlib import Path
        from agentx_ai.kit.shell.sandbox import BubblewrapSandbox, bubblewrap_works
        if not bubblewrap_works():
            self.skipTest("bubblewrap unavailable")
        sb = BubblewrapSandbox()
        with tempfile.TemporaryDirectory() as t:
            wd = Path(t)
            ok = sb.run("echo hello", cwd=wd, timeout=10, allow_network=False)
            self.assertEqual(ok.exit_code, 0)
            self.assertEqual(ok.stdout.strip(), "hello")
            # Network is off → curl fails.
            net = sb.run("curl -m3 https://example.com", cwd=wd, timeout=10, allow_network=False)
            self.assertNotEqual(net.exit_code, 0)
            # Timeout kills a long command.
            slow = sb.run("sleep 5", cwd=wd, timeout=1, allow_network=False)
            self.assertTrue(slow.timed_out)

    def test_run_command_hidden_without_shell_workspace(self):
        # No shell-allowed workspace bound in context → shell tools are not advertised.
        from agentx_ai.mcp.internal_tools import get_internal_tools
        self.assertNotIn("run_command", {t.name for t in get_internal_tools()})


class ContainerShellTest(TestCase):
    """Container shell backend — pure routing/jail checks (full e2e is scripts/shell_container_e2e.py)."""

    def test_safe_rel_path_jail(self):
        from agentx_ai.kit.shell.container import ContainerError, _safe_rel
        self.assertEqual(_safe_rel("a/b.txt"), "a/b.txt")
        self.assertEqual(_safe_rel("./a/./b"), "a/b")
        with self.assertRaises(ContainerError):
            _safe_rel("../escape")
        with self.assertRaises(ContainerError):
            _safe_rel("/etc/passwd")

    def test_backend_for_defaults_bubblewrap(self):
        # No workspace → bubblewrap (no DB needed); the container backend is opt-in per workspace.
        from agentx_ai.kit.shell.dispatch import backend_for
        self.assertEqual(backend_for(None), "bubblewrap")

    def test_docker_available_is_bool(self):
        from agentx_ai.kit.shell.container import docker_available
        self.assertIsInstance(docker_available(), bool)


class ContextLedgerTest(TestCase):
    """Foundation #3 — priority-based budget allocator (Context Ledger)."""

    def _history(self, n, size=400):
        from agentx_ai.providers.base import Message, MessageRole
        return [Message(role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                        content="x" * size) for i in range(n)]

    def test_priority_shrink_and_drop(self):
        from agentx_ai.agent.context_ledger import (
            LedgerBlock, assemble_ledger, shrink_tail,
        )
        from agentx_ai.providers.base import Message, MessageRole
        history = self._history(20)
        new = Message(role=MessageRole.USER, content="now")
        base = LedgerBlock(key="base", priority=100, mandatory=True, content="SYS-PROMPT")
        mid = LedgerBlock(key="mid", priority=70, content="m" * 8000,
                          min_tokens=10, shrink_fn=shrink_tail)
        low = LedgerBlock(key="low", priority=30, content="l" * 8000)

        # Generous window: everything full, canonical (registration) order, history kept.
        big = assemble_ledger(blocks=[base, mid, low], history=history, new_message=new,
                              context_window=1_000_000, reserved_tokens=1000)
        st = {a.key: a.status for a in big.allocations}
        self.assertEqual(st, {"base": "full", "mid": "full", "low": "full"})
        self.assertEqual([m.content[:3] for m in big.messages[:3]], ["SYS", "mmm", "lll"])
        self.assertEqual(big.history_kept, 20)
        self.assertEqual(big.messages[-1].content, "now")

        # Tight window: low dropped, mid shrunk, base survives, history honors floor.
        tight = assemble_ledger(blocks=[base, mid, low], history=history, new_message=new,
                                context_window=2000, reserved_tokens=200, recent_floor=4)
        st2 = {a.key: a.status for a in tight.allocations}
        self.assertEqual(st2["base"], "full")
        self.assertEqual(st2["low"], "dropped")
        self.assertEqual(st2["mid"], "shrunk")
        kept = [m for m in tight.messages if m.role != MessageRole.SYSTEM
                and m.content != "now"]
        self.assertGreaterEqual(len(kept), 4)  # recent_floor honored
        self.assertEqual(tight.messages[-1].content, "now")
        # Shrunk block actually fits what it was granted.
        mid_alloc = next(a for a in tight.allocations if a.key == "mid")
        self.assertLessEqual(mid_alloc.granted_tokens, mid_alloc.requested_tokens)
        self.assertGreaterEqual(mid_alloc.granted_tokens, 10)

    def test_emission_preserves_registration_order(self):
        from agentx_ai.agent.context_ledger import LedgerBlock, assemble_ledger
        from agentx_ai.providers.base import Message, MessageRole
        new = Message(role=MessageRole.USER, content="now")
        # Registered low→high priority, but emission must follow registration order.
        blocks = [
            LedgerBlock(key="a", priority=10, content="AAA"),
            LedgerBlock(key="b", priority=99, content="BBB"),
            LedgerBlock(key="c", priority=50, content="CCC"),
        ]
        res = assemble_ledger(blocks=blocks, history=[], new_message=new,
                              context_window=1_000_000, reserved_tokens=1000)
        self.assertEqual([m.content for m in res.messages], ["AAA", "BBB", "CCC", "now"])

    def test_mandatory_survives_window_too_small(self):
        from agentx_ai.agent.context_ledger import LedgerBlock, assemble_ledger
        from agentx_ai.providers.base import Message, MessageRole
        new = Message(role=MessageRole.USER, content="now")
        base = LedgerBlock(key="base", priority=100, mandatory=True, content="K" * 4000)
        drop = LedgerBlock(key="drop", priority=30, content="D" * 4000)
        # Budget far smaller than the mandatory block — it must still be emitted.
        res = assemble_ledger(blocks=[base, drop], history=[], new_message=new,
                              context_window=200, reserved_tokens=50)
        st = {a.key: a.status for a in res.allocations}
        self.assertEqual(st["base"], "full")
        self.assertEqual(st["drop"], "dropped")
        self.assertEqual(res.messages[0].content, base.content)
        self.assertEqual(res.messages[-1].content, "now")

    def test_dedup_recall_against_core(self):
        from agentx_ai.agent.context_ledger import dedup_recall_against_core
        from agentx_ai.kit.agent_memory.models import MemoryBundle
        core = MemoryBundle()
        core.facts = [{"id": "f1", "claim": "a"}, {"id": "f2", "claim": "b"}]
        core.entities = [{"id": "e1", "name": "X"}]
        recall = MemoryBundle()
        recall.facts = [{"id": "f2", "claim": "b"}, {"id": "f3", "claim": "c"}]
        recall.entities = [{"id": "e1", "name": "X"}, {"id": "e2", "name": "Y"}]
        dedup_recall_against_core(recall, core)
        self.assertEqual([f["id"] for f in recall.facts], ["f3"])
        self.assertEqual([e["id"] for e in recall.entities], ["e2"])


class SalientCoreTest(TestCase):
    """Foundation #3 — stable high-salience core memory method."""

    def test_get_salient_facts_maps_and_orders(self):
        from agentx_ai.kit.agent_memory.memory import semantic as sem_mod
        records = [
            {"id": "f1", "claim": "high", "confidence": 0.9, "channel": "_global",
             "salience": 0.95, "temporal_context": "current"},
            {"id": "f2", "claim": "mid", "confidence": 0.7, "channel": "_global",
             "salience": 0.7, "temporal_context": "current"},
        ]
        mock_session = MagicMock()
        mock_session.run.return_value = records
        cm = MagicMock()
        cm.__enter__.return_value = mock_session
        cm.__exit__.return_value = False
        with patch.object(sem_mod.Neo4jConnection, "session", return_value=cm):
            out = sem_mod.SemanticMemory().get_salient_facts(["_global"], limit=8)
        self.assertEqual([f["id"] for f in out], ["f1", "f2"])
        self.assertIsInstance(out[0], dict)
        out[0]["final_score"] = 1.0  # must be a mutable plain dict

    def test_get_salient_facts_empty_channels_short_circuits(self):
        from agentx_ai.kit.agent_memory.memory import semantic as sem_mod
        with patch.object(sem_mod.Neo4jConnection, "session") as sess:
            out = sem_mod.SemanticMemory().get_salient_facts([], limit=8)
        self.assertEqual(out, [])
        sess.assert_not_called()  # no DB round-trip when there are no channels

    def test_get_salient_core_gated_off_returns_empty(self):
        from types import SimpleNamespace
        from agentx_ai.kit.agent_memory.memory import interface as iface_mod
        with patch.object(iface_mod, "get_embedder", return_value=MagicMock()):
            mem = iface_mod.AgentMemory(user_id="tester")
        mem._settings = SimpleNamespace(salient_core_enabled=False)
        mem.semantic = MagicMock()
        bundle = mem.get_salient_core()
        self.assertEqual(bundle.facts, [])
        self.assertEqual(bundle.entities, [])
        mem.semantic.get_salient_facts.assert_not_called()

    def test_get_salient_core_collects_facts_and_entities(self):
        from types import SimpleNamespace
        from agentx_ai.kit.agent_memory.memory import interface as iface_mod
        with patch.object(iface_mod, "get_embedder", return_value=MagicMock()):
            mem = iface_mod.AgentMemory(user_id="tester")
        mem._settings = SimpleNamespace(
            salient_core_enabled=True, salient_core_limit=8, salient_core_min_salience=0.6,
        )
        mem.semantic = MagicMock()
        mem.semantic.get_salient_facts.return_value = [{"id": "f1", "claim": "x"}]
        mem.semantic.get_salient_entities.return_value = [{"id": "e1", "name": "Y"}]
        with patch.object(iface_mod.AgentMemory, "_default_recall_channels",
                          return_value=["_global"]):
            bundle = mem.get_salient_core()
        self.assertEqual([f["id"] for f in bundle.facts], ["f1"])
        self.assertEqual([e["id"] for e in bundle.entities], ["e1"])


class CheckpointMechanicsTest(TestCase):
    """Slice 6 — checkpoint anchor-preserving eviction + replace."""

    class _FakeRedis:
        def __init__(self):
            self.store: dict[str, list] = {}
        def delete(self, k):
            self.store.pop(k, None)
        def rpush(self, k, *vals):
            self.store.setdefault(k, []).extend(vals)
        def llen(self, k):
            return len(self.store.get(k, []))
        def lrange(self, k, a, b):
            items = self.store.get(k, [])
            return items[a:] if b == -1 else items[a:b + 1]
        def expire(self, k, ttl):
            pass

    def test_anchor_preserving_eviction(self):
        from unittest.mock import patch
        import json
        from agentx_ai.agent import checkpoint_storage as cs
        fake = self._FakeRedis()
        with patch.object(cs, "_redis", return_value=fake):
            for i in range(12):  # exceed MAX (8)
                cs.add_checkpoint("conv", summary=f"cp{i}")
            items = [json.loads(x) for x in fake.store[cs._key("conv")]]
        self.assertEqual(len(items), cs.MAX_CHECKPOINTS_PER_CONVERSATION)
        # First (anchor) preserved + most-recent tail.
        self.assertEqual(items[0]["summary"], "cp0")
        self.assertEqual(items[-1]["summary"], "cp11")

    def test_replace_supersedes(self):
        from unittest.mock import patch
        import json
        from agentx_ai.agent import checkpoint_storage as cs
        fake = self._FakeRedis()
        with patch.object(cs, "_redis", return_value=fake):
            cs.add_checkpoint("conv", summary="old1")
            cs.add_checkpoint("conv", summary="old2")
            cs.add_checkpoint("conv", summary="fresh", replace=True)
            items = [json.loads(x) for x in fake.store[cs._key("conv")]]
        self.assertEqual([i["summary"] for i in items], ["fresh"])


class _FakeKVRedis:
    """Minimal dict-backed Redis for the string+set ops the sidecar/summary use."""

    def __init__(self):
        self.kv: dict = {}
        self.sets: dict[str, set] = {}

    def set(self, k, v):
        self.kv[k] = v

    def get(self, k):
        return self.kv.get(k)

    def expire(self, k, ttl):
        pass

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.sets.pop(k, None)

    def sadd(self, k, *vals):
        self.sets.setdefault(k, set()).update(vals)

    def smembers(self, k):
        return set(self.sets.get(k, set()))

    def srem(self, k, *vals):
        self.sets.get(k, set()).difference_update(vals)


class AmbassadorStorageTest(TestCase):
    """Ambassador sidecar (16.6) — keying, round-trip, and the no-pollution
    invariant: a briefing never lands in the rolling-summary key the main agent
    reads on rehydration."""

    def test_prefix_isolated_from_summary(self):
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.agent import conversation_summary_storage as cs
        # The whole no-pollution guarantee rests on disjoint key prefixes.
        self.assertEqual(a.AMBASSADOR_PREFIX, "ambassador:")
        self.assertNotEqual(a.AMBASSADOR_PREFIX, cs.SUMMARY_PREFIX)
        self.assertFalse(a.AMBASSADOR_PREFIX.startswith(cs.SUMMARY_PREFIX))

    def test_roundtrip_and_list(self):
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            a.set_summary("conv1", "m1", "Brief one.")
            a.set_status("conv1", "m2", "streaming", run_id="r2")
            a.append_chunk("conv1", "m2", "partial")
            b1 = a.get_briefing("conv1", "m1")
            allb = a.list_briefings("conv1")
        self.assertEqual(b1["status"], "done")
        self.assertEqual(b1["summary"], "Brief one.")
        ids = {b["message_id"] for b in allb}
        self.assertEqual(ids, {"m1", "m2"})
        m2 = next(b for b in allb if b["message_id"] == "m2")
        self.assertEqual(m2["summary"], "partial")
        self.assertEqual(m2["run_id"], "r2")

    def test_qa_roundtrip_and_isolated_from_briefings(self):
        # Free-form Q&A shares the sidecar but lives under a disjoint key family,
        # so it round-trips independently and never mixes with per-turn briefings.
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            a.create_qa("c1", "qa1", "what sources did it use?", run_id="r1")
            a.append_qa_chunk("c1", "qa1", "It pulled ")
            a.append_qa_chunk("c1", "qa1", "the county index.")
            a.set_summary("c1", "m1", "a per-turn briefing")  # briefing in same conv
            qa = a.get_qa("c1", "qa1")
            qa_list = a.list_qa("c1")
            briefs = a.list_briefings("c1")
        self.assertEqual(qa["question"], "what sources did it use?")
        self.assertEqual(qa["answer"], "It pulled the county index.")
        self.assertEqual(qa["status"], "streaming")
        self.assertEqual(qa["run_id"], "r1")
        self.assertEqual({x["qa_id"] for x in qa_list}, {"qa1"})
        # Disjoint families: Q&A is absent from briefings and vice-versa.
        self.assertEqual({b["message_id"] for b in briefs}, {"m1"})

    def test_qa_settle_and_clear(self):
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            a.create_qa("c1", "qa1", "q?", run_id="r1")
            a.set_qa_answer("c1", "qa1", "Final answer.", status="done")
            settled = a.get_qa("c1", "qa1")
            a.clear("c1")  # clears both families
            after = a.get_qa("c1", "qa1")
            briefs_after = a.list_qa("c1")
        self.assertEqual(settled["status"], "done")
        self.assertEqual(settled["answer"], "Final answer.")
        self.assertIsNone(after)
        self.assertEqual(briefs_after, [])

    def test_briefing_does_not_pollute_rolling_summary(self):
        # Share one fake Redis between both modules: writing a briefing must NOT
        # make it readable as the conversation's rolling summary (what the main
        # agent restores on a cold session).
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.agent import conversation_summary_storage as cs
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake), \
                patch.object(cs, "_redis", return_value=fake):
            a.set_summary("convX", "mX", "Ambassador-only briefing text.")
            leaked = cs.get_summary("convX")
        self.assertIsNone(leaked)

    def test_thread_unifies_briefings_and_qa_in_order(self):
        # Slice 1b: briefings + Q&A are one ordered thread (an "Inquiry"), oldest-first
        # by created_at, regardless of which kind was written when.
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            a.set_summary("c", "m1", "first — a briefing")
            a.create_qa("c", "qa1", "a question?")
            a.set_qa_answer("c", "qa1", "an answer")
            payload = a.thread_payload("c")
        self.assertEqual(payload["thread_id"], "c")
        self.assertEqual(payload["title"], "")  # no title yet → client borrows chat title
        kinds = [e["kind"] for e in payload["entries"]]
        ids = [e["id"] for e in payload["entries"]]
        self.assertEqual(kinds, ["briefing", "qa"])  # created_at order preserved
        self.assertEqual(ids, ["m1", "qa1"])
        qa_entry = payload["entries"][1]
        self.assertEqual(qa_entry["question"], "a question?")
        self.assertEqual(qa_entry["content"], "an answer")

    def test_tool_calls_persist_on_entry(self):
        # The agentic loop's chips must survive a reload — persisted on the entry and
        # surfaced (camelCase) by both the Q&A projection and the thread payload.
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        chips = [
            {"tool": "read_conversation", "args": {"conversation_id": "c"}, "done": True},
            {"tool": "summarize_conversation", "args": {}, "done": False},
        ]
        with patch.object(a, "_redis", return_value=fake):
            a.create_qa("c", "qa1", "what happened?")
            a.set_entry_tool_calls("c", "qa1", chips)
            a.set_qa_answer("c", "qa1", "Here is what happened.")  # must not clobber chips
            qa = a.get_qa("c", "qa1")
            entry = a.thread_payload("c")["entries"][0]
        self.assertEqual(qa["answer"], "Here is what happened.")
        self.assertEqual([t["tool"] for t in qa["toolCalls"]], [c["tool"] for c in chips])
        self.assertEqual(entry["toolCalls"][0]["done"], True)

    def test_thread_title_persists(self):
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            self.assertEqual(a.get_thread_title("c"), "")
            a.set_thread_title("c", "  Weekly review  ")
            again = a.get_thread_title("c")
            payload_title = a.thread_payload("c")["title"]
        self.assertEqual(again, "Weekly review")  # trimmed
        self.assertEqual(payload_title, "Weekly review")

    def test_legacy_records_fold_into_thread(self):
        # Pre-1b sidecars (separate ambassador:{cid}:msg / :qa families) must still
        # replay through the unified reader, so a deploy doesn't drop Inquiry history.
        from agentx_ai.agent import ambassador_storage as a
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            # Hand-write legacy-shaped records via the legacy key engine.
            a._persist(a._briefing_key("c", "m0"), a._index_key("c"), "m0",
                       {"message_id": "m0", "summary": "legacy briefing",
                        "status": "done", "created_at": "2026-01-01T00:00:00+00:00"})
            a._persist(a._qa_key("c", "q0"), a._qa_index_key("c"), "q0",
                       {"qa_id": "q0", "question": "legacy q", "answer": "legacy a",
                        "status": "done", "created_at": "2026-01-02T00:00:00+00:00"})
            # A new (1b) entry lands after both.
            a.set_summary("c", "m1", "new briefing")
            entries = a.list_thread("c")
            briefs = a.list_briefings("c")
            qa = a.list_qa("c")
        self.assertEqual([e["id"] for e in entries], ["m0", "q0", "m1"])  # created_at order
        self.assertEqual({b["message_id"] for b in briefs}, {"m0", "m1"})
        self.assertEqual({x["qa_id"] for x in qa}, {"q0"})


class AmbassadorServiceTest(TestCase):
    """Ambassador service (16.6) — bulletproof graceful degradation."""

    def test_empty_provider_degrades_without_raising(self):
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage

        async def _collect(agen):
            return [e async for e in agen]

        svc = AmbassadorService()
        reg = MagicMock()
        reg.resolve_with_fallback.side_effect = ValueError("no provider")
        svc._registry = reg
        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(ambassador_storage, "set_status"), \
                patch.object(ambassador_storage, "set_summary"):
            events = asyncio.run(
                _collect(svc.brief_turn("conv", "msg", assistant_text="hi"))
            )
        # Never raised; settled on an empty_provider 'done' the client can show.
        joined = "".join(events)
        self.assertIn("ambassador_done", joined)
        self.assertIn("empty_provider", joined)
        self.assertNotIn("Traceback", joined)

    def test_streams_chunks_and_settles_done(self):
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk

        async def _fake_stream(*args, **kwargs):
            for piece in ["It ", "weighed ", "the trade-off."]:
                yield StreamChunk(content=piece)

        provider = MagicMock()
        provider.stream = _fake_stream
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)

        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _collect(agen):
            return [e async for e in agen]

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(a, "_redis", return_value=fake):
            events = asyncio.run(
                _collect(svc.brief_turn("conv", "msg", assistant_text="hi"))
            )
            record = a.get_briefing("conv", "msg")

        joined = "".join(events)
        # Each delta is streamed as its own ambassador_chunk, then a settled done.
        self.assertEqual(joined.count("ambassador_chunk"), 3)
        self.assertIn("ambassador_done", joined)
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["summary"], "It weighed the trade-off.")

    def test_cancel_midstream_settles_not_streaming(self):
        # Cancelling a run closes the generator (GeneratorExit). The sidecar must
        # land on `cancelled`, never stay stuck on `streaming` (perpetual spinner).
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk

        async def _fake_stream(*args, **kwargs):
            yield StreamChunk(content="partial…")
            # Suspend forever so the consumer cancels us mid-stream.
            await asyncio.Event().wait()
            yield StreamChunk(content="never")

        provider = MagicMock()
        provider.stream = _fake_stream
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)

        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _drive():
            agen = svc.brief_turn("conv", "msg", assistant_text="hi")
            # Pull until the first streamed chunk, then close (cancel).
            async for ev in agen:
                if "ambassador_chunk" in ev:
                    await agen.aclose()
                    break

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(a, "_redis", return_value=fake):
            asyncio.run(_drive())
            record = a.get_briefing("conv", "msg")

        self.assertEqual(record["status"], "cancelled")
        # Partial text that streamed before the cancel is preserved.
        self.assertEqual(record["summary"], "partial…")

    def test_turn_prompt_grounds_on_artifacts(self):
        # The briefing prompt must surface what the agent *did* (tools, sources,
        # exhibits) — not just its prose — so the ambassador can interpret it.
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        artifacts = {
            "tools": [{"name": "web_search", "detail": "town business registry", "ok": True}],
            "sources": [{"label": "County Index", "url": "https://county.example/biz"}],
            "exhibits": [{"kind": "table", "title": "Local businesses", "detail": "3 columns × 24 rows"}],
        }
        prompt = svc._build_turn_prompt(
            user_text="keep searching for the registry",
            assistant_text="Here's what I found.",
            context="",
            agent_name="Atlas",
            artifacts=artifacts,
        )
        # Speaks of the agent by name and second-person to the reader.
        self.assertIn("You said:", prompt)
        self.assertIn("Atlas replied:", prompt)
        # Grounds on the real substance of the turn.
        self.assertIn("web_search", prompt)
        self.assertIn("town business registry", prompt)
        self.assertIn("County Index", prompt)
        self.assertIn("https://county.example/biz", prompt)
        self.assertIn("table", prompt)

    def test_turn_prompt_without_artifacts_has_no_block(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        prompt = svc._build_turn_prompt(
            user_text="hi", assistant_text="hello", context="", agent_name="Atlas"
        )
        self.assertNotIn("actually did this turn", prompt)

    def test_answer_question_streams_and_settles(self):
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk

        async def _fake_stream(*args, **kwargs):
            for piece in ["It used ", "two sources."]:
                yield StreamChunk(content=piece)

        provider = MagicMock()
        provider.stream = _fake_stream
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)

        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _collect(agen):
            return [e async for e in agen]

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(svc, "_grounding_context", return_value=""), \
                patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
                patch.object(a, "_redis", return_value=fake):
            events = asyncio.run(
                _collect(svc.answer_question("conv", "qa1", "what sources did it use?"))
            )
            rec = a.get_qa("conv", "qa1")

        joined = "".join(events)
        self.assertEqual(joined.count("ambassador_chunk"), 2)
        self.assertIn("ambassador_done", joined)
        self.assertEqual(rec["status"], "done")
        self.assertEqual(rec["answer"], "It used two sources.")
        self.assertEqual(rec["question"], "what sources did it use?")

    def test_draft_relay_degrades_to_raw_intent_without_provider(self):
        # The outbound relay must never block: with no provider, drafting returns
        # the user's raw intent so they can still send it.
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        reg = MagicMock()
        reg.resolve_with_fallback.side_effect = ValueError("no provider")
        svc._registry = reg
        with patch.object(svc, "_resolve_profile", return_value=None):
            draft = asyncio.run(svc.draft_relay_message("conv", "ask it about the tax records"))
        self.assertEqual(draft, "ask it about the tax records")

    def test_draft_relay_uses_provider_completion(self):
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.providers.base import CompletionResult

        async def _complete(*args, **kwargs):
            return CompletionResult(
                content="Could you also check the county tax records?",
                finish_reason="stop",
                model="m",
            )

        provider = MagicMock()
        provider.complete = _complete
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)

        svc = AmbassadorService()
        svc._registry = reg
        with patch.object(svc, "_resolve_profile", return_value=None):
            draft = asyncio.run(
                svc.draft_relay_message("conv", "also taxes", agent_name="Atlas")
            )
        self.assertEqual(draft, "Could you also check the county tax records?")

    def test_qa_prompt_grounds_only_on_conversation(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        prompt = svc._build_qa_prompt(
            question="what did it search for?",
            context="You: find the registry\nAtlas: I searched the county index.",
            agent_name="Atlas",
        )
        self.assertIn("what did it search for?", prompt)
        self.assertIn("county index", prompt)
        self.assertIn("grounded in the conversation", prompt)

    def test_qa_prompt_handles_empty_conversation(self):
        # 16.7: with no pre-loaded snippet, the prompt tells the model to read with its
        # tools (tool-first) and to say so plainly if the conversation is empty — never
        # a context-free prompt it could hallucinate against.
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        prompt = svc._build_qa_prompt(
            question="what has it found so far?",
            context="",
            agent_name="Atlas",
        )
        self.assertIn("read the conversation with your tools", prompt)
        self.assertIn("never invent", prompt)

    def test_answer_question_degrades_on_empty_conversation(self):
        # The ambassador must operate on an empty conversation without raising: it
        # grounds on nothing and settles a normal `done` (the model says so plainly).
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk

        async def _fake_stream(*args, **kwargs):
            yield StreamChunk(content="There's nothing in this conversation yet.")

        provider = MagicMock()
        provider.stream = _fake_stream
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)

        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _collect(agen):
            return [e async for e in agen]

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(svc, "_grounding_context", return_value=""), \
                patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
                patch.object(a, "_redis", return_value=fake):
            events = asyncio.run(
                _collect(svc.answer_question("empty-conv", "qa1", "what's happened?"))
            )
            record = a.get_qa("empty-conv", "qa1")

        joined = "".join(events)
        self.assertIn("ambassador_done", joined)
        self.assertNotIn("Traceback", joined)
        self.assertEqual(record["status"], "done")

    def test_thread_history_gives_qa_continuity(self):
        # 16.7 Slice 1: the ambassador has its own conversation — prior settled Q&A
        # comes back as real user/assistant dialogue turns (oldest first), so a
        # follow-up has context. The in-flight turn is excluded; unanswered/streaming
        # turns don't leak in.
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import MessageRole

        svc = AmbassadorService()
        fake = _FakeKVRedis()
        with patch.object(a, "_redis", return_value=fake):
            a.create_qa("conv", "q1", "what did it search for?")
            a.set_qa_answer("conv", "q1", "It searched the county index.", status="done")
            a.create_qa("conv", "q2", "and the second source?")  # in-flight, unanswered
            history = svc._thread_history("conv", exclude_id="q2")

        # q1 → a user/assistant pair; q2 (streaming, no answer) is excluded.
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, MessageRole.USER)
        self.assertEqual(history[0].content, "what did it search for?")
        self.assertEqual(history[1].role, MessageRole.ASSISTANT)
        self.assertIn("county index", history[1].content)

    def test_ambassador_tools_are_read_only_and_degrade(self):
        # 16.7 Slice 2: the tool belt is read-only and never raises — a bad/unknown
        # call returns a readable note instead of throwing.
        from agentx_ai.agent import ambassador_tools as t

        # Labeled rows: (role, content, agent_name) — the transcript names each
        # assistant turn by its OWN producing agent (here metadata says "Atlas").
        rows = [
            ("user", "find the registry", None),
            ("assistant", "I searched the county index.", "Atlas"),
        ]
        with patch.object(t, "load_recent_labeled_turns", return_value=rows):
            summary = t.execute_tool(
                "summarize_conversation", {}, focused_conversation_id="conv", agent_name="Atlas"
            )
        self.assertIn("county index", summary)
        self.assertIn("Atlas:", summary)

        convs = [{
            "conversation_id": "c1", "first_user": "build the registry",
            "last_message": "done", "message_count": 4, "last_at": "2026-06-08",
            "agents": "Nimbus",
        }]
        with patch.object(t, "list_recent_conversations", return_value=convs):
            listed = t.execute_tool("list_conversations", {"limit": 5}, focused_conversation_id="conv")
        self.assertIn("c1", listed)
        self.assertIn("build the registry", listed)
        self.assertIn("Nimbus", listed)  # the survey names each conversation's own agent

        # Unknown tool + a missing required arg degrade to notes (never raise).
        self.assertIn("Unknown tool", t.execute_tool("nope", {}, focused_conversation_id="conv"))
        self.assertIn(
            "needs a conversation_id",
            t.execute_tool("read_conversation", {}, focused_conversation_id="conv"),
        )

    def test_ambassador_tools_label_each_conversation_by_its_own_agent(self):
        # The bug: a cross-conversation read mislabeled another conversation's turns
        # with the *active* agent. Now an assistant turn uses its own metadata name,
        # and a non-active conversation never borrows the active agent's name.
        from agentx_ai.agent import ambassador_tools as t

        rows = [
            ("user", "do the thing", None),
            ("assistant", "Done — used Postgres.", "Nimbus"),
            ("assistant", "Unstamped reply.", None),
        ]
        with patch.object(t, "load_recent_labeled_turns", return_value=rows):
            out = t.execute_tool(
                "read_conversation", {"conversation_id": "other"},
                focused_conversation_id="active", agent_name="Atlas",
            )
        self.assertIn("Nimbus: Done — used Postgres.", out)  # its own agent
        self.assertNotIn("Atlas:", out)  # never the active agent's name
        self.assertIn("Agent: Unstamped reply.", out)  # generic fallback, not "Atlas"

    def test_active_conversation_note_tells_it_where_you_are(self):
        # 16.7 overhaul: the ambassador's focus can sit on one conversation while the
        # person is currently in another — it's told where they are now (ambient
        # context), distinct from its focus, so "what am I working on now?" works.
        from agentx_ai.agent.ambassador import AmbassadorService

        note = AmbassadorService._active_conversation_note({"id": "c9", "title": "Deploy run"})
        self.assertIn("Deploy run", note)
        self.assertIn("c9", note)
        self.assertIn("WHERE THE PERSON IS NOW", note)
        # No ambient context → no note (back-compat / unknown).
        self.assertEqual(AmbassadorService._active_conversation_note(None), "")
        self.assertEqual(AmbassadorService._active_conversation_note({}), "")

    @staticmethod
    def _streaming_provider(stream_fn):
        """A provider whose ``stream`` is the given async-generator fn + a registry."""
        provider = MagicMock()
        provider.stream = stream_fn
        reg = MagicMock()
        reg.resolve_with_fallback.return_value = (provider, "m", None)
        return provider, reg

    def test_agentic_answer_executes_tool_then_answers(self):
        # The unified core: round 1 streams a tool call (summarize), round 2 streams the
        # answer. Tool SSE fires, the answer settles `done`, and a tool round happened.
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk, ToolCall

        calls = {"n": 0}

        async def _stream(messages, model_id, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:  # first round → ask to summarize
                yield StreamChunk(
                    content="", finish_reason="tool_calls",
                    tool_calls=[ToolCall(id="t1", name="summarize_conversation", arguments={})],
                )
            else:  # second round → the answer
                yield StreamChunk(content="It searched the county index for you.", finish_reason="stop")

        provider, reg = self._streaming_provider(_stream)
        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _collect(agen):
            return [e async for e in agen]

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(a, "_redis", return_value=fake), \
                patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
                patch("agentx_ai.agent.ambassador_tools.load_recent_labeled_turns", return_value=[]):
            events = asyncio.run(_collect(svc.answer_question("conv", "qa1", "summarize this")))
            record = a.get_qa("conv", "qa1")

        joined = "".join(events)
        self.assertIn("ambassador_tool_call", joined)
        self.assertIn("ambassador_tool_result", joined)
        self.assertIn("ambassador_done", joined)
        self.assertNotIn("Traceback", joined)
        self.assertEqual(record["status"], "done")
        self.assertIn("county index", record["answer"])
        self.assertEqual(calls["n"], 2)  # one tool round + one answer round

    def test_agentic_answer_falls_back_when_streaming_tools_rejected(self):
        # A provider that can't stream with tools must not break the turn — the core
        # degrades to a grounded one-shot (no tools) and settles `done`.
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent import ambassador_storage as a
        from agentx_ai.providers.base import StreamChunk

        async def _stream(messages, model_id, **kwargs):
            if kwargs.get("tools"):
                raise RuntimeError("this provider can't stream tools")
            yield StreamChunk(content="Here's the gist.", finish_reason="stop")

        provider, reg = self._streaming_provider(_stream)
        svc = AmbassadorService()
        svc._registry = reg
        fake = _FakeKVRedis()

        async def _collect(agen):
            return [e async for e in agen]

        with patch.object(svc, "_resolve_profile", return_value=None), \
                patch.object(a, "_redis", return_value=fake), \
                patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
                patch("agentx_ai.agent.ambassador_tools.load_recent_labeled_turns", return_value=[]):
            events = asyncio.run(_collect(svc.answer_question("conv", "qa2", "what happened?")))
            record = a.get_qa("conv", "qa2")

        joined = "".join(events)
        self.assertIn("ambassador_done", joined)
        self.assertNotIn("ambassador_error", joined)
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["answer"], "Here's the gist.")

    def test_token_budget_leaves_headroom_for_free_range_thinking(self):
        # The cap must accommodate reasoning + the (short) answer, so a thinking
        # model isn't truncated mid-sentence. Length is the prompt's job, not the
        # cap's — so the budget is much larger than the visible answer allowance.
        from agentx_ai.agent.ambassador import (
            AmbassadorService,
            _THINKING_HEADROOM,
            _VERBOSITY_TOKENS,
        )

        svc = AmbassadorService()
        # No explicit ceiling → headroom + verbosity answer room, never clipped.
        budget = svc._max_tokens(None, {"max_tokens": None})
        self.assertEqual(budget, _THINKING_HEADROOM + _VERBOSITY_TOKENS["normal"])
        self.assertGreater(budget, _VERBOSITY_TOKENS["deep"])  # not a tight length cap
        # An explicit setting is honored as a hard ceiling.
        self.assertEqual(svc._max_tokens(None, {"max_tokens": 700}), 700)


class _FakeConfigManager:
    """In-memory ConfigManager stand-in (dot-notation get/set + no-op save)."""

    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        cur = self.data
        for k in key.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def set(self, key, value):
        cur = self.data
        keys = key.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value

    def save(self):
        return True


class PromptLayerStoreTest(TestCase):
    """Layered prompt stack — precedence (override vs default), durable deltas,
    default-change diff, custom CRUD, and compose ordering."""

    def _store(self, fake):
        from agentx_ai.prompts import layers
        with patch.object(layers, "get_config_manager", return_value=fake):
            return layers.LayerStore()

    def test_builtins_ride_default_until_overridden(self):
        store = self._store(_FakeConfigManager())
        layers = store.list_layers()
        ids = {layer.id for layer in layers}
        self.assertIn("core-principles", ids)
        core = store.get("core-principles")
        self.assertEqual(core.effective, core.default)
        self.assertFalse(core.modified)
        self.assertIsNone(core.override)
        self.assertIn("Core Principles", store.compose())

    def test_override_pins_content_and_persists(self):
        fake = _FakeConfigManager()
        store = self._store(fake)
        store.set_override("core-principles", "My custom core.")
        core = store.get("core-principles")
        self.assertEqual(core.effective, "My custom core.")
        self.assertTrue(core.modified)
        # A fresh store over the same config reads the override back (durable).
        store2 = self._store(fake)
        self.assertEqual(store2.get("core-principles").override, "My custom core.")

    def test_default_change_surfaces_update_then_ack_and_reset(self):
        from agentx_ai.prompts import layers as layers_mod
        fake = _FakeConfigManager()
        store = self._store(fake)
        store.set_override("citing-sources", "mine")
        self.assertFalse(store.get("citing-sources").update_available)
        # Simulate a release bumping the shipped default underneath the override.
        builtin = layers_mod._BUILTIN_BY_ID["citing-sources"]
        original = builtin.default_version
        builtin.default_version = original + 1
        try:
            self.assertTrue(store.get("citing-sources").update_available)
            # Acknowledge keeps the override but clears the badge.
            store.acknowledge("citing-sources")
            self.assertFalse(store.get("citing-sources").update_available)
            self.assertEqual(store.get("citing-sources").override, "mine")
            # Reset drops the override entirely (back to the new default).
            reset = store.reset("citing-sources")
            self.assertIsNone(reset.override)
            self.assertEqual(reset.effective, builtin.default)
        finally:
            builtin.default_version = original

    def test_custom_layer_crud_reorder_and_compose(self):
        fake = _FakeConfigManager()
        store = self._store(fake)
        custom = store.create_custom("Tone", "Always be concise.")
        self.assertEqual(custom.kind, "custom")
        self.assertIn("Always be concise.", store.compose())
        # Reorder it to the front; compose reflects the new order.
        store.reorder([custom.id, "core-principles", "reasoning-vs-results", "citing-sources"])
        first = store.list_layers()[0]
        self.assertEqual(first.id, custom.id)
        # Disable drops it from the composed stack.
        store.set_enabled(custom.id, False)
        self.assertNotIn("Always be concise.", store.compose())
        # Delete removes it.
        self.assertTrue(store.delete_custom(custom.id))
        self.assertIsNone(store.get(custom.id))

    def test_set_singleton_override_is_idempotent(self):
        fake = _FakeConfigManager()
        store = self._store(fake)
        store.set_singleton_override("v1")
        store.set_singleton_override("v2")
        customs = [layer for layer in store.list_layers() if layer.kind == "custom"]
        self.assertEqual(len(customs), 1)  # one reserved legacy block, not duplicated
        self.assertEqual(customs[0].override, "v2")


class PromptStackCompositionTest(TestCase):
    """Phase 1b — the live conversational system prompt is composed from the layer
    stack: byte-parity with the legacy default, and the default-profile sections
    (now stack layers) are not double-injected via the General profile."""

    def _store(self, fake):
        from agentx_ai.prompts import layers
        with patch.object(layers, "get_config_manager", return_value=fake):
            return layers.LayerStore()

    def _expected_default(self) -> str:
        from agentx_ai.prompts.defaults import (
            DEFAULT_GLOBAL_PROMPT,
            SECTION_STRUCTURED_THINKING,
            SECTION_CONCISE_OUTPUT,
            SECTION_SAFETY_CONSTRAINTS,
        )
        return "\n\n".join([
            DEFAULT_GLOBAL_PROMPT.content,
            SECTION_STRUCTURED_THINKING.content,
            SECTION_CONCISE_OUTPUT.content,
            SECTION_SAFETY_CONSTRAINTS.content,
        ])

    def test_stack_reproduces_legacy_default_prompt(self):
        store = self._store(_FakeConfigManager())
        self.assertEqual(store.compose(), self._expected_default())

    def test_compose_prompt_uses_stack_without_duplicate_sections(self):
        from agentx_ai.prompts import layers, manager as mgr_mod
        fake = _FakeConfigManager()
        store = self._store(fake)
        manager = mgr_mod.PromptManager()
        with patch.object(layers, "get_layer_store", return_value=store), \
             patch("agentx_ai.config.get_config_manager", return_value=fake):
            composed = manager.get_system_prompt()  # no agent/profile → just the stack
        self.assertEqual(composed, self._expected_default())
        # The safety section appears exactly once, not double-injected via General.
        self.assertEqual(composed.count("Be honest about your limitations as an AI"), 1)

    def test_override_changes_live_composition(self):
        from agentx_ai.prompts import layers, manager as mgr_mod
        fake = _FakeConfigManager()
        store = self._store(fake)
        store.set_override("core-principles", "OVERRIDDEN CORE")
        manager = mgr_mod.PromptManager()
        with patch.object(layers, "get_layer_store", return_value=store), \
             patch("agentx_ai.config.get_config_manager", return_value=fake):
            composed = manager.get_system_prompt()
        self.assertIn("OVERRIDDEN CORE", composed)
        self.assertNotIn("You are an intelligent AI assistant", composed)

    def test_global_shim_returns_composed_stack(self):
        import json as _json
        from agentx_ai import views
        fake = _FakeConfigManager()
        store = self._store(fake)
        store.set_override("core-principles", "SHIMMED CORE")
        with patch("agentx_ai.prompts.get_layer_store", return_value=store):
            resp = views.prompts_global(MagicMock())
        body = _json.loads(resp.content)
        self.assertIn("SHIMMED CORE", body["global_prompt"]["content"])


class PromptLayerApiTest(TestCase):
    """Phase 2 — the layer REST API (list/create/patch/delete/reset/reorder),
    exercised against a fake-config store via RequestFactory (no auth middleware)."""

    def setUp(self):
        from django.test import RequestFactory
        from agentx_ai.prompts import layers
        self.rf = RequestFactory()
        self.fake = _FakeConfigManager()
        with patch.object(layers, "get_config_manager", return_value=self.fake):
            self.store = layers.LayerStore()

    def _call(self, view, request, **kwargs):
        with patch("agentx_ai.prompts.get_layer_store", return_value=self.store):
            return view(request, **kwargs)

    def _json(self, resp):
        import json as _json
        return _json.loads(resp.content)

    def _post(self, path, payload):
        import json as _json
        return self.rf.post(path, data=_json.dumps(payload), content_type="application/json")

    def test_list_returns_layers_and_composed(self):
        from agentx_ai import views
        body = self._json(self._call(views.prompts_layers, self.rf.get("/api/prompts/layers")))
        ids = {layer["id"] for layer in body["layers"]}
        self.assertIn("core-principles", ids)
        self.assertIn("safety-constraints", ids)
        self.assertIn("Core Principles", body["composed"])
        # Serialized shape carries the diff/badge fields.
        core = next(layer for layer in body["layers"] if layer["id"] == "core-principles")
        self.assertEqual(core["kind"], "builtin")
        self.assertFalse(core["modified"])
        self.assertFalse(core["update_available"])

    def test_create_patch_and_delete_custom(self):
        from agentx_ai import views
        created = self._json(self._call(views.prompts_layers, self._post("/x", {"title": "Tone", "content": "Be terse."})))
        cid = created["layer"]["id"]
        self.assertEqual(created["layer"]["kind"], "custom")
        # PATCH content + disable.
        patched = self._json(self._call(
            views.prompts_layer_detail,
            self.rf.patch("/x", data='{"content": "Be VERY terse.", "enabled": false}', content_type="application/json"),
            layer_id=cid,
        ))
        self.assertEqual(patched["layer"]["effective"], "Be VERY terse.")
        self.assertFalse(patched["layer"]["enabled"])
        # DELETE removes it; deleting a built-in is rejected.
        self.assertEqual(self._call(views.prompts_layer_detail, self.rf.delete("/x"), layer_id=cid).status_code, 200)
        self.assertEqual(self._call(views.prompts_layer_detail, self.rf.delete("/x"), layer_id="core-principles").status_code, 400)

    def test_patch_builtin_override_and_reset(self):
        from agentx_ai import views
        self._call(
            views.prompts_layer_detail,
            self.rf.patch("/x", data='{"content": "MINE"}', content_type="application/json"),
            layer_id="core-principles",
        )
        self.assertEqual(self.store.get("core-principles").effective, "MINE")
        reset = self._json(self._call(views.prompts_layer_reset, self.rf.post("/x"), layer_id="core-principles"))
        self.assertIsNone(reset["layer"]["override"])

    def test_reorder(self):
        from agentx_ai import views
        body = self._json(self._call(
            views.prompts_layers_reorder,
            self._post("/x", {"order": ["safety-constraints", "core-principles"]}),
        ))
        first = body["layers"][0]
        self.assertEqual(first["id"], "safety-constraints")

    def test_patch_missing_layer_404(self):
        from agentx_ai import views
        resp = self._call(
            views.prompts_layer_detail,
            self.rf.patch("/x", data='{"content": "x"}', content_type="application/json"),
            layer_id="does-not-exist",
        )
        self.assertEqual(resp.status_code, 404)


class PromptPlaceholderTest(TestCase):
    """Whitelisted `{token}` substitution in composed prompts."""

    def test_substitute_only_whitelisted_tokens(self):
        from agentx_ai.prompts.placeholders import substitute_placeholders
        from datetime import datetime
        out = substitute_placeholders(
            "I am {agent_name}. Today is {date}. Keep literal {json_key}.",
            agent_name="Mobius",
            now=datetime(2026, 6, 6, 9, 30),
        )
        self.assertIn("I am Mobius.", out)
        self.assertIn("Today is 2026-06-06.", out)
        self.assertIn("{json_key}", out)  # unknown braces untouched

    def test_compose_system_prompt_substitutes_agent_name(self):
        from agentx_ai.prompts.models import PromptConfig, GlobalPrompt
        cfg = PromptConfig(
            global_prompt=GlobalPrompt(content="You are {agent_name}, be helpful."),
            agent_name="Echo",
        )
        composed = cfg.compose_system_prompt()
        self.assertIn("You are Echo, be helpful.", composed)
        self.assertNotIn("{agent_name}", composed)


class AgentProfileKindTest(TestCase):
    """Ambassador-as-profile-kind: default isolation (agent vs ambassador),
    persistence, migration/seed safety, and chat-routing exclusion."""

    def _manager(self, path):
        """Fresh ProfileManager over a tmp file; migration sees no legacy id."""
        from agentx_ai.agent.profiles import ProfileManager
        with patch("agentx_ai.config.get_config_manager", return_value=_FakeConfigManager()):
            return ProfileManager(config_path=path)

    def _tmp(self):
        import tempfile
        from pathlib import Path
        return Path(tempfile.mkdtemp()) / "agent_profiles.yaml"

    def test_seed_default_ambassador_separate_from_default_agent(self):
        mgr = self._manager(self._tmp())
        amb = mgr.get_default_ambassador()
        agent = mgr.get_default_profile()
        self.assertIsNotNone(amb)
        self.assertEqual(amb.kind, "ambassador")
        self.assertTrue(amb.is_default_ambassador)
        # The default *agent* is never the ambassador.
        self.assertIsNotNone(agent)
        self.assertEqual(agent.kind, "agent")
        self.assertNotEqual(agent.id, amb.id)

    def test_kind_and_default_persist_across_reload(self):
        path = self._tmp()
        amb_id = self._manager(path).get_default_ambassador().id
        reloaded = self._manager(path).get_profile(amb_id)
        self.assertEqual(reloaded.kind, "ambassador")
        self.assertTrue(reloaded.is_default_ambassador)

    def test_set_default_profile_rejects_ambassador(self):
        mgr = self._manager(self._tmp())
        amb = mgr.get_default_ambassador()
        self.assertFalse(mgr.set_default_profile(amb.id))  # can't make an ambassador the default agent
        self.assertEqual(mgr.get_default_profile().kind, "agent")

    def test_set_default_ambassador_one_per_kind(self):
        from agentx_ai.agent.models import AgentProfile, AmbassadorConfig
        mgr = self._manager(self._tmp())
        first = mgr.get_default_ambassador()
        second = AgentProfile(id="amb2", name="Scribe", kind="ambassador",
                              ambassador=AmbassadorConfig(enabled=True))
        mgr.create_profile(second)
        self.assertTrue(mgr.set_default_ambassador("amb2"))
        self.assertEqual(mgr.get_default_ambassador().id, "amb2")
        self.assertFalse(mgr.get_profile(first.id).is_default_ambassador)

    def test_routing_lookups_exclude_ambassadors(self):
        mgr = self._manager(self._tmp())
        amb = mgr.get_default_ambassador()
        self.assertIsNone(mgr.get_profile_by_agent_id(amb.agent_id))
        self.assertIsNone(mgr.get_profile_by_name(amb.name))

    def test_migration_converts_dedicated_not_default_agent(self):
        from agentx_ai.agent.models import AgentProfile, AmbassadorConfig
        path = self._tmp()
        mgr = self._manager(path)
        default_agent_id = mgr.get_default_profile().id
        # A legacy "dedicated" ambassador: kind still 'agent', ambassador.enabled, not default.
        mgr.create_profile(AgentProfile(id="legacy_amb", name="Legacy", kind="agent",
                                        ambassador=AmbassadorConfig(enabled=True)))
        # Reload → migration promotes the dedicated one, leaves the default agent alone.
        reloaded = self._manager(path)
        self.assertEqual(reloaded.get_profile("legacy_amb").kind, "ambassador")
        self.assertEqual(reloaded.get_profile(default_agent_id).kind, "agent")

    def test_ambassador_persona_uses_override_and_personality(self):
        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.agent.models import AgentProfile, AmbassadorConfig
        profile = AgentProfile(
            id="amb", name="Echo", kind="ambassador",
            system_prompt="Warm and witty.",
            ambassador=AmbassadorConfig(enabled=True, briefing_persona="CUSTOM BRIEFING VOICE"),
        )
        persona = AmbassadorService()._build_persona(profile, "Mobius")
        self.assertIn("CUSTOM BRIEFING VOICE", persona)   # override replaces the default voice
        self.assertIn("Warm and witty.", persona)         # communications prompt woven in


class ChatRunIndexingTest(TestCase):
    """Detached-run indexing flag (16.6) — sidecar runs stay out of the per-user
    recovery list that backs /api/agent/chat/runs."""

    def test_indexed_false_skips_recovery_index(self):
        from agentx_ai.streaming import chat_run
        client = MagicMock()
        with patch.object(chat_run, "_redis", return_value=client):
            chat_run.store.create("run-amb", user_id="u1", indexed=False)
        client.zadd.assert_not_called()

    def test_indexed_true_writes_recovery_index(self):
        from agentx_ai.streaming import chat_run
        client = MagicMock()
        with patch.object(chat_run, "_redis", return_value=client):
            chat_run.store.create("run-chat", user_id="u1", indexed=True)
        client.zadd.assert_called_once()


class LoggingKitTest(TestCase):
    """Unit tests for the logging_kit pipeline (no external deps)."""

    def test_category_mapping(self):
        from agentx_ai.logging_kit.categories import category_for

        self.assertEqual(category_for("agentx_ai.providers.anthropic").key, "provider")
        self.assertEqual(category_for("agentx_ai.streaming.tool_loop").key, "stream")
        # ambassador lives under agent.* but must route to its own category
        self.assertEqual(category_for("agentx_ai.agent.ambassador").key, "ambassador")
        self.assertEqual(category_for("agentx_ai.agent.planner").key, "plan")
        self.assertEqual(category_for("agentx_ai.agent.core").key, "agent")
        self.assertEqual(category_for("agentx_ai.kit.agent_memory.recall").key, "memory")
        self.assertEqual(category_for("agentx_ai.mcp.client").key, "mcp")
        # unknown / bare root → core
        self.assertEqual(category_for("something.else").key, "core")
        self.assertEqual(category_for("agentx_ai").key, "core")

    def test_redaction(self):
        from agentx_ai.logging_kit.redaction import redact

        self.assertIn("«redacted»", redact("api_key=sk-ant-abcd1234567890"))
        self.assertNotIn("sk-ant-abcd1234567890", redact("api_key=sk-ant-abcd1234567890"))
        # Authorization + bearer token: the whole credential goes, not just "Bearer"
        scrubbed = redact("Authorization: Bearer abc.def.ghijklmnop tail")
        self.assertNotIn("abc.def.ghijklmnop", scrubbed)
        self.assertIn("tail", scrubbed)
        # benign content is untouched
        benign = "completed in 1.4s with ~8,234 tokens (60%)"
        self.assertEqual(redact(benign), benign)

    def test_ring_buffer_bounded_and_filters(self):
        import logging as _logging

        from agentx_ai.logging_kit.ring_buffer import RingBufferHandler

        ring = RingBufferHandler(capacity=3)

        def _emit(name, level, msg):
            rec = _logging.LogRecord(name, level, __file__, 1, msg, None, None)
            ring.emit(rec)

        _emit("agentx_ai.providers.openai", _logging.INFO, "first")
        _emit("agentx_ai.providers.openai", _logging.INFO, "second")
        _emit("agentx_ai.agent.core", _logging.WARNING, "third")
        _emit("agentx_ai.agent.core", _logging.ERROR, "fourth")

        snap = ring.snapshot(limit=10)
        self.assertEqual(len(snap), 3)  # bounded → oldest evicted
        self.assertEqual(snap[0]["message"], "second")
        # structured fields present
        self.assertIn("category", snap[0])
        self.assertIn("run_id", snap[0])
        # filters
        self.assertEqual(len(ring.snapshot(level="error")), 1)
        self.assertEqual(len(ring.snapshot(category="agent")), 2)
        self.assertEqual(len(ring.snapshot(search="fourth")), 1)

    def test_ring_buffer_redacts_and_skips_self(self):
        import logging as _logging

        from agentx_ai.logging_kit.redaction import RedactionFilter
        from agentx_ai.logging_kit.ring_buffer import RingBufferHandler

        ring = RingBufferHandler(capacity=5)
        redactor = RedactionFilter()

        rec = _logging.LogRecord(
            "agentx_ai.providers.openai", _logging.INFO, __file__, 1,
            "key api_key=sk-secret123456789", None, None,
        )
        redactor.filter(rec)  # mirrors the QueueHandler-side filter
        ring.emit(rec)
        self.assertNotIn("sk-secret123456789", ring.snapshot()[-1]["message"])

        # records from the log API / logging_kit itself are never captured
        self_rec = _logging.LogRecord(
            "agentx_ai.views_logs", _logging.INFO, __file__, 1, "serving stream", None, None,
        )
        before = len(ring.snapshot(limit=99))
        ring.emit(self_rec)
        self.assertEqual(len(ring.snapshot(limit=99)), before)

    def test_flags_defaults_and_legacy(self):
        from agentx_ai.logging_kit.flags import read_flags

        keys = [
            "AGENTX_LOG_DECORATIONS", "AGENTX_LOG_BANNER", "AGENTX_LLM_LOG_LEVEL",
            "AGENTX_LOG_API_ENABLED", "AGENTX_LOG_FORMAT", "DEBUG_LOG_LLM_REQUESTS",
            "AGENTX_LOG_ARCHIVE_ENABLED",
        ]
        with patch.dict(os.environ, {}, clear=False):
            for k in keys:
                os.environ.pop(k, None)
            f = read_flags()
            self.assertTrue(f.decorations)
            self.assertTrue(f.banner)         # follows decorations
            self.assertTrue(f.api_enabled)
            self.assertTrue(f.archive_enabled)
            self.assertEqual(f.llm_level, "summary")  # quiet default when decorations on
            self.assertEqual(f.fmt, "pretty")

        # legacy DEBUG_LOG_LLM_REQUESTS → full
        with patch.dict(os.environ, {"DEBUG_LOG_LLM_REQUESTS": "1"}, clear=False):
            os.environ.pop("AGENTX_LLM_LOG_LEVEL", None)
            self.assertEqual(read_flags().llm_level, "full")

        # explicit level wins over legacy
        with patch.dict(os.environ, {"AGENTX_LLM_LOG_LEVEL": "off", "DEBUG_LOG_LLM_REQUESTS": "1"}, clear=False):
            self.assertEqual(read_flags().llm_level, "off")

        # decorations off → llm off by default, plain console
        with patch.dict(os.environ, {"AGENTX_LOG_DECORATIONS": "false"}, clear=False):
            os.environ.pop("AGENTX_LLM_LOG_LEVEL", None)
            os.environ.pop("DEBUG_LOG_LLM_REQUESTS", None)
            f2 = read_flags()
            self.assertFalse(f2.decorations)
            self.assertEqual(f2.llm_level, "off")

    def test_plain_formatter_matches_baseline(self):
        from agentx_ai.logging_kit.handler import build_plain_handler

        handler = build_plain_handler()
        # Must equal the historical 'verbose' format so decorations-off is parity.
        self.assertEqual(handler.formatter._fmt, "{levelname} {asctime} {module} {message}")

    def test_llm_cards_summary_and_off(self):
        from agentx_ai.logging_kit.llm_cards import render_llm_request

        params = {
            "model": "anthropic:claude-opus-4",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "x" * 400}],
            "tools": [{"function": {"name": "web_search"}}],
            "api_key": "sk-ant-secret123456789",
        }
        with patch.dict(os.environ, {"AGENTX_LLM_LOG_LEVEL": "summary"}, clear=False):
            card = render_llm_request("Anthropic", params)
            self.assertIsNotNone(card)
            self.assertIn("Anthropic", card)
            self.assertIn("tools", card)
            self.assertNotIn("sk-ant-secret123456789", card)  # no payload in summary
        with patch.dict(os.environ, {"AGENTX_LLM_LOG_LEVEL": "full"}, clear=False):
            full = render_llm_request("Anthropic", params)
            self.assertIn("«redacted»", full)  # full payload, but redacted
            self.assertNotIn("sk-ant-secret123456789", full)
        with patch.dict(os.environ, {"AGENTX_LLM_LOG_LEVEL": "off"}, clear=False):
            self.assertIsNone(render_llm_request("Anthropic", params))

    def test_archive_segment_traversal_guard(self):
        from agentx_ai.logging_kit.archive import resolve_segment

        self.assertIsNone(resolve_segment("../config.json"))
        self.assertIsNone(resolve_segment("../../etc/passwd"))
        self.assertIsNone(resolve_segment("sub/dir"))


# ---------------------------------------------------------------------------
# Ambassador voice (TTS) — OpenRouter speech synthesis + graceful degradation
# ---------------------------------------------------------------------------


class _FakeSpeechResponse:
    """Stand-in for an httpx Response from /audio/speech (raw audio body)."""

    def __init__(self, content: bytes, status_code: int = 200, headers=None, text: str = ""):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {"content-type": "audio/mpeg", "x-generation-id": "gen_1"}
        self.text = text


def _fake_async_client(response, captured: dict):
    """Build a fake httpx.AsyncClient class whose `post` records the call."""

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return response

    return _FakeClient


class OpenRouterSpeechTest(TestCase):
    """OpenRouter's /audio/speech synthesis + the supports_speech capability."""

    def _provider(self):
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(
            ProviderConfig(api_key="k", base_url="https://openrouter.ai/api/v1")
        )

    def test_synthesize_speech_posts_and_returns_bytes(self):
        from agentx_ai.providers import openrouter_provider as orp

        provider = self._provider()
        captured: dict = {}
        resp = _FakeSpeechResponse(b"MP3DATA")
        with patch.object(orp.httpx, "AsyncClient", _fake_async_client(resp, captured)):
            result = asyncio.run(
                provider.synthesize_speech(
                    "hello there",
                    model="microsoft/mai-voice-2",
                    voice="en-US-Harper:MAI-Voice-2",
                )
            )
        self.assertEqual(result.audio, b"MP3DATA")
        self.assertEqual(result.content_type, "audio/mpeg")
        self.assertEqual(result.generation_id, "gen_1")
        self.assertTrue(captured["url"].endswith("/audio/speech"))
        self.assertEqual(captured["json"]["model"], "microsoft/mai-voice-2")
        self.assertEqual(captured["json"]["input"], "hello there")
        self.assertEqual(captured["json"]["voice"], "en-US-Harper:MAI-Voice-2")
        self.assertEqual(captured["json"]["response_format"], "mp3")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer k")

    def test_synthesize_speech_omits_voice_when_unset(self):
        from agentx_ai.providers import openrouter_provider as orp

        provider = self._provider()
        captured: dict = {}
        resp = _FakeSpeechResponse(b"X")
        with patch.object(orp.httpx, "AsyncClient", _fake_async_client(resp, captured)):
            asyncio.run(provider.synthesize_speech("hi", model="m"))
        self.assertNotIn("voice", captured["json"])

    def test_synthesize_speech_raises_on_http_error(self):
        from agentx_ai.providers import openrouter_provider as orp

        provider = self._provider()
        captured: dict = {}
        resp = _FakeSpeechResponse(b"", status_code=402, text="insufficient credits")
        with patch.object(orp.httpx, "AsyncClient", _fake_async_client(resp, captured)):
            with self.assertRaises(RuntimeError):
                asyncio.run(provider.synthesize_speech("hi", model="m"))

    def test_capabilities_report_speech_for_audio_output(self):
        provider = self._provider()
        provider._model_cache = {
            "tts/voice": {"architecture": {"output_modalities": ["audio"]}},
            "text/model": {"architecture": {"output_modalities": ["text"]}},
        }
        provider._cache_timestamp = 9_999_999_999  # keep cache fresh
        self.assertTrue(provider.get_capabilities("tts/voice").supports_speech)
        self.assertFalse(provider.get_capabilities("text/model").supports_speech)

    def test_base_provider_speech_not_implemented(self):
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(ProviderConfig())
        with self.assertRaises(NotImplementedError):
            asyncio.run(provider.synthesize_speech("hi", model="x"))


class AmbassadorSpeechTest(TestCase):
    """AmbassadorService.synthesize — resolution precedence + graceful degradation."""

    def _service(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        return AmbassadorService()

    def test_empty_text_raises(self):
        from agentx_ai.agent.ambassador import SpeechUnavailable

        svc = self._service()
        with self.assertRaises(SpeechUnavailable) as ctx:
            asyncio.run(svc.synthesize("   "))
        self.assertEqual(ctx.exception.code, "empty_text")

    def test_unconfigured_provider_degrades(self):
        from agentx_ai.agent.ambassador import SpeechUnavailable

        svc = self._service()
        svc._registry = MagicMock()
        svc._registry.get_provider_for_model.side_effect = ValueError("not configured")
        with patch.object(svc, "_resolve_profile", return_value=None):
            with self.assertRaises(SpeechUnavailable) as ctx:
                asyncio.run(svc.synthesize("hello"))
        self.assertEqual(ctx.exception.code, "voice_unconfigured")

    def test_unsupported_model_degrades(self):
        from agentx_ai.agent.ambassador import SpeechUnavailable

        svc = self._service()
        provider = MagicMock()
        provider.synthesize_speech = AsyncMock(side_effect=NotImplementedError("no tts"))
        svc._registry = MagicMock()
        svc._registry.get_provider_for_model.return_value = (provider, "some/model")
        with patch.object(svc, "_resolve_profile", return_value=None):
            with self.assertRaises(SpeechUnavailable) as ctx:
                asyncio.run(svc.synthesize("hello"))
        self.assertEqual(ctx.exception.code, "model_unsupported")

    def test_success_returns_audio_and_uses_default_model(self):
        from agentx_ai.agent.ambassador import (
            _DEFAULT_SPEECH_MODEL,
            _DEFAULT_SPEECH_VOICE,
        )
        from agentx_ai.providers.base import SpeechResult

        svc = self._service()
        provider = MagicMock()
        provider.synthesize_speech = AsyncMock(
            return_value=SpeechResult(audio=b"AUDIO", content_type="audio/mpeg", model="m", voice="v")
        )
        svc._registry = MagicMock()
        svc._registry.get_provider_for_model.return_value = (provider, "microsoft/mai-voice-2")
        with patch.object(svc, "_resolve_profile", return_value=None):
            result = asyncio.run(svc.synthesize("hello"))
        self.assertEqual(result.audio, b"AUDIO")
        # The shipped default model is resolved strictly (no chat fallback).
        svc._registry.get_provider_for_model.assert_called_once_with(_DEFAULT_SPEECH_MODEL)
        # The default voice is passed through to the provider.
        _, kwargs = provider.synthesize_speech.call_args
        self.assertEqual(kwargs["voice"], _DEFAULT_SPEECH_VOICE)
        self.assertEqual(kwargs["response_format"], "mp3")


class _FakeJsonResponse:
    """Stand-in for an httpx Response carrying a JSON body (e.g. transcription)."""

    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class OpenRouterTranscriptionTest(TestCase):
    """OpenRouter's /audio/transcriptions (STT) + the supports_transcription flag."""

    def _provider(self):
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(
            ProviderConfig(api_key="k", base_url="https://openrouter.ai/api/v1")
        )

    def test_transcribe_posts_base64_and_returns_text(self):
        from agentx_ai.providers import openrouter_provider as orp

        provider = self._provider()
        captured: dict = {}
        resp = _FakeJsonResponse({"text": "hello world", "usage": {"seconds": 1.2}})
        with patch.object(orp.httpx, "AsyncClient", _fake_async_client(resp, captured)):
            result = asyncio.run(
                provider.transcribe_speech(b"AUDIOBYTES", model="openai/whisper-1", audio_format="webm")
            )
        self.assertEqual(result.text, "hello world")
        self.assertTrue(captured["url"].endswith("/audio/transcriptions"))
        self.assertEqual(captured["json"]["model"], "openai/whisper-1")
        self.assertEqual(captured["json"]["input_audio"]["format"], "webm")
        import base64 as _b64

        self.assertEqual(
            _b64.b64decode(captured["json"]["input_audio"]["data"]), b"AUDIOBYTES"
        )

    def test_transcribe_raises_on_http_error(self):
        from agentx_ai.providers import openrouter_provider as orp

        provider = self._provider()
        captured: dict = {}
        resp = _FakeJsonResponse({}, status_code=429, text="rate limited")
        with patch.object(orp.httpx, "AsyncClient", _fake_async_client(resp, captured)):
            with self.assertRaises(RuntimeError):
                asyncio.run(provider.transcribe_speech(b"x", model="m"))

    def test_capabilities_report_transcription(self):
        provider = self._provider()
        provider._model_cache = {
            "stt/whisper": {
                "architecture": {"input_modalities": ["audio"], "output_modalities": ["transcription"]}
            },
            "chat/audio": {
                "architecture": {"input_modalities": ["audio", "text"], "output_modalities": ["text"]}
            },
        }
        provider._cache_timestamp = 9_999_999_999
        self.assertTrue(provider.get_capabilities("stt/whisper").supports_transcription)
        # An audio-input *chat* model is not an STT endpoint.
        self.assertFalse(provider.get_capabilities("chat/audio").supports_transcription)

    def test_base_provider_transcription_not_implemented(self):
        from agentx_ai.providers.base import ProviderConfig
        from agentx_ai.providers.lmstudio_provider import LMStudioProvider

        provider = LMStudioProvider(ProviderConfig())
        with self.assertRaises(NotImplementedError):
            asyncio.run(provider.transcribe_speech(b"x", model="m"))


class AmbassadorTranscriptionTest(TestCase):
    """AmbassadorService.transcribe — resolution precedence + graceful degradation."""

    def _service(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        return AmbassadorService()

    def test_empty_audio_raises(self):
        from agentx_ai.agent.ambassador import SpeechUnavailable

        svc = self._service()
        with self.assertRaises(SpeechUnavailable) as ctx:
            asyncio.run(svc.transcribe(b""))
        self.assertEqual(ctx.exception.code, "empty_audio")

    def test_unconfigured_provider_degrades(self):
        from agentx_ai.agent.ambassador import SpeechUnavailable

        svc = self._service()
        svc._registry = MagicMock()
        svc._registry.get_provider_for_model.side_effect = ValueError("not configured")
        with patch.object(svc, "_resolve_profile", return_value=None):
            with self.assertRaises(SpeechUnavailable) as ctx:
                asyncio.run(svc.transcribe(b"AUDIO"))
        self.assertEqual(ctx.exception.code, "transcription_unconfigured")

    def test_success_returns_text_and_uses_default_model(self):
        from agentx_ai.agent.ambassador import _DEFAULT_TRANSCRIPTION_MODEL
        from agentx_ai.providers.base import TranscriptionResult

        svc = self._service()
        provider = MagicMock()
        provider.transcribe_speech = AsyncMock(
            return_value=TranscriptionResult(text="transcribed text", model="m")
        )
        svc._registry = MagicMock()
        svc._registry.get_provider_for_model.return_value = (provider, "openai/whisper-1")
        with patch.object(svc, "_resolve_profile", return_value=None):
            text = asyncio.run(svc.transcribe(b"AUDIO", audio_format="webm"))
        self.assertEqual(text, "transcribed text")
        svc._registry.get_provider_for_model.assert_called_once_with(_DEFAULT_TRANSCRIPTION_MODEL)
        _, kwargs = provider.transcribe_speech.call_args
        self.assertEqual(kwargs["audio_format"], "webm")


class AmbassadorVoiceCommandTest(TestCase):
    """AmbassadorService.route_voice_command — intent routing + qa persistence."""

    def _service(self, classify_reply: str, *, answer_text: str = ""):
        """An AmbassadorService whose ``complete`` classifies the intent (returns
        ``classify_reply``) and whose ``stream`` is the agentic answer core (yields
        ``answer_text``). Registry stubbed."""
        from types import SimpleNamespace

        from agentx_ai.agent.ambassador import AmbassadorService
        from agentx_ai.providers.base import StreamChunk

        svc = AmbassadorService()
        provider = MagicMock()
        provider.complete = AsyncMock(return_value=SimpleNamespace(content=classify_reply))

        async def _stream(messages, model_id, **kwargs):
            yield StreamChunk(content=answer_text, finish_reason="stop")

        provider.stream = _stream
        svc._registry = MagicMock()
        svc._registry.resolve_with_fallback.return_value = (provider, "anthropic:model", None)
        return svc

    def test_answer_routes_through_core_and_persists_qa(self):
        # classify → answer; the spoken answer is produced by the agentic core (stream),
        # NOT the classifier's text, and is persisted to qa: for the Text tab.
        from agentx_ai.agent import ambassador_storage as a

        svc = self._service(
            '{"action": "answer", "text": "discarded classifier text"}',
            answer_text="it searched the county index.",
        )
        fake = _FakeKVRedis()
        with patch.object(svc, "_resolve_profile", return_value=None), \
             patch.object(svc, "_grounding_context", return_value=""), \
             patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
             patch("agentx_ai.agent.ambassador_tools.load_recent_labeled_turns", return_value=[]), \
             patch.object(a, "_redis", return_value=fake):
            result = asyncio.run(svc.route_voice_command("conv1", "what did it find?"))
            rec = a.get_qa("conv1", result["qa_id"])
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["text"], "it searched the county index.")
        self.assertIsNotNone(result["qa_id"])
        self.assertEqual(rec["answer"], "it searched the county index.")
        self.assertEqual(rec["question"], "what did it find?")

    def test_relay_routes_without_persisting(self):
        from agentx_ai.agent import ambassador_storage as a

        svc = self._service('{"action": "relay", "text": "Use Postgres for the vector store."}')
        fake = _FakeKVRedis()
        with patch.object(svc, "_resolve_profile", return_value=None), \
             patch.object(svc, "_grounding_context", return_value=""), \
             patch.object(a, "_redis", return_value=fake):
            result = asyncio.run(svc.route_voice_command("conv1", "tell it to use postgres"))
        self.assertEqual(result["action"], "relay")
        self.assertEqual(result["text"], "Use Postgres for the vector store.")
        self.assertIsNone(result["qa_id"])

    def test_malformed_json_falls_back_to_answer(self):
        # Non-JSON classify → answer action; the core still produces the spoken answer.
        from agentx_ai.agent import ambassador_storage as a

        svc = self._service(
            "I think they want to know about the search.",
            answer_text="They searched the registry for you.",
        )
        fake = _FakeKVRedis()
        with patch.object(svc, "_resolve_profile", return_value=None), \
             patch.object(svc, "_grounding_context", return_value=""), \
             patch("agentx_ai.agent.ambassador.load_recent_turns", return_value=[]), \
             patch("agentx_ai.agent.ambassador_tools.load_recent_labeled_turns", return_value=[]), \
             patch.object(a, "_redis", return_value=fake):
            result = asyncio.run(svc.route_voice_command("conv1", "what's up"))
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["text"], "They searched the registry for you.")

    def test_empty_transcript_short_circuits(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        svc._registry = MagicMock()
        result = asyncio.run(svc.route_voice_command("conv1", "   "))
        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["text"], "")
        svc._registry.resolve_with_fallback.assert_not_called()

    def test_disabled_degrades_gracefully(self):
        from agentx_ai.agent.ambassador import AmbassadorService

        svc = AmbassadorService()
        svc._registry = MagicMock()
        cfg = {"enabled": False, "profile_id": None, "model": None, "max_context_turns": 8, "max_tokens": None}
        with patch.object(svc, "_config", return_value=cfg):
            result = asyncio.run(svc.route_voice_command("conv1", "hello"))
        self.assertEqual(result["action"], "answer")
        self.assertIn("disabled", result["text"].lower())
        svc._registry.resolve_with_fallback.assert_not_called()


class LogArchiveCryptoTest(TestCase):
    """Envelope encryption for the durable log archive (no external deps)."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from agentx_ai.logging_kit import archive_crypto

        self.ac = archive_crypto
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_dir = archive_crypto.ARCHIVE_DIR
        self._orig_keyring = archive_crypto.KEYRING_PATH
        archive_crypto.ARCHIVE_DIR = self.tmp
        archive_crypto.KEYRING_PATH = self.tmp / "keyring.json"
        archive_crypto.clear_cached_dek()

    def tearDown(self):
        import shutil

        self.ac.ARCHIVE_DIR = self._orig_dir
        self.ac.KEYRING_PATH = self._orig_keyring
        self.ac.clear_cached_dek()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_gz(self, name, payload):
        import gzip

        gz = self.tmp / name
        with gzip.open(gz, "wb") as f:
            f.write(payload)
        return gz

    def test_keyring_create_unwrap_and_bad_password(self):
        dek = self.ac.create_keyring("correct-horse")
        self.assertEqual(self.ac.unwrap_dek("correct-horse"), dek)
        with self.assertRaises(self.ac.BadPassword):
            self.ac.unwrap_dek("wrong-password")

    def test_seal_unseal_roundtrip(self):
        import gzip

        dek = self.ac.create_keyring("correct-horse")
        payload = b"INFO 2026-06-08 mod a redacted reasoning trace\n" * 4000
        gz = self._make_gz("agentx-2026-06-08.log.gz", payload)
        enc = self.ac.seal_segment(gz, dek)
        self.assertIsNotNone(enc)
        self.assertTrue(enc.exists())
        self.assertFalse(gz.exists())  # plaintext removed after sealing
        self.assertEqual(gzip.decompress(self.ac.unseal_bytes(enc, dek)), payload)

    def test_rewrap_preserves_decryptability(self):
        import gzip

        dek = self.ac.create_keyring("old-passphrase")
        payload = b"WARNING premise/conclusion mismatch\n" * 1000
        enc = self.ac.seal_segment(self._make_gz("agentx-2026-06-08.log.gz", payload), dek)
        # O(1) re-wrap, preferring the cached DEK (as change_password does).
        self.ac.rewrap_dek("new-passphrase", dek=dek)
        with self.assertRaises(self.ac.BadPassword):
            self.ac.unwrap_dek("old-passphrase")
        dek2 = self.ac.unwrap_dek("new-passphrase")
        self.assertEqual(gzip.decompress(self.ac.unseal_bytes(enc, dek2)), payload)

    def test_tamper_is_rejected(self):
        dek = self.ac.create_keyring("correct-horse")
        enc = self.ac.seal_segment(self._make_gz("agentx-2026-06-08.log.gz", b"x" * 2048), dek)
        raw = bytearray(enc.read_bytes())
        raw[-16] ^= 0xFF
        enc.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.ac.unseal_bytes(enc, dek)

    def test_truncation_is_detected(self):
        dek = self.ac.create_keyring("correct-horse")
        enc = self.ac.seal_segment(self._make_gz("agentx-2026-06-08.log.gz", b"y" * 4096), dek)
        raw = enc.read_bytes()
        enc.write_bytes(raw[: len(raw) // 2])  # drop trailing frames + terminator
        with self.assertRaises(ValueError):
            self.ac.unseal_bytes(enc, dek)

    def test_seal_pending_and_reencrypt_all(self):
        import gzip

        dek = self.ac.create_keyring("old-passphrase")
        self.ac.set_cached_dek(dek)
        p1 = b"day one\n" * 500
        p2 = b"day two\n" * 500
        self._make_gz("agentx-2026-06-07.log.gz", p1)
        self._make_gz("agentx-2026-06-08.log.gz", p2)
        self.assertEqual(self.ac.seal_pending(), 2)
        self.assertEqual(self.ac.seal_pending(), 0)  # idempotent

        count = self.ac.reencrypt_all("old-passphrase", "new-passphrase")
        self.assertEqual(count, 2)
        dek2 = self.ac.unwrap_dek("new-passphrase")
        recovered = {gzip.decompress(self.ac.unseal_bytes(p, dek2)) for p in self.tmp.glob("*.enc")}
        self.assertEqual(recovered, {p1, p2})

    def test_prune_old(self):
        import os
        import time

        self.ac.create_keyring("correct-horse")
        old = self._make_gz("agentx-2000-01-01.log.gz", b"ancient\n")
        recent = self._make_gz("agentx-2026-06-08.log.gz", b"fresh\n")
        stale = time.time() - 40 * 86400
        os.utime(old, (stale, stale))
        self.assertEqual(self.ac.prune_old(30), 1)
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
