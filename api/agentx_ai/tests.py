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
from datetime import datetime, timedelta, timezone
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

        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)
        old = now - timedelta(hours=48)

        recent_score = retriever._calculate_recency_score(recent)
        old_score = retriever._calculate_recency_score(old)

        self.assertGreater(recent_score, old_score)

    def test_calculate_recency_score_decays_exponentially(self):
        """Recency score decays with 24-hour half-life."""
        mock_memory = MagicMock()
        retriever = MemoryRetriever(mock_memory)

        now = datetime.now(timezone.utc)
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

    def test_tavily_success(self):
        from unittest.mock import patch

        tavily_payload = {"results": [
            {"title": "T1", "url": "https://a", "content": "snippet one"},
            {"title": "T2", "url": "https://b", "content": "snippet two"},
        ]}
        with patch.object(self.internal_tools, "_http_post_json", return_value=tavily_payload):
            out = self.internal_tools.web_search("hello world")
        self.assertTrue(out["success"])
        self.assertEqual(out["backend"], "tavily")
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["results"][0], {"title": "T1", "url": "https://a", "snippet": "snippet one"})

    def test_fallback_to_brave(self):
        from unittest.mock import patch

        brave_payload = {"web": {"results": [
            {"title": "B1", "url": "https://x", "description": "brave snippet"},
        ]}}
        with patch.object(self.internal_tools, "_http_post_json", side_effect=RuntimeError("tavily down")), \
             patch.object(self.internal_tools, "_http_get_json", return_value=brave_payload):
            out = self.internal_tools.web_search("hello")
        self.assertTrue(out["success"])
        self.assertEqual(out["backend"], "brave")
        self.assertEqual(out["results"][0], {"title": "B1", "url": "https://x", "snippet": "brave snippet"})

    def test_both_down_graceful_empty(self):
        from unittest.mock import patch

        with patch.object(self.internal_tools, "_http_post_json", side_effect=RuntimeError("tavily down")), \
             patch.object(self.internal_tools, "_http_get_json", side_effect=RuntimeError("brave down")):
            out = self.internal_tools.web_search("hello")
        self.assertFalse(out["success"])
        self.assertEqual(out["results"], [])
        self.assertIn("error", out)

    def test_cache_hit_avoids_second_call(self):
        from unittest.mock import patch

        tavily_payload = {"results": [{"title": "T", "url": "https://a", "content": "c"}]}
        with patch.object(
            self.internal_tools, "_http_post_json", return_value=tavily_payload
        ) as mock_post:
            first = self.internal_tools.web_search("cached query")
            second = self.internal_tools.web_search("cached query")
        self.assertEqual(mock_post.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])

    def test_empty_query_rejected(self):
        out = self.internal_tools.web_search("   ")
        self.assertFalse(out["success"])
        self.assertEqual(out["results"], [])


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
    """Slice 5 — universal model fallback (registry) + memory stage inheritance.

    A feature whose configured model is unavailable (provider unconfigured or
    unreachable) must fall back to the active/default model instead of crashing
    the turn; the main chat path stays strict.
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
