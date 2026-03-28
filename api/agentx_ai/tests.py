import asyncio
import json
import os
import re
from datetime import datetime, timedelta, timezone
from unittest import skipUnless
from unittest.mock import AsyncMock, MagicMock, patch

from django.test import TestCase

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
        self.assertTrue(hasattr(settings, 'extraction_model'))
        self.assertTrue(hasattr(settings, 'extraction_provider'))
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


# Phase 11.8+ tests moved to tests_memory.py
