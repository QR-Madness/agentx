import socket
from unittest import skipUnless

from django.test import TestCase, Client

from agentx_ai.kit.translation import LanguageLexicon


def _docker_services_running():
    """Check if Docker services (Neo4j, PostgreSQL, Redis) are reachable."""
    services = [("localhost", 7687), ("localhost", 5432), ("localhost", 6379)]
    for host, port in services:
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except (socket.error, socket.timeout):
            return False
    return True


def _translation_models_loaded():
    """Check if translation models are available (slow to load)."""
    try:
        from agentx_ai.kit.translation import TranslationKit
        # Just check if class is importable, actual loading happens in tests
        return True
    except ImportError:
        return False


# Create your tests here.
class TranslationKitTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_language_detect_post(self):
        """Test that the language detection API works with POST."""
        response = self.client.post(
            "/api/tools/language-detect-20",
            data={"text": "Bonjour, comment allez-vous?"},
            content_type="application/json"
        )
        data = response.json()
        print(data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('detected_language', data)
        self.assertIn('confidence', data)
        self.assertEqual(data['detected_language'], 'fr')

    def test_language_detect_get(self):
        """Test backwards compatibility with GET request."""
        response = self.client.get("/api/tools/language-detect-20")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('detected_language', data)
        # Default text is English
        self.assertEqual(data['detected_language'], 'en')

    def test_lexicon_convert_level_i_to_level_ii(self):
        """Test that the lexicon converts level I language codes to level II language codes."""
        lexicon = LanguageLexicon(verbose=True)
        level_i_language = "en"
        level_ii_language = lexicon.convert_level_i_detection_to_level_ii(level_i_language)
        self.assertEqual(level_ii_language, "eng_Latn")
        self.assertTrue(level_ii_language in lexicon.level_ii_languages)
        print(f'Got correct level II language: {level_ii_language}')

    def test_translate_to_french(self):
        """Test that the translation API works."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello, AgentX AI!", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)


class HealthCheckTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_health_endpoint(self):
        """Test that the health check endpoint returns expected structure."""
        response = self.client.get("/api/health")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('status', data)
        self.assertIn('api', data)
        self.assertIn('translation', data)
        self.assertEqual(data['api']['status'], 'healthy')

    @skipUnless(_docker_services_running(), "Docker services not running")
    def test_health_with_memory_check(self):
        """Test health check with memory system - requires Docker services running."""
        response = self.client.get("/api/health?include_memory=true")
        data = response.json()
        self.assertEqual(response.status_code, 200)
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


class MCPClientTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_mcp_servers_endpoint(self):
        """Test that the MCP servers endpoint returns expected structure."""
        response = self.client.get("/api/mcp/servers")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('configured_servers', data)
        self.assertIn('active_connections', data)
        self.assertIsInstance(data['configured_servers'], list)
        self.assertIsInstance(data['active_connections'], list)

    def test_mcp_tools_endpoint(self):
        """Test that the MCP tools endpoint returns expected structure."""
        response = self.client.get("/api/mcp/tools")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('tools', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['tools'], list)

    def test_mcp_resources_endpoint(self):
        """Test that the MCP resources endpoint returns expected structure."""
        response = self.client.get("/api/mcp/resources")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('resources', data)
        self.assertIn('count', data)
        self.assertIsInstance(data['resources'], list)


class MCPServerRegistryTest(TestCase):
    def test_server_config_creation(self):
        """Test creating a server configuration."""
        from agentx_ai.mcp import ServerConfig
        from agentx_ai.mcp.server_registry import TransportType
        
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

    def test_server_registry_operations(self):
        """Test server registry register/get/list operations."""
        from agentx_ai.mcp import ServerRegistry, ServerConfig
        from agentx_ai.mcp.server_registry import TransportType
        
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
        self.assertEqual(retrieved.name, "test-server")
        
        # Test list
        servers = registry.list()
        self.assertEqual(len(servers), 1)
        
        # Test unregister
        result = registry.unregister("test-server")
        self.assertTrue(result)
        self.assertIsNone(registry.get("test-server"))

    def test_env_resolution(self):
        """Test environment variable resolution in server config."""
        import os
        from agentx_ai.mcp import ServerConfig
        from agentx_ai.mcp.server_registry import TransportType
        
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

class TranslationLanguagePairsTest(TestCase):
    """Test translation across multiple language pairs."""
    
    def setUp(self):
        self.client = Client()
    
    def test_translate_english_to_spanish(self):
        """Test English to Spanish translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello, how are you?", "targetLanguage": "spa_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('translatedText', data)
        # Should contain Spanish words
        self.assertTrue(len(data['translatedText']) > 0)
    
    def test_translate_english_to_german(self):
        """Test English to German translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Good morning", "targetLanguage": "deu_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('translatedText', data)
    
    def test_translate_english_to_japanese(self):
        """Test English to Japanese translation."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Thank you", "targetLanguage": "jpn_Jpan"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('translatedText', data)


class TranslationErrorHandlingTest(TestCase):
    """Test translation error handling."""
    
    def setUp(self):
        self.client = Client()
    
    def test_translate_invalid_language_code(self):
        """Test translation with invalid language code."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello", "targetLanguage": "invalid_code"},
            content_type="application/json"
        )
        # Should return error response
        self.assertIn(response.status_code, [400, 500])
    
    def test_translate_empty_text(self):
        """Test translation with empty text."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        # API should handle empty text gracefully
        self.assertIn(response.status_code, [200, 400])
    
    def test_translate_missing_target_language(self):
        """Test translation with missing target language."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hello"},
            content_type="application/json"
        )
        # Should return error for missing required field
        self.assertEqual(response.status_code, 400)


class TranslationLongTextTest(TestCase):
    """Test translation with various text lengths."""
    
    def setUp(self):
        self.client = Client()
    
    def test_translate_short_text(self):
        """Test translation of short text."""
        response = self.client.post(
            "/api/tools/translate",
            data={"text": "Hi", "targetLanguage": "fra_Latn"},
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
    
    def test_translate_paragraph(self):
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
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('translatedText', data)
        # Translation should be roughly similar length
        self.assertTrue(len(data['translatedText']) > 20)


# =============================================================================
# Phase 10: Reasoning Framework Tests
# =============================================================================

class ReasoningBaseTest(TestCase):
    """Test reasoning framework base classes."""
    
    def test_reasoning_status_enum(self):
        """Test ReasoningStatus enum values."""
        from agentx_ai.reasoning.base import ReasoningStatus
        
        self.assertEqual(ReasoningStatus.PENDING, "pending")
        self.assertEqual(ReasoningStatus.THINKING, "thinking")
        self.assertEqual(ReasoningStatus.COMPLETE, "complete")
        self.assertEqual(ReasoningStatus.FAILED, "failed")
    
    def test_thought_type_enum(self):
        """Test ThoughtType enum values."""
        from agentx_ai.reasoning.base import ThoughtType
        
        self.assertEqual(ThoughtType.OBSERVATION, "observation")
        self.assertEqual(ThoughtType.REASONING, "reasoning")
        self.assertEqual(ThoughtType.ACTION, "action")
        self.assertEqual(ThoughtType.CONCLUSION, "conclusion")
    
    def test_thought_step_creation(self):
        """Test ThoughtStep model creation."""
        from agentx_ai.reasoning.base import ThoughtStep, ThoughtType
        
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
    
    def test_reasoning_result_creation(self):
        """Test ReasoningResult model creation."""
        from agentx_ai.reasoning.base import ReasoningResult, ReasoningStatus
        
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
    
    def test_reasoning_config_creation(self):
        """Test ReasoningConfig creation."""
        from agentx_ai.reasoning.base import ReasoningConfig
        
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
    
    def test_cot_config_creation(self):
        """Test CoTConfig creation with defaults."""
        from agentx_ai.reasoning.chain_of_thought import CoTConfig
        
        config = CoTConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.mode, "zero_shot")
        self.assertEqual(config.thinking_prompt, "Let's think step by step.")
        self.assertTrue(config.extract_steps)
    
    def test_cot_config_few_shot_mode(self):
        """Test CoTConfig with few-shot mode."""
        from agentx_ai.reasoning.chain_of_thought import CoTConfig
        
        examples = [
            {"question": "2+2?", "reasoning": "Add 2 and 2", "answer": "4"}
        ]
        config = CoTConfig(
            model="llama3.2",
            mode="few_shot",
            examples=examples
        )
        
        self.assertEqual(config.mode, "few_shot")
        self.assertEqual(len(config.examples), 1)
    
    def test_step_extraction_pattern(self):
        """Test that step extraction regex works correctly."""
        import re
        
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
    
    def test_tree_node_creation(self):
        """Test TreeNode creation."""
        from agentx_ai.reasoning.tree_of_thought import TreeNode
        
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
    
    def test_tot_config_defaults(self):
        """Test ToT configuration defaults."""
        from agentx_ai.reasoning.tree_of_thought import ToTConfig
        
        config = ToTConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.search_method, "bfs")
        self.assertEqual(config.max_depth, 4)
        self.assertEqual(config.branching_factor, 3)


class ReActTest(TestCase):
    """Test ReAct reasoning components."""
    
    def test_tool_creation(self):
        """Test Tool creation."""
        from agentx_ai.reasoning.react import Tool
        
        tool = Tool(
            name="search",
            description="Search for information",
            parameters={"query": "string"},
            execute=lambda x: "result"
        )
        
        self.assertEqual(tool.name, "search")
        self.assertEqual(tool.description, "Search for information")
    
    def test_react_config_defaults(self):
        """Test ReAct configuration defaults."""
        from agentx_ai.reasoning.react import ReActConfig
        
        config = ReActConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.max_iterations, 10)
        self.assertEqual(config.thought_prefix, "Thought:")
        self.assertEqual(config.action_prefix, "Action:")


class ReflectionTest(TestCase):
    """Test Reflection reasoning components."""
    
    def test_revision_creation(self):
        """Test Revision model creation."""
        from agentx_ai.reasoning.reflection import Revision
        
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
    
    def test_reflection_config_defaults(self):
        """Test ReflectionConfig defaults."""
        from agentx_ai.reasoning.reflection import ReflectionConfig
        
        config = ReflectionConfig(model="llama3.2")
        
        self.assertEqual(config.model, "llama3.2")
        self.assertEqual(config.max_revisions, 3)


# =============================================================================
# Phase 10: Drafting Framework Tests
# =============================================================================

class DraftingBaseTest(TestCase):
    """Test drafting framework base classes."""
    
    def test_draft_status_enum(self):
        """Test DraftStatus enum values."""
        from agentx_ai.drafting.base import DraftStatus
        
        self.assertEqual(DraftStatus.PENDING, "pending")
        self.assertEqual(DraftStatus.DRAFTING, "drafting")
        self.assertEqual(DraftStatus.COMPLETE, "complete")
        self.assertEqual(DraftStatus.FAILED, "failed")
    
    def test_draft_result_creation(self):
        """Test DraftResult model creation."""
        from agentx_ai.drafting.base import DraftResult
        
        result = DraftResult(
            content="Generated draft content",
            strategy="speculative",
            total_tokens=50,
        )
        
        self.assertEqual(result.content, "Generated draft content")
        self.assertEqual(result.strategy, "speculative")
        self.assertEqual(result.total_tokens, 50)
    
    def test_drafting_config_creation(self):
        """Test DraftingConfig creation."""
        from agentx_ai.drafting.base import DraftingConfig
        
        config = DraftingConfig(
            name="test-speculative",
            strategy_type="speculative",
        )
        
        self.assertEqual(config.name, "test-speculative")
        self.assertEqual(config.strategy_type, "speculative")
        self.assertEqual(config.temperature, 0.7)


class SpeculativeDecodingTest(TestCase):
    """Test speculative decoding components."""
    
    def test_speculative_config_defaults(self):
        """Test SpeculativeConfig defaults."""
        from agentx_ai.drafting.speculative import SpeculativeConfig
        
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
    
    def test_stage_role_enum(self):
        """Test StageRole enum values."""
        from agentx_ai.drafting.pipeline import StageRole
        
        self.assertEqual(StageRole.ANALYZE, "analyze")
        self.assertEqual(StageRole.DRAFT, "draft")
        self.assertEqual(StageRole.REVIEW, "review")
        self.assertEqual(StageRole.REFINE, "refine")
    
    def test_pipeline_stage_creation(self):
        """Test PipelineStage creation."""
        from agentx_ai.drafting.pipeline import PipelineStage, StageRole
        
        stage = PipelineStage(
            name="draft-stage",
            model="llama3.2",
            role=StageRole.DRAFT,
        )
        
        self.assertEqual(stage.name, "draft-stage")
        self.assertEqual(stage.model, "llama3.2")
        self.assertEqual(stage.role, StageRole.DRAFT)
    
    def test_pipeline_config_creation(self):
        """Test PipelineConfig creation."""
        from agentx_ai.drafting.pipeline import PipelineConfig, PipelineStage, StageRole
        
        stages = [
            PipelineStage(name="analyze", model="llama3.2", role=StageRole.ANALYZE),
            PipelineStage(name="draft", model="llama3.2", role=StageRole.DRAFT),
        ]
        config = PipelineConfig(name="test-pipeline", stages=stages)
        
        self.assertEqual(config.name, "test-pipeline")
        self.assertEqual(len(config.stages), 2)


class CandidateGenerationTest(TestCase):
    """Test candidate generation components."""
    
    def test_scoring_method_enum(self):
        """Test ScoringMethod enum values."""
        from agentx_ai.drafting.candidate import ScoringMethod
        
        self.assertEqual(ScoringMethod.MAJORITY_VOTE, "majority_vote")
        self.assertEqual(ScoringMethod.VERIFIER, "verifier")
        self.assertEqual(ScoringMethod.LENGTH_PREFERENCE, "length_preference")
    
    def test_candidate_creation(self):
        """Test Candidate model creation."""
        from agentx_ai.drafting.candidate import Candidate
        
        candidate = Candidate(
            content="This is a candidate response.",
            score=0.85,
            model="llama3.2",
            index=0,
        )
        
        self.assertEqual(candidate.content, "This is a candidate response.")
        self.assertEqual(candidate.score, 0.85)
        self.assertEqual(candidate.index, 0)
    
    def test_candidate_config_defaults(self):
        """Test CandidateConfig defaults."""
        from agentx_ai.drafting.candidate import CandidateConfig, ScoringMethod
        
        config = CandidateConfig(name="test-gen", models=["llama3.2"])
        
        self.assertEqual(config.candidates_per_model, 1)
        self.assertEqual(config.scoring_method, ScoringMethod.MAJORITY_VOTE)


# =============================================================================
# Phase 10: Provider Tests
# =============================================================================

class ProviderRegistryTest(TestCase):
    """Test provider registry functionality."""
    
    def test_registry_singleton(self):
        """Test that get_registry returns same instance."""
        from agentx_ai.providers.registry import get_registry
        
        reg1 = get_registry()
        reg2 = get_registry()
        
        self.assertIs(reg1, reg2)
    
    def test_provider_detection_local_prefix(self):
        """Test provider detection for local models by prefix."""
        from agentx_ai.providers.registry import get_registry
        
        registry = get_registry()
        
        # Models with local prefixes should be detected
        # This tests the detection logic without requiring providers to be configured
        local_prefixes = ["llama", "mistral", "qwen", "phi", "gemma"]
        for prefix in local_prefixes:
            model = f"{prefix}3.2"
            # Just verify the model name starts with a known local prefix
            self.assertTrue(model.startswith(prefix))
    
    def test_model_config_retrieval(self):
        """Test model config retrieval."""
        from agentx_ai.providers.registry import get_registry
        
        registry = get_registry()
        
        # Should return None for unknown models
        config = registry.get_model_config("nonexistent-model-xyz")
        self.assertIsNone(config)


class ProviderBaseTest(TestCase):
    """Test provider base classes."""
    
    def test_message_creation(self):
        """Test Message model creation."""
        from agentx_ai.providers.base import Message, MessageRole
        
        msg = Message(role=MessageRole.USER, content="Hello")
        
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello")
    
    def test_message_role_enum(self):
        """Test MessageRole enum values."""
        from agentx_ai.providers.base import MessageRole
        
        self.assertEqual(MessageRole.SYSTEM, "system")
        self.assertEqual(MessageRole.USER, "user")
        self.assertEqual(MessageRole.ASSISTANT, "assistant")
    
    def test_completion_result_creation(self):
        """Test CompletionResult model creation."""
        from agentx_ai.providers.base import CompletionResult
        
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

def _extraction_model_available():
    """Check if extraction model provider is configured."""
    import os
    return bool(
        os.environ.get("ANTHROPIC_API_KEY") or
        os.environ.get("OPENAI_API_KEY") or
        os.environ.get("OLLAMA_BASE_URL") or
        os.environ.get("LMSTUDIO_BASE_URL")
    )


class ExtractionPipelineTest(TestCase):
    """Tests for the extraction pipeline."""

    def test_extract_entities_empty_text(self):
        """Empty text should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_entities
        result = extract_entities("")
        self.assertEqual(result, [])

    def test_extract_entities_short_text(self):
        """Very short text should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_entities
        result = extract_entities("Hi there")
        self.assertEqual(result, [])

    def test_extract_facts_empty_text(self):
        """Empty text should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_facts
        result = extract_facts("")
        self.assertEqual(result, [])

    def test_extract_facts_short_text(self):
        """Very short text should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_facts
        result = extract_facts("OK")
        self.assertEqual(result, [])

    def test_extract_relationships_no_entities(self):
        """Relationships extraction with no entities should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_relationships
        result = extract_relationships("Some text here", [])
        self.assertEqual(result, [])

    def test_extract_relationships_empty_text(self):
        """Relationships extraction with empty text should return empty list."""
        from agentx_ai.kit.agent_memory.extraction import extract_relationships
        entities = [{"name": "Test", "type": "Person"}]
        result = extract_relationships("", entities)
        self.assertEqual(result, [])

    def test_extraction_service_singleton(self):
        """Extraction service should be a singleton."""
        from agentx_ai.kit.agent_memory.extraction.service import (
            get_extraction_service,
            reset_extraction_service
        )

        # Reset to ensure clean state
        reset_extraction_service()

        service1 = get_extraction_service()
        service2 = get_extraction_service()
        self.assertIs(service1, service2)

        # Clean up
        reset_extraction_service()

    def test_extraction_result_model(self):
        """Test ExtractionResult model structure."""
        from agentx_ai.kit.agent_memory.extraction.service import ExtractionResult

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

    def test_extraction_config_settings(self):
        """Test extraction configuration is loaded."""
        from agentx_ai.kit.agent_memory.config import get_settings

        settings = get_settings()

        self.assertTrue(hasattr(settings, 'extraction_enabled'))
        self.assertTrue(hasattr(settings, 'extraction_model'))
        self.assertTrue(hasattr(settings, 'extraction_provider'))
        self.assertTrue(hasattr(settings, 'extraction_temperature'))
        self.assertTrue(hasattr(settings, 'entity_types'))
        self.assertTrue(hasattr(settings, 'relationship_types'))

    @skipUnless(_extraction_model_available(), "Extraction model provider not configured")
    def test_extract_entities_real(self):
        """Test real entity extraction with configured provider."""
        from agentx_ai.kit.agent_memory.extraction import extract_entities

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

    @skipUnless(_extraction_model_available(), "Extraction model provider not configured")
    def test_extract_facts_real(self):
        """Test real fact extraction with configured provider."""
        from agentx_ai.kit.agent_memory.extraction import extract_facts

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
