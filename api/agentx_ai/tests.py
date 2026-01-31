from django.test import TestCase, Client

from agentx_ai.kit.translation import LanguageLexicon

"""

TODO 0) Integrate FAISS for vector database
TODO 1) Implement Django ORM for AI settings, basic storage
TODO 2) Implement neo4j for structured information storage and relationship analysis
 
"""


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
