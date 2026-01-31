/**
 * API client for AgentX backend
 * All calls are routed through the active server configuration
 */

import { getActiveServer, markServerConnected, updateActiveServerMetadata } from './storage';

// === Types ===

export interface ApiError {
  message: string;
  status: number;
  details?: unknown;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  api: { status: string };
  translation?: { status: string; models?: Record<string, unknown> };
  memory?: {
    neo4j?: { status: string };
    postgres?: { status: string };
    redis?: { status: string };
  };
}

export interface ProviderInfo {
  name: string;
  available: boolean;
  models: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  context_length?: number;
  capabilities?: string[];
}

export interface MCPServer {
  name: string;
  status: string;
  tools?: string[];
}

export interface MCPTool {
  name: string;
  description: string;
  server: string;
  inputSchema?: Record<string, unknown>;
}

export interface AgentRunRequest {
  task: string;
  reasoning_strategy?: string;
  drafting_strategy?: string;
  model?: string;
  tools?: string[];
}

export interface AgentRunResponse {
  result: string;
  reasoning_trace?: ReasoningStep[];
  tools_used?: ToolUsage[];
  tokens_used?: number;
  duration_ms?: number;
}

export interface ReasoningStep {
  type: 'thought' | 'action' | 'observation' | 'reflection';
  content: string;
  timestamp?: string;
}

export interface ToolUsage {
  tool: string;
  input: unknown;
  output: unknown;
  success: boolean;
  duration_ms?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  model?: string;
  show_reasoning?: boolean;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  reasoning_trace?: ReasoningStep[];
  tokens_used?: number;
}

export interface TranslateRequest {
  text: string;
  targetLanguage: string;
  sourceLanguage?: string;
}

export interface TranslateResponse {
  original: string;
  translatedText: string;
  sourceLanguage?: string;
  targetLanguage: string;
}

export interface LanguageDetectResponse {
  original: string;
  detected_language: string;
  confidence: number;
}

// === API Client ===

class ApiClient {
  private getBaseUrl(): string {
    const server = getActiveServer();
    if (!server) {
      // Fallback to environment variable or default
      return import.meta.env.VITE_API_URL || 'http://localhost:12319';
    }
    return server.url;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const baseUrl = this.getBaseUrl();
    const url = `${baseUrl}${path}`;

    const defaultHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      let details: unknown;
      try {
        details = await response.json();
      } catch {
        details = await response.text();
      }

      throw {
        message: `API request failed: ${response.statusText}`,
        status: response.status,
        details,
      } as ApiError;
    }

    return response.json();
  }

  // === Health ===

  async health(includeMemory = false): Promise<HealthResponse> {
    const path = includeMemory ? '/api/health?include_memory=true' : '/api/health';
    const result = await this.request<HealthResponse>(path);
    
    // Update cache with health status
    const server = getActiveServer();
    if (server) {
      markServerConnected(server.id);
      updateActiveServerMetadata({
        cache: {
          lastHealthCheck: new Date().toISOString(),
          lastHealthStatus: result.status,
        },
      });
    }
    
    return result;
  }

  // === Providers ===

  async listProviders(): Promise<{ providers: ProviderInfo[] }> {
    return this.request('/api/providers');
  }

  async listModels(): Promise<{ models: ModelInfo[] }> {
    return this.request('/api/providers/models');
  }

  async checkProvidersHealth(): Promise<Record<string, { status: string }>> {
    return this.request('/api/providers/health');
  }

  // === MCP ===

  async listMCPServers(): Promise<{ servers: MCPServer[] }> {
    return this.request('/api/mcp/servers');
  }

  async listMCPTools(): Promise<{ tools: MCPTool[] }> {
    return this.request('/api/mcp/tools');
  }

  async listMCPResources(): Promise<{ resources: unknown[] }> {
    return this.request('/api/mcp/resources');
  }

  // === Agent ===

  async runAgent(request: AgentRunRequest): Promise<AgentRunResponse> {
    return this.request('/api/agent/run', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return this.request('/api/agent/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getAgentStatus(): Promise<{ status: string; active_sessions: number }> {
    return this.request('/api/agent/status');
  }

  // === Translation ===

  async translate(request: TranslateRequest): Promise<TranslateResponse> {
    return this.request('/api/tools/translate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async detectLanguage(text: string): Promise<LanguageDetectResponse> {
    return this.request('/api/tools/language-detect-20', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }
}

// Export singleton instance
export const api = new ApiClient();