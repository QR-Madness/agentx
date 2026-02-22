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
  profile_id?: string;  // Prompt profile to use
  temperature?: number;  // Model temperature (0.0-2.0, default 0.7)
  use_memory?: boolean;  // Enable memory retrieval (default true)
}

export interface ChatResponse {
  response: string;
  session_id: string;
  thinking?: string;  // Extracted thinking content
  has_thinking?: boolean;
  reasoning_trace?: ReasoningStep[];
  tokens_used?: number;
}

// === Prompt Types ===

export interface PromptSection {
  id: string;
  name: string;
  type: string;
  content: string;
  enabled: boolean;
  order: number;
}

export interface PromptProfile {
  id: string;
  name: string;
  description?: string;
  is_default: boolean;
  sections?: PromptSection[];
  sections_count?: number;
  enabled_sections?: number;
}

export interface GlobalPrompt {
  content: string;
  enabled: boolean;
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

// === Config Types ===

export interface ConfigUpdate {
  providers?: {
    lmstudio?: { base_url?: string; timeout?: number };
    anthropic?: { api_key?: string; base_url?: string };
    openai?: { api_key?: string; base_url?: string };
  };
  preferences?: {
    default_model?: string;
    default_reasoning_strategy?: string;
    enable_memory_by_default?: boolean;
  };
  llm_settings?: {
    default_temperature?: number;
    default_max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
  };
}

// === Memory Explorer Types ===

export interface MemoryEntity {
  id: string;
  name: string;
  type: string;
  channel: string;
  salience: number;
  description?: string;
  last_accessed: string;
  access_count: number;
  first_seen?: string;
  aliases?: string[];
}

export interface MemoryFact {
  id: string;
  claim: string;
  confidence: number;
  source: string;
  channel: string;
  source_turn_id?: string;
  created_at: string;
  promoted_from?: string;
  entity_ids: string[];
}

export interface MemoryStrategy {
  id: string;
  description: string;
  tool_sequence: string[];
  success_count: number;
  failure_count: number;
  success_rate: number;
  channel: string;
  last_used?: string;
}

export interface MemoryChannel {
  name: string;
  is_default: boolean;
  item_counts: {
    turns: number;
    entities: number;
    facts: number;
    strategies: number;
    goals: number;
  };
}

export interface PaginatedResponse<T> {
  total: number;
  page: number;
  limit: number;
  has_next: boolean;
  items?: T[];
}

export interface EntitiesResponse extends PaginatedResponse<MemoryEntity> {
  entities: MemoryEntity[];
}

export interface FactsResponse extends PaginatedResponse<MemoryFact> {
  facts: MemoryFact[];
}

export interface StrategiesResponse extends PaginatedResponse<MemoryStrategy> {
  strategies: MemoryStrategy[];
}

export interface EntityGraph {
  entity: MemoryEntity;
  facts: MemoryFact[];
  relationships: Array<{
    type: string;
    target: {
      id: string;
      name: string;
      type: string;
    };
  }>;
}

export interface MemoryStats {
  totals: {
    entities: number;
    facts: number;
    strategies: number;
    turns: number;
  };
  by_channel: Record<string, {
    entities: number;
    facts: number;
    strategies: number;
    turns: number;
  }>;
}

// === Job Types ===

export interface JobStatus {
  name: string;
  description: string;
  interval_minutes: number;
  status: 'idle' | 'running' | 'disabled';
  last_run: string | null;
  last_success: string | null;
  last_error: string | null;
  run_count: number;
  success_count: number;
  failure_count: number;
  avg_duration_ms: number;
  success_rate: number;
}

export interface JobHistory {
  timestamp: string;
  duration_ms: number;
  success: boolean;
  items_processed: number;
  metrics: Record<string, number>;
  error?: string;
}

export interface WorkerStatus {
  id: string;
  status: string;
  uptime_seconds: number;
  jobs_run: number;
  last_heartbeat: string;
}

export interface JobsResponse {
  jobs: JobStatus[];
  worker: WorkerStatus | null;
}

export interface JobDetailResponse {
  job: JobStatus;
  history: JobHistory[];
}

export interface JobRunResult {
  success: boolean;
  duration_ms: number;
  result: Record<string, number>;
  error?: string;
}

export interface ConsolidateResult {
  success: boolean;
  duration_ms: number;
  results: Record<string, Record<string, number>>;
  errors: string[];
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

  // === Prompts ===

  async listPromptProfiles(): Promise<{ profiles: PromptProfile[] }> {
    return this.request('/api/prompts/profiles');
  }

  async getPromptProfile(profileId: string): Promise<{ profile: PromptProfile; composed_prompt: string }> {
    return this.request(`/api/prompts/profiles/${profileId}`);
  }

  async getGlobalPrompt(): Promise<{ global_prompt: GlobalPrompt }> {
    return this.request('/api/prompts/global');
  }

  async updateGlobalPrompt(content: string, enabled = true): Promise<{ global_prompt: GlobalPrompt }> {
    return this.request('/api/prompts/global/update', {
      method: 'POST',
      body: JSON.stringify({ content, enabled }),
    });
  }

  async listPromptSections(): Promise<{ sections: PromptSection[] }> {
    return this.request('/api/prompts/sections');
  }

  async composePrompt(profileId?: string): Promise<{ system_prompt: string; profile_id: string }> {
    const params = profileId ? `?profile_id=${profileId}` : '';
    return this.request(`/api/prompts/compose${params}`);
  }

  async getMCPToolsPrompt(): Promise<{ mcp_tools_prompt: string; tools_count: number }> {
    return this.request('/api/prompts/mcp-tools');
  }

  // === Config ===

  /**
   * Update backend configuration (POST-only for security).
   * Persists to data/config.json and hot-reloads providers.
   */
  async updateConfig(config: ConfigUpdate): Promise<{ status: string; message: string; updated?: string[] }> {
    return this.request('/api/config/update', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // === Memory Explorer ===

  async listMemoryChannels(): Promise<{ channels: MemoryChannel[] }> {
    return this.request('/api/memory/channels');
  }

  async listMemoryEntities(params: {
    channel?: string;
    page?: number;
    limit?: number;
    search?: string;
    type?: string;
  } = {}): Promise<EntitiesResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.search) query.set('search', params.search);
    if (params.type) query.set('type', params.type);
    const queryString = query.toString();
    return this.request(`/api/memory/entities${queryString ? `?${queryString}` : ''}`);
  }

  async getEntityGraph(entityId: string, depth?: number): Promise<EntityGraph> {
    const query = depth ? `?depth=${depth}` : '';
    return this.request(`/api/memory/entities/${entityId}/graph${query}`);
  }

  async listMemoryFacts(params: {
    channel?: string;
    page?: number;
    limit?: number;
    min_confidence?: number;
    search?: string;
  } = {}): Promise<FactsResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.min_confidence !== undefined) query.set('min_confidence', params.min_confidence.toString());
    if (params.search) query.set('search', params.search);
    const queryString = query.toString();
    return this.request(`/api/memory/facts${queryString ? `?${queryString}` : ''}`);
  }

  async listMemoryStrategies(params: {
    channel?: string;
    page?: number;
    limit?: number;
  } = {}): Promise<StrategiesResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    const queryString = query.toString();
    return this.request(`/api/memory/strategies${queryString ? `?${queryString}` : ''}`);
  }

  async getMemoryStats(): Promise<MemoryStats> {
    return this.request('/api/memory/stats');
  }

  // === Jobs ===

  async listJobs(): Promise<JobsResponse> {
    return this.request('/api/jobs');
  }

  async getJob(name: string): Promise<JobDetailResponse> {
    return this.request(`/api/jobs/${encodeURIComponent(name)}`);
  }

  async runJob(name: string): Promise<JobRunResult> {
    return this.request(`/api/jobs/${encodeURIComponent(name)}/run`, {
      method: 'POST',
    });
  }

  async toggleJob(name: string, enabled: boolean): Promise<{ enabled: boolean; job: string }> {
    return this.request(`/api/jobs/${encodeURIComponent(name)}/toggle`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  }

  async consolidateNow(jobs?: string[]): Promise<ConsolidateResult> {
    return this.request('/api/memory/consolidate', {
      method: 'POST',
      body: JSON.stringify(jobs ? { jobs } : {}),
    });
  }

  // === Streaming ===

  /**
   * Stream a chat response using Server-Sent Events.
   * Returns an object with methods to control the stream.
   */
  streamChat(
    request: ChatRequest,
    callbacks: {
      onStart?: (data: { task_id: string; model: string }) => void;
      onChunk?: (content: string) => void;
      onDone?: (data: { task_id: string; thinking?: string; has_thinking?: boolean; total_time_ms: number; session_id: string }) => void;
      onError?: (error: string) => void;
    }
  ): { abort: () => void } {
    const baseUrl = this.getBaseUrl();
    const controller = new AbortController();
    
    fetch(`${baseUrl}/api/agent/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const reader = response.body?.getReader();
        if (!reader) throw new Error('No response body');
        
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          
          // Process complete SSE events
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';  // Keep incomplete line in buffer
          
          let eventType = '';
          let eventData = '';
          
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7);
            } else if (line.startsWith('data: ')) {
              eventData = line.slice(6);
              
              if (eventType && eventData) {
                try {
                  const data = JSON.parse(eventData);
                  
                  switch (eventType) {
                    case 'start':
                      callbacks.onStart?.(data);
                      break;
                    case 'chunk':
                      callbacks.onChunk?.(data.content);
                      break;
                    case 'done':
                      callbacks.onDone?.(data);
                      break;
                    case 'error':
                      callbacks.onError?.(data.error);
                      break;
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data:', e);
                }
                
                eventType = '';
                eventData = '';
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          callbacks.onError?.(error.message);
        }
      });
    
    return {
      abort: () => controller.abort(),
    };
  }
}

// Export singleton instance
export const api = new ApiClient();