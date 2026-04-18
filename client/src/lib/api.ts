/**
 * API client for AgentX backend
 * All calls are routed through the active server configuration
 */

import { getActiveServer, markServerConnected, updateActiveServerMetadata, getAuthToken, clearAuthToken } from './storage';

// === Version Constants ===

/** Client version from package.json (injected by Vite) */
export const CLIENT_VERSION = __APP_VERSION__;

/** Protocol version - must match server exactly */
export const CLIENT_PROTOCOL_VERSION = 1;

/**
 * Compare two semver versions.
 * Returns: -1 if a < b, 0 if a === b, 1 if a > b
 */
export function compareSemver(a: string, b: string): number {
  const partsA = a.split('.').map(Number);
  const partsB = b.split('.').map(Number);

  for (let i = 0; i < 3; i++) {
    const numA = partsA[i] || 0;
    const numB = partsB[i] || 0;
    if (numA < numB) return -1;
    if (numA > numB) return 1;
  }
  return 0;
}

// === Types ===

export interface ApiError {
  message: string;
  status: number;
  details?: unknown;
}

// === Auth Types ===

export interface AuthStatusResponse {
  auth_required: boolean;
  setup_required: boolean;
  auth_bypass_active: boolean;
}

export interface AuthLoginRequest {
  username: string;
  password: string;
}

export interface AuthLoginResponse {
  token: string;
  expires_at: string;
  username: string;
}

export interface AuthSessionResponse {
  user_id: number;
  username: string;
  session_created: string;
  last_active: string;
}

export interface AuthSetupRequest {
  password: string;
  confirm_password: string;
}

export interface AuthChangePasswordRequest {
  old_password: string;
  new_password: string;
}

// === Version Types ===

export interface VersionInfo {
  version: string;
  protocol_version: number;
  min_client_version: string;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  protocol_version?: number;
  min_client_version?: string;
  cluster?: string;
  api: { status: string };
  translation?: { status: string; models?: Record<string, unknown> };
  memory?: {
    neo4j?: { status: string };
    postgres?: { status: string };
    redis?: { status: string };
  };
  storage?: {
    postgres_size_mb: number | null;
    neo4j_size_mb: number | null;
    redis_memory_mb: number | null;
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
  transport?: string;
  tools?: string[];
  tools_count?: number;
  resources_count?: number;
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
  agent_profile_id?: string;  // Agent profile to use (includes model, temperature, etc.)
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

// === Conversation History Types ===

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  preview: string;
  message_count: number;
  channel: string;
  created_at: string | null;
  last_message_at: string | null;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ServerMessage {
  role: 'user' | 'assistant' | 'system' | 'tool' | 'tool_call' | 'tool_result';
  content: string;
  timestamp: string | null;
  turn_index: number;
  metadata?: Record<string, unknown>;
}

export interface ConversationMessagesResponse {
  conversation_id: string;
  messages: ServerMessage[];
  message_count: number;
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

// === Prompt Template Types ===

export type TemplateType = 'system' | 'user' | 'snippet';

export interface PromptTemplate {
  id: string;
  name: string;
  content: string;
  defaultContent: string;
  tags: string[];
  placeholders: string[];
  type: TemplateType;
  isBuiltin: boolean;
  description?: string;
  hasModifications: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface PromptTemplateCreate {
  name: string;
  content: string;
  tags?: string[];
  placeholders?: string[];
  type?: TemplateType;
  description?: string;
}

export interface PromptTemplateUpdate {
  name?: string;
  content?: string;
  tags?: string[];
  placeholders?: string[];
  description?: string;
}

export interface TemplateTag {
  name: string;
  count: number;
}

// === Agent Profile Types ===

export type ReasoningStrategy = 'auto' | 'cot' | 'tot' | 'react' | 'reflection';

export interface AgentProfile {
  id: string;
  name: string;
  avatar?: string;
  description?: string;
  defaultModel?: string;
  temperature: number;
  promptProfileId?: string;
  systemPrompt?: string;
  reasoningStrategy: ReasoningStrategy;
  enableMemory: boolean;
  memoryChannel: string;
  enableTools: boolean;
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface AgentProfileCreate {
  id?: string;
  name: string;
  avatar?: string;
  description?: string;
  default_model?: string;
  temperature?: number;
  prompt_profile_id?: string;
  system_prompt?: string;
  reasoning_strategy?: ReasoningStrategy;
  enable_memory?: boolean;
  memory_channel?: string;
  enable_tools?: boolean;
  is_default?: boolean;
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
    openrouter?: { api_key?: string; site_url?: string; app_name?: string };
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
  prompt_enhancement?: {
    enabled?: boolean;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    system_prompt?: string;
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
  unavailable?: boolean;  // Set when databases are offline
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
  consolidation_active: ConsolidationActiveInfo | null;
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

/** Data for each SSE event type from /api/memory/consolidate/stream */
export interface ConsolidationStreamEvent {
  run_id: string;
  timestamp: string;
}

export interface ConsolidationStartEvent extends ConsolidationStreamEvent {
  jobs: string[];
  total_jobs: number;
  triggered_by: string;
}

export interface ConsolidationJobStartEvent extends ConsolidationStreamEvent {
  job: string;
  index: number;
  total: number;
}

export interface ConsolidationProgressEvent extends ConsolidationStreamEvent {
  job: string;
  stage: string;
  conversation?: string;
  conversation_id?: string;
  turns?: number;
  entities?: number;
  facts?: number;
  relationships?: number;
  conversations_found?: number;
  total_in_neo4j?: number;
  entities_stored?: number;
  facts_stored?: number;
  relationships_stored?: number;
  conversations_processed?: number;
  duration_ms?: number;
}

export interface ConsolidationJobDoneEvent extends ConsolidationStreamEvent {
  job: string;
  success: boolean;
  duration_ms: number;
  result: Record<string, unknown>;
}

export interface ConsolidationDoneEvent extends ConsolidationStreamEvent {
  success: boolean;
  duration_ms: number;
  results: Record<string, unknown>;
  errors: string[];
}

export interface ConsolidationActiveInfo {
  run_id: string;
  started_at: string;
  jobs: string[];
  triggered_by: string;
}

export interface ConsolidationSettings {
  // Extraction
  extraction_enabled: boolean;
  extraction_provider: string;
  extraction_model: string;
  extraction_temperature: number;
  extraction_max_tokens: number;
  extraction_condense_facts: boolean;
  extraction_system_prompt: string;

  // Relevance filter
  relevance_filter_enabled: boolean;
  relevance_filter_provider: string;
  relevance_filter_model: string;
  relevance_filter_max_tokens: number;
  relevance_filter_prompt: string;

  // Entity linking
  entity_linking_enabled: boolean;
  entity_linking_similarity_threshold: number;

  // Quality thresholds
  fact_confidence_threshold: number;
  promotion_min_confidence: number;

  // Job intervals
  job_consolidate_interval: number;
  job_promote_interval: number;
  job_entity_linking_interval: number;

  // Experimental
  contradiction_detection_enabled: boolean;
  correction_detection_enabled: boolean;

  // Read-only (from server)
  entity_types: string[];
  relationship_types: string[];
  default_extraction_prompt: string;
  default_relevance_prompt: string;
}

export interface RecallSettings {
  // Feature toggles
  recall_enable_hybrid: boolean;
  recall_enable_entity_centric: boolean;
  recall_enable_query_expansion: boolean;
  recall_enable_hyde: boolean;
  recall_enable_self_query: boolean;

  // Hybrid search settings
  recall_hybrid_bm25_weight: number;
  recall_hybrid_vector_weight: number;
  recall_hybrid_rrf_k: number;

  // Entity-centric settings
  recall_entity_similarity_threshold: number;
  recall_entity_max_entities: number;
  recall_entity_graph_depth: number;

  // Query expansion settings
  recall_expansion_max_variants: number;

  // HyDE settings
  recall_hyde_provider: string;
  recall_hyde_model: string;
  recall_hyde_temperature: number;
  recall_hyde_max_tokens: number;

  // Self-query settings
  recall_self_query_provider: string;
  recall_self_query_model: string;
  recall_self_query_temperature: number;
  recall_self_query_max_tokens: number;
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
    options: RequestInit = {},
    skipAuth = false
  ): Promise<T> {
    const baseUrl = this.getBaseUrl();
    const url = `${baseUrl}${path}`;

    const defaultHeaders: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    // Add auth token if available (unless skipped for auth endpoints)
    if (!skipAuth) {
      const token = getAuthToken();
      if (token) {
        defaultHeaders['X-Auth-Token'] = token;
      }
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      // Handle 401 Unauthorized - clear token and notify app
      if (response.status === 401) {
        clearAuthToken();
        window.dispatchEvent(new CustomEvent('agentx:auth-required'));
      }

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

  async health(includeMemory = false, includeStorage = false): Promise<HealthResponse> {
    const params = new URLSearchParams();
    if (includeMemory) params.set('include_memory', 'true');
    if (includeStorage) params.set('include_storage', 'true');
    const path = params.toString() ? `/api/health?${params}` : '/api/health';
    const result = await this.request<HealthResponse>(path, {}, true); // Skip auth for health

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

  // === Version ===

  async version(): Promise<VersionInfo> {
    return this.request<VersionInfo>('/api/version', {}, true); // Skip auth for version
  }

  // === Authentication ===

  async authStatus(): Promise<AuthStatusResponse> {
    return this.request<AuthStatusResponse>('/api/auth/status', {}, true);
  }

  async login(credentials: AuthLoginRequest): Promise<AuthLoginResponse> {
    return this.request<AuthLoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    }, true);
  }

  async logout(): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/logout', {
      method: 'POST',
    });
  }

  async authSession(): Promise<AuthSessionResponse> {
    return this.request<AuthSessionResponse>('/api/auth/session');
  }

  async authSetup(data: AuthSetupRequest): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/setup', {
      method: 'POST',
      body: JSON.stringify(data),
    }, true);
  }

  async changePassword(data: AuthChangePasswordRequest): Promise<{ message: string }> {
    return this.request<{ message: string }>('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // === Providers ===

  async listProviders(): Promise<{ providers: ProviderInfo[] }> {
    return this.request('/api/providers');
  }

  async listModels(): Promise<{ models: ModelInfo[] }> {
    return this.request('/api/providers/models', { signal: AbortSignal.timeout(10_000) });
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

  async connectMCPServer(server: string): Promise<{ status: string; server: string; tools_count: number; resources_count: number }> {
    return this.request('/api/mcp/connect', {
      method: 'POST',
      body: JSON.stringify({ server }),
    });
  }

  async connectAllMCPServers(): Promise<{ results: Record<string, { status: string; error?: string }> }> {
    return this.request('/api/mcp/connect', {
      method: 'POST',
      body: JSON.stringify({ all: true }),
    });
  }

  async disconnectMCPServer(server: string): Promise<{ status: string; server: string }> {
    return this.request('/api/mcp/disconnect', {
      method: 'POST',
      body: JSON.stringify({ server }),
    });
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

  // === Tool Output Storage ===

  async listToolOutputs(pattern?: string): Promise<{
    outputs: Array<{
      key: string;
      tool_name: string;
      tool_call_id: string;
      size_chars: number;
      stored_at: string;
    }>;
    count: number;
  }> {
    const params = pattern ? `?pattern=${encodeURIComponent(pattern)}` : '';
    return this.request(`/api/tool-outputs${params}`);
  }

  async getToolOutput(key: string, options?: {
    offset?: number;
    limit?: number;
    metadataOnly?: boolean;
  }): Promise<{
    key: string;
    tool_name: string;
    tool_call_id: string;
    content?: string;
    offset?: number;
    limit?: number;
    total_size?: number;
    size_chars?: number;
    stored_at: string;
  }> {
    const params = new URLSearchParams();
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.metadataOnly) params.set('metadata_only', 'true');
    const query = params.toString() ? `?${params}` : '';
    return this.request(`/api/tool-outputs/${encodeURIComponent(key)}${query}`);
  }

  async deleteToolOutput(key: string): Promise<{ deleted: boolean; key: string }> {
    return this.request(`/api/tool-outputs/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    });
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

  // === Prompt Templates ===

  async listPromptTemplates(params: {
    type?: TemplateType;
    tag?: string;
    search?: string;
  } = {}): Promise<{ templates: PromptTemplate[]; total: number }> {
    const query = new URLSearchParams();
    if (params.type) query.set('type', params.type);
    if (params.tag) query.set('tag', params.tag);
    if (params.search) query.set('search', params.search);
    const queryString = query.toString();

    const response = await this.request<{ templates: Array<{
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }>; total: number }>(`/api/prompts/templates${queryString ? `?${queryString}` : ''}`);

    // Transform snake_case to camelCase
    return {
      templates: response.templates.map(t => ({
        id: t.id,
        name: t.name,
        content: t.content,
        defaultContent: t.default_content,
        tags: t.tags,
        placeholders: t.placeholders,
        type: t.type as TemplateType,
        isBuiltin: t.is_builtin,
        description: t.description,
        hasModifications: t.has_modifications,
        createdAt: t.created_at || '',
        updatedAt: t.updated_at || '',
      })),
      total: response.total,
    };
  }

  async getPromptTemplate(id: string): Promise<{ template: PromptTemplate }> {
    const response = await this.request<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/prompts/templates/${encodeURIComponent(id)}`);

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
    };
  }

  async createPromptTemplate(template: PromptTemplateCreate): Promise<{ template: PromptTemplate; message: string }> {
    const response = await this.request<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>('/api/prompts/templates', {
      method: 'POST',
      body: JSON.stringify(template),
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  }

  async updatePromptTemplate(id: string, updates: PromptTemplateUpdate): Promise<{ template: PromptTemplate; message: string }> {
    const response = await this.request<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>(`/api/prompts/templates/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  }

  async deletePromptTemplate(id: string): Promise<{ deleted: boolean; message: string }> {
    return this.request(`/api/prompts/templates/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  }

  async resetPromptTemplate(id: string): Promise<{ template: PromptTemplate; message: string }> {
    const response = await this.request<{ template: {
      id: string;
      name: string;
      content: string;
      default_content: string;
      tags: string[];
      placeholders: string[];
      type: string;
      is_builtin: boolean;
      description?: string;
      has_modifications: boolean;
      created_at?: string;
      updated_at?: string;
    }; message: string }>(`/api/prompts/templates/${encodeURIComponent(id)}/reset`, {
      method: 'POST',
    });

    return {
      template: {
        id: response.template.id,
        name: response.template.name,
        content: response.template.content,
        defaultContent: response.template.default_content,
        tags: response.template.tags,
        placeholders: response.template.placeholders,
        type: response.template.type as TemplateType,
        isBuiltin: response.template.is_builtin,
        description: response.template.description,
        hasModifications: response.template.has_modifications,
        createdAt: response.template.created_at || '',
        updatedAt: response.template.updated_at || '',
      },
      message: response.message,
    };
  }

  async listPromptTemplateTags(): Promise<{ tags: TemplateTag[]; total: number }> {
    return this.request('/api/prompts/templates/tags');
  }

  async enhancePrompt(prompt: string, context?: Array<{ role: string; content: string }>): Promise<{
    enhanced_prompt: string;
    original_length: number;
    enhanced_length: number;
    model: string;
  }> {
    return this.request('/api/prompts/enhance', {
      method: 'POST',
      body: JSON.stringify({ prompt, context }),
    });
  }

  // === Agent Profiles ===

  async listAgentProfiles(): Promise<{ profiles: AgentProfile[] }> {
    const response = await this.request<{ profiles: Array<{
      id: string;
      name: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    }> }>('/api/agent/profiles');
    // Transform snake_case to camelCase
    return {
      profiles: response.profiles.map(p => ({
        id: p.id,
        name: p.name,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      })),
    };
  }

  async getAgentProfile(id: string): Promise<{ profile: AgentProfile }> {
    const response = await this.request<{ profile: {
      id: string;
      name: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/agent/profiles/${encodeURIComponent(id)}`);
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  }

  async createAgentProfile(profile: AgentProfileCreate): Promise<{ profile: AgentProfile }> {
    const response = await this.request<{ profile: {
      id: string;
      name: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>('/api/agent/profiles', {
      method: 'POST',
      body: JSON.stringify(profile),
    });
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  }

  async updateAgentProfile(id: string, updates: Partial<AgentProfileCreate>): Promise<{ profile: AgentProfile }> {
    const response = await this.request<{ profile: {
      id: string;
      name: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/agent/profiles/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  }

  async deleteAgentProfile(id: string): Promise<{ deleted: boolean }> {
    return this.request(`/api/agent/profiles/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  }

  async setDefaultAgentProfile(id: string): Promise<{ default_profile_id: string }> {
    return this.request(`/api/agent/profiles/${encodeURIComponent(id)}/set-default`, {
      method: 'POST',
    });
  }

  // === Config ===

  /**
   * Get backend configuration.
   * Returns current runtime config with sensitive values redacted.
   */
  async getConfig(): Promise<Record<string, unknown>> {
    return this.request('/api/config');
  }

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

  async getContextLimits(): Promise<{
    lmstudio: { context_window: number; max_output_tokens: number };
    models: Record<string, { context_window: number; max_output_tokens: number }>;
  }> {
    return this.request('/api/config/context-limits');
  }

  async updateContextLimits(limits: {
    lmstudio?: { context_window?: number; max_output_tokens?: number };
    models?: Record<string, { context_window?: number; max_output_tokens?: number }>;
  }): Promise<{ status: string; updated: string[] }> {
    return this.request('/api/config/context-limits', {
      method: 'POST',
      body: JSON.stringify(limits),
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

  /**
   * Stream consolidation progress via SSE.
   * POST triggers + watches; GET watches only (reconnection).
   */
  streamConsolidate(
    options: {
      trigger?: boolean;
      jobs?: string[];
    },
    callbacks: {
      onStart?: (data: ConsolidationStartEvent) => void;
      onJobStart?: (data: ConsolidationJobStartEvent) => void;
      onProgress?: (data: ConsolidationProgressEvent) => void;
      onJobDone?: (data: ConsolidationJobDoneEvent) => void;
      onDone?: (data: ConsolidationDoneEvent) => void;
      onIdle?: () => void;
      onError?: (error: string) => void;
    }
  ): { abort: () => void } {
    const baseUrl = this.getBaseUrl();
    const controller = new AbortController();
    const method = options.trigger ? 'POST' : 'GET';
    const body = options.trigger && options.jobs
      ? JSON.stringify({ jobs: options.jobs })
      : options.trigger ? '{}' : undefined;

    fetch(`${baseUrl}/api/memory/consolidate/stream`, {
      method,
      headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {},
      body,
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
        const eventState = { type: '', data: '' };

        const processLines = (lines: string[]): boolean => {
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventState.type = line.slice(7);
            } else if (line.startsWith('data: ')) {
              eventState.data = line.slice(6);

              if (eventState.type && eventState.data) {
                try {
                  const data = JSON.parse(eventState.data);

                  switch (eventState.type) {
                    case 'start':
                      callbacks.onStart?.(data);
                      break;
                    case 'job_start':
                      callbacks.onJobStart?.(data);
                      break;
                    case 'progress':
                      callbacks.onProgress?.(data);
                      break;
                    case 'job_done':
                      callbacks.onJobDone?.(data);
                      break;
                    case 'done':
                      callbacks.onDone?.(data);
                      return true;
                    case 'idle':
                      callbacks.onIdle?.();
                      return true;
                    case 'error':
                      callbacks.onError?.(data.error);
                      return true;
                  }
                } catch (e) {
                  console.error('Failed to parse consolidation SSE data:', e);
                }

                eventState.type = '';
                eventState.data = '';
              }
            }
          }
          return false;
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          if (processLines(lines)) {
            controller.abort();
            return;
          }
        }

        if (buffer.trim()) {
          processLines(buffer.split('\n'));
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          callbacks.onError?.(error.message);
        }
      });

    return { abort: () => controller.abort() };
  }

  async getConsolidationSettings(): Promise<ConsolidationSettings> {
    return this.request('/api/memory/settings');
  }

  async updateConsolidationSettings(settings: Partial<ConsolidationSettings>): Promise<{ success: boolean; updated: string[] }> {
    return this.request('/api/memory/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  }

  async getRecallSettings(): Promise<RecallSettings> {
    return this.request('/api/memory/recall-settings');
  }

  async updateRecallSettings(settings: Partial<RecallSettings>): Promise<{ success: boolean; updated: string[] }> {
    return this.request('/api/memory/recall-settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  }

  async resetMemory(deleteMemories = false): Promise<{ conversations_reset: number; memories_deleted?: number; success: boolean }> {
    return this.request('/api/memory/reset', {
      method: 'POST',
      body: JSON.stringify({ delete_memories: deleteMemories }),
    });
  }

  // === Conversation History ===

  async listConversations(params?: {
    limit?: number;
    offset?: number;
    channel?: string;
  }): Promise<ConversationListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.offset) searchParams.set('offset', String(params.offset));
    if (params?.channel) searchParams.set('channel', params.channel);
    const qs = searchParams.toString();
    return this.request(`/api/conversations${qs ? `?${qs}` : ''}`);
  }

  async getConversationMessages(conversationId: string): Promise<ConversationMessagesResponse> {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
  }

  async deleteConversation(conversationId: string): Promise<{ message: string; deleted: Record<string, number> }> {
    return this.request(`/api/memory/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    });
  }

  async clearStuckJobs(): Promise<{ success: boolean; cleared_jobs: string[]; message: string }> {
    return this.request('/api/jobs/clear-stuck', {
      method: 'POST',
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
      onStart?: (data: {
        task_id: string;
        model: string;
        model_display_name?: string;
        profile_name?: string;
        agent_name?: string;
        context_window?: number;
        max_output_tokens?: number;
      }) => void;
      onChunk?: (content: string) => void;
      onMemoryContext?: (data: {
        facts: Array<{ claim: string; confidence: number; source?: string }>;
        entities: Array<{ name: string; type: string }>;
        relevant_turns: Array<{ timestamp: string; role: string; content: string }>;
        query: string;
      }) => void;
      onToolCall?: (data: {
        tool: string;
        tool_call_id: string;
        arguments: Record<string, unknown>;
      }) => void;
      onToolResult?: (data: {
        tool: string;
        tool_call_id: string;
        content: string;
        success: boolean;
        duration_ms: number;
      }) => void;
      onDone?: (data: {
        task_id: string;
        thinking?: string;
        has_thinking?: boolean;
        total_time_ms: number;
        session_id: string;
        profile_name?: string;
        agent_name?: string;
        tokens_input?: number;
        tokens_output?: number;
        context_window?: number;
        context_used?: number;
      }) => void;
      onPlanStart?: (data: {
        plan_id: string;
        task: string;
        subtask_count: number;
        complexity: string;
      }) => void;
      onSubtaskStart?: (data: {
        plan_id: string;
        subtask_id: number;
        description: string;
        type: string;
        progress: number;
      }) => void;
      onSubtaskComplete?: (data: {
        plan_id: string;
        subtask_id: number;
        result_preview: string;
        progress: number;
      }) => void;
      onSubtaskFailed?: (data: {
        plan_id: string;
        subtask_id: number;
        error: string;
        progress: number;
      }) => void;
      onPlanComplete?: (data: {
        plan_id: string;
        subtask_count: number;
        completed_count: number;
        total_time_ms: number;
      }) => void;
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
        
        // Helper to process SSE lines
        const processLines = (lines: string[], eventState: { type: string; data: string }) => {
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventState.type = line.slice(7);
            } else if (line.startsWith('data: ')) {
              eventState.data = line.slice(6);

              if (eventState.type && eventState.data) {
                try {
                  const data = JSON.parse(eventState.data);

                  switch (eventState.type) {
                    case 'start':
                      callbacks.onStart?.(data);
                      break;
                    case 'chunk':
                      callbacks.onChunk?.(data.content);
                      break;
                    case 'memory_context':
                      console.log('[API] memory_context received:', data.facts?.length, 'facts,', data.entities?.length, 'entities');
                      callbacks.onMemoryContext?.(data);
                      break;
                    case 'tool_call':
                      callbacks.onToolCall?.(data);
                      break;
                    case 'tool_result':
                      callbacks.onToolResult?.(data);
                      break;
                    case 'plan_start':
                      callbacks.onPlanStart?.(data);
                      break;
                    case 'subtask_start':
                      callbacks.onSubtaskStart?.(data);
                      break;
                    case 'subtask_complete':
                      callbacks.onSubtaskComplete?.(data);
                      break;
                    case 'subtask_failed':
                      callbacks.onSubtaskFailed?.(data);
                      break;
                    case 'plan_complete':
                      callbacks.onPlanComplete?.(data);
                      break;
                    case 'done':
                      callbacks.onDone?.(data);
                      break;
                    case 'close':
                      // Server signals stream is complete - abort to close connection
                      controller.abort();
                      return true; // Signal to stop processing
                    case 'error':
                      callbacks.onError?.(data.error);
                      break;
                  }
                } catch (e) {
                  console.error('Failed to parse SSE data:', e);
                }

                eventState.type = '';
                eventState.data = '';
              }
            }
          }
          return false; // Continue processing
        };

        const eventState = { type: '', data: '' };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE events
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';  // Keep incomplete line in buffer

          if (processLines(lines, eventState)) return;
        }

        // Process any remaining data in buffer after stream ends
        if (buffer.trim()) {
          const finalLines = buffer.split('\n');
          processLines(finalLines, eventState);
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