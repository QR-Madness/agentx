/**
 * Request/response DTOs for the AgentX API, grouped by domain.
 */

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
  /** Whether the provider has the credentials/config to function.
   *  Note: this is a static check; for true reachability use ProvidersHealthResponse. */
  status: 'configured' | 'not_configured';
  models: string[];
  error?: string;
}

/** Response of `GET /api/providers/health` — async ping of every configured provider. */
export interface ProvidersHealthResponse {
  /** `healthy` if all providers passed; `degraded` if any failed. */
  status: 'healthy' | 'degraded';
  providers: Record<string, { status: 'healthy' | 'unhealthy'; error?: string }>;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  context_length?: number;
  context_window?: number;
  max_output_tokens?: number | null;
  supports_tools?: boolean;
  supports_vision?: boolean;
  supports_streaming?: boolean;
  supports_json_mode?: boolean;
  cost_per_1k_input?: number | null;
  cost_per_1k_output?: number | null;
  pricing_currency?: string;
  input_modalities?: string[];
  output_modalities?: string[];
  description?: string | null;
  capabilities?: string[];
}

export interface MCPServer {
  name: string;
  status: string;
  transport?: string;
  tools?: string[];
  tools_count?: number;
  resources_count?: number;
  // Phase 18.2: full config returned by GET /api/mcp/servers
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  url?: string | null;
  headers?: Record<string, string>;
  timeout?: number;
  auto_reconnect?: boolean;
  // Persisted desired-connected state; auto-managed by connect/disconnect and
  // restored on API restart. Preserved through edits, not a user form field.
  auto_connect?: boolean;
  tags?: string[];
  groups?: string[];
  allowed_agent_ids?: string[] | null;
}

export interface MCPServerConfigInput {
  transport: string;
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  url?: string | null;
  headers?: Record<string, string>;
  timeout?: number;
  auto_reconnect?: boolean;
  // Persisted desired-connected state; auto-managed by connect/disconnect and
  // restored on API restart. Preserved through edits, not a user form field.
  auto_connect?: boolean;
  tags?: string[];
  groups?: string[];
  allowed_agent_ids?: string[] | null;
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
  workflow_id?: string;  // Optional Agent Alloy workflow — supervisor takes over the chat
}

export interface ChatResponse {
  response: string;
  session_id: string;
  thinking?: string;  // Extracted thinking content
  has_thinking?: boolean;
  reasoning_trace?: ReasoningStep[];
  tokens_used?: number;
}

/** Per-subtask slice of a plan's Redis-tracked state. */
export interface PlanStatusSubtask {
  id: number;
  status: string;
  description?: string;
  result?: string;
  error?: string;
}

/** Response of GET /api/agent/plans/{id}/status. `found: false` on TTL expiry. */
export interface PlanStatusResponse {
  found: boolean;
  plan_id: string;
  session_id: string;
  status?: string;
  task?: string;
  complexity?: string;
  subtask_count?: number;
  completed_count?: number;
  cancel_requested?: boolean;
  // True when the plan is active with non-terminal work left and carries a
  // structural snapshot — i.e. POST .../resume would actually continue it.
  resumable?: boolean;
  // Seconds until the Redis snapshot expires (how long it stays resumable).
  ttl_seconds?: number | null;
  subtasks?: PlanStatusSubtask[];
}

/** Body for POST /api/agent/plans/{plan_id}/resume. */
export interface PlanResumeRequest {
  session_id: string;
  agent_profile_id?: string;
  model?: string;
  temperature?: number;
  use_memory?: boolean;
}

export interface BackgroundChatRequest {
  message: string;
  session_id?: string;
  profile_id?: string;
  agent_profile_id?: string;
  workflow_id?: string;
  model?: string;
  use_memory?: boolean;
}

export type BackgroundChatStatus = 'queued' | 'running' | 'done' | 'failed';

export interface BackgroundChatJob {
  job_id: string;
  user_id: string;
  message: string;
  status: BackgroundChatStatus | string;
  response?: string;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  session_id?: string;
  profile_id?: string;
  agent_profile_id?: string;
  workflow_id?: string;
  total_tokens?: string;
  total_time_ms?: string;
}

/** A detached chat run (survives tab close); used by recovery surfaces. */
export interface ActiveChatRun {
  run_id: string;
  status: BackgroundChatStatus | string;
  message: string;
  session_id?: string | null;
  created_at: string;
  updated_at: string;
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

/** One block in the layered global-prompt stack ("Prompt Stack"). */
export interface PromptLayer {
  id: string;
  title: string;
  kind: 'builtin' | 'custom';
  /** Shipped default (built-ins only); the sidecar. */
  default: string | null;
  default_version: number;
  /** User edit; null means "use the default". */
  override: string | null;
  base_version: number | null;
  /** override ?? default — what actually ships. */
  effective: string;
  enabled: boolean;
  order: number;
  /** Has an override that differs from the shipped default. */
  modified: boolean;
  /** A release changed the default underneath the user's override. */
  update_available: boolean;
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
  agentId: string;  // Docker-style stable identifier (used by Alloy workflows)
  avatar?: string;
  description?: string;
  // Up to 4 short trait/role labels shown as chips in the agent selector.
  tags?: string[];
  defaultModel?: string;
  temperature: number;
  promptProfileId?: string;
  systemPrompt?: string;
  reasoningStrategy: ReasoningStrategy;
  enableMemory: boolean;
  memoryChannel: string;
  enableTools: boolean;
  // Phase 18.9.x: per-tool gating. Entries are fully-qualified `server.tool`
  // (`_internal.<name>` for built-ins). `allowedTools === null` (or omitted)
  // means all tools enabled; blockedTools always wins.
  allowedTools?: string[] | null;
  blockedTools?: string[];
  // Phase 16.4: when true, other agents may delegate to this profile (ad-hoc
  // delegation). Defaults true server-side.
  availableForDelegation?: boolean;
  // Phase 16.6: optional ambassador section — present when this profile can act
  // as a parallel conversation interpreter.
  ambassador?: AmbassadorSection;
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

/** Phase 16.6 — the extra profile section that makes an agent an ambassador. */
export interface AmbassadorSection {
  enabled: boolean;
  briefingPrompt?: string;
  verbosity?: 'brief' | 'normal' | 'deep';
  speechModel?: string | null;
  voice?: string | null;
}

export interface AgentProfileCreate {
  id?: string;
  name: string;
  avatar?: string;
  description?: string;
  tags?: string[];
  default_model?: string;
  temperature?: number;
  prompt_profile_id?: string;
  system_prompt?: string;
  reasoning_strategy?: ReasoningStrategy;
  enable_memory?: boolean;
  memory_channel?: string;
  enable_tools?: boolean;
  allowed_tools?: string[] | null;
  blocked_tools?: string[];
  available_for_delegation?: boolean;
  // Phase 16.6 — ambassador section (snake_case body). null clears it.
  ambassador?: {
    enabled: boolean;
    briefing_prompt?: string;
    verbosity?: 'brief' | 'normal' | 'deep';
    speech_model?: string | null;
    voice?: string | null;
  } | null;
  is_default?: boolean;
}

// === Agent Alloy Types ===

export type AlloyMemberRole = 'supervisor' | 'specialist';

export interface AlloyWorkflowMember {
  agentId: string;
  role: AlloyMemberRole;
  delegationHint?: string;
}

export interface AlloyWorkflowRoute {
  fromAgentId: string;
  toAgentId: string;
  when: string;
}

export interface AlloyWorkflow {
  id: string;
  name: string;
  description?: string;
  supervisorAgentId: string;
  members: AlloyWorkflowMember[];
  routes: AlloyWorkflowRoute[];
  sharedChannel: string;
  canvas: Record<string, unknown>;
  createdAt: string | null;
  updatedAt: string | null;
}

export interface AlloyWorkflowCreate {
  id: string;
  name: string;
  description?: string;
  supervisorAgentId: string;
  members: AlloyWorkflowMember[];
  routes?: AlloyWorkflowRoute[];
  canvas?: Record<string, unknown>;
}

export type AlloyWorkflowUpdate = Partial<Omit<AlloyWorkflowCreate, 'id'>>;

// SSE event payloads emitted during a workflow-driven chat
export interface DelegationStartEvent {
  delegation_id: string;
  target_agent_id: string;
  tool_call_id: string;
  task: string;
  depth: number;
  supervisor_agent_id: string;
  shared_channel: string;
}

export interface DelegationChunkEvent {
  delegation_id: string;
  target_agent_id: string;
  content: string;
}

export interface DelegationCompleteEvent {
  delegation_id?: string;  // absent on pre-validation failures
  target_agent_id: string;
  tool_call_id: string;
  status: 'success' | 'failed';
  error: string | null;
  result_preview: string;
  // Per-delegation metrics (optional — absent from older servers).
  tokens_input?: number;
  tokens_output?: number;
  duration_ms?: number;
  cost_estimate?: number | null;
  cost_currency?: string | null;
  pricing_snapshot?: Record<string, unknown> | null;
}

export interface DelegationToolCallEvent {
  delegation_id: string;
  target_agent_id: string;
  tool: string;
  tool_call_id: string;
  arguments: Record<string, unknown>;
}

export interface DelegationToolResultEvent {
  delegation_id: string;
  target_agent_id: string;
  tool: string;
  tool_call_id: string;
  content: string;
  success: boolean;
  duration_ms?: number;
}

interface ServerWorkflowMember {
  agent_id: string;
  role: AlloyMemberRole;
  delegation_hint?: string;
}

interface ServerWorkflowRoute {
  from_agent_id: string;
  to_agent_id: string;
  when: string;
}

export interface ServerWorkflow {
  id: string;
  name: string;
  description: string | null;
  supervisor_agent_id: string;
  members: ServerWorkflowMember[];
  routes: ServerWorkflowRoute[];
  shared_channel: string;
  canvas: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
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
    vercel?: { api_key?: string; base_url?: string };
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
  planner?: {
    enabled?: boolean;
    model?: string | null;
    temperature?: number;
    max_tokens?: number;
    prompt_override?: string;
    complexity_threshold?: 'simple' | 'moderate' | 'complex';
    /** Read-only: the built-in decomposition prompt, used to seed the editor. */
    decompose_default?: string;
  };
  search?: {
    backend?: 'tavily' | 'brave';
    fallback_enabled?: boolean;
    max_results?: number;
    cache_ttl_seconds?: number;
    /** Omit when unchanged (redacted) so the stored key isn't overwritten. */
    tavily_api_key?: string;
    brave_api_key?: string;
  };
  alloy?: {
    allow_adhoc_delegation?: boolean;
    max_parallel_delegations?: number;
    max_delegation_depth?: number;
  };
  ambassador?: {
    enabled?: boolean;
    profile_id?: string | null;
    model?: string | null;
    max_context_turns?: number;
    max_tokens?: number;
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

/** A lightweight entity reference carried on a fact's ABOUT links (#538). */
export interface MemoryFactEntity {
  id: string;
  name: string;
  type: string;
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
  /** Entities this fact is ABOUT — {id,name,type}; powers the Mentioned entities section. */
  entities?: MemoryFactEntity[];
}

export interface MemoryFactPatch {
  claim?: string;
  confidence?: number;
  source?: string;
  temporal_context?: 'current' | 'past' | 'future' | null;
}

export interface FactForgetResult {
  success: boolean;
  mode: 'soft' | 'hard';
  fact_id: string;
  fact?: MemoryFact;
}

export interface FactProvenanceOrigin {
  conversation_id: string;
  role: string;
  timestamp: string;
  snippet: string;
}

export interface FactProvenance {
  success: boolean;
  fact_id: string;
  claim?: string;
  source?: string | null;
  source_turn_id?: string | null;
  origin: FactProvenanceOrigin | null;
}

export interface MemoryEntityPatch {
  name?: string;
  type?: string;
  description?: string | null;
  aliases?: string[];
  properties?: Record<string, unknown>;
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

/**
 * A distilled procedure (procedural memory): the "how we work here" delta the
 * `distill_procedures` job mints from corrections/steers + explicit user rules.
 */
export interface MemoryProcedure {
  id: string;
  trigger: string;
  body: string;
  rationale: string;
  scope: string;
  agent_id?: string | null;
  strength: number;
  signal_kinds: string[];
  /** `cand:<id>` and `conv:<conversation_id>` provenance refs. */
  evidence_refs: string[];
  channel: string;
  last_reinforced?: string;
}

export interface ProceduresResponse extends PaginatedResponse<MemoryProcedure> {
  procedures: MemoryProcedure[];
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

/**
 * Round-trippable memory export envelope (POST /api/memory/export).
 *
 * Treated as an opaque blob by the client — it's downloaded as a file and
 * round-tripped back to POST /api/memory/import verbatim. `counts()` server-side
 * mirrors the collection lengths; the client only needs the discriminating
 * header fields for display.
 */
export interface MemoryExport {
  schema_version: number;
  exported_at: string;
  user_id: string;
  channel: string | null;
  embedder: { provider_model: string; dimensions: number };
  conversations: unknown[];
  turns: unknown[];
  entities: unknown[];
  facts: unknown[];
  goals: unknown[];
  strategies: unknown[];
  tool_invocations: unknown[];
  pg_conversation_logs: unknown[];
  pg_tool_invocations: unknown[];
}

/** Per-type result of a memory import (POST /api/memory/import). */
export interface MemoryImportResult {
  mode: 'merge' | 'replace';
  channel: string | null;
  recomputed_embeddings: number;
  imported: Record<string, { created: number; total: number }>;
  pg_conversation_logs: number;
  pg_tool_invocations: number;
}

/** Aggregated token/cost/latency usage from conversation_logs (GET /api/metrics/usage). */
export interface UsageMetrics {
  totals: {
    turns: number;
    tokens_input: number;
    tokens_output: number;
    tokens_total: number;
    cost_total: number;
    cost_currency: string;
    avg_latency_ms: number;
  };
  by_model: Array<{
    model: string;
    turns: number;
    tokens_input: number;
    tokens_output: number;
    tokens_total: number;
    cost_total: number;
  }>;
  by_agent: Array<{
    agent_id: string;       // Docker-style slug, or '_default' when null
    turns: number;
    tokens_input: number;
    tokens_output: number;
    tokens_total: number;
    cost_total: number;
  }>;
  daily: Array<{
    date: string;        // YYYY-MM-DD
    turns: number;
    tokens_total: number;
    cost_total: number;
  }>;
  days: number;
  unavailable?: boolean;  // Set when databases are offline
}

// === Checkpoints (model-authored conversation anchors) ===

export interface Checkpoint {
  summary: string;
  decisions: string[];
  next_step: string;
  created_at: string;
}

export interface CheckpointsResponse {
  checkpoints: Checkpoint[];
  count: number;
}

// === User history (manual recall browse) ===

export interface UserHistoryTurn {
  timestamp: string;
  conversation_id: string | null;
  content: string;
}

export interface UserHistoryFact {
  claim: string;
  confidence: number;
}

export interface UserHistoryResponse {
  topic: string | null;
  summary?: string;
  turn_count: number;
  user_turns: UserHistoryTurn[];
  facts: UserHistoryFact[];
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
  // Bulk default for every stage model: a stage left empty inherits this; if this
  // is empty too, stages inherit the global default chat model.
  feature_default_model: string;

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
  relevance_filter_temperature: number;
  relevance_filter_max_tokens: number;
  relevance_filter_prompt: string;

  // Combined relevance + extraction (handles ~75% of consolidation traffic)
  combined_extraction_model: string;
  combined_extraction_temperature: number;
  combined_extraction_max_tokens: number;

  // Trajectory compression (consolidates older tool-call rounds into a Knowledge block)
  trajectory_compression_enabled: boolean;
  trajectory_compression_model: string;
  trajectory_compression_temperature: number;
  trajectory_compression_max_tokens: number;
  trajectory_compression_threshold_ratio: number;
  trajectory_compression_preserve_recent_rounds: number;

  // Entity linking
  entity_linking_enabled: boolean;
  entity_linking_similarity_threshold: number;
  entity_linking_model: string;
  entity_linking_use_llm_disambiguation: boolean;

  // Quality thresholds
  fact_confidence_threshold: number;
  promotion_min_confidence: number;

  // Job intervals
  job_consolidate_interval: number;
  job_promote_interval: number;
  job_entity_linking_interval: number;

  // Experimental
  contradiction_detection_enabled: boolean;
  contradiction_model: string;
  contradiction_temperature: number;
  contradiction_max_tokens: number;
  correction_detection_enabled: boolean;
  correction_model: string;
  correction_temperature: number;
  correction_max_tokens: number;

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
