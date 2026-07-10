import { getBaseUrl, registerStreamController } from './core';
import { getAuthToken, clearAuthToken, getActiveGatewayToken } from '../storage';
import { classifyStatus, apiErrorMessage, backendErrorMessage } from './errors';
import type { ApiError } from './errors';
import type { ChatRequest, DelegationChunkEvent, DelegationCompleteEvent, DelegationStartEvent, DelegationToolCallEvent, DelegationToolResultEvent, PlanResumeRequest } from './types';
import type { ExhibitWire } from '../exhibits';

/**
 * Callbacks for an SSE chat stream. Shared by `streamChat` (initial POST) and
 * `attachChatRun` (re-attach GET) since a re-attach replays the same event set.
 */
export interface StreamCallbacks {
  onStart?: (data: {
    task_id: string;
    session_id?: string;
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
    /** A rolling summary covers turns older than the verbatim window. */
    context_summarized?: boolean;
    /** Verbatim turns the ledger dropped this turn (0 when everything fit). */
    context_dropped_turns?: number;
    model?: string;
    provider?: string;
    cost_estimate?: number | null;
    cost_currency?: string | null;
    pricing_snapshot?: { cost_per_1k_input: number; cost_per_1k_output: number } | null;
    /** Final finish reason; "length" means the answer hit the token limit. */
    finish_reason?: string | null;
    /** True when the answer was cut off by max_tokens (finish_reason "length"). */
    truncated?: boolean;
    /** Thinking pattern the turn ran with (native/cot/…); absent = none. */
    thinking_pattern?: string | null;
    /** Turn ran in Research Mode (thinking patterns are suppressed there). */
    research?: boolean;
  }) => void;
  /** First event of a detached run — carries the run_id to persist + re-attach to. */
  onRunStarted?: (data: { run_id: string }) => void;
  /** Re-attach found no live run buffer (TTL expired) — fall back to history restore. */
  onRunMissing?: (data: { run_id: string; reason?: string }) => void;
  onPlanStart?: (data: {
    plan_id: string;
    task: string;
    subtask_count: number;
    complexity: string;
  }) => void;
  /** First event of a resumed run — a snapshot of the already-done subtasks so
   *  a fresh client (that never saw plan_start) can paint the plan card. */
  onPlanResumed?: (data: {
    plan_id: string;
    task: string;
    subtask_count: number;
    complexity: string;
    completed_count: number;
    progress: number;
    subtasks: Array<{
      subtask_id: number;
      description: string;
      type: string;
      status: string;
      result_preview: string;
    }>;
  }) => void;
  onSubtaskStart?: (data: {
    plan_id: string;
    subtask_id: number;
    description: string;
    type: string;
    progress: { completed: number; total: number };
  }) => void;
  onSubtaskComplete?: (data: {
    plan_id: string;
    subtask_id: number;
    result_preview: string;
    progress: { completed: number; total: number };
  }) => void;
  onSubtaskFailed?: (data: {
    plan_id: string;
    subtask_id: number;
    error: string;
    progress: { completed: number; total: number };
  }) => void;
  onPlanComplete?: (data: {
    plan_id: string;
    subtask_count: number;
    completed_count: number;
    total_time_ms: number;
  }) => void;
  onPlanCancelled?: (data: {
    plan_id: string;
    subtask_count: number;
    completed_count: number;
    total_time_ms: number;
  }) => void;
  onDelegationStart?: (data: DelegationStartEvent) => void;
  onDelegationChunk?: (data: DelegationChunkEvent) => void;
  onDelegationComplete?: (data: DelegationCompleteEvent) => void;
  onDelegationToolCall?: (data: DelegationToolCallEvent) => void;
  onDelegationToolResult?: (data: DelegationToolResultEvent) => void;
  /** Agent presented a typed exhibit (e.g. a Mermaid diagram) via present_exhibit. */
  onExhibit?: (data: ExhibitWire) => void;
  /**
   * Per-phase activity status (Recalling memory… / Running web_search… / …) so the
   * chat shows a live line instead of a silent "thinking". `phase` is a stable slug;
   * `detail`/`group`/`progress` are optional headroom for deep sub-phases.
   */
  onStatus?: (data: {
    phase: string;
    label: string;
    detail?: string;
    group?: string;
    progress?: number;
  }) => void;
  /** A user steered the running turn mid-stream — echoed so every client shows it. */
  onSteer?: (data: { id: string; message: string }) => void;
  /**
   * Media (a generated image) landed in a workspace — when the conversation had none,
   * the backend fell back to the personal Home store. The client durably attaches it.
   */
  onWorkspaceAttached?: (data: { workspace_id: string }) => void;
  onError?: (error: string) => void;
}

/** Dispatch one parsed SSE event to its callback. Returns true to stop the pump. */
function dispatchSseEvent(
  type: string,
  data: Record<string, unknown>,
  callbacks: StreamCallbacks,
  controller: AbortController,
): boolean {
  switch (type) {
    case 'run_started':
      callbacks.onRunStarted?.(data as { run_id: string });
      break;
    case 'run_missing':
      callbacks.onRunMissing?.(data as { run_id: string; reason?: string });
      break;
    case 'start':
      callbacks.onStart?.(data as Parameters<NonNullable<StreamCallbacks['onStart']>>[0]);
      break;
    case 'chunk':
      callbacks.onChunk?.(data.content as string);
      break;
    case 'memory_context':
      callbacks.onMemoryContext?.(data as Parameters<NonNullable<StreamCallbacks['onMemoryContext']>>[0]);
      break;
    case 'tool_call':
      callbacks.onToolCall?.(data as Parameters<NonNullable<StreamCallbacks['onToolCall']>>[0]);
      break;
    case 'tool_result':
      callbacks.onToolResult?.(data as Parameters<NonNullable<StreamCallbacks['onToolResult']>>[0]);
      break;
    case 'plan_start':
      callbacks.onPlanStart?.(data as Parameters<NonNullable<StreamCallbacks['onPlanStart']>>[0]);
      break;
    case 'plan_resumed':
      callbacks.onPlanResumed?.(data as Parameters<NonNullable<StreamCallbacks['onPlanResumed']>>[0]);
      break;
    case 'subtask_start':
      callbacks.onSubtaskStart?.(data as Parameters<NonNullable<StreamCallbacks['onSubtaskStart']>>[0]);
      break;
    case 'subtask_complete':
      callbacks.onSubtaskComplete?.(data as Parameters<NonNullable<StreamCallbacks['onSubtaskComplete']>>[0]);
      break;
    case 'subtask_failed':
      callbacks.onSubtaskFailed?.(data as Parameters<NonNullable<StreamCallbacks['onSubtaskFailed']>>[0]);
      break;
    case 'plan_complete':
      callbacks.onPlanComplete?.(data as Parameters<NonNullable<StreamCallbacks['onPlanComplete']>>[0]);
      break;
    case 'plan_cancelled':
      callbacks.onPlanCancelled?.(data as Parameters<NonNullable<StreamCallbacks['onPlanCancelled']>>[0]);
      break;
    case 'delegation_start':
      callbacks.onDelegationStart?.(data as unknown as DelegationStartEvent);
      break;
    case 'delegation_chunk':
      callbacks.onDelegationChunk?.(data as unknown as DelegationChunkEvent);
      break;
    case 'delegation_complete':
      callbacks.onDelegationComplete?.(data as unknown as DelegationCompleteEvent);
      break;
    case 'delegation_tool_call':
      callbacks.onDelegationToolCall?.(data as unknown as DelegationToolCallEvent);
      break;
    case 'delegation_tool_result':
      callbacks.onDelegationToolResult?.(data as unknown as DelegationToolResultEvent);
      break;
    case 'exhibit':
      callbacks.onExhibit?.(data as unknown as ExhibitWire);
      break;
    case 'status':
      callbacks.onStatus?.(data as Parameters<NonNullable<StreamCallbacks['onStatus']>>[0]);
      break;
    case 'steer':
      callbacks.onSteer?.(data as Parameters<NonNullable<StreamCallbacks['onSteer']>>[0]);
      break;
    case 'workspace_attached':
      callbacks.onWorkspaceAttached?.(data as { workspace_id: string });
      break;
    case 'done':
      callbacks.onDone?.(data as Parameters<NonNullable<StreamCallbacks['onDone']>>[0]);
      break;
    case 'close':
      // Server signals the run has settled — abort to close the connection.
      controller.abort();
      return true;
    case 'error':
      callbacks.onError?.(data.error as string);
      break;
  }
  return false;
}

/** Read + parse an SSE response body, dispatching events until close/EOF. */
async function pumpSseResponse(
  response: Response,
  controller: AbortController,
  callbacks: StreamCallbacks,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  const processLines = (lines: string[], eventState: { type: string; data: string }): boolean => {
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventState.type = line.slice(7);
      } else if (line.startsWith('data: ')) {
        eventState.data = line.slice(6);
        if (eventState.type && eventState.data) {
          try {
            const data = JSON.parse(eventState.data);
            if (dispatchSseEvent(eventState.type, data, callbacks, controller)) return true;
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
          eventState.type = '';
          eventState.data = '';
        }
      }
    }
    return false;
  };

  const eventState = { type: '', data: '' };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    if (processLines(lines, eventState)) return;
  }

  if (buffer.trim()) {
    processLines(buffer.split('\n'), eventState);
  }
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getAuthToken();
  if (token) headers['X-Auth-Token'] = token;
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) headers['AgentX-Gateway-Token'] = gatewayToken;
  return headers;
}

/** Turn a non-2xx SSE response into an ApiError (parsing the flat {error} body). */
async function streamErrorFromResponse(response: Response): Promise<ApiError> {
  let details: unknown;
  try {
    details = await response.json();
  } catch {
    details = undefined;
  }
  if (response.status === 401) {
    clearAuthToken();
    window.dispatchEvent(new CustomEvent('agentx:auth-required'));
  }
  const message =
    backendErrorMessage(details) ?? `Stream failed: ${response.statusText || response.status}`;
  return { message, status: response.status, kind: classifyStatus(response.status), details } as ApiError;
}

export const streamingApi = {
  // === Streaming ===

  /**
   * Stream a chat response using Server-Sent Events.
   * Returns an object with methods to control the stream.
   */
  streamChat(request: ChatRequest, callbacks: StreamCallbacks): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/chat/stream`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(request),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw await streamErrorFromResponse(response);
        await pumpSseResponse(response, controller, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(apiErrorMessage(error));
        }
      });

    return { abort: () => controller.abort() };
  },

  /**
   * Resume an interrupted plan: continues its not-yet-terminal subtasks,
   * streaming the same event surface as a chat turn (first event is
   * `plan_resumed`). The run is detached server-side like streamChat.
   */
  resumePlan(
    planId: string,
    body: PlanResumeRequest,
    callbacks: StreamCallbacks,
  ): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/plans/${encodeURIComponent(planId)}/resume`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw await streamErrorFromResponse(response);
        await pumpSseResponse(response, controller, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(apiErrorMessage(error));
        }
      });

    return { abort: () => controller.abort() };
  },

  /**
   * Re-attach to a detached chat run: replays buffered events + follows live.
   * Same callback surface as streamChat (the run replays the same event set).
   */
  attachChatRun(runId: string, callbacks: StreamCallbacks): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/chat/stream/attach?run_id=${encodeURIComponent(runId)}`, {
      method: 'GET',
      headers: authHeaders(),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw await streamErrorFromResponse(response);
        await pumpSseResponse(response, controller, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(apiErrorMessage(error));
        }
      });

    return { abort: () => controller.abort() };
  },
};
