import { getBaseUrl, registerStreamController } from './core';
import { getAuthToken, clearAuthToken, getActiveGatewayToken } from '../storage';
import { classifyStatus, apiErrorMessage, backendErrorMessage } from './errors';
import type { ApiError } from './errors';
import type { ChatRequest, DelegationChunkEvent, DelegationCompleteEvent, DelegationStartEvent, DelegationToolCallEvent, DelegationToolResultEvent } from './types';

export const streamingApi = {
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
        model?: string;
        provider?: string;
        cost_estimate?: number | null;
        cost_currency?: string | null;
        pricing_snapshot?: { cost_per_1k_input: number; cost_per_1k_output: number } | null;
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
      onDelegationStart?: (data: DelegationStartEvent) => void;
      onDelegationChunk?: (data: DelegationChunkEvent) => void;
      onDelegationComplete?: (data: DelegationCompleteEvent) => void;
      onDelegationToolCall?: (data: DelegationToolCallEvent) => void;
      onDelegationToolResult?: (data: DelegationToolResultEvent) => void;
      onError?: (error: string) => void;
    }
  ): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const token = getAuthToken();
    if (token) {
      headers['X-Auth-Token'] = token;
    }
    const streamGatewayToken = getActiveGatewayToken();
    if (streamGatewayToken) {
      headers['AgentX-Gateway-Token'] = streamGatewayToken;
    }

    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          // The stream POST can fail before any SSE event is sent (e.g. auth,
          // a typed provider/MCP error). Parse the flat `{"error": "..."}` body
          // so the user sees the real reason rather than a bare status line.
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
            backendErrorMessage(details) ??
            `Stream failed: ${response.statusText || response.status}`;
          throw { message, status: response.status, kind: classifyStatus(response.status), details } as ApiError;
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
                    case 'delegation_start':
                      callbacks.onDelegationStart?.(data);
                      break;
                    case 'delegation_chunk':
                      callbacks.onDelegationChunk?.(data);
                      break;
                    case 'delegation_complete':
                      callbacks.onDelegationComplete?.(data);
                      break;
                    case 'delegation_tool_call':
                      callbacks.onDelegationToolCall?.(data);
                      break;
                    case 'delegation_tool_result':
                      callbacks.onDelegationToolResult?.(data);
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
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(apiErrorMessage(error));
        }
      });

    return {
      abort: () => controller.abort(),
    };
  },
};
