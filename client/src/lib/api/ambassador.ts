/**
 * Ambassador API (Phase 16.6) — the client surface for the parallel,
 * non-polluting conversation interpreter. Kept as its own domain module (rather
 * than folded into the chat stream surface) because an ambassador run is a
 * separate, namespaced (`ambassador_*`) event stream.
 */

import { request as apiRequest, getBaseUrl, registerStreamController } from './core';
import { getAuthToken, getActiveGatewayToken } from '../storage';

export type AmbassadorStatus = 'streaming' | 'done' | 'error' | 'empty_provider' | 'cancelled';

/** One persisted per-turn briefing (sidecar record / live state). */
export interface AmbassadorBriefing {
  message_id: string;
  status: AmbassadorStatus;
  summary: string;
  error?: string;
  run_id?: string;
  created_at?: string;
  updated_at?: string;
}

export interface BriefTurnRequest {
  conversation_id: string;
  message_id: string;
  assistant_text: string;
  user_text?: string;
}

export interface AmbassadorStreamCallbacks {
  onChunk?: (text: string) => void;
  onDone?: (summary: string, status: AmbassadorStatus) => void;
  onError?: (error: string) => void;
  /** The run's event buffer expired — fall back to the persisted briefing. */
  onMissing?: () => void;
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getAuthToken();
  if (token) headers['X-Auth-Token'] = token;
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) headers['AgentX-Gateway-Token'] = gatewayToken;
  return headers;
}

/** Minimal SSE pump for the namespaced ambassador event set. */
async function pumpAmbassadorSse(
  response: Response,
  callbacks: AmbassadorStreamCallbacks,
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
  const state = { type: '', data: '' };

  const handle = (type: string, data: Record<string, unknown>) => {
    switch (type) {
      case 'ambassador_chunk':
        if (typeof data.text === 'string') callbacks.onChunk?.(data.text);
        break;
      case 'ambassador_done':
        callbacks.onDone?.(
          typeof data.summary === 'string' ? data.summary : '',
          (data.status as AmbassadorStatus) || 'done',
        );
        break;
      case 'ambassador_error':
        callbacks.onError?.(typeof data.error === 'string' ? data.error : 'Briefing failed');
        break;
      case 'run_missing':
        callbacks.onMissing?.();
        break;
      default:
        break; // ambassador_start, close — no-op
    }
  };

  const processLines = (lines: string[]) => {
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        state.type = line.slice(7);
      } else if (line.startsWith('data: ')) {
        state.data = line.slice(6);
        if (state.type && state.data) {
          try {
            handle(state.type, JSON.parse(state.data));
          } catch {
            /* ignore malformed frame */
          }
          state.type = '';
          state.data = '';
        }
      }
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    processLines(lines);
  }
  if (buffer.trim()) processLines(buffer.split('\n'));
}

export const ambassadorApi = {
  /** Kick off a parallel briefing of one turn; returns its detached run_id. */
  async briefTurn(req: BriefTurnRequest): Promise<{ run_id: string }> {
    return apiRequest('/api/agent/ambassador/brief-turn', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  /** Replay all persisted briefings for a conversation (cold-open / reload). */
  async fetchAmbassadorBriefings(conversationId: string): Promise<AmbassadorBriefing[]> {
    const res = await apiRequest<{ conversation_id: string; briefings: AmbassadorBriefing[] }>(
      `/api/agent/ambassador/${encodeURIComponent(conversationId)}`,
    );
    return res.briefings ?? [];
  },

  /** Tail a briefing run's `ambassador_*` SSE stream. Returns an abort handle. */
  streamAmbassador(
    runId: string,
    callbacks: AmbassadorStreamCallbacks,
  ): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/agent/ambassador/stream?run_id=${encodeURIComponent(runId)}`, {
      method: 'GET',
      headers: authHeaders(),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Ambassador stream failed: ${response.status}`);
        await pumpAmbassadorSse(response, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(error?.message ?? 'Ambassador stream failed');
        }
      });

    return { abort: () => controller.abort() };
  },
};
