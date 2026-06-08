/**
 * Logs API — the client surface for the server Log panel.
 *
 * The Tauri client is a thin HTTP shell and the Django API is a separate
 * process, so server logs travel over HTTP/SSE (no stdout capture). Records are
 * redacted server-side; this module just reads + tails them. Mirrors the minimal
 * SSE pump used by the ambassador domain.
 */

import { request as apiRequest, getBaseUrl, registerStreamController } from './core';
import { getAuthToken, getActiveGatewayToken } from '../storage';

export interface LogRecord {
  id: number;
  ts: number;
  level: string;
  logger: string;
  category: string;
  run_id?: string | null;
  conversation_id?: string | null;
  agent_id?: string | null;
  message: string;
  exc?: string;
  /** Oversized, already-redacted payload (e.g. a full LLM request) shown
   *  collapsed under the summary — stripped from the console server-side. */
  detail?: string;
}

export interface LogCategoryInfo {
  key: string;
  label: string;
  emoji: string;
  color: string;
}

export interface LogArchiveSegment {
  name: string;
  size: number;
  modified: number;
  compressed: boolean;
  /** Sealed (AES-GCM) segment — needs the vault unlocked (a recent login) to download. */
  encrypted?: boolean;
}

export interface LogArchiveStatus {
  /** Whether the AES-GCM keyring exists (i.e. archives get sealed). */
  keyring_present: boolean;
  /** Whether the data key is cached in server memory — sealed segments are downloadable. */
  unlocked: boolean;
  sealed_segments: number;
  pending_segments: number;
  /** False only when ops forced `AGENTX_LOG_ARCHIVE_ENCRYPT=false`. */
  encryption_enabled: boolean;
  retention_days: number;
  created_at?: string;
  rotated_at?: string;
}

export interface LogFilters {
  level?: string;
  category?: string;
  run_id?: string;
  search?: string;
  since?: number;
  limit?: number;
}

export interface LogStreamCallbacks {
  onLog?: (record: LogRecord) => void;
  onError?: (error: string) => void;
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getAuthToken();
  if (token) headers['X-Auth-Token'] = token;
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) headers['AgentX-Gateway-Token'] = gatewayToken;
  return headers;
}

function buildQuery(filters: LogFilters): string {
  const params = new URLSearchParams();
  if (filters.level) params.set('level', filters.level);
  if (filters.category) params.set('category', filters.category);
  if (filters.run_id) params.set('run_id', filters.run_id);
  if (filters.search) params.set('search', filters.search);
  if (filters.since != null) params.set('since', String(filters.since));
  if (filters.limit != null) params.set('limit', String(filters.limit));
  const q = params.toString();
  return q ? `?${q}` : '';
}

/** Minimal SSE pump for the `log` event set (plus heartbeats / close). */
async function pumpLogSse(response: Response, callbacks: LogStreamCallbacks): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');
  const decoder = new TextDecoder();
  let buffer = '';
  const state = { type: '', data: '' };

  const processLines = (lines: string[]) => {
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        state.type = line.slice(7);
      } else if (line.startsWith('data: ')) {
        state.data = line.slice(6);
        if (state.type === 'log' && state.data) {
          try {
            callbacks.onLog?.(JSON.parse(state.data) as LogRecord);
          } catch {
            /* ignore malformed frame */
          }
        }
        state.type = '';
        state.data = '';
      }
      // ': ping' heartbeats and blank lines fall through (no-op)
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

export const logsApi = {
  async listLogs(filters: LogFilters = {}): Promise<{ logs: LogRecord[]; available: boolean }> {
    return apiRequest(`/api/logs${buildQuery(filters)}`);
  },

  async getLogCategories(): Promise<{ categories: LogCategoryInfo[] }> {
    return apiRequest('/api/logs/categories');
  },

  async listLogArchive(): Promise<{ segments: LogArchiveSegment[] }> {
    return apiRequest('/api/logs/archive');
  },

  async getLogArchiveStatus(): Promise<LogArchiveStatus> {
    return apiRequest('/api/logs/archive/status');
  },

  /** Absolute URL to download an archive segment (auth handled by the browser/session). */
  logArchiveUrl(name: string): string {
    return `${getBaseUrl()}/api/logs/archive/${encodeURIComponent(name)}`;
  },

  /**
   * Fetch + save an archive segment via blob, carrying auth headers. Needed for
   * sealed (`.enc`) segments — the server decrypts on the fly, and a locked vault
   * returns `423` (which a plain `<a download>` couldn't surface). Throws an Error
   * with a readable message on `423`/`422`.
   */
  async downloadLogArchive(segment: LogArchiveSegment): Promise<void> {
    const res = await fetch(this.logArchiveUrl(segment.name), {
      method: 'GET',
      headers: authHeaders(),
    });
    if (!res.ok) {
      let detail = `Download failed (${res.status})`;
      if (res.status === 423) detail = 'Encrypted logs are locked — re-authenticate to unlock.';
      else {
        try {
          const body = await res.json();
          detail = body.detail || body.error || detail;
        } catch {
          /* keep default */
        }
      }
      throw new Error(detail);
    }
    const blob = await res.blob();
    // The server strips `.enc`, returning the inner `.gz`.
    const filename = segment.name.endsWith('.enc') ? segment.name.slice(0, -'.enc'.length) : segment.name;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  /** Tail the live log stream. Returns an abort handle. */
  streamLogs(callbacks: LogStreamCallbacks): { abort: () => void } {
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    registerStreamController(controller);

    fetch(`${baseUrl}/api/logs/stream`, {
      method: 'GET',
      headers: authHeaders(),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Log stream failed: ${response.status}`);
        await pumpLogSse(response, callbacks);
      })
      .catch((error) => {
        if (error?.name !== 'AbortError') {
          callbacks.onError?.(error?.message ?? 'Log stream failed');
        }
      });

    return { abort: () => controller.abort() };
  },
};
