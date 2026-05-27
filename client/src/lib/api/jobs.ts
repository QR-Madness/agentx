import { request as apiRequest, getBaseUrl, registerStreamController } from './core';
import { getAuthToken, getActiveGatewayToken } from '../storage';
import type { ConsolidateResult, ConsolidationDoneEvent, ConsolidationJobDoneEvent, ConsolidationJobStartEvent, ConsolidationProgressEvent, ConsolidationSettings, ConsolidationStartEvent, JobDetailResponse, JobRunResult, JobsResponse, RecallSettings } from './types';

export const jobsApi = {
  // === Jobs ===

  async listJobs(): Promise<JobsResponse> {
    return apiRequest('/api/jobs');
  },

  async getJob(name: string): Promise<JobDetailResponse> {
    return apiRequest(`/api/jobs/${encodeURIComponent(name)}`);
  },

  async runJob(name: string): Promise<JobRunResult> {
    return apiRequest(`/api/jobs/${encodeURIComponent(name)}/run`, {
      method: 'POST',
    });
  },

  async toggleJob(name: string, enabled: boolean): Promise<{ enabled: boolean; job: string }> {
    return apiRequest(`/api/jobs/${encodeURIComponent(name)}/toggle`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  },

  async consolidateNow(jobs?: string[]): Promise<ConsolidateResult> {
    return apiRequest('/api/memory/consolidate', {
      method: 'POST',
      body: JSON.stringify(jobs ? { jobs } : {}),
    });
  },

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
    const baseUrl = getBaseUrl();
    const controller = new AbortController();
    const method = options.trigger ? 'POST' : 'GET';
    const body = options.trigger && options.jobs
      ? JSON.stringify({ jobs: options.jobs })
      : options.trigger ? '{}' : undefined;

    const consolidateHeaders: Record<string, string> = {};
    if (method === 'POST') {
      consolidateHeaders['Content-Type'] = 'application/json';
    }
    const consolidateToken = getAuthToken();
    if (consolidateToken) {
      consolidateHeaders['X-Auth-Token'] = consolidateToken;
    }
    const consolidateGatewayToken = getActiveGatewayToken();
    if (consolidateGatewayToken) {
      consolidateHeaders['AgentX-Gateway-Token'] = consolidateGatewayToken;
    }

    registerStreamController(controller);

    fetch(`${baseUrl}/api/memory/consolidate/stream`, {
      method,
      headers: consolidateHeaders,
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
  },

  async getConsolidationSettings(): Promise<ConsolidationSettings> {
    return apiRequest('/api/memory/settings');
  },

  async updateConsolidationSettings(settings: Partial<ConsolidationSettings>): Promise<{ success: boolean; updated: string[] }> {
    return apiRequest('/api/memory/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  },

  async getRecallSettings(): Promise<RecallSettings> {
    return apiRequest('/api/memory/recall-settings');
  },

  async updateRecallSettings(settings: Partial<RecallSettings>): Promise<{ success: boolean; updated: string[] }> {
    return apiRequest('/api/memory/recall-settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    });
  },

  async resetMemory(deleteMemories = false): Promise<{ conversations_reset: number; memories_deleted?: number; success: boolean }> {
    return apiRequest('/api/memory/reset', {
      method: 'POST',
      body: JSON.stringify({ delete_memories: deleteMemories }),
    });
  },
};
