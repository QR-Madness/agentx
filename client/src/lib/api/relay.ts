import { request as apiRequest } from './core';
import type { ActiveChatRun, BackgroundChatJob, BackgroundChatRequest } from './types';

export const relayApi = {
  // === Background Chat (Relay) ===

  async enqueueBackgroundChat(request: BackgroundChatRequest): Promise<{ job_id: string; status: string }> {
    return apiRequest('/api/chat/background', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async listBackgroundChats(): Promise<{ jobs: BackgroundChatJob[] }> {
    return apiRequest('/api/chat/background');
  },

  async getBackgroundChat(jobId: string): Promise<BackgroundChatJob> {
    return apiRequest(`/api/chat/background/${encodeURIComponent(jobId)}`);
  },

  async dismissBackgroundChat(jobId: string): Promise<{ deleted: boolean }> {
    return apiRequest(`/api/chat/background/${encodeURIComponent(jobId)}`, {
      method: 'DELETE',
    });
  },

  // === Detached Chat Runs (recovery) ===

  async listChatRuns(): Promise<{ runs: ActiveChatRun[] }> {
    return apiRequest('/api/agent/chat/runs');
  },
};
