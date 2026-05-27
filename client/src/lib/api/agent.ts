import { request as apiRequest } from './core';
import type { AgentRunRequest, AgentRunResponse, ChatRequest, ChatResponse } from './types';

export const agentApi = {
  // === Agent ===

  async runAgent(request: AgentRunRequest): Promise<AgentRunResponse> {
    return apiRequest('/api/agent/run', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async chat(request: ChatRequest): Promise<ChatResponse> {
    return apiRequest('/api/agent/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async getAgentStatus(): Promise<{ status: string; active_sessions: number }> {
    return apiRequest('/api/agent/status');
  },
};
