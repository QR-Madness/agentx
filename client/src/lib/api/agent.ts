import { request as apiRequest } from './core';
import type {
  AgentRunRequest,
  AgentRunResponse,
  ChatRequest,
  ChatResponse,
  PlanStatusResponse,
} from './types';

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

  /** Request cooperative cancellation of a running plan. */
  async cancelPlan(planId: string, sessionId: string): Promise<{ cancel_requested: boolean }> {
    return apiRequest('/api/agent/plans/cancel', {
      method: 'POST',
      body: JSON.stringify({ plan_id: planId, session_id: sessionId }),
    });
  },

  /**
   * Read the Redis-tracked state of a plan, for reconciling persisted
   * `running` plans after a reload. Returns `{ found: false }` when the
   * state has expired (1h TTL) rather than throwing.
   */
  async getPlanStatus(planId: string, sessionId: string): Promise<PlanStatusResponse> {
    return apiRequest(
      `/api/agent/plans/${encodeURIComponent(planId)}/status?session_id=${encodeURIComponent(sessionId)}`,
    );
  },
};
