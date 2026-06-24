import { request as apiRequest } from './core';
import type {
  AgentRunRequest,
  AgentRunResponse,
  ChatImageRef,
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

  /**
   * Upload an image for vision input. Stored in the user's Home workspace (no
   * ingestion) and returned as a ref the chat stream carries in `images[]`.
   */
  async uploadChatImage(file: File): Promise<ChatImageRef & { url: string }> {
    const form = new FormData();
    form.append('file', file);
    const res = await apiRequest<{ doc_id: string; workspace_id: string; media_type: string; url: string }>(
      '/api/agent/chat/images',
      { method: 'POST', body: form },
    );
    return { workspace_id: res.workspace_id, doc_id: res.doc_id, media_type: res.media_type, url: res.url };
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

  /** Cooperatively cancel a detached chat run (the Stop button when streaming). */
  async cancelChatRun(runId: string): Promise<{ run_id: string; cancel_requested: boolean }> {
    return apiRequest(`/api/agent/chat/runs/${encodeURIComponent(runId)}/cancel`, {
      method: 'POST',
    });
  },

  /**
   * Steer a running turn: fold a message into the in-flight run. It's drained at
   * the next safe boundary (after a tool round, or instead of ending) and folded
   * in as a fresh user turn so the agent course-corrects without stopping.
   */
  async steerChatRun(
    runId: string,
    message: string,
    mode: 'queue' = 'queue',
  ): Promise<{ run_id: string; steer_accepted: boolean }> {
    return apiRequest(`/api/agent/chat/runs/${encodeURIComponent(runId)}/steer`, {
      method: 'POST',
      body: JSON.stringify({ message, mode }),
    });
  },
};
