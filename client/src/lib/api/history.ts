import { request as apiRequest } from './core';
import type {
  ConversationListResponse,
  ConversationMessagesResponse,
  ConversationStateResponse,
  ConversationStateSlot,
  StateEntryInput,
} from './types';

export const historyApi = {
  // === Conversation History ===

  async listConversations(params?: {
    limit?: number;
    offset?: number;
    channel?: string;
  }): Promise<ConversationListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.offset) searchParams.set('offset', String(params.offset));
    if (params?.channel) searchParams.set('channel', params.channel);
    const qs = searchParams.toString();
    return apiRequest(`/api/conversations${qs ? `?${qs}` : ''}`);
  },

  async getConversationMessages(conversationId: string): Promise<ConversationMessagesResponse> {
    return apiRequest(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
  },

  // === Conversation State (structured working memory) ===

  async getConversationState(conversationId: string): Promise<ConversationStateResponse> {
    return apiRequest(`/api/conversations/${encodeURIComponent(conversationId)}/state`);
  },

  /** Replace a whole slot (add/edit/remove in one call). A user edit is authoritative. */
  async updateConversationState(
    conversationId: string,
    slot: ConversationStateSlot,
    entries: StateEntryInput[],
  ): Promise<ConversationStateResponse> {
    return apiRequest(`/api/conversations/${encodeURIComponent(conversationId)}/state`, {
      method: 'PATCH',
      body: JSON.stringify({ slot, entries }),
    });
  },

  async deleteConversation(conversationId: string): Promise<{ message: string; deleted: Record<string, number> }> {
    return apiRequest(`/api/memory/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    });
  },

  async clearStuckJobs(): Promise<{ success: boolean; cleared_jobs: string[]; message: string }> {
    return apiRequest('/api/jobs/clear-stuck', {
      method: 'POST',
    });
  },
};
