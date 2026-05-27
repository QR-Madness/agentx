import { request as apiRequest } from './core';
import type { ConversationListResponse, ConversationMessagesResponse } from './types';

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
