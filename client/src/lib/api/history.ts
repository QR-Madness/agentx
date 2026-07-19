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
    /** Include archived conversations (excluded by default). */
    includeArchived?: boolean;
  }): Promise<ConversationListResponse> {
    const searchParams = new URLSearchParams();
    if (params?.limit) searchParams.set('limit', String(params.limit));
    if (params?.offset) searchParams.set('offset', String(params.offset));
    if (params?.channel) searchParams.set('channel', params.channel);
    if (params?.includeArchived) searchParams.set('include_archived', '1');
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

  /** Set user-set conversation metadata: a custom title and/or the archived flag.
   *  The execution half of the Ambassador's confirmed-write proposals, and the
   *  manual rename/restore actions. */
  async patchConversationMeta(
    conversationId: string,
    meta: { title?: string; archived?: boolean },
  ): Promise<{ conversation_id: string; title?: string; archived?: boolean }> {
    return apiRequest(`/api/memory/conversations/${encodeURIComponent(conversationId)}/meta`, {
      method: 'PATCH',
      body: JSON.stringify(meta),
    });
  },

  async clearStuckJobs(): Promise<{ success: boolean; cleared_jobs: string[]; message: string }> {
    return apiRequest('/api/jobs/clear-stuck', {
      method: 'POST',
    });
  },
};
