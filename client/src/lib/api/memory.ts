import { request as apiRequest } from './core';
import type { CheckpointsResponse, EntitiesResponse, EntityGraph, FactForgetResult, FactProvenance, FactsResponse, MemoryChannel, MemoryEntity, MemoryEntityPatch, MemoryExport, MemoryFactEntity, MemoryImportResult, MemoryFact, MemoryFactPatch, MemoryStats, ProceduresResponse, StrategiesResponse, UserHistoryResponse } from './types';

export const memoryApi = {
  // === Memory Explorer ===

  async listMemoryChannels(): Promise<{ channels: MemoryChannel[] }> {
    return apiRequest('/api/memory/channels');
  },

  async listMemoryEntities(params: {
    channel?: string;
    page?: number;
    limit?: number;
    search?: string;
    type?: string;
  } = {}): Promise<EntitiesResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.search) query.set('search', params.search);
    if (params.type) query.set('type', params.type);
    const queryString = query.toString();
    return apiRequest(`/api/memory/entities${queryString ? `?${queryString}` : ''}`);
  },

  async getEntityGraph(entityId: string, depth?: number): Promise<EntityGraph> {
    const query = depth ? `?depth=${depth}` : '';
    return apiRequest(`/api/memory/entities/${entityId}/graph${query}`);
  },

  async listMemoryFacts(params: {
    channel?: string;
    page?: number;
    limit?: number;
    min_confidence?: number;
    search?: string;
  } = {}): Promise<FactsResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.min_confidence !== undefined) query.set('min_confidence', params.min_confidence.toString());
    if (params.search) query.set('search', params.search);
    const queryString = query.toString();
    return apiRequest(`/api/memory/facts${queryString ? `?${queryString}` : ''}`);
  },

  async listMemoryStrategies(params: {
    channel?: string;
    page?: number;
    limit?: number;
  } = {}): Promise<StrategiesResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    const queryString = query.toString();
    return apiRequest(`/api/memory/strategies${queryString ? `?${queryString}` : ''}`);
  },

  async listMemoryProcedures(params: {
    channel?: string;
    page?: number;
    limit?: number;
  } = {}): Promise<ProceduresResponse> {
    const query = new URLSearchParams();
    if (params.channel) query.set('channel', params.channel);
    if (params.page) query.set('page', params.page.toString());
    if (params.limit) query.set('limit', params.limit.toString());
    const queryString = query.toString();
    return apiRequest(`/api/memory/procedures${queryString ? `?${queryString}` : ''}`);
  },

  async getMemoryStats(): Promise<MemoryStats> {
    return apiRequest('/api/memory/stats');
  },

  async updateMemoryFact(factId: string, patch: MemoryFactPatch): Promise<{ fact: MemoryFact }> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  },

  async deleteMemoryFact(factId: string): Promise<{ deleted: boolean }> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}`, {
      method: 'DELETE',
    });
  },

  async rememberMemoryFact(factId: string, to?: number): Promise<{ fact: MemoryFact }> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}/remember`, {
      method: 'POST',
      body: JSON.stringify(to !== undefined ? { to } : {}),
    });
  },

  async forgetMemoryFact(factId: string, hard = false): Promise<FactForgetResult> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}/forget`, {
      method: 'POST',
      body: JSON.stringify({ hard }),
    });
  },

  async getFactProvenance(factId: string): Promise<FactProvenance> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}/provenance`);
  },

  /** Link an entity to a fact (ABOUT edge). Returns the fact's updated entity list (#905). */
  async linkFactEntity(factId: string, entityId: string): Promise<{ entities: MemoryFactEntity[] }> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}/entities`, {
      method: 'POST',
      body: JSON.stringify({ entity_id: entityId }),
    });
  },

  /** Unlink an entity from a fact. Returns the fact's remaining entity list (#905). */
  async unlinkFactEntity(factId: string, entityId: string): Promise<{ entities: MemoryFactEntity[] }> {
    return apiRequest(`/api/memory/facts/${encodeURIComponent(factId)}/entities`, {
      method: 'DELETE',
      body: JSON.stringify({ entity_id: entityId }),
    });
  },

  async updateMemoryEntity(entityId: string, patch: MemoryEntityPatch): Promise<{ entity: MemoryEntity }> {
    return apiRequest(`/api/memory/entities/${encodeURIComponent(entityId)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  },

  async deleteMemoryEntity(entityId: string): Promise<{ deleted: boolean }> {
    return apiRequest(`/api/memory/entities/${encodeURIComponent(entityId)}`, {
      method: 'DELETE',
    });
  },

  // === Checkpoints (model-authored conversation anchors) ===

  async getCheckpoints(conversationId: string): Promise<CheckpointsResponse> {
    return apiRequest(`/api/memory/checkpoints?conversation_id=${encodeURIComponent(conversationId)}`);
  },

  async clearCheckpoints(conversationId: string): Promise<{ cleared: number }> {
    return apiRequest(`/api/memory/checkpoints?conversation_id=${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    });
  },

  // === User history (manual recall browse) ===

  async getUserHistory(params: { topic?: string; limit?: number; channel?: string } = {}): Promise<UserHistoryResponse> {
    return apiRequest('/api/memory/user-history', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },

  // === Import / export (round-trippable memory snapshots) ===

  async exportMemory(params: { channel?: string } = {}): Promise<{ export: MemoryExport }> {
    return apiRequest('/api/memory/export', {
      method: 'POST',
      body: JSON.stringify({ channel: params.channel ?? '_all' }),
    });
  },

  async importMemory(
    data: MemoryExport,
    mode: 'merge' | 'replace' = 'merge',
    channel?: string,
  ): Promise<{ imported: MemoryImportResult }> {
    return apiRequest('/api/memory/import', {
      method: 'POST',
      body: JSON.stringify({ data, mode, channel }),
    });
  },
};
