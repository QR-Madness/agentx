import { request as apiRequest } from './core';
import type { EntitiesResponse, EntityGraph, FactsResponse, MemoryChannel, MemoryEntity, MemoryEntityPatch, MemoryFact, MemoryFactPatch, MemoryStats, StrategiesResponse } from './types';

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
};
