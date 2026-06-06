import { request as apiRequest } from './core';
import type { GlobalPrompt, PromptLayer, PromptProfile, PromptSection } from './types';

export const promptsApi = {
  // === Prompt Stack (layers) ===

  async listPromptLayers(): Promise<{ layers: PromptLayer[]; composed: string }> {
    return apiRequest('/api/prompts/layers');
  },

  async createPromptLayer(title: string, content = ''): Promise<{ layer: PromptLayer }> {
    return apiRequest('/api/prompts/layers', {
      method: 'POST',
      body: JSON.stringify({ title, content }),
    });
  },

  async updatePromptLayer(
    layerId: string,
    patch: { content?: string; title?: string; enabled?: boolean },
  ): Promise<{ layer: PromptLayer }> {
    return apiRequest(`/api/prompts/layers/${layerId}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  },

  async deletePromptLayer(layerId: string): Promise<{ deleted: string }> {
    return apiRequest(`/api/prompts/layers/${layerId}`, { method: 'DELETE' });
  },

  async resetPromptLayer(layerId: string): Promise<{ layer: PromptLayer }> {
    return apiRequest(`/api/prompts/layers/${layerId}/reset`, { method: 'POST' });
  },

  async acknowledgePromptLayer(layerId: string): Promise<{ layer: PromptLayer }> {
    return apiRequest(`/api/prompts/layers/${layerId}/acknowledge`, { method: 'POST' });
  },

  async reorderPromptLayers(order: string[]): Promise<{ layers: PromptLayer[] }> {
    return apiRequest('/api/prompts/layers/reorder', {
      method: 'POST',
      body: JSON.stringify({ order }),
    });
  },

  // === Prompts ===

  async listPromptProfiles(): Promise<{ profiles: PromptProfile[] }> {
    return apiRequest('/api/prompts/profiles');
  },

  async getPromptProfile(profileId: string): Promise<{ profile: PromptProfile; composed_prompt: string }> {
    return apiRequest(`/api/prompts/profiles/${profileId}`);
  },

  async getGlobalPrompt(): Promise<{ global_prompt: GlobalPrompt }> {
    return apiRequest('/api/prompts/global');
  },

  async updateGlobalPrompt(content: string, enabled = true): Promise<{ global_prompt: GlobalPrompt }> {
    return apiRequest('/api/prompts/global/update', {
      method: 'POST',
      body: JSON.stringify({ content, enabled }),
    });
  },

  async listPromptSections(): Promise<{ sections: PromptSection[] }> {
    return apiRequest('/api/prompts/sections');
  },

  async composePrompt(profileId?: string): Promise<{ system_prompt: string; profile_id: string }> {
    const params = profileId ? `?profile_id=${profileId}` : '';
    return apiRequest(`/api/prompts/compose${params}`);
  },

  async getMCPToolsPrompt(): Promise<{ mcp_tools_prompt: string; tools_count: number }> {
    return apiRequest('/api/prompts/mcp-tools');
  },
};
