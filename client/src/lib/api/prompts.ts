import { request as apiRequest } from './core';
import type { GlobalPrompt, PromptProfile, PromptSection } from './types';

export const promptsApi = {
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
