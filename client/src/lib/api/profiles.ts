import { request as apiRequest } from './core';
import type { AgentProfile, AgentProfileCreate, ReasoningStrategy } from './types';

export const profilesApi = {
  // === Agent Profiles ===

  async listAgentProfiles(): Promise<{ profiles: AgentProfile[] }> {
    const response = await apiRequest<{ profiles: Array<{
      id: string;
      name: string;
      agent_id?: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    }> }>('/api/agent/profiles');
    // Transform snake_case to camelCase
    return {
      profiles: response.profiles.map(p => ({
        id: p.id,
        name: p.name,
        agentId: p.agent_id || p.id,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      })),
    };
  },

  async getAgentProfile(id: string): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: {
      id: string;
      name: string;
      agent_id?: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/agent/profiles/${encodeURIComponent(id)}`);
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        agentId: p.agent_id || p.id,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  },

  async createAgentProfile(profile: AgentProfileCreate): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: {
      id: string;
      name: string;
      agent_id?: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>('/api/agent/profiles', {
      method: 'POST',
      body: JSON.stringify(profile),
    });
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        agentId: p.agent_id || p.id,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  },

  async updateAgentProfile(id: string, updates: Partial<AgentProfileCreate>): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: {
      id: string;
      name: string;
      agent_id?: string;
      avatar?: string;
      description?: string;
      default_model?: string;
      temperature: number;
      prompt_profile_id?: string;
      system_prompt?: string;
      reasoning_strategy: string;
      enable_memory: boolean;
      memory_channel: string;
      enable_tools: boolean;
      is_default: boolean;
      created_at?: string;
      updated_at?: string;
    } }>(`/api/agent/profiles/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
    const p = response.profile;
    return {
      profile: {
        id: p.id,
        name: p.name,
        agentId: p.agent_id || p.id,
        avatar: p.avatar,
        description: p.description,
        defaultModel: p.default_model,
        temperature: p.temperature,
        promptProfileId: p.prompt_profile_id,
        systemPrompt: p.system_prompt,
        reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
        enableMemory: p.enable_memory,
        memoryChannel: p.memory_channel,
        enableTools: p.enable_tools,
        isDefault: p.is_default,
        createdAt: p.created_at || '',
        updatedAt: p.updated_at || '',
      },
    };
  },

  async deleteAgentProfile(id: string): Promise<{ deleted: boolean }> {
    return apiRequest(`/api/agent/profiles/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  },

  async setDefaultAgentProfile(id: string): Promise<{ default_profile_id: string }> {
    return apiRequest(`/api/agent/profiles/${encodeURIComponent(id)}/set-default`, {
      method: 'POST',
    });
  },
};
