import { request as apiRequest } from './core';
import type { AgentProfile, AgentProfileCreate, ReasoningStrategy } from './types';

// Raw API shape (snake_case) — keeps the mapper honest when the API adds a field.
interface RawAgentProfile {
  id: string;
  name: string;
  agent_id?: string;
  avatar?: string;
  description?: string;
  tags?: string[];
  default_model?: string;
  temperature: number;
  prompt_profile_id?: string;
  system_prompt?: string;
  reasoning_strategy: string;
  enable_memory: boolean;
  memory_channel: string;
  enable_tools: boolean;
  // Phase 18.9.x: per-tool gating. Omitted/`null` = all enabled.
  allowed_tools?: string[] | null;
  blocked_tools?: string[];
  available_for_delegation?: boolean;
  ambassador?: {
    enabled?: boolean;
    briefing_prompt?: string;
    verbosity?: 'brief' | 'normal' | 'deep';
    speech_model?: string | null;
    voice?: string | null;
  } | null;
  is_default: boolean;
  created_at?: string;
  updated_at?: string;
}

function mapProfile(p: RawAgentProfile): AgentProfile {
  return {
    id: p.id,
    name: p.name,
    agentId: p.agent_id || p.id,
    avatar: p.avatar,
    description: p.description,
    tags: p.tags ?? [],
    defaultModel: p.default_model,
    temperature: p.temperature,
    promptProfileId: p.prompt_profile_id,
    systemPrompt: p.system_prompt,
    reasoningStrategy: p.reasoning_strategy as ReasoningStrategy,
    enableMemory: p.enable_memory,
    memoryChannel: p.memory_channel,
    enableTools: p.enable_tools,
    allowedTools: p.allowed_tools ?? null,
    blockedTools: p.blocked_tools ?? [],
    availableForDelegation: p.available_for_delegation ?? true,
    ambassador: p.ambassador
      ? {
          enabled: p.ambassador.enabled ?? false,
          briefingPrompt: p.ambassador.briefing_prompt,
          verbosity: p.ambassador.verbosity,
          speechModel: p.ambassador.speech_model,
          voice: p.ambassador.voice,
        }
      : undefined,
    isDefault: p.is_default,
    createdAt: p.created_at || '',
    updatedAt: p.updated_at || '',
  };
}

export const profilesApi = {
  // === Agent Profiles ===

  async listAgentProfiles(): Promise<{ profiles: AgentProfile[] }> {
    const response = await apiRequest<{ profiles: RawAgentProfile[] }>('/api/agent/profiles');
    return { profiles: response.profiles.map(mapProfile) };
  },

  async getAgentProfile(id: string): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: RawAgentProfile }>(
      `/api/agent/profiles/${encodeURIComponent(id)}`,
    );
    return { profile: mapProfile(response.profile) };
  },

  async createAgentProfile(profile: AgentProfileCreate): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: RawAgentProfile }>('/api/agent/profiles', {
      method: 'POST',
      body: JSON.stringify(profile),
    });
    return { profile: mapProfile(response.profile) };
  },

  async updateAgentProfile(
    id: string,
    updates: Partial<AgentProfileCreate>,
  ): Promise<{ profile: AgentProfile }> {
    const response = await apiRequest<{ profile: RawAgentProfile }>(
      `/api/agent/profiles/${encodeURIComponent(id)}`,
      {
        method: 'PUT',
        body: JSON.stringify(updates),
      },
    );
    return { profile: mapProfile(response.profile) };
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
