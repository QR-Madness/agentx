import { request as apiRequest } from './core';
import type { AgentProfile, AgentProfileCreate, ReasoningStrategy } from './types';

// Raw API shape (snake_case) — keeps the mapper honest when the API adds a field.
interface RawAgentProfile {
  id: string;
  name: string;
  kind?: 'agent' | 'ambassador';
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
  direct_mode?: boolean;
  // Phase 18.9.x: per-tool gating. Omitted/`null` = all enabled.
  allowed_tools?: string[] | null;
  blocked_tools?: string[];
  available_for_delegation?: boolean;
  delegation_hint?: string | null;
  ambassador?: {
    enabled?: boolean;
    briefing_prompt?: string;
    verbosity?: 'brief' | 'normal' | 'deep';
    briefing_persona?: string | null;
    qa_persona?: string | null;
    draft_persona?: string | null;
    voice_mode?: boolean;
    speech_model?: string | null;
    voice?: string | null;
    speech_speed?: number | null;
    transcription_model?: string | null;
  } | null;
  is_default: boolean;
  is_default_ambassador?: boolean;
  created_at?: string;
  updated_at?: string;
}

function mapProfile(p: RawAgentProfile): AgentProfile {
  return {
    id: p.id,
    name: p.name,
    kind: p.kind ?? 'agent',
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
    directMode: p.direct_mode ?? false,
    allowedTools: p.allowed_tools ?? null,
    blockedTools: p.blocked_tools ?? [],
    availableForDelegation: p.available_for_delegation ?? false,
    delegationHint: p.delegation_hint ?? null,
    ambassador: p.ambassador
      ? {
          enabled: p.ambassador.enabled ?? false,
          briefingPrompt: p.ambassador.briefing_prompt,
          verbosity: p.ambassador.verbosity,
          briefingPersona: p.ambassador.briefing_persona,
          qaPersona: p.ambassador.qa_persona,
          draftPersona: p.ambassador.draft_persona,
          voiceMode: p.ambassador.voice_mode ?? false,
          speechModel: p.ambassador.speech_model,
          voice: p.ambassador.voice,
          speechSpeed: p.ambassador.speech_speed,
          transcriptionModel: p.ambassador.transcription_model,
        }
      : undefined,
    isDefault: p.is_default,
    isDefaultAmbassador: p.is_default_ambassador ?? false,
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

  async setDefaultAmbassador(id: string): Promise<{ default_ambassador_id: string }> {
    return apiRequest(`/api/agent/profiles/${encodeURIComponent(id)}/set-default-ambassador`, {
      method: 'POST',
    });
  },

  /** Persist a new profile ordering (list of ids). Returns the authoritative order. */
  async reorderAgentProfiles(order: string[]): Promise<{ order: string[] }> {
    return apiRequest('/api/agent/profiles/reorder', {
      method: 'POST',
      body: JSON.stringify({ order }),
    });
  },
};
