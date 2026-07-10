import { request as apiRequest } from './core';
import type { AgentSkill, AgentSkillInput } from './types';

export const skillsApi = {
  // === Agent Skills (named instruction packs; see Connectors & Tools → Skills) ===

  async listSkills(): Promise<{ skills: AgentSkill[] }> {
    return apiRequest('/api/agent/skills');
  },

  async createSkill(input: AgentSkillInput): Promise<{ skill: AgentSkill }> {
    return apiRequest('/api/agent/skills', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  },

  async updateSkill(id: string, updates: Partial<AgentSkillInput>): Promise<{ skill: AgentSkill }> {
    return apiRequest(`/api/agent/skills/${encodeURIComponent(id)}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  },

  async deleteSkill(id: string): Promise<{ status: string; skill: string }> {
    return apiRequest(`/api/agent/skills/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
  },
};
