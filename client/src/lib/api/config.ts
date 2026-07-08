import { request as apiRequest } from './core';
import type { ConfigUpdate, ModelRoleName, ModelRolesResponse } from './types';

export const configApi = {
  // === Config ===

  /**
   * Get backend configuration.
   * Returns current runtime config with sensitive values redacted.
   */
  async getConfig(): Promise<Record<string, unknown>> {
    return apiRequest('/api/config');
  },

  /**
   * Update backend configuration (POST-only for security).
   * Persists to data/config.json and hot-reloads providers.
   */
  async updateConfig(config: ConfigUpdate): Promise<{ status: string; message: string; updated?: string[] }> {
    return apiRequest('/api/config/update', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  },

  /**
   * Probe the configured web-search backend with a trivial query.
   * Powers the 'Test connection' button in Settings → Web Search.
   */
  async searchHealth(): Promise<{ ok: boolean; backend: string | null; count: number; error?: string | null }> {
    return apiRequest('/api/tools/search-health');
  },

  /**
   * Model roles + membership + effective-model preview (settings overhaul D1).
   * Set roles via updateModelRoles below.
   */
  async getModelRoles(): Promise<ModelRolesResponse> {
    return apiRequest('/api/models/roles');
  },

  /** Set/clear model roles ("" clears; values must be concrete provider:model). */
  async updateModelRoles(
    roles: Partial<Record<ModelRoleName, string>>
  ): Promise<{ status: string; updated?: string[] }> {
    return apiRequest('/api/config/update', {
      method: 'POST',
      body: JSON.stringify({ models: { roles } }),
    });
  },

  /**
   * Clear concrete per-stage consolidation model overrides so every stage
   * follows its model role. For existing installs whose memory settings still
   * pin a specific model per stage (fresh installs already ship "inherit").
   */
  async adoptModelRoles(): Promise<{ success: boolean; adopted?: string[] }> {
    return apiRequest('/api/models/roles/adopt', { method: 'POST' });
  },

  async getContextLimits(): Promise<{
    lmstudio: { context_window: number; max_output_tokens: number };
    models: Record<string, { context_window: number; max_output_tokens: number }>;
  }> {
    return apiRequest('/api/config/context-limits');
  },

  async updateContextLimits(limits: {
    lmstudio?: { context_window?: number; max_output_tokens?: number };
    models?: Record<string, { context_window?: number; max_output_tokens?: number }>;
  }): Promise<{ status: string; updated: string[] }> {
    return apiRequest('/api/config/context-limits', {
      method: 'POST',
      body: JSON.stringify(limits),
    });
  },
};
