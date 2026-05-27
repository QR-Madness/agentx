import { request as apiRequest } from './core';
import type { ConfigUpdate } from './types';

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
