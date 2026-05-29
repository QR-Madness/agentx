import { request as apiRequest } from './core';
import type { ModelInfo, ProviderInfo, ProvidersHealthResponse } from './types';

export const providersApi = {
  // === Providers ===

  async listProviders(): Promise<{ providers: ProviderInfo[] }> {
    return apiRequest('/api/providers');
  },

  async listModels(): Promise<{ models: ModelInfo[] }> {
    return apiRequest('/api/providers/models', { signal: AbortSignal.timeout(10_000) });
  },

  async checkProvidersHealth(): Promise<ProvidersHealthResponse> {
    return apiRequest('/api/providers/health');
  },
};
