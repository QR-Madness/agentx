import { request as apiRequest } from './core';
import type {
  ModelInfo,
  ProviderCatalogEntry,
  ProviderCatalogResponse,
  ProviderCustomInput,
  ProviderInfo,
  ProviderRouteResponse,
  ProvidersHealthResponse,
  ProviderTestResult,
} from './types';

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

  // === Catalog (built-ins + user-registered endpoints) ===

  /** Every reachable backend. Secrets come back as fingerprints, never values. */
  async getProviderCatalog(): Promise<ProviderCatalogResponse> {
    return apiRequest('/api/providers/catalog');
  },

  /** Create or update a custom OpenAI-compatible endpoint. Partial updates merge,
   *  so omitting `api_key` keeps the stored one rather than clearing it. */
  async saveCustomProvider(input: ProviderCustomInput): Promise<{ provider: ProviderCatalogEntry }> {
    return apiRequest('/api/providers/custom', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  },

  /** Unregister a custom endpoint. Reports which agent profiles referenced it so
   *  the caller can say so; those references degrade via fallback, they don't break. */
  async deleteCustomProvider(
    id: string
  ): Promise<{ status: string; id: string; referencing_profiles: string[] }> {
    return apiRequest(`/api/providers/custom/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },

  /** Dry-run an endpoint before saving it — times a `/models` listing. */
  async testProvider(input: {
    base_url: string;
    api_key?: string;
    headers?: Record<string, string>;
    id?: string;
  }): Promise<ProviderTestResult> {
    return apiRequest('/api/providers/test', {
      method: 'POST',
      body: JSON.stringify(input),
      signal: AbortSignal.timeout(20_000),
    });
  },

  /** Where a turn on `model` actually goes, including the fallback behind it. */
  async getProviderRoute(model: string, fallback?: string): Promise<ProviderRouteResponse> {
    const params = new URLSearchParams({ model });
    if (fallback) params.set('fallback', fallback);
    return apiRequest(`/api/providers/route?${params}`, {
      signal: AbortSignal.timeout(15_000),
    });
  },
};
