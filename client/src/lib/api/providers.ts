import { request as apiRequest } from './core';
import type {
  AliasMigrationScan,
  ModelInfo,
  OpenRouterAccount,
  OpenRouterLinkStart,
  OpenRouterLinkStatus,
  ProviderCatalogEntry,
  ProviderCatalogResponse,
  ProviderCustomInput,
  ProviderInfo,
  ProviderRouteResponse,
  ProvidersHealthResponse,
  ProviderTestResult,
  StaleModelRef,
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

  // === OpenRouter account linking (OAuth PKCE) ===

  /** Begin a link. Returns the consent URL to open in the user's real browser;
   *  the PKCE verifier and the minted key stay server-side throughout. */
  async startOpenRouterLink(): Promise<OpenRouterLinkStart> {
    return apiRequest('/api/providers/openrouter/oauth/start', { method: 'POST' });
  },

  /** Poll a pending link. `expired` means stop polling. */
  async getOpenRouterLinkStatus(flowId: string): Promise<OpenRouterLinkStatus> {
    return apiRequest(
      `/api/providers/openrouter/oauth/status?flow_id=${encodeURIComponent(flowId)}`
    );
  },

  /** Abandon a pending link (the user closed the consent tab). */
  async cancelOpenRouterLink(flowId: string): Promise<{ status: string }> {
    return apiRequest('/api/providers/openrouter/oauth/cancel', {
      method: 'POST',
      body: JSON.stringify({ flow_id: flowId }),
    });
  },

  /** Forget the stored OpenRouter key. Local only — revoking it upstream needs
   *  a management key AgentX doesn't hold, so the response carries the URL where
   *  the user can do that themselves. */
  async unlinkOpenRouter(): Promise<{ status: string; had_key: boolean; revoke_url: string }> {
    return apiRequest('/api/providers/openrouter/unlink', { method: 'POST' });
  },

  /** Balance and spend for the stored key. Never throws for an unconfigured or
   *  rejected key — it answers `available: false`. */
  async getOpenRouterAccount(): Promise<OpenRouterAccount> {
    return apiRequest('/api/providers/openrouter/account', {
      signal: AbortSignal.timeout(15_000),
    });
  },

  /** Scan for model ids left stale by OpenRouter's `~` alias rename. Read-only. */
  async scanOpenRouterAliases(): Promise<AliasMigrationScan> {
    return apiRequest('/api/providers/openrouter/alias-migration', {
      signal: AbortSignal.timeout(20_000),
    });
  },

  /** Repair the given references. Opt-in — pass exactly what the user confirmed. */
  async repairOpenRouterAliases(
    refs: StaleModelRef[]
  ): Promise<{ applied: string[]; failed: { location: string; error: string }[]; count: number }> {
    return apiRequest('/api/providers/openrouter/alias-migration', {
      method: 'POST',
      body: JSON.stringify({ refs }),
    });
  },
};
