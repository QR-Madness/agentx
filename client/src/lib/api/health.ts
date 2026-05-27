import { request as apiRequest } from './core';
import { getActiveServer, markServerConnected, updateActiveServerMetadata } from '../storage';
import type { HealthResponse, VersionInfo } from './types';

export const healthApi = {
  // === Health ===

  async health(includeMemory = false, includeStorage = false): Promise<HealthResponse> {
    const params = new URLSearchParams();
    if (includeMemory) params.set('include_memory', 'true');
    if (includeStorage) params.set('include_storage', 'true');
    const path = params.toString() ? `/api/health?${params}` : '/api/health';
    const result = await apiRequest<HealthResponse>(path, {}, true); // Skip auth for health

    // Update cache with health status
    const server = getActiveServer();
    if (server) {
      markServerConnected(server.id);
      updateActiveServerMetadata({
        cache: {
          lastHealthCheck: new Date().toISOString(),
          lastHealthStatus: result.status,
        },
      });
    }

    return result;
  },

  // === Version ===

  async version(): Promise<VersionInfo> {
    return apiRequest<VersionInfo>('/api/version', {}, true); // Skip auth for version
  },
};
