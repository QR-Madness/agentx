import { request as apiRequest } from './core';
import type { UsageMetrics } from './types';

export const metricsApi = {
  /** Aggregate token/cost/latency usage over the last `days` days (default 14). */
  async getUsageMetrics(days?: number): Promise<UsageMetrics> {
    const query = days ? `?days=${days}` : '';
    return apiRequest(`/api/metrics/usage${query}`);
  },
};
