import { request as apiRequest } from './core';

export const toolOutputApi = {
  // === Tool Output Storage ===

  async listToolOutputs(pattern?: string): Promise<{
    outputs: Array<{
      key: string;
      tool_name: string;
      tool_call_id: string;
      size_chars: number;
      stored_at: string;
    }>;
    count: number;
  }> {
    const params = pattern ? `?pattern=${encodeURIComponent(pattern)}` : '';
    return apiRequest(`/api/tool-outputs${params}`);
  },

  async getToolOutput(key: string, options?: {
    offset?: number;
    limit?: number;
    metadataOnly?: boolean;
  }): Promise<{
    key: string;
    tool_name: string;
    tool_call_id: string;
    content?: string;
    offset?: number;
    limit?: number;
    total_size?: number;
    size_chars?: number;
    stored_at: string;
  }> {
    const params = new URLSearchParams();
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.metadataOnly) params.set('metadata_only', 'true');
    const query = params.toString() ? `?${params}` : '';
    return apiRequest(`/api/tool-outputs/${encodeURIComponent(key)}${query}`);
  },

  async deleteToolOutput(key: string): Promise<{ deleted: boolean; key: string }> {
    return apiRequest(`/api/tool-outputs/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    });
  },
};
