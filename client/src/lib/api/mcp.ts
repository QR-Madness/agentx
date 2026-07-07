import { request as apiRequest } from './core';
import type { MCPServer, MCPServerConfigInput, MCPTool } from './types';

export const mcpApi = {
  // === MCP ===

  async listMCPServers(): Promise<{ servers: MCPServer[] }> {
    return apiRequest('/api/mcp/servers');
  },

  async listMCPTools(): Promise<{ tools: MCPTool[] }> {
    // Backend returns `server_name`; map to `server` for the TS contract so
    // the property the UI reads (`t.server`) is actually populated.
    const raw = await apiRequest<{ tools: Array<{
      name: string;
      description: string;
      server_name?: string;
      server?: string;
      input_schema?: Record<string, unknown>;
    }> }>('/api/mcp/tools');
    return {
      tools: raw.tools.map(t => ({
        name: t.name,
        description: t.description,
        server: t.server ?? t.server_name ?? '',
        inputSchema: t.input_schema,
      })),
    };
  },

  async listMCPResources(): Promise<{ resources: unknown[] }> {
    return apiRequest('/api/mcp/resources');
  },

  /** Connect a server. OAuth servers needing consent return
   *  `status: "auth_required"` (HTTP 202) with an `authorization_url` to open;
   *  the connect completes in the background once the user authorizes. */
  async connectMCPServer(server: string): Promise<{
    status: string;
    server: string;
    tools_count?: number;
    resources_count?: number;
    authorization_url?: string;
  }> {
    return apiRequest('/api/mcp/connect', {
      method: 'POST',
      body: JSON.stringify({ server }),
    });
  },

  /** Forget a server's OAuth tokens + registration (forces a fresh sign-in). */
  async resetMCPServerAuth(server: string): Promise<{ status: string; server: string; cleared: boolean }> {
    return apiRequest(`/api/mcp/servers/${encodeURIComponent(server)}/auth/reset`, {
      method: 'POST',
    });
  },

  /** Abort an in-flight OAuth sign-in (the "Cancel" on the sign-in dialog) so a
   *  late browser completion can't flip the server to "signed in". Stored
   *  tokens (if any) are left intact — use resetMCPServerAuth to forget those. */
  async cancelMCPServerAuth(server: string): Promise<{ status: string; server: string; cancelled: boolean }> {
    return apiRequest(`/api/mcp/servers/${encodeURIComponent(server)}/auth/cancel`, {
      method: 'POST',
    });
  },

  async connectAllMCPServers(): Promise<{ results: Record<string, { status: string; error?: string }> }> {
    return apiRequest('/api/mcp/connect', {
      method: 'POST',
      body: JSON.stringify({ all: true }),
    });
  },

  async disconnectMCPServer(server: string): Promise<{ status: string; server: string }> {
    return apiRequest('/api/mcp/disconnect', {
      method: 'POST',
      body: JSON.stringify({ server }),
    });
  },

  async createMCPServer(name: string, config: MCPServerConfigInput): Promise<{ status: string; server: MCPServer }> {
    return apiRequest('/api/mcp/servers', {
      method: 'POST',
      body: JSON.stringify({ name, config }),
    });
  },

  async updateMCPServer(name: string, config: MCPServerConfigInput, rename?: string): Promise<{ status: string; server: MCPServer }> {
    return apiRequest(`/api/mcp/servers/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify(rename && rename !== name ? { config, rename } : { config }),
    });
  },

  async deleteMCPServer(name: string): Promise<{ status: string; server: string }> {
    return apiRequest(`/api/mcp/servers/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
  },

  async validateMCPServer(name: string, config: MCPServerConfigInput): Promise<{ valid: boolean; errors: string[] }> {
    return apiRequest('/api/mcp/servers/validate', {
      method: 'POST',
      body: JSON.stringify({ name, config }),
    });
  },
};
