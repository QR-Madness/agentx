import { request as apiRequest } from './core';

/** Per-workspace shell sandbox. */
export type ShellBackend = 'bubblewrap' | 'container';

/** Lifecycle action on a container-backed shell. */
export type ContainerAction = 'start' | 'stop' | 'reset' | 'remove';

/** Status + live stats of a workspace's shell container (UI resource card). */
export interface ContainerStatus {
  state: 'none' | 'provisioning' | 'running' | 'stopped' | 'unavailable';
  image?: string;
  started_at?: string;
  last_used_at?: number | null;
  idle_gc_at?: number | null;
  memory_usage?: string | null;
  cpu_percent?: string | null;
  install_size?: string | null;
}

/** A file workspace (named container of documents). */
export interface Workspace {
  id: string;
  name: string;
  user_id: string;
  /** Per-workspace opt-in for sandboxed agent shell tools (default false). */
  allow_shell: boolean;
  /** Which shell sandbox this workspace uses. */
  shell_backend: ShellBackend;
  document_count: number;
  used_bytes: number;
  created_at: string | null;
  updated_at: string | null;
}

/** Ingestion status of an uploaded document. */
export type WorkspaceDocumentStatus = 'pending' | 'ready' | 'failed';

/** A document in a workspace (manifest row). */
export interface WorkspaceDocument {
  id: string;
  workspace_id: string;
  filename: string;
  content_type: string | null;
  size_bytes: number;
  sha256: string | null;
  tags: string[];
  summary: string;
  status: WorkspaceDocumentStatus;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export const workspacesApi = {
  async listWorkspaces(): Promise<{ workspaces: Workspace[] }> {
    return apiRequest('/api/workspaces');
  },

  async createWorkspace(name: string): Promise<{ workspace: Workspace }> {
    return apiRequest('/api/workspaces', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  },

  async getWorkspace(id: string): Promise<{ workspace: Workspace }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`);
  },

  async renameWorkspace(id: string, name: string): Promise<{ workspace: Workspace }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    });
  },

  async setWorkspaceShell(id: string, allowShell: boolean): Promise<{ workspace: Workspace }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ allow_shell: allowShell }),
    });
  },

  async setWorkspaceShellBackend(id: string, backend: ShellBackend): Promise<{ workspace: Workspace }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ shell_backend: backend }),
    });
  },

  async getWorkspaceContainer(id: string): Promise<{ container: ContainerStatus }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}/shell/container`);
  },

  async workspaceContainerAction(
    id: string, action: ContainerAction,
  ): Promise<{ container: ContainerStatus }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(id)}/shell/container/${action}`, { method: 'POST' },
    );
  },

  async deleteWorkspace(id: string): Promise<{ status: string; workspace_id: string }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, { method: 'DELETE' });
  },

  async listDocuments(workspaceId: string): Promise<{ documents: WorkspaceDocument[] }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(workspaceId)}/documents`);
  },

  async uploadDocument(workspaceId: string, file: File): Promise<{ document: WorkspaceDocument }> {
    const form = new FormData();
    form.append('file', file);
    return apiRequest(`/api/workspaces/${encodeURIComponent(workspaceId)}/documents`, {
      method: 'POST',
      body: form,
    });
  },

  async deleteDocument(
    workspaceId: string,
    documentId: string,
  ): Promise<{ status: string; document_id: string }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(documentId)}`,
      { method: 'DELETE' },
    );
  },
};
