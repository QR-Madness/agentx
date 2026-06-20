import { request as apiRequest } from './core';

/** A file workspace (named container of documents). */
export interface Workspace {
  id: string;
  name: string;
  user_id: string;
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
