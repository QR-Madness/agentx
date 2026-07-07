import { getBaseUrl, request as apiRequest } from './core';
import { getAuthToken, getActiveGatewayToken } from '../storage';

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const token = getAuthToken();
  if (token) headers['X-Auth-Token'] = token;
  const gatewayToken = getActiveGatewayToken();
  if (gatewayToken) headers['AgentX-Gateway-Token'] = gatewayToken;
  return headers;
}

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

/** A file workspace — surfaces to the user as a **Project** (files + instructions
 * + conversations). Internal naming stays `workspace`. */
export interface Workspace {
  id: string;
  name: string;
  user_id: string;
  /** Project description, shown in the hub (max 500 chars). */
  description: string;
  /** Project instructions, injected into every turn (max 8000 chars). */
  instructions: string;
  /** Per-workspace opt-in for sandboxed agent shell tools (default false). */
  allow_shell: boolean;
  /** Which shell sandbox this workspace uses. */
  shell_backend: ShellBackend;
  document_count: number;
  used_bytes: number;
  created_at: string | null;
  updated_at: string | null;
}

/** Editable project fields (PATCH /api/workspaces/{id}). */
export interface WorkspacePatch {
  name?: string;
  description?: string;
  instructions?: string;
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

  async updateWorkspace(id: string, patch: WorkspacePatch): Promise<{ workspace: Workspace }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  },

  /** The project's conversations (durable server-side membership). */
  async listWorkspaceConversations(
    id: string,
  ): Promise<{ conversations: import('./types').ConversationSummary[]; total: number }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(id)}/conversations`);
  },

  /** Durably add a conversation to a project (moves it if already in another). */
  async linkConversation(
    id: string, conversationId: string,
  ): Promise<{ status: string; workspace_id: string; conversation_id: string }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(id)}/conversations/${encodeURIComponent(conversationId)}`,
      { method: 'PUT' },
    );
  },

  /** Remove a conversation from a project. */
  async unlinkConversation(
    id: string, conversationId: string,
  ): Promise<{ status: string; workspace_id: string; conversation_id: string }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(id)}/conversations/${encodeURIComponent(conversationId)}`,
      { method: 'DELETE' },
    );
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

  /** Re-run ingestion for a document (e.g. after a failed embedding under load). */
  async reingestDocument(
    workspaceId: string,
    documentId: string,
  ): Promise<{ document: WorkspaceDocument }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(documentId)}/reingest`,
      { method: 'POST' },
    );
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

  /** Create a text/markdown document (hub "New document"; 409 on filename collision). */
  async createTextDocument(
    workspaceId: string,
    filename: string,
    content: string,
  ): Promise<{ document: WorkspaceDocument }> {
    return apiRequest(`/api/workspaces/${encodeURIComponent(workspaceId)}/documents/text`, {
      method: 'POST',
      body: JSON.stringify({ filename, content }),
    });
  },

  /** Replace a text document's content (re-ingests). `expectedSha256` is an
   *  optimistic-concurrency check — a mismatch returns 409 with `current_sha256`. */
  async updateTextDocument(
    workspaceId: string,
    documentId: string,
    content: string,
    expectedSha256?: string | null,
  ): Promise<{ document: WorkspaceDocument }> {
    return apiRequest(
      `/api/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(documentId)}/text`,
      {
        method: 'PUT',
        body: JSON.stringify({
          content,
          ...(expectedSha256 ? { expected_sha256: expectedSha256 } : {}),
        }),
      },
    );
  },

  /** Fetch a document's raw bytes via the authed client (a bare <img>/<iframe> src
   *  can't carry the auth header). `sha` cache-busts after edits — the /raw path is
   *  stable but its content changes when a text document is updated. */
  async fetchDocumentBlob(
    workspaceId: string,
    documentId: string,
    sha?: string | null,
  ): Promise<Blob> {
    const path =
      `/api/workspaces/${encodeURIComponent(workspaceId)}/documents/${encodeURIComponent(documentId)}/raw` +
      (sha ? `?v=${encodeURIComponent(sha)}` : '');
    const response = await fetch(`${getBaseUrl()}${path}`, { headers: authHeaders() });
    if (!response.ok) {
      throw { message: `Document fetch failed (${response.status})`, status: response.status, kind: 'http' };
    }
    return response.blob();
  },

  /** Fetch a text document's content as a string. */
  async fetchDocumentText(
    workspaceId: string,
    documentId: string,
    sha?: string | null,
  ): Promise<string> {
    const blob = await this.fetchDocumentBlob(workspaceId, documentId, sha);
    return blob.text();
  },
};
