/**
 * WorkspaceBadge — chat-header chip shown when a conversation has a workspace
 * attached (Document RAG). Resolves the workspace name from its id and opens the
 * Workspaces drawer on click.
 */

import { FolderOpen } from 'lucide-react';
import { api } from '../../lib/api';
import { useApi } from '../../lib/hooks';

export function WorkspaceBadge({ workspaceId, onOpen }: { workspaceId: string; onOpen: () => void }) {
  const { data } = useApi(() => api.getWorkspace(workspaceId), [workspaceId]);
  const name = data?.workspace.name ?? 'Workspace';
  return (
    <button
      type="button"
      onClick={onOpen}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs bg-surface-sunken text-fg-secondary hover:text-fg"
      title={`Attached workspace: ${name} — manage in Workspaces`}
    >
      <FolderOpen size={12} />
      <span className="max-w-[10rem] truncate">{name}</span>
    </button>
  );
}
