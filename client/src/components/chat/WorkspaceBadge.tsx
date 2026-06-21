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
      className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/15 px-2.5 py-1 text-xs font-semibold text-accent transition-colors hover:bg-accent hover:text-fg-inverse"
      title={`Attached workspace: ${name} — manage in Workspaces`}
    >
      <FolderOpen size={14} />
      <span className="max-w-[12rem] truncate">{name}</span>
    </button>
  );
}
