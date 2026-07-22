/**
 * WelcomeProjectRow — the new-conversation project affordance shown under the
 * welcome-hero starters. Projects (files + standing instructions) used to be
 * reachable only from the command palette / sidebar; here they're *offered* at
 * the one moment their context matters — before the first message.
 *
 * Empty conversation, no project → up to a few project chips + a way into the
 * hub. Pick one and the pre-session conversation is tagged (rides `tab.id` meta
 * until the first message lands a session id, exactly like "New chat in this
 * project"); the row flips to the loaded-context state. Attached → name + a
 * one-glance summary + detach.
 */

import { FolderKanban, FolderPlus, Plus, X } from 'lucide-react';
import { api } from '../../lib/api';
import { useApi } from '../../lib/hooks';
import { patchMeta } from '../../lib/conversationMeta';
import { useModal } from '../../contexts/ModalContext';
import { SURFACES } from '../../lib/surfaces';

const HOME_ID = 'ws_home';
const MAX_CHIPS = 4;

export function WelcomeProjectRow({
  convKey,
  sessionId,
  attachedWorkspaceId,
}: {
  convKey: string;
  sessionId?: string;
  attachedWorkspaceId?: string;
}) {
  const { data } = useApi(() => api.listWorkspaces(), []);
  const { openModal } = useModal();

  const projects = (data?.workspaces ?? []).filter(w => w.id !== HOME_ID);
  const attached = projects.find(p => p.id === attachedWorkspaceId) ?? null;

  const attach = (id: string) => {
    patchMeta(convKey, { workspaceId: id });
    // If a session already exists, make membership durable now; otherwise the
    // pre-session meta records it on the first message.
    if (sessionId) api.linkConversation(id, sessionId).catch(() => undefined);
  };
  const detach = () => {
    if (!attachedWorkspaceId) return;
    patchMeta(convKey, { workspaceId: undefined });
    if (sessionId) api.unlinkConversation(attachedWorkspaceId, sessionId).catch(() => undefined);
  };
  const openHub = (id?: string) =>
    openModal(id ? { ...SURFACES.workspaces, props: { initialWorkspaceId: id } } : SURFACES.workspaces);

  // Loaded-context state — the conversation is starting inside a project.
  if (attached) {
    return (
      <div className="welcome-project welcome-project--on">
        <FolderKanban size={13} className="welcome-project-glyph" />
        <button className="welcome-project-name" onClick={() => openHub(attached.id)} title="Open this project">
          {attached.name}
        </button>
        <span className="welcome-project-meta">
          {attached.document_count} file{attached.document_count === 1 ? '' : 's'}
          {attached.instructions ? ' · instructions on' : ''}
        </span>
        <button
          className="welcome-project-detach"
          onClick={detach}
          aria-label="Start without a project"
          title="Start without a project"
        >
          <X size={13} />
        </button>
      </div>
    );
  }

  // No projects yet — a quiet invitation rather than an empty control.
  if (projects.length === 0) {
    return (
      <button className="welcome-project welcome-project--empty" onClick={() => openHub()}>
        <FolderPlus size={13} /> Start a project to bundle files &amp; instructions
      </button>
    );
  }

  // The offer: pick a project to start inside.
  return (
    <div className="welcome-project">
      <span className="welcome-project-lead">
        <FolderKanban size={13} /> Start in a project
      </span>
      <div className="welcome-project-chips">
        {projects.slice(0, MAX_CHIPS).map(p => (
          <button
            key={p.id}
            className="welcome-project-chip"
            onClick={() => attach(p.id)}
            title={p.description || `Start this conversation in ${p.name}`}
          >
            {p.name}
          </button>
        ))}
        {projects.length > MAX_CHIPS ? (
          <button className="welcome-project-chip welcome-project-chip--more" onClick={() => openHub()}>
            +{projects.length - MAX_CHIPS} more
          </button>
        ) : (
          <button className="welcome-project-chip welcome-project-chip--new" onClick={() => openHub()}>
            <Plus size={12} /> New
          </button>
        )}
      </div>
    </div>
  );
}
