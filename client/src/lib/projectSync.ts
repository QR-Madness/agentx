/**
 * One-time migration of client-side project attachments to the server.
 *
 * Before Projects v1, a conversation's workspace attach lived only in
 * localStorage (`conversationMeta.workspaceId`) and was re-sent on every chat
 * turn. Membership is now durable on the server (`workspace_conversations`);
 * this pushes the pre-existing local links up once per server so old
 * conversations appear in their project without needing another message.
 *
 * Best-effort per entry; the done-flag is only set after a pass with no
 * network-level failures so a flaky first run retries next launch.
 */

import { api } from './api';
import { listMetaEntries, patchMeta } from './conversationMeta';
import { getActiveServerId } from './storage';

const SYNC_KEY = (serverId: string) => `agentx:server:${serverId}:projectMembershipSynced`;

/** Home is a personal media dump, not a project — never sync it. */
const HOME_WORKSPACE_ID = 'ws_home';

// Server conversation ids are UUIDs; live pre-session tab ids are not. Only
// UUID-keyed entries are real conversations the server can link.
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function syncLocalProjectLinksOnce(): Promise<void> {
  const serverId = getActiveServerId();
  if (!serverId) return;
  try {
    if (localStorage.getItem(SYNC_KEY(serverId))) return;
  } catch {
    return; // no localStorage — nothing to sync from either
  }

  const links = listMetaEntries().filter(
    ([id, meta]) =>
      UUID_RE.test(id) && !!meta.workspaceId && meta.workspaceId !== HOME_WORKSPACE_ID,
  );
  if (links.length === 0) {
    try {
      localStorage.setItem(SYNC_KEY(serverId), '1');
    } catch { /* best-effort */ }
    return;
  }

  // Validate against the live workspace list first — links pointing at a
  // deleted workspace are stale meta, not something to PUT (they'd just 404).
  let liveIds: Set<string>;
  try {
    const { workspaces } = await api.listWorkspaces();
    liveIds = new Set(workspaces.map((w) => w.id));
  } catch {
    return; // can't validate now — retry next launch
  }

  let clean = true;
  for (const [conversationId, meta] of links) {
    if (!liveIds.has(meta.workspaceId!)) {
      patchMeta(conversationId, { workspaceId: undefined }); // clear the dead link
      continue;
    }
    try {
      await api.linkConversation(meta.workspaceId!, conversationId);
    } catch (err) {
      // 4xx = the conversation/workspace vanished meanwhile — nothing to retry.
      // Anything else (network, 5xx) leaves the flag unset so we retry later.
      const status = (err as { status?: number }).status;
      if (!(typeof status === 'number' && status >= 400 && status < 500)) clean = false;
    }
  }

  if (clean) {
    try {
      localStorage.setItem(SYNC_KEY(serverId), '1');
    } catch {
      /* best-effort */
    }
  }
}
