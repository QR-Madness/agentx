/**
 * Client-side per-conversation metadata — the single, extensible store behind
 * conversation-management (pin / archive / icon / color / group), plus the title
 * override it generalizes. Per-server, persisted to localStorage (there is no
 * server endpoint yet — a future one can supersede this, like the title map it
 * replaces). Reserved fields (`workspaceId`/`fileRefs`) leave room for future
 * workspace/file linking with no behavior today.
 *
 * Keyed by the **session/conversation id** (`tab.sessionId ?? tab.id` for open
 * tabs, `conversation_id` for server conversations) — and since a tab's
 * sessionId IS its conversation_id, meta survives the open→closed transition.
 *
 * Reactivity: a module pub/sub consumed via `useConversationMeta()`
 * (useSyncExternalStore). Components read `getMeta(id)` in render; the hook just
 * forces a re-render on any write.
 *
 * `workspaceId` (a project attach) is now server-backed for established
 * conversations (`workspace_conversations`) — here it remains the carrier for
 * pre-session tabs and the fast path for the chat request's `workspace_id`.
 */

import { useSyncExternalStore } from 'react';
import { getActiveServerId } from './storage';

export interface ConversationMeta {
  title?: string;
  pinned?: boolean;
  archived?: boolean;
  icon?: string;   // avatar id from lib/avatars
  color?: string;  // a CONVERSATION_COLORS key
  group?: string;  // custom group label
  workspaceId?: string; // attached project (workspace); server-backed once a session exists
  // --- reserved for future file linking (no behavior yet) ---
  fileRefs?: string[];
}

type MetaMap = Record<string, ConversationMeta>;

const KEY = (serverId: string) => `agentx:server:${serverId}:convMeta`;
const LEGACY_TITLES_KEY = (serverId: string) => `agentx:server:${serverId}:convTitles`;
const MIGRATED_KEY = (serverId: string) => `agentx:server:${serverId}:convMetaMigrated`;

// In-memory cache so getMeta() doesn't parse localStorage on every row render.
let cache: { serverId: string; map: MetaMap } | null = null;

const listeners = new Set<() => void>();
let version = 0;

function emit(): void {
  version += 1;
  listeners.forEach((l) => l());
}

/** One-time import of the legacy title-override map into meta.title. */
function migrateLegacyTitles(serverId: string, map: MetaMap): void {
  try {
    if (localStorage.getItem(MIGRATED_KEY(serverId))) return;
    const raw = localStorage.getItem(LEGACY_TITLES_KEY(serverId));
    if (raw) {
      const titles = JSON.parse(raw) as Record<string, string>;
      for (const [id, title] of Object.entries(titles)) {
        if (title && !map[id]?.title) map[id] = { ...map[id], title };
      }
    }
    localStorage.setItem(MIGRATED_KEY(serverId), '1');
  } catch {
    /* best-effort migration */
  }
}

function load(serverId: string): MetaMap {
  if (cache && cache.serverId === serverId) return cache.map;
  let map: MetaMap = {};
  try {
    const raw = localStorage.getItem(KEY(serverId));
    map = raw ? (JSON.parse(raw) as MetaMap) : {};
  } catch {
    map = {};
  }
  migrateLegacyTitles(serverId, map);
  cache = { serverId, map };
  return map;
}

function persist(serverId: string, map: MetaMap): void {
  cache = { serverId, map };
  try {
    localStorage.setItem(KEY(serverId), JSON.stringify(map));
  } catch {
    /* localStorage unavailable — meta is best-effort */
  }
  emit();
}

export function getMeta(id: string | null | undefined): ConversationMeta {
  const serverId = getActiveServerId();
  if (!id || !serverId) return {};
  return load(serverId)[id] ?? {};
}

/** Merge a partial into a conversation's meta (deletes keys set to undefined/false/empty). */
export function patchMeta(id: string, partial: Partial<ConversationMeta>): void {
  const serverId = getActiveServerId();
  if (!id || !serverId) return;
  const map = { ...load(serverId) };
  const next: ConversationMeta = { ...map[id], ...partial };
  // Prune falsy/empty so the map stays lean and "is it set?" checks are simple.
  (Object.keys(next) as (keyof ConversationMeta)[]).forEach((k) => {
    const v = next[k];
    if (v === undefined || v === false || v === '' || (Array.isArray(v) && v.length === 0)) {
      delete next[k];
    }
  });
  if (Object.keys(next).length === 0) delete map[id];
  else map[id] = next;
  persist(serverId, map);
}

/**
 * Attach a workspace to a conversation only if it has none yet — used when stored
 * media (a generated/uploaded image) falls back to the Home workspace. Returns true
 * when this call newly attached (so the caller can fire a one-time "saved to Home"
 * notice); false if the conversation already had a workspace.
 */
export function attachWorkspaceOnce(id: string | null | undefined, workspaceId: string): boolean {
  if (!id || !workspaceId) return false;
  if (getMeta(id).workspaceId) return false;
  patchMeta(id, { workspaceId });
  return true;
}

/**
 * Move a conversation's meta to a new key — used when a pre-session tab (keyed by
 * `tab.id`) gets its server `sessionId`, so a pre-session project attach / rename /
 * pin survives the transition instead of being orphaned under the dead tab id.
 * Existing meta under `toId` wins field-by-field (it's the more established key).
 */
export function migrateMeta(fromId: string, toId: string): void {
  if (!fromId || !toId || fromId === toId) return;
  const from = getMeta(fromId);
  if (Object.keys(from).length === 0) return;
  patchMeta(toId, { ...from, ...getMeta(toId) });
  clearMeta(fromId);
}

export function clearMeta(id: string): void {
  const serverId = getActiveServerId();
  if (!id || !serverId) return;
  const map = { ...load(serverId) };
  if (id in map) {
    delete map[id];
    persist(serverId, map);
  }
}

/** Snapshot of all meta entries for the active server (used by the one-time
 *  project-membership sync; not reactive). */
export function listMetaEntries(): [string, ConversationMeta][] {
  const serverId = getActiveServerId();
  if (!serverId) return [];
  return Object.entries(load(serverId));
}

/** All distinct custom group labels currently in use (for the "Move to group" menu). */
export function listGroups(): string[] {
  const serverId = getActiveServerId();
  if (!serverId) return [];
  const groups = new Set<string>();
  for (const m of Object.values(load(serverId))) {
    if (m.group) groups.add(m.group);
  }
  return Array.from(groups).sort((a, b) => a.localeCompare(b));
}

// --- Title shim surface (generalizes conversationTitles.ts) ---

export function getTitleOverride(id: string, serverId?: string): string | undefined {
  void serverId; // active server is the only scope
  return getMeta(id).title;
}

export function setTitleOverride(id: string, title: string): void {
  patchMeta(id, { title: title.trim() });
}

export function clearTitleOverride(id: string): void {
  patchMeta(id, { title: undefined });
}

export function getDisplayTitle(id: string | null | undefined, fallback: string): string {
  if (!id) return fallback;
  return getMeta(id).title ?? fallback;
}

// --- Reactivity hook ---

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** Re-render on any meta change. Read values with getMeta(id) in render. */
export function useConversationMeta(): number {
  return useSyncExternalStore(subscribe, () => version, () => version);
}

/** Test-only: drop the in-memory cache (the app never clears localStorage under
 *  a live server, so this is unnecessary outside tests). */
export function __resetMetaCacheForTests(): void {
  cache = null;
}
