/**
 * Client-side conversation title overrides.
 *
 * There is no server-side conversation rename, so renaming a *past/server*
 * conversation (one not currently an open tab) is stored locally as a
 * `conversation_id → custom title` map, scoped per server. Open tabs keep their
 * renamed title on the tab itself (see `renameTab` in useConversationTabs); this
 * map is the source of truth for everything else, applied via `getDisplayTitle`.
 *
 * Titles are per-device (localStorage) — acceptable for the single-user local
 * app; a future server endpoint could supersede this.
 */

import { getActiveServerId } from './storage';

const KEY = (serverId: string) => `agentx:server:${serverId}:convTitles`;

type TitleMap = Record<string, string>;

function read(serverId?: string): TitleMap {
  const id = serverId ?? getActiveServerId();
  if (!id) return {};
  try {
    const raw = localStorage.getItem(KEY(id));
    return raw ? (JSON.parse(raw) as TitleMap) : {};
  } catch {
    return {};
  }
}

function write(map: TitleMap, serverId?: string): void {
  const id = serverId ?? getActiveServerId();
  if (!id) return;
  localStorage.setItem(KEY(id), JSON.stringify(map));
}

/** The custom title for a conversation, or undefined if none is set. */
export function getTitleOverride(conversationId: string, serverId?: string): string | undefined {
  if (!conversationId) return undefined;
  return read(serverId)[conversationId];
}

/** Set (or, when blank, clear) a conversation's custom title. */
export function setTitleOverride(conversationId: string, title: string, serverId?: string): void {
  if (!conversationId) return;
  const trimmed = title.trim();
  const map = read(serverId);
  if (!trimmed) {
    delete map[conversationId];
  } else {
    map[conversationId] = trimmed;
  }
  write(map, serverId);
}

/** Remove a conversation's custom title (revert to its server-derived title). */
export function clearTitleOverride(conversationId: string, serverId?: string): void {
  const map = read(serverId);
  if (conversationId in map) {
    delete map[conversationId];
    write(map, serverId);
  }
}

/** The display title for a conversation: the override if set, else the fallback. */
export function getDisplayTitle(conversationId: string | null | undefined, fallback: string): string {
  if (!conversationId) return fallback;
  return read()[conversationId] ?? fallback;
}
