/**
 * Recent commands — a small MRU list of command ids, persisted to localStorage,
 * so the palette can surface "what you just did" at the top when the query is
 * empty. Resolution of ids → live commands happens in the palette via the
 * command registry (unknown ids are dropped).
 */

const KEY = 'agentx:recent-commands';
const CAP = 6;

export function getRecentCommandIds(): string[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

export function pushRecentCommand(id: string): void {
  if (!id) return;
  try {
    const next = [id, ...getRecentCommandIds().filter((x) => x !== id)].slice(0, CAP);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* localStorage unavailable — recents are best-effort */
  }
}
