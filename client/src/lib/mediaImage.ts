/**
 * Resolve stored media (a served-blob `…/raw` path) to a displayable object URL.
 *
 * A raw `<img src="/api/…/raw">` can't carry the app's auth header, so we fetch the bytes
 * via the authed API client (`ambassadorApi.fetchMediaBlob`) and hand back an
 * `URL.createObjectURL` blob URL — the same pattern `lib/audio.ts::SpeechPlayer` uses for
 * TTS. Used for generated avatars (`media:` refs) and image exhibits (raw paths). Results
 * are memoized per path (blobs are content-addressed server-side, so a path is stable).
 */

import { ambassadorApi } from './api/ambassador';
import { mediaAvatarPath } from './avatars';

const _cache = new Map<string, Promise<string>>();

// Object URLs pin their Blob in memory until revoked — an unbounded cache slowly
// accumulates every image/clip/video viewed in a session. LRU-cap it: on insert
// beyond the cap, evict + revoke the oldest entry. The cap is generous so eviction
// only fires in long sessions, where the evicted entry's component is almost
// certainly unmounted; a still-mounted consumer of a revoked URL shows its
// load-failure fallback and a re-render re-resolves (the cache re-fetches).
const _CACHE_MAX = 64;

function _touch(rawPath: string, p: Promise<string>): void {
  _cache.delete(rawPath); // re-insert → newest position (Map preserves order)
  _cache.set(rawPath, p);
  if (_cache.size <= _CACHE_MAX) return;
  const [oldestKey, oldestPromise] = _cache.entries().next().value as [string, Promise<string>];
  _cache.delete(oldestKey);
  oldestPromise.then((url) => URL.revokeObjectURL(url)).catch(() => undefined);
}

/** Resolve a served-blob raw path (e.g. `/api/workspaces/{ws}/documents/{doc}/raw`) to an
 * object URL (LRU-cached). Type-agnostic — images, audio, and video all resolve the same way. */
export function resolveMediaBlob(rawPath: string): Promise<string> {
  const cached = _cache.get(rawPath);
  if (cached) {
    _touch(rawPath, cached);
    return cached;
  }

  const p = ambassadorApi
    .fetchMediaBlob(rawPath)
    .then((blob) => URL.createObjectURL(blob))
    .catch((err) => {
      _cache.delete(rawPath); // allow a retry on transient failure
      throw err;
    });
  _touch(rawPath, p);
  return p;
}

/** Back-compat alias (predates audio/video support). */
export const resolveMediaImage = resolveMediaBlob;

/** Resolve a `media:{ws}/{doc}` avatar ref to an object URL (cached). Rejects if not a media ref. */
export function resolveAvatarImage(avatar: string): Promise<string> {
  const path = mediaAvatarPath(avatar);
  if (!path) return Promise.reject(new Error('not a media avatar'));
  return resolveMediaImage(path);
}
