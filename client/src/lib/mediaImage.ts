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

/** Resolve a served-blob raw path (e.g. `/api/workspaces/{ws}/documents/{doc}/raw`) to an object URL (cached). */
export function resolveMediaImage(rawPath: string): Promise<string> {
  const cached = _cache.get(rawPath);
  if (cached) return cached;

  const p = ambassadorApi
    .fetchMediaBlob(rawPath)
    .then((blob) => URL.createObjectURL(blob))
    .catch((err) => {
      _cache.delete(rawPath); // allow a retry on transient failure
      throw err;
    });
  _cache.set(rawPath, p);
  return p;
}

/** Resolve a `media:{ws}/{doc}` avatar ref to an object URL (cached). Rejects if not a media ref. */
export function resolveAvatarImage(avatar: string): Promise<string> {
  const path = mediaAvatarPath(avatar);
  if (!path) return Promise.reject(new Error('not a media avatar'));
  return resolveMediaImage(path);
}
