/**
 * Resolve a `media:` avatar ref to a displayable object URL.
 *
 * A generated avatar is stored as `media:{ws}/{doc}` on the profile. The served blob lives
 * behind the authed API (`…/documents/{doc}/raw`), and an `<img src>` can't carry the app's
 * auth header — so we fetch the bytes via the authed client (`ambassadorApi.fetchMediaBlob`)
 * and hand back an `URL.createObjectURL` blob URL, exactly like `lib/audio.ts::SpeechPlayer`
 * does for TTS. Results are memoized per ref (the blob is content-addressed server-side, so a
 * ref is stable) to avoid refetching across the many render sites.
 */

import { ambassadorApi } from './api/ambassador';
import { mediaAvatarPath } from './avatars';

const _cache = new Map<string, Promise<string>>();

/** Resolve a `media:` avatar ref to an object URL (cached). Rejects if not a media ref or on fetch error. */
export function resolveAvatarImage(avatar: string): Promise<string> {
  const cached = _cache.get(avatar);
  if (cached) return cached;

  const path = mediaAvatarPath(avatar);
  if (!path) return Promise.reject(new Error('not a media avatar'));

  const p = ambassadorApi
    .fetchMediaBlob(path)
    .then((blob) => URL.createObjectURL(blob))
    .catch((err) => {
      _cache.delete(avatar); // allow a retry on transient failure
      throw err;
    });
  _cache.set(avatar, p);
  return p;
}
