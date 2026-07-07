/**
 * openExternal — open a URL in the user's real browser, everywhere.
 *
 * In the Tauri webview a bare `window.open` silently no-ops (no navigation
 * handler), so external links / OAuth sign-in pages never appear. We route
 * through the `opener` plugin (`openUrl`) when running under Tauri, and fall
 * back to `window.open` in browser/dev-web mode.
 *
 * Returns `true` if the open was dispatched, `false` if it failed (blocked
 * popup, plugin error) — callers surface a manual "open this link" affordance
 * on `false`.
 */
import { isTauri } from '@tauri-apps/api/core';
import { openUrl } from '@tauri-apps/plugin-opener';

export async function openExternal(url: string): Promise<boolean> {
  if (isTauri()) {
    try {
      await openUrl(url);
      return true;
    } catch {
      return false;
    }
  }
  try {
    const win = window.open(url, '_blank', 'noopener');
    return win != null;
  } catch {
    return false;
  }
}
