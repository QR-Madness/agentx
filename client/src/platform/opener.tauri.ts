/**
 * opener.tauri — desktop impl. In the Tauri webview a bare `window.open` silently
 * no-ops (no navigation handler), so external links / OAuth pages never appear;
 * the `opener` plugin (`openUrl`) hands the URL to the OS browser instead.
 *
 * Only loaded when `__IS_TAURI__` is true, so this is the sole legal place to
 * import `@tauri-apps/plugin-opener` (enforced by importBoundary.test.ts).
 */

import { openUrl } from '@tauri-apps/plugin-opener';
import type { Opener } from './opener';

export function createTauriOpener(): Opener {
  return {
    async openExternalUrl(url: string): Promise<boolean> {
      try {
        await openUrl(url);
        return true;
      } catch {
        return false;
      }
    },
  };
}
