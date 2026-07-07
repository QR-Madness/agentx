/**
 * platform — the single façade app code imports for anything shell-specific.
 *
 * Consumers write `platform.opener.openExternalUrl(url)` /
 * `platform.window.minimize()` and never know or care which shell they're in.
 * The rule that keeps this honest: `@tauri-apps/*` may ONLY be imported under
 * `src/platform/` (enforced by importBoundary.test.ts). If a new feature needs a
 * Tauri API, add a capability here — never reach around the façade.
 */

import { opener } from './opener';
import { windowControls } from './window';

export { IS_TAURI, isTauriRuntime } from './runtime';
export type { Opener } from './opener';
export type { WindowBackend, ResizeDir } from './window';

export const platform = {
  opener,
  window: windowControls,
} as const;
