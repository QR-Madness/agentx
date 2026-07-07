/**
 * openExternal — open a URL in the user's real browser, everywhere.
 *
 * Thin façade kept for its existing callers; the actual Tauri/web split lives in
 * `src/platform/opener.*` (a bare `window.open` no-ops inside the Tauri webview,
 * so desktop routes through the `opener` plugin instead).
 *
 * Returns `true` if the open was dispatched, `false` if it failed (blocked
 * popup, plugin error) — callers surface a manual "open this link" affordance
 * on `false`.
 */
import { platform } from '../platform';

export function openExternal(url: string): Promise<boolean> {
  return platform.opener.openExternalUrl(url);
}
