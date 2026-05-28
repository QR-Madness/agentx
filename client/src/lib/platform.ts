/**
 * platform.ts — Lightweight runtime environment detection for chrome decisions.
 *
 * We deliberately avoid adding `@tauri-apps/plugin-os` (a Rust dependency) just
 * to branch the window chrome. Webview-side detection is enough:
 *
 *  - `isTauri`            true inside the Tauri webview, false in `task dev:web`
 *                         (browser provides its own frame there).
 *  - `isMac`              macOS keeps the native title bar (we can't test custom
 *                         traffic-light chrome), so we never draw window buttons.
 *  - `showWindowControls` only Windows/Linux desktop gets our min/max/close +
 *                         the frameless drag strip.
 */

export const isTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

export const isMac =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad|iPod/i.test(navigator.userAgent);

/**
 * Render our custom window controls (and treat the strip as the window's top
 * edge) only on frameless desktop platforms — i.e. Tauri on Windows/Linux.
 * macOS uses native decorations; the web build uses the browser chrome.
 */
export const showWindowControls = isTauri && !isMac;
