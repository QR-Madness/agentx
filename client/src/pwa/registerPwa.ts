/**
 * registerPwa — service-worker registration for the web/PWA build.
 *
 * Two jobs:
 *  1. Register the `registerType: 'prompt'` service worker and bridge its
 *     lifecycle to the app's toast system via window events (a React listener,
 *     `PwaToasts`, turns these into styled toasts — this module runs at boot,
 *     outside React).
 *  2. Recover from stale lazy-chunk 404s after a deploy: when a dynamic import's
 *     asset is gone (hashes changed under a still-open tab), reload once —
 *     loop-guarded so a genuinely broken deploy can't wedge in a reload cycle.
 *
 * No-op under Tauri: the desktop shell has its own updater and the PWA plugin is
 * `disable: isTauriBuild`, so `virtual:pwa-register` is a no-op there anyway.
 */

import { registerSW } from 'virtual:pwa-register';

/** Fired when a new SW is waiting; the toast's "Reload" action calls applyPwaUpdate(). */
export const PWA_NEED_REFRESH = 'agentx:pwa-need-refresh';
/** Fired the first time the app is fully cached and usable offline. */
export const PWA_OFFLINE_READY = 'agentx:pwa-offline-ready';

const PRELOAD_RELOAD_KEY = 'agentx:pwa-preload-reload-at';
const PRELOAD_RELOAD_COOLDOWN_MS = 10_000;

let updateSW: ((reloadPage?: boolean) => Promise<void>) | undefined;

export function registerPwa(): void {
  if (__IS_TAURI__) return;

  updateSW = registerSW({
    onNeedRefresh() {
      window.dispatchEvent(new CustomEvent(PWA_NEED_REFRESH));
    },
    onOfflineReady() {
      window.dispatchEvent(new CustomEvent(PWA_OFFLINE_READY));
    },
    onRegisterError(error) {
      console.error('[pwa] service worker registration failed', error);
    },
  });

  installPreloadErrorRecovery();
}

/** Activate the waiting service worker and reload — wired to the toast action. */
export function applyPwaUpdate(): void {
  void updateSW?.(true);
}

/**
 * After a deploy the old index.html can reference chunk hashes that no longer
 * exist; Vite raises `vite:preloadError` when such a lazy import 404s. Reload
 * once to pull the fresh manifest. The sessionStorage cooldown prevents a
 * reload loop if the deploy is actually broken.
 */
function installPreloadErrorRecovery(): void {
  window.addEventListener('vite:preloadError', (event) => {
    event.preventDefault(); // stop the unhandled rejection; we handle it via reload
    const last = Number(sessionStorage.getItem(PRELOAD_RELOAD_KEY) ?? 0);
    if (Date.now() - last < PRELOAD_RELOAD_COOLDOWN_MS) return;
    sessionStorage.setItem(PRELOAD_RELOAD_KEY, String(Date.now()));
    window.location.reload();
  });
}
