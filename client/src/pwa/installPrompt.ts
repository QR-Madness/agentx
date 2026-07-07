/**
 * installPrompt — "Add to Home Screen" affordance for the web/PWA build.
 *
 * Chromium (Android / desktop) fires `beforeinstallprompt`, which we stash so a
 * toast's "Install" action can trigger the native prompt later. iOS Safari has
 * NO such event — install is a manual Share → "Add to Home Screen" — so we
 * detect iOS and surface a one-time textual hint instead.
 *
 * Both paths are suppressed when already running standalone, and the whole
 * module is inert under Tauri.
 */

/** Fired when a native install prompt is available; toast action calls promptInstall(). */
export const PWA_INSTALLABLE = 'agentx:pwa-installable';
/** Fired once on iOS Safari to show the manual "Add to Home Screen" hint. */
export const PWA_IOS_HINT = 'agentx:pwa-ios-hint';

const IOS_HINT_SEEN_KEY = 'agentx:pwa-ios-hint-seen';

/** The `beforeinstallprompt` event isn't in the DOM lib yet. */
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

let deferredPrompt: BeforeInstallPromptEvent | null = null;

function isStandalone(): boolean {
  return (
    window.matchMedia?.('(display-mode: standalone)').matches === true ||
    // iOS Safari exposes this non-standard flag when launched from the home screen.
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

function isIos(): boolean {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function initInstallPrompt(): void {
  if (__IS_TAURI__) return;
  if (isStandalone()) return;

  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault(); // suppress the mini-infobar; we drive our own toast
    deferredPrompt = event as BeforeInstallPromptEvent;
    window.dispatchEvent(new CustomEvent(PWA_INSTALLABLE));
  });

  // iOS has no beforeinstallprompt — offer the manual hint once.
  if (isIos() && localStorage.getItem(IOS_HINT_SEEN_KEY) == null) {
    localStorage.setItem(IOS_HINT_SEEN_KEY, '1');
    // Defer so it doesn't collide with the boot splash dismissal.
    window.setTimeout(() => window.dispatchEvent(new CustomEvent(PWA_IOS_HINT)), 1500);
  }
}

/** Trigger the stashed native install prompt (Chromium). */
export async function promptInstall(): Promise<void> {
  if (!deferredPrompt) return;
  const evt = deferredPrompt;
  deferredPrompt = null;
  try {
    await evt.prompt();
    await evt.userChoice;
  } catch {
    // User dismissed or the prompt is no longer valid — nothing to do.
  }
}
