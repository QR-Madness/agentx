/**
 * opener — open a URL in the user's real browser, everywhere.
 *
 * Resolver + interface. App code imports only this file; the desktop/web impls
 * are picked by the compile-time `__IS_TAURI__` gate over a dynamic `import()`,
 * so the dead shell's module (and its `@tauri-apps` deps) never enter the other
 * bundle. See runtime.ts for why the gate is a literal.
 */

export interface Opener {
  /**
   * Open `url` externally. Returns `true` if the open was dispatched, `false`
   * on failure (blocked popup, plugin error) — callers surface a manual
   * "open this link" affordance on `false`.
   */
  openExternalUrl(url: string): Promise<boolean>;
}

let cached: Promise<Opener> | null = null;

function resolve(): Promise<Opener> {
  if (!cached) {
    cached = __IS_TAURI__
      ? import('./opener.tauri').then((m) => m.createTauriOpener())
      : import('./opener.web').then((m) => m.createWebOpener());
  }
  return cached;
}

export const opener: Opener = {
  async openExternalUrl(url: string): Promise<boolean> {
    return (await resolve()).openExternalUrl(url);
  },
};
