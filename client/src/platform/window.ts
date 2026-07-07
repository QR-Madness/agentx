/**
 * window — custom-chrome window controls for the frameless desktop shell.
 *
 * Resolver + interface. Real on desktop (Tauri), no-ops on web where the browser
 * provides its own frame. App code imports only this file; the impl is picked by
 * the compile-time `__IS_TAURI__` gate over a dynamic `import()`, so
 * `@tauri-apps/api/window` never enters the web bundle.
 *
 * Note: the callers (WindowControls / ResizeHandles) already only mount when
 * `showWindowControls` (runtime Tauri + non-mac, see lib/platform.ts) is true, so
 * the web impl here is purely defensive — its real job is to keep the Tauri
 * import out of those component modules so the boundary holds.
 */

export type ResizeDir =
  | 'North' | 'South' | 'East' | 'West'
  | 'NorthEast' | 'NorthWest' | 'SouthEast' | 'SouthWest';

export interface WindowBackend {
  minimize(): Promise<void>;
  toggleMaximize(): Promise<void>;
  close(): Promise<void>;
  isMaximized(): Promise<boolean>;
  /** Subscribe to resize; resolves to an unsubscribe fn. */
  onResized(cb: () => void): Promise<() => void>;
  startResizeDragging(dir: ResizeDir): Promise<void>;
}

let cached: Promise<WindowBackend> | null = null;

function resolve(): Promise<WindowBackend> {
  if (!cached) {
    cached = __IS_TAURI__
      ? import('./window.tauri').then((m) => m.createTauriWindow())
      : import('./window.web').then((m) => m.createWebWindow());
  }
  return cached;
}

export const windowControls = {
  /** Compile-time: are native window controls available in this shell? */
  isSupported: (): boolean => __IS_TAURI__,
  minimize: (): Promise<void> => resolve().then((w) => w.minimize()),
  toggleMaximize: (): Promise<void> => resolve().then((w) => w.toggleMaximize()),
  close: (): Promise<void> => resolve().then((w) => w.close()),
  isMaximized: (): Promise<boolean> => resolve().then((w) => w.isMaximized()),
  onResized: (cb: () => void): Promise<() => void> => resolve().then((w) => w.onResized(cb)),
  startResizeDragging: (dir: ResizeDir): Promise<void> =>
    resolve().then((w) => w.startResizeDragging(dir)),
};
