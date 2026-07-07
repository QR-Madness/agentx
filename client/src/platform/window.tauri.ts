/**
 * window.tauri — desktop impl over `@tauri-apps/api/window`. Only loaded when
 * `__IS_TAURI__` is true, so this is a legal place to import Tauri
 * (enforced by importBoundary.test.ts).
 */

import { getCurrentWindow } from '@tauri-apps/api/window';
import type { WindowBackend, ResizeDir } from './window';

export function createTauriWindow(): WindowBackend {
  return {
    minimize: () => getCurrentWindow().minimize(),
    toggleMaximize: () => getCurrentWindow().toggleMaximize(),
    close: () => getCurrentWindow().close(),
    isMaximized: () => getCurrentWindow().isMaximized(),
    async onResized(cb: () => void): Promise<() => void> {
      return getCurrentWindow().onResized(() => cb());
    },
    startResizeDragging: (dir: ResizeDir) => getCurrentWindow().startResizeDragging(dir),
  };
}
