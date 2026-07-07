/**
 * window.web — browser/PWA impl. The browser owns the frame, so these are
 * no-ops; `onResized` returns a no-op unsubscribe.
 */

import type { WindowBackend, ResizeDir } from './window';

export function createWebWindow(): WindowBackend {
  return {
    async minimize() {},
    async toggleMaximize() {},
    async close() {},
    async isMaximized() {
      return false;
    },
    async onResized(_cb: () => void): Promise<() => void> {
      return () => {};
    },
    async startResizeDragging(_dir: ResizeDir) {},
  };
}
