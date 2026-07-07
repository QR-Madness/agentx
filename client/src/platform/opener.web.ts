/**
 * opener.web — browser/PWA impl. A normal tab can just open a new one.
 */

import type { Opener } from './opener';

export function createWebOpener(): Opener {
  return {
    async openExternalUrl(url: string): Promise<boolean> {
      try {
        const win = window.open(url, '_blank', 'noopener');
        return win != null;
      } catch {
        return false;
      }
    },
  };
}
