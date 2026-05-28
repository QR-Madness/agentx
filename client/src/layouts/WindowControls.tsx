/**
 * WindowControls — custom minimize / maximize / close for the frameless window.
 *
 * Rendered only on frameless desktop (Windows/Linux) — see `showWindowControls`
 * in lib/platform.ts. macOS keeps its native traffic-lights; the web build uses
 * the browser chrome, so this renders nothing there.
 */

import { useEffect, useState } from 'react';
import { Minus, Square, Copy, X } from 'lucide-react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import './WindowControls.css';

export function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    const win = getCurrentWindow();
    let unlisten: (() => void) | undefined;
    win.isMaximized().then(setMaximized).catch(() => {});
    win
      .onResized(() => {
        win.isMaximized().then(setMaximized).catch(() => {});
      })
      .then(fn => { unlisten = fn; })
      .catch(() => {});
    return () => unlisten?.();
  }, []);

  const win = getCurrentWindow();

  return (
    <div className="window-controls" data-tauri-drag-region={false}>
      <button
        className="window-control"
        onClick={() => win.minimize()}
        title="Minimize"
        aria-label="Minimize"
      >
        <Minus size={15} />
      </button>
      <button
        className="window-control"
        onClick={() => win.toggleMaximize()}
        title={maximized ? 'Restore' : 'Maximize'}
        aria-label={maximized ? 'Restore' : 'Maximize'}
      >
        {maximized ? <Copy size={12} /> : <Square size={12} />}
      </button>
      <button
        className="window-control window-control--close"
        onClick={() => win.close()}
        title="Close"
        aria-label="Close"
      >
        <X size={15} />
      </button>
    </div>
  );
}
