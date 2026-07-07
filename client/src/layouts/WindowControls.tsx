/**
 * WindowControls — custom minimize / maximize / close for the frameless window.
 *
 * Rendered only on frameless desktop (Windows/Linux) — see `showWindowControls`
 * in lib/platform.ts. macOS keeps its native traffic-lights; the web build uses
 * the browser chrome, so this renders nothing there.
 */

import { useEffect, useState } from 'react';
import { Minus, Square, Copy, X } from 'lucide-react';
import { platform } from '../platform';
import './WindowControls.css';

export function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    const sync = () => platform.window.isMaximized().then(setMaximized).catch(() => {});
    sync();
    platform.window
      .onResized(sync)
      .then(fn => { if (cancelled) fn(); else unlisten = fn; })
      .catch(() => {});
    return () => { cancelled = true; unlisten?.(); };
  }, []);

  return (
    <div className="window-controls" data-tauri-drag-region={false}>
      <button
        className="window-control"
        onClick={() => platform.window.minimize()}
        title="Minimize"
        aria-label="Minimize"
      >
        <Minus size={15} />
      </button>
      <button
        className="window-control"
        onClick={() => platform.window.toggleMaximize()}
        title={maximized ? 'Restore' : 'Maximize'}
        aria-label={maximized ? 'Restore' : 'Maximize'}
      >
        {maximized ? <Copy size={12} /> : <Square size={12} />}
      </button>
      <button
        className="window-control window-control--close"
        onClick={() => platform.window.close()}
        title="Close"
        aria-label="Close"
      >
        <X size={15} />
      </button>
    </div>
  );
}
