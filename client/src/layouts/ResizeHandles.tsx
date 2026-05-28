/**
 * ResizeHandles — invisible edge/corner grips for the frameless window.
 *
 * With `decorations: false` the OS no longer draws resize borders, so a drag on
 * the window edge falls through to the titlebar's drag region and just *moves*
 * the window. These thin fixed-position grips sit above everything and call
 * Tauri's `startResizeDragging` so edges/corners resize again.
 *
 * Rendered only on frameless desktop (Windows/Linux) — see `showWindowControls`.
 */

import { getCurrentWindow } from '@tauri-apps/api/window';
import './ResizeHandles.css';

type Dir =
  | 'North' | 'South' | 'East' | 'West'
  | 'NorthEast' | 'NorthWest' | 'SouthEast' | 'SouthWest';

const HANDLES: { dir: Dir; cls: string }[] = [
  { dir: 'North', cls: 'rh-n' },
  { dir: 'South', cls: 'rh-s' },
  { dir: 'East', cls: 'rh-e' },
  { dir: 'West', cls: 'rh-w' },
  { dir: 'NorthEast', cls: 'rh-ne' },
  { dir: 'NorthWest', cls: 'rh-nw' },
  { dir: 'SouthEast', cls: 'rh-se' },
  { dir: 'SouthWest', cls: 'rh-sw' },
];

export function ResizeHandles() {
  const onDown = (dir: Dir) => (e: React.PointerEvent) => {
    // Only the primary button starts a resize.
    if (e.button !== 0) return;
    e.preventDefault();
    getCurrentWindow().startResizeDragging(dir).catch(() => {});
  };

  return (
    <div className="resize-handles" aria-hidden>
      {HANDLES.map(h => (
        <div
          key={h.dir}
          className={`resize-handle ${h.cls}`}
          onPointerDown={onDown(h.dir)}
        />
      ))}
    </div>
  );
}
