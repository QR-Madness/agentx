/**
 * AmbassadorDock — the desktop, non-modal home of the Ambassador panel.
 *
 * A resizable column docked to the right of the conversation (in `AgentXPage`)
 * that *shrinks* the chat instead of overlaying it, so you can watch the agent
 * work and talk to the ambassador at once — the parallel-relay model, finally
 * visible. Renders nothing when closed or when the viewport is too narrow to dock
 * (callers fall back to the full-screen sheet via `useOpenAmbassador`).
 */

import { useCallback, useRef } from 'react';
import { X } from 'lucide-react';
import { useAmbassadorDock } from '../../contexts/AmbassadorDockContext';
import { AmbassadorPanel } from './AmbassadorPanel';
import './AmbassadorDock.css';

export function AmbassadorDock() {
  const { open, width, dockCapable, setOpen, setWidth } = useAmbassadorDock();
  const setWidthRef = useRef(setWidth);
  setWidthRef.current = setWidth;

  const startResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    // Right-docked: width grows as the pointer moves left of the right edge.
    const onMove = (ev: MouseEvent) => setWidthRef.current(window.innerWidth - ev.clientX);
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  if (!dockCapable || !open) return null;

  return (
    <aside className="ambassador-dock" style={{ width }} aria-label="Ambassador">
      <div
        className="ambassador-dock-resizer"
        onMouseDown={startResize}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize ambassador panel"
      />
      {/* Dock chrome owns the close affordance the modal shell normally provides;
          the panel header reserves space for it (pr-12), same as in the drawer. */}
      <button
        type="button"
        className="shell-close-btn"
        data-position="right"
        onClick={() => setOpen(false)}
        aria-label="Close ambassador"
        title="Close"
      >
        <X size={20} />
      </button>
      <div className="ambassador-dock-body">
        <AmbassadorPanel />
      </div>
    </aside>
  );
}
