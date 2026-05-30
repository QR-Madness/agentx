/**
 * DrawerPanel — Slide-out panel from left or right edge
 */

import { useEffect, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';
import type { ModalSize } from '../../contexts/ModalContext';
import './DrawerPanel.css';

interface DrawerPanelProps {
  position?: 'left' | 'right';
  size?: ModalSize;
  /** Render the shell-owned close button. Default true; pass false for content
   *  that renders its own close affordance (see SELF_CLOSING in ModalPortal). */
  showClose?: boolean;
  onClose: () => void;
  children: ReactNode;
}

export function DrawerPanel({ position = 'right', size = 'md', showClose = true, onClose, children }: DrawerPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const isLeft = position === 'left';

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div
        ref={panelRef}
        className={`drawer-panel drawer-${position} drawer-size-${size}`}
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {showClose && (
          <button
            type="button"
            className="shell-close-btn"
            data-position={position}
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            <X size={20} />
          </button>
        )}
        <div className="drawer-content" style={{ [isLeft ? 'left' : 'right']: 0 }}>
          {children}
        </div>
      </div>
    </div>
  );
}
