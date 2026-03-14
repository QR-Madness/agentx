/**
 * DrawerPanel — Slide-out panel from left or right edge
 */

import { useEffect, useRef, type ReactNode } from 'react';
import type { ModalSize } from '../../contexts/ModalContext';

interface DrawerPanelProps {
  position?: 'left' | 'right';
  size?: ModalSize;
  onClose: () => void;
  children: ReactNode;
}

const SIZE_WIDTHS: Record<ModalSize, string> = {
  sm: '360px',
  md: '480px',
  lg: '640px',
  xl: '800px',
  xxl: '1000px',
  full: '100%',
};

export function DrawerPanel({ position = 'right', size = 'md', onClose, children }: DrawerPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const width = SIZE_WIDTHS[size];
  const isLeft = position === 'left';

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div
        ref={panelRef}
        className={`drawer-panel drawer-${position}`}
        style={{ width, maxWidth: '90vw' }}
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="drawer-content" style={{ [isLeft ? 'left' : 'right']: 0 }}>
          {children}
        </div>
      </div>
    </div>
  );
}
