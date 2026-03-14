/**
 * ModalDialog — Centered overlay modal
 */

import { useEffect, type ReactNode } from 'react';
import type { ModalSize } from '../../contexts/ModalContext';

interface ModalDialogProps {
  size?: ModalSize;
  onClose: () => void;
  children: ReactNode;
}

const SIZE_MAX_WIDTHS: Record<ModalSize, string> = {
  sm: '440px',
  md: '600px',
  lg: '800px',
  xl: '1000px',
  full: '95vw',
};

export function ModalDialog({ size = 'md', onClose, children }: ModalDialogProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-dialog"
        style={{ maxWidth: SIZE_MAX_WIDTHS[size] }}
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {children}
      </div>
    </div>
  );
}
