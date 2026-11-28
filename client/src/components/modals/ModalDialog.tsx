/**
 * ModalDialog — Centered overlay modal
 */

import { type ReactNode } from 'react';
import { X } from 'lucide-react';
import { useEscapeToClose, type ModalSize } from '../../contexts/ModalContext';

interface ModalDialogProps {
  size?: ModalSize;
  /** Render the shell-owned close button. Default true; pass false for content
   *  that renders its own close affordance (see SELF_CLOSING in ModalPortal). */
  showClose?: boolean;
  onClose: () => void;
  children: ReactNode;
}

const SIZE_MAX_WIDTHS: Record<ModalSize, string> = {
  sm: '440px',
  md: '600px',
  lg: '800px',
  xl: '1000px',
  xxl: '1200px',
  full: '95vw',
};

export function ModalDialog({ size = 'md', showClose = true, onClose, children }: ModalDialogProps) {
  useEscapeToClose(onClose);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-dialog"
        style={{ maxWidth: SIZE_MAX_WIDTHS[size] }}
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {showClose && (
          <button
            type="button"
            className="shell-close-btn"
            onClick={onClose}
            aria-label="Close"
            title="Close"
          >
            <X size={20} />
          </button>
        )}
        <div className="modal-dialog-body">{children}</div>
      </div>
    </div>
  );
}
