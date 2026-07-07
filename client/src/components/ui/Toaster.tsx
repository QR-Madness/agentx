/**
 * Toaster — renders the {@link NotificationProvider} queue as a stacked,
 * animated toast list portaled into `#toast-root`. Mount once near the app root.
 */

import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react';
import { useNotify, type ToastKind } from '../../contexts/NotificationContext';
import './Toaster.css';

const ICONS: Record<ToastKind, typeof Info> = {
  error: XCircle,
  warning: AlertTriangle,
  success: CheckCircle2,
  info: Info,
};

export function Toaster() {
  const { toasts, dismiss } = useNotify();
  const portalRoot = typeof document !== 'undefined' ? document.getElementById('toast-root') : null;
  if (!portalRoot) return null;

  return createPortal(
    <div className="toaster" role="region" aria-label="Notifications">
      <AnimatePresence initial={false}>
        {toasts.map(toast => {
          const Icon = ICONS[toast.kind];
          return (
            <motion.div
              key={toast.id}
              className={`toast toast--${toast.kind}`}
              role={toast.kind === 'error' ? 'alert' : 'status'}
              initial={{ opacity: 0, y: 12, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, x: 24, scale: 0.96 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              layout
            >
              <span className="toast__icon"><Icon size={18} /></span>
              <div className="toast__body">
                {toast.title && <div className="toast__title">{toast.title}</div>}
                <div className="toast__message">{toast.message}</div>
                {toast.action && (
                  <button
                    type="button"
                    className="toast__action"
                    onClick={() => { toast.action?.onClick(); dismiss(toast.id); }}
                  >
                    {toast.action.label}
                  </button>
                )}
              </div>
              <button
                type="button"
                className="toast__dismiss"
                aria-label="Dismiss notification"
                onClick={() => dismiss(toast.id)}
              >
                <X size={14} />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>,
    portalRoot
  );
}
