/**
 * NotificationContext — app-wide transient toast notifications.
 *
 * The client previously had no consistent way to surface failures: each modal
 * kept its own inline error string and many `.catch {}` swallows dropped errors
 * silently. This provider holds a small auto-dismissing toast queue so any
 * component (or the API error path) can raise visible feedback via `useNotify`.
 *
 * `notifyError` runs the caught value through `toApiError` so it accepts an
 * `ApiError`, a network `TypeError`, or a bare SSE error string interchangeably,
 * and picks a sensible title from the error's `kind`.
 */

import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react';
import { toApiError, type ApiErrorKind } from '../lib/api';

export type ToastKind = 'error' | 'warning' | 'success' | 'info';

export interface Toast {
  id: string;
  kind: ToastKind;
  title?: string;
  message: string;
  /** Auto-dismiss delay in ms. `0` keeps the toast until manually dismissed. */
  duration: number;
}

export interface NotifyOptions {
  kind?: ToastKind;
  title?: string;
  message: string;
  duration?: number;
}

interface NotificationContextValue {
  toasts: Toast[];
  notify: (opts: NotifyOptions) => string;
  notifyError: (err: unknown, title?: string) => string;
  notifySuccess: (message: string, title?: string) => string;
  dismiss: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

/** Max simultaneously-visible toasts; oldest is dropped past this. */
const MAX_TOASTS = 4;
const DEFAULT_DURATION = 6000;
const ERROR_DURATION = 8000;

/** Human-friendly toast title for a normalized API error category. */
function titleForKind(kind: ApiErrorKind): string {
  switch (kind) {
    case 'network': return 'Connection problem';
    case 'auth': return 'Session expired';
    case 'not_found': return 'Not found';
    case 'bad_request': return 'Invalid request';
    case 'upstream': return 'Provider unavailable';
    case 'unavailable': return 'Service unavailable';
    default: return 'Something went wrong';
  }
}

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Per-toast dismiss timers so we can clear them on manual dismiss.
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const notify = useCallback((opts: NotifyOptions): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const toast: Toast = {
      id,
      kind: opts.kind ?? 'info',
      title: opts.title,
      message: opts.message,
      duration: opts.duration ?? (opts.kind === 'error' ? ERROR_DURATION : DEFAULT_DURATION),
    };

    setToasts(prev => {
      const next = [...prev, toast];
      return next.length > MAX_TOASTS ? next.slice(next.length - MAX_TOASTS) : next;
    });

    if (toast.duration > 0) {
      timers.current.set(id, setTimeout(() => dismiss(id), toast.duration));
    }
    return id;
  }, [dismiss]);

  const notifyError = useCallback((err: unknown, title?: string): string => {
    const apiErr = toApiError(err);
    return notify({
      kind: 'error',
      title: title ?? titleForKind(apiErr.kind),
      message: apiErr.message,
    });
  }, [notify]);

  const notifySuccess = useCallback((message: string, title?: string): string => {
    return notify({ kind: 'success', title, message });
  }, [notify]);

  return (
    <NotificationContext.Provider value={{ toasts, notify, notifyError, notifySuccess, dismiss }}>
      {children}
    </NotificationContext.Provider>
  );
}

/** Access the notification API. Must be used within {@link NotificationProvider}. */
export function useNotify(): NotificationContextValue {
  const ctx = useContext(NotificationContext);
  if (!ctx) {
    throw new Error('useNotify must be used within a NotificationProvider');
  }
  return ctx;
}
