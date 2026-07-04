/* Small shared primitives: phase badge, usage gauge, modal shell, toasts. */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Phase } from "../api";

// ---------------------------------------------------------------------------
// Phase badge

const PHASE_STYLE: Record<Phase, string> = {
  up: "bg-emerald-500/15 text-emerald-300 border-emerald-400/30",
  initializing: "bg-amber-500/15 text-amber-300 border-amber-400/30 animate-pulse",
  degraded: "bg-orange-500/15 text-orange-300 border-orange-400/30",
  down: "bg-fg-muted/10 text-fg-muted border-line",
};

export function PhaseBadge({ phase }: { phase: Phase }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${PHASE_STYLE[phase]}`}
    >
      {phase}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Usage gauge (horizontal bar + label)

export function UsageGauge({
  label,
  percent,
  detail,
}: {
  label: string;
  percent: number;
  detail?: string;
}) {
  const clamped = Math.max(0, Math.min(100, percent));
  const tone =
    clamped > 90 ? "bg-orange-400" : clamped > 70 ? "bg-amber-400" : "bg-accent-2";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-8 shrink-0 text-fg-muted">{label}</span>
      <div className="h-1.5 min-w-16 flex-1 overflow-hidden rounded-full bg-sunken">
        <div
          className={`h-full rounded-full transition-all duration-500 ${tone}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
      <span className="w-24 shrink-0 text-right tabular-nums text-fg-secondary">
        {percent.toFixed(1)}%{detail ? ` · ${detail}` : ""}
      </span>
    </div>
  );
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}

// ---------------------------------------------------------------------------
// Modal shell

export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-md rounded-xl border border-line bg-overlay p-5 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-fg">{title}</h2>
          <button
            onClick={onClose}
            className="rounded-md px-2 py-1 text-fg-muted hover:bg-hover hover:text-fg"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toasts

export interface Toast {
  id: number;
  kind: "ok" | "error";
  text: string;
}

interface ToastApi {
  toast: (kind: Toast["kind"], text: string) => void;
}

const ToastContext = createContext<ToastApi>({ toast: () => {} });

export function useToast(): ToastApi {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback((kind: Toast["kind"], text: string) => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, kind, text }]);
    setTimeout(
      () => setToasts((current) => current.filter((t) => t.id !== id)),
      kind === "error" ? 8000 : 4000,
    );
  }, []);

  const value = useMemo(() => ({ toast }), [toast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-96 max-w-[90vw] flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto rounded-lg border px-3 py-2 text-sm shadow-lg ${
              t.kind === "error"
                ? "border-red-400/40 bg-red-950/90 text-red-200"
                : "border-line-strong bg-raised text-fg-secondary"
            }`}
          >
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
