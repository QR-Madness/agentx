/**
 * AmbassadorDockContext — desktop dock state for the Ambassador panel.
 *
 * The Ambassador runs *parallel* to a conversation, so on a wide screen it lives
 * as a docked, resizable column in the chat page (`AgentXPage`) that *shrinks* the
 * conversation instead of a modal drawer that dims/blurs it — both panels stay
 * live. Below the dock breakpoint (narrow screens / mobile) there's no room to
 * push, so callers fall back to the full-screen sheet (see `useOpenAmbassador`).
 *
 * State (open + width) is persisted; `dockCapable` tracks the breakpoint reactively.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

const OPEN_KEY = 'agentx:ambassador-dock-open';
const WIDTH_KEY = 'agentx:ambassador-dock-width';
// Push/shrink needs real room; below this a docked panel would crush the chat, so
// callers fall back to the full-screen sheet instead (DrawerPanel goes full-screen
// at ≤600px; 880px keeps the conversation comfortable when both are visible).
const DOCK_QUERY = '(min-width: 880px)';
const MIN_WIDTH = 340;
const MAX_WIDTH = 680;
const DEFAULT_WIDTH = 440;

export function clampDockWidth(w: number): number {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, w));
}

interface AmbassadorDockValue {
  open: boolean;
  width: number;
  /** True when the viewport is wide enough to dock (else fall back to a sheet). */
  dockCapable: boolean;
  setOpen: (next: boolean) => void;
  toggle: () => void;
  setWidth: (next: number) => void;
}

const AmbassadorDockContext = createContext<AmbassadorDockValue | null>(null);

export function AmbassadorDockProvider({ children }: { children: ReactNode }) {
  const [open, setOpenState] = useState(() => {
    try { return localStorage.getItem(OPEN_KEY) === '1'; } catch { return false; }
  });
  const [width, setWidthState] = useState(() => {
    try { return clampDockWidth(Number(localStorage.getItem(WIDTH_KEY)) || DEFAULT_WIDTH); }
    catch { return DEFAULT_WIDTH; }
  });
  const [dockCapable, setDockCapable] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(DOCK_QUERY).matches,
  );
  const widthRef = useRef(width);
  widthRef.current = width;

  useEffect(() => {
    const mq = window.matchMedia(DOCK_QUERY);
    const handler = (e: MediaQueryListEvent) => setDockCapable(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  const persistOpen = (next: boolean) => {
    try { localStorage.setItem(OPEN_KEY, next ? '1' : '0'); } catch { /* ignore */ }
  };

  const setOpen = useCallback((next: boolean) => {
    setOpenState(next);
    persistOpen(next);
  }, []);

  const toggle = useCallback(() => {
    setOpenState((o) => {
      const next = !o;
      persistOpen(next);
      return next;
    });
  }, []);

  const setWidth = useCallback((next: number) => {
    const w = clampDockWidth(next);
    setWidthState(w);
    try { localStorage.setItem(WIDTH_KEY, String(w)); } catch { /* ignore */ }
  }, []);

  const value = useMemo(
    () => ({ open, width, dockCapable, setOpen, toggle, setWidth }),
    [open, width, dockCapable, setOpen, toggle, setWidth],
  );

  return <AmbassadorDockContext.Provider value={value}>{children}</AmbassadorDockContext.Provider>;
}

export function useAmbassadorDock(): AmbassadorDockValue {
  const ctx = useContext(AmbassadorDockContext);
  if (!ctx) throw new Error('useAmbassadorDock must be used within an AmbassadorDockProvider');
  return ctx;
}
