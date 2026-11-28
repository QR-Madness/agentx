/**
 * Modal Context — Stack-based modal/drawer management with portal rendering
 */

import { createContext, useContext, useEffect, useRef, useState, useCallback, ReactNode } from 'react';

export type ModalType = 'modal' | 'drawer';
export type ModalPosition = 'left' | 'right' | 'center';
export type ModalSize = 'sm' | 'md' | 'lg' | 'xl' | 'xxl' | 'full';

export interface ModalConfig {
  id?: string;
  type: ModalType;
  component: string;
  props?: Record<string, unknown>;
  position?: ModalPosition;
  size?: ModalSize;
}

export interface ModalInstance extends Required<Pick<ModalConfig, 'id' | 'type' | 'component'>> {
  props: Record<string, unknown>;
  position: ModalPosition;
  size: ModalSize;
}

/**
 * A stack entry's say in its own dismissal: `true` lets it close, `false` keeps
 * it open, and a promise asks first (a confirm). A clean guard answers `true`
 * synchronously, so closing a clean entry stays synchronous.
 */
export type DismissGuard = () => boolean | Promise<boolean>;

interface ModalContextValue {
  modals: ModalInstance[];
  openModal: (config: ModalConfig) => string;
  /**
   * Ask entry `id` to close. Its dismissal guards run first (`useDismissGuard`),
   * in the order they registered, and any one of them can keep it open.
   * Resolves `true` once the entry is gone, `false` if a guard kept it.
   * Every close path goes through here: the shells' close button, Escape and
   * backdrop, a surface's own close, `dismissTabSurfaces`.
   */
  closeModal: (id: string) => Promise<boolean>;
  /** Close everything without asking (session expiry: nothing could be saved). */
  closeAll: () => void;
  isOpen: (id: string) => boolean;
  /** Register a guard on entry `entryId`; returns the unregister function. */
  registerDismissGuard: (entryId: string, guard: DismissGuard) => () => void;
}

/** Exported for `useDismissGuard`, which reads it optionally: without a provider it registers nothing. */
export const ModalContext = createContext<ModalContextValue | null>(null);

let modalCounter = 0;

export function ModalProvider({ children }: { children: ReactNode }) {
  const [modals, setModals] = useState<ModalInstance[]>([]);

  const openModal = useCallback((config: ModalConfig): string => {
    const id = config.id || `modal_${++modalCounter}`;

    const instance: ModalInstance = {
      id,
      type: config.type,
      component: config.component,
      props: config.props || {},
      position: config.position || (config.type === 'drawer' ? 'right' : 'center'),
      size: config.size || 'md',
    };

    setModals(prev => {
      const filtered = prev.filter(m => m.id !== id);
      return [...filtered, instance];
    });

    return id;
  }, []);

  const guards = useRef(new Map<string, Set<DismissGuard>>());
  // One close request per entry at a time, so a second Escape or click while
  // the confirm is up does not ask twice.
  const pendingCloses = useRef(new Map<string, Promise<boolean>>());

  const registerDismissGuard = useCallback((entryId: string, guard: DismissGuard) => {
    let set = guards.current.get(entryId);
    if (!set) guards.current.set(entryId, (set = new Set()));
    set.add(guard);
    return () => {
      const current = guards.current.get(entryId);
      current?.delete(guard);
      if (current?.size === 0) guards.current.delete(entryId);
    };
  }, []);

  const closeModal = useCallback((id: string): Promise<boolean> => {
    const pending = pendingCloses.current.get(id);
    if (pending) return pending;

    const remove = () => setModals(prev => prev.filter(m => m.id !== id));
    const queue = [...(guards.current.get(id) ?? [])];

    // Run the guards synchronously until one has to ask. A clean entry closes
    // in this same call, so `closeModal(a); openModal(a)` keeps its order.
    while (queue.length > 0) {
      const answer = queue.shift()!();
      if (answer === false) return Promise.resolve(false);
      if (answer === true) continue;
      const asking = (async () => {
        try {
          if (!(await answer)) return false;
          for (const guard of queue) {
            if (!(await guard())) return false;
          }
          remove();
          return true;
        } finally {
          pendingCloses.current.delete(id);
        }
      })();
      pendingCloses.current.set(id, asking);
      return asking;
    }
    remove();
    return Promise.resolve(true);
  }, []);

  const closeAll = useCallback(() => {
    setModals([]);
  }, []);

  const isOpen = useCallback((id: string) => {
    return modals.some(m => m.id === id);
  }, [modals]);

  return (
    <ModalContext.Provider value={{ modals, openModal, closeModal, closeAll, isOpen, registerDismissGuard }}>
      {children}
    </ModalContext.Provider>
  );
}

export function useModal() {
  const context = useContext(ModalContext);
  if (!context) {
    throw new Error('useModal must be used within a ModalProvider');
  }
  return context;
}

/** The id of the stack entry a component renders inside. ModalPortal sets it; null elsewhere. */
export const ModalEntryContext = createContext<string | null>(null);

// Mount order of every `useEscapeToClose` caller, and the ones listening now.
// A component renders before its children and before anything opened after it,
// so the highest number listening is the innermost, newest layer.
let escapeSeq = 0;
const listeningEscapes = new Set<number>();

/**
 * Escape closes the stack entry this component renders in, but only while that
 * entry is the top of the stack and nothing above it has already taken the key.
 * The modal stack paints in the order it opened (`.modal-layer`), so the entry
 * on top is the one the user sees.
 *
 * - Only the top entry listens at all, so stacked entries never close together:
 *   a drawer over Memory, the profile editor over the Roster.
 * - Within that entry the innermost caller wins: a dialog rendered inside a
 *   drawer (the Projects drawer's document preview) closes itself, not the
 *   drawer. Order is fixed at mount, so a re-render never reshuffles it.
 * - `defaultPrevented` covers the layers above the stack: a Radix Select or
 *   confirm dismisses itself on Escape and prevents the default. The command
 *   palette also stops the event, so it never arrives here.
 * - Outside ModalPortal (no entry), or without a ModalProvider, it always applies.
 *
 * `enabled` gates it for surfaces that stay mounted while closed (`isOpen` props).
 * An entry's dismissal guards still apply: `onClose` is normally `closeModal`.
 */
export function useEscapeToClose(onClose: () => void, enabled = true): void {
  const entryId = useContext(ModalEntryContext);
  const stack = useContext(ModalContext)?.modals;
  const isTop = entryId === null || !stack || stack[stack.length - 1]?.id === entryId;
  const [seq] = useState(() => ++escapeSeq);
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!enabled || !isTop) return;
    listeningEscapes.add(seq);
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      if (seq !== Math.max(...listeningEscapes)) return;
      e.preventDefault();
      onCloseRef.current();
    };
    window.addEventListener('keydown', handler);
    return () => {
      listeningEscapes.delete(seq);
      window.removeEventListener('keydown', handler);
    };
  }, [enabled, isTop, seq]);
}
