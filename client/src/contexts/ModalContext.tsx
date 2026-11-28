/**
 * Modal Context — Stack-based modal/drawer management with portal rendering
 */

import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';

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

interface ModalContextValue {
  modals: ModalInstance[];
  openModal: (config: ModalConfig) => string;
  closeModal: (id: string) => void;
  closeAll: () => void;
  isOpen: (id: string) => boolean;
}

const ModalContext = createContext<ModalContextValue | null>(null);

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

  const closeModal = useCallback((id: string) => {
    setModals(prev => prev.filter(m => m.id !== id));
  }, []);

  const closeAll = useCallback(() => {
    setModals([]);
  }, []);

  const isOpen = useCallback((id: string) => {
    return modals.some(m => m.id === id);
  }, [modals]);

  return (
    <ModalContext.Provider value={{ modals, openModal, closeModal, closeAll, isOpen }}>
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

/**
 * Escape closes the stack entry this component renders in, but only while that
 * entry is the top of the stack and nothing above it has already taken the key.
 * The modal stack paints in the order it opened (`.modal-layer`), so the entry
 * on top is the one the user sees.
 *
 * - Only the top entry listens at all, so stacked entries never close together:
 *   a drawer over Memory, the profile editor over the Roster.
 * - `defaultPrevented` covers the layers above the stack: a Radix Select or
 *   confirm dismisses itself on Escape and prevents the default. The command
 *   palette also stops the event, so it never arrives here.
 * - Outside ModalPortal (no entry), or without a ModalProvider, it always applies.
 *
 * `enabled` gates it for surfaces that stay mounted while closed (`isOpen` props).
 */
export function useEscapeToClose(onClose: () => void, enabled = true): void {
  const entryId = useContext(ModalEntryContext);
  const stack = useContext(ModalContext)?.modals;
  const isTop = entryId === null || !stack || stack[stack.length - 1]?.id === entryId;

  useEffect(() => {
    if (!enabled || !isTop) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      e.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [enabled, isTop, onClose]);
}
