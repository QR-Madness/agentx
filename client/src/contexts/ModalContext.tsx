/**
 * Modal Context — Stack-based modal/drawer management with portal rendering
 */

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';

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
