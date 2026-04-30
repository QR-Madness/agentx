/**
 * ModalPortal — Renders the modal stack via React portal into #modal-root
 */

import { lazy, Suspense, useEffect, type ComponentType } from 'react';
import { createPortal } from 'react-dom';
import { useModal, type ModalInstance } from '../../contexts/ModalContext';
import { useAuth } from '../../contexts/AuthContext';
import { DrawerPanel } from './DrawerPanel';
import { ModalDialog } from './ModalDialog';

type ModalComponentProps = Record<string, unknown> & { onClose: () => void };

/**
 * Lazy-loaded component registry.
 * Keys added here become valid `component` values for openModal().
 */
const MODAL_REGISTRY: Record<string, React.LazyExoticComponent<ComponentType<ModalComponentProps>>> = {
  settings: lazy(() => import('./stubs').then(m => ({ default: m.SettingsModalContent }))),
  unifiedSettings: lazy(() => import('./stubs').then(m => ({ default: m.UnifiedSettingsModalContent }))),
  memory: lazy(() => import('./stubs').then(m => ({ default: m.MemoryModalContent }))),
  tools: lazy(() => import('./stubs').then(m => ({ default: m.ToolsModalContent }))),
  translation: lazy(() => import('./stubs').then(m => ({ default: m.TranslationModalContent }))),
  profileEditor: lazy(() => import('./stubs').then(m => ({ default: m.ProfileEditorModal }))),
  unifiedProfileEditor: lazy(() => import('./stubs').then(m => ({ default: m.UnifiedProfileEditorModalContent }))),
  promptLibrary: lazy(() => import('./PromptLibraryModal').then(m => ({ default: m.PromptLibraryModal }))),
  toolOutput: lazy(() => import('./stubs').then(m => ({ default: m.ToolOutputDrawer }))),
  alloyFactory: lazy(() => import('./AlloyFactoryModal').then(m => ({ default: m.AlloyFactoryModal as ComponentType<ModalComponentProps> }))),
  changePassword: lazy(() => import('./ChangePasswordModal').then(m => ({ default: m.ChangePasswordModal }))),
};

function ModalRenderer({ modal }: { modal: ModalInstance }) {
  const { closeModal } = useModal();
  const Component = MODAL_REGISTRY[modal.component];

  if (!Component) {
    console.warn(`Unknown modal component: ${modal.component}`);
    return null;
  }

  const onClose = () => closeModal(modal.id);

  const content = (
    <Suspense
      fallback={
        <div style={{ padding: 24, color: 'var(--text-secondary)' }}>Loading...</div>
      }
    >
      <Component {...modal.props} onClose={onClose} />
    </Suspense>
  );

  if (modal.type === 'drawer') {
    return (
      <DrawerPanel
        position={modal.position as 'left' | 'right'}
        size={modal.size}
        onClose={onClose}
      >
        {content}
      </DrawerPanel>
    );
  }

  return (
    <ModalDialog size={modal.size} onClose={onClose}>
      {content}
    </ModalDialog>
  );
}

export function ModalPortal() {
  const { modals, closeAll } = useModal();
  const { authRequired, isAuthenticated } = useAuth();
  const portalRoot = document.getElementById('modal-root');

  // Clear all open modals when session expires so hooks inside them
  // don't keep firing unauthenticated API requests.
  useEffect(() => {
    if (authRequired && !isAuthenticated) {
      closeAll();
    }
  }, [authRequired, isAuthenticated, closeAll]);

  if (!portalRoot || modals.length === 0) return null;
  if (authRequired && !isAuthenticated) return null;

  return createPortal(
    <>
      {modals.map(modal => (
        <ModalRenderer key={modal.id} modal={modal} />
      ))}
    </>,
    portalRoot
  );
}
