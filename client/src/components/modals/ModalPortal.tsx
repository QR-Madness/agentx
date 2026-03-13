/**
 * ModalPortal — Renders the modal stack via React portal into #modal-root
 */

import { lazy, Suspense, type ComponentType } from 'react';
import { createPortal } from 'react-dom';
import { useModal, type ModalInstance } from '../../contexts/ModalContext';
import { DrawerPanel } from './DrawerPanel';
import { ModalDialog } from './ModalDialog';

type ModalComponentProps = Record<string, unknown> & { onClose: () => void };

/**
 * Lazy-loaded component registry.
 * Keys added here become valid `component` values for openModal().
 */
const MODAL_REGISTRY: Record<string, React.LazyExoticComponent<ComponentType<ModalComponentProps>>> = {
  settings: lazy(() => import('./stubs').then(m => ({ default: m.SettingsModalContent }))),
  memory: lazy(() => import('./stubs').then(m => ({ default: m.MemoryModalContent }))),
  tools: lazy(() => import('./stubs').then(m => ({ default: m.ToolsModalContent }))),
  translation: lazy(() => import('./stubs').then(m => ({ default: m.TranslationModalContent }))),
  profileEditor: lazy(() => import('./stubs').then(m => ({ default: m.ProfileEditorModal }))),
  promptLibrary: lazy(() => import('./stubs').then(m => ({ default: m.StubModal }))),
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
  const { modals } = useModal();
  const portalRoot = document.getElementById('modal-root');

  if (!portalRoot || modals.length === 0) return null;

  return createPortal(
    <>
      {modals.map(modal => (
        <ModalRenderer key={modal.id} modal={modal} />
      ))}
    </>,
    portalRoot
  );
}
