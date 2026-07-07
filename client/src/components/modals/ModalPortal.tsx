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
  unifiedSettings: lazy(() => import('./stubs').then(m => ({ default: m.UnifiedSettingsModalContent }))),
  memory: lazy(() => import('./stubs').then(m => ({ default: m.MemoryModalContent }))),
  plans: lazy(() => import('./stubs').then(m => ({ default: m.PlansModalContent }))),
  sources: lazy(() => import('./stubs').then(m => ({ default: m.SourcesDrawerContent }))),
  ambassador: lazy(() => import('./stubs').then(m => ({ default: m.AmbassadorDrawerContent }))),
  ambassadorDeck: lazy(() => import('./stubs').then(m => ({ default: m.AmbassadorDeckContent }))),
  tools: lazy(() => import('./stubs').then(m => ({ default: m.ToolsModalContent }))),
  translation: lazy(() => import('./stubs').then(m => ({ default: m.TranslationModalContent }))),
  profileEditor: lazy(() => import('./stubs').then(m => ({ default: m.ProfileEditorModal }))),
  unifiedProfileEditor: lazy(() => import('./stubs').then(m => ({ default: m.UnifiedProfileEditorModalContent }))),
  promptLibrary: lazy(() => import('./PromptLibraryModal').then(m => ({ default: m.PromptLibraryModal }))),
  toolOutput: lazy(() => import('./stubs').then(m => ({ default: m.ToolOutputDrawer }))),
  alloyFactory: lazy(() => import('./AlloyFactoryModal').then(m => ({ default: m.AlloyFactoryModal as ComponentType<ModalComponentProps> }))),
  alloyRunTrace: lazy(() => import('../alloy/AlloyRunTraceModal').then(m => ({ default: m.AlloyRunTraceModal as ComponentType<ModalComponentProps> }))),
  changePassword: lazy(() => import('./ChangePasswordModal').then(m => ({ default: m.ChangePasswordModal }))),
  conversations: lazy(() => import('./stubs').then(m => ({ default: m.ConversationsDrawerContent }))),
  logs: lazy(() => import('./stubs').then(m => ({ default: m.LogsDrawerContent }))),
  toolOutputBrowser: lazy(() => import('./stubs').then(m => ({ default: m.ToolOutputBrowserContent }))),
  workspaces: lazy(() => import('./stubs').then(m => ({ default: m.WorkspacesDrawerContent }))),
};

/**
 * Components that render their own header close button — they opt OUT of the
 * shell-owned close button so we don't double up. Anything NOT listed here gets
 * the shell close button by default, so a new panel can never ship un-closable.
 */
const SELF_CLOSING = new Set<string>([
  'unifiedSettings',
  'tools',
  'unifiedProfileEditor',
  'profileEditor',
  'changePassword',
  'toolOutput',
  'alloyFactory',
  'alloyRunTrace',
  'workspaces',
  // NOTE: 'promptLibrary' intentionally omitted — its standalone modal has no
  // header close button of its own, so it relies on the shell close button.
]);

/**
 * Full-screen immersive surfaces that render their OWN backdrop + fixed
 * `inset: 0` container + close/Escape affordances (the UnifiedSettings
 * pattern). They render BARE — never nested inside ModalDialog:
 *
 * The dialog wrapper has no in-flow content when its child is position:fixed,
 * so `.modal-dialog` collapses to an empty ~2px strip. On Chromium (Windows
 * WebView2) the dialog's `transform` entry animation and the backdrop's
 * `backdrop-filter` make those ancestors the CONTAINING BLOCK for fixed
 * descendants — the surface stays trapped in the collapsed strip and shows as
 * a thin "line". WebKit (Linux/macOS webviews) lets fixed elements escape
 * filtered ancestors, which is why the nesting only ever *looked* fine there.
 * Keyed by component so every openModal call site is covered regardless of
 * the `type` it passes.
 */
const FULLSCREEN_SURFACES = new Set<string>([
  'unifiedSettings',
  'tools',
  'unifiedProfileEditor',
]);

function ModalRenderer({ modal }: { modal: ModalInstance }) {
  const { closeModal } = useModal();
  const Component = MODAL_REGISTRY[modal.component];

  if (!Component) {
    console.warn(`Unknown modal component: ${modal.component}`);
    return null;
  }

  const onClose = () => closeModal(modal.id);
  const showClose = !SELF_CLOSING.has(modal.component);

  const content = (
    <Suspense
      fallback={
        <div style={{ padding: 24, color: 'var(--text-secondary)' }}>Loading...</div>
      }
    >
      <Component {...modal.props} onClose={onClose} />
    </Suspense>
  );

  // Self-contained full-screen surfaces mount bare — no dialog/drawer shell.
  if (FULLSCREEN_SURFACES.has(modal.component)) {
    return content;
  }

  if (modal.type === 'drawer') {
    return (
      <DrawerPanel
        position={modal.position as 'left' | 'right'}
        size={modal.size}
        showClose={showClose}
        onClose={onClose}
      >
        {content}
      </DrawerPanel>
    );
  }

  return (
    <ModalDialog size={modal.size} showClose={showClose} onClose={onClose}>
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
