/**
 * DocumentPreviewModal — an edit in progress is not closed away without asking.
 *
 * The preview is a ModalDialog rendered inside the Projects drawer's stack entry.
 * Before v0.21.311 its markdown editor had no dirty check: its own close, Escape,
 * and anything that dismissed the drawer (⌘T, the palette's navigation, the
 * TopBar's page pills) dropped the draft. And Escape in the preview closed the
 * whole drawer, because the drawer's Escape listener ran first.
 *
 * The stack, the drawer shell and the confirm are the real ones; the drawer's
 * content is a small host that opens one document, as WorkspacesPanel does.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import type { WorkspaceDocument } from '../../lib/api';
import { ModalProvider, useModal } from '../../contexts/ModalContext';
import { ConfirmProvider } from '../ui/ConfirmDialog';
import { SURFACES, dismissTabSurfaces } from '../../lib/surfaces';
import { ModalPortal } from '../modals/ModalPortal';

const api = vi.hoisted(() => ({
  fetchDocumentText: vi.fn(),
  fetchDocumentBlob: vi.fn(),
  updateTextDocument: vi.fn(),
  renameDocument: vi.fn(),
}));
vi.mock('../../lib/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  api,
}));
vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ authRequired: false, isAuthenticated: true }),
}));
vi.mock('../../contexts/NotificationContext', () => {
  const value = { notify: vi.fn(), notifyError: vi.fn(), notifySuccess: vi.fn(), dismiss: vi.fn() };
  return { useNotify: () => value };
});

const notes: WorkspaceDocument = {
  id: 'd1', workspace_id: 'w1', filename: 'notes.md', content_type: 'text/markdown',
  size_bytes: 24, sha256: 'abc', tags: [], summary: '', status: 'ready', error: null,
  created_at: null, updated_at: null,
};

vi.mock('../modals/stubs', async () => {
  const { DocumentPreviewModal } = await import('./DocumentPreviewModal');
  return {
    WorkspacesDrawerContent: function FakeProjects() {
      // Opened from the drawer after it mounted, as in the app: the drawer's
      // Escape listener is then the older one.
      const [previewing, setPreviewing] = useState(false);
      return (
        <div data-testid="projects">
          <button type="button" onClick={() => setPreviewing(true)}>Open notes.md</button>
          {previewing && (
            <DocumentPreviewModal workspaceId="w1" document={notes} onClose={() => setPreviewing(false)} />
          )}
        </div>
      );
    },
  };
});

let modal: ReturnType<typeof useModal>;
function CaptureModalApi() {
  modal = useModal();
  return null;
}

const drawerOpen = () => !!document.querySelector('#modal-root > .modal-layer[data-modal="workspaces"]');
const previewOpen = () => !!screen.queryByText('notes.md');
const prompt = () => screen.queryByRole('dialog', { name: 'Discard unsaved changes?' });

const escape = () =>
  act(() => {
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  });

async function answer(label: 'Discard' | 'Keep editing') {
  fireEvent.click(await screen.findByRole('button', { name: label }));
  await waitFor(() => expect(prompt()).toBeNull());
}

/** Open Projects with notes.md previewed, then enter the editor. */
async function startEditing(draft?: string) {
  act(() => { modal.openModal(SURFACES.workspaces); });
  fireEvent.click(await screen.findByRole('button', { name: 'Open notes.md' }));
  fireEvent.click(await screen.findByRole('button', { name: /Edit/ }));
  const editor = await screen.findByPlaceholderText('Write markdown…');
  if (draft !== undefined) fireEvent.change(editor, { target: { value: draft } });
  return editor;
}

describe('DocumentPreviewModal unsaved edits', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.fetchDocumentText.mockResolvedValue('# Field notes\n');
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: false, media: query, onchange: null,
      addEventListener: vi.fn(), removeEventListener: vi.fn(),
      addListener: vi.fn(), removeListener: vi.fn(), dispatchEvent: vi.fn(),
    }));
    const root = document.createElement('div');
    root.id = 'modal-root';
    document.body.appendChild(root);
    render(
      <ModalProvider>
        <ConfirmProvider>
          <CaptureModalApi />
          <ModalPortal />
        </ConfirmProvider>
      </ModalProvider>,
    );
  });

  afterEach(() => {
    document.getElementById('modal-root')?.remove();
  });

  it('keeps a dirty draft when the drawer is dismissed by navigation, until discarded', async () => {
    const editor = await startEditing('# Field notes\n\nA second paragraph.');

    let left!: Promise<boolean>;
    act(() => { left = dismissTabSurfaces(modal); });
    expect(await screen.findByText('Your edits to notes.md have not been saved.')).toBeInTheDocument();
    await answer('Keep editing');
    expect(await left).toBe(false);
    expect(drawerOpen()).toBe(true);
    expect(editor).toHaveValue('# Field notes\n\nA second paragraph.');

    act(() => { left = dismissTabSurfaces(modal); });
    await answer('Discard');
    expect(await left).toBe(true);
    expect(drawerOpen()).toBe(false);
  });

  it('asks on Escape, and Escape closes the preview only, not the drawer', async () => {
    await startEditing('# Field notes, revised');
    escape();
    await answer('Keep editing');
    expect(previewOpen()).toBe(true);

    escape();
    await answer('Discard');
    expect(previewOpen()).toBe(false);
    expect(drawerOpen()).toBe(true);
  });

  it("asks on the preview's close button", async () => {
    await startEditing('# Field notes, revised');
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    await answer('Keep editing');
    expect(previewOpen()).toBe(true);
  });

  it('closes a clean editor at once, with no prompt', async () => {
    await startEditing();
    escape();
    await waitFor(() => expect(previewOpen()).toBe(false));
    expect(prompt()).toBeNull();
    expect(drawerOpen()).toBe(true);
  });

  it('lets a clean drawer go on navigation at once', async () => {
    await startEditing();
    let left: boolean | undefined;
    await act(async () => { left = await dismissTabSurfaces(modal); });
    expect(left).toBe(true);
    expect(prompt()).toBeNull();
    expect(drawerOpen()).toBe(false);
  });
});
