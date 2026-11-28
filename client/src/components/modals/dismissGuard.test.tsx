/**
 * Dismissal guards — an unsaved edit survives every way of dismissing its surface.
 *
 * Before v0.21.309 the Projects drawer and Memory closed on ⌘T, the palette's
 * navigation and the TopBar's page pills (`dismissTabSurfaces`), as well as on
 * their own close button, Escape and backdrop, and an edit in progress inside
 * them was dropped without a word. An editor now registers `useDismissGuard` on
 * its stack entry, and `closeModal` — which every one of those paths calls —
 * asks "Discard unsaved changes?" first. A clean editor closes at once.
 *
 * The stack, the shells and the confirm are the real ones; only the surface
 * content is a stand-in editor (the real editors have their own tests).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import { ModalProvider, useModal, useEscapeToClose } from '../../contexts/ModalContext';
import { ConfirmProvider } from '../ui/ConfirmDialog';
import { SURFACES, dismissTabSurfaces } from '../../lib/surfaces';
import { ModalPortal } from './ModalPortal';

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ authRequired: false, isAuthenticated: true }),
}));

vi.mock('./stubs', async () => {
  const { useDismissGuard } = await import('../../hooks/useDismissGuard');
  function FakeEditor({ name, onClose }: { name: string; onClose: () => void }) {
    const [draft, setDraft] = useState('');
    useDismissGuard(draft !== '', `Your edits to ${name} have not been saved.`);
    return (
      <div data-testid={name}>
        <textarea aria-label={`${name} draft`} value={draft} onChange={(e) => setDraft(e.target.value)} />
        {/* The surface's own inline close, as WorkspacesPanel and the workbench render. */}
        <button type="button" onClick={onClose}>Close {name}</button>
      </div>
    );
  }
  return {
    // Projects: a drawer, so DrawerPanel owns its Escape and backdrop.
    WorkspacesDrawerContent: ({ onClose }: { onClose: () => void }) => <FakeEditor name="projects" onClose={onClose} />,
    // Memory: a bare full-screen surface that owns its Escape, as the workbench does.
    MemoryModalContent: function FakeMemory({ onClose }: { onClose: () => void }) {
      useEscapeToClose(onClose);
      return <FakeEditor name="memory" onClose={onClose} />;
    },
  };
});

let modal: ReturnType<typeof useModal>;
function CaptureModalApi() {
  modal = useModal();
  return null;
}

const layers = () =>
  [...document.querySelectorAll('#modal-root > .modal-layer')].map((el) => (el as HTMLElement).dataset.modal);

const escape = () =>
  act(() => {
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  });

const prompt = () => screen.queryByRole('dialog', { name: 'Discard unsaved changes?' });

async function open(key: 'workspaces' | 'memory', name: string) {
  act(() => { modal.openModal(SURFACES[key]); });
  await screen.findByTestId(name);
}

function type(name: string, text: string) {
  fireEvent.change(screen.getByLabelText(`${name} draft`), { target: { value: text } });
}

async function answer(label: 'Discard' | 'Keep editing') {
  fireEvent.click(await screen.findByRole('button', { name: label }));
  await waitFor(() => expect(prompt()).toBeNull());
}

describe('dismissal guards', () => {
  beforeEach(() => {
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

  describe('a clean editor closes at once, with no prompt', () => {
    it('through dismissTabSurfaces', async () => {
      await open('workspaces', 'projects');
      let left: boolean | undefined;
      await act(async () => { left = await dismissTabSurfaces(modal); });
      expect(left).toBe(true);
      expect(layers()).toEqual([]);
      expect(prompt()).toBeNull();
    });

    it('through closeModal, in the same call', async () => {
      await open('memory', 'memory');
      act(() => { void modal.closeModal(SURFACES.memory.id!); });
      expect(layers()).toEqual([]);
    });

    it('through Escape', async () => {
      await open('workspaces', 'projects');
      escape();
      expect(layers()).toEqual([]);
      expect(prompt()).toBeNull();
    });
  });

  describe('a dirty editor stays open until the discard is confirmed', () => {
    it('through dismissTabSurfaces (⌘T, the palette, the page pills)', async () => {
      await open('workspaces', 'projects');
      type('projects', 'half a thought');

      let first!: Promise<boolean>;
      act(() => { first = dismissTabSurfaces(modal); });
      expect(await screen.findByText('Your edits to projects have not been saved.')).toBeInTheDocument();
      expect(layers()).toEqual(['workspaces']);

      await answer('Keep editing');
      expect(await first).toBe(false);
      expect(layers()).toEqual(['workspaces']);
      expect(screen.getByLabelText('projects draft')).toHaveValue('half a thought');

      let second!: Promise<boolean>;
      act(() => { second = dismissTabSurfaces(modal); });
      await answer('Discard');
      expect(await second).toBe(true);
      expect(layers()).toEqual([]);
    });

    it('through Escape on a drawer', async () => {
      await open('workspaces', 'projects');
      type('projects', 'half a thought');
      escape();
      expect(await screen.findByRole('button', { name: 'Discard' })).toBeInTheDocument();
      expect(layers()).toEqual(['workspaces']);
      await answer('Discard');
      expect(layers()).toEqual([]);
    });

    it('through Escape on a full-screen surface', async () => {
      await open('memory', 'memory');
      type('memory', 'a revised claim');
      escape();
      await answer('Keep editing');
      expect(layers()).toEqual(['memory']);
    });

    it("through the surface's own close button", async () => {
      await open('memory', 'memory');
      type('memory', 'a revised claim');
      fireEvent.click(screen.getByRole('button', { name: 'Close memory' }));
      await answer('Keep editing');
      expect(layers()).toEqual(['memory']);
      fireEvent.click(screen.getByRole('button', { name: 'Close memory' }));
      await answer('Discard');
      expect(layers()).toEqual([]);
    });

    it("through the drawer's backdrop", async () => {
      await open('workspaces', 'projects');
      type('projects', 'half a thought');
      fireEvent.click(document.querySelector('.drawer-backdrop')!);
      await answer('Keep editing');
      expect(layers()).toEqual(['workspaces']);
    });

    it('asks once when asked twice while the prompt is up', async () => {
      await open('workspaces', 'projects');
      type('projects', 'half a thought');
      let a!: Promise<boolean>;
      let b!: Promise<boolean>;
      act(() => { a = modal.closeModal(SURFACES.workspaces.id!); });
      act(() => { b = modal.closeModal(SURFACES.workspaces.id!); });
      expect(b).toBe(a);
      await answer('Discard');
      expect(await a).toBe(true);
    });

    it('stops dismissTabSurfaces at the surface that stays, closing the clean ones', async () => {
      await open('memory', 'memory');
      await open('workspaces', 'projects');
      type('projects', 'half a thought');
      let left!: Promise<boolean>;
      act(() => { left = dismissTabSurfaces(modal); });
      await answer('Keep editing');
      expect(await left).toBe(false);
      expect(layers()).toEqual(['workspaces']);
    });
  });

  it('closes everything without asking on closeAll (session expiry)', async () => {
    await open('workspaces', 'projects');
    type('projects', 'half a thought');
    act(() => { modal.closeAll(); });
    expect(layers()).toEqual([]);
    expect(prompt()).toBeNull();
  });
});
