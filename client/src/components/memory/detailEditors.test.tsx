/**
 * Memory's entity and fact editors — an edit in progress is not closed away
 * without asking.
 *
 * Before v0.21.309 an open edit form in EntityDetail or FactDetail was dropped
 * by anything that dismissed Memory: its close button, Escape, ⌘T, the palette's
 * navigation, the TopBar's page pills. The detail pane's own close dropped it
 * too. Both editors now register `useDismissGuard` on the Memory entry.
 *
 * The stack and the confirm are real; the workbench is a stand-in that owns its
 * Escape and close the way MemoryWorkbench does, around the real detail editors.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { MemoryFact } from '../../lib/api';
import { ModalProvider, useModal } from '../../contexts/ModalContext';
import { ConfirmProvider } from '../ui/ConfirmDialog';
import { SURFACES, dismissTabSurfaces } from '../../lib/surfaces';
import { ModalPortal } from '../modals/ModalPortal';

const fact: MemoryFact = {
  id: 'f1', claim: 'Rivers carve canyons over millennia', confidence: 0.9, source: 'conversation',
  channel: '_default', created_at: '2026-11-26T12:00:00Z', entity_ids: [],
};

const graph = {
  entity: { id: 'e1', name: 'Ada Lovelace', type: 'person', channel: '_default', salience: 0.8, aliases: [] },
  facts: [],
  relationships: [],
};

vi.mock('../../lib/hooks', () => {
  const mutation = () => ({ mutate: vi.fn(), loading: false });
  return {
    useEntityGraph: () => ({ graph, loading: false, error: null }),
    useUpdateMemoryEntity: mutation,
    useDeleteMemoryEntity: mutation,
    useUpdateMemoryFact: mutation,
    useDeleteMemoryFact: mutation,
    useRememberMemoryFact: mutation,
    useForgetMemoryFact: mutation,
    useLinkFactEntity: mutation,
    useUnlinkFactEntity: mutation,
    useFactProvenance: () => ({ fetch: vi.fn(), loading: false }),
    useMemoryEntities: () => ({ entities: [], loading: false }),
  };
});
vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ authRequired: false, isAuthenticated: true }),
}));

const detail = vi.hoisted(() => ({ kind: 'entity' as 'entity' | 'fact', onDetailClose: vi.fn() }));

vi.mock('../modals/stubs', async () => {
  const { useEscapeToClose } = await import('../../contexts/ModalContext');
  const { EntityDetail } = await import('./EntityDetail');
  const { FactDetail } = await import('./FactDetail');
  return {
    MemoryModalContent: function FakeWorkbench({ onClose }: { onClose: () => void }) {
      useEscapeToClose(onClose);
      return (
        <div data-testid="memory">
          <button type="button" onClick={onClose}>Close Memory</button>
          {detail.kind === 'entity' ? (
            <EntityDetail entityId="e1" onClose={detail.onDetailClose} onDeleted={vi.fn()} onRefreshList={vi.fn()} />
          ) : (
            <FactDetail fact={fact} onClose={detail.onDetailClose} onDeleted={vi.fn()} onUpdated={vi.fn()} onNavigateEntity={vi.fn()} />
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

const memoryOpen = () => !!document.querySelector('#modal-root > .modal-layer[data-modal="memory"]');
const prompt = () => screen.queryByRole('dialog', { name: 'Discard unsaved changes?' });

const escape = () =>
  act(() => {
    document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
  });

async function answer(label: 'Discard' | 'Keep editing') {
  fireEvent.click(await screen.findByRole('button', { name: label }));
  await waitFor(() => expect(prompt()).toBeNull());
}

async function openEditor(kind: 'entity' | 'fact') {
  detail.kind = kind;
  act(() => { modal.openModal(SURFACES.memory); });
  await screen.findByTestId('memory');
  fireEvent.click(screen.getByRole('button', { name: /Edit/ }));
}

/** The first text field of each edit form: the entity's name, the fact's claim. */
const firstField = () => document.querySelector<HTMLInputElement | HTMLTextAreaElement>('.edit-form input, .edit-form textarea')!;

describe('Memory detail editors', () => {
  beforeEach(() => {
    detail.onDetailClose.mockReset();
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

  describe.each(['entity', 'fact'] as const)('the %s editor', (kind) => {
    it('keeps a dirty edit when Memory is dismissed by navigation, until discarded', async () => {
      await openEditor(kind);
      fireEvent.change(firstField(), { target: { value: 'A revised value' } });

      let left!: Promise<boolean>;
      act(() => { left = dismissTabSurfaces(modal); });
      await answer('Keep editing');
      expect(await left).toBe(false);
      expect(memoryOpen()).toBe(true);
      expect(firstField()).toHaveValue('A revised value');

      act(() => { left = dismissTabSurfaces(modal); });
      await answer('Discard');
      expect(await left).toBe(true);
      expect(memoryOpen()).toBe(false);
    });

    it("asks on Escape and on Memory's close button", async () => {
      await openEditor(kind);
      fireEvent.change(firstField(), { target: { value: 'A revised value' } });
      escape();
      await answer('Keep editing');
      fireEvent.click(screen.getByRole('button', { name: 'Close Memory' }));
      await answer('Keep editing');
      expect(memoryOpen()).toBe(true);
    });

    it("asks on the detail pane's own close", async () => {
      await openEditor(kind);
      fireEvent.change(firstField(), { target: { value: 'A revised value' } });
      fireEvent.click(screen.getByRole('button', { name: `Close ${kind}` }));
      await answer('Keep editing');
      expect(detail.onDetailClose).not.toHaveBeenCalled();
      fireEvent.click(screen.getByRole('button', { name: `Close ${kind}` }));
      await answer('Discard');
      expect(detail.onDetailClose).toHaveBeenCalledTimes(1);
    });

    it('lets an open but unchanged editor go at once', async () => {
      await openEditor(kind);
      let left: boolean | undefined;
      await act(async () => { left = await dismissTabSurfaces(modal); });
      expect(left).toBe(true);
      expect(prompt()).toBeNull();
      expect(memoryOpen()).toBe(false);
    });
  });
});
