/**
 * ModalPortal — the modal stack paints, and takes Escape, in the order it opened.
 *
 * Every entry renders inside its own `.modal-layer`. The layers share one
 * z-index, so the later layer paints on top: a drawer opened over Memory covers
 * it, although the drawer shell's own z-index (1000) is lower than a full-screen
 * surface's (2000). `useEscapeToClose` then lets only the top entry close on
 * Escape, so stacked entries never close together. The z-index values
 * themselves are held by `styles/zLayers.test.ts`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { ModalProvider, useModal, useEscapeToClose } from '../../contexts/ModalContext';
import { SURFACES } from '../../lib/surfaces';
import { ModalPortal } from './ModalPortal';

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ authRequired: false, isAuthenticated: true }),
}));

// Light stand-ins for the lazy registry entries this test opens. The full-screen
// ones own their Escape the way the real surfaces do; the Logs drawer gets it
// from its DrawerPanel shell.
vi.mock('./stubs', () => {
  const surface = (name: string) =>
    function FakeSurface({ onClose }: { onClose: () => void }) {
      useEscapeToClose(onClose);
      return <div className="fake-surface" data-testid={name} />;
    };
  return {
    MemoryModalContent: surface('memory'),
    UnifiedSettingsModalContent: surface('settings'),
    LogsDrawerContent: () => <div data-testid="logs" />,
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

describe('ModalPortal stacking', () => {
  beforeEach(() => {
    const root = document.createElement('div');
    root.id = 'modal-root';
    document.body.appendChild(root);
    render(
      <ModalProvider>
        <CaptureModalApi />
        <ModalPortal />
      </ModalProvider>,
    );
  });

  afterEach(() => {
    document.getElementById('modal-root')?.remove();
  });

  it('renders each entry in its own layer, in the order it opened', async () => {
    act(() => { modal.openModal(SURFACES.memory); });
    await screen.findByTestId('memory');
    act(() => { modal.openModal(SURFACES.logs); });
    await screen.findByTestId('logs');

    expect(layers()).toEqual(['memory', 'logs']);
    // The drawer's whole shell (backdrop and panel) sits in the later layer.
    const [memoryLayer, logsLayer] = document.querySelectorAll('#modal-root > .modal-layer');
    expect(memoryLayer.querySelector('.fake-surface')).not.toBeNull();
    expect(logsLayer.querySelector('.drawer-backdrop .drawer-panel')).not.toBeNull();
  });

  it('raises an entry that is opened again to the top', async () => {
    act(() => { modal.openModal(SURFACES.memory); });
    await screen.findByTestId('memory');
    act(() => { modal.openModal(SURFACES.settings); });
    await screen.findByTestId('settings');
    act(() => { modal.openModal(SURFACES.memory); });

    expect(layers()).toEqual(['unifiedSettings', 'memory']);
  });

  it('closes only the top entry on Escape: a drawer over a surface, then the surface', async () => {
    act(() => { modal.openModal(SURFACES.memory); });
    await screen.findByTestId('memory');
    act(() => { modal.openModal(SURFACES.logs); });
    await screen.findByTestId('logs');

    escape();
    expect(layers()).toEqual(['memory']);
    escape();
    expect(layers()).toEqual([]);
  });

  it('closes only the top entry on Escape: a surface over a surface', async () => {
    act(() => { modal.openModal(SURFACES.memory); });
    await screen.findByTestId('memory');
    act(() => { modal.openModal(SURFACES.settings); });
    await screen.findByTestId('settings');

    escape();
    expect(layers()).toEqual(['memory']);
  });

  it('leaves an Escape that a layer above the stack already took', async () => {
    act(() => { modal.openModal(SURFACES.memory); });
    await screen.findByTestId('memory');
    // A Radix Select or confirm dismisses itself in the capture phase and
    // prevents the default, as Radix's DismissableLayer does.
    const radixLike = (e: KeyboardEvent) => { if (e.key === 'Escape') e.preventDefault(); };
    document.addEventListener('keydown', radixLike, true);
    try {
      escape();
      expect(layers()).toEqual(['memory']);
    } finally {
      document.removeEventListener('keydown', radixLike, true);
    }
    escape();
    expect(layers()).toEqual([]);
  });
});
