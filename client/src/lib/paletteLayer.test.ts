import { describe, it, expect, vi, afterEach } from 'vitest';
import { createElement } from 'react';
import { render, act } from '@testing-library/react';
import {
  COMMAND_PALETTE_OVERLAY_CLASS,
  canOpenCommandPalette,
  isFromCommandPalette,
  isModalLayerOpen,
} from './paletteLayer';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/Dialog';
import { ImageLightbox } from '../components/common/ImageLightbox';

afterEach(() => {
  document.body.style.pointerEvents = '';
  document.querySelectorAll(`.${COMMAND_PALETTE_OVERLAY_CLASS}`).forEach((el) => el.remove());
});

/** A stand-in for the open palette: its overlay with a focusable input inside. */
function mountPaletteOverlay(): HTMLInputElement {
  const overlay = document.createElement('div');
  overlay.className = COMMAND_PALETTE_OVERLAY_CLASS;
  const input = document.createElement('input');
  overlay.appendChild(input);
  document.body.appendChild(overlay);
  return input;
}

describe('canOpenCommandPalette', () => {
  it('allows the palette when no modal layer is open', () => {
    expect(isModalLayerOpen()).toBe(false);
    expect(canOpenCommandPalette()).toBe(true);
  });

  it('refuses it while a Radix dialog traps focus (the contract it reads)', () => {
    // Radix's DismissableLayer sets `pointer-events: none` on <body> while a modal
    // layer (Dialog, so every ConfirmDialog) is mounted. If a Radix upgrade stops
    // doing so, this fails, and the guard needs another signal.
    const view = render(
      createElement(
        Dialog,
        { open: true },
        createElement(
          DialogContent,
          null,
          createElement(DialogTitle, null, 'Discard unsaved changes?'),
          createElement(DialogDescription, null, 'A confirm, as useConfirm opens it.'),
        ),
      ),
    );
    expect(isModalLayerOpen()).toBe(true);
    expect(canOpenCommandPalette()).toBe(false);

    view.unmount();
    expect(canOpenCommandPalette()).toBe(true);
  });
});

describe('isFromCommandPalette', () => {
  it('is true for an event from inside the palette, false elsewhere', () => {
    const input = mountPaletteOverlay();
    const seen: boolean[] = [];
    const listener = (e: Event) => seen.push(isFromCommandPalette(e));
    window.addEventListener('keydown', listener, true);
    try {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    } finally {
      window.removeEventListener('keydown', listener, true);
    }
    expect(seen).toEqual([true, false, false]);
  });

  it("keeps the image lightbox's capture-phase Escape off an Escape meant for the palette", () => {
    const onClose = vi.fn();
    render(createElement(ImageLightbox, { src: 'blob:probe', alt: 'probe', onClose }));
    const input = mountPaletteOverlay();

    act(() => { input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); });
    expect(onClose).not.toHaveBeenCalled();

    act(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
