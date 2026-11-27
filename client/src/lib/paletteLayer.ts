/**
 * paletteLayer — where the command palette sits among the app's overlays.
 *
 * The palette is the top app layer: it paints above the modal stack (drawers,
 * dialogs, full-screen surfaces), the Radix popovers and the pickers, with only
 * the toasts and the gates above it (the full ladder is in Development-Notes →
 * Client Surface Map → Layering; `styles/zLayers.test.ts` holds it). Two rules
 * follow from that, and both live here so the layers below can share them
 * without importing the palette itself:
 *
 * - **It owns the keys pressed in it.** A lower layer that listens in the
 *   capture phase would see them first, so it skips events from the palette
 *   (`isFromCommandPalette`). Bubble-phase listeners never see them: the palette
 *   stops their propagation.
 * - **It never opens where it could not take focus** (`canOpenCommandPalette`).
 */

/** Class of the palette's full-viewport overlay (`CommandPalette.css`). */
export const COMMAND_PALETTE_OVERLAY_CLASS = 'cmdk-overlay';

/** True for an event whose target sits inside the open command palette. */
export function isFromCommandPalette(event: Event): boolean {
  const target = event.target;
  return target instanceof Element && target.closest(`.${COMMAND_PALETTE_OVERLAY_CLASS}`) !== null;
}

/**
 * True while a modal Radix layer is open: a `Dialog` (every `ConfirmDialog`), a
 * modal `DropdownMenu`, a `Select`. Radix's DismissableLayer sets
 * `pointer-events: none` on `<body>` for as long as such a layer is mounted, and
 * the layer traps focus. No app code writes that style.
 */
export function isModalLayerOpen(doc: Document = document): boolean {
  return doc.body.style.pointerEvents === 'none';
}

/**
 * Whether ⌘K / the TopBar button may open the palette now. Not while a modal
 * Radix layer is open: the palette would paint above it, but the layer's focus
 * trap would keep the keyboard, so typing would land in the dialog behind it
 * (where Enter confirms).
 */
export function canOpenCommandPalette(doc: Document = document): boolean {
  return !isModalLayerOpen(doc);
}
