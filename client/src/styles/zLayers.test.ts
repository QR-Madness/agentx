/**
 * zLayers — the app's overlay ladder (Development-Notes → Client Surface Map →
 * Layering). Every z-index in the client is compared in one stacking context,
 * so a value picked in isolation can bury a layer that must stay on top. Before
 * v0.21.307 the command palette sat at 1000, under the full-screen surfaces at
 * 2000. Over Memory it still took focus and typing, and Enter ran the command,
 * but nothing of it showed.
 *
 * This test reads every stylesheet and module under src/ as raw text and
 * collects every z-index declared: CSS `z-index:`, Tailwind `z-[N]` and `z-N`,
 * inline `zIndex:`. Test files are skipped. It checks three rules:
 *
 * - The palette is the top app layer. Only the toasts and the gates reach its
 *   z-index; a gate is something the app cannot be used past.
 * - Toasts sit just above the palette, so no layer can hide a notification.
 * - The modal stack (`.modal-layer`, one per ModalPortal entry) sits above the
 *   page chrome and below every overlay opened from inside an entry: the Radix
 *   popovers and dialogs, the Connectors & Tools form overlay, the model picker
 *   and the image lightbox.
 *
 * If this fails: give the new overlay a value from the ladder in
 * Development-Notes, below the palette. Only a gate goes above it, and it then
 * belongs in ABOVE_PALETTE with the reason.
 */

import { describe, it, expect } from 'vitest';

// Raw sources (root-absolute keys like `/src/styles/base.css`).
const sheets = import.meta.glob('/src/**/*.css', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;
const modules = import.meta.glob('/src/**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const PALETTE_CSS = 'src/components/common/CommandPalette.css';
const TOASTER_CSS = 'src/components/ui/Toaster.css';

/** Files whose layers may reach the palette's z-index, and why. */
const ABOVE_PALETTE: Record<string, string> = {
  [TOASTER_CSS]: 'toasts: a notification is never hidden',
  'src/components/ConnectGate.tsx': 'gate: confirm a #connect= link before anything else',
  'src/components/ServerSelector.tsx': "gate: the Connect screen's server dropdown",
  'src/layouts/ResizeHandles.css': "gate: the frameless window's resize grips",
};
const GATES = Object.keys(ABOVE_PALETTE).filter((file) => file !== TOASTER_CSS);

/**
 * Overlays that open from inside a modal-stack entry, so must paint above it.
 * A selector picks one rule out of a stylesheet that also styles the surface
 * itself; without one, every z-index in the file counts.
 */
const OPENED_OVER_THE_STACK: Array<[file: string, selector?: string]> = [
  ['src/components/ui/Dialog.tsx'],
  ['src/components/ui/Popover.tsx'],
  ['src/components/ui/Select.tsx'],
  ['src/components/ui/DropdownMenu.tsx'],
  ['src/components/ui/Tooltip.tsx'],
  ['src/components/toolkit/ToolkitPage.css', '.toolkit-modal-overlay'],
  ['src/components/common/ModelPickerModal.css'],
  ['src/components/common/ImageLightbox.tsx'],
];

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

interface Declaration {
  file: string;
  z: number;
}

/** Every z-index declared under src/, by file. */
function declarations(): Declaration[] {
  const found: Declaration[] = [];
  const add = (path: string, matches: IterableIterator<RegExpMatchArray>) => {
    for (const m of matches) found.push({ file: path.replace(/^\//, ''), z: Number(m[1]) });
  };
  for (const [path, css] of Object.entries(sheets)) {
    add(path, stripComments(css).matchAll(/z-index\s*:\s*(-?\d+)/g));
  }
  for (const [path, source] of Object.entries(modules)) {
    if (/\.test\.tsx?$/.test(path)) continue;
    const code = stripComments(source);
    add(path, code.matchAll(/(?<![\w-])z-\[(-?\d+)\]/g)); // Tailwind arbitrary value
    add(path, code.matchAll(/(?<![\w-])z-(\d+)(?![\w[-])/g)); // Tailwind scale (z-10)
    add(path, code.matchAll(/zIndex\s*:\s*(-?\d+)/g)); // inline style
  }
  return found;
}

/** The z-index a stylesheet's standalone rule for `selector` declares. */
function zOf(file: string, selector: string): number {
  const css = stripComments(sheets[`/${file}`] ?? '');
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const rule = new RegExp(`(?:^|[{};])\\s*${escaped}\\s*\\{([^}]*)\\}`).exec(css);
  const z = rule && /z-index\s*:\s*(-?\d+)/.exec(rule[1]);
  if (!z) throw new Error(`${file}: no rule \`${selector} { z-index: … }\``);
  return Number(z[1]);
}

const zIn = (file: string) => declarations().filter((d) => d.file === file).map((d) => d.z);

describe('overlay z-index ladder', () => {
  const all = declarations();
  const palette = zOf(PALETTE_CSS, '.cmdk-overlay');
  const toasts = zOf(TOASTER_CSS, '.toaster');
  const modalLayer = zOf('src/styles/base.css', '.modal-layer');

  it('reads the sources (an empty raw import would pass the real checks vacuously)', () => {
    expect(all.length).toBeGreaterThan(40);
    expect(zIn('src/layouts/TopBar.css').length).toBeGreaterThan(0);
    for (const file of [...GATES, ...OPENED_OVER_THE_STACK.map(([f]) => f)]) {
      expect(zIn(file).length, `${file} declares no z-index`).toBeGreaterThan(0);
    }
  });

  it('keeps the command palette above every layer but the toasts and the gates', () => {
    const offenders = all
      .filter((d) => d.z >= palette && !(d.file in ABOVE_PALETTE))
      .filter((d) => !(d.file === PALETTE_CSS && d.z === palette))
      .map((d) => `${d.file}: z-index ${d.z}`)
      .sort();

    expect(
      offenders,
      `at or above the command palette (z-index ${palette}), where only toasts and gates belong:\n` +
        offenders.join('\n'),
    ).toEqual([]);
  });

  it('puts the toasts above the palette and the gates above the toasts', () => {
    expect(toasts).toBeGreaterThan(palette);
    for (const gate of GATES) {
      expect(Math.min(...zIn(gate)), gate).toBeGreaterThan(toasts);
    }
  });

  it('keeps the modal stack above the page chrome and below what opens over it', () => {
    expect(modalLayer).toBeGreaterThan(Math.max(...zIn('src/layouts/TopBar.css')));
    for (const [file, selector] of OPENED_OVER_THE_STACK) {
      const z = selector ? zOf(file, selector) : Math.min(...zIn(file));
      expect(z, selector ? `${file} ${selector}` : file).toBeGreaterThan(modalLayer);
    }
  });
});
