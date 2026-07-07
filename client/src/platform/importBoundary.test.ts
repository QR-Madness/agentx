/**
 * importBoundary — the guardrail that keeps the whole two-shell split honest.
 *
 * `@tauri-apps/*` may be imported ONLY under `src/platform/` (specifically the
 * `.tauri.ts` impls). Everywhere else, native capabilities must go through the
 * `platform` façade, so the compile-time `__IS_TAURI__` gate can tree-shake all
 * Tauri code out of the web/PWA bundle. This test loads every `.ts`/`.tsx` file
 * OUTSIDE `src/platform/` (as raw text via Vite's glob) and fails on any
 * `@tauri-apps` import specifier.
 *
 * If this fails: don't import Tauri directly — add a capability to `src/platform/`
 * (an interface + `.tauri.ts` / `.web.ts` pair wired by the `__IS_TAURI__` gate)
 * and call it via `platform.*`.
 */

import { describe, it, expect } from 'vitest';

// Raw source of every module under src/ (root-absolute keys like `/src/lib/x.ts`).
const modules = import.meta.glob('/src/**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/** Matches an import/`import()`/re-export specifier for the `@tauri-apps` scope. */
const TAURI_SPECIFIER = /['"]@tauri-apps\//;

describe('platform import boundary', () => {
  it('no @tauri-apps imports outside src/platform/', () => {
    const offenders = Object.entries(modules)
      .filter(([path]) => !path.startsWith('/src/platform/'))
      .filter(([, source]) => TAURI_SPECIFIER.test(source))
      .map(([path]) => path.replace(/^\//, ''))
      .sort();

    expect(
      offenders,
      `@tauri-apps imported outside src/platform/ — route it through the platform façade:\n` +
        offenders.join('\n'),
    ).toEqual([]);
  });
});
