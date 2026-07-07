/**
 * runtime.ts — the two platform-detection mechanisms, and why they differ.
 *
 * There are deliberately TWO of them:
 *
 *  - `IS_TAURI` (the `__IS_TAURI__` compile-time define) is the **import gate**.
 *    Because it's a literal, `IS_TAURI ? import('x.tauri') : import('x.web')`
 *    branches are statically eliminated by Rollup, so the dead branch's module —
 *    and its transitive `@tauri-apps/*` deps — never enter the other shell's
 *    bundle. This is the ONLY thing that keeps Tauri bytes out of the web build.
 *
 *  - `isTauriRuntime()` (a `__TAURI_INTERNALS__` global probe) is for UI guards
 *    that must behave correctly inside a single already-built bundle — e.g.
 *    `bun run dev`, where one bundle serves both the browser and (potentially) a
 *    webview. Never use it for the import gate: a `@tauri-apps` import reached at
 *    runtime is still emitted into the web bundle.
 *
 * Window-chrome guards (`isTauri` / `isMac` / `showWindowControls`) live in
 * `lib/platform.ts` and stay runtime-based; this module is only the compile-time
 * gate + its runtime sibling.
 */

/** Compile-time platform gate. See `__IS_TAURI__` in `src/vite-env.d.ts`. */
export const IS_TAURI: boolean = __IS_TAURI__;

/** Runtime probe — true inside a Tauri webview, false in a plain browser tab. */
export function isTauriRuntime(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}
