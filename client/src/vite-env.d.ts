/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

declare const __APP_VERSION__: string;
declare const __PROTOCOL_VERSION__: number;

/**
 * Compile-time platform gate. `true` when Vite is invoked by Tauri
 * (`beforeBuildCommand` / `beforeDevCommand` set `TAURI_ENV_PLATFORM`), `false`
 * for the plain web/PWA build (`bun run dev` / `client:build:web`). Because it's
 * a literal, `__IS_TAURI__ ? … : …` branches are statically eliminated, so the
 * dead branch's `import()` — and its transitive `@tauri-apps/*` deps — never
 * enter the other shell's bundle. See src/platform/runtime.ts.
 */
declare const __IS_TAURI__: boolean;

interface ImportMetaEnv {
  /** Public web origin for shareable connection links (desktop share targets the PWA). */
  readonly VITE_PUBLIC_APP_URL?: string;
}
