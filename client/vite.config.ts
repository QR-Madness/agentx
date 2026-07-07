/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";
import { readFileSync } from "fs";
import { load } from "js-yaml";
import { resolve } from "path";

// Load version from versions.yaml (single source of truth)
const versionsPath = resolve(__dirname, "../versions.yaml");
const versions = load(readFileSync(versionsPath, "utf8")) as {
  client: { version: string };
  api: { protocol_version: number };
};

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// True when Tauri drives the renderer build/dev (it sets these when invoking the
// `beforeBuildCommand` / `beforeDevCommand`). The plain web/PWA build never sets
// them → `false`. This one literal is the lever: it gates the `src/platform/`
// import resolvers (so Tauri code is tree-shaken out of the web bundle) and
// disables the PWA service worker in the desktop shell.
// @ts-expect-error process is a nodejs global
const isTauriBuild = process.env.TAURI_ENV_PLATFORM != null || process.env.TAURI_PLATFORM != null;

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    // PWA shell — web build only. Disabled in the Tauri renderer (desktop uses
    // its own updater); when disabled the plugin still exports a no-op
    // `virtual:pwa-register`, so src/pwa/registerPwa.ts can call it either way.
    VitePWA({
      disable: isTauriBuild,
      registerType: "prompt", // soft "New version — Reload" toast, never yanks the page
      injectRegister: false, // we register manually in src/pwa/registerPwa.ts
      includeAssets: ["favicon.png", "apple-touch-icon.png"],
      manifest: {
        name: "AgentX",
        short_name: "AgentX",
        description: "AgentX — AI agent platform",
        start_url: "/",
        scope: "/",
        display: "standalone",
        orientation: "any",
        background_color: "#05070f", // matches the boot splash / theme
        theme_color: "#05070f",
        icons: [
          { src: "/pwa-192.png", sizes: "192x192", type: "image/png" },
          { src: "/pwa-512.png", sizes: "512x512", type: "image/png" },
          {
            src: "/pwa-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        // Precache the shell only; heavy on-demand assets load from network.
        globPatterns: ["**/*.{js,css,html,woff,woff2}"],
        cleanupOutdatedCaches: true,
        navigateFallback: "index.html",
        // Let API / well-known routes 404 honestly instead of returning the SPA.
        navigateFallbackDenylist: [/^\/api\//, /^\/\.well-known\//],
        // Chat app chunks (mermaid/recharts/katex) exceed the 2 MiB default.
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
      },
    }),
  ],

  // Inject version from versions.yaml
  define: {
    __APP_VERSION__: JSON.stringify(versions.client.version),
    __PROTOCOL_VERSION__: versions.api.protocol_version,
    __IS_TAURI__: JSON.stringify(isTauriBuild),
  },

  // Vitest — jsdom + Testing Library. CSS is skipped (css: false) so component
  // CSS imports don't need processing in unit tests.
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
});
