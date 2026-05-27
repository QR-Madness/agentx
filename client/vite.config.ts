/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
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

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],

  // Inject version from versions.yaml
  define: {
    __APP_VERSION__: JSON.stringify(versions.client.version),
    __PROTOCOL_VERSION__: versions.api.protocol_version,
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
