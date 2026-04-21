import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
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
export default defineConfig(async () => ({
  plugins: [react()],

  // Inject version from versions.yaml
  define: {
    __APP_VERSION__: JSON.stringify(versions.client.version),
    __PROTOCOL_VERSION__: versions.api.protocol_version,
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
}));
