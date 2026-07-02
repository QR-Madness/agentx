import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Standalone GUI for the AgentX deployment manager. In dev, API requests are
// proxied to the local manager server (agentx-manager serve, port 12320); in
// production the built dist/ is served by the manager itself (same origin).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:12320",
    },
  },
});
