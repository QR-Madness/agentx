// Brand webfonts, self-hosted (offline-correct for Tauri). base.css resolves
// `var(--font-sans)` / `var(--font-mono)` to these families; without the
// imports the whole app silently falls back to system fonts.
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/700.css";

import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { registerPwa } from "./pwa/registerPwa";
import { initInstallPrompt } from "./pwa/installPrompt";
import { consumeConnectFragment } from "./lib/connectionString";

// Read + strip any `#connect=` share fragment before React renders, so the token
// never lingers in the URL/history; ConnectGate picks up the stashed payload.
consumeConnectFragment();

// Web/PWA shell wiring — both no-op under Tauri (and tree-shaken via __IS_TAURI__).
// Register the service worker + install affordance before the app renders.
registerPwa();
initInstallPrompt();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// Dismiss the boot splash (index.html) once React has committed its first
// frame, keeping a small minimum visible time so it doesn't flash.
const splash = document.getElementById("splash");
if (splash) {
  // Keep the splash up long enough for its animation to play a full beat every
  // launch (and give the UI a touch more time to settle) — enforced even when
  // React paints instantly. Must stay below the fallback timers below.
  const MIN_VISIBLE_MS = 1200;
  const FALLBACK_MS = 2000; // Android WebView can stall chained rAFs — guarantee removal
  const start = performance.now();
  let dismissed = false;
  const hide = () => {
    if (dismissed) return;
    dismissed = true;
    const wait = Math.max(0, MIN_VISIBLE_MS - (performance.now() - start));
    window.setTimeout(() => {
      splash.classList.add("splash-hide");
      window.setTimeout(() => splash.remove(), 450);
    }, wait);
  };
  // Fast path: two RAFs ≈ after the first paint of the real UI (reliable on desktop).
  requestAnimationFrame(() => requestAnimationFrame(hide));
  // Fallback: if rAF never fires (Android WebView), force the dismiss.
  window.setTimeout(hide, FALLBACK_MS);
}
