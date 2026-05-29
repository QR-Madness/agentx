import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// Dismiss the boot splash (index.html) once React has committed its first
// frame, keeping a small minimum visible time so it doesn't flash.
const splash = document.getElementById("splash");
if (splash) {
  const MIN_VISIBLE_MS = 400;
  const start = performance.now();
  const hide = () => {
    const wait = Math.max(0, MIN_VISIBLE_MS - (performance.now() - start));
    window.setTimeout(() => {
      splash.classList.add("splash-hide");
      window.setTimeout(() => splash.remove(), 450);
    }, wait);
  };
  // Two RAFs ≈ after the first paint of the real UI.
  requestAnimationFrame(() => requestAnimationFrame(hide));
}
