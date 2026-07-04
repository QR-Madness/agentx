/* Shared hooks. */

import { useEffect } from "react";

/**
 * Run `fn` now and every `intervalMs` — but only while the tab is visible.
 * Background tabs stop polling entirely (this dashboard is often left open
 * for hours); returning to the tab triggers an immediate refresh.
 */
export function usePolling(
  fn: () => void | Promise<void>,
  intervalMs: number,
  enabled = true,
): void {
  useEffect(() => {
    if (!enabled) return;
    let timer: number | null = null;
    const tick = () => void fn();
    const start = () => {
      if (timer === null) {
        tick();
        timer = window.setInterval(tick, intervalMs);
      }
    };
    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };
    const onVisibility = () =>
      document.visibilityState === "visible" ? start() : stop();
    onVisibility();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [fn, intervalMs, enabled]);
}
