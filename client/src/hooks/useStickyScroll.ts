import { useCallback, useLayoutEffect, useRef, useState } from 'react';

/**
 * Sticky bottom auto-scroll for a streaming / append-only list.
 *
 * The container follows the latest content **while the reader is pinned to the bottom**,
 * but never yanks them away once they scroll up to read history — the same contract the
 * voice captions use, lifted into a reusable hook. Returns the container ref, whether
 * it's currently pinned to the bottom, an `onScroll` handler to wire onto the container,
 * and an explicit `scrollToBottom()` for a "jump to latest" affordance.
 *
 * @param deps       re-run the follow when these change (the item list / streamed text).
 * @param threshold  px from the bottom still counted as "pinned" (default 48).
 */
export function useStickyScroll<T extends HTMLElement = HTMLDivElement>(
  deps: unknown[],
  threshold = 48,
) {
  const ref = useRef<T>(null);
  const pinnedRef = useRef(true);
  const [atBottom, setAtBottom] = useState(true);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const pinned = el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
    pinnedRef.current = pinned;
    setAtBottom((prev) => (prev === pinned ? prev : pinned));
  }, [threshold]);

  const scrollToBottom = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setAtBottom(true);
  }, []);

  // Follow new content only while pinned to the bottom.
  useLayoutEffect(() => {
    const el = ref.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { ref, atBottom, scrollToBottom, onScroll };
}
