/**
 * scrollToAnchor — send the user to a specific element and briefly mark it.
 *
 * Generalized from the Plans-drawer "jump to step" recipe in ChatPanel, which
 * had the shape right but kept it local: find by data-attribute, defer a frame
 * so freshly-mounted content exists, scroll it into view, add a flash class,
 * remove it after the animation.
 *
 * Landing somewhere without saying *where* is disorienting — the scroll alone
 * leaves the user hunting for what changed. The flash answers "this one".
 * Styling lives with the caller's CSS; the reduced-motion guard belongs there
 * too (`@media (prefers-reduced-motion: reduce) { animation: none }`), so the
 * mark still appears for users who don't want movement.
 */

/** How long the flash class stays on. Matches the 1.4s animations in use. */
const FLASH_MS = 1500;

export interface ScrollToAnchorOptions {
  /** Search inside this element instead of the document. */
  within?: ParentNode | null;
  /** Class applied while the element is highlighted. */
  flashClass?: string;
  block?: ScrollLogicalPosition;
  /** Extra frames to wait — for content that mounts in stages (lazy sections). */
  retries?: number;
}

/**
 * Scroll to `[data-<attribute>="<value>"]` and flash it.
 *
 * Returns a cleanup function; call it if the caller unmounts mid-flight so a
 * pending timer can't touch a detached node.
 */
export function scrollToAnchor(
  attribute: string,
  value: string,
  {
    within,
    flashClass = 'flash-target',
    block = 'center',
    retries = 3,
  }: ScrollToAnchorOptions = {},
): () => void {
  const selector = `[data-${attribute}="${CSS.escape(value)}"]`;
  let frame = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let target: HTMLElement | null = null;
  let cancelled = false;

  const attempt = (remaining: number) => {
    if (cancelled) return;
    const root = within ?? document;
    const el = root.querySelector<HTMLElement>(selector);

    if (!el) {
      // A lazy section may still be resolving; try again next frame.
      if (remaining > 0) frame = requestAnimationFrame(() => attempt(remaining - 1));
      return;
    }

    target = el;
    el.scrollIntoView({ behavior: 'smooth', block });
    el.classList.add(flashClass);
    timer = setTimeout(() => el.classList.remove(flashClass), FLASH_MS);
  };

  frame = requestAnimationFrame(() => attempt(retries));

  return () => {
    cancelled = true;
    cancelAnimationFrame(frame);
    if (timer) clearTimeout(timer);
    target?.classList.remove(flashClass);
  };
}
