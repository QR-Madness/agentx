/**
 * DropdownPortal — anchor-positioned floating panel.
 *
 * Centralizes the four bits every chat dropdown was reimplementing:
 *   1. portal render to body (escapes overflow / z-index traps),
 *   2. anchor-relative positioning with automatic flip when the
 *      preferred side has no room,
 *   3. close on Escape + outside click (ignores clicks on the anchor
 *      itself so the trigger toggles cleanly),
 *   4. reposition on window resize and document scroll.
 *
 * Consumers pass the anchor element via `anchorRef` and own the dropdown's
 * visual styling. Positioning is applied via inline style on a wrapping
 * div so the consumer's CSS only needs to care about appearance.
 *
 * Note on positioning: this is intentionally simple. We compute one of
 * `top` / `bottom` (with flip) and one of `left` / `right` (alignment),
 * then let the consumer's CSS size the panel. No collision-aware width
 * clamping yet — add it if/when a future panel needs it.
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';

export type DropdownSide = 'top' | 'bottom';
export type DropdownAlign = 'start' | 'end';

interface DropdownPortalProps {
  isOpen: boolean;
  onClose: () => void;
  anchorRef: RefObject<HTMLElement | null>;
  /** Which side of the anchor to prefer; flips automatically if no room. */
  preferredSide?: DropdownSide;
  /** Align dropdown edge to anchor edge: 'start' = left/left, 'end' = right/right. */
  align?: DropdownAlign;
  /** Pixel gap between anchor and dropdown. */
  gap?: number;
  /** Used for initial flip-room estimation before the dropdown has measured. */
  estimatedHeight?: number;
  /** Extra class on the positioned wrapper (consumers usually set it on their inner element instead). */
  className?: string;
  children: ReactNode;
}

interface ComputedPosition {
  top?: number;
  bottom?: number;
  left?: number;
  right?: number;
}

export function DropdownPortal({
  isOpen,
  onClose,
  anchorRef,
  preferredSide = 'bottom',
  align = 'start',
  gap = 8,
  estimatedHeight = 360,
  className,
  children,
}: DropdownPortalProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<ComputedPosition>({});

  // Compute (and recompute) position. useLayoutEffect avoids a paint flash
  // on open; the measured-height re-measure on the next tick refines the
  // flip decision once the dropdown has actually laid out.
  useLayoutEffect(() => {
    if (!isOpen) return;

    const update = () => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      const measured = wrapperRef.current?.offsetHeight ?? estimatedHeight;
      const roomBelow = window.innerHeight - rect.bottom;
      const roomAbove = rect.top;
      const flip =
        preferredSide === 'bottom'
          ? roomBelow < measured + gap && roomAbove > roomBelow
          : roomAbove < measured + gap && roomBelow > roomAbove;
      const side: DropdownSide = flip
        ? preferredSide === 'bottom' ? 'top' : 'bottom'
        : preferredSide;

      const next: ComputedPosition = {};
      if (side === 'bottom') {
        next.top = rect.bottom + gap;
      } else {
        next.bottom = Math.max(8, window.innerHeight - rect.top + gap);
      }
      if (align === 'start') {
        next.left = rect.left;
      } else {
        next.right = window.innerWidth - rect.right;
      }
      setPosition(next);
    };

    update();
    // Re-measure once mounted in case the actual height differs from the
    // estimate enough to change the flip decision.
    const raf = requestAnimationFrame(update);
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
    };
  }, [isOpen, anchorRef, preferredSide, align, gap, estimatedHeight]);

  // Outside click + Escape. The anchor is excluded so toggling the
  // trigger button closes via the parent's own state, not via this.
  useEffect(() => {
    if (!isOpen) return;

    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (wrapperRef.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [isOpen, onClose, anchorRef]);

  if (!isOpen) return null;

  return createPortal(
    <div
      ref={wrapperRef}
      className={className}
      style={{ position: 'fixed', zIndex: 1000, ...position }}
    >
      {children}
    </div>,
    document.body,
  );
}
