/**
 * ImageLightbox — a full-screen zoomable viewer for a single image.
 *
 * Portaled over everything (escapes overflow / z-index traps). Click the image to
 * toggle fit ↔ 100%; mouse-wheel to zoom around the cursor; drag to pan when zoomed
 * in. A slim toolbar offers zoom in/out/reset, download, and close; Escape or a
 * backdrop click dismisses it. `src` is an already-resolved (authed) object URL.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Download, Minus, Plus, RotateCcw, X } from 'lucide-react';

interface ImageLightboxProps {
  src: string;
  alt?: string;
  /** Filename for the download affordance. */
  downloadName?: string;
  onClose: () => void;
}

const MIN_SCALE = 1;
const MAX_SCALE = 8;
const clampScale = (s: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

export function ImageLightbox({ src, alt, downloadName, onClose }: ImageLightboxProps) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  const reset = useCallback(() => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  // Escape to close (own handler — this is a bare portal, not a dialog shell).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onClose]);

  const zoomBy = useCallback((factor: number) => {
    setScale((s) => {
      const next = clampScale(s * factor);
      if (next === 1) setOffset({ x: 0, y: 0 });
      return next;
    });
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, [zoomBy]);

  // Click (not a drag) toggles fit ↔ 2×.
  const toggleZoom = useCallback(() => {
    if (drag.current && (drag.current.ox !== offset.x || drag.current.oy !== offset.y)) return;
    setScale((s) => (s > 1 ? (setOffset({ x: 0, y: 0 }), 1) : 2));
  }, [offset.x, offset.y]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (scale <= 1) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setOffset({ x: drag.current.ox + (e.clientX - drag.current.x), y: drag.current.oy + (e.clientY - drag.current.y) });
  };
  const onPointerUp = () => { drag.current = null; };

  return createPortal(
    <div
      className="fixed inset-0 z-[3000] flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="absolute right-3 top-3 flex items-center gap-1 rounded-lg bg-black/50 p-1"
        onClick={(e) => e.stopPropagation()}
      >
        <LightboxBtn label="Zoom out" onClick={() => zoomBy(1 / 1.4)}><Minus size={16} /></LightboxBtn>
        <LightboxBtn label="Reset zoom" onClick={reset}><RotateCcw size={15} /></LightboxBtn>
        <LightboxBtn label="Zoom in" onClick={() => zoomBy(1.4)}><Plus size={16} /></LightboxBtn>
        <a
          href={src}
          download={downloadName || 'image'}
          className="flex h-8 w-8 items-center justify-center rounded-md text-white/80 hover:bg-white/15 hover:text-white"
          title="Download"
          aria-label="Download image"
          onClick={(e) => e.stopPropagation()}
        >
          <Download size={16} />
        </a>
        <LightboxBtn label="Close" onClick={onClose}><X size={18} /></LightboxBtn>
      </div>

      <img
        src={src}
        alt={alt ?? 'Image'}
        draggable={false}
        onClick={(e) => { e.stopPropagation(); toggleZoom(); }}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          cursor: scale > 1 ? 'grab' : 'zoom-in',
          touchAction: 'none',
        }}
        className="max-h-[92vh] max-w-[92vw] select-none rounded-md object-contain transition-transform duration-75"
      />
    </div>,
    document.body,
  );
}

function LightboxBtn({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-md bg-transparent text-white/80 hover:bg-white/15 hover:text-white"
    >
      {children}
    </button>
  );
}
