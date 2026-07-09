/**
 * ImageElement — an agent-generated image, displayed inline in the conversation.
 *
 * The element carries a served-blob path (`…/documents/{doc}/raw`), which can't be put
 * directly in an `<img src>` under auth — so we resolve it to an authed object URL
 * (`lib/mediaImage.ts::resolveMediaImage`, the same path avatars use) and show it. A quiet
 * placeholder while loading; a small note if it can't load.
 */

import { useEffect, useState } from 'react';
import { ImageIcon, ImageOff } from 'lucide-react';
import { resolveMediaImage } from '../../../lib/mediaImage';
import { ImageLightbox } from '../../common/ImageLightbox';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

function ImageElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'image') return null;
  const { url, alt, title } = element;
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [zoomed, setZoomed] = useState(false);

  useEffect(() => {
    if (!url) {
      setFailed(true);
      return;
    }
    let alive = true;
    setSrc(null);
    setFailed(false);
    resolveMediaImage(url)
      .then((u) => alive && setSrc(u))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [url]);

  return (
    <div className="flex flex-col gap-1.5">
      {title && <div className="text-sm font-medium text-fg">{title}</div>}
      {failed ? (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-surface-sunken px-3 py-2 text-xs text-fg-muted">
          <ImageOff size={14} /> Couldn't load the image.
        </div>
      ) : src ? (
        <>
          <button
            type="button"
            onClick={() => setZoomed(true)}
            className="block cursor-zoom-in bg-transparent p-0"
            title="Click to zoom"
            aria-label="Zoom image"
          >
            <img
              src={src}
              alt={alt ?? 'Generated image'}
              className="max-h-96 w-auto max-w-full rounded-lg border border-line object-contain"
            />
          </button>
          {zoomed && (
            <ImageLightbox src={src} alt={alt ?? title} downloadName={title} onClose={() => setZoomed(false)} />
          )}
        </>
      ) : (
        <div className="flex h-40 w-full items-center justify-center rounded-lg border border-line bg-surface-sunken text-fg-muted">
          <ImageIcon size={20} className="animate-pulse" />
        </div>
      )}
      {alt && <div className="text-xs text-fg-muted">{alt}</div>}
    </div>
  );
}

export const ImageElement = memoElement(ImageElementImpl);
