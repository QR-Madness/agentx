/**
 * MessageImages — render the images attached to a user turn (vision input).
 *
 * Each `ChatImageRef` points at a served blob (`…/documents/{doc}/raw`) that can't go
 * straight into an `<img src>` under auth, so we resolve it to an authed object URL
 * (`lib/mediaImage.ts::resolveMediaImage`, the same path avatars + generated images use).
 * Used both on the user bubble (live + restored) and the composer preview strip.
 */

import { useEffect, useState } from 'react';
import { ImageIcon, ImageOff, X } from 'lucide-react';
import { resolveMediaImage } from '../../lib/mediaImage';
import type { ChatImageRef } from '../../lib/api/types';

/** The served raw-blob path for an image ref. */
export function imageRefPath(ref: ChatImageRef): string {
  return `/api/workspaces/${encodeURIComponent(ref.workspace_id)}/documents/${encodeURIComponent(ref.doc_id)}/raw`;
}

function Thumb({ imageRef, onRemove }: { imageRef: ChatImageRef; onRemove?: () => void }) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const path = imageRefPath(imageRef);

  useEffect(() => {
    let alive = true;
    setSrc(null);
    setFailed(false);
    resolveMediaImage(path)
      .then((u) => alive && setSrc(u))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [path]);

  return (
    <div className="message-image-thumb">
      {failed ? (
        <div className="message-image-thumb-fallback" title="Couldn't load the image">
          <ImageOff size={16} />
        </div>
      ) : src ? (
        <img src={src} alt="Attached image" />
      ) : (
        <div className="message-image-thumb-loading">
          <ImageIcon size={16} className="animate-pulse" />
        </div>
      )}
      {onRemove && (
        <button
          type="button"
          className="message-image-thumb-remove"
          onClick={onRemove}
          title="Remove image"
          aria-label="Remove image"
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}

export function MessageImages({
  images,
  onRemove,
}: {
  images: ChatImageRef[];
  onRemove?: (index: number) => void;
}) {
  if (!images.length) return null;
  return (
    <div className="message-images">
      {images.map((ref, i) => (
        <Thumb
          key={`${ref.doc_id}-${i}`}
          imageRef={ref}
          onRemove={onRemove ? () => onRemove(i) : undefined}
        />
      ))}
    </div>
  );
}
