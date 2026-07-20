/**
 * AudioElement — an agent-generated (or MCP-passthrough) audio clip, rendered as an
 * inline player. The served-blob url resolves through the authed client to an object
 * URL (`lib/mediaImage.ts::resolveMediaBlob`) since a raw `<audio src>` can't carry auth.
 */

import { useEffect, useState } from 'react';
import { AudioLines, VolumeX } from 'lucide-react';
import { resolveMediaBlob } from '../../../lib/mediaImage';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

function AudioElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'audio') return null;
  const { url, caption, title } = element;
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!url) {
      setFailed(true);
      return;
    }
    let alive = true;
    setSrc(null);
    setFailed(false);
    resolveMediaBlob(url)
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
          <VolumeX size={14} /> Couldn't load the audio.
        </div>
      ) : src ? (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-surface-sunken px-2 py-1.5">
          <AudioLines size={16} className="shrink-0 text-accent" aria-hidden />
          {/* eslint-disable-next-line jsx-a11y/media-has-caption -- generated speech; the caption text renders below */}
          <audio controls preload="metadata" src={src} className="h-9 w-full min-w-48 max-w-full" />
        </div>
      ) : (
        <div className="flex h-12 w-full items-center justify-center rounded-lg border border-line bg-surface-sunken text-fg-muted">
          <AudioLines size={16} className="animate-pulse" />
        </div>
      )}
      {caption && <div className="text-xs text-fg-muted">{caption}</div>}
    </div>
  );
}

export const AudioElement = memoElement(AudioElementImpl);
