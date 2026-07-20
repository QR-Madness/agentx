/**
 * VideoElement — a stored video, rendered as an inline `<video>` player (render-only;
 * video never enters model context). Served-blob url → authed object URL, same as
 * image/audio. Tauri webview codec support varies (WebKitGTK h264 depends on the
 * system GStreamer set), so a playback error degrades to a download link instead of
 * a dead player.
 */

import { useEffect, useState } from 'react';
import { Download, Film, VideoOff } from 'lucide-react';
import { resolveMediaBlob } from '../../../lib/mediaImage';
import { memoElement } from './memoElement';
import type { ElementRenderProps } from './types';

function VideoElementImpl({ element }: ElementRenderProps) {
  if (element.type !== 'video') return null;
  const { url, caption, title } = element;
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [playbackFailed, setPlaybackFailed] = useState(false);

  useEffect(() => {
    if (!url) {
      setFailed(true);
      return;
    }
    let alive = true;
    setSrc(null);
    setFailed(false);
    setPlaybackFailed(false);
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
          <VideoOff size={14} /> Couldn't load the video.
        </div>
      ) : src && playbackFailed ? (
        <a
          href={src}
          download={title || 'video'}
          className="flex items-center gap-2 rounded-lg border border-line bg-surface-sunken px-3 py-2 text-xs text-fg-secondary"
          title="This player can't decode the video here — save it instead"
        >
          <Download size={14} /> Playback isn't supported here — download the video.
        </a>
      ) : src ? (
        // eslint-disable-next-line jsx-a11y/media-has-caption -- agent-produced media; caption text renders below
        <video
          controls
          preload="metadata"
          src={src}
          className="max-h-96 w-auto max-w-full rounded-lg border border-line bg-black object-contain"
          onError={() => setPlaybackFailed(true)}
        />
      ) : (
        <div className="flex h-40 w-full items-center justify-center rounded-lg border border-line bg-surface-sunken text-fg-muted">
          <Film size={20} className="animate-pulse" />
        </div>
      )}
      {caption && <div className="text-xs text-fg-muted">{caption}</div>}
    </div>
  );
}

export const VideoElement = memoElement(VideoElementImpl);
