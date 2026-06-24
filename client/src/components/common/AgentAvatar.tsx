/**
 * AgentAvatar — renders an agent's avatar, whether it's a lucide icon id or a generated
 * `media:` image. For image avatars it resolves the ref to an object URL (authed fetch,
 * cached) and shows an `<img>`; until then (and on any failure) it falls back to the lucide
 * icon, so it's a safe drop-in for the existing `getAvatarIcon(...)` render sites.
 */

import { useEffect, useState } from 'react';
import { getAvatarIcon, isImageAvatar } from '../../lib/avatars';
import { resolveAvatarImage } from '../../lib/mediaImage';

interface AgentAvatarProps {
  avatar?: string;
  size?: number;
  /** Applied to the lucide icon (color etc.); the image is always rounded + cover. */
  className?: string;
  /** Image avatars **fill** their container (for sites that wrap the avatar in a fixed-size
   *  circle — most of them) instead of rendering at the icon-glyph `size`. */
  fill?: boolean;
}

export function AgentAvatar({ avatar, size = 20, className, fill }: AgentAvatarProps) {
  const isImage = isImageAvatar(avatar);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    // Clear any prior image first. Without this, when React reuses this instance
    // for a *different* agent (same position in a list), the previous agent's blob
    // URL lingers in `url` until the new one resolves — which renders as "every
    // agent shows the first avatar". Resetting on every avatar change makes the
    // image strictly track the current `avatar` prop.
    setUrl(null);
    if (!isImage || !avatar) return;
    let alive = true;
    resolveAvatarImage(avatar)
      .then((u) => alive && setUrl(u))
      .catch(() => alive && setUrl(null));
    return () => {
      alive = false;
    };
  }, [avatar, isImage]);

  if (isImage && url) {
    // Fill the wrapping circle (inherit its radius) at boxed sites; else explicit px.
    // `aspect-ratio` (not height:100%) sets the height — webkit2gtk (Tauri) doesn't
    // resolve a flex child's percentage height reliably, which left the image collapsed.
    const style: React.CSSProperties = fill
      ? { width: '100%', aspectRatio: '1 / 1', objectFit: 'cover', borderRadius: 'inherit', display: 'block' }
      : { width: size, height: size };
    return (
      <img
        src={url}
        alt=""
        className={fill ? className : `rounded-full object-cover ${className ?? ''}`}
        style={style}
        onError={() => setUrl(null)}  // a broken blob falls back to the icon, not a broken-image glyph
      />
    );
  }

  const Icon = getAvatarIcon(avatar);
  return <Icon size={size} className={className} />;
}
