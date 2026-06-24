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
  // Keep the resolved blob URL together with the avatar it belongs to. Two bugs this
  // avoids: (1) showing a *stale* image — we only display `resolved.url` when it matches
  // the current `avatar`, so a reused instance never shows a previous agent's picture;
  // (2) getting stuck on the icon — we never imperatively clear the URL, so when `avatar`
  // momentarily flips to undefined and back (e.g. the profile list reloads), the image
  // reappears instantly from `resolved` instead of re-resolving through a blank gap.
  const [resolved, setResolved] = useState<{ avatar: string; url: string } | null>(null);

  useEffect(() => {
    if (!isImage || !avatar) return;
    let alive = true;
    resolveAvatarImage(avatar)
      .then((u) => { if (alive) setResolved({ avatar, url: u }); })
      .catch(() => { /* leave any prior resolution in place; render falls back to the icon */ });
    return () => {
      alive = false;
    };
  }, [avatar, isImage]);

  // Only show the image when the resolved URL is for the *current* avatar.
  const url = isImage && avatar && resolved?.avatar === avatar ? resolved.url : null;

  if (url) {
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
        // In `fill` mode the inline style governs sizing entirely. We deliberately do
        // NOT apply `className` here: at boxed sites it's the *icon glyph* sizing class
        // (e.g. `avatar-trigger__icon` → height:40%), which would squash the photo.
        className={fill ? undefined : `rounded-full object-cover ${className ?? ''}`}
        style={style}
        // A broken blob falls back to the icon: drop the resolution for this avatar so
        // `url` computes to null on the next render.
        onError={() => setResolved((r) => (r?.avatar === avatar ? null : r))}
      />
    );
  }

  const Icon = getAvatarIcon(avatar);
  return <Icon size={size} className={className} />;
}
