/**
 * AgentAvatar — renders an agent's avatar, whether it's a lucide icon id or a generated
 * `media:` image. For image avatars it resolves the ref to an object URL (authed fetch,
 * cached) and shows an `<img>`; until then (and on any failure) it falls back to the lucide
 * icon, so it's a safe drop-in for the existing `getAvatarIcon(...)` render sites.
 */

import { useEffect, useState } from 'react';
import { getAvatarIcon, isImageAvatar } from '../../lib/avatars';
import { resolveAvatarImage } from '../../lib/avatarImage';

interface AgentAvatarProps {
  avatar?: string;
  size?: number;
  /** Applied to the lucide icon (color etc.); the image is always rounded + cover. */
  className?: string;
}

export function AgentAvatar({ avatar, size = 20, className }: AgentAvatarProps) {
  const isImage = isImageAvatar(avatar);
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!isImage || !avatar) {
      setUrl(null);
      return;
    }
    let alive = true;
    resolveAvatarImage(avatar)
      .then((u) => alive && setUrl(u))
      .catch(() => alive && setUrl(null));
    return () => {
      alive = false;
    };
  }, [avatar, isImage]);

  if (isImage && url) {
    return (
      <img
        src={url}
        alt=""
        className={`rounded-full object-cover ${className ?? ''}`}
        style={{ width: size, height: size }}
      />
    );
  }

  const Icon = getAvatarIcon(avatar);
  return <Icon size={size} className={className} />;
}
