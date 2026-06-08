/**
 * GalaxyIcon — a small animated spiral galaxy (currentColor, theme-safe).
 *
 * A glowing core, two tilted spiral arms that slowly rotate, and one orbiting
 * star. Motion is pure CSS (see GalaxyIcon.css) and honors
 * `prefers-reduced-motion`. Inherits color via `currentColor`; callers can amp
 * the glow/speed by styling descendant classes (the TopBar does this on the
 * active nav pill).
 */

import './GalaxyIcon.css';

interface GalaxyIconProps {
  size?: number;
  className?: string;
}

export function GalaxyIcon({ size = 16, className = '' }: GalaxyIconProps) {
  return (
    <svg
      className={`galaxy-icon ${className}`.trim()}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      aria-hidden="true"
    >
      {/* Spiral disk — two tilted arms rotating as one */}
      <g className="galaxy-disk">
        <ellipse cx="12" cy="12" rx="9" ry="3.3" strokeWidth="1.5" opacity="0.75" transform="rotate(22 12 12)" />
        <ellipse cx="12" cy="12" rx="6.4" ry="2.2" strokeWidth="1.3" opacity="0.5" transform="rotate(-30 12 12)" />
      </g>
      {/* Glowing core */}
      <circle className="galaxy-core" cx="12" cy="12" r="2.6" fill="currentColor" stroke="none" />
      {/* Orbiting star */}
      <g className="galaxy-orbit">
        <circle className="galaxy-star" cx="21.2" cy="12" r="1" fill="currentColor" stroke="none" />
      </g>
    </svg>
  );
}
