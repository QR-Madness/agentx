/**
 * MemoryIcon — a small animated glowing brain (currentColor, theme-safe).
 *
 * Two softly-tinted hemispheres under a pulsing glow, with synapses that
 * twinkle in sequence. Motion is pure CSS (see MemoryIcon.css) and honors
 * `prefers-reduced-motion`. Inherits color via `currentColor`; callers can
 * amp the glow/speed by styling descendant classes (GalaxyIcon precedent).
 *
 * Class namespace is `memory-glyph` — NOT `memory-icon`, which is a global
 * class already owned by MemoryInjectionBlock.css (unlayered per-component
 * CSS is global; that collision force-sized this icon to 18px everywhere).
 */

import './MemoryIcon.css';

const LEFT_LOBE =
  'M11.2 4.4C9.8 3.2 7.6 3.6 6.8 5.2C4.9 5.4 3.6 7.1 4 8.9C2.9 9.9 2.7 11.6 3.5 12.8' +
  'C3 14.6 4.1 16.3 5.9 16.7C6.3 18.5 8.1 19.7 9.9 19.3C10.3 19.8 10.7 20.1 11.2 20.3Z';
const RIGHT_LOBE =
  'M12.8 4.4C14.2 3.2 16.4 3.6 17.2 5.2C19.1 5.4 20.4 7.1 20 8.9C21.1 9.9 21.3 11.6 20.5 12.8' +
  'C21 14.6 19.9 16.3 18.1 16.7C17.7 18.5 15.9 19.7 14.1 19.3C13.7 19.8 13.3 20.1 12.8 20.3Z';

interface MemoryIconProps {
  size?: number;
  className?: string;
}

export function MemoryIcon({ size = 16, className = '' }: MemoryIconProps) {
  return (
    <svg
      className={`memory-glyph ${className}`.trim()}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {/* Hemispheres — a soft fill under stroked outlines, glowing as one */}
      <g className="memory-glyph-lobes">
        <path d={`${LEFT_LOBE} ${RIGHT_LOBE}`} fill="currentColor" opacity="0.14" stroke="none" />
        <path d={LEFT_LOBE} />
        <path d={RIGHT_LOBE} />
      </g>
      {/* Synapses — staggered twinkle (delays in MemoryIcon.css) */}
      <circle className="memory-glyph-syn" cx="7.6" cy="8.6" r="1" fill="currentColor" stroke="none" />
      <circle className="memory-glyph-syn" cx="16.4" cy="8.6" r="1" fill="currentColor" stroke="none" />
      <circle className="memory-glyph-syn" cx="9" cy="13.2" r="0.9" fill="currentColor" stroke="none" />
      <circle className="memory-glyph-syn" cx="15" cy="13.2" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}
