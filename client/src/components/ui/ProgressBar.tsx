/**
 * ProgressBar — a slim determinate/indeterminate bar built on semantic tokens.
 *
 * - determinate: pass `value` (0–1); the fill width eases (300ms) so stepped
 *   updates animate smoothly rather than snapping.
 * - indeterminate: omit `value` (or pass `indeterminate`) for a shimmer sweep
 *   used while a phase's total is unknown (e.g. discovery).
 *
 * Respects `prefers-reduced-motion` (no shimmer, no transition).
 */

import { cn } from '../../lib/utils';

type Tone = 'accent' | 'success' | 'warning';

const FILL_TONE: Record<Tone, string> = {
  accent: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
};

export interface ProgressBarProps {
  /** 0–1. Ignored when `indeterminate`. */
  value?: number;
  indeterminate?: boolean;
  tone?: Tone;
  className?: string;
  'aria-label'?: string;
}

export function ProgressBar({
  value = 0,
  indeterminate = false,
  tone = 'accent',
  className,
  'aria-label': ariaLabel,
}: ProgressBarProps) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      role="progressbar"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={indeterminate ? undefined : 100}
      aria-valuenow={indeterminate ? undefined : Math.round(pct)}
      className={cn(
        'relative h-1.5 w-full overflow-hidden rounded-pill bg-surface-sunken',
        className,
      )}
    >
      {indeterminate ? (
        <div className={cn('absolute inset-y-0 w-1/3 rounded-pill ax-progress-shimmer', FILL_TONE[tone])} />
      ) : (
        <div
          className={cn(
            'h-full rounded-pill transition-[width] duration-300 ease-out motion-reduce:transition-none',
            FILL_TONE[tone],
          )}
          style={{ width: `${pct}%` }}
        />
      )}
    </div>
  );
}
