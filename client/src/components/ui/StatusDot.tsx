/**
 * StatusDot — tiny state indicator (design-kit `core/StatusDot`). Wraps the
 * global `.status-dot` classes (base.css) so the glow follows the theme's
 * status tokens. Decorative by default — pair with visible text or a title.
 */

import type { HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type StatusDotTone = 'online' | 'warning' | 'error' | 'inactive';

export interface StatusDotProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: StatusDotTone;
  /** Pulse animation for live states (streaming, consolidating). */
  pulse?: boolean;
}

export function StatusDot({ tone = 'online', pulse = false, className, ...props }: StatusDotProps) {
  return (
    <span
      aria-hidden="true"
      className={cn('status-dot', tone, pulse && 'animate-pulse', className)}
      {...props}
    />
  );
}
