/**
 * Badge — small status pill. Replaces the ad-hoc `.stat-badge` / `.active-badge`
 * styles duplicated across MemoryPanel, SettingsPanel and several modals.
 */

import { forwardRef, type HTMLAttributes } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';
import './Badge.css';

const badgeVariants = cva('ax-badge', {
  variants: {
    variant: {
      neutral: 'ax-badge--neutral',
      accent: 'ax-badge--accent',
      success: 'ax-badge--success',
      warning: 'ax-badge--warning',
      danger: 'ax-badge--danger',
    },
    size: {
      sm: 'ax-badge--sm',
      md: 'ax-badge--md',
    },
  },
  defaultVariants: {
    variant: 'neutral',
    size: 'md',
  },
});

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, size, children, ...props }, ref) => (
    <span ref={ref} className={cn(badgeVariants({ variant, size }), className)} {...props}>
      {children}
    </span>
  )
);

Badge.displayName = 'Badge';
