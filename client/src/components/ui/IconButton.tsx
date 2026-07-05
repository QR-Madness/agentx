/**
 * IconButton — square icon-only control (design-kit `core/IconButton`).
 * Transparent at rest, hover lifts background (+ glow in glowing themes),
 * `active` shows the accent-tint selected state. `aria-label` is required —
 * an icon-only control has no other accessible name.
 */

import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';
import './IconButton.css';

const iconButtonVariants = cva(
  'ax-iconbtn focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
  {
    variants: {
      size: {
        md: '',
        sm: 'ax-iconbtn--sm',
        xs: 'ax-iconbtn--xs',
      },
      tone: {
        default: '',
        danger: 'ax-iconbtn--danger',
        accent: 'ax-iconbtn--accent',
      },
      active: {
        true: 'ax-iconbtn--active',
      },
    },
    defaultVariants: { size: 'md', tone: 'default' },
  }
);

export interface IconButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof iconButtonVariants> {
  'aria-label': string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ className, size, tone, active, type = 'button', ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={cn(iconButtonVariants({ size, tone, active }), className)}
      {...props}
    />
  )
);

IconButton.displayName = 'IconButton';
