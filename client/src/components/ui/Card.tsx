/**
 * Card — thin wrapper over the global `.card` / `.glass` surface styles so the
 * informal `<div className="card">` pattern has a typed, consistent component.
 */

import { forwardRef, type HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Use the translucent glass surface instead of the solid card surface. */
  glass?: boolean;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, glass, children, ...props }, ref) => (
    <div ref={ref} className={cn(glass ? 'glass' : 'card', className)} {...props}>
      {children}
    </div>
  )
);

Card.displayName = 'Card';
