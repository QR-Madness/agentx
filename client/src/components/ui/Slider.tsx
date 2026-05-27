/**
 * Slider — single-value range control built on Radix. Token-styled track/range/
 * thumb for the cosmic theme; the accent token fills the active range.
 */

import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { cn } from '../../lib/utils';

export const Slider = React.forwardRef<
  React.ComponentRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, 'aria-label': ariaLabel, 'aria-labelledby': ariaLabelledby, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      'relative flex w-full touch-none select-none items-center',
      'data-[disabled]:cursor-not-allowed data-[disabled]:opacity-50',
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-[var(--surface-overlay)]">
      <SliderPrimitive.Range className="absolute h-full bg-accent" />
    </SliderPrimitive.Track>
    {/* The focusable thumb carries role="slider"; label it for a11y. */}
    <SliderPrimitive.Thumb
      aria-label={ariaLabel}
      aria-labelledby={ariaLabelledby}
      className={cn(
        'block h-4 w-4 rounded-full border border-accent bg-white shadow-sm transition-colors',
        'outline-none focus-visible:ring-2 focus-visible:ring-accent',
        'disabled:pointer-events-none'
      )}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;
