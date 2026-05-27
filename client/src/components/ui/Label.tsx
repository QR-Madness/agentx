/**
 * Label — accessible form label built on Radix. Pairs with Input/Switch/etc.
 */

import * as React from 'react';
import * as LabelPrimitive from '@radix-ui/react-label';
import { cn } from '../../lib/utils';

export const Label = React.forwardRef<
  React.ComponentRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root>
>(({ className, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    className={cn(
      'text-sm font-medium text-fg select-none',
      'peer-disabled:cursor-not-allowed peer-disabled:opacity-60',
      className
    )}
    {...props}
  />
));
Label.displayName = LabelPrimitive.Root.displayName;
