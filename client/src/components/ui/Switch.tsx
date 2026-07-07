/**
 * Switch — boolean toggle built on Radix. Styling lives in Switch.css (explicit
 * box model so the global button reset can't misalign the thumb). Uses the
 * accent token for the on state, with a soft glow.
 */

import * as React from 'react';
import * as SwitchPrimitive from '@radix-ui/react-switch';
import { cn } from '../../lib/utils';
import './Switch.css';

export const Switch = React.forwardRef<
  React.ComponentRef<typeof SwitchPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitive.Root ref={ref} className={cn('ax-switch', className)} {...props}>
    <SwitchPrimitive.Thumb className="ax-switch__thumb" />
  </SwitchPrimitive.Root>
));
Switch.displayName = SwitchPrimitive.Root.displayName;
