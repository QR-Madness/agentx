/**
 * ThinkingModeMenu — the unified thinking-mode radio menu (patterns + Research),
 * wrapping any trigger (the composer's Mode chip, the Relay's Mode tile).
 *
 * Deliberately an opaque Radix DropdownMenu (no glass) — see the WebKitGTK
 * paint-cost note in ui/DropdownMenu.tsx.
 */

import type { ReactNode } from 'react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '../ui/DropdownMenu';
import type { ThinkingModeOption } from '../../lib/thinkingModes';

interface ThinkingModeMenuProps {
  value: string;
  onChange: (mode: string) => void;
  options: ThinkingModeOption[];
  children: ReactNode;
  align?: 'start' | 'center' | 'end';
}

export function ThinkingModeMenu({
  value,
  onChange,
  options,
  children,
  align = 'start',
}: ThinkingModeMenuProps) {
  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align={align} className="min-w-[15rem]">
        <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
          {options.map(opt => (
            <DropdownMenuRadioItem key={opt.value} value={opt.value} title={opt.hint}>
              <span className="flex flex-col items-start">
                <span>{opt.label}</span>
                <span className="text-2xs text-fg-muted">{opt.hint}</span>
              </span>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
