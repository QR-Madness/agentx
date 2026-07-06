/**
 * ToggleField — labelled checkbox on the Checkbox primitive, with an optional
 * status Badge and hint line. Replaces the `<label><input type="checkbox">`
 * patterns in the memory settings forms.
 */

import { type ReactNode, useId } from 'react';
import { Checkbox, Label, Badge } from '../../ui';
import type { BadgeProps } from '../../ui';

interface ToggleFieldProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  badge?: { text: string; variant?: BadgeProps['variant'] };
  hint?: ReactNode;
  title?: string;
}

export function ToggleField({ checked, onChange, label, badge, hint, title }: ToggleFieldProps) {
  const id = useId();
  return (
    <div className="setting-row checkbox">
      <div className="flex items-center gap-2" title={title}>
        <Checkbox id={id} checked={checked} onCheckedChange={v => onChange(v === true)} />
        <Label htmlFor={id} className="setting-label">
          {label}
          {badge && <Badge variant={badge.variant}>{badge.text}</Badge>}
        </Label>
      </div>
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
