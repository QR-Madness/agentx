/**
 * SelectField — labelled enum picker on the Select primitive. Replaces the
 * hand-rolled `<select>` / ghost-trigger patterns in the settings sections
 * (which read washed-out — see the field-chrome rule).
 */

import type { ReactNode } from 'react';
import {
  Label,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../../ui';

export interface SelectFieldOption {
  value: string;
  label: ReactNode;
}

interface SelectFieldProps {
  label: ReactNode;
  value: string;
  options: SelectFieldOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: ReactNode;
  disabled?: boolean;
}

export function SelectField({
  label, value, options, onChange, placeholder, hint, disabled,
}: SelectFieldProps) {
  return (
    <div className="setting-row">
      <Label>{label}</Label>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
