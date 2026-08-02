/**
 * SelectField — labelled enum picker on the Select primitive. Replaces the
 * hand-rolled `<select>` / ghost-trigger patterns in the settings sections
 * (which read washed-out — see the field-chrome rule).
 *
 * Radix reserves `''` as "cleared" and throws on an item that carries it, but
 * settings routinely need an explicit "inherit / provider default" choice whose
 * stored value IS `''`. That value is swapped for a sentinel at the Radix
 * boundary only: callers keep passing and receiving `''`.
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

/** Stands in for `''` inside Radix, which rejects an empty item value. */
const EMPTY_VALUE = '__empty__';

const toRadix = (value: string) => (value === '' ? EMPTY_VALUE : value);
const fromRadix = (value: string) => (value === EMPTY_VALUE ? '' : value);

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
      <Select
        value={toRadix(value)}
        onValueChange={v => onChange(fromRadix(v))}
        disabled={disabled}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={toRadix(opt.value)}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
