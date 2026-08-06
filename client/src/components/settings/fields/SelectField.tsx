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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../../ui';
import { FieldShell, type FieldChromeProps } from './FieldShell';

export interface SelectFieldOption {
  value: string;
  label: ReactNode;
}

/** Stands in for `''` inside Radix, which rejects an empty item value. */
const EMPTY_VALUE = '__empty__';

const toRadix = (value: string) => (value === '' ? EMPTY_VALUE : value);
const fromRadix = (value: string) => (value === EMPTY_VALUE ? '' : value);

interface SelectFieldProps extends FieldChromeProps {
  label: ReactNode;
  value: string;
  options: SelectFieldOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  hint?: ReactNode;
  disabled?: boolean;
  /** Help/reset announcement text when `label` isn't a plain string. */
  labelText?: string;
}

export function SelectField({
  label, value, options, onChange, placeholder, hint, disabled,
  binding, onReset, badge, labelText,
}: SelectFieldProps) {
  return (
    <FieldShell
      label={label}
      labelText={labelText}
      hint={hint}
      binding={binding}
      onReset={onReset}
      badge={badge}
    >
      {({ id, describedBy }) => (
        <Select
          value={toRadix(value)}
          onValueChange={v => onChange(fromRadix(v))}
          disabled={disabled}
        >
          <SelectTrigger id={id} aria-describedby={describedBy}>
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
      )}
    </FieldShell>
  );
}
