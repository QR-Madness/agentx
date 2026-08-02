/**
 * NumberField — labelled numeric input on the Input primitive. `fallback`
 * mirrors the original `parseInt(...) || fallback` behavior for empty input.
 *
 * Integer by default. Pass a fractional `step` (e.g. 0.05 for a dollar amount)
 * to parse as a float.
 *
 * Bounds come from the explicit `min`/`max`/`step` props when given, and
 * otherwise from the manifest `binding` — so a range declared once server-side
 * reaches the input without being re-typed here.
 *
 * NOTE: because `fallback` applies via `||`, a typed 0 falls back too. Fields
 * where 0 is a real value ("0 = unlimited") should pass `fallback={0}`.
 */

import type { ReactNode } from 'react';
import { Input } from '../../ui';
import { FieldShell, type FieldChromeProps } from './FieldShell';

interface NumberFieldProps extends FieldChromeProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  /** Fractional values (e.g. 0.05) switch parsing to float. */
  step?: number;
  onChange: (value: number) => void;
  /** Value used when the field parses to NaN/0 (mirrors `parseInt(...) || fallback`). */
  fallback?: number;
  title?: string;
  hint?: ReactNode;
}

export function NumberField({
  label, value, min, max, step, onChange, fallback, title, hint,
  binding, onReset, badge,
}: NumberFieldProps) {
  const lo = min ?? binding?.min;
  const hi = max ?? binding?.max;
  const inc = step ?? binding?.step;
  const isDecimal = inc !== undefined && !Number.isInteger(inc);

  return (
    <FieldShell
      label={label}
      labelText={label}
      hint={hint}
      binding={binding}
      onReset={onReset}
      badge={badge}
    >
      {({ id, describedBy }) => (
        <Input
          id={id}
          aria-describedby={describedBy}
          type="number"
          value={value}
          min={lo}
          max={hi}
          step={inc}
          title={title}
          onChange={e => {
            const parsed = isDecimal ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
            onChange(fallback !== undefined ? (parsed || fallback) : parsed);
          }}
        />
      )}
    </FieldShell>
  );
}
