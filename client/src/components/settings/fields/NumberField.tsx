/**
 * NumberField — labelled numeric input on the Input primitive. `fallback`
 * mirrors the original `parseInt(...) || fallback` behavior for empty input.
 *
 * Integer by default. Pass a fractional `step` (e.g. 0.05 for a dollar amount)
 * to parse as a float.
 *
 * NOTE: because `fallback` applies via `||`, a typed 0 falls back too. Fields
 * where 0 is a real value ("0 = unlimited") should pass `fallback={0}`.
 */

import type { ReactNode } from 'react';
import { Label, Input } from '../../ui';

interface NumberFieldProps {
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
}: NumberFieldProps) {
  const isDecimal = step !== undefined && !Number.isInteger(step);
  return (
    <div className="setting-row">
      <Label>{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        title={title}
        onChange={e => {
          const parsed = isDecimal ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
          onChange(fallback !== undefined ? (parsed || fallback) : parsed);
        }}
      />
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
