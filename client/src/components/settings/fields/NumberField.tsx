/**
 * NumberField — labelled integer input on the Input primitive. `fallback`
 * mirrors the original `parseInt(...) || fallback` behavior for empty input.
 */

import type { ReactNode } from 'react';
import { Label, Input } from '../../ui';

interface NumberFieldProps {
  label: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  /** Value used when the field parses to NaN/0 (mirrors `parseInt(...) || fallback`). */
  fallback?: number;
  title?: string;
  hint?: ReactNode;
}

export function NumberField({ label, value, min, max, onChange, fallback, title, hint }: NumberFieldProps) {
  return (
    <div className="setting-row">
      <Label>{label}</Label>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        title={title}
        onChange={e => {
          const parsed = parseInt(e.target.value, 10);
          onChange(fallback !== undefined ? (parsed || fallback) : parsed);
        }}
      />
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
