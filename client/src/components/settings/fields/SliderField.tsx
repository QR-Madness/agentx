/**
 * SliderField — labelled range control with a value readout, on the Slider
 * primitive. Replaces the raw `<input type="range">` + `.setting-value` markup.
 */

import type { ReactNode } from 'react';
import { Label, Slider } from '../../ui';

interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  /** Format the readout (default: 2 decimals). */
  format?: (value: number) => string;
  hint?: ReactNode;
}

export function SliderField({
  label, value, min, max, step, onChange, format = v => v.toFixed(2), hint,
}: SliderFieldProps) {
  return (
    <div className="setting-row">
      <Label>{label}</Label>
      <div className="setting-input-group">
        <Slider
          value={[value]}
          min={min}
          max={max}
          step={step}
          onValueChange={([v]) => onChange(v)}
          aria-label={label}
        />
        <span className="setting-value">{format(value)}</span>
      </div>
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
