/**
 * SliderField — labelled range control with a value readout, on the Slider
 * primitive. Replaces the raw `<input type="range">` + `.setting-value` markup.
 *
 * Bounds stay required here (a slider without them is meaningless); a manifest
 * `binding` supplies the default that reset restores and the help beside the
 * label.
 */

import type { ReactNode } from 'react';
import { Slider } from '../../ui';
import { FieldShell, type FieldChromeProps } from './FieldShell';

interface SliderFieldProps extends FieldChromeProps {
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
  binding, onReset, badge,
}: SliderFieldProps) {
  return (
    <FieldShell
      label={label}
      labelText={label}
      hint={hint}
      binding={binding}
      onReset={onReset}
      badge={badge}
      labelledControl={false}
    >
      {({ describedBy }) => (
        <div className="setting-input-group">
          <Slider
            aria-describedby={describedBy}
            value={[value]}
            min={min}
            max={max}
            step={step}
            onValueChange={([v]) => onChange(v)}
            aria-label={label}
          />
          <span className="setting-value">{format(value)}</span>
        </div>
      )}
    </FieldShell>
  );
}
