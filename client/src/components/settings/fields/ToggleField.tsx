/**
 * ToggleField — labelled checkbox on the Checkbox primitive, with an optional
 * status Badge and hint line. Replaces the `<label><input type="checkbox">`
 * patterns in the memory settings forms.
 *
 * The label sits beside the control here rather than above it, so this keeps
 * its own layout instead of using FieldShell — but it carries the same manifest
 * chrome (changed dot, reset, help) and wires its hint through
 * `aria-describedby`.
 */

import { type ReactNode, useId } from 'react';
import { RotateCcw } from 'lucide-react';
import { Checkbox, Label, Badge } from '../../ui';
import type { BadgeProps } from '../../ui';
import { SettingHelp } from '../SettingHelp';
import { SettingBadge } from '../SettingBadge';
import type { FieldChromeProps } from './FieldShell';

/** `badge` keeps its own free-form shape here — call sites pass arbitrary text
 *  ("Following Fast Utility · gpt-4o-mini"), not just the fixed vocabulary. */
interface ToggleFieldProps extends Omit<FieldChromeProps, 'badge'> {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  badge?: { text: string; variant?: BadgeProps['variant'] };
  hint?: ReactNode;
  title?: string;
  /** Help/reset announcement text when `label` isn't a plain string. */
  labelText?: string;
}

export function ToggleField({
  checked, onChange, label, badge, hint, title,
  binding, onReset, labelText,
}: ToggleFieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const name = labelText ?? (typeof label === 'string' ? label : 'this setting');
  const modified = Boolean(binding?.isModified);

  return (
    <div className="setting-row checkbox">
      <div className="flex items-center gap-2" title={title}>
        <Checkbox
          id={id}
          checked={checked}
          onCheckedChange={v => onChange(v === true)}
          aria-describedby={hint ? hintId : undefined}
        />
        <Label htmlFor={id} className="setting-label">
          {label}
          {badge && <Badge variant={badge.variant}>{badge.text}</Badge>}
        </Label>
        {modified && (
          <span
            className="setting-modified-dot"
            title="Changed from the default"
            aria-label="Changed from the default"
          />
        )}
        {binding?.tier === 'experimental' && <SettingBadge kind="experimental" />}
        <SettingHelp help={binding?.help} label={name} />
        {modified && onReset && (
          <button
            type="button"
            className="setting-reset-btn"
            onClick={onReset}
            title="Reset to default"
            aria-label={`Reset ${name} to default`}
          >
            <RotateCcw size={12} aria-hidden="true" />
          </button>
        )}
      </div>
      {hint && (
        <span className="setting-hint" id={hintId}>
          {hint}
        </span>
      )}
    </div>
  );
}
