/**
 * TextField — labelled text input on the Input primitive (`ax-field` chrome).
 * The settings-kit counterpart of NumberField for free-text values (URLs,
 * names, API hosts). Secrets keep their own explicit-Save flows.
 */

import type { ReactNode } from 'react';
import { Input } from '../../ui';
import { FieldShell, type FieldChromeProps } from './FieldShell';

interface TextFieldProps extends FieldChromeProps {
  label: ReactNode;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'password' | 'url' | 'email';
  hint?: ReactNode;
  title?: string;
  disabled?: boolean;
  /** Help/reset announcement text when `label` isn't a plain string. */
  labelText?: string;
}

export function TextField({
  label, value, onChange, placeholder, type = 'text', hint, title, disabled,
  binding, onReset, badge, labelText,
}: TextFieldProps) {
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
        <Input
          id={id}
          aria-describedby={describedBy}
          type={type}
          value={value}
          placeholder={placeholder}
          title={title}
          disabled={disabled}
          onChange={e => onChange(e.target.value)}
        />
      )}
    </FieldShell>
  );
}
