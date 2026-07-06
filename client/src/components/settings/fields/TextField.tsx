/**
 * TextField — labelled text input on the Input primitive (`ax-field` chrome).
 * The settings-kit counterpart of NumberField for free-text values (URLs,
 * names, API hosts). Secrets keep their own explicit-Save flows.
 */

import type { ReactNode } from 'react';
import { Label, Input } from '../../ui';

interface TextFieldProps {
  label: ReactNode;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'password' | 'url' | 'email';
  hint?: ReactNode;
  title?: string;
  disabled?: boolean;
}

export function TextField({
  label, value, onChange, placeholder, type = 'text', hint, title, disabled,
}: TextFieldProps) {
  return (
    <div className="setting-row">
      <Label>{label}</Label>
      <Input
        type={type}
        value={value}
        placeholder={placeholder}
        title={title}
        disabled={disabled}
        onChange={e => onChange(e.target.value)}
      />
      {hint && <span className="setting-hint">{hint}</span>}
    </div>
  );
}
