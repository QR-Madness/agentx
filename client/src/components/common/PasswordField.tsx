/**
 * PasswordField — a password input on the shared field-kit chrome
 * (`ax-fieldwrap`/`ax-inputwrap`: sunken bg + border + focus glow) with a
 * leading lock, a trailing reveal `IconButton`, and a built-in Caps-Lock warning.
 *
 * Replaces the hand-rolled `.auth-input-wrapper` markup that AuthPage and
 * ChangePasswordModal each duplicated, so password fields look identical
 * everywhere and pick up the same field chrome as the rest of the app.
 */

import { useState, type ReactNode } from 'react';
import { Lock, Eye, EyeOff, AlertTriangle } from 'lucide-react';
import { IconButton } from '../ui/IconButton';
import { cn } from '../../lib/utils';
import './PasswordField.css';

export interface PasswordFieldProps {
  id: string;
  label?: ReactNode;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  autoComplete?: string;
  minLength?: number;
  disabled?: boolean;
  autoFocus?: boolean;
  error?: boolean;
}

export function PasswordField({
  id,
  label,
  value,
  onChange,
  placeholder,
  autoComplete,
  minLength,
  disabled,
  autoFocus,
  error,
}: PasswordFieldProps) {
  const [show, setShow] = useState(false);
  const [capsLock, setCapsLock] = useState(false);

  const syncCapsLock = (e: React.KeyboardEvent<HTMLInputElement>) => {
    setCapsLock(e.getModifierState?.('CapsLock') ?? false);
  };

  return (
    <div className="pw-field">
      {label && <label htmlFor={id}>{label}</label>}
      <span className={cn('ax-fieldwrap ax-inputwrap', error && 'ax-field--error')}>
        <span className="ax-inputwrap__icon">
          <Lock size={16} />
        </span>
        <input
          id={id}
          className="ax-inputwrap__input"
          type={show ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyUp={syncCapsLock}
          onKeyDown={syncCapsLock}
          onBlur={() => setCapsLock(false)}
          placeholder={placeholder}
          autoComplete={autoComplete}
          minLength={minLength}
          disabled={disabled}
          autoFocus={autoFocus}
        />
        <IconButton
          size="sm"
          className="pw-reveal"
          aria-label={show ? 'Hide password' : 'Show password'}
          onClick={() => setShow((s) => !s)}
          tabIndex={-1}
          disabled={disabled}
        >
          {show ? <EyeOff size={16} /> : <Eye size={16} />}
        </IconButton>
      </span>
      {capsLock && (
        <span className="pw-capslock" role="status">
          <AlertTriangle size={12} /> Caps Lock is on
        </span>
      )}
    </div>
  );
}
