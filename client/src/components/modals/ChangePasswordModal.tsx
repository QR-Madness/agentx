/**
 * ChangePasswordModal — Lets the signed-in root user rotate their password.
 *
 * Backed by POST /api/auth/change-password (auth/views.py:auth_change_password).
 * Successful rotation server-side invalidates all other sessions but keeps
 * the current one, so we don't need to re-login here.
 */

import { useState, FormEvent } from 'react';
import { Lock, Eye, EyeOff, AlertCircle, CheckCircle, KeyRound } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
// Reuse the existing auth-* class library (input wrapper, error pill, etc.).
import '../../pages/AuthPage.css';

interface ChangePasswordModalProps {
  onClose: () => void;
}

export function ChangePasswordModal({ onClose }: ChangePasswordModalProps) {
  const { changePassword, sessionInfo } = useAuth();

  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showOld, setShowOld] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (newPassword.length < 8) {
      setError('New password must be at least 8 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    if (newPassword === oldPassword) {
      setError('New password must differ from the current password');
      return;
    }

    setSubmitting(true);
    try {
      const ok = await changePassword(oldPassword, newPassword);
      if (!ok) {
        setError('Failed to change password — check your current password');
        return;
      }
      setSuccess(true);
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
      // Auto-close shortly after success so the user sees the confirmation.
      setTimeout(onClose, 1500);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-content-wrapper" style={{ padding: 24, maxWidth: 480 }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <KeyRound size={22} />
        <div>
          <h2 style={{ margin: 0 }}>Change Password</h2>
          {sessionInfo?.username && (
            <p style={{ margin: '4px 0 0', opacity: 0.7, fontSize: 13 }}>
              Signed in as <strong>{sessionInfo.username}</strong>
            </p>
          )}
        </div>
      </header>

      <form onSubmit={handleSubmit} className="auth-form">
        {error && (
          <div className="auth-error">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="auth-error" style={{ background: 'rgba(34,197,94,0.1)', borderColor: 'rgba(34,197,94,0.3)', color: 'rgb(34,197,94)' }}>
            <CheckCircle size={16} />
            <span>Password changed. Other sessions have been signed out.</span>
          </div>
        )}

        <PasswordField
          id="old-password"
          label="Current Password"
          value={oldPassword}
          onChange={setOldPassword}
          show={showOld}
          onToggleShow={() => setShowOld(s => !s)}
          autoComplete="current-password"
          disabled={submitting || success}
        />

        <PasswordField
          id="new-password"
          label="New Password"
          value={newPassword}
          onChange={setNewPassword}
          show={showNew}
          onToggleShow={() => setShowNew(s => !s)}
          autoComplete="new-password"
          minLength={8}
          disabled={submitting || success}
        />

        <PasswordField
          id="confirm-password"
          label="Confirm New Password"
          value={confirmPassword}
          onChange={setConfirmPassword}
          show={showConfirm}
          onToggleShow={() => setShowConfirm(s => !s)}
          autoComplete="new-password"
          minLength={8}
          disabled={submitting || success}
        />

        <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'flex-end' }}>
          <button
            type="button"
            className="button-secondary"
            onClick={onClose}
            disabled={submitting}
          >
            Close
          </button>
          <button
            type="submit"
            className="button-primary"
            disabled={submitting || success || !oldPassword || !newPassword || !confirmPassword}
          >
            {submitting ? 'Changing…' : 'Change Password'}
          </button>
        </div>
      </form>
    </div>
  );
}

interface PasswordFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  show: boolean;
  onToggleShow: () => void;
  autoComplete: string;
  minLength?: number;
  disabled?: boolean;
}

function PasswordField({
  id, label, value, onChange, show, onToggleShow, autoComplete, minLength, disabled,
}: PasswordFieldProps) {
  return (
    <div className="auth-field">
      <label htmlFor={id}>{label}</label>
      <div className="auth-input-wrapper">
        <Lock size={18} className="auth-input-icon" />
        <input
          id={id}
          type={show ? 'text' : 'password'}
          value={value}
          onChange={e => onChange(e.target.value)}
          autoComplete={autoComplete}
          minLength={minLength}
          disabled={disabled}
        />
        <button
          type="button"
          className="auth-toggle-password"
          onClick={onToggleShow}
          tabIndex={-1}
        >
          {show ? <EyeOff size={18} /> : <Eye size={18} />}
        </button>
      </div>
    </div>
  );
}
