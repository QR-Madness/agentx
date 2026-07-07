/**
 * ChangePasswordModal — Lets the signed-in root user rotate their password.
 *
 * Backed by POST /api/auth/change-password (auth/views.py:auth_change_password).
 * Successful rotation server-side invalidates all other sessions but keeps
 * the current one, so we don't need to re-login here.
 */

import { useState, FormEvent } from 'react';
import { AlertCircle, CheckCircle, KeyRound } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import { PasswordField } from '../common/PasswordField';
// Reuse the auth-* form classes (.auth-form / .auth-error / .auth-field label).
import '../../pages/AuthPage.css';
import { Button } from '../ui';

interface ChangePasswordModalProps {
  onClose: () => void;
}

export function ChangePasswordModal({ onClose }: ChangePasswordModalProps) {
  const { changePassword, sessionInfo } = useAuth();

  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

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
          <div className="auth-error" style={{ background: 'var(--feedback-success-tint)', borderColor: 'var(--feedback-success-border)', color: 'var(--feedback-success)' }}>
            <CheckCircle size={16} />
            <span>Password changed. Other sessions have been signed out.</span>
          </div>
        )}

        <PasswordField
          id="old-password"
          label="Current Password"
          value={oldPassword}
          onChange={setOldPassword}
          autoComplete="current-password"
          disabled={submitting || success}
          autoFocus
        />

        <PasswordField
          id="new-password"
          label="New Password"
          value={newPassword}
          onChange={setNewPassword}
          autoComplete="new-password"
          minLength={8}
          disabled={submitting || success}
        />

        <PasswordField
          id="confirm-password"
          label="Confirm New Password"
          value={confirmPassword}
          onChange={setConfirmPassword}
          autoComplete="new-password"
          minLength={8}
          disabled={submitting || success}
        />

        <div style={{ display: 'flex', gap: 8, marginTop: 16, justifyContent: 'flex-end' }}>
          <Button
            type="button"
            variant="secondary"
            onClick={onClose}
            disabled={submitting}
          >
            Close
          </Button>
          <Button
            type="submit"
            variant="primary"
            disabled={submitting || success || !oldPassword || !newPassword || !confirmPassword}
          >
            {submitting ? 'Changing…' : 'Change Password'}
          </Button>
        </div>
      </form>
    </div>
  );
}
