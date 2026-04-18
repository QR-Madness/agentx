/**
 * AuthPage — Authentication screen for login and initial password setup
 */

import { useState, FormEvent } from 'react';
import { Lock, KeyRound, Eye, EyeOff, Shield, AlertCircle } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import './AuthPage.css';

export function AuthPage() {
  const { setupRequired, login, setupPassword } = useAuth();

  // Form state
  const [username, setUsername] = useState('root');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      if (setupRequired) {
        // Initial password setup
        if (password.length < 8) {
          setError('Password must be at least 8 characters');
          return;
        }
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          return;
        }

        const success = await setupPassword(password, confirmPassword);
        if (!success) {
          setError('Failed to set up password. Please try again.');
          return;
        }

        // After setup, clear form and let user login
        setPassword('');
        setConfirmPassword('');
        // Auth context will update setupRequired
      } else {
        // Login
        const success = await login(username, password);
        if (!success) {
          setError('Invalid username or password');
        }
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <div className="auth-logo">
            <Shield size={48} />
          </div>
          <h1 className="auth-title">
            {setupRequired ? 'Set Up AgentX' : 'Welcome Back'}
          </h1>
          <p className="auth-subtitle">
            {setupRequired
              ? 'Create a password to secure your AgentX server'
              : 'Sign in to continue to AgentX'}
          </p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          {error && (
            <div className="auth-error">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {!setupRequired && (
            <div className="auth-field">
              <label htmlFor="username">Username</label>
              <div className="auth-input-wrapper">
                <KeyRound size={18} className="auth-input-icon" />
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="root"
                  autoComplete="username"
                  disabled={isSubmitting}
                />
              </div>
            </div>
          )}

          <div className="auth-field">
            <label htmlFor="password">
              {setupRequired ? 'New Password' : 'Password'}
            </label>
            <div className="auth-input-wrapper">
              <Lock size={18} className="auth-input-icon" />
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={setupRequired ? 'Enter new password (min 8 characters)' : 'Enter password'}
                autoComplete={setupRequired ? 'new-password' : 'current-password'}
                disabled={isSubmitting}
                minLength={8}
              />
              <button
                type="button"
                className="auth-toggle-password"
                onClick={() => setShowPassword(!showPassword)}
                tabIndex={-1}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          {setupRequired && (
            <div className="auth-field">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <div className="auth-input-wrapper">
                <Lock size={18} className="auth-input-icon" />
                <input
                  id="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm password"
                  autoComplete="new-password"
                  disabled={isSubmitting}
                  minLength={8}
                />
                <button
                  type="button"
                  className="auth-toggle-password"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  tabIndex={-1}
                >
                  {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>
          )}

          <button
            type="submit"
            className="auth-submit button-primary"
            disabled={isSubmitting}
          >
            {isSubmitting
              ? 'Please wait...'
              : setupRequired
                ? 'Set Password'
                : 'Sign In'}
          </button>
        </form>

        {setupRequired && (
          <div className="auth-footer">
            <p className="auth-hint">
              This password will be used to access your AgentX server.
              Make sure to remember it!
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
