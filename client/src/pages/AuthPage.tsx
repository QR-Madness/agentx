/**
 * AuthPage — Authentication screen for login and initial password setup
 */

import { useState, FormEvent } from 'react';
import { Lock, Eye, EyeOff, Shield, AlertCircle, KeyRound, Loader2, WifiOff, CheckCircle2, RefreshCw } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useServer } from '../contexts/ServerContext';
import { ServerSelector } from '../components/ServerSelector';
import './AuthPage.css';

const LOGIN_USERNAME = 'root';

interface ConnectionStatusProps {
  state: 'connecting' | 'unreachable' | 'ready';
  error: string | null;
  serverName: string | null;
  serverUrl: string | null;
  onRetry: () => void;
}

function ConnectionStatus({ state, error, serverName, serverUrl, onRetry }: ConnectionStatusProps) {
  const target = serverName ? `${serverName} (${serverUrl ?? ''})` : (serverUrl ?? 'the server');

  if (state === 'connecting') {
    return (
      <div className="auth-connection-status auth-connection-status--pending">
        <Loader2 size={16} className="auth-connection-spinner" />
        <span>Connecting to {target}…</span>
      </div>
    );
  }

  if (state === 'unreachable') {
    return (
      <div className="auth-connection-status auth-connection-status--error">
        <WifiOff size={16} />
        <div className="auth-connection-status-body">
          <strong>Couldn't reach {target}</strong>
          {error && <span className="auth-connection-status-detail">{error}</span>}
          <span className="auth-connection-status-hint">
            Check the URL above, switch servers, or retry.
          </span>
        </div>
        <button type="button" className="auth-connection-retry" onClick={onRetry}>
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    );
  }

  return (
    <div className="auth-connection-status auth-connection-status--ok">
      <CheckCircle2 size={16} />
      <span>Connected to {target}</span>
    </div>
  );
}

export function AuthPage() {
  const {
    setupRequired,
    login,
    setupPassword,
    connectionState,
    connectionError,
    authRequired,
    checkAuthStatus,
  } = useAuth();
  const { activeServer } = useServer();

  // Form state
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
        const success = await login(LOGIN_USERNAME, password);
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
      <span className="auth-particle auth-particle--1" aria-hidden="true" />
      <span className="auth-particle auth-particle--2" aria-hidden="true" />
      <span className="auth-particle auth-particle--3" aria-hidden="true" />
      <span className="auth-particle auth-particle--4" aria-hidden="true" />
      <span className="auth-particle auth-particle--5" aria-hidden="true" />
      <div className="auth-container">
        <div className="auth-header">
          <div className="auth-logo">
            <Shield size={48} />
          </div>
          <h1 className="auth-title">
            {setupRequired ? 'Set Up AgentX' : 'Welcome to AgentX'}
          </h1>
          <p className="auth-subtitle">
            {setupRequired
              ? 'Create a password to secure your AgentX server'
              : connectionState === 'ready'
                ? (authRequired ? 'Sign in to continue' : 'Connected — entering workspace…')
                : 'Choose a server to continue'}
          </p>
        </div>

        <ServerSelector disabled={isSubmitting} />

        <ConnectionStatus
          state={connectionState}
          error={connectionError}
          serverName={activeServer?.name ?? null}
          serverUrl={activeServer?.url ?? null}
          onRetry={checkAuthStatus}
        />

        {authRequired && connectionState === 'ready' && (
        <form className="auth-form" onSubmit={handleSubmit}>
          {error && (
            <div className="auth-error">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {!setupRequired && (
            <div className="auth-field">
              <label>Username</label>
              <div className="auth-username-static">
                <KeyRound size={16} className="auth-username-icon" />
                <span>{LOGIN_USERNAME}</span>
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
        )}

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
