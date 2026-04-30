/**
 * AuthPage — Authentication screen for login and initial password setup
 */

import { useState, FormEvent } from 'react';
import { Lock, Eye, EyeOff, Shield, AlertCircle, Server, ChevronDown, ChevronUp, KeyRound, Plus, X } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useServer } from '../contexts/ServerContext';
import './AuthPage.css';

const LOGIN_USERNAME = 'root';

function ServerSelector({ disabled }: { disabled: boolean }) {
  const { servers, activeServer, switchServer, addNewServer } = useServer();
  const [open, setOpen] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newGatewayToken, setNewGatewayToken] = useState('');

  const handleSelect = (id: string) => {
    switchServer(id);
    setOpen(false);
  };

  const handleAdd = () => {
    if (!newName.trim() || !newUrl.trim()) return;
    const server = addNewServer(
      newName.trim(),
      newUrl.trim(),
      newGatewayToken.trim() || undefined,
    );
    switchServer(server.id);
    setNewName('');
    setNewUrl('');
    setNewGatewayToken('');
    setShowAddForm(false);
    setOpen(false);
  };

  return (
    <div className={`auth-server-selector${disabled ? ' auth-server-selector--disabled' : ''}`}>
      <label className="auth-server-label">
        <Server size={13} />
        Server
      </label>

      <button
        type="button"
        className="auth-server-current"
        onClick={() => !disabled && setOpen(o => !o)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <Server size={16} className="auth-server-icon" />
        <span className="auth-server-current-info">
          <span className="auth-server-current-name">{activeServer?.name ?? 'No server'}</span>
          <span className="auth-server-current-url">{activeServer?.url ?? ''}</span>
        </span>
        {open ? <ChevronUp size={15} className="auth-server-chevron" /> : <ChevronDown size={15} className="auth-server-chevron" />}
      </button>

      {open && (
        <div className="auth-server-dropdown" role="listbox">
          {servers.map(s => (
            <button
              key={s.id}
              type="button"
              role="option"
              aria-selected={s.id === activeServer?.id}
              className={`auth-server-option${s.id === activeServer?.id ? ' auth-server-option--active' : ''}`}
              onClick={() => handleSelect(s.id)}
            >
              <Server size={14} />
              <span className="auth-server-option-info">
                <span className="auth-server-option-name">{s.name}</span>
                <span className="auth-server-option-url">{s.url}</span>
              </span>
            </button>
          ))}

          {showAddForm ? (
            <div className="auth-server-add-form">
              <input
                type="text"
                className="auth-server-add-input"
                placeholder="Server name"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                autoFocus
              />
              <input
                type="text"
                className="auth-server-add-input"
                placeholder="URL (e.g. http://localhost:12319)"
                value={newUrl}
                onChange={e => setNewUrl(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleAdd()}
              />
              <input
                type="password"
                autoComplete="off"
                className="auth-server-add-input"
                placeholder="Gateway token (optional)"
                value={newGatewayToken}
                onChange={e => setNewGatewayToken(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleAdd()}
              />
              <div className="auth-server-add-actions">
                <button
                  type="button"
                  className="auth-server-add-cancel"
                  onClick={() => {
                    setShowAddForm(false);
                    setNewName('');
                    setNewUrl('');
                    setNewGatewayToken('');
                  }}
                >
                  <X size={13} /> Cancel
                </button>
                <button
                  type="button"
                  className="auth-server-add-confirm"
                  onClick={handleAdd}
                  disabled={!newName.trim() || !newUrl.trim()}
                >
                  <Plus size={13} /> Add
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className="auth-server-add-btn"
              onClick={() => setShowAddForm(true)}
            >
              <Plus size={14} /> Add Server
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function AuthPage() {
  const { setupRequired, login, setupPassword } = useAuth();

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
            {setupRequired ? 'Set Up AgentX' : 'Welcome Back'}
          </h1>
          <p className="auth-subtitle">
            {setupRequired
              ? 'Create a password to secure your AgentX server'
              : 'Sign in to continue to AgentX'}
          </p>
        </div>

        <ServerSelector disabled={isSubmitting} />

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
