/**
 * ServerSelector — Host picker with add/switch and gateway-token capture.
 *
 * Used by the connect screen (`AuthPage`) and the TopBar quick-switch overlay.
 * Switching is handled by `ServerContext.switchServer`, which hard-reloads the
 * window so no in-flight requests, SSE streams, or context state from the
 * previous host can leak into the new one.
 */

import { useState } from 'react';
import { Server, ChevronDown, ChevronUp, Plus, X } from 'lucide-react';
import { useServer } from '../contexts/ServerContext';
import '../pages/AuthPage.css';

interface ServerSelectorProps {
  disabled?: boolean;
  /** When true, render the dropdown expanded by default (e.g. for the TopBar overlay). */
  defaultOpen?: boolean;
  onSwitch?: () => void;
}

export function ServerSelector({ disabled = false, defaultOpen = false, onSwitch }: ServerSelectorProps) {
  const { servers, activeServer, switchServer, addNewServer } = useServer();
  const [open, setOpen] = useState(defaultOpen);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newGatewayToken, setNewGatewayToken] = useState('');

  const handleSelect = (id: string) => {
    onSwitch?.();
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
    onSwitch?.();
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
