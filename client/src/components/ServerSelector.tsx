/**
 * ServerSelector — Host picker with add/switch and gateway-token capture.
 *
 * Used by the connect screen (`AuthPage`) and the TopBar quick-switch overlay.
 * Switching is handled by `ServerContext.switchServer`, which hard-reloads the
 * window so no in-flight requests, SSE streams, or context state from the
 * previous host can leak into the new one.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Server, ChevronDown, ChevronUp, Plus, X, Pencil, Trash2, Check, Share2 } from 'lucide-react';
import { useServer } from '../contexts/ServerContext';
import { useNotify } from '../contexts/NotificationContext';
import { buildConnectUrl } from '../lib/connectionString';
import type { ServerConfig } from '../lib/storage';
import '../pages/AuthPage.css';

interface ServerSelectorProps {
  disabled?: boolean;
  /** When true, render the dropdown expanded by default (e.g. for the TopBar overlay). */
  defaultOpen?: boolean;
  onSwitch?: () => void;
}

export function ServerSelector({ disabled = false, defaultOpen = false, onSwitch }: ServerSelectorProps) {
  const { servers, activeServer, switchServer, addNewServer, updateServerConfig, deleteServer } = useServer();
  const { notifySuccess, notify } = useNotify();
  const [open, setOpen] = useState(defaultOpen);
  const [dropdownRect, setDropdownRect] = useState<{ top: number; left: number; width: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newGatewayToken, setNewGatewayToken] = useState('');

  // Per-row edit + delete-confirm state (keyed by server id).
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [editUrl, setEditUrl] = useState('');
  const [editGatewayToken, setEditGatewayToken] = useState('');
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);

  const openDropdown = useCallback(() => {
    if (disabled || !triggerRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setDropdownRect({ top: rect.bottom + 4, left: rect.left, width: rect.width });
    setOpen(true);
  }, [disabled]);

  const closeDropdown = useCallback(() => {
    setOpen(false);
    setDropdownRect(null);
  }, []);

  // Close on scroll or orientation change.
  // Deliberately ignore height-only resize (Android keyboard opening) — the dropdown
  // is position:fixed via portal so it doesn't shift, and closing on keyboard-open
  // creates an unbreakable collapse loop on real devices.
  useEffect(() => {
    if (!open) return;
    let lastWidth = window.innerWidth;
    const onScroll = (e: Event) => {
      // Only a page/container scroll should dismiss the dropdown. Ignore scrolls
      // that originate *inside* it — notably a long gateway token scrolling its
      // password field horizontally as you type (the capture-phase listener also
      // catches element scrolls, which would otherwise slam the form shut mid-edit).
      const target = e.target as HTMLElement | null;
      if (target?.closest?.('.auth-server-dropdown')) return;
      closeDropdown();
    };
    const onResize = () => {
      const w = window.innerWidth;
      if (w !== lastWidth) { lastWidth = w; closeDropdown(); }
    };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  }, [open, closeDropdown]);

  const handleSelect = (id: string) => {
    onSwitch?.();
    switchServer(id);
    closeDropdown();
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
    closeDropdown();
  };

  const handleEditStart = (id: string) => {
    const server = servers.find(s => s.id === id);
    if (!server) return;
    setConfirmingDeleteId(null);
    setEditingId(id);
    setEditName(server.name);
    setEditUrl(server.url);
    setEditGatewayToken(server.gatewayToken ?? '');
  };

  const handleEditCancel = () => {
    setEditingId(null);
    setEditName('');
    setEditUrl('');
    setEditGatewayToken('');
  };

  const handleEditSave = () => {
    if (!editingId || !editName.trim() || !editUrl.trim()) return;
    updateServerConfig(editingId, {
      name: editName.trim(),
      url: editUrl.trim(),
      gatewayToken: editGatewayToken.trim() || undefined,
    });
    handleEditCancel();
  };

  const handleDelete = (id: string) => {
    deleteServer(id);
    setConfirmingDeleteId(null);
  };

  // Copy a shareable connection link (server address + gateway token) so a
  // recipient can open it and only enter the password. Never includes the
  // per-user auth token — that's set by signing in on the other end.
  const handleShare = async (s: ServerConfig) => {
    const link = buildConnectUrl({ url: s.url, gatewayToken: s.gatewayToken, name: s.name });
    try {
      await navigator.clipboard.writeText(link);
      notifySuccess('Connection link copied — share it so they can sign in.', 'Link copied');
    } catch {
      // Clipboard blocked (e.g. non-secure context) — surface the link to copy manually.
      notify({ kind: 'info', title: 'Copy this connection link', message: link, duration: 0 });
    }
  };

  return (
    <div className={`auth-server-selector${disabled ? ' auth-server-selector--disabled' : ''}`}>
      <label className="auth-server-label">
        <Server size={13} />
        Server
      </label>

      <button
        ref={triggerRef}
        type="button"
        className="auth-server-current"
        onClick={() => (open ? closeDropdown() : openDropdown())}
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

      {open && dropdownRect && createPortal(
        <div
          className="auth-server-dropdown"
          role="listbox"
          style={{
            position: 'fixed',
            top: dropdownRect.top,
            left: dropdownRect.left,
            width: dropdownRect.width,
            zIndex: 9999,
          }}
        >
          {servers.map(s => {
            if (editingId === s.id) {
              return (
                <div key={s.id} className="auth-server-add-form auth-server-edit-form">
                  <input
                    type="text"
                    className="auth-server-add-input"
                    placeholder="Server name"
                    value={editName}
                    onChange={e => setEditName(e.target.value)}
                    autoFocus
                  />
                  <input
                    type="text"
                    className="auth-server-add-input"
                    placeholder="URL (e.g. http://localhost:12319)"
                    value={editUrl}
                    onChange={e => setEditUrl(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleEditSave()}
                  />
                  <input
                    type="password"
                    autoComplete="off"
                    className="auth-server-add-input"
                    placeholder="Gateway token (optional)"
                    value={editGatewayToken}
                    onChange={e => setEditGatewayToken(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleEditSave()}
                  />
                  <div className="auth-server-add-actions">
                    <button type="button" className="auth-server-add-cancel" onClick={handleEditCancel}>
                      <X size={13} /> Cancel
                    </button>
                    <button
                      type="button"
                      className="auth-server-add-confirm"
                      onClick={handleEditSave}
                      disabled={!editName.trim() || !editUrl.trim()}
                    >
                      <Check size={13} /> Save
                    </button>
                  </div>
                </div>
              );
            }

            return (
              <div
                key={s.id}
                className={`auth-server-option${s.id === activeServer?.id ? ' auth-server-option--active' : ''}`}
              >
                <button
                  type="button"
                  role="option"
                  aria-selected={s.id === activeServer?.id}
                  className="auth-server-option-select"
                  onClick={() => handleSelect(s.id)}
                >
                  <Server size={14} />
                  <span className="auth-server-option-info">
                    <span className="auth-server-option-name">{s.name}</span>
                    <span className="auth-server-option-url">{s.url}</span>
                  </span>
                </button>

                {confirmingDeleteId === s.id ? (
                  <div className="auth-server-option-actions">
                    <button
                      type="button"
                      className="auth-server-action-btn auth-server-action-btn--danger"
                      title="Confirm delete"
                      onClick={() => handleDelete(s.id)}
                    >
                      <Check size={14} />
                    </button>
                    <button
                      type="button"
                      className="auth-server-action-btn"
                      title="Cancel"
                      onClick={() => setConfirmingDeleteId(null)}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <div className="auth-server-option-actions">
                    <button
                      type="button"
                      className="auth-server-action-btn"
                      title="Copy connection link"
                      onClick={() => handleShare(s)}
                    >
                      <Share2 size={14} />
                    </button>
                    <button
                      type="button"
                      className="auth-server-action-btn"
                      title="Edit server"
                      onClick={() => handleEditStart(s.id)}
                    >
                      <Pencil size={14} />
                    </button>
                    <button
                      type="button"
                      className="auth-server-action-btn auth-server-action-btn--danger"
                      title="Delete server"
                      onClick={() => setConfirmingDeleteId(s.id)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}

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
        </div>,
        document.body,
      )}
    </div>
  );
}
