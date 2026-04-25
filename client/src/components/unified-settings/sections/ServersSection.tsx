/**
 * ServersSection — Backend server management
 * Extracted from SettingsPanel.tsx lines 311-388
 */

import { useState } from 'react';
import {
  Server,
  Plus,
  Check,
  ExternalLink,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { ServerConfig } from '../../../lib/storage';

export default function ServersSection() {
  const {
    servers,
    activeServer,
    switchServer,
    deleteServer,
    addNewServer
  } = useServer();

  const [showNewServer, setShowNewServer] = useState(false);
  const [newServerName, setNewServerName] = useState('');
  const [newServerUrl, setNewServerUrl] = useState('');

  const handleAddServer = () => {
    if (newServerName.trim() && newServerUrl.trim()) {
      addNewServer(newServerName.trim(), newServerUrl.trim());
      setNewServerName('');
      setNewServerUrl('');
      setShowNewServer(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <div className="section-header">
        <div>
          <h2 className="section-title">
            <Server size={20} className="section-title-icon" />
            Backend Servers
          </h2>
          <p className="section-description">
            Manage connections to AgentX backend servers
          </p>
        </div>
        <button
          className="button-primary"
          onClick={() => setShowNewServer(true)}
        >
          <Plus size={16} />
          Add Server
        </button>
      </div>

      {/* New Server Form */}
      {showNewServer && (
        <div className="card new-server-form">
          <h3>Add New Server</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Server Name</label>
              <input
                type="text"
                value={newServerName}
                onChange={(e) => setNewServerName(e.target.value)}
                placeholder="e.g., Production"
              />
            </div>
            <div className="form-group">
              <label>Server URL</label>
              <input
                type="url"
                value={newServerUrl}
                onChange={(e) => setNewServerUrl(e.target.value)}
                placeholder="e.g., https://api.example.com"
              />
            </div>
          </div>
          <div className="form-actions">
            <button
              className="button-secondary"
              onClick={() => setShowNewServer(false)}
            >
              Cancel
            </button>
            <button
              className="button-primary"
              onClick={handleAddServer}
              disabled={!newServerName.trim() || !newServerUrl.trim()}
            >
              <Plus size={16} />
              Add Server
            </button>
          </div>
        </div>
      )}

      {/* Server List */}
      <div className="server-list">
        {servers.map(server => (
          <ServerCard
            key={server.id}
            server={server}
            isActive={activeServer?.id === server.id}
            onSelect={() => switchServer(server.id)}
            onDelete={() => deleteServer(server.id)}
          />
        ))}
      </div>
    </div>
  );
}

// Server Card Component
interface ServerCardProps {
  server: ServerConfig;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

function ServerCard({ server, isActive, onSelect, onDelete }: ServerCardProps) {
  return (
    <div className={`server-card card ${isActive ? 'active' : ''}`}>
      <div className="server-info" onClick={onSelect}>
        <div className="server-status">
          <span className={`status-dot ${isActive ? 'online' : 'inactive'}`}></span>
        </div>
        <div className="server-details">
          <h3 className="server-name">{server.name}</h3>
          <p className="server-url">{server.url}</p>
          {server.lastConnected && (
            <p className="server-last-connected">
              Last connected: {new Date(server.lastConnected).toLocaleDateString()}
            </p>
          )}
        </div>
        {isActive && (
          <div className="active-badge">
            <Check size={14} />
            Active
          </div>
        )}
      </div>
      <div className="server-actions">
        <button className="button-ghost" title="Open in browser">
          <ExternalLink size={16} />
        </button>
        <button className="button-ghost" title="Test connection">
          <RefreshCw size={16} />
        </button>
        <button
          className="button-ghost danger"
          onClick={onDelete}
          title="Delete server"
        >
          <Trash2 size={16} />
        </button>
      </div>
    </div>
  );
}
