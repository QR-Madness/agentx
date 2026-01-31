import React, { useState } from 'react';
import {
  Settings,
  Server,
  Key,
  Brain,
  Sparkles,
  Database,
  Plus,
  Trash2,
  Check,
  ExternalLink,
  RefreshCw,
  Eye,
  EyeOff,
  ChevronRight
} from 'lucide-react';
import { useServer } from '../../contexts/ServerContext';
import { ServerConfig } from '../../lib/storage';
import '../../styles/SettingsTab.css';

type SettingsSection = 'servers' | 'providers' | 'reasoning' | 'memory';

export const SettingsTab: React.FC = () => {
  const [activeSection, setActiveSection] = useState<SettingsSection>('servers');
  const { 
    servers, 
    activeServer, 
    activeMetadata,
    switchServer, 
    addNewServer, 
    deleteServer,
    updateMetadata 
  } = useServer();

  // New server form state
  const [showNewServer, setShowNewServer] = useState(false);
  const [newServerName, setNewServerName] = useState('');
  const [newServerUrl, setNewServerUrl] = useState('');

  // API key visibility
  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});

  const handleAddServer = () => {
    if (newServerName.trim() && newServerUrl.trim()) {
      addNewServer(newServerName.trim(), newServerUrl.trim());
      setNewServerName('');
      setNewServerUrl('');
      setShowNewServer(false);
    }
  };

  const handleApiKeyChange = (provider: string, value: string) => {
    updateMetadata({
      apiKeys: {
        ...activeMetadata?.apiKeys,
        [provider]: value,
      },
    });
  };

  const toggleApiKeyVisibility = (provider: string) => {
    setShowApiKeys(prev => ({ ...prev, [provider]: !prev[provider] }));
  };

  const sections = [
    { id: 'servers' as const, label: 'Servers', icon: <Server size={18} /> },
    { id: 'providers' as const, label: 'Model Providers', icon: <Key size={18} /> },
    { id: 'reasoning' as const, label: 'Reasoning', icon: <Brain size={18} /> },
    { id: 'memory' as const, label: 'Memory', icon: <Database size={18} /> },
  ];

  return (
    <div className="settings-tab">
      {/* Header */}
      <div className="settings-header fade-in">
        <h1 className="page-title">
          <Settings className="page-icon-svg" />
          <span>Settings</span>
        </h1>
        <p className="page-subtitle">Configure your AgentX environment</p>
      </div>

      <div className="settings-layout">
        {/* Sidebar */}
        <nav className="settings-nav card">
          {sections.map(section => (
            <button
              key={section.id}
              className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => setActiveSection(section.id)}
            >
              <span className="nav-icon">{section.icon}</span>
              <span className="nav-label">{section.label}</span>
              <ChevronRight size={16} className="nav-arrow" />
            </button>
          ))}
        </nav>

        {/* Content */}
        <div className="settings-content">
          {/* Servers Section */}
          {activeSection === 'servers' && (
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
          )}

          {/* Model Providers Section */}
          {activeSection === 'providers' && (
            <div className="settings-section fade-in">
              <div className="section-header">
                <div>
                  <h2 className="section-title">
                    <Key size={20} className="section-title-icon" />
                    Model Providers
                  </h2>
                  <p className="section-description">
                    Configure API keys for AI model providers (stored per-server)
                  </p>
                </div>
              </div>

              {!activeServer ? (
                <div className="empty-state card">
                  <Server size={32} />
                  <p>Select a server first to configure API keys</p>
                </div>
              ) : (
                <div className="providers-list">
                  {['openai', 'anthropic', 'ollama'].map(provider => (
                    <div key={provider} className="provider-card card">
                      <div className="provider-header">
                        <div className="provider-info">
                          <div className="provider-icon">
                            <Sparkles size={20} />
                          </div>
                          <div>
                            <h3 className="provider-name">
                              {provider.charAt(0).toUpperCase() + provider.slice(1)}
                            </h3>
                            <p className="provider-description">
                              {provider === 'ollama' 
                                ? 'Local model server URL' 
                                : 'API key for cloud models'}
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="provider-form">
                        <div className="api-key-input">
                          <input
                            type={showApiKeys[provider] ? 'text' : 'password'}
                            value={activeMetadata?.apiKeys?.[provider as keyof typeof activeMetadata.apiKeys] || ''}
                            onChange={(e) => handleApiKeyChange(provider, e.target.value)}
                            placeholder={provider === 'ollama' 
                              ? 'http://localhost:11434' 
                              : `Enter ${provider} API key`}
                          />
                          <button 
                            className="button-ghost visibility-toggle"
                            onClick={() => toggleApiKeyVisibility(provider)}
                          >
                            {showApiKeys[provider] ? <EyeOff size={16} /> : <Eye size={16} />}
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Reasoning Section */}
          {activeSection === 'reasoning' && (
            <div className="settings-section fade-in">
              <div className="section-header">
                <div>
                  <h2 className="section-title">
                    <Brain size={20} className="section-title-icon" />
                    Reasoning Preferences
                  </h2>
                  <p className="section-description">
                    Configure default reasoning strategies and behavior
                  </p>
                </div>
              </div>

              <div className="preferences-card card">
                <div className="form-group">
                  <label>Default Reasoning Strategy</label>
                  <select
                    value={activeMetadata?.preferences?.defaultReasoningStrategy || 'auto'}
                    onChange={(e) => updateMetadata({
                      preferences: {
                        ...activeMetadata?.preferences,
                        defaultReasoningStrategy: e.target.value,
                      },
                    })}
                  >
                    <option value="auto">Auto (Recommended)</option>
                    <option value="chain_of_thought">Chain of Thought</option>
                    <option value="tree_of_thought">Tree of Thought</option>
                    <option value="react">ReAct</option>
                    <option value="reflection">Reflection</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Default Drafting Strategy</label>
                  <select
                    value={activeMetadata?.preferences?.defaultDraftingStrategy || 'none'}
                    onChange={(e) => updateMetadata({
                      preferences: {
                        ...activeMetadata?.preferences,
                        defaultDraftingStrategy: e.target.value,
                      },
                    })}
                  >
                    <option value="none">None</option>
                    <option value="speculative">Speculative Decoding</option>
                    <option value="pipeline">Multi-Model Pipeline</option>
                    <option value="candidate">Candidate Generation</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Default Model</label>
                  <select
                    value={activeMetadata?.preferences?.defaultModel || 'auto'}
                    onChange={(e) => updateMetadata({
                      preferences: {
                        ...activeMetadata?.preferences,
                        defaultModel: e.target.value,
                      },
                    })}
                  >
                    <option value="auto">Auto Select</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
                    <option value="claude-3-5-haiku">Claude 3.5 Haiku</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Memory Section */}
          {activeSection === 'memory' && (
            <div className="settings-section fade-in">
              <div className="section-header">
                <div>
                  <h2 className="section-title">
                    <Database size={20} className="section-title-icon" />
                    Memory & Storage
                  </h2>
                  <p className="section-description">
                    Configure agent memory and data retention
                  </p>
                </div>
              </div>

              <div className="memory-info card">
                <div className="info-row">
                  <span className="info-label">Session Storage</span>
                  <span className="info-value">Local (Browser)</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Server Data</span>
                  <span className="info-value">PostgreSQL + Neo4j</span>
                </div>
                <div className="info-row">
                  <span className="info-label">Cache</span>
                  <span className="info-value">Redis</span>
                </div>
              </div>

              <div className="card">
                <h3 className="subsection-title">Local Data</h3>
                <p className="subsection-description">
                  Clear locally stored preferences and cached data.
                </p>
                <button className="button-secondary danger">
                  <Trash2 size={16} />
                  Clear Local Data
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Server Card Component
interface ServerCardProps {
  server: ServerConfig;
  isActive: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

const ServerCard: React.FC<ServerCardProps> = ({ server, isActive, onSelect, onDelete }) => {
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
};
