import React, { useState, useEffect } from 'react';
import {
  Settings,
  Server,
  Key,
  Sparkles,
  Database,
  Plus,
  Trash2,
  Check,
  ExternalLink,
  RefreshCw,
  Eye,
  EyeOff,
  ChevronRight,
  AlertTriangle,
  Upload,
  Layers,
  Cpu,
  Save,
} from 'lucide-react';
import { useServer } from '../../contexts/ServerContext';
import { ServerConfig } from '../../lib/storage';
import { api, ConfigUpdate } from '../../lib/api';
import '../../styles/SettingsPanel.css';

type SettingsSection = 'servers' | 'providers' | 'models' | 'memory';

interface ContextLimits {
  lmstudio: { context_window: number; max_output_tokens: number };
  anthropic: { context_window: number; max_output_tokens: number };
  openai: { context_window: number; max_output_tokens: number };
  models: Record<string, { context_window: number; max_output_tokens: number }>;
}

export const SettingsPanel: React.FC = () => {
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

  // Provider settings state (local copy for editing)
  const [providerSettings, setProviderSettings] = useState<{
    lmstudio: string;
    anthropic: string;
    openai: string;
  }>({ lmstudio: '', anthropic: '', openai: '' });
  const [savingConfig, setSavingConfig] = useState(false);
  const [configSaveMessage, setConfigSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Context limits state
  const [contextLimits, setContextLimits] = useState<ContextLimits | null>(null);
  const [loadingContextLimits, setLoadingContextLimits] = useState(false);
  const [savingContextLimits, setSavingContextLimits] = useState(false);
  const [contextLimitsMessage, setContextLimitsMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

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

  const handleProviderSettingChange = (provider: 'lmstudio' | 'anthropic' | 'openai', value: string) => {
    setProviderSettings(prev => ({ ...prev, [provider]: value }));
    // Also update localStorage
    handleApiKeyChange(provider, value);
  };

  const handleSaveProviderSettings = async () => {
    // Confirm with user
    const confirmed = window.confirm(
      'Saving will update server configuration and may affect running models. Continue?'
    );
    if (!confirmed) return;

    setSavingConfig(true);
    setConfigSaveMessage(null);

    try {
      const config: ConfigUpdate = {
        providers: {
          lmstudio: providerSettings.lmstudio ? { base_url: providerSettings.lmstudio } : undefined,
          anthropic: providerSettings.anthropic ? { api_key: providerSettings.anthropic } : undefined,
          openai: providerSettings.openai ? { api_key: providerSettings.openai } : undefined,
        },
      };

      await api.updateConfig(config);
      setConfigSaveMessage({ type: 'success', text: 'Settings saved and applied to server' });

      // Clear message after 3 seconds
      setTimeout(() => setConfigSaveMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save config:', error);
      setConfigSaveMessage({ type: 'error', text: 'Failed to save settings to server' });
    } finally {
      setSavingConfig(false);
    }
  };

  // Initialize provider settings from activeMetadata
  useEffect(() => {
    if (activeMetadata?.apiKeys) {
      setProviderSettings({
        lmstudio: activeMetadata.apiKeys.lmstudio || '',
        anthropic: activeMetadata.apiKeys.anthropic || '',
        openai: activeMetadata.apiKeys.openai || '',
      });
    }
  }, [activeMetadata]);

  // Fetch context limits when models section is active
  useEffect(() => {
    if (activeSection === 'models') {
      fetchContextLimits();
    }
  }, [activeSection]);

  const fetchContextLimits = async () => {
    setLoadingContextLimits(true);
    try {
      const limits = await api.getContextLimits();
      setContextLimits(limits);
    } catch (error) {
      console.error('Failed to fetch context limits:', error);
    } finally {
      setLoadingContextLimits(false);
    }
  };

  const handleContextLimitChange = (
    provider: 'lmstudio' | 'anthropic' | 'openai',
    field: 'context_window' | 'max_output_tokens',
    value: number
  ) => {
    if (!contextLimits) return;
    setContextLimits({
      ...contextLimits,
      [provider]: {
        ...contextLimits[provider],
        [field]: value,
      },
    });
  };

  const handleSaveContextLimits = async () => {
    if (!contextLimits) return;

    setSavingContextLimits(true);
    setContextLimitsMessage(null);

    try {
      await api.updateContextLimits(contextLimits);
      setContextLimitsMessage({ type: 'success', text: 'Context limits saved' });
      setTimeout(() => setContextLimitsMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save context limits:', error);
      setContextLimitsMessage({ type: 'error', text: 'Failed to save context limits' });
    } finally {
      setSavingContextLimits(false);
    }
  };

  const settingsSections = [
    { id: 'servers' as const, label: 'Servers', icon: <Server size={18} /> },
    { id: 'providers' as const, label: 'Model Providers', icon: <Key size={18} /> },
    { id: 'models' as const, label: 'Model Limits', icon: <Layers size={18} /> },
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
          {settingsSections.map(section => (
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
                    Configure API keys and URLs for AI model providers
                  </p>
                </div>
                <button
                  className="button-primary"
                  onClick={handleSaveProviderSettings}
                  disabled={savingConfig}
                >
                  {savingConfig ? (
                    <RefreshCw size={16} className="spin" />
                  ) : (
                    <Upload size={16} />
                  )}
                  Save to Server
                </button>
              </div>

              {/* Warning Banner */}
              <div className="config-warning">
                <AlertTriangle size={16} />
                <span>Changes are applied immediately when saved and affect all running models</span>
              </div>

              {/* Save Message */}
              {configSaveMessage && (
                <div className={`config-message ${configSaveMessage.type}`}>
                  {configSaveMessage.type === 'success' ? <Check size={16} /> : <AlertTriangle size={16} />}
                  <span>{configSaveMessage.text}</span>
                </div>
              )}

              {!activeServer ? (
                <div className="empty-state card">
                  <Server size={32} />
                  <p>Select a server first to configure API keys</p>
                </div>
              ) : (
                <div className="providers-list">
                  {/* LM Studio - Local */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon local">
                          <Server size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            LM Studio
                            <span className="provider-badge local">Local</span>
                          </h3>
                          <p className="provider-description">
                            Local model server URL (OpenAI-compatible)
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="provider-form">
                      <div className="api-key-input">
                        <input
                          type={showApiKeys.lmstudio ? 'text' : 'password'}
                          value={providerSettings.lmstudio}
                          onChange={(e) => handleProviderSettingChange('lmstudio', e.target.value)}
                          placeholder="http://192.168.x.x:1234/v1"
                        />
                        <button
                          className="button-ghost visibility-toggle"
                          onClick={() => toggleApiKeyVisibility('lmstudio')}
                        >
                          {showApiKeys.lmstudio ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Anthropic - Cloud Primary */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon cloud">
                          <Sparkles size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            Anthropic
                            <span className="provider-badge primary">Primary</span>
                          </h3>
                          <p className="provider-description">
                            API key for Claude models
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="provider-form">
                      <div className="api-key-input">
                        <input
                          type={showApiKeys.anthropic ? 'text' : 'password'}
                          value={providerSettings.anthropic}
                          onChange={(e) => handleProviderSettingChange('anthropic', e.target.value)}
                          placeholder="sk-ant-..."
                        />
                        <button
                          className="button-ghost visibility-toggle"
                          onClick={() => toggleApiKeyVisibility('anthropic')}
                        >
                          {showApiKeys.anthropic ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* OpenAI - Experimental */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon experimental">
                          <Sparkles size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            OpenAI
                            <span className="provider-badge experimental">Experimental</span>
                          </h3>
                          <p className="provider-description">
                            API key for GPT models
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="provider-form">
                      <div className="api-key-input">
                        <input
                          type={showApiKeys.openai ? 'text' : 'password'}
                          value={providerSettings.openai}
                          onChange={(e) => handleProviderSettingChange('openai', e.target.value)}
                          placeholder="sk-..."
                        />
                        <button
                          className="button-ghost visibility-toggle"
                          onClick={() => toggleApiKeyVisibility('openai')}
                        >
                          {showApiKeys.openai ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Model Limits Section */}
          {activeSection === 'models' && (
            <div className="settings-section fade-in">
              <div className="section-header">
                <div>
                  <h2 className="section-title">
                    <Layers size={20} className="section-title-icon" />
                    Model Context Limits
                  </h2>
                  <p className="section-description">
                    Configure context window and output token limits per provider
                  </p>
                </div>
                <button
                  className="button-primary"
                  onClick={handleSaveContextLimits}
                  disabled={savingContextLimits || !contextLimits}
                >
                  {savingContextLimits ? (
                    <RefreshCw size={16} className="spin" />
                  ) : (
                    <Save size={16} />
                  )}
                  Save Limits
                </button>
              </div>

              {/* Save Message */}
              {contextLimitsMessage && (
                <div className={`config-message ${contextLimitsMessage.type}`}>
                  {contextLimitsMessage.type === 'success' ? <Check size={16} /> : <AlertTriangle size={16} />}
                  <span>{contextLimitsMessage.text}</span>
                </div>
              )}

              {loadingContextLimits ? (
                <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
                  <RefreshCw size={24} className="spin" style={{ marginBottom: '0.5rem' }} />
                  <p>Loading context limits...</p>
                </div>
              ) : contextLimits ? (
                <div className="providers-list">
                  {/* LM Studio Limits */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon local">
                          <Server size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            LM Studio
                            <span className="provider-badge local">Local</span>
                          </h3>
                          <p className="provider-description">
                            Context limits for local models
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="context-limits-form">
                      <div className="form-group">
                        <label>Context Window (tokens)</label>
                        <input
                          type="number"
                          value={contextLimits.lmstudio.context_window}
                          onChange={(e) => handleContextLimitChange('lmstudio', 'context_window', parseInt(e.target.value) || 0)}
                          min={1024}
                          max={1000000}
                          step={1024}
                        />
                        <span className="form-hint">
                          {(contextLimits.lmstudio.context_window / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                      <div className="form-group">
                        <label>Max Output Tokens</label>
                        <input
                          type="number"
                          value={contextLimits.lmstudio.max_output_tokens}
                          onChange={(e) => handleContextLimitChange('lmstudio', 'max_output_tokens', parseInt(e.target.value) || 0)}
                          min={256}
                          max={131072}
                          step={256}
                        />
                        <span className="form-hint">
                          {(contextLimits.lmstudio.max_output_tokens / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Anthropic Limits */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon cloud">
                          <Sparkles size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            Anthropic
                            <span className="provider-badge primary">Claude</span>
                          </h3>
                          <p className="provider-description">
                            Context limits for Claude models (up to 1M for Opus)
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="context-limits-form">
                      <div className="form-group">
                        <label>Context Window (tokens)</label>
                        <input
                          type="number"
                          value={contextLimits.anthropic.context_window}
                          onChange={(e) => handleContextLimitChange('anthropic', 'context_window', parseInt(e.target.value) || 0)}
                          min={1024}
                          max={1000000}
                          step={1024}
                        />
                        <span className="form-hint">
                          {(contextLimits.anthropic.context_window / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                      <div className="form-group">
                        <label>Max Output Tokens</label>
                        <input
                          type="number"
                          value={contextLimits.anthropic.max_output_tokens}
                          onChange={(e) => handleContextLimitChange('anthropic', 'max_output_tokens', parseInt(e.target.value) || 0)}
                          min={256}
                          max={131072}
                          step={256}
                        />
                        <span className="form-hint">
                          {(contextLimits.anthropic.max_output_tokens / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* OpenAI Limits */}
                  <div className="provider-card card">
                    <div className="provider-header">
                      <div className="provider-info">
                        <div className="provider-icon experimental">
                          <Cpu size={20} />
                        </div>
                        <div>
                          <h3 className="provider-name">
                            OpenAI
                            <span className="provider-badge experimental">GPT</span>
                          </h3>
                          <p className="provider-description">
                            Context limits for GPT models
                          </p>
                        </div>
                      </div>
                    </div>
                    <div className="context-limits-form">
                      <div className="form-group">
                        <label>Context Window (tokens)</label>
                        <input
                          type="number"
                          value={contextLimits.openai.context_window}
                          onChange={(e) => handleContextLimitChange('openai', 'context_window', parseInt(e.target.value) || 0)}
                          min={1024}
                          max={128000}
                          step={1024}
                        />
                        <span className="form-hint">
                          {(contextLimits.openai.context_window / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                      <div className="form-group">
                        <label>Max Output Tokens</label>
                        <input
                          type="number"
                          value={contextLimits.openai.max_output_tokens}
                          onChange={(e) => handleContextLimitChange('openai', 'max_output_tokens', parseInt(e.target.value) || 0)}
                          min={256}
                          max={16384}
                          step={256}
                        />
                        <span className="form-hint">
                          {(contextLimits.openai.max_output_tokens / 1000).toFixed(0)}k tokens
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state card">
                  <AlertTriangle size={32} />
                  <p>Failed to load context limits</p>
                  <button className="button-secondary" onClick={fetchContextLimits}>
                    <RefreshCw size={16} />
                    Retry
                  </button>
                </div>
              )}
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
