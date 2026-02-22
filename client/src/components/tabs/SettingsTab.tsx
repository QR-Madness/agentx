import React, { useState, useEffect } from 'react';
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
  ChevronRight,
  FileText,
  Edit3,
  Save,
  AlertTriangle,
  Upload
} from 'lucide-react';
import { useServer } from '../../contexts/ServerContext';
import { ServerConfig } from '../../lib/storage';
import { api, PromptProfile, PromptSection, GlobalPrompt, ConfigUpdate } from '../../lib/api';
import '../../styles/SettingsTab.css';

type SettingsSection = 'servers' | 'providers' | 'prompts' | 'reasoning' | 'memory';

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

  // Provider settings state (local copy for editing)
  const [providerSettings, setProviderSettings] = useState<{
    lmstudio: string;
    anthropic: string;
    openai: string;
  }>({ lmstudio: '', anthropic: '', openai: '' });
  const [savingConfig, setSavingConfig] = useState(false);
  const [configSaveMessage, setConfigSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Prompts state
  const [profiles, setProfiles] = useState<PromptProfile[]>([]);
  const [_sections, setSections] = useState<PromptSection[]>([]);
  const [globalPrompt, setGlobalPrompt] = useState<GlobalPrompt | null>(null);
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);
  const [selectedProfileDetails, setSelectedProfileDetails] = useState<{profile: PromptProfile; composed_prompt: string} | null>(null);
  const [editingGlobal, setEditingGlobal] = useState(false);
  const [globalPromptDraft, setGlobalPromptDraft] = useState('');
  const [loadingPrompts, setLoadingPrompts] = useState(false);
  const [mcpToolsPrompt, setMcpToolsPrompt] = useState<string>('');
  const [mcpToolsCount, setMcpToolsCount] = useState(0);

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

  // Fetch prompts data when prompts section is active
  useEffect(() => {
    if (activeSection === 'prompts') {
      fetchPromptsData();
    }
  }, [activeSection]);

  const fetchPromptsData = async () => {
    setLoadingPrompts(true);
    try {
      const [profilesRes, sectionsRes, globalRes, mcpRes] = await Promise.all([
        api.listPromptProfiles(),
        api.listPromptSections(),
        api.getGlobalPrompt(),
        api.getMCPToolsPrompt(),
      ]);
      setProfiles(profilesRes.profiles);
      setSections(sectionsRes.sections);
      setGlobalPrompt(globalRes.global_prompt);
      setGlobalPromptDraft(globalRes.global_prompt.content);
      setMcpToolsPrompt(mcpRes.mcp_tools_prompt);
      setMcpToolsCount(mcpRes.tools_count);
    } catch (error) {
      console.error('Failed to fetch prompts data:', error);
    } finally {
      setLoadingPrompts(false);
    }
  };

  const handleSelectProfile = async (profileId: string) => {
    setSelectedProfileId(profileId);
    try {
      const details = await api.getPromptProfile(profileId);
      setSelectedProfileDetails(details);
    } catch (error) {
      console.error('Failed to fetch profile details:', error);
    }
  };

  const handleSaveGlobalPrompt = async () => {
    try {
      const result = await api.updateGlobalPrompt(globalPromptDraft, true);
      setGlobalPrompt(result.global_prompt);
      setEditingGlobal(false);
    } catch (error) {
      console.error('Failed to save global prompt:', error);
    }
  };

  const settingsSections = [
    { id: 'servers' as const, label: 'Servers', icon: <Server size={18} /> },
    { id: 'providers' as const, label: 'Model Providers', icon: <Key size={18} /> },
    { id: 'prompts' as const, label: 'Prompts', icon: <FileText size={18} /> },
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

          {/* Prompts Section */}
          {activeSection === 'prompts' && (
            <div className="settings-section fade-in">
              <div className="section-header">
                <div>
                  <h2 className="section-title">
                    <FileText size={20} className="section-title-icon" />
                    Prompt Management
                  </h2>
                  <p className="section-description">
                    Configure system prompts, profiles, and MCP tool prompts
                  </p>
                </div>
                <button 
                  className="button-secondary"
                  onClick={fetchPromptsData}
                  disabled={loadingPrompts}
                >
                  <RefreshCw size={16} className={loadingPrompts ? 'spin' : ''} />
                  Refresh
                </button>
              </div>

              {loadingPrompts ? (
                <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
                  <RefreshCw size={24} className="spin" style={{ marginBottom: '0.5rem' }} />
                  <p>Loading prompts...</p>
                </div>
              ) : (
                <>
                  {/* Global Prompt */}
                  <div className="card prompt-card">
                    <div className="prompt-card-header">
                      <h3>Global Prompt</h3>
                      <p className="prompt-card-description">
                        Applied to all conversations regardless of profile
                      </p>
                    </div>
                    {editingGlobal ? (
                      <div className="prompt-editor">
                        <textarea
                          value={globalPromptDraft}
                          onChange={(e) => setGlobalPromptDraft(e.target.value)}
                          rows={8}
                          placeholder="Enter your global system prompt..."
                        />
                        <div className="prompt-editor-actions">
                          <button 
                            className="button-secondary"
                            onClick={() => {
                              setEditingGlobal(false);
                              setGlobalPromptDraft(globalPrompt?.content || '');
                            }}
                          >
                            Cancel
                          </button>
                          <button 
                            className="button-primary"
                            onClick={handleSaveGlobalPrompt}
                          >
                            <Save size={16} />
                            Save
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="prompt-preview">
                        <pre>{globalPrompt?.content || 'No global prompt set'}</pre>
                        <button 
                          className="button-ghost edit-btn"
                          onClick={() => setEditingGlobal(true)}
                        >
                          <Edit3 size={16} />
                          Edit
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Prompt Profiles */}
                  <div className="card prompt-card">
                    <div className="prompt-card-header">
                      <h3>Prompt Profiles</h3>
                      <p className="prompt-card-description">
                        Named collections of prompt sections for different use cases
                      </p>
                    </div>
                    <div className="profiles-grid">
                      {profiles.map(profile => (
                        <button
                          key={profile.id}
                          className={`profile-card ${selectedProfileId === profile.id ? 'selected' : ''}`}
                          onClick={() => handleSelectProfile(profile.id)}
                        >
                          <div className="profile-card-header">
                            <span className="profile-name">{profile.name}</span>
                            {profile.is_default && <span className="default-badge">Default</span>}
                          </div>
                          {profile.description && (
                            <p className="profile-description">{profile.description}</p>
                          )}
                          <div className="profile-meta">
                            <span>{profile.sections_count || 0} sections</span>
                          </div>
                        </button>
                      ))}
                    </div>

                    {selectedProfileDetails && (
                      <div className="profile-details">
                        <h4>Composed Prompt Preview</h4>
                        <pre className="composed-prompt-preview">
                          {selectedProfileDetails.composed_prompt || 'No content'}
                        </pre>
                        <h4>Sections</h4>
                        <div className="sections-list">
                          {selectedProfileDetails.profile.sections?.map(section => (
                            <div 
                              key={section.id} 
                              className={`section-item ${section.enabled ? '' : 'disabled'}`}
                            >
                              <div className="section-item-header">
                                <span className="section-name">{section.name}</span>
                                <span className="section-type">{section.type}</span>
                              </div>
                              <pre className="section-content">{section.content}</pre>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* MCP Tools Prompt */}
                  <div className="card prompt-card">
                    <div className="prompt-card-header">
                      <h3>MCP Tools Prompt</h3>
                      <p className="prompt-card-description">
                        Auto-generated from {mcpToolsCount} available MCP tools (read-only)
                      </p>
                    </div>
                    <div className="prompt-preview">
                      <pre>{mcpToolsPrompt || 'No MCP tools available'}</pre>
                    </div>
                  </div>
                </>
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
