import { useState, useEffect } from 'react';
import {
  Key,
  Server,
  Sparkles,
  Eye,
  EyeOff,
  RefreshCw,
  Check,
  AlertTriangle,
  Upload,
} from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { api, ConfigUpdate } from '../../../lib/api';

export default function ProvidersSection() {
  const { activeServer, activeMetadata, updateMetadata } = useServer();

  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [providerSettings, setProviderSettings] = useState<{
    lmstudio: string;
    anthropic: string;
    openai: string;
  }>({ lmstudio: '', anthropic: '', openai: '' });
  const [savingConfig, setSavingConfig] = useState(false);
  const [configSaveMessage, setConfigSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (activeMetadata?.apiKeys) {
      setProviderSettings({
        lmstudio: activeMetadata.apiKeys.lmstudio || '',
        anthropic: activeMetadata.apiKeys.anthropic || '',
        openai: activeMetadata.apiKeys.openai || '',
      });
    }
  }, [activeMetadata]);

  const toggleApiKeyVisibility = (provider: string) => {
    setShowApiKeys(prev => ({ ...prev, [provider]: !prev[provider] }));
  };

  const handleProviderSettingChange = (provider: 'lmstudio' | 'anthropic' | 'openai', value: string) => {
    setProviderSettings(prev => ({ ...prev, [provider]: value }));
    updateMetadata({
      apiKeys: {
        ...activeMetadata?.apiKeys,
        [provider]: value,
      },
    });
  };

  const handleSaveProviderSettings = async () => {
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
      setTimeout(() => setConfigSaveMessage(null), 3000);
    } catch (error) {
      console.error('Failed to save config:', error);
      setConfigSaveMessage({ type: 'error', text: 'Failed to save settings to server' });
    } finally {
      setSavingConfig(false);
    }
  };

  return (
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

      <div className="config-warning">
        <AlertTriangle size={16} />
        <span>Changes are applied immediately when saved and affect all running models</span>
      </div>

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
  );
}
