import { useState, useEffect } from 'react';
import {
  Key,
  Server,
  Eye,
  EyeOff,
  RefreshCw,
  Check,
  AlertTriangle,
  Upload,
} from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { api, ConfigUpdate } from '../../../lib/api';
import anthropicIcon from '../../../assets/providers/anthropic-dark.svg';
import openaiIcon from '../../../assets/providers/openai-light.svg';
import openrouterIcon from '../../../assets/providers/open-router-dark.svg';
import lmstudioIcon from '../../../assets/providers/lmstudio.svg';
import vercelIcon from '../../../assets/providers/vercel.svg';

export default function ProvidersSection() {
  const { activeServer, activeMetadata, updateMetadata } = useServer();

  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [providerSettings, setProviderSettings] = useState<{
    lmstudio: string;
    anthropic: string;
    openai: string;
    openrouter: string;
    vercel: string;
  }>({ lmstudio: '', anthropic: '', openai: '', openrouter: '', vercel: '' });
  const [savingConfig, setSavingConfig] = useState(false);
  const [configSaveMessage, setConfigSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (activeMetadata?.apiKeys) {
      setProviderSettings({
        lmstudio: activeMetadata.apiKeys.lmstudio || '',
        anthropic: activeMetadata.apiKeys.anthropic || '',
        openai: activeMetadata.apiKeys.openai || '',
        openrouter: activeMetadata.apiKeys.openrouter || '',
        vercel: activeMetadata.apiKeys.vercel || '',
      });
    }
  }, [activeMetadata]);

  const toggleApiKeyVisibility = (provider: string) => {
    setShowApiKeys(prev => ({ ...prev, [provider]: !prev[provider] }));
  };

  const handleProviderSettingChange = (provider: 'lmstudio' | 'anthropic' | 'openai' | 'openrouter' | 'vercel', value: string) => {
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
          openrouter: providerSettings.openrouter ? { api_key: providerSettings.openrouter } : undefined,
          vercel: providerSettings.vercel ? { api_key: providerSettings.vercel } : undefined,
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
                  <img src={lmstudioIcon} alt="" width={20} height={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    LM Studio
                    <span className="provider-badge local">Offline</span>
                  </h3>
                  <p className="provider-description">
                    Local model server (OpenAI-compatible) — recommended for sensitive data
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
                  <img src={anthropicIcon} alt="" width={20} height={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    Anthropic
                    <span className="provider-badge primary">High-Reasoning</span>
                  </h3>
                  <p className="provider-description">
                    Claude models — best for complex reasoning tasks
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
                  <img src={openaiIcon} alt="" width={20} height={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    OpenAI
                    <span className="provider-badge cloud">Cloud</span>
                  </h3>
                  <p className="provider-description">
                    GPT models — day-to-day operations and offloading
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

          <div className="provider-card card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon cloud">
                  <img src={openrouterIcon} alt="" width={20} height={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    OpenRouter
                    <span className="provider-badge router">Cloud Router</span>
                  </h3>
                  <p className="provider-description">
                    Unified access to many providers and orgs through one key
                  </p>
                </div>
              </div>
            </div>
            <div className="provider-form">
              <div className="api-key-input">
                <input
                  type={showApiKeys.openrouter ? 'text' : 'password'}
                  value={providerSettings.openrouter}
                  onChange={(e) => handleProviderSettingChange('openrouter', e.target.value)}
                  placeholder="sk-or-..."
                />
                <button
                  className="button-ghost visibility-toggle"
                  onClick={() => toggleApiKeyVisibility('openrouter')}
                >
                  {showApiKeys.openrouter ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
          </div>

          <div className="provider-card card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon cloud">
                  <img src={vercelIcon} alt="" width={20} height={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    Vercel AI Gateway
                    <span className="provider-badge cloud">Cloud</span>
                  </h3>
                  <p className="provider-description">
                    Vercel-hosted gateway — day-to-day operations and offloading
                  </p>
                </div>
              </div>
            </div>
            <div className="provider-form">
              <div className="api-key-input">
                <input
                  type={showApiKeys.vercel ? 'text' : 'password'}
                  value={providerSettings.vercel}
                  onChange={(e) => handleProviderSettingChange('vercel', e.target.value)}
                  placeholder="vck_..."
                />
                <button
                  className="button-ghost visibility-toggle"
                  onClick={() => toggleApiKeyVisibility('vercel')}
                >
                  {showApiKeys.vercel ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
