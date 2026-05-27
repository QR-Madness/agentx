import { useState, useEffect, type ReactNode } from 'react';
import {
  Key,
  Server,
  Eye,
  EyeOff,
  AlertTriangle,
  Upload,
} from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { useNotify } from '../../../contexts/NotificationContext';
import { api, ConfigUpdate } from '../../../lib/api';
import { Button, Card, Badge, SectionHeader, Input } from '../../ui';
import type { BadgeProps } from '../../ui';
import anthropicIcon from '../../../assets/providers/anthropic-dark.svg';
import openaiIcon from '../../../assets/providers/openai-light.svg';
import openrouterIcon from '../../../assets/providers/open-router-dark.svg';
import lmstudioIcon from '../../../assets/providers/lmstudio.svg';
import vercelIcon from '../../../assets/providers/vercel.svg';

type ProviderKey = 'lmstudio' | 'anthropic' | 'openai' | 'openrouter' | 'vercel';

interface ProviderDef {
  key: ProviderKey;
  name: string;
  description: string;
  icon: ReactNode;
  /** Tile gradient variant. */
  tile: 'local' | 'cloud' | 'experimental';
  badge: { label: string; variant: BadgeProps['variant'] };
  placeholder: string;
}

const PROVIDERS: ProviderDef[] = [
  {
    key: 'lmstudio',
    name: 'LM Studio',
    description: 'Local model server (OpenAI-compatible) — recommended for sensitive data',
    icon: <img src={lmstudioIcon} alt="" width={20} height={20} />,
    tile: 'local',
    badge: { label: 'Offline', variant: 'neutral' },
    placeholder: 'http://192.168.x.x:1234/v1',
  },
  {
    key: 'anthropic',
    name: 'Anthropic',
    description: 'Claude models — best for complex reasoning tasks',
    icon: <img src={anthropicIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    badge: { label: 'High-Reasoning', variant: 'accent' },
    placeholder: 'sk-ant-...',
  },
  {
    key: 'openai',
    name: 'OpenAI',
    description: 'GPT models — day-to-day operations and offloading',
    icon: <img src={openaiIcon} alt="" width={20} height={20} />,
    tile: 'experimental',
    badge: { label: 'Cloud', variant: 'neutral' },
    placeholder: 'sk-...',
  },
  {
    key: 'openrouter',
    name: 'OpenRouter',
    description: 'Unified access to many providers and orgs through one key',
    icon: <img src={openrouterIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    badge: { label: 'Cloud Router', variant: 'accent' },
    placeholder: 'sk-or-...',
  },
  {
    key: 'vercel',
    name: 'Vercel AI Gateway',
    description: 'Vercel-hosted gateway — day-to-day operations and offloading',
    icon: <img src={vercelIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    badge: { label: 'Cloud', variant: 'neutral' },
    placeholder: 'vck_...',
  },
];

type ProviderSettings = Record<ProviderKey, string>;

const EMPTY_SETTINGS: ProviderSettings = {
  lmstudio: '',
  anthropic: '',
  openai: '',
  openrouter: '',
  vercel: '',
};

export default function ProvidersSection() {
  const { activeServer, activeMetadata, updateMetadata } = useServer();
  const { notifyError, notifySuccess } = useNotify();

  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [providerSettings, setProviderSettings] = useState<ProviderSettings>(EMPTY_SETTINGS);
  const [savingConfig, setSavingConfig] = useState(false);

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

  const handleProviderSettingChange = (provider: ProviderKey, value: string) => {
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
      notifySuccess('Settings saved and applied to server', 'Providers');
    } catch (error) {
      notifyError(error, 'Failed to save provider settings');
    } finally {
      setSavingConfig(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Key size={20} />}
        title="Model Providers"
        description="Configure API keys and URLs for AI model providers"
        actions={
          <Button variant="primary" onClick={handleSaveProviderSettings} loading={savingConfig}>
            <Upload size={16} />
            Save to Server
          </Button>
        }
      />

      <div className="config-warning">
        <AlertTriangle size={16} />
        <span>Changes are applied immediately when saved and affect all running models</span>
      </div>

      {!activeServer ? (
        <Card className="empty-state">
          <Server size={32} />
          <p>Select a server first to configure API keys</p>
        </Card>
      ) : (
        <div className="providers-list">
          {PROVIDERS.map(provider => (
            <Card key={provider.key} className="provider-card">
              <div className="provider-header">
                <div className="provider-info">
                  <div className={`provider-icon ${provider.tile}`}>{provider.icon}</div>
                  <div>
                    <h3 className="provider-name">
                      {provider.name}
                      <Badge variant={provider.badge.variant} size="sm">
                        {provider.badge.label}
                      </Badge>
                    </h3>
                    <p className="provider-description">{provider.description}</p>
                  </div>
                </div>
              </div>
              <div className="api-key-input">
                <Input
                  type={showApiKeys[provider.key] ? 'text' : 'password'}
                  value={providerSettings[provider.key]}
                  onChange={(e) => handleProviderSettingChange(provider.key, e.target.value)}
                  placeholder={provider.placeholder}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="visibility-toggle"
                  onClick={() => toggleApiKeyVisibility(provider.key)}
                  aria-label={showApiKeys[provider.key] ? 'Hide value' : 'Show value'}
                >
                  {showApiKeys[provider.key] ? <EyeOff size={16} /> : <Eye size={16} />}
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
