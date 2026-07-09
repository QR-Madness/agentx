import { useState, useEffect, type ReactNode } from 'react';
import {
  Key,
  Server,
  Eye,
  EyeOff,
  AlertTriangle,
  Upload,
  Cpu,
  Boxes,
  Languages,
} from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { useNotify } from '../../../contexts/NotificationContext';
import { useHealth } from '../../../lib/hooks';
import { api, ConfigUpdate } from '../../../lib/api';
import { Button, Card, Badge, SectionHeader, Input } from '../../ui';
import type { BadgeProps } from '../../ui';
import { useConfirm } from '../../ui/ConfirmDialog';
import anthropicIcon from '../../../assets/providers/anthropic-dark.svg';
import openaiIcon from '../../../assets/providers/openai-light.svg';
import openrouterIcon from '../../../assets/providers/open-router-dark.svg';
import lmstudioIcon from '../../../assets/providers/lmstudio.svg';
import vercelIcon from '../../../assets/providers/vercel.svg';

type ProviderKey = 'lmstudio' | 'anthropic' | 'openai' | 'openrouter' | 'vercel';
type ProviderTier = 'primary' | 'beta' | 'local';

interface ProviderDef {
  key: ProviderKey;
  name: string;
  description: string;
  icon: ReactNode;
  /** Tile gradient variant (icon chip only). */
  tile: 'local' | 'cloud' | 'experimental';
  /** Capability tier — drives grouping + the badge. */
  tier: ProviderTier;
  badge: { label: string; variant: BadgeProps['variant'] };
  /** Optional capability note under the description. */
  note?: string;
  placeholder: string;
}

// Ordered by capability tier: OpenRouter is the de-facto primary (most features
// resolve there first + image/voice are OpenRouter-only today); Anthropic/OpenAI/
// Vercel are Beta; LM Studio is local/offline.
const PROVIDERS: ProviderDef[] = [
  {
    key: 'openrouter',
    name: 'OpenRouter',
    description: 'Unified access to many providers and orgs through one key.',
    icon: <img src={openrouterIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    tier: 'primary',
    badge: { label: 'Recommended', variant: 'accent' },
    note: 'Powers image generation & voice today; most features resolve here first.',
    placeholder: 'sk-or-...',
  },
  {
    key: 'anthropic',
    name: 'Anthropic',
    description: 'Claude models — best for complex reasoning tasks.',
    icon: <img src={anthropicIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    tier: 'beta',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'sk-ant-...',
  },
  {
    key: 'openai',
    name: 'OpenAI',
    description: 'GPT models — day-to-day operations and offloading.',
    icon: <img src={openaiIcon} alt="" width={20} height={20} />,
    tile: 'experimental',
    tier: 'beta',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'sk-...',
  },
  {
    key: 'vercel',
    name: 'Vercel AI Gateway',
    description: 'Vercel-hosted gateway — day-to-day operations and offloading.',
    icon: <img src={vercelIcon} alt="" width={20} height={20} />,
    tile: 'cloud',
    tier: 'beta',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'vck_...',
  },
  {
    key: 'lmstudio',
    name: 'LM Studio',
    description: 'Local model server (OpenAI-compatible).',
    icon: <img src={lmstudioIcon} alt="" width={20} height={20} />,
    tile: 'local',
    tier: 'local',
    badge: { label: 'Local', variant: 'neutral' },
    note: 'Recommended for sensitive / offline processing.',
    placeholder: 'http://192.168.x.x:1234/v1',
  },
];

const TIERS: { tier: ProviderTier; label: string }[] = [
  { tier: 'primary', label: 'Primary' },
  { tier: 'beta', label: 'Beta' },
  { tier: 'local', label: 'Local' },
];

type ProviderSettings = Record<ProviderKey, string>;

const EMPTY_SETTINGS: ProviderSettings = {
  lmstudio: '',
  anthropic: '',
  openai: '',
  openrouter: '',
  vercel: '',
};

/** Human-readable compute device for the on-device tiles. */
function deviceLabel(device?: string): string {
  if (!device) return '—';
  if (device === 'cpu') return 'CPU';
  if (device.startsWith('cuda')) return 'CUDA (GPU)';
  return device;
}

export default function ProvidersSection() {
  const { activeServer, activeMetadata, updateMetadata } = useServer();
  const { notifyError, notifySuccess } = useNotify();
  const confirm = useConfirm();
  // On-device engine status (device + locked models). Read-only; degrades to the
  // known locked model ids when health is loading/unavailable.
  const { data: health } = useHealth(false, false);

  // The client reaches the API through the cluster's Nginx gateway when a gateway
  // token is set — i.e. a remote cluster, where LM Studio's localhost server
  // isn't reachable (no link-connection support yet).
  const onRemoteCluster = !!activeServer?.gatewayToken;

  const [showApiKeys, setShowApiKeys] = useState<Record<string, boolean>>({});
  const [providerSettings, setProviderSettings] = useState<ProviderSettings>(EMPTY_SETTINGS);
  const [savingConfig, setSavingConfig] = useState(false);
  // Last-loaded (or last-saved) state — the dirty-check baseline for the
  // explicit Save button (secrets stay on explicit save, never autosave).
  const [baseline, setBaseline] = useState<ProviderSettings | null>(null);

  useEffect(() => {
    if (!activeMetadata) return;
    const loaded: ProviderSettings = {
      lmstudio: activeMetadata.apiKeys?.lmstudio || '',
      anthropic: activeMetadata.apiKeys?.anthropic || '',
      openai: activeMetadata.apiKeys?.openai || '',
      openrouter: activeMetadata.apiKeys?.openrouter || '',
      vercel: activeMetadata.apiKeys?.vercel || '',
    };
    if (activeMetadata.apiKeys) setProviderSettings(loaded);
    // Capture the baseline once per session (server switches hard-reload the
    // app); every later metadata change is a local edit, not a load.
    setBaseline(prev => prev ?? loaded);
  }, [activeMetadata]);

  const dirty = PROVIDERS.some(
    p => providerSettings[p.key] !== (baseline ?? EMPTY_SETTINGS)[p.key]
  );

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
    const confirmed = await confirm({
      title: 'Save provider settings?',
      body: 'Saving will update server configuration and may affect running models.',
      confirmLabel: 'Save to Server',
    });
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
      setBaseline({ ...providerSettings });
      notifySuccess('Settings saved and applied to server', 'Providers');
    } catch (error) {
      notifyError(error, 'Failed to save provider settings');
    } finally {
      setSavingConfig(false);
    }
  };

  const renderProviderCard = (provider: ProviderDef) => {
    const clusterBlocked = provider.key === 'lmstudio' && onRemoteCluster;
    return (
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
              {clusterBlocked ? (
                <p className="provider-note warning">
                  Local connection only — not available on this remote cluster yet.
                </p>
              ) : (
                provider.note && <p className="provider-note">{provider.note}</p>
              )}
            </div>
          </div>
        </div>
        <div className="api-key-input">
          <Input
            type={showApiKeys[provider.key] ? 'text' : 'password'}
            value={providerSettings[provider.key]}
            onChange={(e) => handleProviderSettingChange(provider.key, e.target.value)}
            placeholder={provider.placeholder}
            disabled={clusterBlocked}
          />
          <Button
            variant="ghost"
            size="icon"
            className="visibility-toggle"
            onClick={() => toggleApiKeyVisibility(provider.key)}
            aria-label={showApiKeys[provider.key] ? 'Hide value' : 'Show value'}
            disabled={clusterBlocked}
          >
            {showApiKeys[provider.key] ? <EyeOff size={16} /> : <Eye size={16} />}
          </Button>
        </div>
      </Card>
    );
  };

  const device = deviceLabel(health?.compute?.device);
  const embeddingModel = health?.embeddings?.model ?? 'BAAI/bge-m3';
  const embeddingProvider = health?.embeddings?.provider ?? 'local';
  const embeddingDims = health?.embeddings?.dimensions;
  const translationModel =
    (health?.translation?.models?.translation as string | undefined) ??
    'facebook/nllb-200-distilled-600M';
  const translationLoaded = health?.translation?.status === 'healthy';

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Key size={20} />}
        title="Model Providers"
        description="Configure API keys and URLs for AI model providers"
        actions={
          <>
            {dirty && <span className="text-warning text-xs">Unsaved changes</span>}
            <Button
              variant="primary"
              onClick={handleSaveProviderSettings}
              loading={savingConfig}
              disabled={!dirty || savingConfig}
            >
              <Upload size={16} />
              Save to Server
            </Button>
          </>
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
        <>
          {TIERS.map(({ tier, label }) => {
            const group = PROVIDERS.filter(p => p.tier === tier);
            if (group.length === 0) return null;
            return (
              <div key={tier} className="provider-tier">
                <div className="provider-tier-eyebrow">{label}</div>
                <div className="providers-list">
                  {group.map(renderProviderCard)}
                </div>
              </div>
            );
          })}

          {/* On-device engines — read-only, locked models. */}
          <SectionHeader
            icon={<Cpu size={20} />}
            title="On-device processing"
            description="Local embedding + translation engines. Locked models; run on GPU when CUDA is available."
          />
          <div className="providers-list">
            <Card className="provider-card ondevice-card">
              <div className="provider-header">
                <div className="provider-info">
                  <div className="provider-icon local"><Boxes size={20} /></div>
                  <div>
                    <h3 className="provider-name">
                      Embeddings
                      <Badge variant="neutral" size="sm">Local · Locked</Badge>
                    </h3>
                    <p className="provider-description">Semantic memory storage &amp; recall.</p>
                    <p className="provider-note">Cloud/PaaS embeddings (still BGE-M3) coming later.</p>
                  </div>
                </div>
              </div>
              <div className="ondevice-meta">
                <span><b>Model</b> {embeddingModel}</span>
                <span><b>Provider</b> {embeddingProvider}</span>
                {embeddingDims != null && <span><b>Dimensions</b> {embeddingDims}</span>}
                <span><b>Device</b> {device}</span>
              </div>
            </Card>

            <Card className="provider-card ondevice-card">
              <div className="provider-header">
                <div className="provider-info">
                  <div className="provider-icon local"><Languages size={20} /></div>
                  <div>
                    <h3 className="provider-name">
                      Translation
                      <Badge variant="neutral" size="sm">Local · Locked</Badge>
                    </h3>
                    <p className="provider-description">NLLB-200 distilled — 200+ languages.</p>
                    <p className="provider-note">Cloud translation TBD.</p>
                  </div>
                </div>
              </div>
              <div className="ondevice-meta">
                <span><b>Model</b> {translationModel}</span>
                <span><b>Device</b> {device}</span>
                <span><b>Status</b> {translationLoaded ? 'loaded' : 'loads on first use'}</span>
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
