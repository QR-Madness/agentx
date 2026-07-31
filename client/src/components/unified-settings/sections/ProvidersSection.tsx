/**
 * ProvidersSection — where AgentX gets its intelligence.
 *
 * Rebuilt around **live state**. The previous version rendered seven identical
 * static tiles with a password box each: a linked account and an empty box
 * looked the same, `/api/providers/health` existed and was never called, and
 * nothing on the page could tell you which provider a turn would actually use.
 *
 * Three things changed:
 *  1. The **Supply Line** rail states the live route for the active agent.
 *  2. Cards lead with connection state (reachable, model count, key on file)
 *     and keep credentials behind an expand.
 *  3. The provider list comes from `GET /api/providers/catalog`, so
 *     user-registered endpoints appear beside the built-ins.
 *
 * **Secrets are server-side truth.** Key state arrives as a fingerprint from the
 * catalog; the client no longer mirrors keys into browser storage, which used to
 * mean anything written server-side (a cluster `.env`, another client) left the
 * field looking empty. Typing into a key field is an explicit replacement, and
 * secrets still save on an explicit press — never autosave.
 */

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { AlertTriangle, Boxes, Cpu, Key, Languages, Plus, Server, Upload } from 'lucide-react';
import { useServer } from '../../../contexts/ServerContext';
import { useNotify } from '../../../contexts/NotificationContext';
import { useAgentProfile } from '../../../contexts/AgentProfileContext';
import { useHealth, useProvidersHealth } from '../../../lib/hooks';
import {
  api,
  apiErrorMessage,
  type ConfigUpdate,
  type ProviderCatalogEntry,
  type ProviderTestResult,
} from '../../../lib/api';
import { Badge, Button, Card, SectionHeader } from '../../ui';
import { useConfirm } from '../../ui/ConfirmDialog';
import { ProviderConnectionCard, type ProviderPresentation } from '../providers/ProviderConnectionCard';
import { EndpointSheet } from '../providers/EndpointSheet';
import { SupplyLine } from '../providers/SupplyLine';
import anthropicIcon from '../../../assets/providers/anthropic-dark.svg';
import openaiIcon from '../../../assets/providers/openai-light.svg';
import openrouterIcon from '../../../assets/providers/open-router-dark.svg';
import lmstudioIcon from '../../../assets/providers/lmstudio.svg';
import vercelIcon from '../../../assets/providers/vercel.svg';

/** Health polling. 20s keeps reachability honest without hammering providers —
 *  each poll pings every configured backend. */
const HEALTH_POLL_MS = 20_000;

const providerImage = (src: string) => <img src={src} alt="" width={20} height={20} />;

/** Presentation for the shipped providers. The *catalog* decides what exists;
 *  this only decides how a known one looks. An unknown id falls back gracefully. */
const PRESENTATION: Record<string, ProviderPresentation> = {
  openrouter: {
    description: 'Unified access to many providers and orgs through one key.',
    icon: providerImage(openrouterIcon),
    tile: 'cloud',
    badge: { label: 'Recommended', variant: 'accent' },
    note: 'Powers image generation and voice today; most features resolve here first.',
    placeholder: 'sk-or-…',
  },
  anthropic: {
    description: 'Claude models — best for complex reasoning.',
    icon: providerImage(anthropicIcon),
    tile: 'cloud',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'sk-ant-…',
  },
  openai: {
    description: 'GPT models — day-to-day operations and offloading.',
    icon: providerImage(openaiIcon),
    tile: 'experimental',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'sk-…',
  },
  vercel: {
    description: 'Vercel-hosted gateway — day-to-day operations and offloading.',
    icon: providerImage(vercelIcon),
    tile: 'cloud',
    badge: { label: 'Beta', variant: 'warning' },
    placeholder: 'vck_…',
  },
  lmstudio: {
    description: 'Local model server (OpenAI-compatible).',
    icon: providerImage(lmstudioIcon),
    tile: 'local',
    badge: { label: 'Local', variant: 'neutral' },
    note: 'Recommended for sensitive or offline processing.',
    placeholder: 'http://192.168.x.x:1234/v1',
  },
};

/** A registered endpoint we ship no artwork for. */
const CUSTOM_PRESENTATION: ProviderPresentation = {
  description: 'An OpenAI-compatible endpoint you connected.',
  icon: <Boxes size={20} />,
  tile: 'cloud',
  placeholder: 'sk-…',
};

function presentationFor(entry: ProviderCatalogEntry): ProviderPresentation {
  return PRESENTATION[entry.id] ?? CUSTOM_PRESENTATION;
}

/** Built-ins first (in shipped order), then custom endpoints alphabetically —
 *  a stable reading order that doesn't reshuffle as connections come and go. */
function orderEntries(entries: ProviderCatalogEntry[]): ProviderCatalogEntry[] {
  const builtinOrder = Object.keys(PRESENTATION);
  return [...entries].sort((a, b) => {
    if (a.builtin !== b.builtin) return a.builtin ? -1 : 1;
    if (a.builtin) return builtinOrder.indexOf(a.id) - builtinOrder.indexOf(b.id);
    return a.label.localeCompare(b.label);
  });
}

/** Human-readable compute device for the on-device tiles. */
function deviceLabel(device?: string): string {
  if (!device) return '—';
  if (device === 'cpu') return 'CPU';
  if (device.startsWith('cuda')) return 'CUDA (GPU)';
  return device;
}

export default function ProvidersSection() {
  const { activeServer } = useServer();
  const { activeProfile } = useAgentProfile();
  const { notifyError, notifySuccess } = useNotify();
  const confirm = useConfirm();

  // On-device engine status (device + locked models). Read-only.
  const { data: health } = useHealth(false, false);
  const { providers: providersHealth, loading: healthLoading, refresh: refreshHealth } =
    useProvidersHealth({ pollInterval: HEALTH_POLL_MS });

  const [catalog, setCatalog] = useState<ProviderCatalogEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  // The global default model. A profile that pins no model of its own runs on
  // this — the resolution the Supply Line exists to make visible.
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  // Credential edits in flight, keyed by provider id. Empty string = untouched;
  // secrets never autosave, so this stays local until an explicit save.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [sheetOpen, setSheetOpen] = useState(false);
  // Bumped after any change that could alter resolution, so the rail re-resolves.
  const [routeToken, setRouteToken] = useState(0);

  // The client reaches the API through the cluster's Nginx gateway when a
  // gateway token is set — i.e. a remote cluster, where LM Studio's localhost
  // server isn't reachable (no link-connection support yet).
  const onRemoteCluster = !!activeServer?.gatewayToken;

  const loadCatalog = useCallback(async () => {
    try {
      const response = await api.getProviderCatalog();
      setCatalog(response.providers);
      setCatalogError(null);
    } catch (error) {
      setCatalogError(apiErrorMessage(error));
    }
  }, []);

  useEffect(() => {
    if (!activeServer) return;
    void loadCatalog();
    api
      .getConfig()
      .then((config) => {
        // `/api/config` is an untyped bag; read the one key we need defensively.
        const preferences = (config as { preferences?: { default_model?: string } }).preferences;
        setDefaultModel(preferences?.default_model || null);
      })
      // Advisory only — without it the rail just falls back to the profile model.
      .catch(() => setDefaultModel(null));
  }, [activeServer, loadCatalog]);

  const entries = useMemo(() => orderEntries(catalog), [catalog]);
  const dirty = Object.values(drafts).some((value) => value.trim().length > 0);
  const connectedCount = entries.filter((entry) => entry.configured).length;
  const modelCount = Object.values(providersHealth ?? {}).reduce(
    (total, entry) => total + (entry.models_available ?? 0),
    0
  );

  const setDraft = (id: string, value: string) =>
    setDrafts((previous) => ({ ...previous, [id]: value }));

  const handleSave = async () => {
    const pending = Object.entries(drafts).filter(([, value]) => value.trim().length > 0);
    if (pending.length === 0) return;

    const confirmed = await confirm({
      title: 'Save provider settings?',
      body: 'Saving updates the server configuration and applies immediately to running models.',
      confirmLabel: 'Save to server',
    });
    if (!confirmed) return;

    setSaving(true);
    try {
      const providers: NonNullable<ConfigUpdate['providers']> = {};
      for (const [id, value] of pending) {
        const entry = entries.find((candidate) => candidate.id === id);
        providers[id] =
          entry?.credential === 'base_url' ? { base_url: value.trim() } : { api_key: value.trim() };
      }
      await api.updateConfig({ providers });
      setDrafts({});
      await loadCatalog();
      refreshHealth();
      setRouteToken((token) => token + 1);
      notifySuccess('Settings saved and applied to the server', 'Providers');
    } catch (error) {
      notifyError(error, 'Failed to save provider settings');
    } finally {
      setSaving(false);
    }
  };

  /**
   * Two genuinely different tests behind one button.
   *
   * A custom endpoint has a URL we can probe directly, including an unsaved one
   * the user is still typing — that's what `/providers/test` is for. A built-in
   * cloud provider has **no** stored base URL (its provider class hardcodes the
   * address), so there is nothing to point a probe at; its real reachability
   * check is the same health ping the dashboard uses. Re-running that and
   * reporting the result keeps one honest button instead of a dead one.
   */
  const handleTest = async (entry: ProviderCatalogEntry): Promise<ProviderTestResult | null> => {
    const draft = drafts[entry.id]?.trim();
    const baseUrl = entry.credential === 'base_url' ? draft || entry.base_url : entry.base_url;

    if (baseUrl) {
      try {
        return await api.testProvider({
          base_url: baseUrl,
          api_key: entry.credential === 'base_url' ? undefined : draft || undefined,
          id: entry.id,
        });
      } catch (error) {
        notifyError(error, "Couldn't test the connection");
        return null;
      }
    }

    if (!entry.configured) {
      notifyError(
        entry.credential === 'base_url'
          ? 'Add a server URL first, then test it.'
          : 'Add a key first, then test it.',
        "Can't test this connection"
      );
      return null;
    }

    const started = performance.now();
    try {
      const health = await api.checkProvidersHealth();
      refreshHealth();
      const result = health.providers[entry.id];
      return {
        reachable: result?.status === 'healthy',
        base_url: entry.label,
        elapsed_ms: Math.round(performance.now() - started),
        models_available: result?.models_available ?? 0,
        models: [],
        error: result?.error ?? (result ? null : 'provider not reported'),
      };
    } catch (error) {
      notifyError(error, "Couldn't test the connection");
      return null;
    }
  };

  const handleDelete = async (entry: ProviderCatalogEntry) => {
    const confirmed = await confirm({
      title: `Remove ${entry.label}?`,
      body:
        `AgentX will forget this endpoint and its key. Agents still pointing at ` +
        `${entry.id}:… fall back to their own model rather than failing.`,
      confirmLabel: 'Remove',
      danger: true,
    });
    if (!confirmed) return;

    try {
      const result = await api.deleteCustomProvider(entry.id);
      await loadCatalog();
      refreshHealth();
      setRouteToken((token) => token + 1);
      notifySuccess(
        result.referencing_profiles.length > 0
          ? `${entry.label} removed. ${result.referencing_profiles.join(', ')} will fall back.`
          : `${entry.label} removed`,
        'Providers'
      );
    } catch (error) {
      notifyError(error, `Couldn't remove ${entry.label}`);
    }
  };

  const device = deviceLabel(health?.compute?.device);
  const embeddingModel = health?.embeddings?.model ?? 'BAAI/bge-m3';
  const embeddingProvider = health?.embeddings?.provider ?? 'local';
  const embeddingDims = health?.embeddings?.dimensions;
  const translationModel =
    (health?.translation?.models?.translation as string | undefined) ??
    'facebook/nllb-200-distilled-600M';
  const translationLoaded = health?.translation?.status === 'healthy';

  const onDeviceCard = (
    title: string,
    icon: ReactNode,
    description: string,
    note: string,
    facts: [string, string | number][]
  ) => (
    <Card className="provider-card ondevice-card">
      <div className="provider-header">
        <div className="provider-info">
          <div className="provider-icon local">{icon}</div>
          <div>
            <h3 className="provider-name">
              {title}
              <Badge variant="neutral" size="sm">
                Local · Locked
              </Badge>
            </h3>
            <p className="provider-description">{description}</p>
            <p className="provider-note">{note}</p>
          </div>
        </div>
      </div>
      <div className="ondevice-meta">
        {facts.map(([label, value]) => (
          <span key={label}>
            <b>{label}</b> {value}
          </span>
        ))}
      </div>
    </Card>
  );

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Key size={20} />}
        title="Model Providers"
        description="Where AgentX gets its intelligence."
        actions={
          <>
            {dirty && <span className="text-warning text-xs">Unsaved changes</span>}
            <Button variant="primary" onClick={handleSave} loading={saving} disabled={!dirty || saving}>
              <Upload size={16} />
              Save to server
            </Button>
          </>
        }
      />

      {!activeServer ? (
        <Card className="empty-state">
          <Server size={32} />
          <p>Select a server first to configure providers</p>
        </Card>
      ) : (
        <>
          {catalogError && (
            <div className="config-warning">
              <AlertTriangle size={16} />
              <span>Couldn&rsquo;t load providers — {catalogError}</span>
            </div>
          )}

          <SupplyLine
            model={activeProfile?.defaultModel ?? null}
            fallbackModel={defaultModel}
            agentName={activeProfile?.name ?? 'Your agent'}
            refreshToken={routeToken}
          />

          <div className="connection-tally">
            <span>
              <b>{connectedCount}</b> of {entries.length} connected
            </span>
            {modelCount > 0 && (
              <span>
                <b>{modelCount.toLocaleString()}</b> models available
              </span>
            )}
          </div>

          <div className="providers-list">
            {entries.map((entry) => (
              <ProviderConnectionCard
                key={entry.id}
                entry={entry}
                presentation={presentationFor(entry)}
                health={providersHealth?.[entry.id]}
                healthLoading={healthLoading}
                blocked={entry.id === 'lmstudio' && onRemoteCluster}
                draft={drafts[entry.id] ?? ''}
                onDraftChange={(value) => setDraft(entry.id, value)}
                onTest={() => handleTest(entry)}
                onDelete={entry.builtin ? undefined : () => void handleDelete(entry)}
              />
            ))}

            <button type="button" className="connection-add" onClick={() => setSheetOpen(true)}>
              <span className="connection-add-icon" aria-hidden>
                <Plus size={18} />
              </span>
              <span className="connection-add-text">
                <span className="connection-add-title">Connect an endpoint</span>
                <span className="connection-add-sub">
                  Groq, Together, DeepSeek, Ollama, vLLM, or your own
                </span>
              </span>
            </button>
          </div>

          <EndpointSheet
            open={sheetOpen}
            onOpenChange={setSheetOpen}
            takenIds={entries.map((entry) => entry.id)}
            onSaved={() => {
              void loadCatalog();
              refreshHealth();
              setRouteToken((token) => token + 1);
            }}
          />

          {/* On-device engines — read-only, locked models. */}
          <SectionHeader
            icon={<Cpu size={20} />}
            title="On-device processing"
            description="Local embedding and translation engines. Locked models; they run on GPU when CUDA is available."
          />
          <div className="providers-list">
            {onDeviceCard(
              'Embeddings',
              <Boxes size={20} />,
              'Semantic memory storage and recall.',
              'Cloud embeddings (still BGE-M3) coming later.',
              [
                ['Model', embeddingModel],
                ['Provider', embeddingProvider],
                ...(embeddingDims != null
                  ? ([['Dimensions', embeddingDims]] as [string, string | number][])
                  : []),
                ['Device', device],
              ]
            )}
            {onDeviceCard(
              'Translation',
              <Languages size={20} />,
              'NLLB-200 distilled — 200+ languages.',
              'Cloud translation to be decided.',
              [
                ['Model', translationModel],
                ['Device', device],
                ['Status', translationLoaded ? 'loaded' : 'loads on first use'],
              ]
            )}
          </div>
        </>
      )}
    </div>
  );
}
