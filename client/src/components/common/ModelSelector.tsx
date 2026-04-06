/**
 * ModelSelector — Reusable model selection component with provider tabs
 *
 * Fetches available models from the API and displays them grouped by provider.
 * Supports optional "System default" option and compact mode for settings forms.
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { RefreshCw, Search } from 'lucide-react';
import { api, type ModelInfo } from '../../lib/api';
import './ModelSelector.css';

/** Shared model cache so multiple selectors don't each fire a request */
let modelCache: ModelInfo[] | null = null;
let modelCachePromise: Promise<ModelInfo[]> | null = null;

function fetchModelsOnce(): Promise<ModelInfo[]> {
  if (modelCache) return Promise.resolve(modelCache);
  if (!modelCachePromise) {
    modelCachePromise = api
      .listModels()
      .then(({ models }) => {
        modelCache = models;
        return models;
      })
      .catch((err) => {
        console.error('Failed to fetch models:', err);
        modelCachePromise = null;
        return [];
      });
  }
  return modelCachePromise;
}

/** Invalidate the shared cache so the next mount re-fetches */
export function invalidateModelCache() {
  modelCache = null;
  modelCachePromise = null;
}

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  lmstudio: 'LM Studio',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
};

const KNOWN_PROVIDER_ORDER = ['anthropic', 'openrouter', 'lmstudio', 'openai'];

interface ModelSelectorProps {
  /** Currently selected model id (empty string = system default) */
  value: string;
  /** Called when the user picks a model */
  onChange: (modelId: string) => void;
  /** Called when the selected provider changes (useful for settings that store provider separately) */
  onProviderChange?: (provider: string) => void;
  /** Show "System default" radio at top (default true) */
  showDefault?: boolean;
  /** Compact single-line display for settings rows (default false) */
  compact?: boolean;
  /** Optional label shown above the selector */
  label?: string;
  /** Lock to a single provider (hides tabs) */
  provider?: string;
}

export function ModelSelector({
  value,
  onChange,
  onProviderChange,
  showDefault = true,
  compact = false,
  label,
  provider: lockedProvider,
}: ModelSelectorProps) {
  const [models, setModels] = useState<ModelInfo[]>(modelCache ?? []);
  const [loading, setLoading] = useState(!modelCache);
  const [selectedProvider, setSelectedProvider] = useState(lockedProvider ?? '');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch models (uses shared cache)
  useEffect(() => {
    let cancelled = false;
    fetchModelsOnce().then((m) => {
      if (cancelled) return;
      setModels(m);
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
    invalidateModelCache();
    fetchModelsOnce().then((m) => {
      setModels(m);
      setLoading(false);
    });
  }, []);

  // Derive sorted provider list
  const providers = useMemo(() => {
    const set = new Set(models.map((m) => m.provider));
    return Array.from(set).sort((a, b) => {
      const ai = KNOWN_PROVIDER_ORDER.indexOf(a);
      const bi = KNOWN_PROVIDER_ORDER.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
  }, [models]);

  // Auto-select provider from current value, or first available
  useEffect(() => {
    if (lockedProvider) { setSelectedProvider(lockedProvider); return; }
    if (loading) return;
    if (value) {
      const match = models.find((m) => m.id === value);
      if (match) { setSelectedProvider(match.provider); return; }
    }
    if (!selectedProvider || !providers.includes(selectedProvider)) {
      setSelectedProvider(providers[0] ?? '');
    }
  }, [loading, value, models, providers, lockedProvider]);

  // Notify parent when provider changes
  useEffect(() => {
    if (selectedProvider && onProviderChange) {
      onProviderChange(selectedProvider);
    }
  }, [selectedProvider, onProviderChange]);

  const filtered = useMemo(() => {
    let result = selectedProvider ? models.filter((m) => m.provider === selectedProvider) : [];
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter((m) => m.name.toLowerCase().includes(query));
    }
    return result;
  }, [models, selectedProvider, searchQuery]);

  const formatContext = (n?: number) => (n ? `${(n / 1000).toFixed(0)}k context` : '');

  // ── Compact mode: dropdown ──
  if (compact) {
    return (
      <div className="model-selector-compact">
        {label && <label className="model-selector-label">{label}</label>}
        <div className="model-selector-compact-row">
          {!lockedProvider && (
            <select
              className="model-selector-provider-select"
              value={selectedProvider}
              onChange={(e) => { setSelectedProvider(e.target.value); onChange(''); }}
              disabled={loading}
            >
              {providers.map((p) => (
                <option key={p} value={p}>{PROVIDER_LABELS[p] ?? p}</option>
              ))}
            </select>
          )}
          <select
            className="model-selector-model-select"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            disabled={loading}
          >
            {showDefault && <option value="">System default</option>}
            {filtered.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}{m.context_length ? ` (${formatContext(m.context_length)})` : ''}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="model-selector-refresh"
            onClick={refresh}
            disabled={loading}
            title="Refresh models"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
          </button>
        </div>
      </div>
    );
  }

  // ── Full mode: provider tabs + radio list ──
  return (
    <div className="model-selector">
      {label && <label className="model-selector-label">{label}</label>}
      <div className="model-selector-tabs">
        {/* Provider tab bar */}
        {!lockedProvider && (
          <div className="model-selector-tab-bar">
            {providers.map((p) => {
              const count = models.filter((m) => m.provider === p).length;
              return (
                <button
                  key={p}
                  type="button"
                  className={`model-selector-tab ${selectedProvider === p ? 'active' : ''}`}
                  onClick={() => setSelectedProvider(p)}
                  disabled={loading}
                >
                  {PROVIDER_LABELS[p] ?? p.charAt(0).toUpperCase() + p.slice(1)}
                  {count > 0 && <span className="model-selector-tab-count">{count}</span>}
                </button>
              );
            })}
            <button
              type="button"
              className="model-selector-tab model-selector-refresh-tab"
              onClick={refresh}
              disabled={loading}
              title="Refresh models"
            >
              <RefreshCw size={13} className={loading ? 'spin' : ''} />
            </button>
          </div>
        )}

        {/* Search input - show for providers with many models */}
        {!loading && models.filter((m) => m.provider === selectedProvider).length > 10 && (
          <div className="model-selector-search">
            <Search size={14} className="model-selector-search-icon" />
            <input
              type="text"
              className="model-selector-search-input"
              placeholder="Search models..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="model-selector-search-clear"
                onClick={() => setSearchQuery('')}
              >
                ×
              </button>
            )}
          </div>
        )}

        {/* Model list */}
        <div className="model-selector-list">
          {loading ? (
            <div className="model-selector-status">
              <RefreshCw size={16} className="spin" />
              <span>Loading models...</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="model-selector-status">
              {selectedProvider
                ? 'No models available — is the provider running?'
                : 'Select a provider'}
            </div>
          ) : (
            <>
              {showDefault && (
                <label className={`model-selector-option ${value === '' ? 'selected' : ''}`}>
                  <input
                    type="radio"
                    name="model-selector"
                    value=""
                    checked={value === ''}
                    onChange={() => onChange('')}
                  />
                  <span className="model-selector-option-body">
                    <span className="model-selector-option-name">System default</span>
                    <span className="model-selector-option-meta">Use the system's default model</span>
                  </span>
                </label>
              )}
              {filtered.map((m) => (
                <label
                  key={m.id}
                  className={`model-selector-option ${value === m.id ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="model-selector"
                    value={m.id}
                    checked={value === m.id}
                    onChange={() => onChange(m.id)}
                  />
                  <span className="model-selector-option-body">
                    <span className="model-selector-option-name">{m.name}</span>
                    {m.context_length && (
                      <span className="model-selector-option-meta">
                        {formatContext(m.context_length)}
                      </span>
                    )}
                  </span>
                </label>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
