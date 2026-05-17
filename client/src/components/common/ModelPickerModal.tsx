/**
 * ModelPickerModal — Phase 18.4 fullscreen filterable model picker.
 *
 * Replaces the inline ModelSelector in the agent-profile editor. Modeled on
 * ToolkitPage (backdrop + motion + body-overflow lock). Filters by provider
 * and by capability flag (tools / vision / json-mode / streaming), with a
 * search box and rich per-row metadata.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, RefreshCw, Search, Check, Wrench, Eye, Braces, Zap, Loader2,
} from 'lucide-react';
import type { ModelInfo } from '../../lib/api';
import {
  fetchModelsOnce, invalidateModelCache,
} from './ModelSelector';
import { ParallaxBackground } from '../unified-settings/animations/ParallaxBackground';
import {
  backdropVariants, containerVariants,
} from '../unified-settings/animations/transitions';
import './ModelPickerModal.css';

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  lmstudio: 'LM Studio',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  vercel: 'Vercel Gateway',
};

const KNOWN_PROVIDER_ORDER = ['anthropic', 'openrouter', 'vercel', 'lmstudio', 'openai'];

type CapabilityKey = 'tools' | 'vision' | 'json' | 'streaming';

const CAPABILITIES: { key: CapabilityKey; label: string; icon: React.ReactNode; match: (m: ModelInfo) => boolean }[] = [
  { key: 'tools',     label: 'Tools',     icon: <Wrench size={13} />, match: m => !!m.supports_tools },
  { key: 'vision',    label: 'Vision',    icon: <Eye size={13} />,    match: m => !!m.supports_vision },
  { key: 'json',      label: 'JSON mode', icon: <Braces size={13} />, match: m => !!m.supports_json_mode },
  { key: 'streaming', label: 'Streaming', icon: <Zap size={13} />,    match: m => m.supports_streaming !== false },
];

interface ModelPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  value: string;
  onChange: (modelId: string) => void;
  /** Show "System default" row at the top */
  showDefault?: boolean;
}

function formatContext(n?: number | null): string {
  if (!n) return '';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M ctx`;
  if (n >= 1000) return `${Math.round(n / 1000)}k ctx`;
  return `${n} ctx`;
}

function formatMaxOut(n?: number | null): string {
  if (!n) return '';
  if (n >= 1000) return `${Math.round(n / 1000)}k out`;
  return `${n} out`;
}

function formatPrice(input?: number | null, output?: number | null, currency = 'USD'): string {
  if (input == null && output == null) return '';
  const sym = currency === 'USD' ? '$' : '';
  // values are per-1k tokens → show per-1M for readability
  const fmt = (v: number) => {
    const per1m = v * 1000;
    if (per1m < 0.01) return `${sym}${per1m.toFixed(4)}`;
    if (per1m < 1) return `${sym}${per1m.toFixed(3)}`;
    return `${sym}${per1m.toFixed(2)}`;
  };
  const i = input != null ? fmt(input) : '—';
  const o = output != null ? fmt(output) : '—';
  return `${i} / ${o} per 1M`;
}

export function ModelPickerModal({
  isOpen, onClose, value, onChange, showDefault = true,
}: ModelPickerModalProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedProviders, setSelectedProviders] = useState<Set<string>>(new Set());
  const [selectedCaps, setSelectedCaps] = useState<Set<CapabilityKey>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [pending, setPending] = useState<string>(value);

  // Sync pending selection whenever the modal opens with a new value
  useEffect(() => { if (isOpen) setPending(value); }, [isOpen, value]);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setLoading(true);
    fetchModelsOnce().then(m => {
      if (cancelled) return;
      setModels(m);
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (isOpen) document.body.style.overflow = 'hidden';
    else document.body.style.overflow = '';
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  const refresh = useCallback(() => {
    setLoading(true);
    invalidateModelCache();
    fetchModelsOnce().then(m => {
      setModels(m);
      setLoading(false);
    });
  }, []);

  const providers = useMemo(() => {
    const set = new Set(models.map(m => m.provider));
    return Array.from(set).sort((a, b) => {
      const ai = KNOWN_PROVIDER_ORDER.indexOf(a);
      const bi = KNOWN_PROVIDER_ORDER.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
  }, [models]);

  const filtered = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return models.filter(m => {
      if (selectedProviders.size && !selectedProviders.has(m.provider)) return false;
      for (const cap of selectedCaps) {
        const spec = CAPABILITIES.find(c => c.key === cap);
        if (spec && !spec.match(m)) return false;
      }
      if (q && !(m.name.toLowerCase().includes(q) || (m.description ?? '').toLowerCase().includes(q))) {
        return false;
      }
      return true;
    });
  }, [models, selectedProviders, selectedCaps, searchQuery]);

  const toggleProvider = (p: string) => {
    setSelectedProviders(prev => {
      const next = new Set(prev);
      if (next.has(p)) next.delete(p);
      else next.add(p);
      return next;
    });
  };
  const toggleCap = (k: CapabilityKey) => {
    setSelectedCaps(prev => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };

  const handleConfirm = () => {
    onChange(pending);
    onClose();
  };

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="model-picker-backdrop"
            variants={backdropVariants}
            initial="initial" animate="animate" exit="exit"
            transition={{ duration: 0.25 }}
            onClick={onClose}
          />
          <motion.div
            className="model-picker-container"
            variants={containerVariants}
            initial="initial" animate="animate" exit="exit"
          >
            <ParallaxBackground />
            <div className="model-picker-header">
              <div className="header-left">
                <h1>Choose model</h1>
                <p>{loading ? 'Loading…' : `${filtered.length} of ${models.length} models`}</p>
              </div>
              <div className="header-right">
                <button className="mp-button" onClick={refresh} disabled={loading} title="Refresh models">
                  <RefreshCw size={14} className={loading ? 'spin' : ''} />
                  <span>Refresh</span>
                </button>
                <button className="mp-button" onClick={onClose} title="Close">
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="model-picker-layout">
              <aside className="model-picker-filters">
                <div className="filter-group">
                  <h3>Providers</h3>
                  {providers.length === 0 && !loading && (
                    <div className="filter-empty">No providers configured</div>
                  )}
                  {providers.map(p => {
                    const count = models.filter(m => m.provider === p).length;
                    const on = selectedProviders.has(p);
                    return (
                      <button
                        key={p}
                        type="button"
                        className={`filter-chip ${on ? 'on' : ''}`}
                        onClick={() => toggleProvider(p)}
                      >
                        <span>{PROVIDER_LABELS[p] ?? p}</span>
                        <span className="filter-chip-count">{count}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="filter-group">
                  <h3>Capabilities</h3>
                  {CAPABILITIES.map(c => {
                    const on = selectedCaps.has(c.key);
                    return (
                      <button
                        key={c.key}
                        type="button"
                        className={`filter-chip ${on ? 'on' : ''}`}
                        onClick={() => toggleCap(c.key)}
                      >
                        {c.icon}
                        <span>{c.label}</span>
                      </button>
                    );
                  })}
                </div>
                {(selectedProviders.size > 0 || selectedCaps.size > 0) && (
                  <button
                    type="button"
                    className="filter-clear"
                    onClick={() => { setSelectedProviders(new Set()); setSelectedCaps(new Set()); }}
                  >
                    Clear filters
                  </button>
                )}
              </aside>

              <div className="model-picker-main">
                <div className="model-picker-search">
                  <Search size={14} className="search-icon" />
                  <input
                    type="text"
                    placeholder="Search by name or description…"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                  />
                  {searchQuery && (
                    <button type="button" className="search-clear" onClick={() => setSearchQuery('')}>×</button>
                  )}
                </div>

                <div className="model-picker-list">
                  {loading ? (
                    <div className="model-picker-status">
                      <Loader2 size={16} className="spin" />
                      <span>Loading models…</span>
                    </div>
                  ) : (
                    <>
                      {showDefault && (
                        <ModelRow
                          selected={pending === ''}
                          onClick={() => setPending('')}
                          title="System default"
                          subtitle="Use the system's default model"
                        />
                      )}
                      {filtered.length === 0 ? (
                        <div className="model-picker-status">No models match the current filters.</div>
                      ) : filtered.map(m => (
                        <ModelRow
                          key={m.id}
                          selected={pending === m.id}
                          onClick={() => setPending(m.id)}
                          title={m.name}
                          subtitle={m.description ?? undefined}
                          provider={PROVIDER_LABELS[m.provider] ?? m.provider}
                          badges={[
                            formatContext(m.context_length ?? m.context_window),
                            formatMaxOut(m.max_output_tokens ?? undefined),
                            formatPrice(m.cost_per_1k_input, m.cost_per_1k_output, m.pricing_currency),
                          ].filter(Boolean)}
                          caps={CAPABILITIES.filter(c => c.match(m))}
                        />
                      ))}
                    </>
                  )}
                </div>

                <div className="model-picker-footer">
                  <button type="button" className="mp-button" onClick={onClose}>Cancel</button>
                  <button type="button" className="mp-button primary" onClick={handleConfirm}>
                    <Check size={14} />
                    <span>Use this model</span>
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body,
  );
}

interface ModelRowProps {
  selected: boolean;
  onClick: () => void;
  title: string;
  subtitle?: string;
  provider?: string;
  badges?: string[];
  caps?: typeof CAPABILITIES;
}

function ModelRow({ selected, onClick, title, subtitle, provider, badges, caps }: ModelRowProps) {
  return (
    <button
      type="button"
      className={`model-row ${selected ? 'selected' : ''}`}
      onClick={onClick}
    >
      <div className="model-row-main">
        <div className="model-row-title-line">
          <span className="model-row-title">{title}</span>
          {provider && <span className="model-row-provider">{provider}</span>}
          {selected && <Check size={14} className="model-row-check" />}
        </div>
        {subtitle && <div className="model-row-subtitle" title={subtitle}>{subtitle}</div>}
      </div>
      <div className="model-row-meta">
        {(badges ?? []).map((b, i) => (
          <span key={i} className="model-row-badge">{b}</span>
        ))}
        {(caps ?? []).map(c => (
          <span key={c.key} className="model-row-cap" title={c.label}>{c.icon}</span>
        ))}
      </div>
    </button>
  );
}
