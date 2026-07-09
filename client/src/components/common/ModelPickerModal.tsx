/**
 * ModelPickerModal — Phase 18.4 fullscreen filterable model picker.
 *
 * Replaces the inline ModelSelector in the agent-profile editor. Modeled on
 * ToolkitPage (backdrop + motion + body-overflow lock). Filters by provider
 * and by capability flag (tools / vision / json-mode / streaming), with a
 * search box and rich per-row metadata.
 *
 * Built for large catalogs (OpenRouter can be 500+ models): per-model derived
 * data (capabilities, badges, search haystack) is computed once per catalog,
 * search filtering is deferred via useDeferredValue, rows are memoized, and
 * off-screen rows skip layout/paint via `content-visibility` in the CSS.
 */

import {
  memo, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState,
} from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, RefreshCw, Search, Check, Wrench, Eye, Braces, Zap, Loader2, Volume2, Mic,
  Image as ImageIcon, AlertTriangle,
} from 'lucide-react';
import type { ModelInfo } from '../../lib/api';
import {
  fetchModelsOnce, invalidateModelCache, pushRecentModel, readRecentModels, writeRecentModel,
} from './modelCatalog';
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

type CapabilityKey = 'tools' | 'vision' | 'image' | 'json' | 'streaming' | 'speech' | 'transcription';

// Image *output* (generation): the model emits images (`output_modalities` includes "image"),
// distinct from `vision` which is image *input*.
const _outputsImage = (m: ModelInfo): boolean =>
  !!m.supports_image || !!m.output_modalities?.includes('image');

const CAPABILITIES: { key: CapabilityKey; label: string; icon: React.ReactNode; match: (m: ModelInfo) => boolean }[] = [
  { key: 'tools',         label: 'Tools',     icon: <Wrench size={13} />, match: m => !!m.supports_tools },
  { key: 'vision',        label: 'Vision',    icon: <Eye size={13} />,    match: m => !!m.supports_vision },
  { key: 'image',         label: 'Image gen', icon: <ImageIcon size={13} />, match: _outputsImage },
  { key: 'json',          label: 'JSON mode', icon: <Braces size={13} />, match: m => !!m.supports_json_mode },
  { key: 'streaming',     label: 'Streaming', icon: <Zap size={13} />,    match: m => m.supports_streaming !== false },
  { key: 'speech',        label: 'Speech',    icon: <Volume2 size={13} />, match: m => !!m.supports_speech },
  { key: 'transcription', label: 'Transcribe', icon: <Mic size={13} />,   match: m => !!m.supports_transcription },
];

type CapabilitySpec = (typeof CAPABILITIES)[number];

/** Per-model derived data, computed once per catalog (not per keystroke / per row render). */
interface PreparedModel {
  id: string;
  name: string;
  description?: string;
  provider: string;
  providerLabel: string;
  badges: string[];
  caps: CapabilitySpec[];
  capKeys: ReadonlySet<CapabilityKey>;
  /** Lowercase search haystack — includes the model id so id fragments match. */
  haystack: string;
  /** OpenRouter `:latest` route — reports no context window (falls back to ~8k)
   *  and can break image generation. Flagged so the row can warn. */
  warnLatest: boolean;
}

interface ModelPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  value: string;
  onChange: (modelId: string) => void;
  /** Show "System default" row at the top */
  showDefault?: boolean;
  /** Restrict the list to models matching this capability (e.g. 'speech' for TTS). */
  requireCapability?: CapabilityKey;
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

function prepareModel(m: ModelInfo): PreparedModel {
  const caps = CAPABILITIES.filter(c => c.match(m));
  return {
    id: m.id,
    name: m.name,
    description: m.description ?? undefined,
    provider: m.provider,
    providerLabel: PROVIDER_LABELS[m.provider] ?? m.provider,
    badges: [
      formatContext(m.context_length ?? m.context_window),
      formatMaxOut(m.max_output_tokens ?? undefined),
      formatPrice(m.cost_per_1k_input, m.cost_per_1k_output, m.pricing_currency),
    ].filter(Boolean),
    caps,
    capKeys: new Set(caps.map(c => c.key)),
    haystack: `${m.id} ${m.name} ${m.description ?? ''}`.toLowerCase(),
    warnLatest: m.provider === 'openrouter' && /:latest$/i.test(m.id),
  };
}

export function ModelPickerModal({
  isOpen, onClose, value, onChange, showDefault = true, requireCapability,
}: ModelPickerModalProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedProviders, setSelectedProviders] = useState<Set<string>>(new Set());
  const [selectedCaps, setSelectedCaps] = useState<Set<CapabilityKey>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [pending, setPending] = useState<string>(value);
  const [recents, setRecents] = useState<string[]>([]);
  const listRef = useRef<HTMLDivElement>(null);
  const initialScrollDone = useRef(false);

  // Sync pending selection whenever the modal opens with a new value
  useEffect(() => { if (isOpen) setPending(value); }, [isOpen, value]);
  useEffect(() => {
    if (isOpen) {
      setRecents(readRecentModels());
      initialScrollDone.current = false;
    }
  }, [isOpen]);

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

  // Derived per-model data, computed once per catalog.
  const prepared = useMemo(() => models.map(prepareModel), [models]);
  const preparedById = useMemo(() => new Map(prepared.map(p => [p.id, p])), [prepared]);

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

  const providerCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const m of models) counts[m.provider] = (counts[m.provider] ?? 0) + 1;
    return counts;
  }, [models]);

  // Defer search filtering so typing stays responsive on large catalogs.
  const deferredQuery = useDeferredValue(searchQuery);

  const filtered = useMemo(() => {
    const q = deferredQuery.trim().toLowerCase();
    return prepared.filter(p => {
      if (requireCapability && !p.capKeys.has(requireCapability)) return false;
      if (selectedProviders.size && !selectedProviders.has(p.provider)) return false;
      for (const cap of selectedCaps) {
        if (!p.capKeys.has(cap)) return false;
      }
      if (q && !p.haystack.includes(q)) return false;
      return true;
    });
  }, [prepared, selectedProviders, selectedCaps, deferredQuery, requireCapability]);

  // "Recent" group — only when the list isn't narrowed by search or filters.
  const isNarrowed = searchQuery.trim() !== '' || selectedProviders.size > 0 || selectedCaps.size > 0;
  const recentRows = useMemo(() => {
    if (isNarrowed) return [];
    const out: PreparedModel[] = [];
    for (const id of recents) {
      const p = preparedById.get(id);
      if (p && (!requireCapability || p.capKeys.has(requireCapability))) out.push(p);
    }
    return out;
  }, [isNarrowed, recents, preparedById, requireCapability]);

  // Ordered ids of every visible row ('' = the System default row) for keyboard nav.
  const visibleIds = useMemo(() => {
    const ids: string[] = [];
    if (showDefault) ids.push('');
    for (const p of recentRows) ids.push(p.id);
    for (const p of filtered) ids.push(p.id);
    return ids;
  }, [showDefault, recentRows, filtered]);

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

  const selectRow = useCallback((id: string) => setPending(id), []);
  const confirmWith = useCallback((id: string) => {
    if (id) {
      writeRecentModel(id);
      setRecents(prev => pushRecentModel(prev, id));
    }
    onChange(id);
    onClose();
  }, [onChange, onClose]);

  // Keyboard: Escape closes, ArrowUp/Down move the pending selection (works
  // while the search input is focused — arrows don't type), Enter confirms.
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return; }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (!visibleIds.length) return;
        e.preventDefault();
        const delta = e.key === 'ArrowDown' ? 1 : -1;
        setPending(prev => {
          const idx = visibleIds.indexOf(prev);
          if (idx === -1) return delta === 1 ? visibleIds[0] : visibleIds[visibleIds.length - 1];
          return visibleIds[(idx + delta + visibleIds.length) % visibleIds.length];
        });
        return;
      }
      if (e.key === 'Enter') {
        // Focused buttons (filter chips, footer, refresh) keep their native
        // Enter → click behavior; model rows and the search input confirm.
        const t = e.target as HTMLElement | null;
        if (t instanceof HTMLButtonElement && !t.classList.contains('model-row')) return;
        e.preventDefault();
        confirmWith(pending);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, onClose, visibleIds, confirmWith, pending]);

  // Keep the pending row visible: centered on first paint after load (so the
  // currently-selected model is in view on open), nearest-edge afterwards
  // (keyboard nav; a click never causes a jump since the row is on screen).
  useEffect(() => {
    if (!isOpen || loading) return;
    const block = initialScrollDone.current ? 'nearest' : 'center';
    initialScrollDone.current = true;
    listRef.current
      ?.querySelector(`[data-model-id="${CSS.escape(pending)}"]`)
      ?.scrollIntoView({ block });
  }, [isOpen, loading, pending]);

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
                    const on = selectedProviders.has(p);
                    return (
                      <button
                        key={p}
                        type="button"
                        className={`filter-chip ${on ? 'on' : ''}`}
                        onClick={() => toggleProvider(p)}
                      >
                        <span>{PROVIDER_LABELS[p] ?? p}</span>
                        <span className="filter-chip-count">{providerCounts[p] ?? 0}</span>
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
                    placeholder="Search by name, id, or description…"
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                  />
                  {searchQuery && (
                    <button type="button" className="search-clear" onClick={() => setSearchQuery('')}>×</button>
                  )}
                </div>

                <div className="model-picker-list" ref={listRef}>
                  {loading ? (
                    <div className="model-picker-status">
                      <Loader2 size={16} className="spin" />
                      <span>Loading models…</span>
                    </div>
                  ) : (
                    <>
                      {showDefault && (
                        <ModelRow
                          id=""
                          selected={pending === ''}
                          onSelect={selectRow}
                          onConfirm={confirmWith}
                          title="System default"
                          subtitle="Use the system's default model"
                        />
                      )}
                      {recentRows.length > 0 && (
                        <>
                          <div className="model-list-heading">Recent</div>
                          {recentRows.map(p => (
                            <ModelRow
                              key={`recent:${p.id}`}
                              id={p.id}
                              selected={pending === p.id}
                              onSelect={selectRow}
                              onConfirm={confirmWith}
                              record={p}
                            />
                          ))}
                          <div className="model-list-heading">All models</div>
                        </>
                      )}
                      {filtered.length === 0 ? (
                        <div className="model-picker-status">No models match the current filters.</div>
                      ) : filtered.map(p => (
                        <ModelRow
                          key={p.id}
                          id={p.id}
                          selected={pending === p.id}
                          onSelect={selectRow}
                          onConfirm={confirmWith}
                          record={p}
                        />
                      ))}
                    </>
                  )}
                </div>

                <div className="model-picker-footer">
                  <button type="button" className="mp-button" onClick={onClose}>Cancel</button>
                  <button type="button" className="mp-button primary" onClick={() => confirmWith(pending)}>
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
  id: string;
  selected: boolean;
  onSelect: (id: string) => void;
  /** Double-click (or Enter) confirms the row immediately. */
  onConfirm: (id: string) => void;
  /** Precomputed catalog record; omitted for the synthetic "System default" row. */
  record?: PreparedModel;
  /** Fallback title/subtitle for the synthetic row. */
  title?: string;
  subtitle?: string;
}

const ModelRow = memo(function ModelRow({
  id, selected, onSelect, onConfirm, record, title, subtitle,
}: ModelRowProps) {
  const rowTitle = record?.name ?? title;
  const rowSubtitle = record?.description ?? subtitle;
  return (
    <button
      type="button"
      className={`model-row ${selected ? 'selected' : ''}`}
      data-model-id={id}
      onClick={() => onSelect(id)}
      onDoubleClick={() => onConfirm(id)}
    >
      <div className="model-row-main">
        <div className="model-row-title-line">
          <span className="model-row-title">{rowTitle}</span>
          {record && <span className="model-row-provider">{record.providerLabel}</span>}
          {record?.warnLatest && (
            <span
              className="model-row-warn"
              title="`:latest` routes don't report a context window (falls back to ~8k) and can break image generation — pin a concrete version."
            >
              <AlertTriangle size={13} />
            </span>
          )}
          {selected && <Check size={14} className="model-row-check" />}
        </div>
        {rowSubtitle && <div className="model-row-subtitle" title={rowSubtitle}>{rowSubtitle}</div>}
      </div>
      {record && (
        <div className="model-row-meta">
          {record.badges.map((b, i) => (
            <span key={i} className="model-row-badge">{b}</span>
          ))}
          {record.caps.map(c => (
            <span key={c.key} className="model-row-cap" title={c.label}>{c.icon}</span>
          ))}
        </div>
      )}
    </button>
  );
});
