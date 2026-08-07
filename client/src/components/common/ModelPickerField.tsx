/**
 * ModelPickerField — a settings-row trigger that opens the full ModelPickerModal.
 *
 * Drop-in replacement for the old compact `ModelSelector` dropdown: shows the
 * selected model (name / provider / context badge / capability icons) on a button
 * that opens the proper filterable picker. For settings that store the provider
 * separately (e.g. `extraction_provider`), `onProviderChange` is fired with the
 * picked model's provider.
 *
 * **Manifest chrome.** Model-valued settings are the single most common kind
 * left in the settings area, and this control sat outside the field kit — so
 * every one of them rendered with no default chip, no reset, no help popover,
 * and no `data-setting` anchor, which meant settings search could *find* them
 * and then fail to land on them. Pass a `binding` and it renders through
 * `FieldShell` like the rest of the kit. Pass none — as the profile editors do,
 * where there is no manifest — and the markup is exactly what it always was.
 */

import { useEffect, useState } from 'react';
import { ChevronRight, Wrench, Eye, Image as ImageIcon } from 'lucide-react';
import type { ModelInfo } from '../../lib/api';
import { fetchModelsOnce } from './modelCatalog';
import { ModelPickerModal } from './ModelPickerModal';
import { FieldShell, type FieldChromeProps } from '../settings/fields/FieldShell';
import './ModelPickerField.css';

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  lmstudio: 'LM Studio',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  vercel: 'Vercel Gateway',
};

interface ModelPickerFieldProps extends FieldChromeProps {
  /** Currently selected model id (empty string = inherit / system default) */
  value: string;
  /** Called when the user picks a model */
  onChange: (modelId: string) => void;
  /** Called with the picked model's provider (for settings that store it separately) */
  onProviderChange?: (provider: string) => void;
  /** Show the "System default" row in the picker (default false for settings rows) */
  showDefault?: boolean;
  /** Optional label above the trigger */
  label?: string;
  /** Optional hint below the trigger */
  hint?: string;
  /** Restrict the picker to models with this capability (forwarded to ModelPickerModal) */
  requireCapability?: 'tools' | 'vision' | 'image' | 'json' | 'streaming' | 'speech' | 'transcription';
  /** Trigger text when no model is selected (default "System default") */
  placeholder?: string;
}

export function ModelPickerField({
  value,
  onChange,
  onProviderChange,
  showDefault = false,
  label,
  hint,
  requireCapability,
  placeholder = 'System default',
  binding,
  onReset,
  badge,
}: ModelPickerFieldProps) {
  const [open, setOpen] = useState(false);
  const [catalog, setCatalog] = useState<ModelInfo[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetchModelsOnce().then(m => { if (!cancelled) setCatalog(m); });
    return () => { cancelled = true; };
  }, []);

  const selected = value ? catalog.find(m => m.id === value) : undefined;
  const ctx = selected?.context_length ?? selected?.context_window;
  const selectedName = (() => {
    if (!value) return placeholder;
    if (selected) return selected.name;
    // Model not in the catalog (yet) — fall back to the id, stripping a provider prefix.
    const parts = value.split(':');
    return parts.length > 1 ? parts.slice(1).join(':') : value;
  })();
  const providerLabel = selected
    ? (PROVIDER_LABELS[selected.provider] ?? selected.provider)
    : (value.includes(':') ? value.split(':')[0] : '');

  const handleChange = (modelId: string) => {
    onChange(modelId);
    if (onProviderChange) {
      const picked = catalog.find(m => m.id === modelId);
      if (picked) onProviderChange(picked.provider);
    }
  };

  const trigger = (ids?: { id: string; describedBy?: string }) => (
    <button
      type="button"
      id={ids?.id}
      aria-describedby={ids?.describedBy}
      className="mpf-trigger"
      onClick={() => setOpen(true)}
    >
      <div className="mpf-trigger-main">
        <span className="mpf-trigger-name">{selectedName}</span>
        {providerLabel && <span className="mpf-trigger-provider">{providerLabel}</span>}
      </div>
      <div className="mpf-trigger-meta">
        {ctx && (
          <span className="mpf-trigger-badge">
            {ctx >= 1000 ? `${Math.round(ctx / 1000)}k ctx` : `${ctx} ctx`}
          </span>
        )}
        {selected?.supports_tools && (
          <span className="mpf-trigger-cap" title="Tools"><Wrench size={12} /></span>
        )}
        {selected?.supports_vision && (
          <span className="mpf-trigger-cap" title="Vision"><Eye size={12} /></span>
        )}
        {(selected?.supports_image || selected?.output_modalities?.includes('image')) && (
          <span className="mpf-trigger-cap" title="Image generation"><ImageIcon size={12} /></span>
        )}
        <ChevronRight size={14} className="mpf-trigger-chev" />
      </div>
    </button>
  );

  const picker = (
    <ModelPickerModal
      isOpen={open}
      onClose={() => setOpen(false)}
      value={value}
      onChange={handleChange}
      showDefault={showDefault}
      requireCapability={requireCapability}
    />
  );

  // Bound to the manifest: FieldShell owns the label row (dot, help, reset) and
  // the anchor. `mpf-field` stays on for the picker's own spacing rules.
  if (binding) {
    return (
      <>
        <FieldShell
          className="setting-row mpf-field"
          label={label ?? ''}
          labelText={label ?? binding.entry.key}
          hint={hint}
          binding={binding}
          onReset={onReset}
          badge={badge}
        >
          {trigger}
        </FieldShell>
        {picker}
      </>
    );
  }

  return (
    <div className="mpf-field">
      {label && <span className="mpf-label">{label}</span>}
      {trigger()}
      {hint && <span className="mpf-hint">{hint}</span>}
      {picker}
    </div>
  );
}
