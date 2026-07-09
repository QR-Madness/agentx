/**
 * ModelsSection ("Model Limits") — local model context limits + compaction.
 *
 * Context limits are built on the settings field kit + autosave
 * (useSettingsAutosave); the diff is top-level-key based, so nested edits
 * replace the whole `lmstudio` object. (The global default model lives in the
 * Model Roles section.)
 */

import { useState } from 'react';
import {
  Layers,
  Server,
  RefreshCw,
  AlertTriangle,
  Gauge,
  Trash2,
} from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, Card, Badge, SectionHeader, IconButton } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import { NumberField, SliderField, SaveStatusChip } from '../../settings/fields';

interface ContextLimits extends Record<string, unknown> {
  lmstudio: { context_window: number; max_output_tokens: number };
  models: Record<string, { context_window: number; max_output_tokens: number }>;
}

// Compaction knobs (context.* config). Sizing the verbatim window + the
// rolling-summary trigger point so the digest kicks in near the REAL budget.
interface ContextRatios extends Record<string, unknown> {
  verbatim_budget_ratio: number;
  summary_trigger_ratio: number;
  recent_floor: number;
}

const RATIO_DEFAULTS: ContextRatios = {
  verbatim_budget_ratio: 0.9,
  summary_trigger_ratio: 0.85,
  recent_floor: 4,
};

// Sane default for a fresh per-model override — the user tunes it to their
// model's real window (the whole point of this escape hatch).
const NEW_OVERRIDE_WINDOW = 200_000;
const NEW_OVERRIDE_MAX_OUTPUT = 8_192;

export default function ModelsSection() {
  const { notifyError } = useNotify();

  // Context limits — autosave (baseline-diff on top-level keys, so edits
  // replace the whole `lmstudio` object).
  const { settings, loading, status, update, refresh } = useSettingsAutosave<ContextLimits>({
    load: () => api.getContextLimits(),
    save: async changed => {
      await api.updateContextLimits(changed);
    },
    onError: err => notifyError(err, 'Context limits'),
  });

  // Compaction knobs (context.* — verbatim window + rolling-summary trigger).
  const {
    settings: ratios,
    update: updateRatios,
    status: ratioStatus,
  } = useSettingsAutosave<ContextRatios>({
    load: async () => {
      const cfg = await api.getConfig();
      const c = (cfg.context ?? {}) as Partial<ContextRatios>;
      return {
        verbatim_budget_ratio: c.verbatim_budget_ratio ?? RATIO_DEFAULTS.verbatim_budget_ratio,
        summary_trigger_ratio: c.summary_trigger_ratio ?? RATIO_DEFAULTS.summary_trigger_ratio,
        recent_floor: c.recent_floor ?? RATIO_DEFAULTS.recent_floor,
      };
    },
    save: async changed => {
      await api.updateConfig({ context: changed });
    },
    onError: err => notifyError(err, 'Context compaction'),
  });

  // Per-model context-window overrides (escape hatch for any provider — e.g. an
  // OpenRouter `:latest` route that reports no window and falls back to ~8k).
  const [newModelId, setNewModelId] = useState('');
  const models = settings?.models ?? {};

  const addModelOverride = (modelId: string) => {
    if (!modelId || models[modelId]) return;
    update({
      models: {
        ...models,
        [modelId]: {
          context_window: NEW_OVERRIDE_WINDOW,
          max_output_tokens: NEW_OVERRIDE_MAX_OUTPUT,
        },
      },
    });
    setNewModelId('');
  };

  const editModelOverride = (
    modelId: string,
    patch: Partial<{ context_window: number; max_output_tokens: number }>,
  ) => {
    update({ models: { ...models, [modelId]: { ...models[modelId], ...patch } } });
  };

  const removeModelOverride = async (modelId: string) => {
    try {
      await api.updateContextLimits({ models: { [modelId]: null } });
      await refresh();
    } catch (err) {
      notifyError(err, 'Failed to remove the override');
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Layers size={20} />}
        title="Model Context Limits"
        description="LM Studio hardware limits, plus per-model overrides for any provider (e.g. an OpenRouter :latest route that reports no window and falls back to ~8k)"
        actions={<SaveStatusChip status={status} />}
      />

      {loading ? (
        <Card className="empty-state">
          <RefreshCw size={24} className="spin" />
          <p>Loading context limits...</p>
        </Card>
      ) : settings ? (
        <div className="providers-list">
          <Card className="provider-card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon local">
                  <Server size={20} />
                </div>
                <div>
                  <h3 className="provider-name">
                    LM Studio
                    <Badge variant="neutral" size="sm">Local</Badge>
                  </h3>
                  <p className="provider-description">
                    Hardware limits for local models (API providers use their own per-model capabilities)
                  </p>
                </div>
              </div>
            </div>
            <div className="context-limits-form">
              <NumberField
                label="Context Window (tokens)"
                value={settings.lmstudio.context_window}
                min={1024}
                max={1000000}
                fallback={1024}
                onChange={v =>
                  update({ lmstudio: { ...settings.lmstudio, context_window: v } })
                }
                title={`≈ ${(settings.lmstudio.context_window / 1000).toFixed(0)}k tokens`}
              />
              <NumberField
                label="Max Output Tokens"
                value={settings.lmstudio.max_output_tokens}
                min={256}
                max={131072}
                fallback={256}
                onChange={v =>
                  update({ lmstudio: { ...settings.lmstudio, max_output_tokens: v } })
                }
                title={`≈ ${(settings.lmstudio.max_output_tokens / 1000).toFixed(0)}k tokens`}
              />
            </div>
          </Card>

          <Card className="provider-card">
            <div className="provider-header">
              <div className="provider-info">
                <div className="provider-icon">
                  <Layers size={20} />
                </div>
                <div>
                  <h3 className="provider-name">Per-Model Overrides</h3>
                  <p className="provider-description">
                    Force a model's context window when its provider reports the wrong one.
                    Leave unset to use the provider's own capability.
                  </p>
                </div>
              </div>
            </div>

            {Object.keys(models).length > 0 && (
              <div className="model-overrides-list">
                {Object.entries(models).map(([modelId, limits]) => (
                  <div key={modelId} className="model-override-row">
                    <div className="model-override-head">
                      <span className="model-override-id font-mono">{modelId}</span>
                      <IconButton
                        size="sm"
                        tone="danger"
                        aria-label={`Remove override for ${modelId}`}
                        onClick={() => void removeModelOverride(modelId)}
                      >
                        <Trash2 size={16} />
                      </IconButton>
                    </div>
                    <div className="context-limits-form">
                      <NumberField
                        label="Context Window (tokens)"
                        value={limits.context_window}
                        min={1024}
                        max={2000000}
                        fallback={NEW_OVERRIDE_WINDOW}
                        onChange={v => editModelOverride(modelId, { context_window: v })}
                        title={`≈ ${(limits.context_window / 1000).toFixed(0)}k tokens`}
                      />
                      <NumberField
                        label="Max Output Tokens"
                        value={limits.max_output_tokens}
                        min={256}
                        max={200000}
                        fallback={NEW_OVERRIDE_MAX_OUTPUT}
                        onChange={v => editModelOverride(modelId, { max_output_tokens: v })}
                        title={`≈ ${(limits.max_output_tokens / 1000).toFixed(0)}k tokens`}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            <ModelPickerField
              value={newModelId}
              onChange={addModelOverride}
              label="Add an override"
              placeholder="Pick a model…"
              hint="Select a model to pin its context window."
            />
          </Card>
        </div>
      ) : (
        <Card className="empty-state">
          <AlertTriangle size={32} />
          <p>Failed to load context limits</p>
          <Button variant="secondary" onClick={() => void refresh()}>
            <RefreshCw size={16} />
            Retry
          </Button>
        </Card>
      )}

      <SectionHeader
        icon={<Gauge size={20} />}
        title="Context Compaction"
        description="When the rolling summary kicks in and how much of the window stays verbatim. Set these against the model's REAL window — with a correct window, most chats never compact."
        actions={<SaveStatusChip status={ratioStatus} />}
      />
      {ratios && (
        <div className="providers-list">
          <Card className="provider-card">
            <div className="context-limits-form">
              <SliderField
                label="Verbatim window budget"
                value={ratios.verbatim_budget_ratio}
                min={0.5}
                max={0.98}
                step={0.01}
                onChange={v => updateRatios({ verbatim_budget_ratio: v })}
                format={v => `${Math.round(v * 100)}%`}
              />
              <SliderField
                label="Rolling-summary trigger"
                value={ratios.summary_trigger_ratio}
                min={0.5}
                max={0.98}
                step={0.01}
                onChange={v => updateRatios({ summary_trigger_ratio: v })}
                format={v => `${Math.round(v * 100)}%`}
              />
              <NumberField
                label="Recent turns kept verbatim (floor)"
                value={ratios.recent_floor}
                min={1}
                max={50}
                fallback={RATIO_DEFAULTS.recent_floor}
                onChange={v => updateRatios({ recent_floor: v })}
              />
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
