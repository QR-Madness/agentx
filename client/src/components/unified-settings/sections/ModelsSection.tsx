/**
 * ModelsSection — global default model + local model context limits.
 *
 * The default-model picker saves immediately (optimistic, with revert).
 * Context limits are built on the settings field kit + autosave
 * (useSettingsAutosave); the diff is top-level-key based, so nested edits
 * replace the whole `lmstudio` object.
 */

import { useState, useEffect } from 'react';
import {
  Layers,
  Server,
  RefreshCw,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, Card, Badge, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import { NumberField, SaveStatusChip } from '../../settings/fields';

interface ContextLimits extends Record<string, unknown> {
  lmstudio: { context_window: number; max_output_tokens: number };
  models: Record<string, { context_window: number; max_output_tokens: number }>;
}

export default function ModelsSection() {
  const { notifyError, notifySuccess } = useNotify();

  // Global default model (preferences.default_model) — the fallback floor when an
  // agent profile doesn't pin its own model. Empty = use the agent profile's model.
  const [defaultModel, setDefaultModel] = useState<string>('');
  const [savingDefaultModel, setSavingDefaultModel] = useState(false);

  useEffect(() => {
    fetchDefaultModel();
  }, []);

  const fetchDefaultModel = async () => {
    try {
      const config = await api.getConfig();
      const prefs = (config.preferences ?? {}) as { default_model?: string | null };
      setDefaultModel(prefs.default_model ?? '');
    } catch {
      // Non-fatal: the picker just starts on "System default".
    }
  };

  const handleDefaultModelChange = async (modelId: string) => {
    const previous = defaultModel;
    setDefaultModel(modelId);  // optimistic
    setSavingDefaultModel(true);
    try {
      await api.updateConfig({ preferences: { default_model: modelId } });
      notifySuccess(
        modelId ? 'Default model updated' : 'Default model cleared',
        'Models',
      );
    } catch (error) {
      setDefaultModel(previous);  // revert on failure
      notifyError(error, 'Failed to update the default model');
    } finally {
      setSavingDefaultModel(false);
    }
  };

  // Context limits — autosave (baseline-diff on top-level keys, so edits
  // replace the whole `lmstudio` object).
  const { settings, loading, status, update, refresh } = useSettingsAutosave<ContextLimits>({
    load: () => api.getContextLimits(),
    save: async changed => {
      await api.updateContextLimits(changed);
    },
    onError: err => notifyError(err, 'Context limits'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Sparkles size={20} />}
        title="Default Model"
        description="The model new agents and ad-hoc requests fall back to when a profile doesn't pin its own"
      />
      <div className="providers-list">
        <Card className="provider-card">
          <ModelPickerField
            value={defaultModel}
            onChange={handleDefaultModelChange}
            showDefault
            label="Global default model"
            hint={
              savingDefaultModel
                ? 'Saving…'
                : defaultModel
                  ? 'Used when an agent profile has no model of its own.'
                  : 'No global default — each agent uses its profile model.'
            }
          />
        </Card>
      </div>

      <SectionHeader
        icon={<Layers size={20} />}
        title="Model Context Limits"
        description="Configure context window and output token limits for local models (LM Studio)"
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
    </div>
  );
}
