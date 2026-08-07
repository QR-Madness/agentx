/**
 * PlannerSection — Task planner configuration
 *
 * Settings live under `planner.*` in data/config.json and are consumed by
 * TaskPlanner / Agent.run(). Built on the settings field kit + autosave
 * (useSettingsAutosave — the section exemplar for the settings overhaul).
 */

import { ListTree, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  SaveStatusChip,
  SelectField,
  SliderField,
  ToggleField,
} from '../../settings/fields';
import { sectionBinder, useSettingsManifest } from '../SettingsManifestContext';

type ComplexityThreshold = 'simple' | 'moderate' | 'complex';

/** Local field name → `store:key`. `prompt_override` is absent on purpose: it
 *  is edited on Prompts → Feature Prompts, and now declares itself onto that
 *  screen rather than inheriting this one from its config root. */
const KEYS: Partial<Record<keyof PlannerSettings & string, string>> = {
  enabled: 'config:planner.enabled',
  model: 'config:planner.model',
  temperature: 'config:planner.temperature',
  max_tokens: 'config:planner.max_tokens',
  max_subtasks: 'config:planner.max_subtasks',
  complexity_threshold: 'config:planner.complexity_threshold',
};

interface PlannerSettings extends Record<string, unknown> {
  enabled: boolean;
  model: string;
  temperature: number;
  max_tokens: number;
  max_subtasks: number;
  complexity_threshold: ComplexityThreshold;
}

export default function PlannerSection() {
  const { notifyError } = useNotify();
  const manifest = useSettingsManifest();

  const { settings, loading, status, update } = useSettingsAutosave<PlannerSettings>({
    load: async () => {
      const config = await api.getConfig();
      const p = (config.planner || {}) as Partial<PlannerSettings> & {
        model?: string | null;
      };
      return {
        enabled: p.enabled ?? true,
        model: p.model || '',
        temperature: p.temperature ?? 0.3,
        max_tokens: p.max_tokens ?? 1000,
        max_subtasks: p.max_subtasks ?? 6,
        complexity_threshold: (p.complexity_threshold as ComplexityThreshold) || 'complex',
      };
    },
    save: async changed => {
      const payload: Record<string, unknown> = { ...changed };
      // Empty string → null so backend falls back to agent default model.
      if ('model' in payload) {
        payload.model = String(payload.model ?? '').trim() ? payload.model : null;
      }
      await api.updateConfig({ planner: payload });
    },
    onError: err => notifyError(err, 'Task Planner settings'),
  });

  const bind = sectionBinder<PlannerSettings>(manifest, KEYS, settings, update);

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<ListTree size={20} />}
        title="Task Planner"
        description="Configure how the agent decomposes complex tasks into subtasks before execution."
        actions={<SaveStatusChip status={status} />}
      />

      {loading || !settings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <ToggleField
            checked={settings.enabled}
            onChange={enabled => update({ enabled })}
            label="Enable Planner"
            hint="When off, every task takes the single-pass path regardless of complexity."
            {...bind('enabled')}
          />

          <ModelPickerField
            label="Planner Model"
            value={settings.model}
            onChange={model => update({ model })}
            showDefault={true}
            {...bind('model')}
          />

          <SelectField
            label="Complexity Threshold"
            value={settings.complexity_threshold}
            onChange={v => update({ complexity_threshold: v as ComplexityThreshold })}
            hint="Minimum complexity that triggers decomposition."
            options={[
              { value: 'simple', label: 'Simple — always decompose' },
              { value: 'moderate', label: 'Moderate — decompose moderate or complex tasks' },
              { value: 'complex', label: 'Complex — only decompose complex tasks' },
            ]}
            {...bind('complexity_threshold')}
          />

          <NumberField
            label="Max subtasks"
            value={settings.max_subtasks}
            min={1}
            max={20}
            fallback={6}
            onChange={max_subtasks => update({ max_subtasks })}
            title="Hard cap on the pieces one request may be split into. Each subtask is execution overhead as well as planning overhead."
            {...bind('max_subtasks')}
          />

          <SliderField
            label="Temperature"
            value={settings.temperature}
            min={0}
            max={1}
            step={0.1}
            onChange={temperature => update({ temperature })}
            format={v => v.toFixed(1)}
            {...bind('temperature')}
          />

          <NumberField
            label="Max Tokens"
            value={settings.max_tokens}
            min={100}
            max={4000}
            fallback={1000}
            onChange={max_tokens => update({ max_tokens })}
            title="Maximum length of the plan response."
            {...bind('max_tokens')}
          />

          <p className="setting-hint">
            The custom planner prompt now lives in Prompts → Feature Prompts.
          </p>
        </div>
      )}
    </div>
  );
}
