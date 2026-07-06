/**
 * PlannerSection — Task planner configuration
 *
 * Settings live under `planner.*` in data/config.json and are consumed by
 * TaskPlanner / Agent.run(). Built on the settings field kit + autosave
 * (useSettingsAutosave — the section exemplar for the settings overhaul).
 */

import { useRef } from 'react';
import { ListTree, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  PromptField,
  SaveStatusChip,
  SelectField,
  SliderField,
  ToggleField,
} from '../../settings/fields';

type ComplexityThreshold = 'simple' | 'moderate' | 'complex';

interface PlannerSettings extends Record<string, unknown> {
  enabled: boolean;
  model: string;
  temperature: number;
  max_tokens: number;
  prompt_override: string;
  complexity_threshold: ComplexityThreshold;
}

export default function PlannerSection() {
  const { notifyError } = useNotify();
  // The built-in decomposition prompt (read-only) used to seed the editor so
  // the user can customize from the real default instead of a blank box.
  const defaultPromptRef = useRef('');

  const { settings, loading, status, update } = useSettingsAutosave<PlannerSettings>({
    load: async () => {
      const config = await api.getConfig();
      const p = (config.planner || {}) as Partial<PlannerSettings> & {
        model?: string | null;
        decompose_default?: string;
      };
      const builtinPrompt = p.decompose_default || '';
      defaultPromptRef.current = builtinPrompt;
      return {
        enabled: p.enabled ?? true,
        model: p.model || '',
        temperature: p.temperature ?? 0.3,
        max_tokens: p.max_tokens ?? 1000,
        // Seed the editor with the built-in prompt when no override is saved.
        prompt_override: p.prompt_override || builtinPrompt,
        complexity_threshold: (p.complexity_threshold as ComplexityThreshold) || 'complex',
      };
    },
    save: async changed => {
      const payload: Record<string, unknown> = { ...changed };
      // Empty string → null so backend falls back to agent default model.
      if ('model' in payload) {
        payload.model = String(payload.model ?? '').trim() ? payload.model : null;
      }
      // If the editor still matches the built-in prompt, persist empty so the
      // planner keeps tracking the live default instead of pinning a copy.
      if ('prompt_override' in payload) {
        const text = String(payload.prompt_override ?? '');
        payload.prompt_override =
          text.trim() === defaultPromptRef.current.trim() ? '' : text;
      }
      await api.updateConfig({ planner: payload });
    },
    onError: err => notifyError(err, 'Task Planner settings'),
  });

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
          />

          <div className="setting-row">
            <ModelPickerField
              label="Planner Model"
              value={settings.model}
              onChange={model => update({ model })}
              showDefault={true}
            />
          </div>

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
          />

          <SliderField
            label="Temperature"
            value={settings.temperature}
            min={0}
            max={1}
            step={0.1}
            onChange={temperature => update({ temperature })}
            format={v => v.toFixed(1)}
          />

          <NumberField
            label="Max Tokens"
            value={settings.max_tokens}
            min={100}
            max={4000}
            fallback={1000}
            onChange={max_tokens => update({ max_tokens })}
            title="Maximum length of the plan response."
          />

          <PromptField
            label="Custom Planner Prompt"
            value={settings.prompt_override}
            onChange={prompt_override => update({ prompt_override })}
            onReset={() => update({ prompt_override: defaultPromptRef.current })}
            placeholder="Override the planner's decomposition system prompt..."
            rows={10}
            defaultText={defaultPromptRef.current}
          />
        </div>
      )}
    </div>
  );
}
