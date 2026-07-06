/**
 * PromptsSection — Prompt enhancement configuration
 *
 * Settings live under `prompt_enhancement.*` in data/config.json. Built on the
 * settings field kit + autosave (useSettingsAutosave).
 */

import { Sparkles, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  PromptField,
  SaveStatusChip,
  SliderField,
  ToggleField,
} from '../../settings/fields';

interface PromptEnhanceSettings extends Record<string, unknown> {
  enabled: boolean;
  model: string;
  temperature: number;
  max_tokens: number;
  system_prompt: string;
}

export default function PromptsSection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<PromptEnhanceSettings>({
    load: async () => {
      const config = await api.getConfig();
      const p = (config.prompt_enhancement || {}) as Partial<PromptEnhanceSettings>;
      return {
        enabled: p.enabled ?? true,
        model: p.model || 'anthropic:claude-3-5-haiku-latest',
        temperature: p.temperature ?? 0.7,
        max_tokens: p.max_tokens ?? 1000,
        system_prompt: p.system_prompt || '',
      };
    },
    save: async changed => {
      await api.updateConfig({ prompt_enhancement: changed });
    },
    onError: err => notifyError(err, 'Prompt Enhancement settings'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Sparkles size={20} />}
        title="Prompt Enhancement"
        description="Configure AI-powered prompt enhancement for better results"
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
            label="Enable Prompt Enhancement"
            hint="Allow AI to improve your prompts before sending"
          />

          <div className="setting-row">
            <ModelPickerField
              label="Enhancement Model"
              value={settings.model}
              onChange={model => update({ model })}
              showDefault={false}
            />
          </div>

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
            title="Maximum output length for enhanced prompt."
          />

          <PromptField
            label="Custom System Prompt"
            value={settings.system_prompt}
            onChange={system_prompt => update({ system_prompt })}
            onReset={() => update({ system_prompt: '' })}
            placeholder="Enter custom instructions for the enhancement model..."
            rows={4}
          />
        </div>
      )}
    </div>
  );
}
