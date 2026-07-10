/**
 * ThinkingSection ("Thinking Patterns") — how agents reason in chat.
 *
 * Settings live under `reasoning.*` in data/config.json and are consumed by
 * reasoning/chat_patterns.py (pattern resolution, the auto classifier, the
 * step_back pre-call, self-consistency sampling, and the thinking output
 * floor). Built on the settings field kit + autosave.
 */

import { Brain, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  SaveStatusChip,
  SettingsSection,
  ToggleField,
} from '../../settings/fields';

interface ThinkingSettings extends Record<string, unknown> {
  chat_patterns_enabled: boolean;
  auto_classifier_enabled: boolean;
  classifier_model: string;
  classifier_min_chars: number;
  step_back_model: string;
  cot_enabled: boolean;
  step_back_enabled: boolean;
  reflection_enabled: boolean;
  self_consistency_enabled: boolean;
  sc_model: string;
  sc_k: number;
  min_output_tokens: number;
}

export default function ThinkingSection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<ThinkingSettings>({
    load: async () => {
      const cfg = await api.getConfig();
      const r = (cfg.reasoning ?? {}) as Record<string, unknown>;
      return {
        chat_patterns_enabled: (r.chat_patterns_enabled as boolean) ?? true,
        auto_classifier_enabled: (r.auto_classifier_enabled as boolean) ?? true,
        classifier_model: (r.classifier_model as string) || '',
        classifier_min_chars: (r.classifier_min_chars as number) ?? 240,
        step_back_model: (r.step_back_model as string) || '',
        cot_enabled: (r.cot_enabled as boolean) ?? true,
        step_back_enabled: (r.step_back_enabled as boolean) ?? true,
        reflection_enabled: (r.reflection_enabled as boolean) ?? true,
        self_consistency_enabled: (r.self_consistency_enabled as boolean) ?? true,
        sc_model: (r.sc_model as string) || '',
        sc_k: (r.sc_k as number) ?? 3,
        min_output_tokens: (r.min_output_tokens as number) ?? 0,
      };
    },
    save: async changed => {
      await api.updateConfig({ reasoning: changed });
    },
    onError: err => notifyError(err, 'Thinking Patterns settings'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Brain size={20} />}
        title="Thinking Patterns"
        description="How agents reason in chat: step-by-step scaffolds, principles-first step-back, self-critiquing reflection, and consensus sampling — compiled into the live streamed turn. Per-agent defaults live on the profile; the composer chip overrides per conversation."
        actions={<SaveStatusChip status={status} />}
      />

      {loading || !settings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <SettingsSection
            title="Patterns"
            description="Which patterns are available (to both explicit selection and Auto). A model with native reasoning is never given a redundant step-by-step scaffold by Auto."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.chat_patterns_enabled}
                onChange={v => update({ chat_patterns_enabled: v })}
                label="Enable thinking patterns"
                hint="Master switch — off restores plain turns (native model thinking still streams)."
              />
              <ToggleField
                checked={settings.cot_enabled}
                onChange={v => update({ cot_enabled: v })}
                label="Step-by-step (chain of thought)"
              />
              <ToggleField
                checked={settings.step_back_enabled}
                onChange={v => update({ step_back_enabled: v })}
                label="Step-back (principles first)"
                hint="One small hidden pre-call distills governing principles before the turn."
              />
              <ToggleField
                checked={settings.reflection_enabled}
                onChange={v => update({ reflection_enabled: v })}
                label="Reflection (draft → critique → improve)"
                hint="Also gates the multi-pass Reflect deeply pattern."
              />
              <ToggleField
                checked={settings.self_consistency_enabled}
                onChange={v => update({ self_consistency_enabled: v })}
                label="Consensus (self-consistency sampling)"
                hint="Auto only picks it for short calculation/logic turns with no tools."
              />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Auto Selection"
            description="Auto picks per message: instant keyword heuristics first; an optional bounded LLM tiebreak only when they're unsure on a substantial message."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.auto_classifier_enabled}
                onChange={v => update({ auto_classifier_enabled: v })}
                label="LLM tiebreak"
                hint="One tiny classification call (≤150 tokens, 5s cap) when heuristics are unconfident."
              />
              <div className="setting-row">
                <ModelPickerField
                  label="Classifier model"
                  value={settings.classifier_model}
                  onChange={v => update({ classifier_model: v })}
                  placeholder="Fast Utility role"
                  hint="Empty follows the Fast Utility model role."
                />
              </div>
              <NumberField
                label="Tiebreak minimum message length (chars)"
                value={settings.classifier_min_chars}
                min={0}
                max={5000}
                fallback={240}
                onChange={v => update({ classifier_min_chars: v })}
                title="Below this the tiebreak never fires — trivial messages stay zero-cost."
              />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Pattern Models & Budgets"
            description="The extra-call patterns run on the active turn model unless overridden here."
          >
            <div className="settings-grid">
              <div className="setting-row">
                <ModelPickerField
                  label="Step-back model"
                  value={settings.step_back_model}
                  onChange={v => update({ step_back_model: v })}
                  placeholder="Active turn model"
                  hint="Empty uses the conversation's own model for the principles pre-call."
                />
              </div>
              <div className="setting-row">
                <ModelPickerField
                  label="Consensus sampling model"
                  value={settings.sc_model}
                  onChange={v => update({ sc_model: v })}
                  placeholder="Active turn model"
                  hint="Samples are the k× cost — point them at a cheaper model if needed."
                />
              </div>
              <NumberField
                label="Consensus samples (k)"
                value={settings.sc_k}
                min={2}
                max={5}
                fallback={3}
                onChange={v => update({ sc_k: v })}
              />
              <NumberField
                label="Thinking output floor (tokens, 0 = auto)"
                value={settings.min_output_tokens}
                min={0}
                max={65536}
                fallback={0}
                onChange={v => update({ min_output_tokens: v })}
                title="Thinking spends output tokens before the visible answer. 0 floors automatically when a pattern is active or the model reasons natively."
              />
            </div>
          </SettingsSection>
        </div>
      )}
    </div>
  );
}
