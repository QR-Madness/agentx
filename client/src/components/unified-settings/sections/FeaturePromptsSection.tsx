/**
 * FeaturePromptsSection — every overridable feature prompt in one place
 * (prompts nav group): extraction, relevance filter, planner decomposition,
 * prompt enhancement.
 *
 * Two autosave scopes: the memory prompts persist via the consolidation
 * settings API; the agent-feature prompts persist under config keys
 * (`planner.prompt_override`, `prompt_enhancement.system_prompt`). Shipped
 * defaults come from `getFeaturePromptDefaults` and power Diff/reset.
 */

import { useCallback, useRef, useState } from 'react';
import { FileText, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import type { FeaturePromptDefaults } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { PromptField, SaveStatusChip, SettingsSection } from '../../settings/fields';

interface MemoryPromptSettings extends Record<string, unknown> {
  extraction_system_prompt: string;
  relevance_filter_prompt: string;
}

interface ConfigPromptSettings extends Record<string, unknown> {
  planner_prompt: string;
  enhancement_prompt: string;
}

export default function FeaturePromptsSection() {
  const { notifyError } = useNotify();

  // Shipped defaults — fetched once, shared by both autosave loads. The ref
  // holds the resolved value for save-time transforms; the state re-renders
  // the Diff buttons once defaults arrive.
  const defaultsRef = useRef<FeaturePromptDefaults | null>(null);
  const defaultsPromiseRef = useRef<Promise<FeaturePromptDefaults> | null>(null);
  const [defaults, setDefaults] = useState<FeaturePromptDefaults | null>(null);
  const loadDefaults = useCallback(() => {
    defaultsPromiseRef.current ??= api.getFeaturePromptDefaults().then(d => {
      defaultsRef.current = d;
      setDefaults(d);
      return d;
    });
    return defaultsPromiseRef.current;
  }, []);

  // Memory prompts — consolidation settings keys (empty = shipped default).
  const memory = useSettingsAutosave<MemoryPromptSettings>({
    load: async () => {
      await loadDefaults();
      const s = await api.getConsolidationSettings();
      return {
        extraction_system_prompt: s.extraction_system_prompt || '',
        relevance_filter_prompt: s.relevance_filter_prompt || '',
      };
    },
    save: async changed => {
      await api.updateConsolidationSettings(changed);
    },
    onError: err => notifyError(err, 'Feature Prompts (memory)'),
  });

  // Agent-feature prompts — config keys. The planner editor is seeded with the
  // shipped default when no override is saved; if the text still matches the
  // default at save time, persist empty so the planner keeps tracking the live
  // default instead of pinning a copy. The enhancement prompt has no seeding.
  const config = useSettingsAutosave<ConfigPromptSettings>({
    load: async () => {
      const d = await loadDefaults();
      const cfg = await api.getConfig();
      const planner = (cfg.planner || {}) as { prompt_override?: string };
      const enhance = (cfg.prompt_enhancement || {}) as { system_prompt?: string };
      return {
        planner_prompt: planner.prompt_override || d.planner_prompt,
        enhancement_prompt: enhance.system_prompt || '',
      };
    },
    save: async changed => {
      const payload: Record<string, unknown> = {};
      if ('planner_prompt' in changed) {
        const text = String(changed.planner_prompt ?? '');
        const builtin = defaultsRef.current?.planner_prompt ?? '';
        payload.planner = {
          prompt_override: text.trim() === builtin.trim() ? '' : text,
        };
      }
      if ('enhancement_prompt' in changed) {
        payload.prompt_enhancement = { system_prompt: changed.enhancement_prompt };
      }
      await api.updateConfig(payload);
    },
    onError: err => notifyError(err, 'Feature Prompts (agent)'),
  });

  const loading = memory.loading || config.loading || !memory.settings || !config.settings;
  // One chip: surface whichever scope is doing something.
  const status = memory.status !== 'idle' ? memory.status : config.status;

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<FileText size={20} />}
        title="Feature Prompts"
        description="Override the built-in prompts behind extraction, relevance filtering, planning, and prompt enhancement."
        actions={<SaveStatusChip status={status} />}
      />

      {loading ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <SettingsSection
            title="Memory extraction"
            description="Prompts driving memory consolidation. Models, temperatures, and stage toggles live in Memory → Consolidation."
          >
            <PromptField
              label="Extraction System Prompt"
              value={memory.settings!.extraction_system_prompt}
              onChange={extraction_system_prompt => memory.update({ extraction_system_prompt })}
              onReset={() => memory.update({ extraction_system_prompt: '' })}
              placeholder="Override the extraction system prompt (entities, facts, relationships)..."
              rows={8}
              defaultText={defaults?.extraction_system_prompt}
            />
            <PromptField
              label="Relevance Filter Prompt"
              value={memory.settings!.relevance_filter_prompt}
              onChange={relevance_filter_prompt => memory.update({ relevance_filter_prompt })}
              onReset={() => memory.update({ relevance_filter_prompt: '' })}
              placeholder="Override the relevance filter prompt (skip non-informative turns)..."
              rows={6}
              defaultText={defaults?.relevance_filter_prompt}
            />
          </SettingsSection>

          <SettingsSection
            title="Agent features"
            description="Prompts behind task planning and prompt enhancement. Models and limits live in their own sections."
          >
            <PromptField
              label="Custom Planner Prompt"
              value={config.settings!.planner_prompt}
              onChange={planner_prompt => config.update({ planner_prompt })}
              onReset={() =>
                config.update({ planner_prompt: defaultsRef.current?.planner_prompt ?? '' })
              }
              placeholder="Override the planner's decomposition system prompt..."
              rows={10}
              defaultText={defaults?.planner_prompt}
            />
            <PromptField
              label="Enhancement System Prompt"
              value={config.settings!.enhancement_prompt}
              onChange={enhancement_prompt => config.update({ enhancement_prompt })}
              onReset={() => config.update({ enhancement_prompt: '' })}
              placeholder="Enter custom instructions for the enhancement model..."
              rows={4}
              defaultText={defaults?.prompt_enhancement_prompt}
            />
          </SettingsSection>
        </div>
      )}
    </div>
  );
}
