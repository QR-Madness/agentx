/**
 * ContextSection ("Conversation Context") — the single home for every
 * in-conversation context-management technique:
 *
 *   - the verbatim window + compaction triggers (context.*)
 *   - the structured conversation state + its rolling digest (the default
 *     compaction target, INV-CTX-1)
 *   - the compaction summarizer (session.rolling_summary.*)
 *   - in-turn trajectory compression of tool rounds (trajectory_compression.*,
 *     moved here from Memory → Consolidation)
 *   - tool-output compression of oversized single results (compression.*)
 *   - episodic thread leads + rehydration bounds (memory.* / context.*)
 *
 * One flat autosave object; save() buckets changed keys back into their
 * config sections. Built on the settings field kit + useSettingsAutosave.
 */

import { Layers3, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  SaveStatusChip,
  SettingsSection,
  SliderField,
  ToggleField,
} from '../../settings/fields';

interface ContextSettings extends Record<string, unknown> {
  // context.*
  verbatim_budget_ratio: number;
  summary_trigger_ratio: number;
  recent_floor: number;
  preassembly_summary_enabled: boolean;
  conversation_state_enabled: boolean;
  conversation_state_compaction_enabled: boolean;
  rehydrate_max_turns: number;
  max_input_tokens: number;
  // session.rolling_summary.*
  compaction_enabled: boolean;
  compaction_model: string;
  compaction_max_tokens: number;
  // trajectory_compression.*
  trajectory_enabled: boolean;
  trajectory_threshold_ratio: number;
  trajectory_preserve_recent_rounds: number;
  trajectory_model: string;
  trajectory_max_knowledge_chars: number;
  // compression.* (tool output)
  tool_output_enabled: boolean;
  tool_output_model: string;
  tool_output_max_summary_chars: number;
  // memory.*
  episodic_leads_enabled: boolean;
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function ContextSection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<ContextSettings>({
    load: async () => {
      const cfg = await api.getConfig();
      const c = (cfg.context ?? {}) as Record<string, unknown>;
      const rs = ((cfg.session as Record<string, unknown> | undefined)?.rolling_summary ??
        {}) as Record<string, unknown>;
      const tc = (cfg.trajectory_compression ?? {}) as Record<string, unknown>;
      const cp = (cfg.compression ?? {}) as Record<string, unknown>;
      const mem = (cfg.memory ?? {}) as Record<string, unknown>;
      return {
        verbatim_budget_ratio: (c.verbatim_budget_ratio as number) ?? 0.9,
        summary_trigger_ratio: (c.summary_trigger_ratio as number) ?? 0.85,
        recent_floor: (c.recent_floor as number) ?? 4,
        preassembly_summary_enabled: (c.preassembly_summary_enabled as boolean) ?? true,
        conversation_state_enabled: (c.conversation_state_enabled as boolean) ?? true,
        conversation_state_compaction_enabled:
          (c.conversation_state_compaction_enabled as boolean) ?? true,
        rehydrate_max_turns: (c.rehydrate_max_turns as number) ?? 400,
        max_input_tokens: (c.max_input_tokens as number) ?? 0,
        compaction_enabled: (rs.enabled as boolean) ?? true,
        compaction_model: (rs.model as string) || '',
        compaction_max_tokens: (rs.max_tokens as number) ?? 800,
        trajectory_enabled: (tc.enabled as boolean) ?? true,
        trajectory_threshold_ratio: (tc.threshold_ratio as number) ?? 0.75,
        trajectory_preserve_recent_rounds: (tc.preserve_recent_rounds as number) ?? 2,
        trajectory_model: (tc.model as string) || '',
        trajectory_max_knowledge_chars: (tc.max_knowledge_chars as number) ?? 3000,
        tool_output_enabled: (cp.enabled as boolean) ?? true,
        tool_output_model: (cp.model as string) || '',
        tool_output_max_summary_chars: (cp.max_summary_chars as number) ?? 2000,
        episodic_leads_enabled: (mem.episodic_leads_enabled as boolean) ?? true,
      };
    },
    save: async changed => {
      // Bucket the flat autosave diff back into its config sections.
      const context: Record<string, unknown> = {};
      const rolling: Record<string, unknown> = {};
      const trajectory: Record<string, unknown> = {};
      const compression: Record<string, unknown> = {};
      const memory: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(changed)) {
        if (key.startsWith('compaction_')) {
          rolling[key.replace('compaction_', '')] = value;
        } else if (key.startsWith('trajectory_')) {
          trajectory[key.replace('trajectory_', '')] = value;
        } else if (key.startsWith('tool_output_')) {
          compression[key.replace('tool_output_', '')] = value;
        } else if (key === 'episodic_leads_enabled') {
          memory[key] = value;
        } else {
          context[key] = value;
        }
      }
      const payload: Record<string, unknown> = {};
      if (Object.keys(context).length) payload.context = context;
      if (Object.keys(rolling).length) payload.session = { rolling_summary: rolling };
      if (Object.keys(trajectory).length) payload.trajectory_compression = trajectory;
      if (Object.keys(compression).length) payload.compression = compression;
      if (Object.keys(memory).length) payload.memory = memory;
      await api.updateConfig(payload);
    },
    onError: err => notifyError(err, 'Conversation Context settings'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Layers3 size={20} />}
        title="Conversation Context"
        description="How each turn's prompt is built and compressed: the verbatim window, the conversation-state digest that covers aged-out turns (nothing leaves view uncovered), and the in-turn compressors."
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
            title="Verbatim Window & Compaction"
            description="The recent transcript stays word-for-word up to the verbatim budget; older turns are folded into the conversation-state digest. Sized against the model's REAL window — with a correct window (see Model Limits), most chats never compact."
          >
            <div className="settings-grid">
              <SliderField
                label="Verbatim window budget"
                value={settings.verbatim_budget_ratio}
                min={0.5}
                max={0.98}
                step={0.01}
                onChange={v => update({ verbatim_budget_ratio: v })}
                format={pct}
                hint="Fraction of the model's context window the verbatim transcript may use."
              />
              <SliderField
                label="Compaction pre-warm trigger"
                value={settings.summary_trigger_ratio}
                min={0.5}
                max={0.98}
                step={0.01}
                onChange={v => update({ summary_trigger_ratio: v })}
                format={pct}
                hint="Fraction of the turn's history budget at which the post-turn pre-warm folds older turns into the digest — slightly below the verbatim ceiling so the digest is ready before it's needed."
              />
              <NumberField
                label="Recent turns always kept verbatim"
                value={settings.recent_floor}
                min={1}
                max={50}
                fallback={4}
                onChange={v => update({ recent_floor: v })}
                title="Floor of most-recent turns that never age out, even under pressure."
              />
              <ToggleField
                checked={settings.preassembly_summary_enabled}
                onChange={v => update({ preassembly_summary_enabled: v })}
                label="Just-in-time coverage backstop"
                hint="Before assembling an over-budget turn, refresh the digest so the turns about to leave view are covered first (no silent context loss). A deterministic fallback digest stands in if the summarizer is unavailable."
              />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Conversation State"
            description="A structured, user-editable working memory per conversation — goals, decisions, open threads, artifacts — plus the rolling digest of aged-out turns. The agent updates it deliberately; you can edit it any time from the composer badge."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.conversation_state_enabled}
                onChange={v => update({ conversation_state_enabled: v })}
                label="Enable conversation state"
                hint="Off hides the state block and the agent's update_conversation_state tool."
              />
              <ToggleField
                checked={settings.conversation_state_compaction_enabled}
                onChange={v => update({ conversation_state_compaction_enabled: v })}
                label="Compact into the state digest"
                hint="Default: aged-out turns roll into the state object's digest. Off falls back to the legacy free-prose rolling summary."
              />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Compaction Summarizer"
            description="The model that folds aged-out turns into the digest, and how large the digest may grow (it is re-summarized in place each pass, so this bounds its steady-state size)."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.compaction_enabled}
                onChange={v => update({ compaction_enabled: v })}
                label="Automatic compaction"
                hint="Master switch for both compaction targets. Off means turns past the budget drop with no coverage — not recommended."
              />
              <div className="setting-row">
                <ModelPickerField
                  label="Summarizer model"
                  value={settings.compaction_model}
                  onChange={v => update({ compaction_model: v })}
                  placeholder="Summarizer role"
                  hint="Empty follows the summarizer model role (Model Roles)."
                />
            </div>
            <NumberField
              label="Digest size budget (tokens)"
              value={settings.compaction_max_tokens}
              min={200}
              max={4000}
              fallback={800}
              onChange={v => update({ compaction_max_tokens: v })}
            />
            </div>
          </SettingsSection>

          <SettingsSection
            title="In-Turn Trajectory Compression"
            description="During long multi-tool turns, older tool-call rounds are consolidated into a Knowledge block so the working transcript keeps room to think. This is within one turn — conversation-level compaction above is between turns."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.trajectory_enabled}
                onChange={v => update({ trajectory_enabled: v })}
                label="Enable trajectory compression"
              />
              <SliderField
                label="Trigger threshold"
                value={settings.trajectory_threshold_ratio}
                min={0.5}
                max={0.95}
                step={0.05}
                onChange={v => update({ trajectory_threshold_ratio: v })}
                format={pct}
                hint="Fires when the in-turn context crosses this fraction of its ceiling."
              />
              <NumberField
                label="Recent rounds kept intact"
                value={settings.trajectory_preserve_recent_rounds}
                min={1}
                max={5}
                fallback={2}
                onChange={v => update({ trajectory_preserve_recent_rounds: v })}
              />
              <div className="setting-row">
                <ModelPickerField
                  label="Compression model"
                  value={settings.trajectory_model}
                  onChange={v => update({ trajectory_model: v })}
                  placeholder="Summarizer role"
                  hint="Empty follows the summarizer model role."
                />
            </div>
            <NumberField
              label="Knowledge block size (chars)"
              value={settings.trajectory_max_knowledge_chars}
              min={500}
              max={10000}
              fallback={3000}
              onChange={v => update({ trajectory_max_knowledge_chars: v })}
            />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Tool-Output Compression"
            description="A single oversized tool result is summarized task-aware (with a structure index) instead of flooding the turn; the full output stays retrievable."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.tool_output_enabled}
                onChange={v => update({ tool_output_enabled: v })}
                label="Enable tool-output compression"
              />
              <div className="setting-row">
                <ModelPickerField
                  label="Compression model"
                  value={settings.tool_output_model}
                  onChange={v => update({ tool_output_model: v })}
                  placeholder="Summarizer role"
                  hint="Empty follows the summarizer model role."
                />
            </div>
            <NumberField
              label="Summary size (chars)"
              value={settings.tool_output_max_summary_chars}
              min={500}
              max={10000}
              fallback={2000}
              onChange={v => update({ tool_output_max_summary_chars: v })}
            />
            </div>
          </SettingsSection>

          <SettingsSection
            title="Recall & Rehydration"
            description="How a resumed conversation reloads, and whether past-conversation pointers are offered."
          >
            <div className="settings-grid">
              <ToggleField
                checked={settings.episodic_leads_enabled}
                onChange={v => update({ episodic_leads_enabled: v })}
                label="Episodic thread leads"
                hint='On phrasing like "when did we…", offer lightweight pointers into past conversations that the agent can expand on demand (read_thread) — never full transcripts.'
              />
              <NumberField
                label="Rehydration window (turns)"
                value={settings.rehydrate_max_turns}
                min={20}
                max={2000}
                fallback={400}
                onChange={v => update({ rehydrate_max_turns: v })}
                title="Max turns reloaded from durable history when a conversation resumes cold. Beyond it, an in-prompt notice points the agent at memory recall."
              />
              <NumberField
                label="Per-turn input cap (tokens, 0 = off)"
                value={settings.max_input_tokens}
                min={0}
                max={1000000}
                fallback={0}
                onChange={v => update({ max_input_tokens: v })}
                title="Optional spend guard: caps the in-turn context ceiling below the model window. Leave 0 to let agents use the model's full length."
              />
            </div>
          </SettingsSection>
        </div>
      )}
    </div>
  );
}
