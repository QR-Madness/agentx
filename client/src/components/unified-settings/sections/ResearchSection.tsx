/**
 * ResearchSection — Research Mode settings.
 *
 * Surfaces the per-conversation Research Mode dials: the global on/off, the
 * elevated per-turn search budget (the primary cost lever), the tool-round cap,
 * the default deep-research depth (cost/quality dial), and the deep-research tool
 * switch. Spans three config namespaces (`research.*`, `search.research_per_turn_limit`,
 * `web_research.enabled`), split back out in `save`. Built on the field kit + autosave.
 */

import { Telescope, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { NumberField, SaveStatusChip, SelectField, ToggleField } from '../../settings/fields';

interface ResearchSettings extends Record<string, unknown> {
  enabled: boolean;
  perTurnLimit: number;      // search.research_per_turn_limit (0 = unlimited)
  maxToolRounds: number;     // research.max_tool_rounds
  defaultDepth: string;      // research.default_depth (mini | auto | pro)
  webResearchEnabled: boolean; // web_research.enabled
}

// Tavily pricing for the projected-cost estimate: $0.008/credit; deep research
// burns ~5/10/20 credits by tier and charges ~3 budget units per call.
const USD_PER_CREDIT = 0.008;
const CREDITS_BY_DEPTH: Record<string, number> = { mini: 5, auto: 10, pro: 20 };
const DEEP_BUDGET_WEIGHT = 3;

const DEPTH_OPTIONS = [
  { value: 'mini', label: 'Mini — fast, cheapest (~$0.04/deep call)' },
  { value: 'auto', label: 'Auto — provider decides (~$0.08/deep call)' },
  { value: 'pro', label: 'Pro — deepest, priciest (~$0.16/deep call)' },
];

export default function ResearchSection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<ResearchSettings>({
    load: async () => {
      const config = await api.getConfig();
      const r = (config.research || {}) as Partial<{
        enabled: boolean; max_tool_rounds: number; default_depth: string;
      }>;
      const s = (config.search || {}) as Partial<{ research_per_turn_limit: number }>;
      const w = (config.web_research || {}) as Partial<{ enabled: boolean }>;
      return {
        enabled: r.enabled ?? true,
        perTurnLimit: s.research_per_turn_limit ?? 40,
        maxToolRounds: r.max_tool_rounds ?? 40,
        defaultDepth: r.default_depth ?? 'auto',
        webResearchEnabled: w.enabled ?? true,
      };
    },
    save: async changed => {
      // Route each changed key back to its config namespace (config/update is an
      // allowlisted per-section handler — see views.config_update).
      const research: Record<string, unknown> = {};
      const search: Record<string, unknown> = {};
      const webResearch: Record<string, unknown> = {};
      if ('enabled' in changed) research.enabled = changed.enabled;
      if ('maxToolRounds' in changed) research.max_tool_rounds = changed.maxToolRounds;
      if ('defaultDepth' in changed) research.default_depth = changed.defaultDepth;
      if ('perTurnLimit' in changed) search.research_per_turn_limit = changed.perTurnLimit;
      if ('webResearchEnabled' in changed) webResearch.enabled = changed.webResearchEnabled;
      const payload: Record<string, unknown> = {};
      if (Object.keys(research).length) payload.research = research;
      if (Object.keys(search).length) payload.search = search;
      if (Object.keys(webResearch).length) payload.web_research = webResearch;
      await api.updateConfig(payload);
    },
    onError: err => notifyError(err, 'Research settings'),
  });

  // Rough upper-bound cost per research turn: if every budgeted call were a deep
  // research at the chosen depth. Real turns mix cheap web_search calls, so this
  // is a ceiling, not an expectation.
  const projectedMax = (() => {
    if (!settings || settings.perTurnLimit <= 0) return null;
    const deepCalls = Math.floor(settings.perTurnLimit / DEEP_BUDGET_WEIGHT);
    const credits = CREDITS_BY_DEPTH[settings.defaultDepth] ?? 10;
    return deepCalls * credits * USD_PER_CREDIT;
  })();

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Telescope size={20} />}
        title="Research Mode"
        description="A per-conversation mode for deep, cited research with an elevated search budget and a self-reviewing report."
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
            label="Enable Research Mode"
            hint="Show the Research chip in the composer and let a conversation run a rigorous, cited research engagement. Turning this off hides the chip everywhere."
          />

          <NumberField
            label="Search budget per research turn"
            value={settings.perTurnLimit}
            min={0}
            max={200}
            fallback={40}
            onChange={perTurnLimit => update({ perTurnLimit })}
            title="Max web_search / web_research calls a single research turn may make. Deep research charges ~3 against this. 0 = unlimited (uncapped spend)."
          />
          <p className="setting-hint">
            {settings.perTurnLimit <= 0
              ? 'Unlimited — a research turn may spend without a cap. Watch usage.'
              : projectedMax !== null
                ? `Up to ~$${projectedMax.toFixed(2)} per research turn if every call is a deep “${settings.defaultDepth}” search — real turns mix cheaper web searches, so this is a ceiling.`
                : null}
          </p>

          <SelectField
            label="Default research depth"
            value={settings.defaultDepth}
            options={DEPTH_OPTIONS}
            onChange={defaultDepth => update({ defaultDepth })}
            hint="The deep-research effort tier the agent starts from; it may still escalate to “pro” for the hardest questions. The main cost/quality dial."
          />

          <NumberField
            label="Max tool rounds"
            value={settings.maxToolRounds}
            min={10}
            max={80}
            fallback={40}
            onChange={maxToolRounds => update({ maxToolRounds })}
            title="Tool-use rounds a research turn may take. Kept generous so the search budget — not tool rounds — governs how deep research goes."
          />

          <ToggleField
            checked={settings.webResearchEnabled}
            onChange={webResearchEnabled => update({ webResearchEnabled })}
            label="Deep-research tool (web_research)"
            hint="Allow the agentic deep-research tool (slower, costs more per call, but far richer). When off, research relies on ordinary web search + extraction."
          />
        </div>
      )}
    </div>
  );
}
