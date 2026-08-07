/**
 * ResearchSection — Research Mode settings.
 *
 * The per-conversation Research Mode dials: the global on/off, the elevated
 * per-turn search budget (the primary cost lever), the tool-round cap, the
 * default deep-research depth (the cost/quality dial), and the deep-research
 * tool — plus how a single deep call is weighed, cached and waited on.
 *
 * Spans three config namespaces (`research.*`, `search.research_per_turn_limit`,
 * `web_research.*`), split back out in `save`. Built on the field kit + autosave.
 */

import { Telescope, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import {
  NumberField,
  SaveStatusChip,
  SelectField,
  SettingsSection,
  ToggleField,
} from '../../settings/fields';
import { sectionBinder, useSettingsManifest } from '../SettingsManifestContext';

interface ResearchSettings extends Record<string, unknown> {
  enabled: boolean;
  perTurnLimit: number;      // search.research_per_turn_limit (0 = unlimited)
  maxToolRounds: number;     // research.max_tool_rounds
  defaultDepth: string;      // research.default_depth (mini | auto | pro)
  minMaxTokens: number;      // research.min_max_tokens
  webResearchEnabled: boolean; // web_research.enabled
  budgetWeight: number;      // web_research.budget_weight
  researchCacheTtl: number;  // web_research.cache_ttl_seconds
  pollTimeout: number;       // web_research.poll_timeout_seconds
  pollInterval: number;      // web_research.poll_interval_seconds
}

/**
 * Local field name → `store:key`. This screen spans three config roots, and
 * `search.research_per_turn_limit` is the clearest case of why the mapping is
 * declared rather than inferred: it is stored under `search` and rendered here,
 * with the rest of the research budget it belongs to.
 */
const KEYS: Partial<Record<keyof ResearchSettings & string, string>> = {
  enabled: 'config:research.enabled',
  perTurnLimit: 'config:search.research_per_turn_limit',
  maxToolRounds: 'config:research.max_tool_rounds',
  defaultDepth: 'config:research.default_depth',
  minMaxTokens: 'config:research.min_max_tokens',
  webResearchEnabled: 'config:web_research.enabled',
  budgetWeight: 'config:web_research.budget_weight',
  researchCacheTtl: 'config:web_research.cache_ttl_seconds',
  pollTimeout: 'config:web_research.poll_timeout_seconds',
  pollInterval: 'config:web_research.poll_interval_seconds',
};

// Tavily pricing for the projected-cost estimate: $0.008/credit; deep research
// burns ~5/10/20 credits by tier. The budget units a deep call charges is now a
// setting on this same screen, so the estimate reads it rather than assuming 3.
const USD_PER_CREDIT = 0.008;
const CREDITS_BY_DEPTH: Record<string, number> = { mini: 5, auto: 10, pro: 20 };

const DEPTH_OPTIONS = [
  { value: 'mini', label: 'Mini — fast, cheapest (~$0.04/deep call)' },
  { value: 'auto', label: 'Auto — provider decides (~$0.08/deep call)' },
  { value: 'pro', label: 'Pro — deepest, priciest (~$0.16/deep call)' },
];

export default function ResearchSection() {
  const { notifyError } = useNotify();
  const manifest = useSettingsManifest();

  const { settings, loading, status, update } = useSettingsAutosave<ResearchSettings>({
    load: async () => {
      const config = await api.getConfig();
      const r = (config.research || {}) as Partial<{
        enabled: boolean; max_tool_rounds: number; default_depth: string;
        min_max_tokens: number;
      }>;
      const s = (config.search || {}) as Partial<{ research_per_turn_limit: number }>;
      const w = (config.web_research || {}) as Partial<{
        enabled: boolean; budget_weight: number; cache_ttl_seconds: number;
        poll_timeout_seconds: number; poll_interval_seconds: number;
      }>;
      return {
        enabled: r.enabled ?? true,
        perTurnLimit: s.research_per_turn_limit ?? 40,
        maxToolRounds: r.max_tool_rounds ?? 40,
        defaultDepth: r.default_depth ?? 'auto',
        minMaxTokens: r.min_max_tokens ?? 16384,
        webResearchEnabled: w.enabled ?? true,
        budgetWeight: w.budget_weight ?? 3,
        researchCacheTtl: w.cache_ttl_seconds ?? 1800,
        pollTimeout: w.poll_timeout_seconds ?? 240,
        pollInterval: w.poll_interval_seconds ?? 5,
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
      if ('minMaxTokens' in changed) research.min_max_tokens = changed.minMaxTokens;
      if ('perTurnLimit' in changed) search.research_per_turn_limit = changed.perTurnLimit;
      if ('webResearchEnabled' in changed) webResearch.enabled = changed.webResearchEnabled;
      if ('budgetWeight' in changed) webResearch.budget_weight = changed.budgetWeight;
      if ('researchCacheTtl' in changed) webResearch.cache_ttl_seconds = changed.researchCacheTtl;
      if ('pollTimeout' in changed) webResearch.poll_timeout_seconds = changed.pollTimeout;
      if ('pollInterval' in changed) webResearch.poll_interval_seconds = changed.pollInterval;
      const payload: Record<string, unknown> = {};
      if (Object.keys(research).length) payload.research = research;
      if (Object.keys(search).length) payload.search = search;
      if (Object.keys(webResearch).length) payload.web_research = webResearch;
      await api.updateConfig(payload);
    },
    onError: err => notifyError(err, 'Research settings'),
  });

  const bind = sectionBinder<ResearchSettings>(manifest, KEYS, settings, update);

  // Rough upper-bound cost per research turn: if every budgeted call were a deep
  // research at the chosen depth. Real turns mix cheap web_search calls, so this
  // is a ceiling, not an expectation.
  const projectedMax = (() => {
    if (!settings || settings.perTurnLimit <= 0) return null;
    const deepCalls = Math.floor(settings.perTurnLimit / Math.max(1, settings.budgetWeight));
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
            {...bind('enabled')}
          />

          <NumberField
            label="Search budget per research turn"
            value={settings.perTurnLimit}
            min={0}
            max={200}
            fallback={40}
            onChange={perTurnLimit => update({ perTurnLimit })}
            title="Max web_search / web_research calls a single research turn may make. Deep research charges ~3 against this. 0 = unlimited (uncapped spend)."
            {...bind('perTurnLimit')}
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
            {...bind('defaultDepth')}
          />

          <NumberField
            label="Max tool rounds"
            value={settings.maxToolRounds}
            min={10}
            max={80}
            fallback={40}
            onChange={maxToolRounds => update({ maxToolRounds })}
            title="Tool-use rounds a research turn may take. Kept generous so the search budget — not tool rounds — governs how deep research goes."
            {...bind('maxToolRounds')}
          />

          <ToggleField
            checked={settings.webResearchEnabled}
            onChange={webResearchEnabled => update({ webResearchEnabled })}
            label="Deep-research tool (web_research)"
            hint="Allow the agentic deep-research tool (slower, costs more per call, but far richer). When off, research relies on ordinary web search + extraction."
            {...bind('webResearchEnabled')}
          />

          {/* These five were writable over the API but had no control anywhere —
              including the budget weight the search-budget help points at. */}
          <SettingsSection
            title="Deep research tuning"
            description="How a single deep-research call is weighed, cached, and waited on. Sensible as shipped; reach for these when research is costing or timing out more than it should."
          >
            <NumberField
              label="Report output floor (tokens)"
              value={settings.minMaxTokens}
              min={1024}
              max={200000}
              fallback={16384}
              onChange={minMaxTokens => update({ minMaxTokens })}
              title="Minimum output budget for a research completion — a report must fit alongside the thinking that produced it"
              {...bind('minMaxTokens')}
            />

            <NumberField
              label="Deep-research budget weight"
              value={settings.budgetWeight}
              min={1}
              max={20}
              fallback={3}
              onChange={budgetWeight => update({ budgetWeight })}
              title="Budget units one deep-research call charges against the turn's search budget"
              {...bind('budgetWeight')}
            />

            <NumberField
              label="Deep-research cache (seconds)"
              value={settings.researchCacheTtl}
              min={0}
              max={86400}
              fallback={1800}
              onChange={researchCacheTtl => update({ researchCacheTtl })}
              title="How long an identical deep-research query is reused instead of re-run. 0 = no caching"
              {...bind('researchCacheTtl')}
            />

            <NumberField
              label="Report wait limit (seconds)"
              value={settings.pollTimeout}
              min={30}
              max={900}
              fallback={240}
              onChange={pollTimeout => update({ pollTimeout })}
              title="Total wall-clock wait for a deep-research report before the tool gives up"
              {...bind('pollTimeout')}
            />

            <NumberField
              label="Status check interval (seconds)"
              value={settings.pollInterval}
              min={1}
              max={60}
              fallback={5}
              onChange={pollInterval => update({ pollInterval })}
              title="Gap between checks while waiting for a report"
              {...bind('pollInterval')}
            />
          </SettingsSection>
        </div>
      )}
    </div>
  );
}
