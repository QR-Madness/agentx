/**
 * SearchSection — the control plane for the agent's web tools.
 *
 * Grouped into Backend & keys / Defaults / Budgets / Source policy. Non-secret
 * knobs autosave via useSettingsAutosave (settings field kit); the API keys keep
 * an explicit "Save keys" flow (secrets are never autosaved). Keys arrive
 * redacted from GET /api/config (e.g. "***1234"); we only send a key back when
 * the user has typed a new value (not the redacted mask).
 *
 * Every knob here is deliberately operator-owned: the model decides *what* to
 * search for, the operator decides how much that is allowed to cost and which
 * corners of the web may ground an answer.
 */

import { useState } from 'react';
import { Globe, RefreshCw, Save, Eye, EyeOff, Wifi, Check, X } from 'lucide-react';
import { api } from '../../../lib/api';
import type { SearchBackendHealth } from '../../../lib/api/types';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { Badge, Button, Input, SectionHeader } from '../../ui';
import {
  NumberField,
  SaveStatusChip,
  SelectField,
  SettingsSection,
  TextField,
  ToggleField,
} from '../../settings/fields';
import { SettingHelp } from '../../settings/SettingHelp';
import {
  bindSetting,
  sectionBinder,
  useSettingsManifest,
} from '../SettingsManifestContext';

type Backend = 'tavily' | 'brave';

interface SearchSettings extends Record<string, unknown> {
  backend: Backend;
  fallback_enabled: boolean;
  max_results: number;
  cache_ttl_seconds: number;
  timeout: number;
  // Defaults applied when the model doesn't specify them ('' = no opinion).
  default_search_depth: string;
  default_chunks_per_source: number;
  safesearch: string;
  country: string;
  search_lang: string;
  // Budgets
  per_turn_limit: number;
  per_turn_cost_usd: number;
  research_per_turn_cost_usd: number;
  // Brave
  brave_grounding_default: boolean;
  brave_context_max_tokens: number;
  brave_context_max_tokens_per_url: number;
  brave_context_threshold: string;
  brave_answers_enabled: boolean;
  // Source policy (sent as one object)
  trusted: string;
  blocked: string;
  goggle: string;
}

/**
 * Local field name → `store:key` in the settings manifest. Everything the
 * manifest knows — shipped default, declared bounds, authored help — reaches
 * the control through this.
 *
 * `trusted`/`blocked`/`goggle` are absent deliberately: they are three inputs
 * over one `source_policy` dict, so they bind once, by hand, below.
 */
const KEYS: Partial<Record<keyof SearchSettings & string, string>> = {
  backend: 'config:search.backend',
  fallback_enabled: 'config:search.fallback_enabled',
  max_results: 'config:search.max_results',
  cache_ttl_seconds: 'config:search.cache_ttl_seconds',
  timeout: 'config:search.timeout',
  default_search_depth: 'config:search.default_search_depth',
  default_chunks_per_source: 'config:search.default_chunks_per_source',
  safesearch: 'config:search.safesearch',
  country: 'config:search.country',
  search_lang: 'config:search.search_lang',
  per_turn_limit: 'config:search.per_turn_limit',
  per_turn_cost_usd: 'config:search.per_turn_cost_usd',
  research_per_turn_cost_usd: 'config:search.research_per_turn_cost_usd',
  brave_grounding_default: 'config:search.brave_grounding_default',
  brave_context_max_tokens: 'config:search.brave_context_max_tokens',
  brave_context_max_tokens_per_url: 'config:search.brave_context_max_tokens_per_url',
  brave_context_threshold: 'config:search.brave_context_threshold',
  brave_answers_enabled: 'config:search.brave_answers_enabled',
};

const isRedacted = (v: string) => v.startsWith('***');

/** Textarea-free list editing: one domain per comma. Empty entries dropped. */
const toList = (v: string): string[] =>
  v.split(',').map(s => s.trim()).filter(Boolean);

export default function SearchSection() {
  const { notifyError, notifySuccess } = useNotify();
  const manifest = useSettingsManifest();

  // Secrets stay out of the autosave draft — explicit Save only.
  const [keys, setKeys] = useState<{ tavily_api_key: string; brave_api_key: string }>({
    tavily_api_key: '',
    brave_api_key: '',
  });
  const [showKeys, setShowKeys] = useState<{ tavily: boolean; brave: boolean }>({
    tavily: false,
    brave: false,
  });
  const [savingKeys, setSavingKeys] = useState(false);
  const [testing, setTesting] = useState(false);
  const [health, setHealth] = useState<SearchBackendHealth[] | null>(null);

  const { settings, loading, status, update, refresh } = useSettingsAutosave<SearchSettings>({
    load: async () => {
      const config = await api.getConfig();
      const s = (config.search || {}) as Partial<SearchSettings> & {
        tavily_api_key?: string;
        brave_api_key?: string;
        source_policy?: { trusted?: string[]; blocked?: string[]; goggle?: string };
      };
      // Seed the key inputs alongside the autosaved knobs (single fetch).
      setKeys({
        tavily_api_key: s.tavily_api_key || '',
        brave_api_key: s.brave_api_key || '',
      });
      const policy = s.source_policy || {};
      return {
        backend: s.backend === 'brave' ? 'brave' : 'tavily',
        fallback_enabled: s.fallback_enabled ?? true,
        max_results: s.max_results ?? 5,
        cache_ttl_seconds: s.cache_ttl_seconds ?? 300,
        timeout: s.timeout ?? 15,
        default_search_depth: s.default_search_depth ?? '',
        default_chunks_per_source: s.default_chunks_per_source ?? 0,
        safesearch: s.safesearch ?? '',
        country: s.country ?? '',
        search_lang: s.search_lang ?? '',
        per_turn_limit: s.per_turn_limit ?? 8,
        per_turn_cost_usd: s.per_turn_cost_usd ?? 0,
        research_per_turn_cost_usd: s.research_per_turn_cost_usd ?? 0,
        brave_grounding_default: s.brave_grounding_default ?? true,
        brave_context_max_tokens: s.brave_context_max_tokens ?? 4096,
        brave_context_max_tokens_per_url: s.brave_context_max_tokens_per_url ?? 1024,
        brave_context_threshold: s.brave_context_threshold ?? 'balanced',
        brave_answers_enabled: s.brave_answers_enabled ?? false,
        trusted: (policy.trusted || []).join(', '),
        blocked: (policy.blocked || []).join(', '),
        goggle: policy.goggle || '',
      };
    },
    save: async changed => {
      // The three policy fields are edited separately but persist as one object,
      // so any touch of them re-sends the whole thing.
      const { trusted, blocked, goggle, ...rest } = changed;
      const payload: Record<string, unknown> = { ...rest };
      if (trusted !== undefined || blocked !== undefined || goggle !== undefined) {
        payload.source_policy = {
          trusted: toList(trusted ?? settings?.trusted ?? ''),
          blocked: toList(blocked ?? settings?.blocked ?? ''),
          goggle: (goggle ?? settings?.goggle ?? '').trim(),
        };
      }
      await api.updateConfig({ search: payload });
    },
    onError: err => notifyError(err, 'Web Search settings'),
  });

  const bind = sectionBinder<SearchSettings>(manifest, KEYS, settings, update);

  /**
   * Source policy binds once, against the reconstructed dict.
   *
   * It is written whole (a per-leaf patch would drop its siblings), so the
   * manifest carries one entry for all three inputs. Binding each input
   * separately would compare a comma-separated string to a dict and report
   * "changed from default" forever; resetting one would write the dict into a
   * text field. So: one binding, attached to the first input, with a reset that
   * puts all three back.
   */
  const policyBinding = bindSetting(
    manifest, 'config', 'search.source_policy',
    settings
      ? {
          trusted: toList(settings.trusted),
          blocked: toList(settings.blocked),
          goggle: settings.goggle.trim(),
        }
      : undefined,
  );
  const policyDefault = (policyBinding?.defaultValue ?? {}) as {
    trusted?: string[]; blocked?: string[]; goggle?: string;
  };
  const resetPolicy = policyBinding
    ? () => update({
        trusted: (policyDefault.trusted ?? []).join(', '),
        blocked: (policyDefault.blocked ?? []).join(', '),
        goggle: policyDefault.goggle ?? '',
      })
    : undefined;

  /** Authored help for a key with no field-kit control of its own. */
  const helpFor = (key: string) =>
    manifest?.entries.get(`config:${key}`)?.help;

  // A key is sendable when the user typed a new value (not empty, not the mask).
  const tavilyChanged = !!keys.tavily_api_key && !isRedacted(keys.tavily_api_key);
  const braveChanged = !!keys.brave_api_key && !isRedacted(keys.brave_api_key);

  const handleSaveKeys = async () => {
    setSavingKeys(true);
    try {
      // Only send keys the user actually changed (skip empty + redacted mask);
      // the backend skips omitted/None values so stored keys aren't overwritten.
      const payload: NonNullable<Parameters<typeof api.updateConfig>[0]['search']> = {};
      if (tavilyChanged) payload.tavily_api_key = keys.tavily_api_key;
      if (braveChanged) payload.brave_api_key = keys.brave_api_key;
      await api.updateConfig({ search: payload });
      notifySuccess('API keys saved', 'Web Search');
      // Re-fetch so key fields show the freshly-redacted values.
      await refresh();
    } catch (error) {
      notifyError(error, 'Failed to save API keys');
    } finally {
      setSavingKeys(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      const res = await api.searchHealth();
      setHealth(res.backends ?? null);
      if (res.ok) {
        notifySuccess(`Reached "${res.backend}" (${res.count} result${res.count === 1 ? '' : 's'})`, 'Web Search');
      } else {
        notifyError(res.error || 'No backend answered', 'Search test failed');
      }
    } catch (error) {
      notifyError(error, 'Search test failed');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Globe size={20} />}
        title="Web Search"
        description="How the agent searches the web — and what that's allowed to cost."
        actions={<SaveStatusChip status={status} />}
      />

      {loading || !settings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          {/* ---------------------------------------------------- Backend */}
          <SettingsSection
            title="Backend & keys"
            description="Tavily brings the research suite (extract, crawl, map, deep research). Brave returns pre-extracted page content and supports Goggles."
          >
            <SelectField
              label="Search Backend"
              value={settings.backend}
              onChange={v => update({ backend: v as Backend })}
              hint="Tried first for every search"
              options={[
                { value: 'tavily', label: 'Tavily (recommended)' },
                { value: 'brave', label: 'Brave' },
              ]}
              {...bind('backend')}
            />

            <ToggleField
              checked={settings.fallback_enabled}
              onChange={fallback_enabled => update({ fallback_enabled })}
              label="Fallback to other backend"
              hint="If the primary errors or returns nothing, try the other"
              {...bind('fallback_enabled')}
            />

            {/* Tavily API key — secrets keep explicit Save (never autosaved) */}
            <div className="setting-row">
              <label className="setting-label">
                <span className="flex items-center gap-1.5">
                  Tavily API Key
                  <Badge variant="accent" size="sm">Recommended</Badge>
                  {/* Secrets carry no field-kit chrome — no default to compare
                      against, nothing to reset to — but the authored help still
                      has the billing detail worth reading before you paste a key. */}
                  <SettingHelp help={helpFor('search.tavily_api_key')} label="Tavily API Key" />
                </span>
                <span className="setting-hint">
                  From tavily.com — generous free tier; unlocks web_extract / web_map / web_crawl /
                  web_research
                </span>
              </label>
              <div className="api-key-input">
                <Input
                  type={showKeys.tavily ? 'text' : 'password'}
                  value={keys.tavily_api_key}
                  onChange={(e) => setKeys((p) => ({ ...p, tavily_api_key: e.target.value }))}
                  placeholder="tvly-..."
                  autoComplete="off"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="visibility-toggle"
                  onClick={() => setShowKeys((p) => ({ ...p, tavily: !p.tavily }))}
                  aria-label={showKeys.tavily ? 'Hide value' : 'Show value'}
                >
                  {showKeys.tavily ? <EyeOff size={16} /> : <Eye size={16} />}
                </Button>
              </div>
            </div>

            {/* Brave API key */}
            <div className="setting-row">
              <label className="setting-label">
                <span className="flex items-center gap-1.5">
                  Brave API Key
                  <SettingHelp help={helpFor('search.brave_api_key')} label="Brave API Key" />
                </span>
                <span className="setting-hint">
                  From api.search.brave.com — the Search plan covers grounding; deep research needs
                  the separate Answers plan
                </span>
              </label>
              <div className="api-key-input">
                <Input
                  type={showKeys.brave ? 'text' : 'password'}
                  value={keys.brave_api_key}
                  onChange={(e) => setKeys((p) => ({ ...p, brave_api_key: e.target.value }))}
                  placeholder="BSA..."
                  autoComplete="off"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="visibility-toggle"
                  onClick={() => setShowKeys((p) => ({ ...p, brave: !p.brave }))}
                  aria-label={showKeys.brave ? 'Hide value' : 'Show value'}
                >
                  {showKeys.brave ? <EyeOff size={16} /> : <Eye size={16} />}
                </Button>
              </div>
            </div>

            {/* Actions — keys only; the knobs above autosave */}
            <div className="setting-actions">
              <Button variant="secondary" onClick={handleTest} loading={testing}>
                <Wifi size={16} />
                {testing ? 'Testing...' : 'Test connection'}
              </Button>
              <Button
                variant="primary"
                onClick={handleSaveKeys}
                loading={savingKeys}
                disabled={!tavilyChanged && !braveChanged}
              >
                <Save size={16} />
                {savingKeys ? 'Saving...' : 'Save keys'}
              </Button>
            </div>

            {health && (
              <ul className="search-health-list">
                {health.map(b => (
                  <li key={b.backend} className="search-health-row">
                    <span className="search-health-name">
                      {b.ok ? (
                        <Check size={14} className="text-success" aria-hidden />
                      ) : (
                        <X size={14} className="text-error" aria-hidden />
                      )}
                      {b.label}
                      {b.active && <Badge variant="accent" size="sm">Active</Badge>}
                    </span>
                    <span className="search-health-detail">
                      {b.ok ? b.tools.join(', ') : (b.error || 'unavailable')}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </SettingsSection>

          {/* --------------------------------------------------- Defaults */}
          <SettingsSection
            title="Search defaults"
            description="Applied when the agent doesn't ask for something specific. Leave blank to let the provider decide."
          >
            <NumberField
              label="Max Results"
              value={settings.max_results}
              min={1}
              max={20}
              fallback={5}
              onChange={max_results => update({ max_results })}
              title="Results returned per search (1–20)"
              {...bind('max_results')}
            />

            <SelectField
              label="Search depth"
              value={settings.default_search_depth}
              onChange={v => update({ default_search_depth: v })}
              hint="Tavily: deeper digs find more but cost more (advanced bills 2 credits, the rest 1)"
              options={[
                { value: '', label: 'Provider default' },
                { value: 'ultra-fast', label: 'Ultra-fast — lowest latency' },
                { value: 'fast', label: 'Fast — relevant chunks, quick' },
                { value: 'basic', label: 'Basic — balanced' },
                { value: 'advanced', label: 'Advanced — deepest (2 credits)' },
              ]}
              {...bind('default_search_depth')}
            />

            <NumberField
              label="Chunks per source"
              value={settings.default_chunks_per_source}
              min={0}
              max={5}
              fallback={0}
              onChange={default_chunks_per_source => update({ default_chunks_per_source })}
              title="How much of each result to keep (1–5). 0 = provider default"
              {...bind('default_chunks_per_source')}
            />

            <SelectField
              label="Safe search"
              value={settings.safesearch}
              onChange={v => update({ safesearch: v })}
              hint="Brave only"
              options={[
                { value: '', label: 'Provider default' },
                { value: 'off', label: 'Off' },
                { value: 'moderate', label: 'Moderate' },
                { value: 'strict', label: 'Strict' },
              ]}
              {...bind('safesearch')}
            />

            <TextField
              label="Country"
              value={settings.country}
              onChange={country => update({ country })}
              placeholder="e.g. GB"
              hint="Bias results toward a country. Blank = no preference"
              {...bind('country')}
            />

            <TextField
              label="Results language"
              value={settings.search_lang}
              onChange={search_lang => update({ search_lang })}
              placeholder="e.g. en"
              hint="Preferred language for results. Blank = no preference"
              {...bind('search_lang')}
            />
          </SettingsSection>

          {/* ---------------------------------------------------- Budgets */}
          <SettingsSection
            title="Budgets"
            description="Ceilings for a single turn. Whichever runs out first stops that turn's searching — set either to 0 to disable it."
          >
            <NumberField
              label="Searches per turn"
              value={settings.per_turn_limit}
              min={0}
              max={100}
              fallback={0}
              onChange={per_turn_limit => update({ per_turn_limit })}
              title="Max searches in one turn. 0 = unlimited"
              {...bind('per_turn_limit')}
            />

            <NumberField
              label="Spend per turn (USD)"
              value={settings.per_turn_cost_usd}
              min={0}
              max={100}
              step={0.05}
              fallback={0}
              onChange={per_turn_cost_usd => update({ per_turn_cost_usd })}
              title="Dollar ceiling for one turn's searches. 0 = no cost ceiling"
              {...bind('per_turn_cost_usd')}
            />

            <NumberField
              label="Spend per research turn (USD)"
              value={settings.research_per_turn_cost_usd}
              min={0}
              max={100}
              step={0.05}
              fallback={0}
              onChange={research_per_turn_cost_usd => update({ research_per_turn_cost_usd })}
              title="Dollar ceiling while Research Mode is on. 0 = no cost ceiling"
              {...bind('research_per_turn_cost_usd')}
            />

            <NumberField
              label="Search timeout (seconds)"
              value={settings.timeout}
              min={5}
              max={120}
              fallback={15}
              onChange={timeout => update({ timeout })}
              title="Hard per-call cap — a hung search blocks the turn until it returns"
              {...bind('timeout')}
            />

            <NumberField
              label="Cache lifetime (seconds)"
              value={settings.cache_ttl_seconds}
              min={0}
              max={3600}
              fallback={0}
              onChange={cache_ttl_seconds => update({ cache_ttl_seconds })}
              title="Repeat searches are served free from cache for this long. 0 = no caching"
              {...bind('cache_ttl_seconds')}
            />
          </SettingsSection>

          {/* ------------------------------------------------ Brave detail */}
          <SettingsSection
            title="Brave grounding"
            description="Brave can return pre-extracted page content instead of a link list — search and extract in one call. These bound how much context that may claim."
          >
            <ToggleField
              checked={settings.brave_grounding_default}
              onChange={brave_grounding_default => update({ brave_grounding_default })}
              label="Return page content by default"
              hint="Off = a plain link list, cheaper but the agent must extract separately"
              {...bind('brave_grounding_default')}
            />

            <NumberField
              label="Context budget (tokens)"
              value={settings.brave_context_max_tokens}
              min={1024}
              max={32768}
              fallback={4096}
              onChange={brave_context_max_tokens => update({ brave_context_max_tokens })}
              title="Ceiling on one grounded search's returned content"
              {...bind('brave_context_max_tokens')}
            />

            <NumberField
              label="Per-page budget (tokens)"
              value={settings.brave_context_max_tokens_per_url}
              min={128}
              max={8192}
              fallback={1024}
              onChange={brave_context_max_tokens_per_url =>
                update({ brave_context_max_tokens_per_url })}
              title="Most any single page may contribute, so one long result can't take the whole budget"
              {...bind('brave_context_max_tokens_per_url')}
            />

            <SelectField
              label="Relevance threshold"
              value={settings.brave_context_threshold}
              onChange={v => update({ brave_context_threshold: v })}
              hint="How picky to be about which passages come back"
              options={[
                { value: 'strict', label: 'Strict — fewer, more relevant' },
                { value: 'balanced', label: 'Balanced' },
                { value: 'lenient', label: 'Lenient — more, less relevant' },
              ]}
              {...bind('brave_context_threshold')}
            />

            <ToggleField
              checked={settings.brave_answers_enabled}
              onChange={brave_answers_enabled => update({ brave_answers_enabled })}
              label="Brave deep research"
              hint="Needs the separate Answers plan on your key — leave off unless you have it, or every deep-research call will fail"
              {...bind('brave_answers_enabled')}
            />
          </SettingsSection>

          {/* ----------------------------------------------- Source policy */}
          <SettingsSection
            title="Source policy"
            description="Which corners of the web may ground an answer. Applied to every search — comma-separated domains."
          >
            <TextField
              label="Preferred sources"
              value={settings.trusted}
              onChange={trusted => update({ trusted })}
              placeholder="arxiv.org, *.edu"
              hint="Favoured when the agent hasn't scoped a search itself. A preference, not a restriction — over-narrowing returns nothing"
              binding={policyBinding}
              onReset={resetPolicy}
            />

            <TextField
              label="Blocked sources"
              value={settings.blocked}
              onChange={blocked => update({ blocked })}
              placeholder="pinterest.com, quora.com"
              hint="Never returned. A hard floor — the agent cannot search past it"
            />

            <TextField
              label="Brave Goggle"
              value={settings.goggle}
              onChange={goggle => update({ goggle })}
              placeholder="https://…/my.goggle  or  $discard,site=example.com"
              hint="Brave only — a hosted Goggle URL, or inline rules for finer re-ranking than the lists above"
            />
          </SettingsSection>
        </div>
      )}
    </div>
  );
}
