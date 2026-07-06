/**
 * SearchSection — Web search backend configuration (Track B/C)
 *
 * Picks the backend (Tavily / Brave), holds the API keys, and toggles fallback.
 * Non-secret knobs autosave via useSettingsAutosave (settings field kit); the
 * API keys keep an explicit "Save keys" flow (secrets are never autosaved).
 * Keys arrive redacted from GET /api/config (e.g. "***1234"); we only send a
 * key back when the user has typed a new value (not the redacted mask).
 */

import { useState } from 'react';
import { Globe, RefreshCw, Save, Eye, EyeOff, Wifi } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { Badge, Button, Input, SectionHeader } from '../../ui';
import { NumberField, SaveStatusChip, SelectField, ToggleField } from '../../settings/fields';

type Backend = 'tavily' | 'brave';

interface SearchSettings extends Record<string, unknown> {
  backend: Backend;
  fallback_enabled: boolean;
  max_results: number;
}

const isRedacted = (v: string) => v.startsWith('***');

export default function SearchSection() {
  const { notifyError, notifySuccess } = useNotify();

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

  const { settings, loading, status, update, refresh } = useSettingsAutosave<SearchSettings>({
    load: async () => {
      const config = await api.getConfig();
      const s = (config.search || {}) as Partial<SearchSettings> & {
        tavily_api_key?: string;
        brave_api_key?: string;
      };
      // Seed the key inputs alongside the autosaved knobs (single fetch).
      setKeys({
        tavily_api_key: s.tavily_api_key || '',
        brave_api_key: s.brave_api_key || '',
      });
      return {
        backend: s.backend === 'brave' ? 'brave' : 'tavily',
        fallback_enabled: s.fallback_enabled ?? true,
        max_results: s.max_results ?? 5,
      };
    },
    save: async changed => {
      await api.updateConfig({ search: changed });
    },
    onError: err => notifyError(err, 'Web Search settings'),
  });

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
        description="Backend for the agent's web_search tool. Tavily is the default; Brave is the fallback."
        actions={<SaveStatusChip status={status} />}
      />

      {loading || !settings ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <SelectField
            label="Search Backend"
            value={settings.backend}
            onChange={v => update({ backend: v as Backend })}
            hint="Primary provider for web search"
            options={[
              { value: 'tavily', label: 'Tavily (recommended)' },
              { value: 'brave', label: 'Brave' },
            ]}
          />

          <ToggleField
            checked={settings.fallback_enabled}
            onChange={fallback_enabled => update({ fallback_enabled })}
            label="Fallback to other backend"
            hint="If the primary errors or returns nothing, try the other"
          />

          <NumberField
            label="Max Results"
            value={settings.max_results}
            min={1}
            max={10}
            fallback={5}
            onChange={max_results => update({ max_results })}
            title="Results returned per search (1–10)"
          />

          {/* Tavily API key — secrets keep explicit Save (never autosaved) */}
          <div className="setting-row">
            <label className="setting-label">
              <span className="flex items-center gap-1.5">
                Tavily API Key
                <Badge variant="accent" size="sm">Recommended</Badge>
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
              <span>Brave API Key</span>
              <span className="setting-hint">Used as the fallback backend</span>
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
        </div>
      )}
    </div>
  );
}
