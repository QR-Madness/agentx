/**
 * SearchSection — Web search backend configuration (Track B/C)
 *
 * Picks the backend (Tavily / Brave), holds the API keys, and toggles fallback.
 * API keys arrive redacted from GET /api/config (e.g. "***1234"); we only send a
 * key back on save when the user has typed a new value (not the redacted mask).
 */

import { useEffect, useState } from 'react';
import { Globe, RefreshCw, Save, Eye, EyeOff, Wifi } from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Badge, Button, Input, SectionHeader } from '../../ui';

type Backend = 'tavily' | 'brave';

const isRedacted = (v: string) => v.startsWith('***');

export default function SearchSection() {
  const { notifyError, notifySuccess } = useNotify();

  const [settings, setSettings] = useState<{
    backend: Backend;
    fallback_enabled: boolean;
    max_results: number;
    tavily_api_key: string;
    brave_api_key: string;
  }>({
    backend: 'tavily',
    fallback_enabled: true,
    max_results: 5,
    tavily_api_key: '',
    brave_api_key: '',
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showKeys, setShowKeys] = useState<{ tavily: boolean; brave: boolean }>({
    tavily: false,
    brave: false,
  });

  useEffect(() => {
    void fetchSettings();
  }, []);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const config = await api.getConfig();
      const s = (config.search || {}) as {
        backend?: Backend;
        fallback_enabled?: boolean;
        max_results?: number;
        tavily_api_key?: string;
        brave_api_key?: string;
      };
      setSettings({
        backend: s.backend === 'brave' ? 'brave' : 'tavily',
        fallback_enabled: s.fallback_enabled ?? true,
        max_results: s.max_results ?? 5,
        tavily_api_key: s.tavily_api_key || '',
        brave_api_key: s.brave_api_key || '',
      });
    } catch (error) {
      notifyError(error, 'Failed to load search settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const update: NonNullable<Parameters<typeof api.updateConfig>[0]['search']> = {
        backend: settings.backend,
        fallback_enabled: settings.fallback_enabled,
        max_results: settings.max_results,
      };
      // Only send keys the user actually changed (skip empty + redacted mask).
      if (settings.tavily_api_key && !isRedacted(settings.tavily_api_key)) {
        update.tavily_api_key = settings.tavily_api_key;
      }
      if (settings.brave_api_key && !isRedacted(settings.brave_api_key)) {
        update.brave_api_key = settings.brave_api_key;
      }
      await api.updateConfig({ search: update });
      notifySuccess('Search settings saved', 'Web Search');
      // Re-fetch so key fields show the freshly-redacted values.
      void fetchSettings();
    } catch (error) {
      notifyError(error, 'Failed to save search settings');
    } finally {
      setSaving(false);
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
      />

      {loading ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          {/* Backend selector */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Search Backend</span>
              <span className="setting-hint">Primary provider for web search</span>
            </label>
            <select
              className="form-input"
              value={settings.backend}
              onChange={(e) => setSettings((p) => ({ ...p, backend: e.target.value as Backend }))}
            >
              <option value="tavily">Tavily (recommended)</option>
              <option value="brave">Brave</option>
            </select>
          </div>

          {/* Fallback toggle */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Fallback to other backend</span>
              <span className="setting-hint">If the primary errors or returns nothing, try the other</span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.fallback_enabled}
                onChange={(e) => setSettings((p) => ({ ...p, fallback_enabled: e.target.checked }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {/* Max results */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Max Results</span>
              <span className="setting-hint">Results returned per search (1–10)</span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.max_results}
                onChange={(e) =>
                  setSettings((p) => ({ ...p, max_results: parseInt(e.target.value) || 5 }))
                }
                min={1}
                max={10}
                step={1}
              />
            </div>
          </div>

          {/* Tavily API key */}
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
                value={settings.tavily_api_key}
                onChange={(e) => setSettings((p) => ({ ...p, tavily_api_key: e.target.value }))}
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
                value={settings.brave_api_key}
                onChange={(e) => setSettings((p) => ({ ...p, brave_api_key: e.target.value }))}
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

          {/* Actions */}
          <div className="setting-actions">
            <Button variant="secondary" onClick={handleTest} loading={testing}>
              <Wifi size={16} />
              {testing ? 'Testing...' : 'Test connection'}
            </Button>
            <Button variant="primary" onClick={handleSave} loading={saving}>
              <Save size={16} />
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
