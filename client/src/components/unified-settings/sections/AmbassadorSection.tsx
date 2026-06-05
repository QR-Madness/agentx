/**
 * AmbassadorSection — global Ambassador (Phase 16.6) configuration.
 *
 * The ambassador is a parallel conversation interpreter (an agent profile with
 * an `ambassador` section). Here the user picks the *global default* ambassador
 * profile (`ambassador.profile_id`) and toggles the feature. Settings live under
 * `ambassador.*` in data/config.json.
 */

import { useEffect, useState } from 'react';
import { Radio, RefreshCw, Save } from 'lucide-react';
import { api } from '../../../lib/api';
import type { AgentProfile } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, SectionHeader } from '../../ui';

interface AmbassadorSettings {
  enabled: boolean;
  profile_id: string; // '' = use the default agent profile
  max_context_turns: number;
}

const DEFAULT_SETTINGS: AmbassadorSettings = {
  enabled: true,
  profile_id: '',
  max_context_turns: 8,
};

export default function AmbassadorSection() {
  const { notifyError, notifySuccess } = useNotify();
  const [settings, setSettings] = useState<AmbassadorSettings>(DEFAULT_SETTINGS);
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void load();
  }, []);

  const load = async () => {
    setLoading(true);
    try {
      const [config, profileRes] = await Promise.all([
        api.getConfig(),
        api.listAgentProfiles(),
      ]);
      setProfiles(profileRes.profiles);
      const a = (config.ambassador || {}) as Partial<AmbassadorSettings> & {
        profile_id?: string | null;
      };
      setSettings({
        enabled: a.enabled ?? true,
        profile_id: a.profile_id || '',
        max_context_turns: a.max_context_turns ?? 8,
      });
    } catch (error) {
      notifyError(error, 'Failed to load ambassador settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateConfig({
        ambassador: {
          enabled: settings.enabled,
          // Empty → null so the backend falls back to the default agent profile.
          profile_id: settings.profile_id.trim() ? settings.profile_id : null,
          max_context_turns: settings.max_context_turns,
        },
      });
      notifySuccess('Ambassador settings saved', 'Ambassador');
    } catch (error) {
      notifyError(error, 'Failed to save ambassador settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Radio size={20} />}
        title="Ambassador"
        description="A dedicated agent that runs parallel to a conversation and briefs you on a turn — without entering the conversation itself. Use the CC button on any reply."
      />

      {loading ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          <div className="setting-row">
            <label className="setting-label">
              <span>Enable Ambassador</span>
              <span className="setting-hint">
                When off, the CC button reports the ambassador as disabled.
              </span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.enabled}
                onChange={(e) => setSettings((prev) => ({ ...prev, enabled: e.target.checked }))}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Ambassador Profile</span>
              <span className="setting-hint">
                Which agent profile acts as the global ambassador. Its model and persona, plus its
                profile's ambassador section, shape the briefing. Defaults to your default agent.
              </span>
            </label>
            <select
              className="form-input"
              value={settings.profile_id}
              onChange={(e) => setSettings((prev) => ({ ...prev, profile_id: e.target.value }))}
            >
              <option value="">Default agent profile</option>
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                  {p.ambassador?.enabled ? ' — ambassador' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="setting-row">
            <label className="setting-label">
              <span>Grounding Turns</span>
              <span className="setting-hint">
                How many recent turns the ambassador reads (read-only) to ground each briefing.
              </span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.max_context_turns}
                onChange={(e) =>
                  setSettings((prev) => ({
                    ...prev,
                    max_context_turns: parseInt(e.target.value) || 8,
                  }))
                }
                min={1}
                max={40}
                step={1}
              />
            </div>
          </div>

          <div className="setting-actions">
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
