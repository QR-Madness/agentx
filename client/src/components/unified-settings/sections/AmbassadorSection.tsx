/**
 * AmbassadorSection — global Ambassador configuration.
 *
 * An ambassador is its own profile *kind* (a parallel conversation interpreter).
 * Here the user toggles the feature globally and picks the **default ambassador**
 * (the one briefings use) from their ambassador profiles; per-ambassador settings
 * (personality, voices, verbosity) live in the ambassador profile editor. The
 * global model + grounding-turns defaults persist under `ambassador.*`.
 */

import { useEffect, useState } from 'react';
import { Radio, RefreshCw, Save, Plus, Pencil } from 'lucide-react';
import { api } from '../../../lib/api';
import type { AgentProfile } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { useModal } from '../../../contexts/ModalContext';
import { Button, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';

interface AmbassadorSettings {
  enabled: boolean;
  model: string; // '' = use the profile's model / floor
  max_context_turns: number;
  aideEnabled: boolean; // aide swarm: condense conversations via cheap parallel aides
}

const DEFAULT_SETTINGS: AmbassadorSettings = {
  enabled: true,
  model: '',
  max_context_turns: 8,
  aideEnabled: true,
};

export default function AmbassadorSection() {
  const { notifyError, notifySuccess } = useNotify();
  const { openModal } = useModal();
  const [settings, setSettings] = useState<AmbassadorSettings>(DEFAULT_SETTINGS);
  const [ambassadors, setAmbassadors] = useState<AgentProfile[]>([]);
  const [defaultAmbassadorId, setDefaultAmbassadorId] = useState('');
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
      const ambs = profileRes.profiles.filter((p) => p.kind === 'ambassador');
      setAmbassadors(ambs);
      setDefaultAmbassadorId(ambs.find((p) => p.isDefaultAmbassador)?.id ?? ambs[0]?.id ?? '');
      const a = (config.ambassador || {}) as Partial<AmbassadorSettings> & {
        model?: string | null;
        aide?: { enabled?: boolean };
      };
      setSettings({
        enabled: a.enabled ?? true,
        model: a.model || '',
        max_context_turns: a.max_context_turns ?? 8,
        aideEnabled: a.aide?.enabled ?? true,
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
      // The default ambassador is now a profile flag; clear the legacy profile_id
      // override so the default-ambassador mechanism is authoritative.
      if (defaultAmbassadorId) {
        await api.setDefaultAmbassador(defaultAmbassadorId);
      }
      await api.updateConfig({
        ambassador: {
          enabled: settings.enabled,
          profile_id: null,
          model: settings.model.trim() ? settings.model : null,
          max_context_turns: settings.max_context_turns,
          aide: { enabled: settings.aideEnabled },
        },
      });
      notifySuccess('Ambassador settings saved', 'Ambassador');
      await load();
    } catch (error) {
      notifyError(error, 'Failed to save ambassador settings');
    } finally {
      setSaving(false);
    }
  };

  const handleNewAmbassador = async () => {
    try {
      const { profile } = await api.createAgentProfile({
        name: 'New Ambassador',
        kind: 'ambassador',
        avatar: 'radio',
      });
      openModal({
        id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor',
        size: 'full', props: { initialProfileId: profile.id },
      });
    } catch (error) {
      notifyError(error, 'Failed to create ambassador');
    }
  };

  const handleEditAmbassador = () => {
    if (!defaultAmbassadorId) return;
    openModal({
      id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor',
      size: 'full', props: { initialProfileId: defaultAmbassadorId },
    });
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
              <span>Default Ambassador</span>
              <span className="setting-hint">
                Which ambassador profile briefings use. Edit its personality and voices in the
                ambassador profile editor.
              </span>
            </label>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <select
                className="form-input"
                value={defaultAmbassadorId}
                onChange={(e) => setDefaultAmbassadorId(e.target.value)}
                disabled={ambassadors.length === 0}
                style={{ flex: 1, minWidth: 160 }}
              >
                {ambassadors.length === 0 && <option value="">No ambassadors yet</option>}
                {ambassadors.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <Button variant="secondary" size="sm" onClick={handleEditAmbassador} disabled={!defaultAmbassadorId}>
                <Pencil size={14} /> Edit
              </Button>
              <Button variant="secondary" size="sm" onClick={() => void handleNewAmbassador()}>
                <Plus size={14} /> New
              </Button>
            </div>
          </div>

          <div className="setting-row">
            <ModelPickerField
              label="Ambassador Model"
              value={settings.model}
              onChange={(modelId) => setSettings((prev) => ({ ...prev, model: modelId }))}
              showDefault={true}
            />
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

          <div className="setting-row">
            <label className="setting-label">
              <span>Aide swarm</span>
              <span className="setting-hint">
                When surveying or summarizing conversations, fan out cheap parallel "aide" models
                that each condense one conversation, so the ambassador stays high-level instead of
                reading full transcripts. Off ⇒ it reads transcripts directly.
              </span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.aideEnabled}
                onChange={(e) => setSettings((prev) => ({ ...prev, aideEnabled: e.target.checked }))}
              />
              <span className="toggle-slider"></span>
            </label>
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
