/**
 * AlloySection — Multi-agent (Agent Alloy) delegation settings (Track D)
 *
 * Surfaces the previously config-only delegation controls: the global ad-hoc
 * delegation switch (lets any agent delegate to any delegatable profile) and the
 * fan-out concurrency bound.
 */

import { useEffect, useState } from 'react';
import { Users, RefreshCw, Save } from 'lucide-react';
import { api } from '../../../lib/api';
import { useNotify } from '../../../contexts/NotificationContext';
import { Button, SectionHeader } from '../../ui';

export default function AlloySection() {
  const { notifyError, notifySuccess } = useNotify();

  const [settings, setSettings] = useState<{
    allow_adhoc_delegation: boolean;
    max_parallel_delegations: number;
    max_delegation_depth: number;
  }>({
    allow_adhoc_delegation: false,
    max_parallel_delegations: 3,
    max_delegation_depth: 3,
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    void fetchSettings();
  }, []);

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const config = await api.getConfig();
      const a = (config.alloy || {}) as {
        allow_adhoc_delegation?: boolean;
        max_parallel_delegations?: number;
        max_delegation_depth?: number;
      };
      setSettings({
        allow_adhoc_delegation: a.allow_adhoc_delegation ?? false,
        max_parallel_delegations: a.max_parallel_delegations ?? 3,
        max_delegation_depth: a.max_delegation_depth ?? 3,
      });
    } catch (error) {
      notifyError(error, 'Failed to load delegation settings');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.updateConfig({
        alloy: {
          allow_adhoc_delegation: settings.allow_adhoc_delegation,
          max_parallel_delegations: settings.max_parallel_delegations,
          max_delegation_depth: settings.max_delegation_depth,
        },
      });
      notifySuccess('Delegation settings saved', 'Multi-Agent');
    } catch (error) {
      notifyError(error, 'Failed to save delegation settings');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Users size={20} />}
        title="Multi-Agent Delegation"
        description="Let agents hand subtasks to other agents (Agent Alloy)."
      />

      {loading ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading settings...</span>
        </div>
      ) : (
        <div className="settings-content">
          {/* Ad-hoc delegation toggle */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Ad-hoc delegation</span>
              <span className="setting-hint">
                Give every agent a `delegate_to` tool so it can hand subtasks to any
                profile marked “available for delegation”.
              </span>
            </label>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.allow_adhoc_delegation}
                onChange={(e) =>
                  setSettings((p) => ({ ...p, allow_adhoc_delegation: e.target.checked }))
                }
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          {/* Max parallel delegations */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Max parallel delegations</span>
              <span className="setting-hint">
                How many specialists may run at once when an agent fans out in a
                single turn (1–8).
              </span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.max_parallel_delegations}
                onChange={(e) =>
                  setSettings((p) => ({
                    ...p,
                    max_parallel_delegations: parseInt(e.target.value) || 3,
                  }))
                }
                min={1}
                max={8}
                step={1}
              />
            </div>
          </div>

          {/* Max delegation depth */}
          <div className="setting-row">
            <label className="setting-label">
              <span>Max delegation depth</span>
              <span className="setting-hint">
                How many delegation hops deep a chain may go before it's rejected (1–5).
              </span>
            </label>
            <div className="input-with-hint">
              <input
                type="number"
                className="form-input"
                value={settings.max_delegation_depth}
                onChange={(e) =>
                  setSettings((p) => ({
                    ...p,
                    max_delegation_depth: parseInt(e.target.value) || 3,
                  }))
                }
                min={1}
                max={5}
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
