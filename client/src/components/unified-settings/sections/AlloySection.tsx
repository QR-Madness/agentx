/**
 * AlloySection — Multi-agent (Agent Alloy) delegation settings (Track D)
 *
 * Surfaces the previously config-only delegation controls under `alloy.*`:
 * the global ad-hoc delegation switch (lets any agent delegate to any
 * delegatable profile), fan-out concurrency/depth bounds, the per-delegation
 * timeout, and specialist tool inheritance. Built on the settings field kit +
 * autosave (useSettingsAutosave).
 */

import { Users, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { SectionHeader } from '../../ui';
import { NumberField, SaveStatusChip, ToggleField } from '../../settings/fields';

interface AlloySettings extends Record<string, unknown> {
  allow_adhoc_delegation: boolean;
  max_parallel_delegations: number;
  max_delegation_depth: number;
  delegation_timeout_seconds: number;
}

export default function AlloySection() {
  const { notifyError } = useNotify();

  const { settings, loading, status, update } = useSettingsAutosave<AlloySettings>({
    load: async () => {
      const config = await api.getConfig();
      const a = (config.alloy || {}) as Partial<AlloySettings>;
      return {
        allow_adhoc_delegation: a.allow_adhoc_delegation ?? true,
        max_parallel_delegations: a.max_parallel_delegations ?? 3,
        max_delegation_depth: a.max_delegation_depth ?? 3,
        delegation_timeout_seconds: a.delegation_timeout_seconds ?? 300,
      };
    },
    save: async changed => {
      await api.updateConfig({ alloy: changed });
    },
    onError: err => notifyError(err, 'Agent Teams settings'),
  });

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Users size={20} />}
        title="Agent Teams"
        description="Let agents delegate subtasks to each other, ad-hoc or in structured teams."
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
            checked={settings.allow_adhoc_delegation}
            onChange={allow_adhoc_delegation => update({ allow_adhoc_delegation })}
            label="Ad-hoc delegation"
            hint={
              <>
                Give every agent a `delegate_to` tool (plus a roster prompt) so it can
                hand subtasks to any profile that joined the team roster.
              </>
            }
          />

          <NumberField
            label="Max parallel delegations"
            value={settings.max_parallel_delegations}
            min={1}
            max={8}
            fallback={3}
            onChange={max_parallel_delegations => update({ max_parallel_delegations })}
            title="How many teammates may run at once when an agent fans out in a single turn (1–8)."
          />

          <NumberField
            label="Max delegation depth"
            value={settings.max_delegation_depth}
            min={1}
            max={5}
            fallback={3}
            onChange={max_delegation_depth => update({ max_delegation_depth })}
            title="How many delegation hops deep a chain may go before it's rejected (1–5)."
          />

          <NumberField
            label="Delegation timeout (seconds)"
            value={settings.delegation_timeout_seconds}
            min={30}
            max={3600}
            fallback={300}
            onChange={delegation_timeout_seconds => update({ delegation_timeout_seconds })}
            title="How long a delegated subtask may run before it's cancelled."
          />
        </div>
      )}
    </div>
  );
}
