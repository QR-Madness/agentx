/**
 * AmbassadorSection — global Ambassador configuration.
 *
 * An ambassador is its own profile *kind* (a parallel conversation interpreter).
 * Here the user toggles the feature globally and picks the **default ambassador**
 * (the one briefings use) from their ambassador profiles; per-ambassador settings
 * (personality, voices, verbosity) live in the ambassador profile editor. The
 * global model + grounding-turns defaults persist under `ambassador.*`. Built on
 * the settings field kit + autosave (useSettingsAutosave).
 */

import { useState } from 'react';
import { Radio, RefreshCw, Plus, Pencil } from 'lucide-react';
import { api } from '../../../lib/api';
import type { AgentProfile } from '../../../lib/api';
import { useSettingsAutosave } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { useModal } from '../../../contexts/ModalContext';
import { Button, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import {
  NumberField,
  SaveStatusChip,
  SelectField,
  ToggleField,
} from '../../settings/fields';

interface AmbassadorSettings extends Record<string, unknown> {
  enabled: boolean;
  model: string; // '' = use the profile's model / floor
  max_context_turns: number;
  aideEnabled: boolean; // aide swarm: condense conversations via cheap parallel aides
  dispatchEnabled: boolean; // hand a task to a chosen worker (mints a new conversation)
  defaultAmbassadorId: string; // profile flag, persisted via setDefaultAmbassador
}

export default function AmbassadorSection() {
  const { notifyError } = useNotify();
  const { openModal } = useModal();
  const [ambassadors, setAmbassadors] = useState<AgentProfile[]>([]);

  const { settings, loading, status, update } = useSettingsAutosave<AmbassadorSettings>({
    load: async () => {
      const [config, profileRes] = await Promise.all([
        api.getConfig(),
        api.listAgentProfiles(),
      ]);
      const ambs = profileRes.profiles.filter(p => p.kind === 'ambassador');
      setAmbassadors(ambs);
      const a = (config.ambassador || {}) as Partial<AmbassadorSettings> & {
        model?: string | null;
        aide?: { enabled?: boolean };
        dispatch?: { enabled?: boolean };
      };
      return {
        enabled: a.enabled ?? true,
        model: a.model || '',
        max_context_turns: a.max_context_turns ?? 8,
        aideEnabled: a.aide?.enabled ?? true,
        dispatchEnabled: a.dispatch?.enabled ?? true,
        defaultAmbassadorId:
          ambs.find(p => p.isDefaultAmbassador)?.id ?? ambs[0]?.id ?? '',
      };
    },
    save: async changed => {
      // The default ambassador is a profile flag, not config; persist it via
      // its own endpoint, then clear the legacy `profile_id` override below so
      // the default-ambassador mechanism stays authoritative.
      if ('defaultAmbassadorId' in changed) {
        const id = String(changed.defaultAmbassadorId ?? '');
        if (id) await api.setDefaultAmbassador(id);
      }
      const payload: Record<string, unknown> = { profile_id: null };
      if ('enabled' in changed) payload.enabled = changed.enabled;
      // Empty string → null so the backend falls back to the profile's model.
      if ('model' in changed) {
        payload.model = String(changed.model ?? '').trim() ? changed.model : null;
      }
      if ('max_context_turns' in changed) {
        payload.max_context_turns = changed.max_context_turns;
      }
      // The backend merges aide/dispatch sub-keys individually.
      if ('aideEnabled' in changed) payload.aide = { enabled: changed.aideEnabled };
      if ('dispatchEnabled' in changed) {
        payload.dispatch = { enabled: changed.dispatchEnabled };
      }
      await api.updateConfig({ ambassador: payload });
    },
    onError: err => notifyError(err, 'Ambassador settings'),
  });

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
    const id = settings?.defaultAmbassadorId;
    if (!id) return;
    openModal({
      id: 'profile-editor', type: 'modal', component: 'unifiedProfileEditor',
      size: 'full', props: { initialProfileId: id },
    });
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Radio size={20} />}
        title="Ambassador"
        description="A dedicated agent that runs parallel to a conversation and briefs you on a turn — without entering the conversation itself. Use the CC button on any reply."
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
            label="Enable Ambassador"
            hint="When off, the CC button reports the ambassador as disabled."
          />

          <SelectField
            label="Default Ambassador"
            value={settings.defaultAmbassadorId}
            onChange={defaultAmbassadorId => update({ defaultAmbassadorId })}
            options={ambassadors.map(p => ({ value: p.id, label: p.name }))}
            placeholder="No ambassadors yet"
            disabled={ambassadors.length === 0}
            hint="Which ambassador profile briefings use. Edit its personality and voices in the ambassador profile editor."
          />

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleEditAmbassador}
              disabled={!settings.defaultAmbassadorId}
            >
              <Pencil size={14} /> Edit
            </Button>
            <Button variant="secondary" size="sm" onClick={() => void handleNewAmbassador()}>
              <Plus size={14} /> New
            </Button>
          </div>

          <div className="setting-row">
            <ModelPickerField
              label="Ambassador Model"
              value={settings.model}
              onChange={model => update({ model })}
              showDefault={true}
            />
          </div>

          <NumberField
            label="Grounding Turns"
            value={settings.max_context_turns}
            min={1}
            max={40}
            fallback={8}
            onChange={max_context_turns => update({ max_context_turns })}
            title="How many recent turns the ambassador reads (read-only) to ground each briefing."
          />

          <ToggleField
            checked={settings.aideEnabled}
            onChange={aideEnabled => update({ aideEnabled })}
            label="Aide swarm"
            hint={
              <>
                When surveying or summarizing conversations, fan out cheap parallel
                "aide" models that each condense one conversation, so the ambassador
                stays high-level instead of reading full transcripts. Off ⇒ it reads
                transcripts directly.
              </>
            }
          />

          <ToggleField
            checked={settings.dispatchEnabled}
            onChange={dispatchEnabled => update({ dispatchEnabled })}
            label="Dispatch to workers"
            hint="Let the ambassador hand a task to a worker you pick — it opens a new conversation with that agent and runs it. Off ⇒ the Dispatch option is hidden."
          />
        </div>
      )}
    </div>
  );
}
