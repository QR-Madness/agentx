/**
 * ModelRolesSection — the three model roles (settings overhaul D1).
 *
 * One model per workload cluster: Fast Utility (quick classification &
 * extraction), Deep Reasoning (consolidation & distillation), Summarizer
 * (compression & recaps). A member setting left empty follows its role; an
 * explicit member value always wins — the member list below each role shows
 * the live resolution (`GET /api/models/roles`) so there's no guessing.
 */

import { useState } from 'react';
import { Boxes, RefreshCw } from 'lucide-react';
import { api } from '../../../lib/api';
import type { ModelRoleMember, ModelRoleName } from '../../../lib/api';
import { useApi } from '../../../lib/hooks';
import { useNotify } from '../../../contexts/NotificationContext';
import { Badge, SectionHeader } from '../../ui';
import { ModelPickerField } from '../../common/ModelPickerField';
import { SettingsSection } from '../../settings/fields';

const ROLE_ORDER: ModelRoleName[] = ['fast_utility', 'deep_reasoning', 'summarizer'];

/** Human-readable tail of a provider:model id. */
function shortModel(id: string): string {
  if (!id) return '';
  const parts = id.split(':');
  return parts.length > 1 ? parts.slice(1).join(':') : id;
}

function MemberChip({ member }: { member: ModelRoleMember }) {
  const { following } = member;
  return (
    <div className="flex items-center justify-between gap-3 py-1.5 border-b border-line-subtle last:border-b-0">
      <span className="text-sm text-fg-secondary">{member.label}</span>
      <span className="flex items-center gap-2 min-w-0">
        {following === 'role' && (
          <Badge variant="accent">following role</Badge>
        )}
        {following === 'explicit' && (
          <Badge variant="neutral">custom</Badge>
        )}
        {following === 'fallback' && (
          <Badge variant="neutral">fallback chain</Badge>
        )}
        <span
          className="text-xs font-mono text-fg-muted truncate max-w-[16rem]"
          title={member.effective || 'resolved by the provider fallback chain'}
        >
          {member.effective ? shortModel(member.effective) : 'auto'}
        </span>
      </span>
    </div>
  );
}

export default function ModelRolesSection() {
  const { notifyError } = useNotify();
  const [saving, setSaving] = useState<ModelRoleName | null>(null);
  const { data, loading, refresh } = useApi(() => api.getModelRoles(), [], {
    onError: err => notifyError(err, 'Model roles'),
  });

  const setRole = async (role: ModelRoleName, model: string) => {
    setSaving(role);
    try {
      await api.updateModelRoles({ [role]: model });
      await refresh();
    } catch (err) {
      notifyError(err, 'Failed to update the model role');
    } finally {
      setSaving(null);
    }
  };

  return (
    <div className="settings-section fade-in">
      <SectionHeader
        icon={<Boxes size={20} />}
        title="Model Roles"
        description="Set one model per workload — features left on their default follow their role automatically. An explicit per-feature model always wins."
      />

      {loading || !data ? (
        <div className="loading-state">
          <RefreshCw size={24} className="spin" />
          <span>Loading roles...</span>
        </div>
      ) : (
        <div className="settings-content">
          {ROLE_ORDER.map(name => {
            const role = data.roles[name];
            if (!role) return null;
            const members = data.members.filter(m => m.role === name);
            return (
              <SettingsSection key={name} title={role.label} description={role.description}>
                <ModelPickerField
                  value={role.model}
                  onChange={model => void setRole(name, model)}
                  showDefault={true}
                  placeholder="Not set — members use their own defaults"
                  hint={saving === name ? 'Saving…' : undefined}
                />
                <div className="mt-2">
                  {members.map(m => (
                    <MemberChip key={m.member} member={m} />
                  ))}
                </div>
              </SettingsSection>
            );
          })}
        </div>
      )}
    </div>
  );
}
