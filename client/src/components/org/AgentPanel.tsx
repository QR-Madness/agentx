/**
 * AgentPanel — one agent's place in the organization, in the dock: which teams
 * it is on and in what role, who it reports to and directs, a quick edit of the
 * profile fields the organization reads (role, roster, specialty), and the links
 * out to the surfaces that own the rest (the profile editor for identity, the
 * Roster for the full dossier).
 */

import { useEffect, useState } from 'react';
import { ArrowUpRight, PenLine, Plus, UserRound } from 'lucide-react';
import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useNotify } from '../../contexts/NotificationContext';
import type { AgentProfile, AgentProfileCreate } from '../../lib/api';
import { modelShortLabel } from '../../lib/modelLabel';
import { editableOrgRole, ORG_ROLE_OPTIONS } from '../../lib/orgRoles';
import type { OrgIssue, StructuralTier } from '../../lib/orgChart';
import type { OrgPlacement } from '../../lib/orgPlacement';
import { AgentAvatar } from '../common/AgentAvatar';
import { Button } from '../ui/Button';
import { Input } from '../ui/Field';
import { SegmentedControl } from '../ui/SegmentedControl';
import { Switch } from '../ui/Switch';
import { TierChip } from './TierChip';

interface AgentPanelProps {
  profile: AgentProfile;
  placement: OrgPlacement;
  tier: StructuralTier | undefined;
  issues: OrgIssue[];
  onOpenTeam: (teamId: string) => void;
  onOpenAgent: (agentId: string) => void;
  onNewTeam: (managerAgentId: string) => void;
  onEditProfile: (profileId: string) => void;
  onOpenDossier: (profileId: string) => void;
}

export function AgentPanel({
  profile,
  placement,
  tier,
  issues,
  onOpenTeam,
  onOpenAgent,
  onNewTeam,
  onEditProfile,
  onOpenDossier,
}: AgentPanelProps) {
  const isAmbassador = profile.kind === 'ambassador';
  const specialty = (profile.delegationHint || profile.description || '').trim();

  return (
    <div className="org-panel">
      <div className="org-panel-head">
        <span className="org-panel-avatar" aria-hidden>
          <AgentAvatar avatar={profile.avatar} size={40} fill />
        </span>
        <div className="org-panel-ident">
          <div className="org-panel-name">
            {profile.name}
            {isAmbassador ? <TierChip tier="ambassador" /> : tier && tier !== 'member' && <TierChip tier={tier} />}
          </div>
          <div className="org-panel-id">{profile.agentId}</div>
        </div>
      </div>
      {specialty && <p className="org-panel-specialty">{specialty}</p>}

      {issues.length > 0 && (
        <ul className="org-panel-issues">
          {issues.map((issue) => (
            <li key={issue.message}>{issue.message}</li>
          ))}
        </ul>
      )}

      <section className="org-section">
        <h3 className="org-eyebrow">Place in the organization</h3>
        {isAmbassador ? (
          <p className="org-quiet">
            The ambassador sits above the chain of command and outside it. It can dispatch work to
            any agent, and no agent delegates to it.
          </p>
        ) : placement.teams.length === 0 ? (
          <p className="org-quiet">
            On no team.
            {profile.availableForDelegation
              ? ' Other independent agents can hand it subtasks.'
              : ' It is not offered to other agents for delegation.'}
          </p>
        ) : (
          <ul className="org-placement">
            {placement.teams.map((t) => (
              <li key={`${t.role}:${t.teamId}`}>
                <button type="button" className="org-placement-row" onClick={() => onOpenTeam(t.teamId)}>
                  <span className={`org-role org-role--${t.role}`}>{t.role}</span>
                  <span className="org-placement-team">{t.teamName}</span>
                  <ArrowUpRight size={13} />
                </button>
              </li>
            ))}
          </ul>
        )}

        {placement.superiors.length > 0 && (
          <ChipLine label="Reports to" agents={placement.superiors} onOpenAgent={onOpenAgent} />
        )}
        {placement.subordinates.length > 0 && (
          <ChipLine label="Directs" agents={placement.subordinates} onOpenAgent={onOpenAgent} />
        )}
      </section>

      <section className="org-section">
        <h3 className="org-eyebrow">Profile</h3>
        {!isAmbassador && <QuickEdit profile={profile} />}
        <dl className="org-kv">
          {isAmbassador && (
            <>
              <dt>Role</dt>
              <dd>Ambassador</dd>
            </>
          )}
          <dt>Model</dt>
          <dd>{modelShortLabel(profile.defaultModel) ?? 'Inherits the default'}</dd>
        </dl>
      </section>

      <div className="org-panel-actions">
        <Button variant="secondary" size="sm" onClick={() => onEditProfile(profile.id)}>
          <PenLine size={14} />
          Edit profile
        </Button>
        {!isAmbassador && (
          <Button variant="ghost" size="sm" onClick={() => onOpenDossier(profile.id)}>
            <UserRound size={14} />
            Dossier
          </Button>
        )}
        {!isAmbassador && (profile.orgLevel === 'manager' || tier === 'manager') && (
          <Button variant="ghost" size="sm" onClick={() => onNewTeam(profile.agentId)}>
            <Plus size={14} />
            New team
          </Button>
        )}
      </div>
    </div>
  );
}

function ChipLine({
  label,
  agents,
  onOpenAgent,
}: {
  label: string;
  agents: AgentProfile[];
  onOpenAgent: (agentId: string) => void;
}) {
  return (
    <div className="org-chipline">
      <span className="org-chipline-label">{label}</span>
      {agents.map((a) => (
        <button key={a.agentId} type="button" className="org-chip" onClick={() => onOpenAgent(a.agentId)}>
          <AgentAvatar avatar={a.avatar} size={14} />
          {a.name}
        </button>
      ))}
    </div>
  );
}

/**
 * The profile fields the organization reads, edited in place: role, roster and
 * specialty. Role and roster save as soon as they change; the specialty saves on
 * request. Choosing Manager makes the server apply the report-only tool limits, as
 * the profile editor does.
 */
function QuickEdit({ profile }: { profile: AgentProfile }) {
  const { updateProfile } = useAgentProfile();
  const { notifyError } = useNotify();
  const [saving, setSaving] = useState<'role' | 'roster' | 'specialty' | null>(null);
  const [specialty, setSpecialty] = useState(profile.delegationHint ?? '');
  useEffect(() => setSpecialty(profile.delegationHint ?? ''), [profile.id, profile.delegationHint]);

  const role = editableOrgRole(profile.orgLevel);
  const specialtyDirty = specialty.trim() !== (profile.delegationHint ?? '').trim();

  const save = async (
    field: 'role' | 'roster' | 'specialty',
    updates: Partial<AgentProfileCreate>,
    what: string,
  ) => {
    setSaving(field);
    const saved = await updateProfile(profile.id, updates);
    setSaving(null);
    if (!saved) {
      notifyError(new Error(`${profile.name}'s ${what} could not be saved. It is unchanged.`), 'Quick edit');
    }
  };

  return (
    <div className="org-quick">
      <div className="org-quick-row">
        <span className="org-quick-label">Role</span>
        <SegmentedControl
          size="sm"
          ariaLabel={`${profile.name}'s role`}
          value={role}
          options={ORG_ROLE_OPTIONS}
          onChange={(next) => {
            if (next !== role && saving === null) void save('role', { org_level: next }, 'role');
          }}
        />
      </div>
      {role !== 'manager' && (
        <p className="org-quick-hint">Manager also applies the report-only tool limits.</p>
      )}
      <div className="org-quick-row">
        <span className="org-quick-label" id={`roster-${profile.id}`}>
          Join the team roster
        </span>
        <Switch
          aria-labelledby={`roster-${profile.id}`}
          checked={Boolean(profile.availableForDelegation)}
          disabled={saving === 'roster'}
          onCheckedChange={(on) => void save('roster', { available_for_delegation: on }, 'roster setting')}
        />
      </div>
      <div className="org-quick-field">
        <span className="org-quick-label">Specialty</span>
        <div className="org-quick-inline">
          <Input
            value={specialty}
            maxLength={200}
            onChange={(e) => setSpecialty(e.target.value)}
            placeholder="What this agent is best at"
            aria-label={`${profile.name}'s specialty`}
          />
          {specialtyDirty && (
            <Button
              variant="secondary"
              size="sm"
              disabled={saving === 'specialty'}
              onClick={() => void save('specialty', { delegation_hint: specialty.trim() || null }, 'specialty')}
            >
              Save
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

