/**
 * AgentPanel — one agent's place in the organization, in the dock: which teams
 * it is on and in what role, who it reports to and directs, and the links out
 * to the surfaces that own the rest (the profile editor for identity, the
 * Roster for the full dossier).
 */

import { ArrowUpRight, PenLine, Plus, UserRound } from 'lucide-react';
import type { AgentProfile } from '../../lib/api';
import { modelShortLabel } from '../../lib/modelLabel';
import type { OrgIssue, StructuralTier } from '../../lib/orgChart';
import type { OrgPlacement } from '../../lib/orgPlacement';
import { AgentAvatar } from '../common/AgentAvatar';
import { Button } from '../ui/Button';
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

const ROLE_NAME: Record<NonNullable<AgentProfile['orgLevel']>, string> = {
  executive: 'Executive',
  manager: 'Manager',
  lead: 'Lead',
  agent: 'Agent',
};

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
        <dl className="org-kv">
          <dt>Role</dt>
          <dd>{isAmbassador ? 'Ambassador' : ROLE_NAME[profile.orgLevel ?? 'agent']}</dd>
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
