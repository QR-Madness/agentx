/**
 * OrgOutline — the organization as a nested list: the same structure the canvas
 * draws, for narrow screens (where it replaces the canvas) and for anyone who
 * would rather scan a list. Rows select exactly what the canvas boxes select.
 */

import { Crown, Network, Radio, UserRound, Users } from 'lucide-react';
import type { AgentProfile } from '../../lib/api';
import type { OrgModel, OrgTeam } from '../../lib/orgChart';
import { AgentAvatar } from '../common/AgentAvatar';
import { TierChip, type ChipTier } from './TierChip';
import type { OrgSelection } from './types';

interface OrgOutlineProps {
  model: OrgModel;
  selection: OrgSelection;
  /** Lower-cased search text; rows that don't match are hidden. */
  query: string;
  onSelectAgent: (agentId: string) => void;
  onSelectTeam: (teamId: string) => void;
}

function matches(profile: AgentProfile, query: string): boolean {
  if (!query) return true;
  return (
    profile.name.toLowerCase().includes(query) ||
    profile.agentId.toLowerCase().includes(query) ||
    (profile.delegationHint ?? '').toLowerCase().includes(query)
  );
}

export function OrgOutline({ model, selection, query, onSelectAgent, onSelectTeam }: OrgOutlineProps) {
  /** `shown` keeps a row that doesn't match on its own (its team's name did). */
  const agentRow = (profile: AgentProfile, tier: ChipTier | null, key: string, indent = 0, shown = false) => {
    if (!shown && !matches(profile, query)) return null;
    const active = selection.kind === 'agent' && selection.agentId === profile.agentId;
    return (
      <button
        key={key}
        type="button"
        className={`org-outline-row${active ? ' active' : ''}`}
        style={{ paddingInlineStart: `${0.6 + indent * 0.9}rem` }}
        onClick={() => onSelectAgent(profile.agentId)}
      >
        <span className="org-avatar" aria-hidden>
          <AgentAvatar avatar={profile.avatar} size={16} fill />
        </span>
        <span className="org-outline-name">{profile.name}</span>
        {tier && <TierChip tier={tier} />}
      </button>
    );
  };

  const teamBlock = (team: OrgTeam, indent: number) => {
    // A team found by its name shows all its people, as the chart does.
    const teamMatches = !query || team.name.toLowerCase().includes(query);
    const rows = [
      team.lead ? agentRow(team.lead, 'lead', `${team.id}/lead`, indent + 1, teamMatches) : null,
      ...team.members.map((m) => agentRow(m, null, `${team.id}/${m.agentId}`, indent + 2, teamMatches)),
    ].filter(Boolean);
    if (!teamMatches && rows.length === 0) return null;
    const active = selection.kind === 'team' && selection.teamId === team.id;
    return (
      <div key={team.id} className="org-outline-team">
        <button
          type="button"
          className={`org-outline-row org-outline-row--team${active ? ' active' : ''}`}
          style={{ paddingInlineStart: `${0.6 + indent * 0.9}rem` }}
          onClick={() => onSelectTeam(team.id)}
        >
          <Users size={14} />
          <span className="org-outline-name">{team.name}</span>
          <span className="org-outline-count">{team.members.length + (team.lead ? 1 : 0)}</span>
        </button>
        {rows}
      </div>
    );
  };

  const ambassadors = model.ambassadors.map((p) => agentRow(p, 'ambassador', `amb/${p.agentId}`)).filter(Boolean);
  const managers = model.managers
    .map((entry) => {
      const head = agentRow(entry.profile, 'manager', `mgr/${entry.profile.agentId}`);
      const teams = entry.teams.map((t) => teamBlock(t, 1)).filter(Boolean);
      if (!head && teams.length === 0) return null;
      return (
        <div key={entry.profile.agentId} className="org-outline-manager">
          {head}
          {teams}
        </div>
      );
    })
    .filter(Boolean);
  const orgFree = model.orgFreeTeams.map((t) => teamBlock(t, 0)).filter(Boolean);
  const independents = model.independents.map((p) => agentRow(p, null, `ind/${p.agentId}`)).filter(Boolean);

  const empty =
    ambassadors.length + managers.length + orgFree.length + independents.length === 0;

  return (
    <nav className="org-outline" aria-label="Organization outline">
      {empty && <p className="org-quiet org-outline-empty">{query ? 'Nothing matches.' : 'No agents yet.'}</p>}
      {ambassadors.length > 0 && (
        <section>
          <h3 className="org-outline-label"><Radio size={12} /> Ambassador</h3>
          {ambassadors}
        </section>
      )}
      {managers.length > 0 && (
        <section>
          <h3 className="org-outline-label"><Crown size={12} /> Organization</h3>
          {managers}
        </section>
      )}
      {orgFree.length > 0 && (
        <section>
          <h3 className="org-outline-label"><Network size={12} /> Teams without a manager</h3>
          {orgFree}
        </section>
      )}
      {independents.length > 0 && (
        <section>
          <h3 className="org-outline-label"><UserRound size={12} /> Independent</h3>
          {independents}
        </section>
      )}
    </nav>
  );
}
