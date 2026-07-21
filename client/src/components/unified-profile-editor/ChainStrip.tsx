/**
 * ChainStrip — the agent's live position in the chain of command, drawn as a
 * small clickable constellation inside the Team membership card:
 *
 *     (superior) ─ ((THIS AGENT)) ─ (subordinate) (subordinate)
 *
 * Derived entirely from contexts the editor already holds (profiles +
 * workflows via lib/orgPlacement) — no new API. Clicking a neighbor hops the
 * editor to that profile, so you can walk the org without leaving the
 * dossier. Org-free agents get a quiet caption instead of a constellation.
 */

import { useAgentProfile } from '../../contexts/AgentProfileContext';
import { useAlloyWorkflow } from '../../contexts/AlloyWorkflowContext';
import { orgPlacement } from '../../lib/orgPlacement';
import type { AgentProfile } from '../../lib/api';
import { AgentAvatar } from '../common/AgentAvatar';

interface ChainStripProps {
  profile: AgentProfile;
  /** Hop the editor to another profile (ProfileNav's select). */
  onHop?: (profileId: string) => void;
}

const MAX_NAMED = 3;

function nameList(list: AgentProfile[]): string {
  const names = list.map(p => p.name);
  if (names.length <= MAX_NAMED) return names.join(', ');
  return `${names.slice(0, MAX_NAMED).join(', ')} +${names.length - MAX_NAMED}`;
}

function Node({ profile, self, onHop }: { profile: AgentProfile; self?: boolean; onHop?: (id: string) => void }) {
  const cls = `chain-strip__node${self ? ' chain-strip__node--self' : ''}`;
  if (self || !onHop) {
    return (
      <span className={cls}>
        <AgentAvatar avatar={profile.avatar} size={16} />
        {profile.name}
      </span>
    );
  }
  return (
    <button type="button" className={cls} onClick={() => onHop(profile.id)} title={`Open ${profile.name}`}>
      <AgentAvatar avatar={profile.avatar} size={16} />
      {profile.name}
    </button>
  );
}

export function ChainStrip({ profile, onHop }: ChainStripProps) {
  const { profiles } = useAgentProfile();
  const { workflows } = useAlloyWorkflow();

  const placement = orgPlacement(profile.agentId, profiles, workflows);
  const { inOrg, teams, superiors, subordinates } = placement;

  if (teams.length === 0) {
    return (
      <div className="chain-strip">
        <span className="chain-strip__caption">
          {profile.orgLevel !== 'agent'
            ? 'No team yet — give this officer a team in Agent Teams'
            : 'Org-free — engages the org through its officers'}
          {' · '}
          {profile.availableForDelegation ? 'on the flat roster' : 'off the flat roster'}
        </span>
      </div>
    );
  }

  return (
    <div className="chain-strip">
      <div className="chain-strip__row">
        {superiors.map(p => (
          <Node key={p.id} profile={p} onHop={onHop} />
        ))}
        {superiors.length > 0 && <span className="chain-strip__link" aria-hidden />}
        <Node profile={profile} self />
        {subordinates.length > 0 && <span className="chain-strip__link" aria-hidden />}
        {subordinates.map(p => (
          <Node key={p.id} profile={p} onHop={onHop} />
        ))}
      </div>
      <div className="chain-strip__meta">
        <span className="chain-strip__caption">
          {superiors.length > 0 && <>reports to {nameList(superiors)}</>}
          {superiors.length > 0 && subordinates.length > 0 && ' · '}
          {subordinates.length > 0 && <>works {nameList(subordinates)}</>}
          {superiors.length === 0 && subordinates.length === 0 && 'alone on its team'}
          {!inOrg && ' · team has no manager (org-free)'}
        </span>
        <span className="chain-strip__teams">
          {teams.map(t => (
            <span key={`${t.teamId}-${t.role}`} className="chain-strip__team">
              {t.teamName} · {t.role}
            </span>
          ))}
        </span>
      </div>
    </div>
  );
}
