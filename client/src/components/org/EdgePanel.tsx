/**
 * EdgePanel — one line of the organization chart, explained (read-only): who
 * hands work to whom on which team, who can escalate back, and links to both
 * agents and the team. Lines are structure derived from the teams; they are
 * changed by editing a team, not here.
 */

import { ArrowUpRight } from 'lucide-react';
import type { AgentProfile } from '../../lib/api';
import type { ResolvedChartEdge } from '../../lib/orgChart';
import { AgentAvatar } from '../common/AgentAvatar';

interface EdgePanelProps {
  edge: ResolvedChartEdge;
  onOpenAgent: (agentId: string) => void;
  onOpenTeam: (teamId: string) => void;
}

export function EdgePanel({ edge, onOpenAgent, onOpenTeam }: EdgePanelProps) {
  const { from, to, teamName } = edge;
  const directs = edge.kind === 'directs';
  return (
    <div className="org-panel">
      <h3 className="org-eyebrow">{directs ? 'Chain of command' : 'Team line'}</h3>
      <h2 className="org-panel-title">
        {directs ? `${from.name} directs ${to.name}` : `${from.name} works with ${to.name}`}
      </h2>
      <p className="org-quiet">
        {directs
          ? `${to.name} leads ${teamName}, which ${from.name} manages. Managers hand work only to the leads of their teams; ${to.name} reports to ${from.name}.`
          : `${from.name} leads ${teamName} and hands ${to.name} tasks. ${to.name} can escalate back to ${from.name}: the chevron at the lead's end of the line.`}
      </p>

      <section className="org-section">
        <div className="org-chipline">
          <span className="org-chipline-label">{directs ? 'Manager' : 'Lead'}</span>
          <AgentChip agent={from} onOpenAgent={onOpenAgent} />
        </div>
        <div className="org-chipline">
          <span className="org-chipline-label">{directs ? 'Lead' : 'Member'}</span>
          <AgentChip agent={to} onOpenAgent={onOpenAgent} />
        </div>
        <button type="button" className="org-placement-row" onClick={() => onOpenTeam(edge.teamId)}>
          <span className="org-placement-team">{teamName}</span>
          <ArrowUpRight size={13} />
        </button>
      </section>

      <p className="org-quiet">Lines follow the teams: change one by editing the team.</p>
    </div>
  );
}

function AgentChip({ agent, onOpenAgent }: { agent: AgentProfile; onOpenAgent: (agentId: string) => void }) {
  return (
    <button type="button" className="org-chip" onClick={() => onOpenAgent(agent.agentId)}>
      <AgentAvatar avatar={agent.avatar} size={14} />
      {agent.name}
    </button>
  );
}
