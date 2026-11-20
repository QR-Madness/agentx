/**
 * OrgOverview — what the dock shows when nothing is selected: the size of the
 * organization and anything in its structure that needs attention.
 */

import { AlertTriangle, Plus } from 'lucide-react';
import type { OrgIssue, OrgModel } from '../../lib/orgChart';
import { Button } from '../ui/Button';

interface OrgOverviewProps {
  model: OrgModel;
  /** The legend explains the chart, so it is left out where no chart is shown. */
  showLegend?: boolean;
  onOpenIssue: (issue: OrgIssue) => void;
  onNewTeam: () => void;
}

export function OrgOverview({ model, showLegend = true, onOpenIssue, onNewTeam }: OrgOverviewProps) {
  const inTeams = model.tierByAgentId.size;
  const stats: Array<[string, number]> = [
    ['Managers', model.managers.length],
    ['Teams', model.teams.length],
    ['Agents placed', inTeams],
    ['Independent', model.independents.length],
  ];

  return (
    <div className="org-panel">
      <h2 className="org-panel-title">Organization</h2>
      <p className="org-quiet">
        A manager delegates to the leads of the teams it owns, and each lead works its own
        members. Select a team to edit it, or an agent to see where it sits.
      </p>

      <div className="org-stats">
        {stats.map(([label, value]) => (
          <div key={label} className="org-stat">
            <span className="org-stat-value">{value}</span>
            <span className="org-stat-label">{label}</span>
          </div>
        ))}
      </div>

      {model.teams.length === 0 && (
        <div className="org-empty">
          <p>
            There are no teams yet. A team is one lead and the members it can delegate to. Give a
            team a manager and it becomes part of the organization.
          </p>
          <Button variant="primary" size="sm" onClick={onNewTeam}>
            <Plus size={14} />
            Create a team
          </Button>
        </div>
      )}

      <section className="org-section">
        <h3 className="org-eyebrow">Needs attention ({model.issues.length})</h3>
        {model.issues.length === 0 ? (
          <p className="org-quiet">Nothing. Every team resolves, and roles match the structure.</p>
        ) : (
          <ul className="org-issues">
            {model.issues.map((issue) => (
              <li key={issue.message}>
                <button type="button" className="org-issue" onClick={() => onOpenIssue(issue)}>
                  <AlertTriangle size={13} />
                  <span>{issue.message}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {showLegend && (
        <section className="org-section">
          <h3 className="org-eyebrow">Reading the chart</h3>
          <ul className="org-legend">
            <li>
              <span className="org-legend-line" aria-hidden /> A line runs from whoever delegates to
              whoever receives the work.
            </li>
            <li>
              <span className="org-legend-escalate" aria-hidden /> A chevron at a lead marks that its
              members can escalate back to it.
            </li>
            <li>
              <span className="org-legend-apex" aria-hidden /> The ambassador sits above the chain and
              outside it: it can dispatch to anyone.
            </li>
            <li>
              <span className="org-legend-lane" aria-hidden /> Independent agents are on no team and keep
              the flat roster.
            </li>
          </ul>
        </section>
      )}
    </div>
  );
}
