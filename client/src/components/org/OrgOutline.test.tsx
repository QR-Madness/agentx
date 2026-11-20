import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from '../../lib/api';
import { buildOrgModel } from '../../lib/orgChart';
import { OrgOutline } from './OrgOutline';

function agent(agentId: string, onRoster = false): AgentProfile {
  return {
    id: agentId, name: agentId.toUpperCase(), kind: 'agent', agentId, orgLevel: 'agent',
    availableForDelegation: onRoster,
  } as AgentProfile;
}

function team(id: string, lead: string, members: string[]): AlloyWorkflow {
  const list: AlloyWorkflowMember[] = [
    { agentId: lead, role: 'supervisor' },
    ...members.map((agentId) => ({ agentId, role: 'specialist' as const })),
  ];
  return { id, name: id.toUpperCase(), supervisorAgentId: lead, members: list } as AlloyWorkflow;
}

describe('OrgOutline', () => {
  it('marks agents on the team roster, for sight and for screen readers', () => {
    const amb = { ...agent('amb', true), kind: 'ambassador' } as AgentProfile;
    const model = buildOrgModel(
      [{ ...agent('l1'), orgLevel: 'lead' } as AgentProfile, agent('a1', true), agent('a2'), agent('free', true), amb],
      [team('t1', 'l1', ['a1', 'a2'])],
    );
    render(
      <OrgOutline model={model} selection={{ kind: 'none' }} query="" onSelectAgent={vi.fn()} onSelectTeam={vi.fn()} />,
    );
    expect(screen.getByRole('button', { name: /A1.*on the team roster/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /FREE.*on the team roster/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^A2$/ })).toBeTruthy(); // off the roster: no badge
    // An ambassador is never a delegation target, whatever its flag says.
    expect(screen.queryByRole('button', { name: /AMB.*on the team roster/ })).toBeNull();
  });
});
