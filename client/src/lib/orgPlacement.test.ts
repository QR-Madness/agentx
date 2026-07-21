import { describe, expect, it } from 'vitest';
import { groupRoster, orgPlacement } from './orgPlacement';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from './api/types';

// Abstract org shaped like the real thing: one manager, two managed teams,
// plus one org-free team (no manager) to pin the manager-anchored rule.
function agent(agentId: string, kind: 'agent' | 'ambassador' = 'agent'): AgentProfile {
  return { id: agentId, name: agentId.toUpperCase(), kind, agentId } as AgentProfile;
}

function team(
  id: string,
  supervisorAgentId: string,
  memberAgentIds: string[],
  managerAgentId?: string,
): AlloyWorkflow {
  const members: AlloyWorkflowMember[] = [
    { agentId: supervisorAgentId, role: 'supervisor' },
    ...memberAgentIds.map((agentId) => ({ agentId, role: 'specialist' as const })),
  ];
  return { id, name: id.toUpperCase(), supervisorAgentId, managerAgentId, members } as AlloyWorkflow;
}

const profiles: AgentProfile[] = [
  agent('m'),
  agent('l1'),
  agent('l2'),
  agent('l3'),
  agent('a1'),
  agent('a2'),
  agent('a3'),
  agent('a4'),
  agent('free'),
  agent('amb', 'ambassador'),
];

const workflows: AlloyWorkflow[] = [
  // 'amb' as a member + a dangling id pin the neighbor filtering.
  team('t1', 'l1', ['a1', 'a2', 'amb', 'ghost'], 'm'),
  team('t2', 'l2', ['a3'], 'm'),
  team('t3', 'l3', ['a4']), // org-free team — no manager anchor
];

const names = (list: AgentProfile[]) => list.map((p) => p.agentId);

describe('orgPlacement', () => {
  it('derives the manager: subordinate leads, no superiors', () => {
    const p = orgPlacement('m', profiles, workflows);
    expect(p.inOrg).toBe(true);
    expect(names(p.subordinates)).toEqual(['l1', 'l2']);
    expect(p.superiors).toEqual([]);
    expect(p.teams).toEqual([
      { teamId: 't1', teamName: 'T1', role: 'manager' },
      { teamId: 't2', teamName: 'T2', role: 'manager' },
    ]);
  });

  it('derives a lead: manager above, own members below', () => {
    const p = orgPlacement('l1', profiles, workflows);
    expect(p.inOrg).toBe(true);
    expect(names(p.superiors)).toEqual(['m']);
    expect(names(p.subordinates)).toEqual(['a1', 'a2']); // amb + ghost dropped
    expect(p.teams).toEqual([{ teamId: 't1', teamName: 'T1', role: 'lead' }]);
  });

  it('derives a member: its lead above, nothing below', () => {
    const p = orgPlacement('a3', profiles, workflows);
    expect(p.inOrg).toBe(true);
    expect(names(p.superiors)).toEqual(['l2']);
    expect(p.subordinates).toEqual([]);
  });

  it('inOrg is manager-anchored: an unowned team confers no membership', () => {
    const lead = orgPlacement('l3', profiles, workflows);
    expect(lead.inOrg).toBe(false);
    expect(lead.superiors).toEqual([]); // no owning manager
    expect(names(lead.subordinates)).toEqual(['a4']); // team structure still shown
    expect(orgPlacement('a4', profiles, workflows).inOrg).toBe(false);
  });

  it('an unplaced agent is org-free with no neighbors', () => {
    const p = orgPlacement('free', profiles, workflows);
    expect(p).toEqual({ inOrg: false, teams: [], superiors: [], subordinates: [] });
  });
});

describe('groupRoster', () => {
  it('groups manager → teams (lead first) → org-free → ambassador, each profile once', () => {
    const groups = groupRoster(profiles, workflows);
    expect(groups.map((g) => [g.kind, g.label, names(g.profiles)])).toEqual([
      ['manager', 'Manager', ['m']],
      ['team', 'T1', ['l1', 'a1', 'a2']],
      ['team', 'T2', ['l2', 'a3']],
      ['team', 'T3', ['l3', 'a4']],
      ['org-free', 'Org-free', ['free']],
      ['ambassador', 'Ambassador', ['amb']],
    ]);
  });

  it('a manager who also leads a team stays in the manager group', () => {
    const doubleHat = [team('t1', 'm', ['a1'], 'm')];
    const groups = groupRoster(profiles, doubleHat);
    expect(groups[0]).toMatchObject({ kind: 'manager', profiles: [expect.objectContaining({ agentId: 'm' })] });
    expect(names(groups.find((g) => g.teamId === 't1')!.profiles)).toEqual(['a1']);
  });

  it('empty workflows yields org-free + ambassador only', () => {
    const groups = groupRoster(profiles, []);
    expect(groups.map((g) => g.kind)).toEqual(['org-free', 'ambassador']);
    expect(groups[0].profiles).toHaveLength(9);
  });
});
