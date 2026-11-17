import { describe, expect, it } from 'vitest';
import { buildOrgModel, chartNodeId, layoutOrgChart, CHART, type ChartNode } from './orgChart';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from './api/types';

// The same abstract org as orgPlacement.test.ts: one manager with two teams,
// one team with no manager, one agent on no team, one ambassador.
function agent(
  agentId: string,
  orgLevel: AgentProfile['orgLevel'] = 'agent',
  kind: 'agent' | 'ambassador' = 'agent',
): AgentProfile {
  return { id: agentId, name: agentId.toUpperCase(), kind, agentId, orgLevel } as AgentProfile;
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
  agent('m', 'manager'),
  agent('l1', 'lead'),
  agent('l2', 'lead'),
  agent('l3', 'lead'),
  agent('a1'),
  agent('a2'),
  agent('a3'),
  agent('a4'),
  agent('free'),
  agent('amb', 'agent', 'ambassador'),
];

const workflows: AlloyWorkflow[] = [
  team('t1', 'l1', ['a1', 'a2'], 'm'),
  team('t2', 'l2', ['a3'], 'm'),
  team('t3', 'l3', ['a4']),
];

const ids = (list: AgentProfile[]) => list.map((p) => p.agentId);

describe('buildOrgModel', () => {
  const model = buildOrgModel(profiles, workflows);

  it('groups teams under their manager and keeps manager-less teams apart', () => {
    expect(model.managers.map((m) => [m.profile.agentId, m.teams.map((t) => t.id)])).toEqual([
      ['m', ['t1', 't2']],
    ]);
    expect(model.orgFreeTeams.map((t) => t.id)).toEqual(['t3']);
    expect(model.teams.map((t) => t.id)).toEqual(['t1', 't2', 't3']);
  });

  it('resolves each team to its lead and members, lead excluded from members', () => {
    const t1 = model.teams[0];
    expect(t1.lead?.agentId).toBe('l1');
    expect(ids(t1.members)).toEqual(['a1', 'a2']);
  });

  it('lists agents on no team as independent and ambassadors separately', () => {
    expect(ids(model.independents)).toEqual(['free']);
    expect(ids(model.ambassadors)).toEqual(['amb']);
  });

  it('records the highest structural tier per agent', () => {
    expect(model.tierByAgentId.get('m')).toBe('manager');
    expect(model.tierByAgentId.get('l1')).toBe('lead');
    expect(model.tierByAgentId.get('a1')).toBe('member');
    expect(model.tierByAgentId.has('free')).toBe(false);
  });

  it('reports nothing for a consistent org', () => {
    expect(model.issues).toEqual([]);
  });

  it('keeps the higher tier when an agent leads one team and is a member of another', () => {
    const both = buildOrgModel(profiles, [
      team('t1', 'l1', ['a1'], 'm'),
      team('t2', 'l2', ['l1'], 'm'),
    ]);
    expect(both.tierByAgentId.get('l1')).toBe('lead');
    expect(ids(both.teams[1].members)).toEqual(['l1']);
  });
});

describe('buildOrgModel issues', () => {
  it('flags ids that resolve to no agent, and ambassadors named on a team', () => {
    const model = buildOrgModel(profiles, [team('t1', 'l1', ['a1', 'ghost', 'amb'], 'm')]);
    expect(model.teams[0].unresolved).toEqual([
      { agentId: 'ghost', role: 'member', reason: 'missing' },
      { agentId: 'amb', role: 'member', reason: 'ambassador' },
    ]);
    expect(ids(model.teams[0].members)).toEqual(['a1']);
    const messages = model.issues.filter((i) => i.kind === 'unresolved-agent').map((i) => i.message);
    expect(messages).toEqual([
      'T1 names "ghost" as its member, but no agent has that id.',
      'T1 names AMB as its member, but an ambassador cannot be on a team.',
    ]);
  });

  it('treats a team whose manager id is dangling as manager-less', () => {
    const model = buildOrgModel(profiles, [team('t1', 'l1', ['a1'], 'gone')]);
    expect(model.managers).toEqual([]);
    expect(model.orgFreeTeams.map((t) => t.id)).toEqual(['t1']);
    expect(model.issues.map((i) => i.kind)).toContain('unresolved-agent');
  });

  it('flags a team with no members', () => {
    const model = buildOrgModel(profiles, [team('t1', 'l1', [], 'm')]);
    expect(model.issues).toContainEqual({
      kind: 'team-empty',
      teamId: 't1',
      message: 'T1 has no members, so its lead has no one to delegate to.',
    });
  });

  it('flags a manager or lead by structure whose role was never set', () => {
    const plain = profiles.map((p) => ({ ...p, orgLevel: 'agent' as const }));
    const model = buildOrgModel(plain, [team('t1', 'l1', ['a1'], 'm')]);
    const flagged = model.issues.filter((i) => i.kind === 'role-not-applied').map((i) => i.agentId);
    expect(flagged).toEqual(['m', 'l1']);
  });

  it('asks for no roles on a team outside the organization', () => {
    const plain = profiles.map((p) => ({ ...p, orgLevel: 'agent' as const }));
    const model = buildOrgModel(plain, [team('t1', 'l1', ['a1'])]);
    expect(model.issues.filter((i) => i.kind === 'role-not-applied')).toEqual([]);
  });

  it('flags a manager or lead role with no team to match', () => {
    const model = buildOrgModel(profiles, []);
    expect(model.issues.map((i) => [i.kind, i.agentId])).toEqual([
      ['role-without-place', 'm'],
      ['role-without-place', 'l1'],
      ['role-without-place', 'l2'],
      ['role-without-place', 'l3'],
    ]);
  });
});

describe('layoutOrgChart', () => {
  const layout = layoutOrgChart(buildOrgModel(profiles, workflows));
  const node = (id: string): ChartNode => {
    const found = layout.nodes.find((n) => n.id === id);
    if (!found) throw new Error(`no node ${id}`);
    return found;
  };
  const topLevel = layout.nodes.filter((n) => !n.parentId && n.kind !== 'band');

  it('is deterministic', () => {
    expect(layoutOrgChart(buildOrgModel(profiles, workflows))).toEqual(layout);
  });

  it('gives every node a unique id and every edge two existing ends', () => {
    const nodeIds = layout.nodes.map((n) => n.id);
    expect(new Set(nodeIds).size).toBe(nodeIds.length);
    for (const e of layout.edges) {
      expect(nodeIds).toContain(e.source);
      expect(nodeIds).toContain(e.target);
    }
  });

  it('stacks the tiers top to bottom: apex, band, managers, teams', () => {
    const apex = node(chartNodeId.ambassador('amb'));
    const band = node(chartNodeId.band);
    const manager = node(chartNodeId.manager('m'));
    const teamNode = node(chartNodeId.team('t1'));
    expect(apex.y + apex.height).toBeLessThan(band.y);
    expect(band.y + band.height).toBeLessThan(manager.y);
    expect(manager.y + manager.height).toBeLessThan(teamNode.y);
  });

  it('centres the manager over its own teams and the apex over the chart', () => {
    const t1 = node(chartNodeId.team('t1'));
    const t2 = node(chartNodeId.team('t2'));
    const manager = node(chartNodeId.manager('m'));
    expect(manager.x + manager.width / 2).toBeCloseTo((t1.x + t2.x + t2.width) / 2);
    const apex = node(chartNodeId.ambassador('amb'));
    expect(apex.x + apex.width / 2).toBeCloseTo(layout.width / 2);
  });

  it('never overlaps two top-level boxes', () => {
    for (let i = 0; i < topLevel.length; i++) {
      for (let j = i + 1; j < topLevel.length; j++) {
        const a = topLevel[i];
        const b = topLevel[j];
        const apart =
          a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y;
        expect(apart, `${a.id} overlaps ${b.id}`).toBe(true);
      }
    }
  });

  const expectChildrenContained = (nodes: ChartNode[]) => {
    const groups = nodes.filter((n) => n.kind === 'team' || n.kind === 'lane');
    for (const group of groups) {
      const children = nodes.filter((n) => n.parentId === group.id);
      children.forEach((child, i) => {
        expect(child.x).toBeGreaterThanOrEqual(CHART.teamPad);
        expect(child.x + child.width).toBeLessThanOrEqual(group.width - CHART.teamPad);
        expect(child.y).toBeGreaterThanOrEqual(CHART.teamHeadH);
        expect(child.y + child.height).toBeLessThanOrEqual(group.height - CHART.teamPad);
        for (const other of children.slice(i + 1)) {
          const apart =
            child.x + child.width <= other.x || other.x + other.width <= child.x ||
            child.y + child.height <= other.y || other.y + other.height <= child.y;
          expect(apart, `${child.id} overlaps ${other.id}`).toBe(true);
        }
      });
    }
  };

  it('keeps every child inside its group, below the header, without overlap', () => {
    expectChildrenContained(layout.nodes);
  });

  it('runs the independent lane under the teams, as wide as they are', () => {
    const lane = node(chartNodeId.lane);
    const t1 = node(chartNodeId.team('t1'));
    const t3 = node(chartNodeId.team('t3'));
    for (const t of [t1, node(chartNodeId.team('t2')), t3]) {
      expect(lane.y).toBeGreaterThan(t.y + t.height);
    }
    expect(lane.x).toBeCloseTo(t1.x);
    expect(lane.x + lane.width).toBeCloseTo(t3.x + t3.width);
    expect(lane.y + lane.height).toBe(layout.height);
  });

  it('wraps a long lane into rows and centres a narrow team tier over it', () => {
    const many = Array.from({ length: 7 }, (_, i) => agent(`free${i}`));
    const wide = layoutOrgChart(buildOrgModel([...profiles, ...many], [team('t1', 'l1', ['a1'], 'm')]));
    expectChildrenContained(wide.nodes);
    const lane = wide.nodes.find((n) => n.id === chartNodeId.lane)!;
    const only = wide.nodes.find((n) => n.id === chartNodeId.team('t1'))!;
    const rows = new Set(wide.nodes.filter((n) => n.parentId === lane.id).map((n) => n.y));
    expect(rows.size).toBeGreaterThan(1);
    expect(lane.width).toBeGreaterThan(only.width);
    expect(only.x + only.width / 2).toBeCloseTo(lane.x + lane.width / 2);
    expect(wide.width).toBe(lane.width);
  });

  it('lays out an org with no teams as the apex over the lane', () => {
    const flat = layoutOrgChart(buildOrgModel([agent('x'), agent('y'), agent('amb', 'agent', 'ambassador')], []));
    const lane = flat.nodes.find((n) => n.id === chartNodeId.lane)!;
    const apex = flat.nodes.find((n) => n.id === chartNodeId.ambassador('amb'))!;
    expect(flat.nodes.filter((n) => n.parentId === lane.id)).toHaveLength(2);
    expect(apex.x + apex.width / 2).toBeCloseTo(flat.width / 2);
    expect(lane.y).toBeGreaterThan(apex.y + apex.height);
  });

  it('draws manager → lead and lead → member, and nothing to the ambassador', () => {
    expect(layout.edges.map((e) => [e.kind, e.source, e.target])).toEqual([
      ['works', chartNodeId.lead('t1', 'l1'), chartNodeId.member('t1', 'a1')],
      ['works', chartNodeId.lead('t1', 'l1'), chartNodeId.member('t1', 'a2')],
      ['directs', chartNodeId.manager('m'), chartNodeId.lead('t1', 'l1')],
      ['works', chartNodeId.lead('t2', 'l2'), chartNodeId.member('t2', 'a3')],
      ['directs', chartNodeId.manager('m'), chartNodeId.lead('t2', 'l2')],
      ['works', chartNodeId.lead('t3', 'l3'), chartNodeId.member('t3', 'a4')],
    ]);
    const apexId = chartNodeId.ambassador('amb');
    expect(layout.edges.some((e) => e.source === apexId || e.target === apexId)).toBe(false);
  });

  it('puts independent agents in their own lane', () => {
    const free = node(chartNodeId.independent('free'));
    expect(free.parentId).toBe(chartNodeId.lane);
  });

  it('lays out an empty org as just the band', () => {
    const empty = layoutOrgChart(buildOrgModel([], []));
    expect(empty.nodes.map((n) => n.kind)).toEqual(['band']);
    expect(empty.width).toBe(CHART.agentW);
  });

  it('shows the same agent once per team it is on', () => {
    const shared = layoutOrgChart(
      buildOrgModel(profiles, [team('t1', 'l1', ['a1'], 'm'), team('t2', 'l2', ['a1'], 'm')]),
    );
    expect(shared.nodes.filter((n) => n.agentId === 'a1').map((n) => n.id)).toEqual([
      chartNodeId.member('t1', 'a1'),
      chartNodeId.member('t2', 'a1'),
    ]);
  });
});
