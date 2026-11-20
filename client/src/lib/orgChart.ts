/**
 * Org chart — the whole organization as one derived model, plus a tiered
 * layout for the Organization canvas.
 *
 * Like `orgPlacement.ts` (one agent's place) this is pure derivation over the
 * two stores the client already holds: profiles and workflows. Nothing here is
 * stored; the chart is recomputed from them (INV-9). Structure comes from the
 * workflows, never from a profile's declared `orgLevel` — where the two
 * disagree, that is reported as an issue instead of drawn as an edge.
 *
 * `buildOrgModel` answers "what is the org?"; `layoutOrgChart` answers "where
 * does each box go?". The layout is deterministic for a given model, so the
 * canvas does not jump between renders.
 */

import type { AgentProfile, AlloyWorkflow, LayoutPositions } from './api/types';

export type StructuralTier = 'manager' | 'lead' | 'member';

export interface UnresolvedRef {
  agentId: string;
  role: StructuralTier;
  /** `missing` = no profile has this id; `ambassador` = the profile is not an agent. */
  reason: 'missing' | 'ambassador';
}

export interface OrgTeam {
  id: string;
  name: string;
  workflow: AlloyWorkflow;
  manager: AgentProfile | null;
  lead: AgentProfile | null;
  /** Resolved members, in the team's own order, without the lead. */
  members: AgentProfile[];
  unresolved: UnresolvedRef[];
}

export interface OrgManagerEntry {
  profile: AgentProfile;
  teams: OrgTeam[];
}

export type OrgIssueKind =
  | 'unresolved-agent'
  | 'team-empty'
  | 'role-not-applied'
  | 'role-without-place';

export interface OrgIssue {
  kind: OrgIssueKind;
  message: string;
  teamId?: string;
  agentId?: string;
}

export interface OrgModel {
  ambassadors: AgentProfile[];
  /** Managers in the order their first team appears. */
  managers: OrgManagerEntry[];
  /** Teams with no resolvable manager, in workflow order. */
  orgFreeTeams: OrgTeam[];
  /** Every team, in workflow order. */
  teams: OrgTeam[];
  /** Agents on no team and managing none. */
  independents: AgentProfile[];
  /** Highest structural tier per agent id (manager > lead > member). */
  tierByAgentId: Map<string, StructuralTier>;
  issues: OrgIssue[];
}

const TIER_RANK: Record<StructuralTier, number> = { member: 1, lead: 2, manager: 3 };

const ROLE_LABEL: Record<NonNullable<AgentProfile['orgLevel']>, string> = {
  executive: 'Executive',
  manager: 'Manager',
  lead: 'Lead',
  agent: 'Agent',
};

function roleLabel(profile: AgentProfile): string {
  return ROLE_LABEL[profile.orgLevel ?? 'agent'];
}

export function buildOrgModel(profiles: AgentProfile[], workflows: AlloyWorkflow[]): OrgModel {
  const byId = new Map<string, AgentProfile>();
  for (const p of profiles) byId.set(p.agentId, p);

  const tierByAgentId = new Map<string, StructuralTier>();
  const raise = (agentId: string, tier: StructuralTier) => {
    const current = tierByAgentId.get(agentId);
    if (!current || TIER_RANK[tier] > TIER_RANK[current]) tierByAgentId.set(agentId, tier);
  };

  const issues: OrgIssue[] = [];

  /** Resolve an id to an agent-kind profile, recording why when it doesn't. */
  const resolve = (
    agentId: string,
    role: StructuralTier,
    unresolved: UnresolvedRef[],
  ): AgentProfile | null => {
    const profile = byId.get(agentId);
    if (profile && profile.kind === 'agent') return profile;
    unresolved.push({ agentId, role, reason: profile ? 'ambassador' : 'missing' });
    return null;
  };

  const teams: OrgTeam[] = workflows.map((wf) => {
    const unresolved: UnresolvedRef[] = [];
    const manager = wf.managerAgentId ? resolve(wf.managerAgentId, 'manager', unresolved) : null;
    const lead = wf.supervisorAgentId ? resolve(wf.supervisorAgentId, 'lead', unresolved) : null;

    const members: AgentProfile[] = [];
    const seen = new Set<string>([wf.supervisorAgentId]);
    for (const m of wf.members) {
      if (m.role !== 'specialist' || seen.has(m.agentId)) continue;
      seen.add(m.agentId);
      const profile = resolve(m.agentId, 'member', unresolved);
      if (profile) members.push(profile);
    }

    if (manager) raise(manager.agentId, 'manager');
    if (lead) raise(lead.agentId, 'lead');
    for (const m of members) raise(m.agentId, 'member');

    return { id: wf.id, name: wf.name, workflow: wf, manager, lead, members, unresolved };
  });

  const managers: OrgManagerEntry[] = [];
  const managerIndex = new Map<string, OrgManagerEntry>();
  const orgFreeTeams: OrgTeam[] = [];
  for (const team of teams) {
    if (!team.manager) {
      orgFreeTeams.push(team);
      continue;
    }
    let entry = managerIndex.get(team.manager.agentId);
    if (!entry) {
      entry = { profile: team.manager, teams: [] };
      managerIndex.set(team.manager.agentId, entry);
      managers.push(entry);
    }
    entry.teams.push(team);
  }

  const independents = profiles.filter((p) => p.kind === 'agent' && !tierByAgentId.has(p.agentId));
  const ambassadors = profiles.filter((p) => p.kind === 'ambassador');

  for (const team of teams) {
    for (const ref of team.unresolved) {
      const named = byId.get(ref.agentId)?.name ?? `"${ref.agentId}"`;
      issues.push({
        kind: 'unresolved-agent',
        teamId: team.id,
        agentId: ref.agentId,
        message:
          ref.reason === 'ambassador'
            ? `${team.name} names ${named} as its ${ref.role}, but an ambassador cannot be on a team.`
            : `${team.name} names ${named} as its ${ref.role}, but no agent has that id.`,
      });
    }
    if (team.members.length === 0) {
      issues.push({
        kind: 'team-empty',
        teamId: team.id,
        message: `${team.name} has no members, so its lead has no one to delegate to.`,
      });
    }
    // Mirrors the server's soft checks (alloy/manager.py), which only log, and
    // only for teams with a manager: a team outside the organization needs no
    // roles. The Manager role is what applies the report-only tool limits and
    // the manager charter.
    if (team.manager && team.manager.orgLevel !== 'manager') {
      issues.push({
        kind: 'role-not-applied',
        teamId: team.id,
        agentId: team.manager.agentId,
        message: `${team.manager.name} manages ${team.name}, but its role is ${roleLabel(team.manager)}. Set the role to Manager so the report-only tool limits and the manager charter apply.`,
      });
    }
    if (team.manager && team.lead && team.lead.orgLevel !== 'lead') {
      issues.push({
        kind: 'role-not-applied',
        teamId: team.id,
        agentId: team.lead.agentId,
        message: `${team.lead.name} leads ${team.name}, but its role is ${roleLabel(team.lead)}.`,
      });
    }
  }

  for (const p of profiles) {
    if (p.kind !== 'agent') continue;
    const tier = tierByAgentId.get(p.agentId);
    if (p.orgLevel === 'manager' && tier !== 'manager') {
      issues.push({
        kind: 'role-without-place',
        agentId: p.agentId,
        message: `${p.name} has the Manager role but owns no team.`,
      });
    } else if (p.orgLevel === 'lead' && tier !== 'lead' && tier !== 'manager') {
      issues.push({
        kind: 'role-without-place',
        agentId: p.agentId,
        message: `${p.name} has the Lead role but leads no team.`,
      });
    }
  }

  return { ambassadors, managers, orgFreeTeams, teams, independents, tierByAgentId, issues };
}

/**
 * Where an agent already sits, as short phrases ("Leads Marketing", "On
 * Research") — shown beside candidates in the team editor so adding someone to
 * a second team is a choice, not an accident. `excludeTeamId` drops the team
 * being edited.
 */
export function affiliations(agentId: string, model: OrgModel, excludeTeamId?: string): string[] {
  const teams = model.teams.filter((t) => t.id !== excludeTeamId);
  const managed = teams.filter((t) => t.manager?.agentId === agentId);
  const led = teams.filter((t) => t.lead?.agentId === agentId);
  const on = teams.filter((t) => t.members.some((m) => m.agentId === agentId));
  const out: string[] = [];
  if (managed.length === 1) out.push(`Manages ${managed[0].name}`);
  else if (managed.length > 1) out.push(`Manages ${managed.length} teams`);
  if (led.length > 0) out.push(`Leads ${led.map((t) => t.name).join(', ')}`);
  if (on.length > 0) out.push(`On ${on.map((t) => t.name).join(', ')}`);
  return out;
}

// ─── Layout ────────────────────────────────────────────────────────────────

export type ChartNodeKind =
  | 'ambassador'
  | 'manager'
  | 'lead'
  | 'member'
  | 'independent'
  | 'team'
  | 'lane'
  | 'band';

export interface ChartNode {
  id: string;
  kind: ChartNodeKind;
  /** Relative to the parent when `parentId` is set, else to the canvas. */
  x: number;
  y: number;
  width: number;
  height: number;
  parentId?: string;
  agentId?: string;
  teamId?: string;
  label?: string;
}

export interface ChartEdge {
  id: string;
  source: string;
  target: string;
  /** `directs` = manager → lead; `works` = lead → member. */
  kind: 'directs' | 'works';
}

export interface OrgChartLayout {
  nodes: ChartNode[];
  edges: ChartEdge[];
  width: number;
  height: number;
}

/** Box metrics, shared with the canvas so cards render at the size laid out. */
export const CHART = {
  agentW: 216,
  agentH: 58,
  teamW: 256,
  teamPad: 12,
  teamHeadH: 40,
  memberH: 46,
  memberGap: 8,
  memberIndent: 22,
  colGap: 28,
  clusterGap: 56,
  rowGap: 56,
  bandH: 30,
  laneGap: 10,
  laneRowGap: 40,
} as const;

function groupHeight(leadCount: number, memberCount: number): number {
  const rows = leadCount * (CHART.agentH + CHART.memberGap) + memberCount * (CHART.memberH + CHART.memberGap);
  // Header, the rows, and bottom padding (the last row's gap doubles as part of it).
  return CHART.teamHeadH + Math.max(rows, CHART.memberH) + CHART.teamPad;
}

/**
 * The stored-layout key of a box the user may drag, or null for one that stays where
 * the derived layout puts it. Only top-level boxes move — ambassadors, managers and
 * whole teams; leads, members and lane cards sit inside a parent. Keys are ids, never
 * names, so a rename keeps its position (INV-3).
 */
export function layoutKeyOf(node: ChartNode): string | null {
  if (node.parentId) return null;
  if (node.kind === 'team' && node.teamId) return `team:${node.teamId}`;
  if ((node.kind === 'manager' || node.kind === 'ambassador') && node.agentId) {
    return `agent:${node.agentId}`;
  }
  return null;
}

/** Keep only the positions of boxes still on the chart (a deleted team, a demoted manager). */
export function pruneLayoutPositions(
  positions: LayoutPositions,
  layout: OrgChartLayout,
): LayoutPositions {
  const live = new Set<string>();
  for (const node of layout.nodes) {
    const key = layoutKeyOf(node);
    if (key) live.add(key);
  }
  return Object.fromEntries(Object.entries(positions).filter(([key]) => live.has(key)));
}

export const chartNodeId = {
  ambassador: (agentId: string) => `amb:${agentId}`,
  manager: (agentId: string) => `mgr:${agentId}`,
  team: (teamId: string) => `team:${teamId}`,
  lead: (teamId: string, agentId: string) => `team:${teamId}/lead:${agentId}`,
  member: (teamId: string, agentId: string) => `team:${teamId}/member:${agentId}`,
  independent: (agentId: string) => `ind:${agentId}`,
  lane: 'lane:independent',
  band: 'band:executive',
} as const;

/**
 * Tiered layout, top to bottom: the ambassador apex, the reserved executive
 * band, managers, one column per team (lead first, members indented under it),
 * and last a lane of independent agents running the width of the chart. Each
 * manager is centred over its own teams; teams without a manager follow to the
 * right. The lane sits below rather than beside the teams so the chart stays
 * close to the shape of the space it is drawn in.
 */
export function layoutOrgChart(model: OrgModel): OrgChartLayout {
  const nodes: ChartNode[] = [];
  const edges: ChartEdge[] = [];

  const yApex = 0;
  const yBand = yApex + CHART.agentH + 26;
  const yManagers = yBand + CHART.bandH + 26;
  const yTeams = yManagers + CHART.agentH + CHART.rowGap;

  let cursorX = 0;
  let bottom = yTeams;

  const placeTeam = (team: OrgTeam, x: number) => {
    const teamNodeId = chartNodeId.team(team.id);
    const height = groupHeight(team.lead ? 1 : 0, team.members.length);
    nodes.push({
      id: teamNodeId, kind: 'team', x, y: yTeams, width: CHART.teamW, height,
      teamId: team.id, label: team.name,
    });
    bottom = Math.max(bottom, yTeams + height);

    let y = CHART.teamHeadH;
    let leadNodeId: string | null = null;
    if (team.lead) {
      leadNodeId = chartNodeId.lead(team.id, team.lead.agentId);
      nodes.push({
        id: leadNodeId, kind: 'lead', parentId: teamNodeId,
        x: CHART.teamPad, y, width: CHART.teamW - 2 * CHART.teamPad, height: CHART.agentH,
        agentId: team.lead.agentId, teamId: team.id,
      });
      y += CHART.agentH + CHART.memberGap;
    }
    for (const member of team.members) {
      const memberNodeId = chartNodeId.member(team.id, member.agentId);
      nodes.push({
        id: memberNodeId, kind: 'member', parentId: teamNodeId,
        x: CHART.teamPad + CHART.memberIndent, y,
        width: CHART.teamW - 2 * CHART.teamPad - CHART.memberIndent, height: CHART.memberH,
        agentId: member.agentId, teamId: team.id,
      });
      if (leadNodeId) {
        edges.push({ id: `works:${team.id}:${member.agentId}`, source: leadNodeId, target: memberNodeId, kind: 'works' });
      }
      y += CHART.memberH + CHART.memberGap;
    }
    return leadNodeId;
  };

  for (const entry of model.managers) {
    if (cursorX > 0) cursorX += CHART.clusterGap;
    const startX = cursorX;
    const managerNodeId = chartNodeId.manager(entry.profile.agentId);
    for (const team of entry.teams) {
      const leadNodeId = placeTeam(team, cursorX);
      if (leadNodeId) {
        edges.push({ id: `directs:${entry.profile.agentId}:${team.id}`, source: managerNodeId, target: leadNodeId, kind: 'directs' });
      }
      cursorX += CHART.teamW + CHART.colGap;
    }
    cursorX -= CHART.colGap;
    const span = cursorX - startX;
    nodes.push({
      id: managerNodeId, kind: 'manager',
      x: startX + (span - CHART.agentW) / 2, y: yManagers, width: CHART.agentW, height: CHART.agentH,
      agentId: entry.profile.agentId,
    });
  }

  model.orgFreeTeams.forEach((team, i) => {
    if (cursorX > 0) cursorX += i === 0 ? CHART.clusterGap : CHART.colGap;
    placeTeam(team, cursorX);
    cursorX += CHART.teamW;
  });

  const teamsWidth = cursorX;
  const hasTeams = model.teams.length > 0;

  // The lane is a grid as wide as the teams above it, and never narrower than
  // two cards (or the cards it holds, when there are fewer).
  const cardW = CHART.teamW - 2 * CHART.teamPad;
  const laneWidthFor = (cols: number) => cols * cardW + (cols - 1) * CHART.laneGap + 2 * CHART.teamPad;
  const independentCount = model.independents.length;
  const minLaneWidth = independentCount > 0 ? laneWidthFor(Math.min(independentCount, hasTeams ? 2 : 3)) : 0;
  const laneWidth = independentCount > 0 ? Math.max(teamsWidth, minLaneWidth) : 0;

  // Never narrower than the apex row, so an empty org still lays out sensibly.
  const apexSpan =
    model.ambassadors.length * CHART.agentW + Math.max(0, model.ambassadors.length - 1) * CHART.colGap;
  const width = Math.max(teamsWidth, laneWidth, apexSpan, CHART.agentW);

  // Centre the manager and team tier when something below or above it is wider.
  const teamsShift = (width - teamsWidth) / 2;
  if (teamsShift > 0) {
    for (const n of nodes) if (!n.parentId) n.x += teamsShift;
  }

  if (independentCount > 0) {
    const cols = Math.max(1, Math.floor((laneWidth - 2 * CHART.teamPad + CHART.laneGap) / (cardW + CHART.laneGap)));
    const rows = Math.ceil(independentCount / cols);
    const laneY = hasTeams ? bottom + CHART.laneRowGap : yTeams;
    const height = CHART.teamHeadH + rows * (CHART.memberH + CHART.memberGap) + CHART.teamPad;
    nodes.push({
      id: chartNodeId.lane, kind: 'lane', x: (width - laneWidth) / 2, y: laneY, width: laneWidth, height,
      label: 'Independent',
    });
    model.independents.forEach((p, i) => {
      nodes.push({
        id: chartNodeId.independent(p.agentId), kind: 'independent', parentId: chartNodeId.lane,
        x: CHART.teamPad + (i % cols) * (cardW + CHART.laneGap),
        y: CHART.teamHeadH + Math.floor(i / cols) * (CHART.memberH + CHART.memberGap),
        width: cardW, height: CHART.memberH,
        agentId: p.agentId,
      });
    });
    bottom = laneY + height;
  }

  let apexX = (width - apexSpan) / 2;
  for (const amb of model.ambassadors) {
    nodes.push({
      id: chartNodeId.ambassador(amb.agentId), kind: 'ambassador',
      x: apexX, y: yApex, width: CHART.agentW, height: CHART.agentH, agentId: amb.agentId,
    });
    apexX += CHART.agentW + CHART.colGap;
  }

  nodes.push({
    id: chartNodeId.band, kind: 'band', x: 0, y: yBand, width, height: CHART.bandH,
    label: 'Executive tier — reserved',
  });

  return { nodes, edges, width, height: bottom };
}

// ─── Edges, explained ─────────────────────────────────────────────────────

/** One line of the chart, resolved for the dock: who hands work to whom, on which team. */
export interface ResolvedChartEdge {
  kind: ChartEdge['kind'];
  /** Who hands the work down: a manager (`directs`) or a lead (`works`). */
  from: AgentProfile;
  /** Who receives it: a lead (`directs`) or a member (`works`). */
  to: AgentProfile;
  teamId: string;
  teamName: string;
}

/** Resolve a chart edge id to its two agents and team, or null once it is gone. */
export function resolveChartEdge(
  layout: OrgChartLayout,
  model: OrgModel,
  profilesByAgentId: Map<string, AgentProfile>,
  edgeId: string,
): ResolvedChartEdge | null {
  const edge = layout.edges.find((e) => e.id === edgeId);
  if (!edge) return null;
  const nodes = new Map(layout.nodes.map((n) => [n.id, n]));
  const source = nodes.get(edge.source);
  const target = nodes.get(edge.target);
  const from = source?.agentId ? profilesByAgentId.get(source.agentId) : undefined;
  const to = target?.agentId ? profilesByAgentId.get(target.agentId) : undefined;
  const team = model.teams.find((t) => t.id === (target?.teamId ?? source?.teamId));
  if (!from || !to || !team) return null;
  return { kind: edge.kind, from, to, teamId: team.id, teamName: team.name };
}
