/**
 * Org placement — client-side mirror of the backend chain-of-command
 * derivation (`api/agentx_ai/alloy/org_chart.py`). Pure functions over the
 * two stores the client already holds (profiles + workflows); no new API.
 *
 * Mirrored rules that must not drift (pinned by orgPlacement.test.ts):
 * - `inOrg` is manager-anchored: an agent is in the org iff it manages a
 *   team, or any team it leads / belongs to has a `managerAgentId`.
 * - Structure comes from workflows, not from `orgLevel` — the declared tier
 *   is presentation (chips), never an edge.
 * - Non-agent kinds (ambassadors) and dangling ids never appear as chain
 *   neighbors.
 *
 * Presentation note: unlike backend `chain_targets` (delegation legality,
 * which has no lead→manager upward edge in v1), this module describes
 * *placement* — a lead's superior IS the manager of a team it leads.
 */

import type { AgentProfile, AlloyWorkflow } from './api/types';

export type PlacementRole = 'manager' | 'lead' | 'member';

export interface PlacementTeam {
  teamId: string;
  teamName: string;
  role: PlacementRole;
}

export interface OrgPlacement {
  inOrg: boolean;
  /** Teams this agent touches, in workflow-list order (manager roles first). */
  teams: PlacementTeam[];
  /** Who this agent reports to: member → its leads, lead → owning managers. */
  superiors: AgentProfile[];
  /** Who this agent directs: manager → leads of owned teams, lead → its members. */
  subordinates: AgentProfile[];
}

export type RosterGroupKind = 'manager' | 'team' | 'org-free' | 'ambassador';

export interface RosterGroup {
  kind: RosterGroupKind;
  /** Set for `team` groups. */
  teamId?: string;
  label: string;
  profiles: AgentProfile[];
}

function byAgentId(profiles: AgentProfile[]): Map<string, AgentProfile> {
  const map = new Map<string, AgentProfile>();
  for (const p of profiles) map.set(p.agentId, p);
  return map;
}

/** Resolve chain-neighbor ids to real agent profiles (ambassadors and
 *  dangling ids drop out), deduped, order-preserving. */
function resolveAgents(ids: string[], profiles: Map<string, AgentProfile>): AgentProfile[] {
  const out: AgentProfile[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const p = profiles.get(id);
    if (p && p.kind === 'agent') out.push(p);
  }
  return out;
}

function memberIds(wf: AlloyWorkflow): string[] {
  return wf.members
    .filter((m) => m.role === 'specialist')
    .map((m) => m.agentId);
}

/** Derive one agent's placement in the org chart. */
export function orgPlacement(
  agentId: string,
  profiles: AgentProfile[],
  workflows: AlloyWorkflow[],
): OrgPlacement {
  const profileMap = byAgentId(profiles);
  const managed = workflows.filter((wf) => wf.managerAgentId === agentId);
  const led = workflows.filter((wf) => wf.supervisorAgentId === agentId);
  const memberOf = workflows.filter((wf) => memberIds(wf).includes(agentId));

  const teams: PlacementTeam[] = [
    ...managed.map((wf) => ({ teamId: wf.id, teamName: wf.name, role: 'manager' as const })),
    ...led.map((wf) => ({ teamId: wf.id, teamName: wf.name, role: 'lead' as const })),
    ...memberOf.map((wf) => ({ teamId: wf.id, teamName: wf.name, role: 'member' as const })),
  ];

  const inOrg =
    managed.length > 0 || [...led, ...memberOf].some((wf) => Boolean(wf.managerAgentId));

  const superiors = resolveAgents(
    [
      ...memberOf.map((wf) => wf.supervisorAgentId),
      ...led.map((wf) => wf.managerAgentId ?? ''),
    ].filter((id) => id !== agentId),
    profileMap,
  );

  const subordinates = resolveAgents(
    [
      ...managed.map((wf) => wf.supervisorAgentId),
      ...led.flatMap((wf) => memberIds(wf)),
    ].filter((id) => id !== agentId),
    profileMap,
  );

  return { inOrg, teams, superiors, subordinates };
}

/**
 * Group the roster for the profile nav: managers first, then one group per
 * team (lead first, then members in member order), then org-free agents,
 * then ambassadors. Every profile appears exactly once — first placement
 * wins (a manager who also leads a team stays in the manager group).
 * Within a group, the incoming profile order (the user's persisted order)
 * is preserved except for the lead-first rule.
 */
export function groupRoster(
  profiles: AgentProfile[],
  workflows: AlloyWorkflow[],
): RosterGroup[] {
  const placed = new Set<string>();
  const groups: RosterGroup[] = [];
  const take = (p: AgentProfile | undefined): AgentProfile | null => {
    if (!p || placed.has(p.agentId)) return null;
    placed.add(p.agentId);
    return p;
  };

  const agents = profiles.filter((p) => p.kind === 'agent');
  const profileMap = byAgentId(agents);

  const managers = agents.filter((p) =>
    workflows.some((wf) => wf.managerAgentId === p.agentId),
  );
  const managerGroup = managers.map(take).filter((p): p is AgentProfile => p !== null);
  if (managerGroup.length > 0) {
    groups.push({ kind: 'manager', label: 'Manager', profiles: managerGroup });
  }

  for (const wf of workflows) {
    const roster: AgentProfile[] = [];
    const lead = take(profileMap.get(wf.supervisorAgentId));
    if (lead) roster.push(lead);
    for (const id of memberIds(wf)) {
      const member = take(profileMap.get(id));
      if (member) roster.push(member);
    }
    if (roster.length > 0) {
      groups.push({ kind: 'team', teamId: wf.id, label: wf.name, profiles: roster });
    }
  }

  const orgFree = agents.filter((p) => !placed.has(p.agentId));
  if (orgFree.length > 0) {
    for (const p of orgFree) placed.add(p.agentId);
    groups.push({ kind: 'org-free', label: 'Org-free', profiles: orgFree });
  }

  const ambassadors = profiles.filter((p) => p.kind === 'ambassador');
  if (ambassadors.length > 0) {
    groups.push({ kind: 'ambassador', label: 'Ambassador', profiles: ambassadors });
  }

  return groups;
}
