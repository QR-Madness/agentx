/** What the Organization dock is showing. `overview` is the narrow-screen way
 *  to reach what the dock shows by default on wide ones. */
export type OrgSelection =
  | { kind: 'none' }
  | { kind: 'overview' }
  | { kind: 'agent'; agentId: string }
  | { kind: 'team'; teamId: string }
  /** A line of the chart (a ChartEdge id), explained read-only in the dock. */
  | { kind: 'edge'; edgeId: string }
  | { kind: 'new-team'; managerAgentId?: string };
