/** What the Organization dock is showing. */
export type OrgSelection =
  | { kind: 'none' }
  | { kind: 'agent'; agentId: string }
  | { kind: 'team'; teamId: string }
  | { kind: 'new-team'; managerAgentId?: string };
