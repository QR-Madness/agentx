/**
 * The organization roles an agent can be given, as both the profile editor and the
 * Organization dock offer them. `executive` is reserved (the empty executive band)
 * and is shown, and edited, as Agent.
 */

import type { AgentProfile } from './api';

export type OrgRole = 'agent' | 'lead' | 'manager';

export const ORG_ROLE_OPTIONS: { value: OrgRole; label: string; title: string }[] = [
  { value: 'agent', label: 'Agent', title: 'A team member — works tasks handed down by its lead' },
  { value: 'lead', label: 'Lead', title: 'Runs a team hands-on: works its members, may do manual work' },
  {
    value: 'manager',
    label: 'Manager',
    title: 'Staffing director — reports and directives only, delegates to the leads of its teams',
  },
];

/** The role to show and edit for a profile (`executive` and unset read as Agent). */
export function editableOrgRole(level: AgentProfile['orgLevel']): OrgRole {
  return level === 'lead' || level === 'manager' ? level : 'agent';
}
