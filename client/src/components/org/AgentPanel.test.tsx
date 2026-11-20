import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { AgentProfile } from '../../lib/api';
import type { OrgPlacement } from '../../lib/orgPlacement';

const state = vi.hoisted(() => ({
  updateProfile: vi.fn(),
  notifyError: vi.fn(),
}));

vi.mock('../../contexts/AgentProfileContext', () => ({
  useAgentProfile: () => ({ updateProfile: state.updateProfile }),
}));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotify: () => ({ notifyError: state.notifyError }),
}));

import { AgentPanel } from './AgentPanel';

const placement: OrgPlacement = { inOrg: false, teams: [], superiors: [], subordinates: [] };

function renderPanel(overrides: Partial<AgentProfile> = {}) {
  const profile = {
    id: 'p1',
    agentId: 'bold-cosmic-falcon',
    name: 'Kestrel',
    kind: 'agent',
    orgLevel: 'agent',
    availableForDelegation: false,
    delegationHint: 'archival research',
    ...overrides,
  } as AgentProfile;
  render(
    <AgentPanel
      profile={profile}
      placement={placement}
      tier={undefined}
      issues={[]}
      onOpenTeam={vi.fn()}
      onOpenAgent={vi.fn()}
      onNewTeam={vi.fn()}
      onEditProfile={vi.fn()}
      onOpenDossier={vi.fn()}
    />,
  );
  return profile;
}

beforeEach(() => {
  state.updateProfile.mockReset().mockResolvedValue({ id: 'p1' });
  state.notifyError.mockReset();
});

describe('AgentPanel quick edit', () => {
  it('saves a new role at once and says Manager brings the tool limits', async () => {
    renderPanel();
    expect(screen.getByText(/Manager also applies the report-only tool limits/)).toBeTruthy();
    await userEvent.click(screen.getByRole('radio', { name: 'Manager' }));
    expect(state.updateProfile).toHaveBeenCalledWith('p1', { org_level: 'manager' });
  });

  it('shows a reserved executive as Agent and does not resave the current role', async () => {
    renderPanel({ orgLevel: 'executive' });
    const agent = screen.getByRole('radio', { name: 'Agent' });
    expect(agent.getAttribute('aria-checked')).toBe('true');
    await userEvent.click(agent);
    expect(state.updateProfile).not.toHaveBeenCalled();
  });

  it('toggles the roster at once', async () => {
    renderPanel();
    await userEvent.click(screen.getByRole('switch', { name: 'Join the team roster' }));
    expect(state.updateProfile).toHaveBeenCalledWith('p1', { available_for_delegation: true });
  });

  it('saves the specialty only when it changed, and clears it to the description fallback', async () => {
    renderPanel();
    expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
    const field = screen.getByLabelText("Kestrel's specialty");
    await userEvent.clear(field);
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(state.updateProfile).toHaveBeenCalledWith('p1', { delegation_hint: null });
    await userEvent.type(field, '  maps and charts ');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(state.updateProfile).toHaveBeenLastCalledWith('p1', { delegation_hint: 'maps and charts' });
  });

  it('says so when a save fails', async () => {
    state.updateProfile.mockResolvedValue(null);
    renderPanel();
    await userEvent.click(screen.getByRole('radio', { name: 'Lead' }));
    expect(state.notifyError).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Kestrel's role could not be saved. It is unchanged." }),
      'Quick edit',
    );
  });

  it('leaves an ambassador read-only', () => {
    renderPanel({ kind: 'ambassador' });
    expect(screen.queryByRole('radiogroup')).toBeNull();
    expect(screen.queryByRole('switch')).toBeNull();
    expect(screen.getByText('Ambassador')).toBeTruthy();
  });
});
