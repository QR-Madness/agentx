import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { AgentProfile, AlloyWorkflow, AlloyWorkflowMember } from '../../lib/api';

const state = vi.hoisted(() => ({
  profiles: [] as AgentProfile[],
  workflows: [] as AlloyWorkflow[],
  modals: [] as Array<{ component: string }>,
  confirm: vi.fn(),
  notifyError: vi.fn(),
  openModal: vi.fn(),
  createWorkflow: vi.fn(),
  updateWorkflow: vi.fn(),
  deleteWorkflow: vi.fn(),
}));

// React Flow needs layout APIs jsdom lacks; the chart is covered by orgChart.test.ts.
vi.mock('./OrgCanvas', () => ({ OrgCanvas: () => <div data-testid="org-canvas" /> }));
vi.mock('../../contexts/AgentProfileContext', () => ({
  useAgentProfile: () => ({ profiles: state.profiles }),
}));
vi.mock('../../contexts/AlloyWorkflowContext', () => ({
  useAlloyWorkflow: () => ({
    workflows: state.workflows,
    createWorkflow: state.createWorkflow,
    updateWorkflow: state.updateWorkflow,
    deleteWorkflow: state.deleteWorkflow,
  }),
}));
vi.mock('../../contexts/ModalContext', () => ({
  useModal: () => ({ modals: state.modals, openModal: state.openModal }),
}));
vi.mock('../../contexts/NotificationContext', () => ({
  useNotify: () => ({ notifyError: state.notifyError }),
}));
vi.mock('../ui/ConfirmDialog', () => ({ useConfirm: () => state.confirm }));

import { OrgManager } from './OrgManager';

function agent(agentId: string, orgLevel: AgentProfile['orgLevel'] = 'agent'): AgentProfile {
  return { id: agentId, name: agentId.toUpperCase(), kind: 'agent', agentId, orgLevel } as AgentProfile;
}

function team(id: string, lead: string, members: string[], manager?: string): AlloyWorkflow {
  const list: AlloyWorkflowMember[] = [
    { agentId: lead, role: 'supervisor' },
    ...members.map((agentId) => ({ agentId, role: 'specialist' as const })),
  ];
  return { id, name: id.toUpperCase(), supervisorAgentId: lead, managerAgentId: manager, members: list } as AlloyWorkflow;
}

/** The chart's breakpoint is a max-width query; answer it. */
function setNarrow(narrow: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: query.includes('max-width') ? narrow : false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

beforeEach(() => {
  state.profiles = [agent('m', 'manager'), agent('l1', 'lead'), agent('a1'), agent('free')];
  state.workflows = [team('t1', 'l1', ['a1'], 'm')];
  state.modals = [{ component: 'orgManager' }];
  for (const fn of [state.confirm, state.notifyError, state.openModal, state.createWorkflow, state.updateWorkflow, state.deleteWorkflow]) {
    fn.mockReset();
  }
  setNarrow(false);
});

describe('OrgManager', () => {
  it('opens on the chart and the overview, and an issue opens what it is about', async () => {
    state.workflows = [team('t1', 'l1', ['a1', 'ghost'], 'm')];
    render(<OrgManager onClose={vi.fn()} />);

    expect(screen.getByTestId('org-canvas')).toBeInTheDocument();
    expect(screen.getByText('Needs attention (1)')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /no agent has that id/ }));
    expect(screen.getByRole('heading', { name: 'T1' })).toBeInTheDocument();
  });

  it('on a narrow screen shows the outline, and a team opens full-width with a way back', async () => {
    setNarrow(true);
    render(<OrgManager onClose={vi.fn()} />);

    expect(screen.queryByTestId('org-canvas')).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: /^T1/ }));
    expect(screen.getByRole('heading', { name: 'T1' })).toBeInTheDocument();
    expect(screen.queryByRole('navigation', { name: 'Organization outline' })).toBeNull();

    await userEvent.click(screen.getByRole('button', { name: 'Organization' }));
    expect(screen.getByRole('navigation', { name: 'Organization outline' })).toBeInTheDocument();
  });

  it('asks before discarding unsaved edits, and keeps them when told to', async () => {
    setNarrow(true);
    render(<OrgManager onClose={vi.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: /^T1/ }));
    await userEvent.type(screen.getByLabelText('When to delegate to A1'), 'x');

    state.confirm.mockResolvedValueOnce(false);
    await userEvent.click(screen.getByRole('button', { name: 'Organization' }));
    expect(state.confirm).toHaveBeenCalledWith(expect.objectContaining({ title: 'Discard unsaved changes?' }));
    expect(screen.getByRole('heading', { name: 'T1' })).toBeInTheDocument();

    state.confirm.mockResolvedValueOnce(true);
    await userEvent.click(screen.getByRole('button', { name: 'Organization' }));
    expect(screen.getByRole('navigation', { name: 'Organization outline' })).toBeInTheDocument();
  });

  it('closes on Escape only while it is the top surface', () => {
    const onClose = vi.fn();
    const { rerender, container } = render(<OrgManager onClose={onClose} />);

    state.modals = [{ component: 'orgManager' }, { component: 'unifiedProfileEditor' }];
    rerender(<OrgManager onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    expect(container.ownerDocument.querySelector('.org-container')).toHaveClass('is-covered');

    state.modals = [{ component: 'orgManager' }];
    rerender(<OrgManager onClose={onClose} />);
    expect(container.ownerDocument.querySelector('.org-container')).not.toHaveClass('is-covered');
    fireEvent.keyDown(window, { key: 'Escape' });
    return vi.waitFor(() => expect(onClose).toHaveBeenCalled());
  });

  it('keeps the team open and says so when a delete fails', async () => {
    render(<OrgManager onClose={vi.fn()} initialTeamId="t1" />);
    state.confirm.mockResolvedValueOnce(true);
    state.deleteWorkflow.mockResolvedValueOnce(false);

    await userEvent.click(screen.getByRole('button', { name: 'Delete team' }));
    expect(state.deleteWorkflow).toHaveBeenCalledWith('t1');
    expect(state.notifyError).toHaveBeenCalled();
    expect(screen.getByRole('heading', { name: 'T1' })).toBeInTheDocument();
  });

  it('starts a new team owned by the manager whose panel it was started from', async () => {
    render(<OrgManager onClose={vi.fn()} initialAgentId="m" />);
    const details = screen.getByRole('region', { name: 'Details' });
    await userEvent.click(within(details).getByRole('button', { name: 'New team' }));
    expect(screen.getByRole('heading', { name: 'New team' })).toBeInTheDocument();
    expect(screen.getByLabelText('Owning manager')).toHaveTextContent('M');
  });

  it('opens the profile editor above itself for Edit profile', async () => {
    render(<OrgManager onClose={vi.fn()} initialAgentId="a1" />);
    await userEvent.click(screen.getByRole('button', { name: 'Edit profile' }));
    expect(state.openModal).toHaveBeenCalledWith(
      expect.objectContaining({ component: 'unifiedProfileEditor', props: { initialProfileId: 'a1' } }),
    );
  });
});
